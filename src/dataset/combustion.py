import glob
import os
from typing import Tuple

import numpy as np
import pyarrow as pa
import torch
from torch.utils.data import Dataset


def _list_shards(path: str) -> list:
    """The .arrow shards under `path` (or `path` itself if it is a file)."""
    if os.path.isfile(path):
        return [path]
    shards = sorted(glob.glob(os.path.join(path, "*.arrow")))
    if not shards:
        raise FileNotFoundError(f"No .arrow shards found under {path}")
    return shards


class CombustionDataset(Dataset):
    """RealPDEBench real_combustion: one run per row, a float32 (nt, nx, ny) blob.

    Runs are named `<blend>NH3_<phi>`, e.g. `20NH3_1.05`. Items are single frames,
    or windows of `horizon` frames when return_traj is True.
    """

    def __init__(
        self,
        path: str,
        split: str,
        sim_ids: list = None,
        resolution: list = None,
        normalizer=None,
        return_traj: bool = False,
        horizon: int = None,
        t_downsample: int = 1,
        window_stride: int = None,
        dx: float = 1.0,
        dt: float = 1.0,
    ) -> None:
        """sim_ids: the runs of this split (None: all). dx, dt: placeholders, the archive has no coordinates."""
        super().__init__()
        self.path = path
        self.split = split
        self.return_traj = return_traj
        self.normalizer = normalizer
        self.t_downsample = max(1, int(t_downsample))
        self.shards = _list_shards(path)

        self.records = []
        for shard in self.shards:
            table = pa.ipc.open_stream(pa.memory_map(shard, "rb")).read_all()
            for row in range(table.num_rows):
                sim_id = table.column("sim_id")[row].as_py()
                sim_id = sim_id[:-3] if sim_id.endswith(".h5") else sim_id
                if sim_ids is not None and sim_id not in sim_ids:
                    continue
                self.records.append(
                    {
                        "shard": shard,
                        "row": row,
                        "sim_id": sim_id,
                        "nt": table.column("shape_t")[row].as_py(),
                        "nx": table.column("shape_h")[row].as_py(),
                        "ny": table.column("shape_w")[row].as_py(),
                    }
                )
            del table

        if sim_ids is not None:
            missing = sorted(set(sim_ids) - {r["sim_id"] for r in self.records})
            if missing:
                raise ValueError(
                    f"Split '{split}' requested runs that are not in the archive: "
                    f"{missing}. Present runs: {list_sim_ids(path)}. "
                    f"If you expected these, some .arrow shards are missing from {path}."
                )

        if not self.records:
            raise ValueError(
                f"No runs selected under {path} for split '{split}'. "
                f"Requested sim_ids={sim_ids}"
            )

        shapes = {(r["nx"], r["ny"]) for r in self.records}
        if len(shapes) > 1:
            by_shape = {}
            for r in self.records:
                by_shape.setdefault((r["nx"], r["ny"]), []).append(r["sim_id"])
            raise ValueError(
                f"Runs in {path} do not share a spatial shape: "
                + "; ".join(f"{s}: {sorted(ids)}" for s, ids in sorted(by_shape.items()))
                + ". Restrict sim_ids to one resolution, or resample first."
            )

        self.nx = self.records[0]["nx"]
        self.ny = self.records[0]["ny"]
        # runs differ by a frame or two
        raw_nt = min(r["nt"] for r in self.records)
        self.nt = raw_nt // self.t_downsample

        if resolution is not None:
            exp_nt, exp_nx, exp_ny = resolution
            if (exp_nx, exp_ny) != (self.nx, self.ny):
                raise ValueError(
                    f"resolution {resolution} does not match archive "
                    f"spatial shape ({self.nx}, {self.ny})"
                )
            self.nt = min(self.nt, exp_nt)

        self.horizon = horizon if horizon is not None else self.nt
        self.horizon = min(self.horizon, self.nt)

        self.dx = torch.tensor(float(dx))
        self.dy = torch.tensor(float(dx))
        self.dt = torch.tensor(float(dt) * self.t_downsample)
        self.t = torch.arange(self.horizon, dtype=torch.float32) * self.dt

        if self.return_traj:
            stride = window_stride if window_stride is not None else self.horizon
            self.stride = max(1, int(stride))
            self.windows_per_sim = 1 + (self.nt - self.horizon) // self.stride
            self.n_items = len(self.records) * self.windows_per_sim
        else:
            self.stride = None
            self.windows_per_sim = None
            self.n_items = len(self.records) * self.nt

        # [NH3 blend %, phi], unnormalised (TrainModule.get_data normalises)
        self.cond = torch.tensor(
            [list(parse_sim_id(r["sim_id"])) for r in self.records], dtype=torch.float32
        )

        self._tables = None
        self._pid = None

        print("Data loaded from: {}".format(path))
        print(
            f"split: {split}, runs: {len(self.records)}, "
            f"nt: {self.nt} (x{self.t_downsample} downsample), "
            f"nx: {self.nx}, ny: {self.ny}"
        )
        print(f"runs: {sorted(r['sim_id'] for r in self.records)}")
        if self.return_traj:
            print(
                f"horizon: {self.horizon}, stride: {self.stride}, "
                f"windows/run: {self.windows_per_sim}, items: {self.n_items}"
            )
        else:
            print(f"frames (items): {self.n_items}")
        print("\n")

    def __getstate__(self):
        # workers reopen their own memory maps
        state = self.__dict__.copy()
        state["_tables"] = None
        state["_pid"] = None
        return state

    def _get_blob(self, shard: str, row: int) -> Tuple[np.ndarray, int]:
        """Zero-copy float32 view of one run's blob."""
        if self._tables is None or self._pid != os.getpid():
            self._tables = {}
            self._pid = os.getpid()

        if shard not in self._tables:
            table = pa.ipc.open_stream(pa.memory_map(shard, "rb")).read_all()
            column = table.column("observed").combine_chunks()
            chunk = column.chunk(0) if isinstance(column, pa.ChunkedArray) else column
            buffers = chunk.buffers()
            offsets = np.frombuffer(
                buffers[1],
                dtype=np.int64 if pa.types.is_large_binary(chunk.type) else np.int32,
            )
            self._tables[shard] = {
                "table": table,  # owns the buffers below
                "offsets": offsets,
                "data": buffers[2],
                "base": chunk.offset,
            }

        entry = self._tables[shard]
        idx = entry["base"] + row
        start, end = int(entry["offsets"][idx]), int(entry["offsets"][idx + 1])
        flat = np.frombuffer(
            entry["data"], dtype=np.float32, count=(end - start) // 4, offset=start
        )
        return flat, start

    def _read_frames(self, record: dict, frames: np.ndarray) -> torch.Tensor:
        """Read the given (already downsampled) frame indices of one run."""
        flat, _ = self._get_blob(record["shard"], record["row"])
        stride = record["nx"] * record["ny"]
        out = np.stack([flat[f * stride : (f + 1) * stride] for f in frames])
        out = out.reshape(len(frames), record["nx"], record["ny"])
        return torch.from_numpy(np.ascontiguousarray(out))

    def __len__(self) -> int:
        return self.n_items

    def __getitem__(self, i: int) -> dict:
        if self.return_traj:
            sim_idx, window = divmod(i, self.windows_per_sim)
            record = self.records[sim_idx]
            start = window * self.stride
            frames = (np.arange(start, start + self.horizon)) * self.t_downsample
            u = self._read_frames(record, frames).unsqueeze(-1)
            u = self.normalizer.normalize(u) if self.normalizer is not None else u
            u_input, u_label = u[0], u
        else:
            sim_idx, frame = divmod(i, self.nt)
            record = self.records[sim_idx]
            u = self._read_frames(record, np.array([frame * self.t_downsample]))
            u = u[0].unsqueeze(-1)
            u = self.normalizer.normalize(u) if self.normalizer is not None else u
            u_input, u_label = u, u

        return {
            "input_fields": u_input,
            "output_fields": u_label,
            "constant_scalars": self.cond[sim_idx],
            "dx": self.dx,
            "dt": self.dt,
            "t": self.t,
        }


def list_sim_ids(path: str) -> list:
    """Every run present in the archive at `path`, sorted."""
    sim_ids = []
    for shard in _list_shards(path):
        table = pa.ipc.open_stream(pa.memory_map(shard, "rb")).read_all()
        for row in range(table.num_rows):
            sim_id = table.column("sim_id")[row].as_py()
            sim_ids.append(sim_id[:-3] if sim_id.endswith(".h5") else sim_id)
        del table
    return sorted(sim_ids)


def parse_sim_id(sim_id: str) -> Tuple[float, float]:
    """`20NH3_1.05` -> (20.0, 1.05): ammonia blend %, equivalence ratio."""
    blend, ratio = sim_id.split("_", 1)
    return float(blend.upper().replace("NH3", "")), float(ratio)


def split_sim_ids(
    path: str,
    val_sims: list = None,
    n_val: int = 3,
    stratify: bool = True,
) -> Tuple[list, list]:
    """Train/valid split by whole run.

    stratify: hold out the median-phi run of each blend group with 2+ runs (the
    blends are unevenly sampled). Otherwise n_val evenly spaced runs. val_sims
    overrides both.
    """
    all_sims = list_sim_ids(path)

    if val_sims is not None:
        val = [s[:-3] if s.endswith(".h5") else s for s in val_sims]
        unknown = sorted(set(val) - set(all_sims))
        if unknown:
            raise ValueError(
                f"val_sims not present in archive: {unknown}. Available: {all_sims}"
            )
    elif stratify:
        groups = {}
        for s in all_sims:
            groups.setdefault(parse_sim_id(s)[0], []).append(s)
        val = []
        for blend in sorted(groups):
            runs = sorted(groups[blend], key=lambda s: parse_sim_id(s)[1])
            if len(runs) < 2:
                print(
                    f"[split] blend {blend:g}% NH3 has a single run "
                    f"({runs[0]}); keeping it in train."
                )
                continue
            val.append(runs[len(runs) // 2])
    else:
        n_val = min(n_val, max(1, len(all_sims) - 1))
        idx = np.linspace(0, len(all_sims) - 1, n_val + 2)[1:-1].round().astype(int)
        val = [all_sims[j] for j in sorted(set(idx.tolist()))]

    val = sorted(set(val))
    train = [s for s in all_sims if s not in set(val)]
    if not train:
        raise ValueError(f"Split leaves no training runs (val={val})")
    return train, val
