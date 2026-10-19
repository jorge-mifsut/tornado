"""WeatherBench-2 ERA5 at 1.5 degrees: [t2m, u10, v10, z500] on 121 x 240, 6-hourly.

Latitude runs south to north. Splits are by calendar year. Each frame is
conditioned on sin/cos of the fraction of the year and of the day.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

VARS = ["t2m", "u10", "v10", "z500"]
DATE_FEATURES = ["sin_doy", "cos_doy", "sin_tod", "cos_tod"]
NUM_DATE_FEATURES = len(DATE_FEATURES)

_EPOCH = np.datetime64("1970-01-01T00:00:00", "h")


def date_features(hours: np.ndarray) -> np.ndarray:
    """[N] hours since 1970-01-01 -> [N, 4] sin/cos of year- and day-fraction."""
    hours = np.asarray(hours, dtype=np.int64)
    times = _EPOCH + hours.astype("timedelta64[h]")

    year = times.astype("datetime64[Y]")
    year_start = year.astype("datetime64[h]")
    next_year = (year + 1).astype("datetime64[h]")
    elapsed = (times - year_start).astype("timedelta64[h]").astype(np.float64)
    length = (next_year - year_start).astype("timedelta64[h]").astype(np.float64)
    doy = elapsed / length

    tod = (times - times.astype("datetime64[D]")).astype(
        "timedelta64[h]"
    ).astype(np.float64) / 24.0

    two_pi = 2.0 * np.pi
    return np.stack(
        [
            np.sin(two_pi * doy),
            np.cos(two_pi * doy),
            np.sin(two_pi * tod),
            np.cos(two_pi * tod),
        ],
        axis=-1,
    ).astype(np.float32)


def load_meta(path: str) -> dict:
    meta_path = os.path.join(path, "era5_1p5_meta.json")
    if not os.path.isfile(meta_path):
        if os.path.isdir(path):
            present = sorted(os.listdir(path))[:10]
            what = f"it contains {present}" if present else "it is empty"
        else:
            what = "the directory does not exist"
        raise FileNotFoundError(
            f"No converted ERA5 dataset at {path} -- {what}.\n"
            f"The yearly NetCDFs have to be converted once before training:\n"
            f"    python dataset/prepare_era5.py --src <dir of .nc files> "
            f"--dst {path} --years 2010-2018 --train-years 2010-2015\n"
            f"(or: sbatch bash/era5_prepare.sh). If you converted them somewhere "
            f"else, point data.dataset.path and data.normalizer.stat_path there."
        )
    with open(meta_path) as f:
        return json.load(f)


def available_years(path: str) -> list:
    """Years for which a converted .npy exists under `path`."""
    years = []
    for name in os.listdir(path):
        if name.startswith("era5_1p5_") and name.endswith(".npy") and "_time" not in name:
            stem = name[len("era5_1p5_") : -len(".npy")]
            if stem.isdigit():
                years.append(int(stem))
    return sorted(years)


class ERA5Dataset(Dataset):
    """One item is a frame, or a window of `horizon` frames when return_traj is True.

    Windows do not cross year boundaries.
    """

    def __init__(
        self,
        path: str,
        split: str,
        years: list,
        normalizer=None,
        return_traj: bool = False,
        horizon: int = None,
        t_downsample: int = 1,
        window_stride: int = None,
        dx: float = 1.5,
        dt: float = 6.0,
    ) -> None:
        """path: output of dataset/prepare_era5.py. dx in degrees, dt in hours."""
        super().__init__()
        self.path = path
        self.split = split
        self.return_traj = return_traj
        self.normalizer = normalizer
        self.t_downsample = max(1, int(t_downsample))

        self.meta = load_meta(path)
        self.lat = np.asarray(self.meta["lat"], dtype=np.float64)
        self.lon = np.asarray(self.meta["lon"], dtype=np.float64)
        self.nlat, self.nlon = len(self.lat), len(self.lon)
        self.vars = self.meta["vars"]
        self.n_channels = len(self.vars)

        present = available_years(path)
        years = sorted(int(y) for y in years)
        missing = sorted(set(years) - set(present))
        if missing:
            raise FileNotFoundError(
                f"Split '{split}' requested years {missing} but only {present} are "
                f"converted under {path}. Run dataset/prepare_era5.py for them."
            )
        self.years = years

        self.records = []
        for year in years:
            hours = np.load(os.path.join(path, f"era5_1p5_{year}_time.npy"))
            hours = hours[:: self.t_downsample]
            self.records.append(
                {
                    "year": year,
                    "field_path": os.path.join(path, f"era5_1p5_{year}.npy"),
                    "nt": len(hours),
                    "hours": hours,
                    "date": date_features(hours),
                }
            )

        self.dx = torch.tensor(float(dx))
        self.dy = torch.tensor(float(dx))
        self.dt = torch.tensor(float(dt) * self.t_downsample)

        if self.return_traj:
            nt_min = min(r["nt"] for r in self.records)
            self.horizon = min(horizon if horizon is not None else nt_min, nt_min)
            stride = window_stride if window_stride is not None else self.horizon
            self.stride = max(1, int(stride))
            self.index = []
            for ri, r in enumerate(self.records):
                n_windows = 1 + (r["nt"] - self.horizon) // self.stride
                self.index.extend((ri, w * self.stride) for w in range(n_windows))
        else:
            self.horizon = 1
            self.stride = None
            self.index = [
                (ri, t) for ri, r in enumerate(self.records) for t in range(r["nt"])
            ]

        self.t = torch.arange(self.horizon, dtype=torch.float32) * self.dt

        # opened lazily, per worker
        self._maps = None
        self._pid = None

        print(f"Data loaded from: {path}")
        print(
            f"split: {split}, years: {years}, frames: "
            f"{sum(r['nt'] for r in self.records)} "
            f"(x{self.t_downsample} downsample, dt={float(self.dt):g}h), "
            f"nlat: {self.nlat}, nlon: {self.nlon}, channels: {self.vars}"
        )
        if self.return_traj:
            print(
                f"horizon: {self.horizon}, stride: {self.stride}, "
                f"items: {len(self.index)}"
            )
        else:
            print(f"frames (items): {len(self.index)}")
        print("\n")

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_maps"] = None
        state["_pid"] = None
        return state

    def _map(self, record: dict) -> np.ndarray:
        if self._maps is None or self._pid != os.getpid():
            self._maps = {}
            self._pid = os.getpid()
        key = record["field_path"]
        if key not in self._maps:
            self._maps[key] = np.load(key, mmap_mode="r")
        return self._maps[key]

    def _read(self, record: dict, start: int, count: int) -> torch.Tensor:
        """Read `count` frames from `start`, both in downsampled index space."""
        arr = self._map(record)
        if self.t_downsample == 1:
            block = arr[start : start + count]
        else:
            rows = (np.arange(start, start + count) * self.t_downsample)
            block = arr[rows]
        return torch.from_numpy(np.ascontiguousarray(block))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict:
        record_idx, start = self.index[i]
        record = self.records[record_idx]

        u = self._read(record, start, self.horizon)
        if self.normalizer is not None:
            u = self.normalizer.normalize(u)

        date = torch.from_numpy(record["date"][start : start + self.horizon])
        hours = torch.from_numpy(record["hours"][start : start + self.horizon])

        if self.return_traj:
            u_input, u_label = u[0], u
        else:
            u_input = u_label = u[0]

        return {
            "input_fields": u_input,
            "output_fields": u_label,
            "constant_scalars": date[0],
            "time_scalars": date,
            "time_hours": hours,
            "dx": self.dx,
            "dt": self.dt,
            "t": self.t,
        }


def split_years(
    train_years: list = None, val_years: list = None, test_years: list = None
) -> dict:
    """Default split: train 2010-2015, valid 2016, test 2017-2018."""
    return {
        "train": list(train_years) if train_years else list(range(2010, 2016)),
        "valid": list(val_years) if val_years else [2016],
        "test": list(test_years) if test_years else [2017, 2018],
    }
