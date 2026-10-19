"""Stage 1: roll out a model and store its latent ensemble trajectories.

    python -m evals.rollout --run <run> --rollout 50 [--nfe 5 -K 16 ...]

Writes <eval_dir>/<run>/<tag>/meta.json and batch_XXXXX.pt (latents z (K, R, B, ...),
sample ids, RNG state). Interrupted rollouts resume with the same draws.
"""

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Optional

import torch
from tqdm import tqdm

from evals.core import (
    Conditioning,
    Stepper,
    default_base_dir,
    default_eval_dir,
    estimate_amplitude,
    eval_data,
    eval_loader,
    load_processor,
    run_config,
    sample_ids,
)

WINDOWED = ("real_combustion", "era5")


@dataclass
class RolloutSpec:
    """Everything that determines a rollout; `tag` names its store."""

    run: str
    rollout: int = 20
    nfe: int = 5
    K: int = 16
    ckpt: str = "best"
    sampler: Optional[str] = None      # TORNADO: em|euler|heun|milstein|amplitude|fixed; FM: euler|heun|midpoint
    amplitude: Optional[float] = None  # for sampler="amplitude"; None estimates it
    seed: int = 0
    split: str = "valid"
    window: Optional[int] = None       # combustion / ERA5; default rollout + 1
    batch_size: Optional[int] = None
    subset: Optional[Dict] = None      # {"n": 64, "seed": 0}
    sigma_scale: Optional[float] = None  # inference-only diagnostic

    def resolve(self, pde: str) -> "RolloutSpec":
        if self.window is None and pde in WINDOWED:
            self.window = self.rollout + 1
        return self

    def tag(self) -> str:
        parts = [self.split, f"nfe{self.nfe}", f"K{self.K}", self.ckpt]
        if self.sampler:
            parts.append(self.sampler)
            if self.sampler == "amplitude" and self.amplitude is not None:
                parts.append(f"g{self.amplitude:g}")
        if self.window is not None:
            parts.append(f"w{self.window}")
        if self.batch_size is not None:
            parts.append(f"b{self.batch_size}")
        if self.subset is not None:
            parts.append(f"sub{self.subset['n']}s{self.subset.get('seed', 0)}")
        if self.sigma_scale is not None:
            parts.append(f"sig{self.sigma_scale:g}")
        parts += [f"r{self.rollout}", f"s{self.seed}"]
        return "_".join(parts)


@dataclass
class Paths:
    base_dir: Path = field(default_factory=default_base_dir)
    eval_dir: Optional[Path] = None
    ae_dir: Optional[str] = None
    data_overrides: Optional[Dict] = None    # dotted keys into configs/data/<pde>.yaml

    def __post_init__(self):
        self.base_dir = Path(self.base_dir)
        self.eval_dir = Path(self.eval_dir) if self.eval_dir else default_eval_dir(self.base_dir)


def store_dir(spec: RolloutSpec, paths: Paths) -> Path:
    return paths.eval_dir / spec.run / spec.tag()


def is_complete(sdir: Path) -> bool:
    meta = sdir / "meta.json"
    return meta.exists() and json.loads(meta.read_text()).get("complete", False)


def _git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       cwd=Path(__file__).resolve().parents[1]).strip()
    except Exception:
        return None


def _rng_state():
    state = {"cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state):
    torch.set_rng_state(state["cpu"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _atomic_save(obj, path: Path):
    tmp = path.with_name(f".{path.name}.tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


@torch.inference_mode()
def run_rollout(spec: RolloutSpec, paths: Paths, *, device: Optional[str] = None,
                num_workers: int = 2, force: bool = False) -> Path:
    """Write (or complete) the store for `spec`. Returns its directory."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    pde = run_config(paths.base_dir, spec.run)["data"]["pde"]
    spec.resolve(pde)
    sdir = store_dir(spec, paths)
    if is_complete(sdir) and not force:
        print(f"[rollout] complete: {sdir}")
        return sdir
    if force and sdir.exists():
        for f in sdir.glob("batch_*.pt"):
            f.unlink()
    sdir.mkdir(parents=True, exist_ok=True)

    dm = eval_data(pde, split=spec.split, window=spec.window, batch_size=spec.batch_size,
                   num_workers=num_workers, overrides=paths.data_overrides)
    loader = eval_loader(dm, spec.subset)

    torch.manual_seed(spec.seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(spec.seed)

    proc, info = load_processor(paths.base_dir, spec.run, nfe=spec.nfe, ckpt_type=spec.ckpt,
                                normalizer=dm.normalizer, device=device,
                                sigma_scale=spec.sigma_scale, ae_dir=paths.ae_dir)
    meta_path = sdir / "meta.json"
    if meta_path.exists() and not force:
        old = json.loads(meta_path.read_text())
        if old.get("ckpt_fingerprint") != info["ckpt_fingerprint"]:
            raise RuntimeError(
                f"{sdir} was started from a different checkpoint ({old.get('ckpt_fingerprint')} "
                f"vs {info['ckpt_fingerprint']}); rerun with --force")

    amplitude = spec.amplitude
    if spec.sampler == "amplitude" and amplitude is None:
        amplitude = estimate_amplitude(proc, loader, device)
        print(f"[rollout] estimated constant amplitude g = {amplitude:.6g}")
        torch.manual_seed(spec.seed)
        if device.startswith("cuda"):
            torch.cuda.manual_seed_all(spec.seed)

    stepper = Stepper(proc, sampler=spec.sampler, amplitude=amplitude)
    meta = {"spec": asdict(spec), "tag": spec.tag(), "pde": pde, "complete": False,
            "amplitude_used": amplitude, "git_commit": _git_commit(), **info}
    meta_path.write_text(json.dumps(meta, indent=2))

    offset, resumed = 0, False
    n_batches = 0
    for bi, batch in enumerate(tqdm(loader, desc=f"rollout {spec.run}")):
        B = batch["input_fields"].shape[0]
        bpath = sdir / f"batch_{bi:05d}.pt"
        n_batches = bi + 1
        if bpath.exists():
            offset += B
            resumed = True
            continue
        if resumed and bi > 0:
            prev = torch.load(sdir / f"batch_{bi - 1:05d}.pt", map_location="cpu", weights_only=False)
            _set_rng_state(prev["rng"])
            resumed = False

        u0 = batch["input_fields"].to(device)
        T = batch["output_fields"].shape[1]
        R = min(spec.rollout, T - 1)
        cond = Conditioning(proc, batch, device)
        K = spec.K

        state = stepper.init(u0, cond.at(0), K)
        zs, means = [], []
        for t in range(R):
            state, z_dec, z_mean = stepper.step(state, cond.flat(t, K), K, B)
            zs.append(z_dec.float().cpu())
            if z_mean is not None:
                means.append(z_mean.float().cpu())

        out = {
            "z": torch.stack(zs, 1),
            "z_mean": torch.stack(means, 1) if means else None,
            "ids": sample_ids(batch, B, offset, pde),
            "index": list(range(offset, offset + B)),
            "rng": _rng_state(),
        }
        _atomic_save(out, bpath)
        offset += B
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    meta.update({"complete": True, "n_batches": n_batches, "n_samples": offset})
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[rollout] wrote {sdir} ({offset} samples)")
    return sdir


def add_spec_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--run", required=True, help="run directory name under --base-dir")
    p.add_argument("--rollout", type=int, default=20)
    p.add_argument("--nfe", type=int, default=5)
    p.add_argument("-K", "--ensemble", type=int, default=16)
    p.add_argument("--ckpt", default="best", choices=["best", "last"])
    p.add_argument("--sampler", default=None)
    p.add_argument("--amplitude", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split", default="valid", choices=["valid", "test"])
    p.add_argument("--window", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--subset", type=int, default=None, help="random subset size of the evaluation set")
    p.add_argument("--sigma-scale", type=float, default=None,
                   help="DIAGNOSTIC: inference-only sigma_scale override (not reportable)")


def add_path_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--base-dir", default=None, help="training runs (default $TORNADO_LOGS or $SCRATCH/...)")
    p.add_argument("--eval-dir", default=None, help="stores (default $TORNADO_EVAL_DIR or <base-dir>/eval)")
    p.add_argument("--ae-dir", default=None, help="override the runs' data.autoencoder.checkpoint_path")
    p.add_argument("--data", nargs="*", default=[], metavar="KEY=VALUE",
                   help="overrides of configs/data/<pde>.yaml, e.g. dataset.path=/my/data")


def spec_from_args(a) -> RolloutSpec:
    return RolloutSpec(run=a.run, rollout=a.rollout, nfe=a.nfe, K=a.ensemble, ckpt=a.ckpt,
                       sampler=a.sampler, amplitude=a.amplitude, seed=a.seed,
                       split=a.split, window=a.window, batch_size=a.batch_size,
                       subset=None if a.subset is None else {"n": a.subset, "seed": 0},
                       sigma_scale=a.sigma_scale)


def paths_from_args(a) -> Paths:
    import yaml

    overrides = {}
    for kv in a.data:
        k, v = kv.split("=", 1)
        overrides[k] = yaml.safe_load(v)
    kw = {"eval_dir": a.eval_dir, "ae_dir": a.ae_dir, "data_overrides": overrides or None}
    if a.base_dir:
        kw["base_dir"] = a.base_dir
    return Paths(**kw)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_spec_args(p)
    add_path_args(p)
    p.add_argument("--force", action="store_true")
    p.add_argument("--num-workers", type=int, default=2)
    a = p.parse_args()
    torch.set_float32_matmul_precision("high")
    run_rollout(spec_from_args(a), paths_from_args(a), num_workers=a.num_workers, force=a.force)


if __name__ == "__main__":
    main()
