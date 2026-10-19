"""Runs, checkpoints, evaluation data, conditioning and ensemble stepping."""

import copy
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from einops import rearrange
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from common.utils import get_yaml
from dataset.datamodule import PDEDataModule
from modules.train_module import TrainModule, _with_legacy_names

REPO = Path(__file__).resolve().parents[1]

TORNADO_MODELS = ("tornado", "tornado_ar")
GAUSSIAN_MODELS = ("gaussian_nll", "moment_stepper")

LABELS = {
    "tornado": "TORNADO",
    "tornado_ar": "TORNADO-AR",
    "flow_matching": "FM",
    "interpolant": "SI",
    "ddim": "DDIM",
    "ddpm": "DDPM",
    "gaussian_nll": "Gaussian NLL",
    "moment_stepper": "VAE matching",
    "mc_dropout": "MC Dropout",
}

# "amplitude" and "fixed" are eval-only (tornado_eval_step)
TORNADO_SAMPLERS = ("em", "euler", "heun", "milstein", "amplitude", "fixed")


def default_base_dir() -> Path:
    """$TORNADO_LOGS, else $SCRATCH/stochastic_interpolants/logs (the training log_dir)."""
    if os.environ.get("TORNADO_LOGS"):
        return Path(os.environ["TORNADO_LOGS"])
    return Path(os.environ.get("SCRATCH", ".")) / "stochastic_interpolants" / "logs"


def default_eval_dir(base_dir: Path) -> Path:
    """$TORNADO_EVAL_DIR, else <base_dir>/eval."""
    return Path(os.environ.get("TORNADO_EVAL_DIR", Path(base_dir) / "eval"))



def find_ckpt(run_dir: Path, ckpt_type: str) -> Path:
    run_dir = Path(run_dir)
    if ckpt_type == "last":
        path = run_dir / "last.ckpt"
        if not path.exists():
            raise FileNotFoundError(f"No last.ckpt in {run_dir}")
        return path
    if ckpt_type == "best":
        ckpts = sorted(run_dir.glob("*best.ckpt*"))
        if not ckpts:
            raise FileNotFoundError(f"No '*best.ckpt*' in {run_dir}")
        return ckpts[0]
    raise ValueError(f"ckpt_type must be 'best' or 'last', got {ckpt_type!r}")


def ckpt_fingerprint(path: Path) -> Dict:
    """Size and mtime, to tell rewritten last.ckpt files apart."""
    st = Path(path).stat()
    return {"name": Path(path).name, "size": st.st_size, "mtime": int(st.st_mtime)}


def run_config(base_dir: Path, run: str) -> dict:
    """The run's config.yml, with `cone` renamed to `tornado`."""
    cfg = get_yaml(str(Path(base_dir) / run / "config.yml"))
    return copy.deepcopy(_with_legacy_names(cfg))


def model_name(cfg: dict) -> str:
    return cfg["model"]["model_name"]


def configure_sampling(cfg: dict, nfe: int, sigma_scale: Optional[float] = None) -> int:
    """Set the sampler's step count in place; returns the effective NFE.

    sigma_scale: inference-only override, a diagnostic (training used another value).
    """
    m = cfg["model"]
    name = m["model_name"]
    if "from_checkpoint" in m:
        m["from_checkpoint"] = None
    if name == "ddim":
        m["ddim"]["num_ddim_steps"] = nfe
    elif name == "flow_matching":
        m["flow_matching"]["num_refinement_steps"] = nfe
        m["flow_matching"]["num_train_steps"] = nfe
    elif name in TORNADO_MODELS:
        m[name]["integrator_steps"] = nfe
        if name == "tornado_ar":
            m[name]["from_checkpoint"] = None
    elif name == "interpolant":
        m["interpolant"]["num_refinement_steps"] = nfe
    elif name == "mc_dropout":
        nfe = 1

    if sigma_scale is not None:
        block = m.get(name)
        if not (isinstance(block, dict) and "sigma_scale" in block):
            raise ValueError(f"sigma_scale override: model '{name}' has no sigma_scale")
        block["sigma_scale"] = float(sigma_scale)
    return nfe


def load_processor(base_dir: Path, run: str, *, nfe: int, ckpt_type: str, normalizer,
                   device: str, sigma_scale: Optional[float] = None,
                   ae_dir: Optional[str] = None) -> Tuple[TrainModule, Dict]:
    """A trained processor in eval mode. Returns (proc, info)."""
    cfg = run_config(base_dir, run)
    eff_nfe = configure_sampling(cfg, nfe, sigma_scale)
    if ae_dir is not None:
        cfg["data"]["autoencoder"]["checkpoint_path"] = str(ae_dir)
    ckpt_path = find_ckpt(Path(base_dir) / run, ckpt_type)

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    proc = TrainModule(cfg, normalizer=normalizer)
    proc.load_state_dict(ckpt["state_dict"], strict=True)
    proc.eval().to(device)
    if proc.model_name == "mc_dropout":
        for m in proc.modules():
            if m.__class__.__name__.startswith("Dropout"):
                m.train()
    for p in proc.parameters():
        p.requires_grad_(False)

    info = {"model": proc.model_name, "nfe": eff_nfe, "ckpt": str(ckpt_path),
            "ckpt_fingerprint": ckpt_fingerprint(ckpt_path), "pde": proc.pde}
    return proc, info


def label_of(run: str, base_dir: Optional[Path] = None) -> str:
    if base_dir is not None:
        try:
            return LABELS.get(model_name(run_config(base_dir, run)), run)
        except Exception:
            pass
    for key in sorted(LABELS, key=len, reverse=True):
        if run.startswith(key + "_"):
            return LABELS[key]
    return "TORNADO" if run.startswith("cone_") else run



def _set_dotted(d: dict, key: str, value) -> None:
    *path, last = key.split(".")
    for p in path:
        d = d.setdefault(p, {})
    d[last] = value


def eval_data(pde: str, *, split: str = "valid", window: Optional[int] = None,
              batch_size: Optional[int] = None, num_workers: int = 2,
              overrides: Optional[Dict] = None) -> PDEDataModule:
    """Datamodule whose validation loader is the evaluation set.

    window: frames per window (combustion, ERA5). split='test': ERA5 test years.
    overrides: dotted keys into configs/data/<pde>.yaml.
    """
    dc = get_yaml(str(REPO / "configs" / "data" / f"{pde}.yaml"))
    dc["ae"] = False
    dc["num_workers"] = num_workers
    if batch_size is not None:
        dc["batch_size"] = batch_size
    if window is not None:
        dc["dataset"]["val_horizon"] = window
        dc["dataset"]["val_window_stride"] = window
    if split == "test":
        test_years = dc["dataset"].get("test_years")
        if not test_years:
            raise ValueError(f"split='test' needs dataset.test_years in configs/data/{pde}.yaml")
        dc["dataset"]["val_years"] = test_years
    elif split != "valid":
        raise ValueError(f"split must be 'valid' or 'test', got {split!r}")
    for k, v in (overrides or {}).items():
        _set_dotted(dc, k, v)
    dc = OmegaConf.to_container(OmegaConf.create(dc), resolve=True)  # ${oc.env:...} paths
    return PDEDataModule(dataconfig=dc)


def eval_loader(dm: PDEDataModule, subset: Optional[Dict] = None) -> DataLoader:
    """The validation loader, optionally on a random subset {"n": ..., "seed": ...}."""
    if subset is None:
        return dm.val_dataloader()
    ds = dm.val_dataset
    rng = np.random.default_rng(subset.get("seed", 0))
    idx = sorted(rng.choice(len(ds), size=min(subset["n"], len(ds)), replace=False).tolist())
    return DataLoader(Subset(ds, idx), batch_size=dm.batch_size, shuffle=False,
                      num_workers=dm.num_workers, pin_memory=False)


def sample_ids(batch: Dict, B: int, offset: int, pde: str) -> List[str]:
    """Run condition (combustion), initial time (ERA5), or a running index."""
    if pde == "era5":
        hours = batch.get("time_hours")
        if hours is not None:
            epoch = np.datetime64("1970-01-01T00", "h")
            return [str(epoch + np.timedelta64(int(hours[b, 0]), "h")) for b in range(B)]
    elif pde == "real_combustion":
        cond = batch.get("constant_scalars")
        if cond is not None:
            return [f"{float(cond[b, 0]):g}NH3_{float(cond[b, 1]):g}" for b in range(B)]
    return [f"sample{offset + b}" for b in range(B)]


class Conditioning:
    """Per-frame conditioning of a batch: run constants (combustion) or the date (ERA5)."""

    def __init__(self, proc: TrainModule, batch: Dict, device: str):
        cond = batch.get("constant_scalars")
        if cond is not None:
            cond = cond.to(device)
            if proc.pde == "real_combustion":
                cond = proc._normalize_combustion_cond(cond)
        time_cond = batch.get("time_scalars")
        self.cond = cond
        self.time_cond = None if time_cond is None else time_cond.to(device).float()

    def at(self, i: int):
        """[B, D] for frame i, or None."""
        if self.time_cond is None:
            return self.cond
        return self.time_cond[:, min(i, self.time_cond.shape[1] - 1)]

    def flat(self, i: int, K: int):
        """at(i) repeated over K members: [K*B, D]."""
        c = self.at(i)
        if c is None:
            return None
        return c.unsqueeze(0).expand(K, *c.shape).reshape(K * c.shape[0], -1)



@torch.no_grad()
def tornado_eval_step(proc: TrainModule, x: torch.Tensor, cond, method: str,
                      amplitude: Optional[float] = None) -> torch.Tensor:
    """Learned drift with a prescribed diffusion (Table 2): constant `amplitude`,
    or "fixed" g(t) = sqrt(sigma_scale t (1 - t))."""
    sch = proc.scheduler
    integ = sch.integrator
    timesteps = torch.linspace(0, 1, sch.integrator_steps + 1, device=x.device)
    y = x.clone()
    for i in range(len(timesteps) - 1):
        t, t_next = timesteps[i], timesteps[i + 1]
        dt = t_next - t
        drift, _ = integ._evaluate_model(x, y, proc.model, t, sch._sigma_from_raw, cond=cond)
        dW = torch.sqrt(dt) * torch.randn_like(y, device=y.device)
        if method == "amplitude":
            if amplitude is None:
                raise ValueError("sampler 'amplitude' needs an amplitude")
            amp = amplitude
        elif method == "fixed":
            amp = math.sqrt(sch.sigma_scale) * torch.sqrt(t * (1.0 - t))
        else:
            raise ValueError(f"unknown eval-only sampler {method!r}")
        y = y + dt * drift + amp * dW
    return y


@torch.no_grad()
def estimate_amplitude(proc: TrainModule, loader, device: str, n_batches: int = 4) -> float:
    """RMS of the learned g over one EM step on the first n_batches batches."""
    sch = proc.scheduler
    integ = sch.integrator
    sq, n = 0.0, 0
    for bi, batch in enumerate(loader):
        if bi >= n_batches:
            break
        cond = Conditioning(proc, batch, device)
        x = proc.encode(batch["input_fields"].to(device), cond.at(0), return_posterior=False)
        y = x.clone()
        timesteps = torch.linspace(0, 1, sch.integrator_steps + 1, device=device)
        for i in range(len(timesteps) - 1):
            t, dt = timesteps[i], timesteps[i + 1] - timesteps[i]
            drift, g = integ._evaluate_model(x, y, proc.model, t, sch._sigma_from_raw, cond=cond.at(0))
            sq += float(g.double().pow(2).sum())
            n += g.numel()
            y = y + dt * drift + g * torch.sqrt(dt) * torch.randn_like(y)
    return math.sqrt(sq / max(n, 1))


class Stepper:
    """Advances a K-member latent ensemble by one autoregressive step per call."""

    def __init__(self, proc: TrainModule, sampler: Optional[str] = None,
                 amplitude: Optional[float] = None):
        self.proc = proc
        self.name = proc.model_name
        self.sampler = sampler
        self.amplitude = amplitude
        if self.name in TORNADO_MODELS:
            self.sampler = sampler or "em"
            if self.sampler not in TORNADO_SAMPLERS:
                raise ValueError(f"TORNADO sampler must be one of {TORNADO_SAMPLERS}")
        elif sampler is not None and self.name != "flow_matching":
            raise ValueError(f"model '{self.name}' takes no sampler override")

    def init(self, u0: torch.Tensor, cond0, K: int):
        """K copies of the encoded initial condition."""
        proc = self.proc
        if self.name in GAUSSIAN_MODELS:
            _, post0 = proc.encode(u0, cond0, return_posterior=True)
            mu0 = rearrange(post0.mean, "b c h w -> b h w c")
            sigma0 = rearrange(torch.sqrt(post0.var), "b c h w -> b h w c")
            return (mu0.unsqueeze(0).expand(K, *mu0.shape).contiguous(),
                    sigma0.unsqueeze(0).expand(K, *sigma0.shape).contiguous())
        z0 = proc.encode(u0, cond0, return_posterior=False)
        return z0.unsqueeze(0).expand(K, *z0.shape).contiguous()

    def step(self, state, cond_flat, K: int, B: int):
        """(next_state, decoded latent, Gaussian-baseline mean or None), each (K, B, ...)."""
        proc = self.proc
        if self.name in GAUSSIAN_MODELS:
            z_mu, z_sigma = state
            flat = (z_mu.reshape(K * B, *z_mu.shape[2:]), z_sigma.reshape(K * B, *z_sigma.shape[2:]))
            nmu, nsig = proc.forward(flat, cond_flat)
            nmu = nmu.reshape(K, B, *nmu.shape[1:])
            nsig = nsig.reshape(K, B, *nsig.shape[1:])
            z_dec = nmu + nsig * torch.randn_like(nmu)
            return (nmu, nsig), z_dec, nmu

        z_flat = state.reshape(K * B, *state.shape[2:])
        if self.name in TORNADO_MODELS and self.sampler in ("amplitude", "fixed"):
            z_next = tornado_eval_step(proc, z_flat, cond_flat, self.sampler, self.amplitude)
        else:
            z_next = proc.forward(z_flat, cond_flat, method_val=self.sampler)
        z_next = z_next.reshape(K, B, *z_next.shape[1:])
        return z_next, z_next, None
