"""Latent error vs ensemble variance by error tercile (Table 14).

Per latent point: mse to the encoded truth and ensemble variance, channel-averaged;
points are split into terciles of mse.
"""

import copy
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from einops import rearrange

from evals.core import Conditioning, eval_data, eval_loader, load_processor, run_config
from evals.rollout import store_dir

GROUPS = [("low", 0.0, 1.0 / 3.0), ("mid", 1.0 / 3.0, 2.0 / 3.0), ("high", 2.0 / 3.0, 1.0)]


def quantile_threshold(x: torch.Tensor, q: float) -> torch.Tensor:
    x = x.reshape(-1)
    n = x.numel()
    k = min(max(int(q * (n - 1)) + 1, 1), n)
    return torch.kthvalue(x, k).values


@torch.inference_mode()
def pointwise(spec, paths, steps: int, device: str):
    """Flattened per-point (mse, var) over steps 1..steps and every sample."""
    spec = copy.deepcopy(spec)
    pde = run_config(paths.base_dir, spec.run)["data"]["pde"]
    spec.resolve(pde)
    sdir = store_dir(spec, paths)
    dm = eval_data(pde, split=spec.split, window=spec.window, batch_size=spec.batch_size,
                   num_workers=0, overrides=paths.data_overrides)
    proc, _ = load_processor(paths.base_dir, spec.run, nfe=spec.nfe, ckpt_type=spec.ckpt,
                             normalizer=dm.normalizer, device=device, ae_dir=paths.ae_dir)
    mses, vars_ = [], []
    for bi, batch in enumerate(eval_loader(dm, spec.subset)):
        st = torch.load(sdir / f"batch_{bi:05d}.pt", map_location="cpu", weights_only=False)
        cond = Conditioning(proc, batch, device)
        gt = batch["output_fields"]
        for t in range(min(steps, st["z"].shape[1])):
            _, post = proc.encode(gt[:, t + 1].to(device), cond.at(t + 1), return_posterior=True)
            y_lat = rearrange(post.mean * proc.scale_factor, "b c h w -> b h w c")
            z = st["z"][:, t].to(device)  # (K, B, h, w, c)
            mses.append(((z.mean(0) - y_lat) ** 2).mean(-1).reshape(-1).cpu())
            vars_.append(z.var(dim=0, unbiased=False).mean(-1).reshape(-1).cpu())
    return torch.cat(mses), torch.cat(vars_)


def terciles(mse: torch.Tensor, var: torch.Tensor) -> List[Dict]:
    q1 = quantile_threshold(mse, GROUPS[1][1])
    q2 = quantile_threshold(mse, GROUPS[1][2])
    masks = {"low": mse <= q1, "mid": (mse > q1) & (mse <= q2), "high": mse > q2}
    rows = []
    for name, m in masks.items():
        mm, vv = float(mse[m].mean()), float(var[m].mean())
        rows.append({"tercile": name, "mse": mm, "var": vv, "var_over_mse": vv / mm,
                     "fraction": float(m.float().mean())})
    return rows


def report(name, exp, paths, out: Path, device: str, specs) -> None:
    rows = []
    for row, spec in specs:
        mse, var = pointwise(spec, paths, exp["steps"], device)
        for r in terciles(mse, var):
            rows.append({"label": row["label"], **r})
    df = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{name}.csv", index=False)
    print(df.to_string(index=False))
