"""Stage 2: decode a rollout store into <store>/metrics.csv, one row per (step, sample, K).

    python -m evals.metrics --run <run> --rollout 50 [options of evals.rollout]

Columns: mse, mae (ensemble mean), crps, sharpness, cov/width@90/95, energy,
rank_0..rank_K, latent rmse_z, sharpness_z, rmse_vae, enstrophy ens_*, and for
ERA5 mse/crps/sharpness_<v> and rank_<v>_*. Physical units, except ERA5
(normalised). SSR and RMSCE are computed per group in evals/reduce.py.
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from evals import scores as S
from evals.core import Conditioning, eval_data, eval_loader, load_processor, run_config
from evals.rollout import (
    Paths,
    RolloutSpec,
    add_path_args,
    add_spec_args,
    is_complete,
    paths_from_args,
    run_rollout,
    spec_from_args,
    store_dir,
)


def _pixel_rows(u: torch.Tensor, y: torch.Tensor, *, era5: bool,
                u_n=None, y_n=None, w=None) -> List[dict]:
    """Scores of each element. u: (K, B, ...), y: (B, ...)."""
    B = y.shape[0]
    mu = u.mean(dim=0)
    rows = []
    for b in range(B):
        if era5:
            em = S.era5_metrics(u_n[:, b : b + 1], y_n[b : b + 1], w)
            row = {"mse": em["mse"], "crps": em["crps"], "sharpness": em["sharpness"]}
            for key in ("mse", "crps", "sharpness"):
                row.update({f"{key}_{v}": x for v, x in zip(S.ERA5_VARS, em[f"{key}_c"])})
            ranks = S.rank_counts(u_n[:, b : b + 1], y_n[b : b + 1], w)
            for c, v in enumerate(S.ERA5_VARS):
                rc = S.rank_counts(u_n[:, b : b + 1, ..., c : c + 1], y_n[b : b + 1, ..., c : c + 1], w)
                row.update({f"rank_{v}_{r}": float(x) for r, x in enumerate(rc.tolist())})
        else:
            ub, yb = u[:, b : b + 1], y[b : b + 1]
            c90 = S.central_interval_coverage_width(ub, yb, p=0.90)
            c95 = S.central_interval_coverage_width(ub, yb, p=0.95)
            row = {
                "mse": float(S.mse(mu[b : b + 1], yb)),
                "mae": float(S.mae(mu[b : b + 1], yb)),
                "crps": float(S.crps_ensemble_l1(ub, yb)),
                "sharpness": float(S.sharpness_std(ub)),
                "cov@90": float(c90["cov@90"]), "width@90": float(c90["width@90"]),
                "cov@95": float(c95["cov@95"]), "width@95": float(c95["width@95"]),
                "energy": float(S.energy_score(ub, yb, beta=1.0)),
            }
            ranks = S.rank_counts(ub, yb)
        row.update({f"rank_{r}": float(c) for r, c in enumerate(ranks.tolist())})
        rows.append(row)
    return rows


@torch.inference_mode()
def compute_metrics(spec: RolloutSpec, paths: Paths, *, k_subsets: Optional[List[int]] = None,
                    latent: bool = True, device: Optional[str] = None, num_workers: int = 2,
                    force: bool = False) -> Path:
    """Write <store>/metrics.csv; k_subsets adds rows for the first k members."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    pde = run_config(paths.base_dir, spec.run)["data"]["pde"]
    spec.resolve(pde)
    sdir = store_dir(spec, paths)
    if not is_complete(sdir):
        raise FileNotFoundError(f"no complete rollout store at {sdir}; run evals.rollout first")
    out = sdir / "metrics.csv"
    ks = sorted(set([spec.K] + [k for k in (k_subsets or []) if k < spec.K]))
    info_path = sdir / "metrics.json"
    if out.exists() and info_path.exists() and not force:
        if json.loads(info_path.read_text()).get("K_values") == ks:
            print(f"[metrics] exists: {out}")
            return out

    meta = json.loads((sdir / "meta.json").read_text())
    dm = eval_data(pde, split=spec.split, window=spec.window, batch_size=spec.batch_size,
                   num_workers=num_workers, overrides=paths.data_overrides)
    loader = eval_loader(dm, spec.subset)
    proc, _ = load_processor(paths.base_dir, spec.run, nfe=spec.nfe, ckpt_type=spec.ckpt,
                             normalizer=dm.normalizer, device=device,
                             sigma_scale=spec.sigma_scale, ae_dir=paths.ae_dir)
    era5 = pde == "era5"
    # encoding the true frame samples the posterior
    torch.manual_seed(spec.seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(spec.seed)

    rows = []
    for bi, batch in enumerate(tqdm(loader, desc=f"metrics {spec.run}")):
        st = torch.load(sdir / f"batch_{bi:05d}.pt", map_location="cpu", weights_only=False)
        z_all, ids = st["z"], st["ids"]
        u0 = batch["input_fields"]
        B = u0.shape[0]
        if len(ids) != B:
            raise RuntimeError(f"batch {bi}: store has {len(ids)} samples, loader {B}")
        gt = batch["output_fields"]
        K, R = z_all.shape[0], z_all.shape[1]
        cond = Conditioning(proc, batch, device)
        w = S.era5_weights(u0.shape[1], device) if era5 else None
        out_shape = tuple(u0.shape[1:])

        ens_ic = S.total_enstrophy(proc.normalizer.denormalize(u0.to(device))).cpu()

        def decode(z, i):
            """(k, B, ...) latents of frame i -> (normalised, physical) fields."""
            k = z.shape[0]
            u_n = proc.decode(z.reshape(k * B, *z.shape[2:]).to(device), cond.flat(i, k))
            u_n = u_n.reshape(k, B, *out_shape)
            return u_n, proc.normalizer.denormalize(u_n)

        for t in range(R):
            z = z_all[:, t]
            u_n, u = decode(z, t + 1)
            y_n = gt[:, t + 1].to(device)
            y = proc.normalizer.denormalize(y_n)

            base = [{"step": t + 1, "sample": ids[b], "index": st["index"][b], "batch": bi}
                    for b in range(B)]
            for k in ks:
                pr = _pixel_rows(u[:k], y, era5=era5, u_n=u_n[:k], y_n=y_n, w=w)
                for b in range(B):
                    pr[b].update(base[b], K=k)
                if k == spec.K:
                    full = pr
                else:
                    rows.extend(pr)

            if latent:
                z_true = proc.encode(y_n, cond.at(t + 1), return_posterior=False)
                u_rec_n = proc.decode(z_true, cond.at(t + 1))
                u_rec = proc.normalizer.denormalize(u_rec_n)
                z_ens = (st["z_mean"][:, t] if st["z_mean"] is not None else z).to(device)
                z_spread = z_ens.std(dim=0, unbiased=False)
                z_err = z_ens.mean(dim=0) - z_true
                for b in range(B):
                    s_z = float(torch.sqrt(z_spread[b].pow(2).mean()))
                    e_z = float(torch.sqrt(z_err[b].pow(2).mean()))
                    if era5:
                        m_vae = float(np.sqrt(max(S.w_mse(u_rec_n[b : b + 1], y_n[b : b + 1], w), 0.0)))
                    else:
                        m_vae = float(np.sqrt(max(float(S.mse(u_rec[b : b + 1], y[b : b + 1])), 0.0)))
                    full[b].update({"sharpness_z": s_z, "rmse_z": e_z, "rmse_vae": m_vae})
                    if not era5:
                        full[b]["ens_rec"] = float(S.total_enstrophy(u_rec[b]))
            if not era5:
                ens_pred = S.total_enstrophy(u).cpu()
                ens_gt = S.total_enstrophy(y).cpu()
                for b in range(B):
                    full[b]["ens_ic"] = float(ens_ic[b])
                    full[b]["ens_gt"] = float(ens_gt[b])
                    full[b].update({f"ens_m{k}": float(ens_pred[k, b]) for k in range(K)})
            rows.extend(full)
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    front = ["step", "sample", "index", "batch", "K"]
    df = df[front + [c for c in df.columns if c not in front]]
    tmp = out.with_name(".metrics.csv.tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(out)
    info_path.write_text(json.dumps({"K_values": ks, "latent": latent, "rows": len(df),
                                     "store": meta["tag"]}, indent=2))
    print(f"[metrics] wrote {out} ({len(df)} rows)")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_spec_args(p)
    add_path_args(p)
    p.add_argument("--k-subsets", type=int, nargs="*", default=None,
                   help="also score the first k members, e.g. 4 8 16 32 (ensemble-size sweep)")
    p.add_argument("--no-latent", action="store_true", help="skip the latent-space diagnostics")
    p.add_argument("--rollout-if-missing", action="store_true", help="run stage 1 first if needed")
    p.add_argument("--force", action="store_true")
    p.add_argument("--num-workers", type=int, default=2)
    a = p.parse_args()
    torch.set_float32_matmul_precision("high")
    spec, paths = spec_from_args(a), paths_from_args(a)
    if a.rollout_if_missing:
        run_rollout(spec, paths, num_workers=a.num_workers)
    compute_metrics(spec, paths, k_subsets=a.k_subsets, latent=not a.no_latent,
                    num_workers=a.num_workers, force=a.force)


if __name__ == "__main__":
    main()
