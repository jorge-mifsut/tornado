"""VAE studies (Tables 6, 7): reconstruction VRMSE, posterior KL, and RMSCE of K
decoded posterior samples, next to the paired TORNADO run's metrics."""

from pathlib import Path

import pandas as pd
import torch

from common.loss import VRMSE
from common.utils import get_yaml
from evals import scores as S
from evals.core import eval_data, find_ckpt
from evals.reduce import OnlineMeanVar
from modules.ae_module import AutoencoderModule


def load_vae(base_dir: Path, run: str, ckpt_type: str, normalizer, device: str) -> AutoencoderModule:
    cfg = get_yaml(str(Path(base_dir) / run / "config.yml"))
    ae = AutoencoderModule(cfg, normalizer=normalizer)
    ckpt = torch.load(str(find_ckpt(Path(base_dir) / run, ckpt_type)), map_location=device,
                      weights_only=False)
    ae.load_state_dict(ckpt["state_dict"])
    ae.eval().to(device)
    for p in ae.parameters():
        p.requires_grad_(False)
    return ae, cfg["model"].get("kl_weight")


@torch.inference_mode()
def vae_metrics(ae, dm, *, K: int, steps: int, device: str, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    recon, recon_n, kl, n = 0.0, 0, 0.0, 0
    rmsce = OnlineMeanVar()
    for batch in dm.val_dataloader():
        cond = batch.get("constant_scalars")
        cond = None if cond is None else cond.to(device).float()
        x = batch["input_fields"].to(device)
        B = x.shape[0]
        _, post = ae.encode(x, cond)
        rec = ae.decode(post.mean, cond)
        recon += float(VRMSE(dm.normalizer.denormalize(rec), dm.normalizer.denormalize(x))) * B
        recon_n += B
        kl += float(post.kl().sum())
        n += B

        gt = batch["output_fields"]
        for t in range(1, min(steps, gt.shape[1] - 1) + 1):
            y_n = gt[:, t].to(device)
            _, post_t = ae.encode(y_n, cond)
            lat = torch.stack([post_t.sample() for _ in range(K)], 0)
            cflat = None if cond is None else cond.unsqueeze(0).expand(K, *cond.shape).reshape(K * B, -1)
            dec = ae.decode(lat.reshape(K * B, *lat.shape[2:]), cflat).reshape(K, B, *y_n.shape[1:])
            samples = dm.normalizer.denormalize(dec)
            y = dm.normalizer.denormalize(y_n)
            counts = S.rank_counts(samples, y).cpu().numpy()
            rmsce.update(float(S.rmsce_from_ranks(counts)), weight=B)
    return {"vae_recon": recon / max(recon_n, 1), "vae_kl": kl / max(n, 1), "vae_rmsce": rmsce.mean}


def report(name, exp, paths, out: Path, device: str, specs, summarize, table_of) -> None:
    from evals.reduce import per_batch

    rows = []
    for row, spec in specs:
        pde = get_yaml(str(paths.base_dir / row["vae"] / "config.yml"))["data"]["pde"]
        dm = eval_data(pde, batch_size=spec.batch_size, num_workers=0, overrides=paths.data_overrides)
        ae, beta = load_vae(paths.base_dir, row["vae"], row.get("vae_ckpt", "best"), dm.normalizer, device)
        r = {"label": row.get("label", row["vae"]), "beta": row.get("beta", beta), "vae": row["vae"],
             "run": row["run"]}
        r.update(vae_metrics(ae, dm, K=spec.K, steps=exp["steps"], device=device))
        df = table_of(spec, paths)
        r["tornado_rmsce_step1"] = float(per_batch(df, steps=1)["summary"]["rmsce"])
        vals = summarize(df, exp, ["mse", "rmsce", "sharpness"])
        r.update({f"tornado_{k}": v[0] for k, v in vals.items()})
        rows.append(r)
        del ae
    table = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / f"{name}.csv", index=False)
    print(table.to_string(index=False))
