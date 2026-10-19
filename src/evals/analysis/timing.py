"""Wall-clock per latent forecast step for K members of a batch (Table 10; H100, B=32, K=16)."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from evals.core import TORNADO_MODELS, Conditioning, eval_data, load_processor, run_config


@torch.inference_mode()
def time_step(paths, run: str, *, nfe: int, ckpt: str, batch_size: int, K: int,
              warmup: int = 50, iters: int = 200, repeats: int = 3) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("timing needs a GPU")
    device = "cuda"
    pde = run_config(paths.base_dir, run)["data"]["pde"]
    dm = eval_data(pde, batch_size=batch_size, num_workers=0, overrides=paths.data_overrides)
    proc, info = load_processor(paths.base_dir, run, nfe=nfe, ckpt_type=ckpt,
                                normalizer=dm.normalizer, device=device, ae_dir=paths.ae_dir)
    batch = next(iter(dm.val_dataloader()))
    u0 = batch["input_fields"][:batch_size].to(device)
    B = u0.shape[0]
    cond = Conditioning(proc, {k: v[:B] if torch.is_tensor(v) else v for k, v in batch.items()}, device)
    z = proc.encode(u0, cond.at(0), return_posterior=False)
    z = z.unsqueeze(0).expand(K, *z.shape).reshape(K * B, *z.shape[1:]).contiguous()
    c = cond.flat(0, K)
    method = "em" if info["model"] in TORNADO_MODELS else None

    cur = z.clone()
    for _ in range(warmup):
        cur = proc.forward(cur, c, method_val=method)
    times = []
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(repeats):
        cur = z.clone()
        torch.cuda.synchronize()
        start.record()
        for _ in range(iters):
            cur = proc.forward(cur, c, method_val=method)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / iters)
    return {"ms_per_step": float(np.mean(times)), "ms_std": float(np.std(times)),
            "gpu": torch.cuda.get_device_name(0), "B": B, "K": K, "nfe": nfe}


def report(name, exp, paths, out: Path, device: str) -> None:
    rows = []
    for row in exp["rows"]:
        for nfe in exp["nfes"]:
            r = time_step(paths, row["run"], nfe=nfe, ckpt=row.get("ckpt", "best"),
                          batch_size=exp.get("batch_size", 32), K=exp.get("K", 16))
            rows.append({"label": row["label"], "run": row["run"], **r})
            print(f"  {row['label']:10s} NFE={nfe:3d}  {r['ms_per_step']:.2f} ms/step")
    table = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / f"{name}.csv", index=False)
    print(table.pivot(index="label", columns="nfe", values="ms_per_step").to_string())
