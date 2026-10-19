"""Reproduce the paper's tables and figures from evals/experiments.yaml.

    python -m evals.reproduce --list
    python -m evals.reproduce table1_kolmogorov [--stage rollout|metrics|report]
    python -m evals.reproduce all --stage report
"""

import argparse
import copy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

from evals import figures as F
from evals import reduce as Rd
from evals.metrics import compute_metrics
from evals.rollout import Paths, RolloutSpec, add_path_args, paths_from_args, run_rollout, store_dir
from evals.core import run_config

EXPERIMENTS = Path(__file__).with_name("experiments.yaml")
SPEC_KEYS = {"rollout", "nfe", "K", "ckpt", "sampler", "amplitude", "seed", "split",
             "window", "batch_size", "subset", "sigma_scale"}
MODEL_KINDS = {"panels", "panel_single", "latent_terciles", "vae_table", "timing"}


def load_experiments(path: Path = EXPERIMENTS) -> Dict[str, dict]:
    doc = yaml.safe_load(open(path))
    return {k: v for k, v in doc["experiments"].items()}



def row_specs(exp: dict) -> List[Tuple[dict, RolloutSpec]]:
    """(row, spec) for every store an experiment reads; NFE sweeps expand per NFE."""
    out = []
    base = exp.get("store", {})
    for row in exp.get("rows", []):
        st = {**base, **row.get("store", {})}
        unknown = set(st) - SPEC_KEYS
        if unknown:
            raise ValueError(f"unknown store keys {unknown}")
        nfes = exp.get("nfes") if exp["kind"] == "nfe_sweep" else [st.get("nfe", 5)]
        for nfe in nfes:
            spec = RolloutSpec(run=row["run"], **{**st, "nfe": nfe})
            out.append(({**row, "nfe": nfe}, spec))
    return out


def stage_rollout(exp: dict, paths: Paths, device: str, num_workers: int) -> None:
    for _, spec in row_specs(exp):
        run_rollout(copy.deepcopy(spec), paths, device=device, num_workers=num_workers)


def stage_metrics(exp: dict, paths: Paths, device: str, num_workers: int) -> None:
    for _, spec in row_specs(exp):
        compute_metrics(copy.deepcopy(spec), paths, k_subsets=exp.get("k_subsets"),
                        device=device, num_workers=num_workers)


def metrics_path(spec: RolloutSpec, paths: Paths) -> Path:
    spec = copy.deepcopy(spec)
    spec.resolve(run_config(paths.base_dir, spec.run)["data"]["pde"])
    return store_dir(spec, paths) / "metrics.csv"


def table_of(spec: RolloutSpec, paths: Paths, K: Optional[int] = None) -> pd.DataFrame:
    p = metrics_path(spec, paths)
    if not p.exists():
        raise FileNotFoundError(f"{p} missing: run the rollout and metrics stages first")
    return Rd.load_table(p, K=K)



def summarize(df: pd.DataFrame, exp: dict, cols: List[str]) -> Dict[str, Tuple[float, float]]:
    """{column: (value, +-)} over steps 1..steps (evals/reduce.py)."""
    s = Rd.per_batch(df, steps=exp.get("steps"))["summary"]
    return {c: (s[c], s.get(f"{c}_err", float("nan"))) for c in cols if c in s}


def per_step(df: pd.DataFrame, exp: dict) -> pd.DataFrame:
    return Rd.per_batch(df, steps=exp.get("steps"))["per_step"]



def _fmt(v: float, s: float, scale: float, prec: int) -> str:
    if v != v:
        return "--"
    txt = f"{v / scale:.{prec}f}"
    if s == s:
        txt += f" $\\pm$ {s / scale:.{prec}f}"
    return txt


def write_table(name: str, exp: dict, rows: List[dict], out: Path) -> pd.DataFrame:
    """CSV at full precision and a LaTeX table with the best value in bold."""
    mets = exp["metrics"]
    df = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{name}.csv", index=False)

    best = {}
    for m in mets:
        col = m["col"]
        if col not in df.columns:
            continue
        vals = df[col].to_numpy(float)
        if m.get("better") == "none":
            continue
        if m.get("better", "low") == "one":
            key = np.abs(vals - 1.0)
        elif m.get("better") == "high":
            key = -vals
        else:
            key = vals
        best[col] = int(np.nanargmin(key)) if np.isfinite(key).any() else None

    header = " & ".join(["Model"] + [m.get("label", m["col"]) for m in mets]) + r" \\"
    lines = [r"\begin{tabular}{l" + "c" * len(mets) + "}", r"\toprule", header, r"\midrule"]
    for i, r in df.iterrows():
        cells = [str(r["label"])]
        for m in mets:
            col = m["col"]
            cell = _fmt(r.get(col, float("nan")), r.get(f"{col}_spread", float("nan")),
                        m.get("scale", 1.0), m.get("prec", 3))
            if best.get(col) == i:
                cell = r"\textbf{" + cell + "}"
            cells.append(cell)
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / f"{name}.tex").write_text("\n".join(lines) + "\n")
    print(f"  saved {out / name}.csv / .tex")
    print(df[["label"] + [m["col"] for m in mets if m["col"] in df.columns]].to_string(index=False))
    return df


def _average_groups(rows: List[dict], cols: List[str]) -> List[dict]:
    """Average rows that share a `group`."""
    out, seen = [], {}
    for r in rows:
        g = r.get("group")
        if g is None:
            out.append(r)
        elif g in seen:
            seen[g].append(r)
        else:
            seen[g] = [r]
            out.append({"__group__": g})
    merged = []
    for r in out:
        if "__group__" not in r:
            merged.append(r)
            continue
        members = seen[r["__group__"]]
        m = {"label": members[0]["label"], "runs": "+".join(x["run"] for x in members)}
        for c in cols:
            m[c] = float(np.mean([x[c] for x in members]))
            m[f"{c}_spread"] = float(np.mean([x[f"{c}_spread"] for x in members]))
        merged.append(m)
    return merged


def report_table(name, exp, paths, out):
    cols = [m["col"] for m in exp["metrics"]]
    rows = []
    for row, spec in row_specs(exp):
        vals = summarize(table_of(spec, paths), exp, cols)
        r = {"label": row["label"], "run": row["run"], "group": row.get("group")}
        for c in cols:
            v, s = vals.get(c, (float("nan"), float("nan")))
            r[c], r[f"{c}_spread"] = v, s
        rows.append(r)
    rows = _average_groups(rows, cols)
    for r in rows:
        r.pop("group", None)
    write_table(name, exp, rows, out)


def report_per_var(name, exp, paths, out):
    """One table per ERA5 variable (Table 15)."""
    for v in exp["vars"]:
        sub = copy.deepcopy(exp)
        sub["metrics"] = [{**m, "col": f"{m['col']}_{v}"} for m in exp["metrics"]]
        report_table(f"{name}_{v}", sub, paths, out)


def report_paired(name, exp, paths, out):
    tables = {row["label"]: table_of(spec, paths) for row, spec in row_specs(exp)}
    ref = tables[exp["reference"]]
    rows = []
    for label, df in tables.items():
        if label == exp["reference"]:
            continue
        for r in Rd.paired(df, ref, exp["metrics"], steps=exp.get("steps")):
            rows.append({"label": label, "vs": exp["reference"], **r})
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / f"{name}.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))



def report_per_step(name, exp, paths, out):
    curves = {}
    for row, spec in row_specs(exp):
        curves[row["label"]] = per_step(table_of(spec, paths), exp)
    out.mkdir(parents=True, exist_ok=True)
    pd.concat([c.assign(label=k) for k, c in curves.items()]).to_csv(out / f"{name}.csv", index=False)
    F.per_step_curves(curves, exp["metrics"], out / name)


def _sweep_points(exp, paths, x_of) -> Dict[str, pd.DataFrame]:
    pts: Dict[str, List[dict]] = {}
    cols = exp["metrics"]
    for row, spec in row_specs(exp):
        vals = summarize(table_of(spec, paths), exp, cols)
        rec = {"x": x_of(row, spec)}
        for c in cols:
            rec[c], rec[f"{c}_err"] = vals.get(c, (float("nan"), float("nan")))
        pts.setdefault((row.get("panel"), row["label"]), []).append(rec)
    return {k: pd.DataFrame(v) for k, v in pts.items()}


def report_sweep(name, exp, paths, out, x_of, xlabel, xscale=None):
    pts = _sweep_points(exp, paths, x_of)
    out.mkdir(parents=True, exist_ok=True)
    pd.concat([d.assign(panel=p, label=l) for (p, l), d in pts.items()]).to_csv(out / f"{name}.csv", index=False)
    for panel in dict.fromkeys(p for p, _ in pts):
        sub = {l: d for (p, l), d in pts.items() if p == panel}
        tag = name if panel is None else f"{name}_{panel}"
        F.metric_vs_x(sub, "x", exp["metrics"], out / tag, xlabel=xlabel,
                      xscale=exp.get("xscale", xscale), err=exp.get("errorbars", False))


def report_k_sweep(name, exp, paths, out):
    pts = {}
    for row, spec in row_specs(exp):
        recs = []
        for k in exp["k_subsets"]:
            vals = summarize(table_of(spec, paths, K=k), exp, exp["metrics"])
            rec = {"x": k}
            for c in exp["metrics"]:
                rec[c], rec[f"{c}_err"] = vals[c]
            recs.append(rec)
        pts[row["label"]] = pd.DataFrame(recs)
    out.mkdir(parents=True, exist_ok=True)
    pd.concat([d.assign(label=l) for l, d in pts.items()]).to_csv(out / f"{name}.csv", index=False)
    F.metric_vs_x(pts, "x", exp["metrics"], out / name, xlabel="ensemble size $K$",
                  xscale="log2", err=True, ncols=exp.get("ncols"))


def report_pit(name, exp, paths, out):
    steps = range(1, exp["steps"] + 1)
    hists = {row["label"]: Rd.pooled_ranks(table_of(spec, paths), steps) for row, spec in row_specs(exp)}
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({l: h / h.sum() for l, h in hists.items()}).rename_axis("rank").to_csv(out / f"{name}.csv")
    F.pit_histograms(hists, out / name)


def report_reliability(name, exp, paths, out):
    n = exp["steps"]
    panels = {"Reliability (one-step)": {}, f"Reliability (rollout 1–{n})": {}}
    rows = []
    for row, spec in row_specs(exp):
        df = table_of(spec, paths)
        for title, steps in zip(panels, (1, n)):
            p, obs = Rd.reliability(Rd.pooled_ranks(df, range(1, steps + 1)))
            rmsce = Rd.per_batch(df, steps=steps)["summary"]["rmsce"]
            panels[title][row["label"]] = (p, obs, rmsce)
            rows += [{"panel": title, "label": row["label"], "p": a, "coverage": b} for a, b in zip(p, obs)]
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / f"{name}.csv", index=False)
    F.reliability_diagrams(panels, out / name)


def report_scatter(name, exp, paths, out):
    width = exp.get("width", "width@90")
    pts = {}
    for row, spec in row_specs(exp):
        vals = summarize(table_of(spec, paths), exp, [width, "rmsce"])
        pts[row["label"]] = (vals[width][0], vals["rmsce"][0])
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"label": k, width: v[0], "rmsce": v[1]} for k, v in pts.items()]).to_csv(
        out / f"{name}.csv", index=False)
    F.calibration_scatter(pts, out / name, xlabel=f"PI width @{width[-2:]}%  (absolute sharpness)",
                          ylabel="RMSCE  (calibration error)")


def report_enstrophy_windows(name, exp, paths, out):
    from scipy.stats import wasserstein_distance

    W = exp["window"]
    rows = []
    for row, spec in row_specs(exp):
        df = table_of(spec, paths)
        mcols = [c for c in df.columns if c.startswith("ens_m")]
        n = int(df["step"].max()) // W
        for w in range(n):
            sel = df[(df["step"] > w * W) & (df["step"] <= (w + 1) * W)]
            pred = sel[mcols].to_numpy(float).ravel()
            gt = sel["ens_gt"].to_numpy(float)
            rows.append({"label": row["label"], "window": w, "window_label": f"{w * W + 1}-{(w + 1) * W}",
                         "wasserstein": wasserstein_distance(pred, gt),
                         "n_pred": pred.size, "n_gt": gt.size})
    table = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / f"{name}.csv", index=False)
    F.wasserstein_windows(table, out / name)


def report_enstrophy_traj(name, exp, paths, out):
    for idx in exp["samples"]:
        curves = {}
        for row, spec in row_specs(exp):
            df = table_of(spec, paths)
            d = df[df["index"] == idx].sort_values("step")
            mcols = sorted([c for c in d.columns if c.startswith("ens_m")], key=lambda c: int(c[5:]))
            z0 = float(d["ens_ic"].iloc[0])
            t = np.concatenate([[0], d["step"].to_numpy()])
            gt = np.concatenate([[z0], d["ens_gt"].to_numpy(float)]) / z0
            pred = np.concatenate([np.full((len(mcols), 1), z0), d[mcols].to_numpy(float).T], 1) / z0
            curves[row["label"]] = {"t": t, "gt": gt, "pred": pred}
        F.enstrophy_trajectories(curves, out / f"{name}_sample{idx}")



@torch.inference_mode()
def decode_sample(spec: RolloutSpec, paths: Paths, index: int, steps: int, device: str,
                  member: int = 0) -> Dict:
    """Truth, mean, std and member fields of one sample, (T+1, H, W, C)."""
    from evals.core import Conditioning, eval_data, eval_loader, load_processor

    spec = copy.deepcopy(spec)
    pde = run_config(paths.base_dir, spec.run)["data"]["pde"]
    spec.resolve(pde)
    sdir = store_dir(spec, paths)
    dm = eval_data(pde, split=spec.split, window=spec.window, batch_size=spec.batch_size,
                   num_workers=0, overrides=paths.data_overrides)
    proc, _ = load_processor(paths.base_dir, spec.run, nfe=spec.nfe, ckpt_type=spec.ckpt,
                             normalizer=dm.normalizer, device=device, ae_dir=paths.ae_dir)
    seen = 0
    for bi, batch in enumerate(eval_loader(dm, spec.subset)):
        B = batch["input_fields"].shape[0]
        if index < seen + B:
            b = index - seen
            break
        seen += B
    else:
        raise IndexError(f"sample {index} beyond the evaluation set")
    st = torch.load(sdir / f"batch_{bi:05d}.pt", map_location="cpu", weights_only=False)
    cond = Conditioning(proc, batch, device)
    K = st["z"].shape[0]
    T = min(steps, st["z"].shape[1])
    u0 = proc.normalizer.denormalize(batch["input_fields"][b : b + 1].to(device))[0].cpu()
    mem = [u0.unsqueeze(0).expand(K, *u0.shape)]
    for t in range(T):
        z = st["z"][:, t, b].to(device)
        c = cond.at(t + 1)
        c = None if c is None else c[b : b + 1].expand(K, -1)
        mem.append(proc.normalizer.denormalize(proc.decode(z, c)).cpu())
    mem = torch.stack(mem, 0)
    truth = proc.normalizer.denormalize(batch["output_fields"][b, : T + 1].to(device)).cpu()
    return {"truth": truth, "mean": mem.mean(1), "std": mem.std(1), "member": mem[:, member]}


def report_panels(name, exp, paths, out, device):
    results = {}
    for row, spec in row_specs(exp):
        results[row["label"]] = decode_sample(spec, paths, exp.get("sample", 0), exp["steps"], device,
                                              member=exp.get("member", 0))
    T = min(r["truth"].shape[0] for r in results.values())
    frames = F.frame_indices(T, exp["frame_gap"])
    geo = exp.get("geo", False)
    channels = exp.get("channels", {"field": 0})
    for cname, ch in channels.items():
        opts = exp.get("channel_opts", {}).get(cname, {})
        tag = name if len(channels) == 1 else f"{name}_{cname}"
        F.figure_traj(results, ch, frames, out / f"{tag}_traj", geo=geo, view=exp.get("view", "member"),
                      unit=opts.get("unit", ""), cmap=opts.get("cmap"),
                      vmax_percentile=opts.get("vmax_percentile", 100.0),
                      vmin_zero=opts.get("vmin_zero", False), symmetric=opts.get("symmetric", False))
        F.figure_errci(results, ch, frames, out / f"{tag}_errci", geo=geo, unit=opts.get("unit", ""))


def report_panel_single(name, exp, paths, out, device):
    (row, spec), = row_specs(exp)
    res = decode_sample(spec, paths, exp.get("sample", 0), exp["steps"], device)
    frames = F.frame_indices(res["truth"].shape[0], exp["frame_gap"])
    F.figure_single(res, 0, frames, out / name, row["label"])


def report(name: str, exp: dict, paths: Paths, out: Path, device: str) -> None:
    kind = exp["kind"]
    print(f"[report] {name} ({kind})")
    if kind == "table":
        report_table(name, exp, paths, out)
    elif kind == "per_var":
        report_per_var(name, exp, paths, out)
    elif kind == "paired":
        report_paired(name, exp, paths, out)
    elif kind == "per_step":
        report_per_step(name, exp, paths, out)
    elif kind == "sweep":
        report_sweep(name, exp, paths, out, lambda row, spec: row["x"], exp.get("xlabel", "x"))
    elif kind == "nfe_sweep":
        report_sweep(name, exp, paths, out, lambda row, spec: spec.nfe, "NFE")
    elif kind == "k_sweep":
        report_k_sweep(name, exp, paths, out)
    elif kind == "pit":
        report_pit(name, exp, paths, out)
    elif kind == "reliability":
        report_reliability(name, exp, paths, out)
    elif kind == "scatter":
        report_scatter(name, exp, paths, out)
    elif kind == "enstrophy_windows":
        report_enstrophy_windows(name, exp, paths, out)
    elif kind == "enstrophy_traj":
        report_enstrophy_traj(name, exp, paths, out)
    elif kind == "panels":
        report_panels(name, exp, paths, out, device)
    elif kind == "panel_single":
        report_panel_single(name, exp, paths, out, device)
    elif kind == "latent_terciles":
        from evals.analysis.latent_terciles import report as lt_report
        lt_report(name, exp, paths, out, device, row_specs(exp))
    elif kind == "vae_table":
        from evals.analysis.vae import report as vae_report
        vae_report(name, exp, paths, out, device, row_specs(exp), summarize, table_of)
    elif kind == "timing":
        from evals.analysis.timing import report as timing_report
        timing_report(name, exp, paths, out, device)
    else:
        raise ValueError(f"unknown experiment kind {kind!r}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("experiments", nargs="*", help="experiment names, or 'all'")
    p.add_argument("--list", action="store_true")
    p.add_argument("--stage", default="all", choices=["rollout", "metrics", "report", "all"])
    p.add_argument("--out", default="results/paper", help="where tables and figures go")
    p.add_argument("--file", default=str(EXPERIMENTS), help="experiment list")
    p.add_argument("--num-workers", type=int, default=2)
    add_path_args(p)
    a = p.parse_args()

    exps = load_experiments(Path(a.file))
    if a.list or not a.experiments:
        w = max(len(k) for k in exps)
        for k, v in exps.items():
            print(f"{k:{w}s}  {v['kind']:18s}  {v.get('paper', '')}")
        return
    names = list(exps) if a.experiments == ["all"] else a.experiments
    paths = paths_from_args(a)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision("high")
    for name in names:
        exp = exps[name]
        if exp["kind"] not in ("timing",):
            if a.stage in ("rollout", "all"):
                stage_rollout(exp, paths, device, a.num_workers)
            if a.stage in ("metrics", "all"):
                stage_metrics(exp, paths, device, a.num_workers)
        if a.stage in ("report", "all"):
            report(name, exp, paths, Path(a.out), device)


if __name__ == "__main__":
    main()
