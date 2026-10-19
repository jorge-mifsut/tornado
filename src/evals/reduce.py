"""Reduce metrics.csv to reported numbers.

Each metric is computed per step on groups of GROUP trajectories (RMSCE from the
group's rank histogram, SSR = RMS spread / RMSE), then averaged over groups and
steps. +- is 1.96 x the standard error over (step, group) values.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from evals import scores as S

GROUP = 32  # the evaluation batch size of every data config

_MEAN_COLS = ["mse", "mae", "crps", "cov@90", "width@90", "cov@95", "width@95", "energy"]
_RMS_COLS = ["sharpness", "sharpness_z", "rmse_z", "rmse_vae"]


def load_table(path: Path, K: Optional[int] = None) -> pd.DataFrame:
    """A metrics.csv, restricted to one ensemble size (default: the largest)."""
    df = pd.read_csv(path)
    if "K" in df.columns:
        K = int(df["K"].max()) if K is None else int(K)
        df = df[df["K"] == K].reset_index(drop=True)
        df = df.dropna(axis=1, how="all")
    return df


def rank_cols(df: pd.DataFrame, var: Optional[str] = None) -> List[str]:
    """rank_0 .. rank_K, or ERA5's per-variable rank_<var>_0 .. rank_<var>_K."""
    prefix = "rank_" if var is None else f"rank_{var}_"
    cols = [c for c in df.columns if c.startswith(prefix) and c[len(prefix):].isdigit()]
    return sorted(cols, key=lambda c: int(c[len(prefix):]))


def era5_vars(df: pd.DataFrame) -> List[str]:
    """ERA5 variables with per-variable columns in the table."""
    return [v for v in S.ERA5_VARS if f"sharpness_{v}" in df.columns]


@dataclass
class OnlineMeanVar:
    """Weighted running mean/variance."""

    mean_: float = 0.0
    m2: float = 0.0
    n: int = 0

    def update(self, x: float, weight: int = 1):
        w = int(weight)
        if w <= 0:
            return
        x = float(x)
        n1 = self.n + w
        delta = x - self.mean_
        self.mean_ += (w / n1) * delta
        self.m2 += w * delta * (x - self.mean_)
        self.n = n1

    @property
    def mean(self) -> float:
        return self.mean_

    @property
    def stderr(self) -> float:
        var = self.m2 / max(1, self.n)
        return float(np.sqrt(max(0.0, var)) / np.sqrt(max(1, self.n)))


def _ratio(num: float, den_sq: float) -> float:
    return num / (np.sqrt(den_sq) + 1e-12) if den_sq > 1e-24 else float("nan")


def group_values(g: pd.DataFrame) -> Dict[str, float]:
    """Metrics of one (step, group), computed on the group as a whole."""
    out = {c: float(g[c].mean()) for c in _MEAN_COLS if c in g.columns}
    out.update({c: float(np.sqrt((g[c] ** 2).mean())) for c in _RMS_COLS if c in g.columns})
    out["rmse"] = float(np.sqrt(out["mse"]))
    out["ssr"] = _ratio(out["sharpness"], out["mse"])
    ranks = rank_cols(g)
    if ranks:
        counts = g[ranks].to_numpy(dtype=np.float64).sum(axis=0)
        out["rmsce"] = float(S.rmsce_from_ranks(counts))
        out["mace"] = float(S.mace_from_ranks(counts))
    if "rmse_z" in out:
        out["ssr_z"] = _ratio(out["sharpness_z"], out["rmse_z"] ** 2)
    if "rmse_vae" in out:
        out["ssr_corrected"] = _ratio(out["sharpness"], out["mse"] - out["rmse_vae"] ** 2)
    for v in era5_vars(g):
        out[f"mse_{v}"] = float(g[f"mse_{v}"].mean())
        out[f"crps_{v}"] = float(g[f"crps_{v}"].mean())
        out[f"sharpness_{v}"] = float(np.sqrt((g[f"sharpness_{v}"] ** 2).mean()))
        out[f"ssr_{v}"] = _ratio(out[f"sharpness_{v}"], out[f"mse_{v}"])
        rv = rank_cols(g, v)
        if rv:
            out[f"rmsce_{v}"] = float(S.rmsce_from_ranks(g[rv].to_numpy(dtype=np.float64).sum(axis=0)))
    return out


def _grouped(df: pd.DataFrame, steps: Optional[int]) -> pd.DataFrame:
    if steps is not None:
        df = df[df["step"] <= steps]
    df = df.copy()
    order = {i: k for k, i in enumerate(sorted(df["index"].unique()))}
    df["group"] = df["index"].map(order) // GROUP
    return df


def _with_abs_log(d: Dict[str, float]) -> Dict[str, float]:
    """Add abs_log_<k> = |log ssr*| for every SSR key."""
    for k in [k for k in d if k.startswith("ssr") and not k.endswith("_err")]:
        if np.isfinite(d[k]) and d[k] > 0:
            d[f"abs_log_{k}"] = abs(float(np.log(d[k])))
    return d


def per_batch(df: pd.DataFrame, steps: Optional[int] = None) -> Dict[str, object]:
    """Per-step means (+-1.96 stderr over groups) and the mean over steps 1..steps."""
    df = _grouped(df, steps)
    R = int(df["step"].max())
    per_t: Dict[str, List[OnlineMeanVar]] = {}
    overall: Dict[str, OnlineMeanVar] = {}
    for (t, _), g in df.groupby(["step", "group"], sort=True):
        for m, v in group_values(g).items():
            if not np.isfinite(v):
                continue
            per_t.setdefault(m, [OnlineMeanVar() for _ in range(R)])[t - 1].update(v, weight=len(g))
            overall.setdefault(m, OnlineMeanVar()).update(v, weight=len(g))
    rows = []
    for t in range(R):
        row = {"step": t + 1}
        for m, acc in per_t.items():
            if acc[t].n:
                row[m] = acc[t].mean
                row[f"{m}_err"] = 1.96 * acc[t].stderr
        rows.append(_with_abs_log(row))
    summary = {m: acc.mean for m, acc in overall.items()}
    summary.update({f"{m}_err": 1.96 * acc.stderr for m, acc in overall.items()})
    return {"per_step": pd.DataFrame(rows), "summary": _with_abs_log(summary)}


def paired(df: pd.DataFrame, ref: pd.DataFrame, metrics: Sequence[str],
           steps: Optional[int] = None) -> List[Dict]:
    """Per-(step, group) differences to a reference model; negative is better.

    SSR is scored as |SSR - 1|; p_value is a paired t-test over groups.
    """
    from scipy import stats

    def values(x):
        x = _grouped(x, steps)
        return pd.DataFrame([{"step": t, "group": k, **group_values(g)}
                             for (t, k), g in x.groupby(["step", "group"], sort=True)])

    a, b = values(df), values(ref)
    if not set(df["index"]) == set(ref["index"]):
        raise ValueError("paired comparison needs the same evaluation trajectories")
    cols = [c for c in metrics if c in a.columns and c in b.columns]
    m = a.merge(b, on=["step", "group"], suffixes=("", "_ref"))
    rows = []
    for c in cols:
        x = (m[c] - 1.0).abs() if c == "ssr" else m[c]
        y = (m[f"{c}_ref"] - 1.0).abs() if c == "ssr" else m[f"{c}_ref"]
        d = (x - y).to_numpy()
        p = stats.ttest_1samp(d, 0.0).pvalue if d.size > 1 and d.std() > 0 else np.nan
        rows.append({
            "metric": c,
            "delta_mean": float(d.mean()),
            "rel_delta_pct": 100.0 * float(d.mean()) / float(y.mean()),
            "win_rate": float((d < 0).mean()),
            "n_groups": int(d.size),
            "p_value": float(p),
        })
    return rows


def pooled_ranks(df: pd.DataFrame, steps: Optional[Sequence[int]] = None) -> np.ndarray:
    """Rank histogram summed over all entries of the given steps (for figures)."""
    if steps is not None:
        df = df[df["step"].isin(list(steps))]
    return df[rank_cols(df)].to_numpy(dtype=np.float64).sum(axis=0)


def reliability(counts: np.ndarray, num_bins: int = 50):
    """(levels p, observed coverage G(p)) of a rank histogram."""
    import torch

    ps = torch.linspace(0.0, 1.0, num_bins).numpy().astype(np.float64)
    return ps, S.pit_cdf_from_ranks(counts, num_bins)
