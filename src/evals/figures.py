"""Paper figures, from reductions (evals/reduce.py) or decoded fields. Colours are fixed per method."""

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

COLORS = {"TORNADO": "#494e84", "FM": "#dd164c", "DDIM": "#da9100", "SI": "#00a693"}
EXTRA = ["#74afda", "#8e44ad", "#000000", "#7f7f7f", "#2ca02c"]
LINESTYLES = {"TORNADO": "-", "FM": "--", "DDIM": "-.", "SI": ":"}
METRIC_TITLES = {
    "mse": "MSE", "crps": "CRPS", "rmsce": "RMSCE", "ssr": "Spread-Skill",
    "abs_log_ssr": r"$|\log \mathrm{SSR}|$", "sharpness": "Sharpness", "rmse": "RMSE",
    "width@95": "Interval width @95%", "width@90": "Interval width @90%",
    "cov@95": "Coverage @95%", "energy": "Energy score", "mae": "MAE",
}
REF_LINES = {"ssr": 1.0, "cov@95": 0.95, "cov@90": 0.90}

TRAJ_CMAP = sns.color_palette("icefire", as_cmap=True)
ERR_CMAP = sns.color_palette("rocket", as_cmap=True)
CI_CMAP = sns.color_palette("mako", as_cmap=True)


def color_of(label: str, i: int = 0) -> str:
    for key, c in COLORS.items():
        if label == key or label.startswith(key + " "):
            return c
    return EXTRA[i % len(EXTRA)]


def style_of(label: str, i: int = 0) -> Dict:
    ls = LINESTYLES.get(label, ["-", "--", "-.", ":"][i % 4])
    return {"color": color_of(label, i), "linestyle": ls}


def apply_style(hide_left_spine: bool = False) -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": not hide_left_spine,
        "axes.edgecolor": "#b0b0b0", "axes.labelcolor": "#2b2b2b", "text.color": "#2b2b2b",
        "xtick.color": "#6b6b6b", "ytick.color": "#6b6b6b",
        "xtick.direction": "out", "ytick.direction": "out", "legend.frameon": False,
    })


def soften(hex_color: str, amount: float) -> tuple:
    """Blend toward white, keeping hue."""
    r, g, b = mcolors.to_rgb(hex_color)
    a = max(0.0, min(1.0, amount))
    return tuple(c + (1.0 - c) * a for c in (r, g, b))


def save(fig, out: Path, dpi: int = 220) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.with_suffix('.png')} (+ .pdf)")



def per_step_curves(curves: Dict[str, "pd.DataFrame"], metrics: Sequence[str], out: Path) -> None:
    """One panel per metric, one line per method (paper Figs 3, 14, 15).

    curves[label] has a `step` column and a column per metric.
    """
    apply_style()
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 3.6))
    axes = np.atleast_1d(axes)
    for i, (label, df) in enumerate(curves.items()):
        st = style_of(label, i)
        for ax, m in zip(axes, metrics):
            if m not in df.columns:
                continue
            x, y = df["step"].to_numpy(), df[m].to_numpy(float)
            ax.plot(x, y, linewidth=1.8, label=label, **st)
    for ax, m in zip(axes, metrics):
        ax.set_title(METRIC_TITLES.get(m, m), fontsize=13)
        ax.set_xlabel("t")
        ax.grid(True, color="#ededed", linewidth=0.8)
        if m in REF_LINES:
            ax.axhline(REF_LINES[m], color="#9a9a9a", linestyle=(0, (4, 3)), linewidth=1.0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", fontsize=10)
    fig.tight_layout(rect=[0, 0, 0.9, 1])
    save(fig, out)


def metric_vs_x(points: Dict[str, "pd.DataFrame"], x: str, metrics: Sequence[str], out: Path, *,
                xlabel: str, xscale: Optional[str] = None, err: bool = False,
                ncols: Optional[int] = None) -> None:
    """Metric against a swept quantity (NFE, K, lambda_ED, sigma), one line per method."""
    apply_style()
    ncols = ncols or len(metrics)
    nrows = int(np.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.9 * ncols, 3.3 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for i, (label, df) in enumerate(points.items()):
        df = df.sort_values(x)
        st = style_of(label, i)
        for ax, m in zip(axes, metrics):
            if m not in df.columns:
                continue
            e = df[f"{m}_err"].to_numpy(float) if err and f"{m}_err" in df.columns else None
            ax.errorbar(df[x], df[m], yerr=e, marker="o", markersize=4, linewidth=1.8,
                        capsize=2.5 if e is not None else 0, label=label, **st)
    for ax, m in zip(axes, metrics):
        ax.set_title(METRIC_TITLES.get(m, m), fontsize=13)
        ax.set_xlabel(xlabel)
        ax.grid(True, color="#ededed", linewidth=0.8)
        if xscale == "log2":
            ax.set_xscale("log", base=2)
        elif xscale == "symlog":
            ax.set_xscale("symlog", linthresh=0.01)
        elif xscale:
            ax.set_xscale(xscale)
        if m in REF_LINES:
            ax.axhline(REF_LINES[m], color="#9a9a9a", linestyle=(0, (4, 3)), linewidth=1.0)
    for ax in axes[len(metrics):]:
        ax.set_visible(False)
    if len(points) > 1:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=10,
                   bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    save(fig, out)



def pit_histograms(hists: Dict[str, np.ndarray], out: Path, soften_by: float = 0.2) -> None:
    """PIT histograms with ONE shared y-axis (paper Figs 2, 13)."""
    apply_style(hide_left_spine=True)
    n = len(hists)
    fig, axes = plt.subplots(1, n, figsize=(3.1 * n, 2.9), sharey=True)
    axes = np.atleast_1d(axes)
    freqs = {k: h / max(h.sum(), 1e-300) for k, h in hists.items()}
    hi = max(f.max() for f in freqs.values())
    for i, (ax, (label, f)) in enumerate(zip(axes, freqs.items())):
        K = len(f) - 1
        pit = np.arange(K + 1) / K
        uniform = 1.0 / (K + 1)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color="#e8e8e8", linewidth=0.8)
        ax.bar(pit, f, width=(1.0 / K) * 0.92, color=soften(color_of(label, i), soften_by),
               edgecolor="none", zorder=2)
        ax.axhline(uniform, color="#2b2b2b", linestyle=(0, (4, 3)), linewidth=1.1, zorder=3)
        ax.set_title(label, fontsize=14, pad=8)
        ax.set_xlabel("PIT", fontsize=12)
        ax.set_xlim(-0.04, 1.04)
        ax.set_ylim(0, hi * 1.08)
        ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.tick_params(labelsize=10, length=3)
    for ax in axes[1:]:
        ax.tick_params(axis="y", length=0)
    axes[0].set_ylabel("frequency", fontsize=12)
    axes[0].annotate("uniform", xy=(0.10, uniform), xytext=(0, 3), textcoords="offset points",
                     fontsize=9, ha="left", va="bottom")
    fig.tight_layout()
    save(fig, out)


def reliability_diagrams(panels: Dict[str, Dict[str, tuple]], out: Path) -> None:
    """panels[title][label] = (p, obs, rmsce): one panel per title (paper Fig 12)."""
    apply_style()
    fig, axes = plt.subplots(1, len(panels), figsize=(4.8 * len(panels), 4.6), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (title, curves) in zip(axes, panels.items()):
        ax.set_axisbelow(True)
        ax.grid(True, color="#ededed", linewidth=0.8)
        ax.plot([0, 1], [0, 1], color="#9a9a9a", linestyle=(0, (4, 3)), linewidth=1.1, label="ideal")
        for i, (label, (p, obs, rmsce)) in enumerate(curves.items()):
            ax.plot(p, obs, color=color_of(label, i), linewidth=1.9, label=f"{label}  ({rmsce:.3f})")
        ax.set_title(title, fontsize=14, pad=8)
        ax.set_xlabel("nominal level $p$", fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.legend(fontsize=10, loc="upper left", title="RMSCE", title_fontsize=10)
    axes[0].set_ylabel(r"observed coverage $\hat{G}(p)$", fontsize=12)
    fig.tight_layout()
    save(fig, out)


def calibration_scatter(points: Dict[str, tuple], out: Path, xlabel: str, ylabel: str) -> None:
    """points[label] = (x, y): calibration error against interval width (paper Fig 11)."""
    apply_style()
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    ax.set_axisbelow(True)
    ax.grid(True, color="#ededed", linewidth=0.8)
    xs, ys = [v[0] for v in points.values()], [v[1] for v in points.values()]
    for i, (label, (x, y)) in enumerate(points.items()):
        c = color_of(label, i)
        ax.scatter(x, y, s=190, color=soften(c, 0.2), edgecolor=c, linewidth=1.2, zorder=3)
        ax.annotate(label, (x, y), xytext=(11, 4), textcoords="offset points", fontsize=11)
    lo, hi = min(xs), max(xs)
    ax.set_xlim(lo - 0.12 * (hi - lo + 1e-12), hi + 0.30 * (hi - lo + 1e-12))
    ylo, yhi = min(ys), max(ys)
    ax.set_ylim(ylo - 0.14 * (yhi - ylo + 1e-12), yhi + 0.14 * (yhi - ylo + 1e-12))
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.annotate("better", xy=(0.03, 0.05), xycoords="axes fraction", fontsize=10,
                color="#8a8a8a", style="italic")
    fig.tight_layout()
    save(fig, out)



def wasserstein_windows(table: "pd.DataFrame", out: Path) -> None:
    """W1 distance to the ground-truth enstrophy distribution per window (paper Fig 8)."""
    apply_style()
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for i, (label, df) in enumerate(table.groupby("label", sort=False)):
        st = style_of(label, i)
        ax.plot(df["window"], df["wasserstein"], marker="o", markersize=4, linewidth=1.8,
                label=label, **st)
    ax.set_xticks(sorted(table["window"].unique()))
    ax.set_xticklabels(table.drop_duplicates("window")["window_label"], rotation=45)
    ax.set_xlabel("Rollout window")
    ax.set_ylabel("Wasserstein distance")
    ax.grid(True, color="#ededed", linewidth=0.8)
    ax.legend(fontsize=10)
    fig.tight_layout()
    save(fig, out)


def enstrophy_trajectories(curves: Dict[str, Dict[str, np.ndarray]], out: Path) -> None:
    """Normalised total enstrophy of one trajectory: truth vs ensemble mean +- std (Figs 9, 10)."""
    apply_style()
    n = len(curves)
    ncols = min(2, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.0 * nrows), sharex=True, sharey=True,
                             squeeze=False)
    for ax, (label, c) in zip(axes.ravel(), curves.items()):
        t, gt, pred = c["t"], c["gt"], c["pred"]
        mu, sd = pred.mean(0), pred.std(0)
        ax.fill_between(t, mu - sd, mu + sd, color="0.6", alpha=0.45, linewidth=0)
        ax.plot(t, mu, color="0.35", linewidth=1.0, label="Prediction mean +/- std")
        ax.plot(t, gt, color="crimson", linewidth=1.0, label="Ground truth", zorder=5)
        ax.set_title(label, fontsize=11)
        ax.grid(True, alpha=0.25)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    axes[0][0].set_ylabel(r"Normalized total enstrophy $\sum\omega_t^2/\sum\omega_0^2$", fontsize=8)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.03))
    fig.supxlabel("Timestep", fontsize=10)
    fig.tight_layout()
    save(fig, out)


# margins in inches
LEFT_IN, RIGHT_IN, TOP_IN, BOTTOM_IN = 1.25, 1.55, 0.25, 0.75


def _serif():
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "serif"]


def frame_indices(n_frames: int, gap: int) -> List[int]:
    idx = list(range(0, n_frames, gap))
    if (n_frames - 1) not in idx:
        idx.append(n_frames - 1)
    return idx


def _figure_size(n_rows, n_cols, cell_w, cell_h, row_gap, col_gap, pair_gap=0.0, n_pairs=0):
    w = LEFT_IN + n_cols * cell_w + (n_cols - 1) * col_gap + RIGHT_IN
    h = TOP_IN + n_rows * cell_h + (n_rows - 1) * row_gap + max(n_pairs - 1, 0) * pair_gap + BOTTOM_IN
    return w, h


def _panel_grid(fig, n_rows, n_cols, *, cell_w, cell_h, row_gap, col_gap, pair_gap=0.0, pair_size=0):
    W, H = fig.get_size_inches()
    w, h = cell_w / W, cell_h / H
    cg, rg, pg = col_gap / W, row_gap / H, pair_gap / H
    left = LEFT_IN / W
    axes = []
    y = 1.0 - TOP_IN / H
    for r in range(n_rows):
        if pair_size and r and r % pair_size == 0:
            y -= pg
        row = []
        for c in range(n_cols):
            ax = fig.add_axes([left + c * (w + cg), y - h, w, h])
            ax.set_axis_off()
            row.append(ax)
        axes.append(row)
        y -= h + rg
    return axes


def _cbar_axes(fig, y0: float, height: float):
    W, _ = fig.get_size_inches()
    return fig.add_axes([1.0 - (RIGHT_IN - 0.35) / W, y0, 0.18 / W, height])


def _show(ax, field, cmap, vmin, vmax, geo, lat=None, lon=None):
    if geo:
        from common.geo import plot_map
        return plot_map(ax, field, lat, lon, cmap=cmap, vmin=vmin, vmax=vmax, coast_color="k")
    return ax.imshow(field, vmin=vmin, vmax=vmax, cmap=cmap)


def _row_label(ax, text, size, italic=False):
    ax.text(-0.06, 0.5, text, transform=ax.transAxes, fontsize=size, fontfamily="serif",
            fontstyle="italic" if italic else "normal", ha="right", va="center", rotation=90)


def _col_labels(axes_row, frames, size):
    for ax, t in zip(axes_row, frames):
        ax.text(0.5, -0.18, f"t={t}", transform=ax.transAxes, fontsize=size, fontfamily="serif",
                fontstyle="italic", ha="center", va="top")


def _geo_grid(shape):
    from common.geo import pole_grid_lat
    return pole_grid_lat(shape[1]), np.arange(shape[2]) * (360.0 / shape[2])


def figure_traj(results: Dict, channel: int, frames: List[int], out: Path, *, geo: bool,
                view: str = "member", unit: str = "", cmap=None, vmax_percentile: float = 100.0,
                vmin_zero: bool = False, symmetric: bool = False) -> None:
    """Ground truth on top, one row per method, shared colour scale.

    results[label] = {"truth", "mean", "std", "member"}, each (T+1, H, W, C).
    """
    _serif()
    labels = list(results)
    n_rows, n_cols = 1 + len(labels), len(frames)
    cell_w, cell_h = (4.0, 2.2) if geo else (2.6, 2.6)
    cmap = cmap if cmap is not None else TRAJ_CMAP
    fig = plt.figure(figsize=_figure_size(n_rows, n_cols, cell_w, cell_h, 0.10, 0.08))
    axes = _panel_grid(fig, n_rows, n_cols, cell_w=cell_w, cell_h=cell_h, row_gap=0.10, col_gap=0.08)

    truth = next(iter(results.values()))["truth"]
    fields = [truth] + [results[r][view] for r in labels]
    sel = [f[frames][..., channel] for f in fields]
    if symmetric:
        m = max(float(abs(s).max()) for s in sel)
        lo, hi = -m, m
    else:
        lo = 0.0 if vmin_zero else min(float(s.min()) for s in sel)
        hi = (max(float(s.max()) for s in sel) if vmax_percentile >= 100.0
              else max(float(np.percentile(np.asarray(s), vmax_percentile)) for s in sel))
    lat = lon = None
    if geo:
        lat, lon = _geo_grid(truth.shape)

    im = None
    for c, t in enumerate(frames):
        im = _show(axes[0][c], np.asarray(truth[t, ..., channel]), cmap, lo, hi, geo, lat, lon)
    _row_label(axes[0][0], "Ground Truth", 15)
    for r, label in enumerate(labels, start=1):
        seq = results[label][view]
        for c, t in enumerate(frames):
            _show(axes[r][c], np.asarray(seq[t, ..., channel]), cmap, lo, hi, geo, lat, lon)
        _row_label(axes[r][0], label, 15)
    _col_labels(axes[-1], frames, 14)
    cb = fig.colorbar(im, cax=_cbar_axes(fig, 0.18, 0.64))
    if unit:
        cb.set_label(unit, fontsize=13, fontfamily="serif")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(out).with_suffix(".png"), dpi=180)
    fig.savefig(Path(out).with_suffix(".pdf"))
    plt.close(fig)
    print(f"  saved {Path(out).with_suffix('.png')} (+ .pdf)")


def figure_errci(results: Dict, channel: int, frames: List[int], out: Path, *, geo: bool,
                 unit: str = "") -> None:
    """Two rows per method: |ensemble mean - truth| and the 95% half-width 1.96 std."""
    _serif()
    labels = list(results)
    n_rows, n_cols = 2 * len(labels), len(frames)
    cell_w, cell_h = (4.0, 2.2) if geo else (2.6, 2.6)
    fig = plt.figure(figsize=_figure_size(n_rows, n_cols, cell_w, cell_h, 0.10, 0.08,
                                         pair_gap=0.42, n_pairs=len(labels)))
    axes = _panel_grid(fig, n_rows, n_cols, cell_w=cell_w, cell_h=cell_h, row_gap=0.10,
                       col_gap=0.08, pair_gap=0.42, pair_size=2)
    err = {r: np.abs(np.asarray(results[r]["mean"][..., channel]) - np.asarray(results[r]["truth"][..., channel]))
           for r in labels}
    ci = {r: 1.96 * np.asarray(results[r]["std"][..., channel]) for r in labels}
    err_max = max(err[r][frames].max() for r in labels)
    ci_max = max(ci[r][frames].max() for r in labels)
    lat = lon = None
    if geo:
        lat, lon = _geo_grid(results[labels[0]]["truth"].shape)
    im_e = im_c = None
    for i, label in enumerate(labels):
        for c, t in enumerate(frames):
            im_e = _show(axes[2 * i][c], err[label][t], ERR_CMAP, 0, err_max, geo, lat, lon)
            im_c = _show(axes[2 * i + 1][c], ci[label][t], CI_CMAP, 0, ci_max, geo, lat, lon)
        _row_label(axes[2 * i][0], "Error", 13, italic=True)
        _row_label(axes[2 * i + 1][0], "95% CI", 13, italic=True)
        top, bot = axes[2 * i][0].get_position().y1, axes[2 * i + 1][0].get_position().y0
        fig.text(0.012, (top + bot) / 2, label, fontsize=17, fontfamily="serif", ha="left",
                 va="center", rotation=90)
    _col_labels(axes[-1], frames, 14)
    fig.colorbar(im_e, cax=_cbar_axes(fig, 0.55, 0.35)).set_label(
        f"Error [{unit}]" if unit else "Error", fontsize=13, fontfamily="serif")
    fig.colorbar(im_c, cax=_cbar_axes(fig, 0.10, 0.35)).set_label(
        f"95% CI [{unit}]" if unit else "95% CI", fontsize=13, fontfamily="serif")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(out).with_suffix(".png"), dpi=180)
    fig.savefig(Path(out).with_suffix(".pdf"))
    plt.close(fig)
    print(f"  saved {Path(out).with_suffix('.png')} (+ .pdf)")


def figure_single(res: Dict, channel: int, frames: List[int], out: Path, label: str) -> None:
    """One method, four rows: truth, mean prediction, |error|, 95% CI (paper Fig 4)."""
    _serif()
    n_cols = len(frames)
    fig = plt.figure(figsize=_figure_size(4, n_cols, 2.0, 2.0, 0.08, 0.06))
    axes = _panel_grid(fig, 4, n_cols, cell_w=2.0, cell_h=2.0, row_gap=0.08, col_gap=0.06)
    truth, mean = np.asarray(res["truth"][..., channel]), np.asarray(res["mean"][..., channel])
    err, ci = np.abs(mean - truth), 1.96 * np.asarray(res["std"][..., channel])
    lo, hi = min(truth[frames].min(), mean[frames].min()), max(truth[frames].max(), mean[frames].max())
    emax = max(err[frames].max(), ci[frames].max())
    rows = [(truth, TRAJ_CMAP, lo, hi, "Ground Truth"), (mean, TRAJ_CMAP, lo, hi, f"{label} Prediction"),
            (err, ERR_CMAP, 0, emax, f"{label} Error"), (ci, CI_CMAP, 0, emax, f"{label} 95% CI")]
    for r, (f, cmap, vmin, vmax, name) in enumerate(rows):
        for c, t in enumerate(frames):
            im = axes[r][c].imshow(f[t], cmap=cmap, vmin=vmin, vmax=vmax)
        _row_label(axes[r][0], name, 10)
        if r in (1, 3):
            y0 = axes[r][0].get_position().y0
            fig.colorbar(im, cax=_cbar_axes(fig, y0, axes[r - 1][0].get_position().y1 - y0))
    _col_labels(axes[-1], frames, 10)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(out).with_suffix(".png"), dpi=180)
    fig.savefig(Path(out).with_suffix(".pdf"))
    plt.close(fig)
    print(f"  saved {Path(out).with_suffix('.png')} (+ .pdf)")
