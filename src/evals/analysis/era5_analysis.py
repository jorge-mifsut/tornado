"""ERA5 reference baselines and dataset statistics.

Persistence and climatology RMSE / ACC by lead time on the test years
(climatology from 2010-2015), the size of the diurnal and seasonal cycles, and
per-latitude statistics. All cos(latitude) weighted.

    python -m evals.analysis.era5_analysis --path <era5_1p5 dir> --outdir results/era5_analysis
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from common.geo import latitude_weights, plot_map  # noqa: E402
from common.plotting import ERA5_CMAPS  # noqa: E402
from dataset.era5 import load_meta  # noqa: E402

VARS = ["t2m", "u10", "v10", "z500"]
UNITS = {"t2m": "K", "u10": "m/s", "v10": "m/s", "z500": "m$^2$/s$^2$"}
EPOCH = np.datetime64("1970-01-01T00", "h")


def load_years(path: Path, years: list) -> tuple:
    """Concatenated fields and timestamps for `years`, as (T, nlat, nlon, C)."""
    fields, hours = [], []
    for y in years:
        fields.append(np.load(path / f"era5_1p5_{y}.npy", mmap_mode="r"))
        hours.append(np.load(path / f"era5_1p5_{y}_time.npy"))
    return fields, np.concatenate(hours)


def wmean(x, w, axis=(-3, -2)):
    """cos(latitude)-weighted mean over the (lat, lon) axes of `x`."""
    return (x * w).sum(axis=axis) / np.broadcast_to(w, x.shape).sum(axis=axis)




def build_climatology(fields, hours, window_days: int = 15) -> np.ndarray:
    """[366, 4, nlat, nlon, C] mean per (day of year, hour // 6), smoothed over +-7 days."""
    nlat, nlon, C = fields[0].shape[1:]
    times = EPOCH + hours.astype("timedelta64[h]")
    doy = ((times.astype("datetime64[D]") - times.astype("datetime64[Y]"))
           .astype("timedelta64[D]").astype(int))  # 0-365
    hod = (times - times.astype("datetime64[D]")).astype("timedelta64[h]").astype(int) // 6

    total = np.zeros((366, 4, nlat, nlon, C), dtype=np.float32)
    count = np.zeros((366, 4), dtype=np.int64)

    offset = 0
    for arr in fields:
        n = arr.shape[0]
        for i in range(n):
            g = offset + i
            total[doy[g], hod[g]] += arr[i]
            count[doy[g], hod[g]] += 1
        offset += n

    half = window_days // 2
    smooth = np.zeros_like(total)
    scount = np.zeros_like(count)
    for d in range(-half, half + 1):
        smooth += np.roll(total, d, axis=0)
        scount += np.roll(count, d, axis=0)

    scount = np.maximum(scount, 1)[..., None, None, None]
    return (smooth / scount).astype(np.float32)


def climatology_for(clim, hours):
    """Look up the climatological field for each timestamp in `hours`."""
    times = EPOCH + hours.astype("timedelta64[h]")
    doy = ((times.astype("datetime64[D]") - times.astype("datetime64[Y]"))
           .astype("timedelta64[D]").astype(int))
    hod = (times - times.astype("datetime64[D]")).astype("timedelta64[h]").astype(int) // 6
    return clim[doy, hod]


def baseline_scores(fields, hours, clim, w, leads) -> dict:
    """Persistence RMSE and ACC (anomalies from `clim`) per lead, and climatology RMSE."""
    data = np.concatenate([np.asarray(f) for f in fields], axis=0)  # (T, lat, lon, C)
    T = data.shape[0]
    anom = data - climatology_for(clim, hours)

    out = {"lead_hours": [], "persistence_rmse": [], "persistence_acc": []}

    clim_rmse = np.sqrt(wmean((anom**2), w).mean(axis=0))
    out["climatology_rmse"] = clim_rmse.tolist()

    chunk = 256
    for lead in leads:
        step = lead // 6
        if step >= T:
            continue
        n = T - step
        se = np.zeros(len(VARS))
        acc_sum = np.zeros(len(VARS))
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            a, b = data[s:e], data[s + step : e + step]
            se += wmean(((b - a) ** 2), w).sum(axis=0)

            aa, ab = anom[s:e], anom[s + step : e + step]
            num = wmean(aa * ab, w)
            den = np.sqrt(wmean(aa**2, w) * wmean(ab**2, w))
            acc_sum += (num / np.maximum(den, 1e-12)).sum(axis=0)

        rmse = np.sqrt(se / n)
        acc = acc_sum / n

        out["lead_hours"].append(int(lead))
        out["persistence_rmse"].append(rmse.tolist())
        out["persistence_acc"].append(acc.tolist())
        print(
            f"  lead {lead:4d} h  RMSE "
            + "  ".join(f"{v}={r:9.3f}" for v, r in zip(VARS, rmse))
            + "   ACC " + " ".join(f"{v}={c:.3f}" for v, c in zip(VARS, acc))
        )
    return out


def fig_climatology(clim, lat, lon, outdir: Path):
    """Annual-mean field and its temporal standard deviation, per variable."""
    mean = clim.mean(axis=(0, 1))
    std = clim.std(axis=(0, 1))

    fig, axes = plt.subplots(len(VARS), 2, figsize=(13, 2.6 * len(VARS)))
    for i, v in enumerate(VARS):
        im = plot_map(axes[i][0], mean[..., i], lat, lon, cmap=ERA5_CMAPS[v],
                      title=f"{v} climatological mean [{UNITS[v]}]")
        fig.colorbar(im, ax=axes[i][0], fraction=0.025, pad=0.01)
        im = plot_map(axes[i][1], std[..., i], lat, lon, cmap="magma",
                      title=f"{v} seasonal+diurnal std of the climatology")
        fig.colorbar(im, ax=axes[i][1], fraction=0.025, pad=0.01)
    fig.tight_layout()
    fig.savefig(outdir / "era5_climatology.png", dpi=130)
    plt.close(fig)
    print(f"  saved {outdir / 'era5_climatology.png'}")


def fig_baselines(scores, outdir: Path):
    leads = np.array(scores["lead_hours"])
    prmse = np.array(scores["persistence_rmse"])  # (L, C)
    pacc = np.array(scores["persistence_acc"])
    crmse = np.array(scores["climatology_rmse"])  # (C,)

    fig, axes = plt.subplots(2, len(VARS), figsize=(4 * len(VARS), 7))
    for i, v in enumerate(VARS):
        ax = axes[0][i]
        ax.plot(leads / 24, prmse[:, i], "o-", label="persistence")
        ax.axhline(crmse[i], color="k", ls="--", label="climatology")
        ax.set_title(f"{v}  RMSE [{UNITS[v]}]")
        ax.set_xlabel("lead time [days]")
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)
            ax.annotate("12 h: out of phase\nwith the diurnal cycle",
                        xy=(0.5, prmse[1, i]), xytext=(2.6, prmse[1, i] * 0.99),
                        fontsize=7, ha="left", va="center",
                        arrowprops=dict(arrowstyle="->", lw=0.7))

        ax = axes[1][i]
        ax.plot(leads / 24, pacc[:, i], "o-", color="C1")
        ax.axhline(0.6, color="k", ls=":", lw=1)
        ax.set_ylim(-0.1, 1.02)
        ax.set_title(f"{v}  persistence ACC")
        ax.set_xlabel("lead time [days]")
        ax.grid(alpha=0.3)
    fig.suptitle("Reference forecasts on the held-out years, cos(lat) weighted "
                 "(ACC=0.6 marks the usual skill threshold)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outdir / "era5_baselines.png", dpi=130)
    plt.close(fig)
    print(f"  saved {outdir / 'era5_baselines.png'}")


def fig_cycles(fields, hours, w, lat, outdir: Path, chunk: int = 128):
    """Diurnal and seasonal cycles: global mean and one mid-latitude land point."""
    times = EPOCH + hours.astype("timedelta64[h]")
    hod = (times - times.astype("datetime64[D]")).astype("timedelta64[h]").astype(int)
    month = times.astype("datetime64[M]").astype(int) % 12 + 1

    ilat = int(np.argmin(np.abs(np.asarray(lat) - 40.0)))
    ilon = int(260 / 1.5)

    gm_parts, pt_parts = [], []
    for f in fields:
        for s in range(0, f.shape[0], chunk):
            block = np.asarray(f[s : s + chunk], dtype=np.float32)
            gm_parts.append(wmean(block, w))
            pt_parts.append(block[:, ilat, ilon, 0])
    gm = np.concatenate(gm_parts)  # (T, C) global area-weighted mean
    pt = np.concatenate(pt_parts)  # (T,)  t2m at 40N, 100W

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    ax.bar([0, 6, 12, 18],
           [gm[hod == h, 0].mean() - gm[:, 0].mean() for h in [0, 6, 12, 18]],
           width=4, color="C0")
    ax.set_xlabel("hour (UTC)")
    ax.set_ylabel("global mean t2m anomaly [K]")
    ax.set_title("Diurnal cycle, global mean\n(small: local cycles cancel)")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.bar([0, 6, 12, 18], [pt[hod == h].mean() - pt.mean() for h in [0, 6, 12, 18]],
           width=4, color="C1")
    ax.set_xlabel("hour (UTC)")
    ax.set_ylabel("t2m anomaly [K]")
    ax.set_title("Diurnal cycle at 40N, 100W\n(land point, local time ~UTC-7)")
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(range(1, 13), [gm[month == m, 0].mean() for m in range(1, 13)], "o-")
    ax.set_xlabel("month")
    ax.set_ylabel("global mean t2m [K]")
    ax.set_title("Seasonal cycle, global mean")
    ax.grid(alpha=0.3)

    fig.suptitle("Signal available to the sin/cos date conditioning "
                 "(training years)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(outdir / "era5_cycles.png", dpi=130)
    plt.close(fig)
    print(f"  saved {outdir / 'era5_cycles.png'}")

    ilat_range = float(np.ptp([pt[hod == h].mean() for h in [0, 6, 12, 18]]))
    return {
        "global_diurnal_range_t2m_K": float(
            np.ptp([gm[hod == h, 0].mean() for h in [0, 6, 12, 18]])
        ),
        "point_40N_100W_diurnal_range_t2m_K": ilat_range,
        "global_seasonal_range_t2m_K": float(
            np.ptp([gm[month == m, 0].mean() for m in range(1, 13)])
        ),
    }


def fig_latitude(fields, w, lat, outdir: Path, chunk: int = 128):
    """Per-latitude variability and the area weights."""
    n = 0
    s1 = s2 = None
    for f in fields:
        for s in range(0, f.shape[0], chunk):
            block = np.asarray(f[s : s + chunk], dtype=np.float64)
            if s1 is None:
                s1 = np.zeros(block.shape[1::2], dtype=np.float64)  # (nlat, C)
                s2 = np.zeros_like(s1)
            s1 += block.sum(axis=(0, 2))
            s2 += (block**2).sum(axis=(0, 2))
            n += block.shape[0] * block.shape[2]
    mean = s1 / n
    zonal_std = np.sqrt(np.maximum(s2 / n - mean**2, 0.0))  # (nlat, C)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for i, v in enumerate(VARS):
        axes[0].plot(lat, zonal_std[:, i] / zonal_std[:, i].max(), label=v)
    axes[0].set_xlabel("latitude [deg]")
    axes[0].set_ylabel("std, normalised per variable")
    axes[0].set_title("Variability by latitude")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].plot(lat, w.ravel(), color="C3")
    axes[1].set_xlabel("latitude [deg]")
    axes[1].set_ylabel("weight (mean 1)")
    axes[1].set_title("cos(latitude) area weights\n"
                      "poles carry ~1/50 of an equatorial cell")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(outdir / "era5_latitude.png", dpi=130)
    plt.close(fig)
    print(f"  saved {outdir / 'era5_latitude.png'}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--path", type=Path, required=True,
                   help="directory produced by dataset/prepare_era5.py")
    p.add_argument("--outdir", type=Path, default=Path("results/era5_analysis"))
    p.add_argument("--train-years", type=int, nargs="+",
                   default=list(range(2010, 2016)))
    p.add_argument("--test-years", type=int, nargs="+", default=[2017, 2018])
    p.add_argument("--leads", type=int, nargs="+",
                   default=[6, 12, 24, 48, 72, 120, 168, 240])
    args = p.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    meta = load_meta(str(args.path))
    lat = np.asarray(meta["lat"])
    lon = np.asarray(meta["lon"])
    w = latitude_weights(lat).reshape(-1, 1, 1)

    print(f"grid: {len(lat)} x {len(lon)} at {meta['resolution_deg']} deg, "
          f"{meta['dt_hours']}-hourly, channels {meta['vars']}")

    print("\n[1/5] climatology from the training years "
          f"{args.train_years} (15-day window)")
    tr_fields, tr_hours = load_years(args.path, args.train_years)
    clim = build_climatology(tr_fields, tr_hours)

    print("\n[2/5] climatology and variability maps")
    fig_climatology(clim, lat, lon, args.outdir)

    print("\n[3/5] reference forecasts on the held-out years "
          f"{args.test_years}")
    te_fields, te_hours = load_years(args.path, args.test_years)
    scores = baseline_scores(te_fields, te_hours, clim, w, args.leads)
    fig_baselines(scores, args.outdir)

    print("\n[4/5] diurnal and seasonal cycles")
    cycles = fig_cycles(tr_fields, tr_hours, w, lat, args.outdir)
    for k, v in cycles.items():
        print(f"  {k}: {v:.3f}")

    print("\n[5/5] latitude structure")
    fig_latitude(tr_fields, w, lat, args.outdir)

    report = {
        "grid": {"nlat": len(lat), "nlon": len(lon),
                 "resolution_deg": meta["resolution_deg"],
                 "dt_hours": meta["dt_hours"]},
        "vars": VARS,
        "splits": {"train": args.train_years, "test": args.test_years},
        "baselines": scores,
        "cycles": cycles,
    }
    out = args.outdir / "era5_analysis.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
