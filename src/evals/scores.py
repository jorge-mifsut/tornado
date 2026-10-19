"""Ensemble scores. samples: (K, B, ...), y: (B, ...); each reduces over all entries."""

from typing import Dict

import numpy as np
import torch



def mse(pred_mean: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (pred_mean - y).pow(2).mean()


def mae(pred_mean: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (pred_mean - y).abs().mean()


def crps_ensemble_l1(samples: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Empirical CRPS per entry, averaged: E|X - y| - 0.5 E|X - X'|."""
    K = samples.shape[0]
    x = samples.reshape(K, -1)
    yt = y.reshape(-1)

    term1 = (x - yt).abs().mean(dim=0)

    sum_abs = torch.zeros_like(yt)
    for k in range(K):
        sum_abs += (x[k : k + 1] - x).abs().sum(dim=0)
    term2 = sum_abs / (K * K)

    return (term1 - 0.5 * term2).mean()


def sharpness_std(samples: torch.Tensor) -> torch.Tensor:
    """RMS of the per-entry ensemble std."""
    std = samples.std(dim=0, unbiased=False)
    return torch.sqrt((std.pow(2)).mean())


def central_interval_coverage_width(
    samples: torch.Tensor, y: torch.Tensor, p: float = 0.9
) -> Dict[str, torch.Tensor]:
    """Empirical central-interval coverage and mean width at nominal mass p."""
    lo_q, hi_q = (1.0 - p) / 2.0, 1.0 - (1.0 - p) / 2.0
    qlo = torch.quantile(samples, lo_q, dim=0)
    qhi = torch.quantile(samples, hi_q, dim=0)
    cover = ((y >= qlo) & (y <= qhi)).float().mean()
    width = (qhi - qlo).mean()
    return {f"cov@{int(p * 100)}": cover, f"width@{int(p * 100)}": width}


def energy_score(samples: torch.Tensor, y: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """ES = E||X - y||^beta - 0.5 E||X - X'||^beta, per element over its spatial dims."""
    K, B = samples.shape[:2]
    x = samples.reshape(K, B, -1)
    yt = y.reshape(B, -1)

    term1 = torch.norm(x - yt.unsqueeze(0), dim=-1).pow(beta).mean(dim=0)

    sum_pair = torch.zeros(B, device=samples.device, dtype=samples.dtype)
    for k in range(K):
        sum_pair += torch.norm(x[k : k + 1] - x, dim=-1).pow(beta).sum(dim=0)
    term2 = sum_pair / (K * K)

    return (term1 - 0.5 * term2).mean()


def total_enstrophy(u: torch.Tensor) -> torch.Tensor:
    """(..., H, W, C) -> (...): mean over the snapshot of u^2."""
    return u.float().square().mean(dim=(-3, -2, -1))


# ERA5: cos(latitude) weighted, in normalised units

ERA5_VARS = ["t2m", "u10", "v10", "z500"]


def era5_weights(nlat: int, device, dtype=torch.float32) -> torch.Tensor:
    """cos(latitude) weights, [1, nlat, 1, 1], normalised to mean 1."""
    from common.geo import latitude_weights, pole_grid_lat

    w = latitude_weights(pole_grid_lat(nlat))
    return torch.as_tensor(w, device=device, dtype=dtype).reshape(1, nlat, 1, 1)


def w_mse(pred: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> float:
    num = (w * (pred - y).pow(2)).sum()
    return float(num / w.expand_as(pred).sum())


def _w_mean_per_channel(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Area-weighted mean over all but the channel axis -> [C]."""
    dims = tuple(range(x.ndim - 1))
    return (x * w).sum(dim=dims) / w.expand_as(x).sum(dim=dims)


def era5_metrics(samples: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> Dict[str, object]:
    """Per-channel and channel-averaged mse, crps, sharpness of one element."""
    K = samples.shape[0]

    mse_c = _w_mean_per_channel((samples.mean(dim=0) - y).pow(2), w)
    sharp_c = torch.sqrt(_w_mean_per_channel(samples.std(dim=0, unbiased=False).pow(2), w))

    term1 = (samples - y.unsqueeze(0)).abs().mean(dim=0)
    term2 = torch.zeros_like(term1)
    for k in range(K):
        term2 += (samples[k : k + 1] - samples).abs().sum(dim=0)
    crps_c = _w_mean_per_channel(term1 - 0.5 * term2 / (K * K), w)

    return {
        "mse": float(mse_c.mean()),
        "crps": float(crps_c.mean()),
        "sharpness": float(torch.sqrt(sharp_c.pow(2).mean())),
        "mse_c": mse_c.tolist(),
        "crps_c": crps_c.tolist(),
        "sharpness_c": sharp_c.tolist(),
    }


# Rank histograms: the PIT of a K-member ensemble is r/K, r = #{x_k <= y}

def rank_counts(samples: torch.Tensor, y: torch.Tensor, w: torch.Tensor = None) -> torch.Tensor:
    """(K+1,) histogram of ensemble ranks, area weighted when `w` is given."""
    K = samples.shape[0]
    rank = (samples <= y.unsqueeze(0)).sum(dim=0)
    weights = None if w is None else w.expand_as(rank).reshape(-1).to(torch.float64)
    return torch.bincount(rank.reshape(-1), weights=weights, minlength=K + 1).to(torch.float64)


def pit_cdf_from_ranks(counts: np.ndarray, num_bins: int = 50) -> np.ndarray:
    """Observed coverage G(p) at num_bins levels from rank counts (..., K+1)."""
    counts = np.asarray(counts, dtype=np.float64)
    K = counts.shape[-1] - 1
    u = (torch.arange(K + 1, dtype=torch.float32) / K).numpy()
    ps = torch.linspace(0.0, 1.0, num_bins).numpy()
    included = (u[:, None] <= ps[None, :]).astype(np.float64)
    total = counts.sum(axis=-1, keepdims=True)
    return (counts @ included) / np.maximum(total, 1e-300)


def rmsce_from_ranks(counts: np.ndarray, num_bins: int = 50) -> np.ndarray:
    """RMSCE from rank counts (..., K+1)."""
    ps = torch.linspace(0.0, 1.0, num_bins).numpy().astype(np.float64)
    obs = pit_cdf_from_ranks(counts, num_bins)
    return np.sqrt(((obs - ps) ** 2).mean(axis=-1))


def mace_from_ranks(counts: np.ndarray, num_bins: int = 50) -> np.ndarray:
    ps = torch.linspace(0.0, 1.0, num_bins).numpy().astype(np.float64)
    obs = pit_cdf_from_ranks(counts, num_bins)
    return np.abs(obs - ps).mean(axis=-1)


def rmsce_floor(K: int, num_levels: int = 50) -> float:
    """RMSCE of a perfectly calibrated K-member ensemble (its PIT is discrete)."""
    ps = np.linspace(0.0, 1.0, num_levels)
    obs = (np.floor(ps * K + 1e-9) + 1) / (K + 1)
    return float(np.sqrt(np.mean((obs - ps) ** 2)))
