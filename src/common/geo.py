"""Geographic helpers for the ERA5 / WeatherBench experiments.

Two things live here:

  * latitude weights -- ERA5 is on an equiangular lat/lon grid, so a cell near the
    pole covers far less area than one at the equator. Every area-average
    (training loss, RMSE, CRPS, spread) has to weight by cos(latitude) or it
    silently reports a polar-dominated number. This is the WeatherBench
    convention: w(phi) = cos(phi), renormalised to mean 1 over the grid.

  * coastline overlay -- `cartopy` is not available on the compute nodes, so the
    Natural Earth 110m coastline (public domain) is vendored as a NaN-separated
    polyline in `common/assets/coastline_110m.npz` and drawn with plain
    matplotlib. 37 KB, no runtime download, no extra dependency.
"""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

_ASSET = os.path.join(os.path.dirname(__file__), "assets", "coastline_110m.npz")


# --------------------------------------------------------------------------- #
# Latitude weighting
# --------------------------------------------------------------------------- #
def latitude_weights(lat, normalize: bool = True):
    """cos(lat) weights for an equiangular grid, as a numpy float64 array.

    Args:
        lat: latitudes in degrees, shape [nlat].
        normalize: rescale so the weights average to 1. Keeps a weighted MSE on
            the same scale as the unweighted one, so thresholds and logged values
            stay comparable across the two.
    """
    lat = np.asarray(lat, dtype=np.float64)
    w = np.cos(np.deg2rad(lat))
    # At +-90 poles have cos = 0 exactly. A zero weight
    # would drop them from every metric and, in a training, cut their
    # gradients. Floor them at half the smallest interior weight, which
    # is the area a pole cap actually covers relative to its neighbours.
    w = np.clip(w, 1e-8, None)
    if lat.size > 2:
        interior = w[1:-1].min()
        w[0] = max(w[0], 0.5 * interior)
        w[-1] = max(w[-1], 0.5 * interior)
    if normalize:
        w = w / w.mean()
    return w


def pole_grid_lat(nlat: int):
    """Latitudes of an equiangular "with poles" grid, south -> north.

    ERA5 at 1.5 degrees is 121 rows running -90 .. +90 inclusive, which is exactly
    `linspace(-90, 90, nlat)`. Deriving the latitudes from the row count lets the
    model build its area weights without reading the dataset metadata, so a
    checkpoint stays loadable when the data directory has moved.
    """
    return np.linspace(-90.0, 90.0, int(nlat))


def latitude_weights_torch(lat, normalize: bool = True, device=None, dtype=None):
    """`latitude_weights` as a torch tensor."""
    import torch

    w = latitude_weights(lat, normalize=normalize)
    return torch.as_tensor(w, device=device, dtype=dtype or torch.float32)


def lat_weighted_mean(x, weights, lat_dim: int):
    """Weighted mean of `x` over every dim from `lat_dim` onward.

    Args:
        x: tensor with a latitude axis at `lat_dim`.
        weights: 1-D tensor [nlat], already normalised.
        lat_dim: index of the latitude axis in `x`.
    """
    shape = [1] * x.ndim
    shape[lat_dim] = weights.numel()
    w = weights.reshape(shape).to(device=x.device, dtype=x.dtype)
    dims = tuple(range(lat_dim, x.ndim))
    return (x * w).sum(dim=dims) / w.expand_as(x).sum(dim=dims)


# --------------------------------------------------------------------------- #
# Coastlines
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=4)
def _raw_coastline():
    data = np.load(_ASSET)
    return data["lon"].astype(np.float64), data["lat"].astype(np.float64)


@lru_cache(maxsize=4)
def coastline(lon_min: float = 0.0, lon_max: float = 360.0):
    """Coastline vertices remapped onto [lon_min, lon_max], NaN-separated.

    Natural Earth stores longitudes in [-180, 180]; ERA5 uses [0, 360). Shifting
    a polyline into the other convention makes any segment that crosses the seam
    jump the full width of the map, drawing a spurious horizontal line. Those
    segments are cut by inserting a NaN.
    """
    lon, lat = _raw_coastline()
    span = lon_max - lon_min
    lon = np.where(np.isnan(lon), lon, (lon - lon_min) % span + lon_min)

    # Break wherever consecutive vertices of the same polyline jump more than
    # half the map: that is the seam, not a real coastal segment.
    out_lon, out_lat = [lon[:1]], [lat[:1]]
    jump = np.abs(np.diff(lon)) > span / 2.0
    idx = np.flatnonzero(jump) + 1
    prev = 0
    for i in idx:
        out_lon.extend([lon[prev:i], [np.nan]])
        out_lat.extend([lat[prev:i], [np.nan]])
        prev = i
    out_lon.append(lon[prev:])
    out_lat.append(lat[prev:])
    return (
        np.concatenate([np.atleast_1d(a) for a in out_lon[1:]]),
        np.concatenate([np.atleast_1d(a) for a in out_lat[1:]]),
    )


def draw_coastlines(ax, lon_min: float = 0.0, lon_max: float = 360.0,
                    color: str = "k", linewidth: float = 0.4, alpha: float = 0.75,
                    zorder: int = 3):
    """Overlay the world coastline on an axis whose data coords are degrees."""
    lon, lat = coastline(lon_min, lon_max)
    ax.plot(lon, lat, color=color, linewidth=linewidth, alpha=alpha,
            zorder=zorder, solid_capstyle="round")
    return ax


def grid_extent(lat, lon):
    """imshow `extent` for a cell-centred lat/lon grid, in (l, r, b, t) order."""
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    dlat = float(np.abs(np.diff(lat)).mean())
    dlon = float(np.abs(np.diff(lon)).mean())
    return (
        float(lon.min()) - dlon / 2,
        float(lon.max()) + dlon / 2,
        float(lat.min()) - dlat / 2,
        float(lat.max()) + dlat / 2,
    )


def recenter_longitude(field, lon, center_lon: float = 0.0):
    """Reorder a periodic-longitude field so `center_lon` is in the middle.

    ERA5 is stored on 0 .. 358.5, so plotting it as-is puts the Greenwich
    meridian at both edges and the dateline through the centre -- Europe and
    Africa split down the left and right margins. Every conventional world map
    centres on Greenwich, and a reader matching a forecast against a coastline
    should not have to mentally re-wrap it.

    Longitudes are mapped into [center-180, center+180) and the field is
    reordered to match; `argsort` rather than a fixed roll so this works for any
    grid spacing and any centre. Display only -- nothing upstream is touched.
    """
    lon = np.asarray(lon, dtype=np.float64)
    shifted = ((lon - center_lon + 180.0) % 360.0) - 180.0 + center_lon
    order = np.argsort(shifted)
    return np.asarray(field)[..., order], shifted[order]


def plot_map(ax, field, lat, lon, cmap="RdBu_r", vmin=None, vmax=None,
             coast: bool = True, coast_color: str = "k", title: str = None,
             origin: str = None, center_lon: float = 0.0):
    """imshow one [nlat, nlon] field in geographic coordinates with coastlines.

    `origin` defaults to "lower" when latitude increases with index (the ERA5
    south -> north ordering) and "upper" otherwise, so the map is never upside
    down regardless of how the source file was written.

    `center_lon` recentres the periodic longitude axis; 0 (Greenwich in the
    middle) is the conventional world-map view and the default. Pass None to
    plot the array in its stored order.
    """
    lat = np.asarray(lat)
    field = np.asarray(field)
    if center_lon is not None:
        field, lon = recenter_longitude(field, lon, center_lon)
    if origin is None:
        origin = "lower" if lat[0] < lat[-1] else "upper"
    extent = grid_extent(lat, lon)
    im = ax.imshow(field, origin=origin, extent=extent, cmap=cmap,
                   vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest")
    if coast:
        draw_coastlines(ax, extent[0], extent[1], color=coast_color)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=8)
    return im
