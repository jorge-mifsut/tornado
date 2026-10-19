"""Convert the WeatherBench-2 ERA5 yearly NetCDFs to memory-mappable arrays.

Writes era5_1p5_<year>.npy [T, 121, 240, 4], era5_1p5_<year>_time.npy (hours
since 1970), era5_1p5_meta.json and era5_1p5_norm_stats.pkl (training years).

    python dataset/prepare_era5.py --src <dir of .nc> --dst <out dir> --train-years 2010-2015
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np

VARS = ["t2m", "u10", "v10", "z500"]
LONG_NAMES = {
    "t2m": "2m_temperature",
    "u10": "10m_u_component_of_wind",
    "v10": "10m_v_component_of_wind",
    "z500": "geopotential (level=500)",
}
UNITS = {"t2m": "K", "u10": "m s-1", "v10": "m s-1", "z500": "m2 s-2"}

EPOCH = np.datetime64("1970-01-01T00:00:00", "h")


def _parse_years(spec: str) -> list:
    """'2010-2015' or '2016' or '2017,2018' -> [int, ...]."""
    years = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            years.extend(range(int(a), int(b) + 1))
        else:
            years.append(int(part))
    return sorted(set(years))


class _H5Reader:
    """NetCDF-4 reader on h5py alone, for nodes without xarray."""

    def __init__(self, path):
        import h5py

        self.f = h5py.File(str(path), "r")
        missing = [v for v in VARS if v not in self.f]
        if missing:
            raise KeyError(f"{path} is missing variables {missing}; has {list(self.f)}")

        self.lat = np.asarray(self.f["latitude"][:], dtype=np.float64)
        self.lon = np.asarray(self.f["longitude"][:], dtype=np.float64)

        # "hours since <ref>": the reference date differs from our epoch
        t = self.f["time"]
        units = t.attrs.get("units", b"")
        units = units.decode() if isinstance(units, bytes) else str(units)
        step, _, ref = units.partition(" since ")
        step = step.strip().lower()
        if step not in ("hours", "hour"):
            raise ValueError(f"expected hourly time units, got {units!r}")
        ref = np.datetime64(ref.strip().replace(" ", "T").rstrip("Z"), "h")
        self.times = (ref + np.asarray(t[:], dtype="int64").astype("timedelta64[h]"))

        self.nt = self.f[VARS[0]].shape[0]
        self.nlat, self.nlon = len(self.lat), len(self.lon)

    def read(self, var, start, stop):
        return np.asarray(self.f[var][start:stop], dtype=np.float32)

    def close(self):
        self.f.close()


class _XarrayReader:
    """Same interface via xarray, used when it is available."""

    def __init__(self, path):
        import xarray as xr

        self.ds = xr.open_dataset(str(path), engine="h5netcdf")
        self.lat = self.ds.latitude.values.astype(np.float64)
        self.lon = self.ds.longitude.values.astype(np.float64)
        self.times = self.ds.time.values.astype("datetime64[h]")
        self.nt = self.ds.sizes["time"]
        self.nlat = self.ds.sizes["latitude"]
        self.nlon = self.ds.sizes["longitude"]

    def read(self, var, start, stop):
        return self.ds[var].isel(time=slice(start, stop)).values.astype(np.float32)

    def close(self):
        self.ds.close()


def open_year(path, backend: str = "auto"):
    """Open one yearly NetCDF with whichever backend is installed."""
    if backend == "h5py":
        return _H5Reader(path)
    if backend == "xarray":
        return _XarrayReader(path)
    try:
        import xarray  # noqa: F401
        import h5netcdf  # noqa: F401

        return _XarrayReader(path)
    except ImportError:
        return _H5Reader(path)


def convert_year(src: Path, dst: Path, year: int, overwrite: bool = False,
                 backend: str = "auto") -> tuple:
    """Write <year>.npy / <year>_time.npy. Returns (n_frames, lat, lon)."""
    src_file = src / f"era5_t2m_u10_v10_z500_{year}.nc"
    if not src_file.exists():
        raise FileNotFoundError(
            f"{src_file} not found. --src must be the directory holding the yearly "
            f"NetCDFs written by download_weatherbench.py. Present there: "
            f"{sorted(p.name for p in src.iterdir())[:10] if src.is_dir() else 'not a directory'}"
        )

    out_field = dst / f"era5_1p5_{year}.npy"
    out_time = dst / f"era5_1p5_{year}_time.npy"

    ds = open_year(src_file, backend)
    lat, lon, times = ds.lat, ds.lon, ds.times
    nt, nlat, nlon = ds.nt, ds.nlat, ds.nlon

    if out_field.exists() and out_time.exists() and not overwrite:
        existing = np.load(out_field, mmap_mode="r")
        if existing.shape == (nt, nlat, nlon, len(VARS)):
            print(f"  {year}: already converted ({existing.shape}), skipping")
            ds.close()
            return nt, lat, lon
        print(f"  {year}: shape mismatch {existing.shape}, reconverting")

    tmp = out_field.with_suffix(".npy.partial")
    arr = np.lib.format.open_memmap(
        tmp, mode="w+", dtype=np.float32, shape=(nt, nlat, nlon, len(VARS))
    )
    chunk = 128
    for start in range(0, nt, chunk):
        stop = min(start + chunk, nt)
        for ci, var in enumerate(VARS):
            arr[start:stop, :, :, ci] = ds.read(var, start, stop)
    arr.flush()
    del arr
    tmp.replace(out_field)

    hours = (times - EPOCH).astype("int64")
    np.save(out_time, hours)
    ds.close()

    print(f"  {year}: {nt} frames -> {out_field.name} "
          f"({str(times[0])} .. {str(times[-1])})")
    return nt, lat, lon


def compute_stats(dst: Path, years: list) -> dict:
    """Per-channel mean/std over `years`, unweighted (WeatherBench-2 convention)."""
    n = 0
    s1 = np.zeros(len(VARS), dtype=np.float64)
    s2 = np.zeros(len(VARS), dtype=np.float64)
    vmin = np.full(len(VARS), np.inf)
    vmax = np.full(len(VARS), -np.inf)

    for year in years:
        arr = np.load(dst / f"era5_1p5_{year}.npy", mmap_mode="r")
        for start in range(0, arr.shape[0], 128):
            block = np.asarray(arr[start : start + 128], dtype=np.float64)
            s1 += block.sum(axis=(0, 1, 2))
            s2 += (block**2).sum(axis=(0, 1, 2))
            vmin = np.minimum(vmin, block.min(axis=(0, 1, 2)))
            vmax = np.maximum(vmax, block.max(axis=(0, 1, 2)))
            n += block.shape[0] * block.shape[1] * block.shape[2]
        del arr
        print(f"  stats: {year} done")

    mean = s1 / n
    std = np.sqrt(np.maximum(s2 / n - mean**2, 0.0))
    return {
        "mean": mean.astype(np.float64),
        "std": std.astype(np.float64),
        "min": vmin,
        "max": vmax,
        "n": int(n),
        "years": years,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, required=True, help="dir of yearly .nc files")
    p.add_argument("--dst", type=Path, required=True, help="output dir")
    p.add_argument("--years", default="2010-2018", help="years to convert")
    p.add_argument(
        "--train-years",
        default="2010-2015",
        help="years the normalisation statistics are fitted on",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "xarray", "h5py"],
        help="NetCDF reader; 'auto' prefers xarray + h5netcdf, else h5py",
    )
    args = p.parse_args()

    args.dst.mkdir(parents=True, exist_ok=True)
    years = _parse_years(args.years)
    train_years = _parse_years(args.train_years)

    missing = sorted(set(train_years) - set(years))
    if missing:
        raise ValueError(
            f"--train-years {missing} are not in --years; the statistics would be "
            f"fitted on files this run does not produce."
        )

    print(f"Converting {len(years)} years from {args.src}")
    lat = lon = None
    counts = {}
    for year in years:
        nt, lat, lon = convert_year(args.src, args.dst, year, args.overwrite,
                                    args.backend)
        counts[year] = nt

    meta = {
        "vars": VARS,
        "long_names": [LONG_NAMES[v] for v in VARS],
        "units": [UNITS[v] for v in VARS],
        "lat": lat.tolist(),
        "lon": lon.tolist(),
        "nlat": len(lat),
        "nlon": len(lon),
        "dt_hours": 6,
        "resolution_deg": 1.5,
        "frames_per_year": counts,
        "epoch": "1970-01-01T00:00:00",
        "time_units": "hours",
    }
    with open(args.dst / "era5_1p5_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {args.dst / 'era5_1p5_meta.json'}")

    print(f"Fitting normalisation statistics on {train_years}")
    stats = compute_stats(args.dst, train_years)
    stat_path = args.dst / "era5_1p5_norm_stats.pkl"
    with open(stat_path, "wb") as f:
        pickle.dump([stats["mean"], stats["std"]], f)
    with open(args.dst / "era5_1p5_norm_stats.json", "w") as f:
        json.dump({k: (v.tolist() if isinstance(v, np.ndarray) else v)
                   for k, v in stats.items()}, f, indent=2)

    print(f"wrote {stat_path}")
    for i, v in enumerate(VARS):
        print(
            f"  {v:5s} mean={stats['mean'][i]:12.4f} std={stats['std'][i]:11.4f} "
            f"min={stats['min'][i]:11.3f} max={stats['max'][i]:11.3f}"
        )

    total = sum(counts.values())
    size_gb = total * len(lat) * len(lon) * len(VARS) * 4 / 1e9
    print(f"\n{total} frames total, {size_gb:.2f} GB on disk")


if __name__ == "__main__":
    main()
