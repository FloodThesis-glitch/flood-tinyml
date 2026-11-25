"""Look inside the CAMELSH archives without extracting them.

Usage (from the project root, with the .venv active):
    python inspect_camelsh.py "C:\\path\\Hourly2.zip"
    python inspect_camelsh.py "C:\\path\\Hourly2.zip" --gauge 01013500

It lists the archive, extracts ONE gauge file into data/raw/camelsh_sample/
and prints the variables, units, time range and how much of each variable
is actually observed. Nothing is modified and nothing else is extracted.
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

SAMPLE_DIR = Path("data/raw/camelsh_sample")
KEYWORDS = ("stream", "flow", "water", "stage", "level", "prec", "rain")


def find_member(names: list[str], gauge: str | None) -> str:
    """Pick the archive member for a gauge id (or the first .nc file)."""
    files = [n for n in names if n.lower().endswith(".nc")]
    if not files:
        raise SystemExit("No .nc files found in the archive.")
    if gauge is None:
        return files[0]
    for name in files:
        if Path(name).stem == gauge:
            return name
    raise SystemExit(f"Gauge {gauge} not found. First files: {files[:5]}")


def summarise(path: Path) -> None:
    """Print the structure and data coverage of one NetCDF file."""
    with xr.open_dataset(path) as ds:
        print(f"\n=== {path.name} ===")
        print("dimensions :", dict(ds.sizes))
        print("coordinates:", list(ds.coords))
        print("attributes :", {k: str(v)[:80] for k, v in list(ds.attrs.items())[:8]})
        time = pd.DatetimeIndex(ds["time"].values) if "time" in ds.coords else None
        if time is not None:
            steps = pd.Series(time).diff().dropna().value_counts().head(3)
            print(f"time range : {time.min()} -> {time.max()}  ({len(time)} steps)")
            print("time steps :", {str(k): int(v) for k, v in steps.items()})
        print(f"\n{'variable':26s} {'dtype':8s} {'units':14s} {'valid %':>8s} {'min':>12s} {'max':>12s}")
        for name, var in ds.data_vars.items():
            values = np.asarray(var.values, dtype=float).ravel()
            ok = np.isfinite(values)
            lo = np.nanmin(values) if ok.any() else float("nan")
            hi = np.nanmax(values) if ok.any() else float("nan")
            units = str(var.attrs.get("units", "?"))[:14]
            print(f"{name:26s} {str(var.dtype):8s} {units:14s} {100 * ok.mean():8.1f} {lo:12.4g} {hi:12.4g}")
        if time is not None:
            print("\nyears with >= 8000 valid hours (nearly complete):")
            for name, var in ds.data_vars.items():
                if any(k in name.lower() for k in KEYWORDS) and var.ndim == 1 and var.shape[0] == len(time):
                    ok = pd.Series(np.isfinite(np.asarray(var.values, dtype=float)), index=time)
                    per_year = ok.groupby(ok.index.year).sum()
                    full = per_year[per_year >= 8000]
                    print(f"  {name:24s} {len(full):2d} of {per_year.size} years"
                          + (f"  ({full.index.min()}-{full.index.max()})" if len(full) else ""))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("archive", help="path to Hourly2.zip")
    parser.add_argument("--gauge", help="USGS gauge id to open (default: first file)")
    args = parser.parse_args()

    archive = Path(args.archive)
    if not archive.is_file():
        sys.exit(f"File not found: {archive}")
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        n_nc = sum(n.lower().endswith(".nc") for n in names)
        print(f"{archive.name}: {len(names)} entries, {n_nc} NetCDF files")
        print("first entries:", names[:5])
        member = find_member(names, args.gauge)
        target = SAMPLE_DIR / Path(member).name
        with z.open(member) as src, open(target, "wb") as dst:
            dst.write(src.read())
        print(f"extracted one file: {member} -> {target} ({target.stat().st_size / 1e6:.1f} MB)")
    summarise(target)


if __name__ == "__main__":
    main()
