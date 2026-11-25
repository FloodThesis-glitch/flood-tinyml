"""Inventory of the CAMELSH hourly archive: what does each gauge really have?

Usage (from the project root, with the .venv active):
    python scan_camelsh.py "C:\\path\\Hourly2.zip" --limit 50     # quick test
    python scan_camelsh.py "C:\\path\\Hourly2.zip"                # all gauges (slow)

For every gauge it records, for streamflow and for water level: valid hours,
first and last year with data, years with enough data, and a PROVISIONAL
event count (hours at or above the 99th percentile of the gauge's whole
record, events closer than 12 h merged). The event count is only to see
which gauges are worth a closer look; real labels will use a threshold from
training data only. Results go to results/camelsh_inventory.csv.
Files are extracted one at a time into a temporary folder and deleted.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time as clock
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

OUT_PATH = Path("results/camelsh_inventory.csv")
MERGE_GAP_H = 12
QUANTILE = 0.99
VARIABLES = {"q": "streamflow", "wl": "water_level"}


def series_stats(values: np.ndarray, years: np.ndarray) -> dict[str, float]:
    """Coverage and provisional event count for one hourly series.

    Args:
        values: Hourly values (NaN where missing).
        years: Calendar year of every hour, same length as values.
    """
    ok = np.isfinite(values)
    n_valid = int(ok.sum())
    stats: dict[str, float] = {"valid_hours": n_valid,
                               "valid_pct": round(100.0 * n_valid / values.size, 1)}
    if n_valid == 0:
        return {**stats, "first_year": np.nan, "last_year": np.nan, "years_ge_4000": 0,
                "years_ge_8000": 0, "p99": np.nan, "events_prov": 0}
    per_year = np.bincount(years[ok] - years.min(), minlength=years.max() - years.min() + 1)
    with_data = np.flatnonzero(per_year) + years.min()
    p99 = float(np.quantile(values[ok], QUANTILE))
    above = np.flatnonzero(np.where(ok, values, -np.inf) >= p99)
    events = 0 if above.size == 0 else 1 + int((np.diff(above) > MERGE_GAP_H).sum())
    return {**stats, "first_year": int(with_data.min()), "last_year": int(with_data.max()),
            "years_ge_4000": int((per_year >= 4000).sum()),
            "years_ge_8000": int((per_year >= 8000).sum()),
            "p99": round(p99, 4), "events_prov": events}


def scan_file(path: Path) -> dict[str, float]:
    """Read one gauge file and return its coverage statistics."""
    row: dict[str, float] = {}
    with xr.open_dataset(path) as ds:
        years = pd.DatetimeIndex(ds["time"].values).year.to_numpy()
        row["n_hours"] = int(years.size)
        for prefix, name in VARIABLES.items():
            values = (np.asarray(ds[name].values, dtype=float) if name in ds
                      else np.full(years.size, np.nan))
            row.update({f"{prefix}_{k}": v for k, v in series_stats(values, years).items()})
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("archive", help="path to Hourly2.zip")
    parser.add_argument("--limit", type=int, help="only scan the first N gauges (test run)")
    args = parser.parse_args()
    archive = Path(args.archive)
    if not archive.is_file():
        sys.exit(f"File not found: {archive}")

    rows, failures = [], []
    tmp_dir = Path(tempfile.mkdtemp(prefix="camelsh_"))
    started = clock.time()
    try:
        with zipfile.ZipFile(archive) as z:
            members = sorted(n for n in z.namelist() if n.lower().endswith(".nc"))
            if args.limit:
                members = members[: args.limit]
            for i, member in enumerate(members, 1):
                target = tmp_dir / Path(member).name
                try:
                    with z.open(member) as src, open(target, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    rows.append({"gauge": Path(member).stem.split("_")[0], **scan_file(target)})
                except Exception as exc:  # keep going; report at the end
                    failures.append((member, f"{type(exc).__name__}: {exc}"))
                finally:
                    target.unlink(missing_ok=True)
                if i % 250 == 0 or i == len(members):
                    elapsed = clock.time() - started
                    eta = elapsed / i * (len(members) - i)
                    print(f"{i}/{len(members)} gauges  elapsed {elapsed/60:.1f} min  ETA {eta/60:.1f} min", flush=True)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    table = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {len(table)} gauges to {OUT_PATH}; {len(failures)} failed")
    for member, why in failures[:5]:
        print("  failed:", member, why)

    print("\n=== SUMMARY (provisional; coverage only, no models involved) ===")
    for prefix, label in (("q", "streamflow"), ("wl", "water level")):
        t = table
        print(f"{label}:")
        print(f"  gauges with any data              : {int((t[f'{prefix}_valid_hours'] > 0).sum())}")
        print(f"  gauges with >= 10 years of >=4000 h: {int((t[f'{prefix}_years_ge_4000'] >= 10).sum())}")
        print(f"  gauges with >= 20 years of >=4000 h: {int((t[f'{prefix}_years_ge_4000'] >= 20).sum())}")
        print(f"  ... and >= 30 provisional events  : "
              f"{int(((t[f'{prefix}_years_ge_4000'] >= 10) & (t[f'{prefix}_events_prov'] >= 30)).sum())} (with >= 10 years)")


if __name__ == "__main__":
    main()
