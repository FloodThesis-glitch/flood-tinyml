"""Turn the processed FlowDB windows into model-ready arrays.

The labels and features are PRECOMPUTED in the data file (see its README);
this module does not relabel anything. It:

    1. loads one development fold (train + validation),
    2. keeps only usable windows (eligible and with complete features),
    3. builds the event keys and crossing times that src/metrics.py needs,
    4. reports windows, positives and events per split and per river.

The 2018 test year is NOT in the files we have yet, so no test split is
built here. When it arrives, add it as another named split; the report
already warns if a test split has fewer than the required number of events.

Run  python -m src.data_prep  from the project root for the report.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.metrics import prepare_events
from src.utils import get_path, load_config, require

# Column names that belong to the processed-file schema itself.
FOLD_COLUMN = "fold"
SPLIT_COLUMN = "split"
IN_FLOOD_COLUMN = "in_flood"
EVENT_ID_COLUMN = "event_id"
COMPLETE_COLUMN = "features_complete"
KEY_SEPARATOR = "-"


@dataclass
class SplitData:
    """One split (for example 'train') ready for models and metrics.

    All per-window arrays have one entry per USABLE window and share order.

    Attributes:
        name: Split name ('train', 'validation', later 'test').
        fold: Cross-validation fold the split belongs to.
        horizon_h: Prediction horizon N in hours.
        feature_names: Column names of x, in order.
        x: Feature matrix, shape (windows, features).
        y: 1 if the level crosses the flood threshold within N hours.
        event_key: Per window, 'river-eventid' of the event it heads toward,
            or None when the window is negative.
        times_h: Per window, hours since the dataset origin.
        river: Per window, the river it belongs to.
        crossing_h: Every event of the split -> hour of its first crossing
            (built from ALL rows, including unusable windows).
        event_river: Every event key -> its river.
    """

    name: str
    fold: int
    horizon_h: int
    feature_names: list[str]
    x: np.ndarray
    y: np.ndarray
    event_key: np.ndarray
    times_h: np.ndarray
    river: np.ndarray
    crossing_h: dict[str, float]
    event_river: dict[str, str]


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def required_columns(cfg: dict[str, Any], horizon_h: int) -> list[str]:
    """List the columns the processed file must contain for this setup."""
    data = cfg["data"]
    return [
        data["time_column"], data["river_column"],
        FOLD_COLUMN, SPLIT_COLUMN, IN_FLOOD_COLUMN, EVENT_ID_COLUMN,
        COMPLETE_COLUMN,
        f"label_{horizon_h}h", f"target_event_{horizon_h}h",
        f"eligible_{horizon_h}h",
        *cfg["features"]["tabular_columns"],
    ]


def load_processed(cfg: dict[str, Any]) -> pd.DataFrame:
    """Read the processed CSV named in config.yaml."""
    path: Path = get_path(cfg, "data_processed") / require(cfg, "data.processed_file")
    if not path.is_file():
        raise FileNotFoundError(
            f"Processed data file not found: {path}\n"
            "Unzip the FlowDB bundle into data/processed/flowdb first."
        )
    return pd.read_csv(path, parse_dates=[cfg["data"]["time_column"]])


# --------------------------------------------------------------------------
# Building splits
# --------------------------------------------------------------------------
def build_split(
    block: pd.DataFrame,
    name: str,
    fold: int,
    horizon_h: int,
    cfg: dict[str, Any],
    origin: pd.Timestamp,
) -> SplitData:
    """Build one split from the rows of a single fold and split name.

    Args:
        block: All rows (usable or not) of this fold and split.
        name: Split name.
        fold: Fold number (stored for reporting).
        horizon_h: Prediction horizon N in hours.
        cfg: Loaded config.
        origin: Time that corresponds to hour 0 (same for every split).
    """
    time_col = cfg["data"]["time_column"]
    river_col = cfg["data"]["river_column"]
    feature_cols: list[str] = list(cfg["features"]["tabular_columns"])

    # Keep only windows the model may be trained or scored on.
    usable = (block[f"eligible_{horizon_h}h"] == 1) & (block[COMPLETE_COLUMN] == 1)
    rows = block[usable]

    x = rows[feature_cols].to_numpy(dtype=float)
    if not np.isfinite(x).all():
        raise ValueError(f"Split '{name}' has NaN features among usable windows.")
    y = rows[f"label_{horizon_h}h"].to_numpy(dtype=int)
    times_h = ((rows[time_col] - origin).dt.total_seconds() / 3600.0).to_numpy(dtype=float)
    river = rows[river_col].astype(str).to_numpy(dtype=object)
    target = rows[f"target_event_{horizon_h}h"].to_numpy(dtype=int)
    event_key = np.array(
        [f"{r}{KEY_SEPARATOR}{t}" if t > 0 else None for r, t in zip(river, target)],
        dtype=object,
    )

    # Event crossing time = first in-flood hour, using ALL rows of the split.
    flood = block[(block[IN_FLOOD_COLUMN] == 1) & (block[EVENT_ID_COLUMN] > 0)]
    flood_river = flood[river_col].astype(str).to_numpy(dtype=object)
    flood_id = flood[EVENT_ID_COLUMN].to_numpy(dtype=int)
    flood_h = ((flood[time_col] - origin).dt.total_seconds() / 3600.0).to_numpy(dtype=float)
    crossing_h: dict[str, float] = {}
    event_river: dict[str, str] = {}
    for r, e, h in zip(flood_river, flood_id, flood_h):
        key = f"{r}{KEY_SEPARATOR}{e}"
        crossing_h[key] = min(h, crossing_h.get(key, h))
        event_river[key] = r

    unknown = {k for k in event_key if k is not None} - set(crossing_h)
    if unknown:
        raise ValueError(
            f"Split '{name}': windows point to events with no in-flood rows: "
            f"{sorted(unknown)[:5]}"
        )

    return SplitData(
        name=name, fold=fold, horizon_h=horizon_h, feature_names=feature_cols,
        x=x, y=y, event_key=event_key, times_h=times_h, river=river,
        crossing_h=crossing_h, event_river=event_river,
    )


def build_development_splits(
    df: pd.DataFrame,
    cfg: dict[str, Any],
    horizon_h: int | None = None,
    fold: int | None = None,
) -> dict[str, SplitData]:
    """Build the train and validation splits of one development fold.

    Rows are duplicated across folds in the file, so a fold is always
    selected first; folds are never mixed.

    Args:
        df: The processed data frame (all folds).
        cfg: Loaded config.
        horizon_h: Horizon N in hours; defaults to task.horizon_hours.
        fold: Fold number; defaults to split.cv_fold.
    """
    horizon = int(horizon_h if horizon_h is not None else require(cfg, "task.horizon_hours"))
    if horizon not in cfg["task"]["horizon_options"]:
        raise ValueError(f"Horizon {horizon} h is not in task.horizon_options")
    fold_number = int(fold if fold is not None else require(cfg, "split.cv_fold"))

    missing = [c for c in required_columns(cfg, horizon) if c not in df.columns]
    if missing:
        raise KeyError(f"Processed file is missing columns: {missing}")

    fold_df = df[df[FOLD_COLUMN] == fold_number]
    if fold_df.empty:
        raise ValueError(
            f"Fold {fold_number} not found; available folds: "
            f"{sorted(df[FOLD_COLUMN].unique().tolist())}"
        )

    origin = df[cfg["data"]["time_column"]].min()
    return {
        name: build_split(fold_df[fold_df[SPLIT_COLUMN] == name], name,
                          fold_number, horizon, cfg, origin)
        for name in ("train", "validation")
    }


def load_development_data(
    cfg: dict[str, Any] | None = None,
    horizon_h: int | None = None,
    fold: int | None = None,
) -> dict[str, SplitData]:
    """Load the processed file and build the development splits."""
    config = cfg if cfg is not None else load_config()
    return build_development_splits(load_processed(config), config, horizon_h, fold)


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def event_report(split: SplitData) -> pd.DataFrame:
    """Windows, positives and events per river, plus a pooled 'ALL' row.

    'events_evaluable' counts events with at least one usable window 1 to N
    hours before the crossing (the events event-level recall is measured on);
    'events_unevaluable' are events lost to missing data.
    """
    prepared = prepare_events(split.event_key, split.times_h, split.crossing_h, split.horizon_h)
    evaluable = {key for key, _, _ in prepared}

    rows = []
    for river in sorted(set(split.river) | set(split.event_river.values())):
        mask = split.river == river
        keys = [k for k, r in split.event_river.items() if r == river]
        rows.append({
            "river": river,
            "windows": int(mask.sum()),
            "positives": int(split.y[mask].sum()),
            "events_total": len(keys),
            "events_evaluable": sum(k in evaluable for k in keys),
        })
    table = pd.DataFrame(rows).set_index("river")
    table = pd.concat([table, table.sum().to_frame("ALL").T]).astype(int)
    table["events_unevaluable"] = table["events_total"] - table["events_evaluable"]
    return table


def check_min_events(split: SplitData, minimum: int) -> str | None:
    """Return a warning if the split has fewer evaluable events than needed.

    Proposal Step 3 and rule 8: with fewer than the minimum number of
    independent test events, the study must be reported as exploratory.
    Returns None when the count is sufficient.
    """
    n_events = int(event_report(split).loc["ALL", "events_evaluable"])
    if n_events < minimum:
        return (
            f"WARNING: split '{split.name}' has only {n_events} evaluable events "
            f"(minimum {minimum}). Report results as exploratory and avoid "
            "claims the event count cannot support."
        )
    return None


def print_split_report(splits: dict[str, SplitData], minimum_test_events: int) -> None:
    """Print the per-split tables and every applicable warning."""
    first = next(iter(splits.values()))
    print(f"Fold {first.fold}, horizon N = {first.horizon_h} h, "
          f"features: {first.feature_names}\n")
    for name, split in splits.items():
        print(f"--- {name} ---")
        print(event_report(split).to_string())
        print()
    if "test" in splits:
        message = check_min_events(splits["test"], minimum_test_events)
        print(message or f"Test split has at least {minimum_test_events} evaluable events.")
    else:
        print("NOTE: no test split loaded (the 2018 file is not available yet), "
              "so the minimum-test-events check cannot run.")
    print("NOTE: events from different rivers can come from the same storm, "
          "so pooled events may not be fully independent.")


if __name__ == "__main__":
    configuration = load_config()
    print_split_report(
        load_development_data(configuration),
        minimum_test_events=int(require(configuration, "split.min_test_events")),
    )
