"""Unit tests for src/data_prep.py on a tiny hand-made table.

THE HAND-MADE TABLE (hourly rows, columns like the real processed file)

Each block is 12 hourly rows. A block with an event has it start flooding
at hour `onset` (two in-flood rows, never eligible). The three hours before
the onset are positive windows at N = 3, the six hours before at N = 6.

    fold 2 train       R1: start 0,   onset 8, window at hour 6 incomplete
    fold 2 train       R2: start 0,   no event
    fold 2 validation  R1: start 100, onset 108
    fold 2 validation  R2: start 100, onset 108, hours 105-107 incomplete
    fold 1 validation  R1: start 200, onset 208   (a decoy: must be ignored)

Hour 0 is the earliest row (train R1), so times in hours equal the h above.
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.data_prep import (
    build_development_splits,
    check_min_events,
    event_report,
)

BASE = pd.Timestamp("2020-01-01 00:00")

CFG = {
    "data": {"time_column": "time", "river_column": "river"},
    "task": {"horizon_hours": 3, "horizon_options": [3, 6]},
    "features": {"tabular_columns": ["cfs_rel", "dcfs_rel", "rain_1h", "rain_3h",
                                     "rain_6h", "rain_24h", "rain_prev_7d"]},
    "split": {"cv_fold": 2},
}
# Constant per-column values so feature ORDER can be checked.
FEATURE_VALUES = {"cfs_rel": 0.1, "dcfs_rel": 0.2, "rain_1h": 1.0, "rain_3h": 3.0,
                  "rain_6h": 6.0, "rain_24h": 24.0, "rain_prev_7d": 168.0}


def make_block(river: str, split: str, fold: int, start: int,
               onset: int | None = None, incomplete: tuple[int, ...] = (),
               event_id: int = 1) -> list[dict]:
    """Build 12 hourly rows; see the module docstring for the layout."""
    rows = []
    for h in range(start, start + 12):
        in_flood = int(onset is not None and onset <= h <= onset + 1)
        row = {
            "time": BASE + pd.Timedelta(hours=h), "river": river,
            "fold": fold, "split": split, "in_flood": in_flood,
            "event_id": event_id if in_flood else 0,
            "features_complete": 0 if h in incomplete else 1,
        }
        for n in (3, 6):
            positive = int(onset is not None and onset - n <= h < onset)
            row[f"label_{n}h"] = positive
            row[f"target_event_{n}h"] = event_id if positive else 0
            row[f"eligible_{n}h"] = 0 if in_flood else 1
        row.update(FEATURE_VALUES)
        rows.append(row)
    return rows


def make_table() -> pd.DataFrame:
    rows = (
        make_block("R1", "train", 2, 0, onset=8, incomplete=(6,))
        + make_block("R2", "train", 2, 0)
        + make_block("R1", "validation", 2, 100, onset=108)
        + make_block("R2", "validation", 2, 100, onset=108, incomplete=(105, 106, 107))
        + make_block("R1", "validation", 1, 200, onset=208)
    )
    return pd.DataFrame(rows)


class TestBuildSplits(unittest.TestCase):
    def setUp(self) -> None:
        self.df = make_table()
        self.splits = build_development_splits(self.df, CFG)  # fold 2, N = 3

    def test_only_train_and_validation_are_built(self) -> None:
        self.assertEqual(list(self.splits), ["train", "validation"])

    def test_unusable_windows_are_dropped(self) -> None:
        # train R1: 12 rows - 2 in-flood - 1 incomplete = 9; train R2: 12.
        self.assertEqual(len(self.splits["train"].y), 21)
        # validation R1: 10 usable; R2: 12 - 2 in-flood - 3 incomplete = 7.
        self.assertEqual(len(self.splits["validation"].y), 17)

    def test_positives_after_filtering(self) -> None:
        # train R1 positives are hours 5, 6, 7; hour 6 is incomplete -> 2.
        self.assertEqual(int(self.splits["train"].y.sum()), 2)
        # validation: R1 has 3, R2 loses all 3 to missing features.
        self.assertEqual(int(self.splits["validation"].y.sum()), 3)

    def test_folds_are_never_mixed(self) -> None:
        val = self.splits["validation"]
        self.assertLess(val.times_h.max(), 150)   # decoy fold 1 starts at 200
        self.assertEqual(len(val.y), 17)

    def test_other_fold_can_be_selected(self) -> None:
        other = build_development_splits(self.df, CFG, fold=1)
        self.assertEqual(len(other["validation"].y), 10)   # only the decoy block

    def test_unknown_fold_raises(self) -> None:
        with self.assertRaises(ValueError):
            build_development_splits(self.df, CFG, fold=9)

    def test_feature_order_follows_config(self) -> None:
        train = self.splits["train"]
        self.assertEqual(train.feature_names, CFG["features"]["tabular_columns"])
        np.testing.assert_allclose(
            train.x[0], [FEATURE_VALUES[c] for c in train.feature_names])

    def test_event_keys_and_crossing_times(self) -> None:
        val = self.splits["validation"]
        self.assertEqual(set(val.crossing_h), {"R1-1", "R2-1"})  # same id, distinct keys
        self.assertAlmostEqual(val.crossing_h["R1-1"], 108.0)
        self.assertAlmostEqual(val.crossing_h["R2-1"], 108.0)
        # Windows of positives carry their event key; negatives carry None.
        keyed = {k for k in val.event_key if k is not None}
        self.assertEqual(keyed, {"R1-1"})   # R2's positive windows were dropped
        self.assertTrue(all(k is None for k in val.event_key[val.y == 0]))

    def test_times_are_hours_since_global_origin(self) -> None:
        train = self.splits["train"]
        self.assertAlmostEqual(train.times_h.min(), 0.0)
        self.assertAlmostEqual(self.splits["validation"].times_h.min(), 100.0)

    def test_missing_column_raises(self) -> None:
        with self.assertRaises(KeyError):
            build_development_splits(self.df.drop(columns=["rain_24h"]), CFG)

    def test_nan_features_among_usable_windows_raise(self) -> None:
        bad = self.df.copy()
        bad.loc[0, "rain_1h"] = np.nan   # row 0 is a usable train window
        with self.assertRaises(ValueError):
            build_development_splits(bad, CFG)


class TestEventReport(unittest.TestCase):
    def test_counts_at_three_hours(self) -> None:
        val = build_development_splits(make_table(), CFG, horizon_h=3)["validation"]
        report = event_report(val)
        # R2's only event lost all its usable pre-onset windows at N = 3.
        self.assertEqual(report.loc["ALL", "windows"], 17)
        self.assertEqual(report.loc["ALL", "positives"], 3)
        self.assertEqual(report.loc["ALL", "events_total"], 2)
        self.assertEqual(report.loc["ALL", "events_evaluable"], 1)
        self.assertEqual(report.loc["ALL", "events_unevaluable"], 1)
        self.assertEqual(report.loc["R2", "events_unevaluable"], 1)
        self.assertEqual(report.loc["R1", "events_evaluable"], 1)

    def test_counts_depend_on_the_horizon(self) -> None:
        # At N = 6, R2 keeps hours 102-104 (lead 6, 5, 4 h) -> evaluable again.
        val = build_development_splits(make_table(), CFG, horizon_h=6)["validation"]
        report = event_report(val)
        self.assertEqual(report.loc["ALL", "positives"], 9)    # 6 (R1) + 3 (R2)
        self.assertEqual(report.loc["ALL", "events_evaluable"], 2)
        self.assertEqual(report.loc["ALL", "events_unevaluable"], 0)

    def test_pooled_row_equals_sum_of_rivers(self) -> None:
        train = build_development_splits(make_table(), CFG)["train"]
        report = event_report(train)
        rivers = report.drop(index="ALL")
        self.assertEqual(report.loc["ALL", "windows"], rivers["windows"].sum())


class TestMinimumEventsWarning(unittest.TestCase):
    def setUp(self) -> None:
        self.val = build_development_splits(make_table(), CFG)["validation"]

    def test_warns_when_too_few_events(self) -> None:
        message = check_min_events(self.val, minimum=10)
        self.assertIsNotNone(message)
        self.assertIn("exploratory", message)
        self.assertIn("1 evaluable events", message)

    def test_no_warning_when_enough_events(self) -> None:
        self.assertIsNone(check_min_events(self.val, minimum=1))


if __name__ == "__main__":
    unittest.main()
