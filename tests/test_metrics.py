"""Unit tests for src/metrics.py using a tiny hand-made example.

THE HAND-MADE EXAMPLE (horizon N = 3 hours, one window per hour, t = 0..11)

    t         : 0  1  2  3  4  5  6  7  8  9  10 11
    event     : -  -  A  A  A  -  -  B  B  B  -  -
    y_true    : 0  0  1  1  1  0  0  1  1  1  0  0

Event A first crosses the flood threshold at t = 5, event B at t = 10.
So the windows that can warn for A are t = 2, 3, 4 (lead 3, 2, 1 hours)
and for B are t = 7, 8, 9 (lead 3, 2, 1 hours).
"""
from __future__ import annotations

import math
import unittest

import numpy as np

from src.metrics import event_metrics, select_threshold, window_metrics

N_HOURS = 3.0
TIMES = list(range(12))
KEYS = [None, None, "A", "A", "A", None, None, "B", "B", "B", None, None]
CROSSING = {"A": 5.0, "B": 10.0}
Y_TRUE = [0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0]


class TestWindowMetrics(unittest.TestCase):
    def test_hand_example(self) -> None:
        # Warnings at t = 0, 3, 4, 11.
        y_pred = [1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1]
        m = window_metrics(Y_TRUE, y_pred, c_fn=10, c_fp=1)
        # TP: t=3,4          -> 2
        # FP: t=0,11         -> 2
        # FN: t=2,7,8,9      -> 4
        # TN: the other 4    -> 4
        self.assertEqual((m["tp"], m["fp"], m["fn"], m["tn"]), (2, 2, 4, 4))
        self.assertAlmostEqual(m["accuracy"], 6 / 12)
        self.assertAlmostEqual(m["precision"], 2 / 4)
        self.assertAlmostEqual(m["recall"], 2 / 6)
        self.assertAlmostEqual(m["f1"], 0.4)           # 2*0.5*(1/3)/(0.5+1/3)
        self.assertAlmostEqual(m["false_alarm_rate"], 2 / 6)   # FP/(FP+TN)
        self.assertAlmostEqual(m["false_alarm_ratio"], 2 / 4)  # FP/(TP+FP)
        self.assertAlmostEqual(m["cost"], 10 * 4 + 1 * 2)      # 42

    def test_cost_ratios(self) -> None:
        y_pred = [1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1]
        self.assertAlmostEqual(window_metrics(Y_TRUE, y_pred, 5, 1)["cost"], 22)
        self.assertAlmostEqual(window_metrics(Y_TRUE, y_pred, 20, 1)["cost"], 82)

    def test_undefined_values_are_nan_not_zero(self) -> None:
        m = window_metrics([0, 0, 0], [0, 0, 0])  # no positives, no warnings
        self.assertTrue(math.isnan(m["precision"]))
        self.assertTrue(math.isnan(m["recall"]))
        self.assertEqual(m["fp"], 0)

    def test_rejects_bad_input(self) -> None:
        with self.assertRaises(ValueError):
            window_metrics([0, 1], [0, 2])
        with self.assertRaises(ValueError):
            window_metrics([0, 1, 1], [0, 1])


class TestEventMetrics(unittest.TestCase):
    def test_one_detected_one_missed(self) -> None:
        # Warnings at t = 0, 3, 4, 11. Event A: first warning t=3 -> lead
        # 5-3 = 2 h. Event B: no warning in 7..9 -> missed.
        y_pred = [1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1]
        m = event_metrics(y_pred, KEYS, TIMES, CROSSING, N_HOURS)
        self.assertEqual(m["n_events_evaluable"], 2)
        self.assertEqual(m["n_detected"], 1)
        self.assertAlmostEqual(m["event_recall"], 0.5)
        self.assertEqual(m["missed_events"], ["B"])
        self.assertAlmostEqual(m["lead_time_median_h"], 2.0)
        self.assertAlmostEqual(m["lead_time_min_h"], 2.0)
        self.assertAlmostEqual(m["lead_frac_median"], 2 / 3)

    def test_both_detected_lead_statistics(self) -> None:
        # Warnings at t = 4, 7, 8. A: lead 5-4 = 1 h. B: first warning t=7,
        # lead 10-7 = 3 h (the later warning at t=8 does not change it).
        y_pred = [0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0]
        m = event_metrics(y_pred, KEYS, TIMES, CROSSING, N_HOURS)
        self.assertAlmostEqual(m["event_recall"], 1.0)
        self.assertAlmostEqual(m["lead_time_median_h"], 2.0)   # median(1, 3)
        self.assertAlmostEqual(m["lead_time_min_h"], 1.0)
        self.assertAlmostEqual(m["lead_frac_median"], 2 / 3)
        self.assertAlmostEqual(m["lead_frac_min"], 1 / 3)

    def test_warning_after_crossing_does_not_count(self) -> None:
        # Warnings at t = 5, 6 come at or after A's crossing (t = 5).
        y_pred = [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
        m = event_metrics(y_pred, KEYS, TIMES, CROSSING, N_HOURS)
        self.assertEqual(m["n_detected"], 0)
        self.assertTrue(math.isnan(m["lead_time_median_h"]))

    def test_warning_earlier_than_horizon_does_not_count(self) -> None:
        # t = 1 is 4 h before A's crossing, outside the 3 h horizon.
        keys = list(KEYS)
        keys[1] = "A"  # pretend this window was labelled with event A
        y_pred = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        m = event_metrics(y_pred, keys, TIMES, CROSSING, N_HOURS)
        self.assertEqual(m["n_detected"], 0)

    def test_dip_window_inside_event_is_ignored(self) -> None:
        # A window at t = 6 labelled with event A comes AFTER A's crossing
        # (a dip inside an ongoing event). Its warning must not detect A.
        keys = list(KEYS)
        keys[6] = "A"
        y_pred = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        m = event_metrics(y_pred, keys, TIMES, CROSSING, N_HOURS)
        self.assertEqual(m["n_detected"], 0)
        self.assertEqual(m["missed_events"], ["A", "B"])

    def test_event_without_usable_window_is_reported_not_missed(self) -> None:
        # Event C has no windows at all (for example sensor data missing).
        crossing = dict(CROSSING, C=20.0)
        y_pred = [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]
        m = event_metrics(y_pred, KEYS, TIMES, crossing, N_HOURS)
        self.assertEqual(m["n_events_total"], 3)
        self.assertEqual(m["n_events_evaluable"], 2)
        self.assertEqual(m["n_events_unevaluable"], 1)
        self.assertAlmostEqual(m["event_recall"], 1.0)


class TestSelectThreshold(unittest.TestCase):
    """Scores chosen so the best threshold can be found by hand.

        t      : 0    1    2    3    4    5    6    7    8    9    10   11
        score  : .10  .20  .60  .70  .80  .35  .15  .30  .40  .50  .05  .55
        y_true : 0    0    1    1    1    0    0    1    1    1    0    0

    Event A detected iff threshold <= 0.80; event B iff threshold <= 0.50.
    With c_fn = 1 and c_fp = 10 (false alarms expensive):
        tau .30 -> FN0 FP2 cost 20      tau .55 -> FN3 FP1 cost 13
        tau .35 -> FN1 FP2 cost 21      tau .60 -> FN3 FP0 cost  3 (B missed)
        tau .40 -> FN1 FP1 cost 11      tau .70 -> cost 4,  tau .80 -> cost 5
        tau .50 -> FN2 FP1 cost 12
    """

    SCORES = [.10, .20, .60, .70, .80, .35, .15, .30, .40, .50, .05, .55]

    def _select(self, min_event_recall: float) -> dict:
        return select_threshold(
            self.SCORES, Y_TRUE, KEYS, TIMES, CROSSING, N_HOURS,
            c_fn=1, c_fp=10, min_event_recall=min_event_recall,
        )

    def test_recall_constraint_is_binding(self) -> None:
        # Cheapest overall is tau .60 (cost 3) but it misses event B, so with
        # a required event recall of 1.0 the answer must be tau .40 (cost 11).
        r = self._select(1.0)
        self.assertAlmostEqual(r["threshold"], 0.40)
        self.assertAlmostEqual(r["cost"], 11)
        self.assertAlmostEqual(r["event_recall"], 1.0)
        self.assertTrue(r["feasible"])

    def test_relaxed_constraint_picks_cheapest(self) -> None:
        r = self._select(0.5)
        self.assertAlmostEqual(r["threshold"], 0.60)
        self.assertAlmostEqual(r["cost"], 3)
        self.assertTrue(r["feasible"])

    def test_infeasible_target_is_flagged(self) -> None:
        # Event recall cannot exceed 1.0, so 1.5 is unreachable. The result
        # falls back to the best achievable recall (1.0) at lowest cost.
        r = self._select(1.5)
        self.assertFalse(r["feasible"])
        self.assertAlmostEqual(r["event_recall"], 1.0)
        self.assertAlmostEqual(r["threshold"], 0.40)

    def test_no_evaluable_events_raises(self) -> None:
        with self.assertRaises(ValueError):
            select_threshold(
                self.SCORES, Y_TRUE, KEYS, TIMES, {"Z": 99.0}, N_HOURS,
                min_event_recall=1.0,
            )


if __name__ == "__main__":
    unittest.main()
