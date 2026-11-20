"""Unit tests for src/baselines.py.

The rule test reuses the 12-window hand example from test_metrics.py
(horizon N = 3 h; event A crosses at t = 5, event B at t = 10; the windows
that can warn for A are t = 2, 3, 4 and for B are t = 7, 8, 9).

Here event B is a FLASHY rise: its level stays low but the rise is fast.
A level rule must therefore lower its alarm level a lot (paying many false
alarms) to catch B, while a rate rule catches B cheaply.
"""
from __future__ import annotations

import unittest

import numpy as np

from src.baselines import (
    fit_logistic_regression,
    fixed_threshold_scores,
    logistic_scores,
    rate_of_rise_scores,
)
from src.metrics import select_threshold

N_HOURS = 3.0
TIMES = list(range(12))
KEYS = [None, None, "A", "A", "A", None, None, "B", "B", "B", None, None]
CROSSING = {"A": 5.0, "B": 10.0}
Y_TRUE = [0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0]

#               t0   t1   t2   t3   t4   t5   t6   t7   t8   t9   t10  t11
LEVEL = np.array([.50, .45, .60, .70, .80, .55, .40, .20, .25, .30, .35, .38])
RATE = np.array([.00, .01, .02, .03, .04, .02, .00, .10, .15, .20, .01, .00])


class TestRuleBaselines(unittest.TestCase):
    def test_scores_pass_values_through(self) -> None:
        np.testing.assert_allclose(fixed_threshold_scores(LEVEL), LEVEL)
        np.testing.assert_allclose(rate_of_rise_scores(RATE), RATE)

    def test_nan_is_rejected(self) -> None:
        bad = LEVEL.copy()
        bad[3] = np.nan
        with self.assertRaises(ValueError):
            fixed_threshold_scores(bad)
        with self.assertRaises(ValueError):
            rate_of_rise_scores(bad)

    def _select(self, scores: np.ndarray) -> dict:
        return select_threshold(
            scores, Y_TRUE, KEYS, TIMES, CROSSING, N_HOURS,
            c_fn=10, c_fp=1, min_event_recall=1.0,
        )

    def test_level_rule_hand_result(self) -> None:
        # B's highest level is .30, so detecting B needs alarm level <= .30.
        # Every negative window has level >= .35, so all 6 become false
        # alarms. The cheapest alarm level that catches both events is .20
        # (FN 0, FP 6): cost 6.
        r = self._select(fixed_threshold_scores(LEVEL))
        self.assertAlmostEqual(r["threshold"], 0.20)
        self.assertAlmostEqual(r["cost"], 6)
        self.assertEqual((r["fn"], r["fp"]), (0, 6))

    def test_rate_rule_hand_result(self) -> None:
        # Rate .02 catches both events (FN 0) with a single false alarm
        # (t = 5, rate .02): cost 1. Rate .01 would add 2 more false alarms.
        r = self._select(rate_of_rise_scores(RATE))
        self.assertAlmostEqual(r["threshold"], 0.02)
        self.assertAlmostEqual(r["cost"], 1)
        self.assertEqual((r["fn"], r["fp"]), (0, 1))

    def test_rate_rule_beats_level_rule_on_flashy_event(self) -> None:
        level = self._select(fixed_threshold_scores(LEVEL))["cost"]
        rate = self._select(rate_of_rise_scores(RATE))["cost"]
        self.assertLess(rate, level)


class TestLogisticRegression(unittest.TestCase):
    @staticmethod
    def _toy_data(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(400, 3))
        y = (x[:, 0] + 0.3 * rng.normal(size=400) > 1.0).astype(int)
        return x, y

    def test_scores_are_probabilities_and_rank_positives_higher(self) -> None:
        x, y = self._toy_data()
        model = fit_logistic_regression(x, y, seed=1)
        s = logistic_scores(model, x)
        self.assertTrue(((s >= 0) & (s <= 1)).all())
        self.assertGreater(s[y == 1].mean(), s[y == 0].mean())

    def test_same_seed_gives_identical_scores(self) -> None:
        x, y = self._toy_data()
        a = logistic_scores(fit_logistic_regression(x, y, seed=7), x)
        b = logistic_scores(fit_logistic_regression(x, y, seed=7), x)
        np.testing.assert_array_equal(a, b)

    def test_scaler_is_fitted_on_training_data_only(self) -> None:
        x, y = self._toy_data()
        x_other = x + 100.0  # data the model must never learn from
        model = fit_logistic_regression(x, y, seed=1)
        scaler = model.named_steps["scale"]
        np.testing.assert_allclose(scaler.mean_, x.mean(axis=0))
        self.assertFalse(np.allclose(scaler.mean_, x_other.mean(axis=0)))

    def test_class_weights_are_balanced(self) -> None:
        x, y = self._toy_data()
        model = fit_logistic_regression(x, y, seed=1)
        self.assertEqual(model.named_steps["logreg"].class_weight, "balanced")

    def test_bad_input_is_rejected(self) -> None:
        x, y = self._toy_data()
        with self.assertRaises(ValueError):
            fit_logistic_regression(x, np.zeros_like(y), seed=1)  # one class
        x_nan = x.copy()
        x_nan[0, 0] = np.nan
        with self.assertRaises(ValueError):
            fit_logistic_regression(x_nan, y, seed=1)
        with self.assertRaises(ValueError):
            fit_logistic_regression(x, y[:-1], seed=1)  # length mismatch


if __name__ == "__main__":
    unittest.main()
