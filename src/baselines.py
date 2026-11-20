"""Baseline early-warning methods.

Three baselines from the proposal:

    1. Fixed-threshold rule: warn when the current level is high.
    2. Rate-of-rise rule:    warn when the level is rising quickly.
    3. Logistic regression:  a linear model on the window features.

Every baseline produces a SCORE per window (higher = more dangerous). The
decision threshold is then chosen on VALIDATION data only with
metrics.select_threshold, exactly as for the other models, so the
comparison is fair. The rule baselines therefore have a tuned alarm level
and rate, which makes them stronger than an untuned sensor alarm.

All functions work on plain arrays so they do not depend on a dataset's
column names. Feature selection and column names live in data_prep.py.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _check_finite(array: np.ndarray, name: str) -> np.ndarray:
    """Return a float array, refusing NaN or infinite values.

    Only windows with complete features should reach the baselines; a NaN
    here means the eligibility filter was skipped.
    """
    values = np.asarray(array, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(
            f"{name} contains NaN or infinite values; keep only windows with "
            "complete features (features_complete == 1)."
        )
    return values


def fixed_threshold_scores(level: np.ndarray) -> np.ndarray:
    """Score for the fixed-threshold rule: the current level itself.

    'Warn when level >= alarm level' is the same as thresholding this score.
    Pass the level relative to the flood level (for FlowDB: cfs_rel) so the
    alarm level is comparable across rivers.

    Args:
        level: 1-D array, current level (or flow) per window.
    """
    values = _check_finite(level, "level").ravel()
    return values


def rate_of_rise_scores(rate_of_rise: np.ndarray) -> np.ndarray:
    """Score for the rate-of-rise rule: the one-step rise itself.

    'Warn when the level rises faster than a set rate' is the same as
    thresholding this score. Pass the relative rise per time step (for
    FlowDB: dcfs_rel).

    Args:
        rate_of_rise: 1-D array, rise per time step for each window.
    """
    values = _check_finite(rate_of_rise, "rate_of_rise").ravel()
    return values


def fit_logistic_regression(
    x_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    c: float = 1.0,
    max_iter: int = 1000,
) -> Pipeline:
    """Fit a standardised, class-weighted logistic regression.

    The scaler is part of the pipeline and is fitted on TRAINING data only.
    Class weights are 'balanced' (rare floods count more), which is the
    class-weighting option from proposal Step 5.

    Args:
        x_train: 2-D array (windows x features), training windows only.
        y_train: 1-D array of 0/1 labels for those windows.
        seed: Random seed (from config.yaml).
        c: Inverse regularisation strength (smaller = stronger).
        max_iter: Solver iteration limit.

    Returns:
        A fitted scikit-learn Pipeline (scale -> logistic regression).
    """
    features = _check_finite(x_train, "x_train")
    if features.ndim != 2:
        raise ValueError("x_train must be 2-D (windows x features)")
    labels = np.asarray(y_train).astype(int).ravel()
    if labels.shape[0] != features.shape[0]:
        raise ValueError("x_train and y_train must have the same length")
    if np.unique(labels).size < 2:
        raise ValueError("y_train needs both classes (0 and 1) to fit")

    model = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    C=c,
                    class_weight="balanced",
                    max_iter=max_iter,
                    random_state=seed,
                ),
            ),
        ]
    )
    model.fit(features, labels)
    return model


def logistic_scores(model: Pipeline, x: np.ndarray) -> np.ndarray:
    """Predicted flood probability for each window (the score).

    Args:
        model: Pipeline returned by fit_logistic_regression.
        x: 2-D array of windows, same feature order as in training.
    """
    features = _check_finite(x, "x")
    if features.ndim != 2:
        raise ValueError("x must be 2-D (windows x features)")
    return model.predict_proba(features)[:, 1]
