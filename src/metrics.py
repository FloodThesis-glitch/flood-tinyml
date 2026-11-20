"""Evaluation metrics for flood early warning.

Window-level metrics score every prediction window on its own. Event-level
metrics ask the question that matters for warning: was each flood event
announced in time? Also contains the threshold-selection rule from the
proposal (minimise cost subject to a minimum event recall).

Conventions used throughout:
    * A warning is issued when score >= threshold.
    * Times are in hours (floats). Pass ONLY eligible windows (rows with
      eligible_Nh == 1 and features_complete == 1).
    * An event is identified by a hashable key that is unique across the
      whole evaluation set (for example "river-eventid"). Windows that
      belong to no event simply carry a key that is not in `crossing_h`.
"""
from __future__ import annotations

from typing import Any, Hashable, Mapping, Sequence

import numpy as np

# Tolerance for floating-point comparison of hour values.
_EPS = 1e-9


def _safe_div(numerator: float, denominator: float) -> float:
    """Divide, returning NaN (not 0) when the denominator is zero."""
    return float("nan") if denominator == 0 else numerator / denominator


def _as_binary(values: Sequence[int] | np.ndarray, name: str) -> np.ndarray:
    """Convert to a 1-D int array and check that it holds only 0 and 1."""
    array = np.asarray(values).astype(int).ravel()
    if not np.isin(array, (0, 1)).all():
        raise ValueError(f"{name} must contain only 0 and 1")
    return array


# --------------------------------------------------------------------------
# Window-level metrics
# --------------------------------------------------------------------------
def window_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    c_fn: float = 10.0,
    c_fp: float = 1.0,
) -> dict[str, float]:
    """Window-level confusion counts, scores and cost.

    Args:
        y_true: 1 if a flood crossing follows within N hours, else 0.
        y_pred: 1 if the model issued a warning, else 0.
        c_fn: Cost of one missed (false-negative) window.
        c_fp: Cost of one false-alarm (false-positive) window.

    Returns:
        Dictionary with tp, fp, fn, tn, accuracy, precision, recall, f1,
        false_alarm_rate (FP / (FP + TN)), false_alarm_ratio
        (FP / (TP + FP), equal to 1 - precision) and cost
        (c_fn * FN + c_fp * FP). Undefined ratios are NaN.
    """
    truth = _as_binary(y_true, "y_true")
    pred = _as_binary(y_pred, "y_pred")
    if truth.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same length")

    tp = int(((truth == 1) & (pred == 1)).sum())
    fp = int(((truth == 0) & (pred == 1)).sum())
    fn = int(((truth == 1) & (pred == 0)).sum())
    tn = int(((truth == 0) & (pred == 0)).sum())

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    if np.isnan(precision) or np.isnan(recall) or (precision + recall) == 0:
        f1 = float("nan") if (np.isnan(precision) or np.isnan(recall)) else 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": _safe_div(tp + tn, tp + fp + fn + tn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_alarm_rate": _safe_div(fp, fp + tn),
        "false_alarm_ratio": _safe_div(fp, tp + fp),
        "cost": float(c_fn * fn + c_fp * fp),
    }


# --------------------------------------------------------------------------
# Event-level metrics
# --------------------------------------------------------------------------
def prepare_events(
    event_key: Sequence[Hashable] | np.ndarray,
    times_h: Sequence[float] | np.ndarray,
    crossing_h: Mapping[Hashable, float],
    horizon_h: float,
) -> list[tuple[Hashable, np.ndarray, np.ndarray]]:
    """Find, for every event, the windows where a warning would count.

    A warning counts for an event only if it is issued strictly before the
    threshold crossing and no more than `horizon_h` hours before it. Windows
    that belong to the event but come after its first crossing (short dips
    inside an ongoing event) are therefore ignored.

    Returns:
        List of (key, window_indices, lead_hours) for each event that has at
        least one such window. Events with none are not evaluable and are
        left out (event_metrics reports how many).
    """
    keys = np.asarray(event_key, dtype=object).ravel()
    times = np.asarray(times_h, dtype=float).ravel()
    if keys.shape != times.shape:
        raise ValueError("event_key and times_h must have the same length")

    # Group window positions by event key in a single pass.
    groups: dict[Hashable, list[int]] = {}
    for position, key in enumerate(keys):
        if key in crossing_h:
            groups.setdefault(key, []).append(position)

    prepared: list[tuple[Hashable, np.ndarray, np.ndarray]] = []
    for key, positions in groups.items():
        member = np.array(positions, dtype=int)
        lead = crossing_h[key] - times[member]
        valid = (lead > _EPS) & (lead <= horizon_h + _EPS)
        if valid.any():
            prepared.append((key, member[valid], lead[valid]))
    return prepared


def event_metrics_from_prepared(
    y_pred: Sequence[int] | np.ndarray,
    prepared: list[tuple[Hashable, np.ndarray, np.ndarray]],
    n_events_total: int,
    horizon_h: float,
) -> dict[str, Any]:
    """Event-level recall and lead time from pre-computed event windows."""
    pred = _as_binary(y_pred, "y_pred")

    detected = 0
    missed: list[Hashable] = []
    leads: list[float] = []
    for key, indices, lead in prepared:
        fired = pred[indices] == 1
        if fired.any():
            detected += 1
            leads.append(float(lead[fired].max()))  # earliest warning
        else:
            missed.append(key)

    n_evaluable = len(prepared)
    lead_arr = np.array(leads, dtype=float)
    has_leads = lead_arr.size > 0
    return {
        "n_events_total": int(n_events_total),
        "n_events_evaluable": n_evaluable,
        "n_events_unevaluable": int(n_events_total) - n_evaluable,
        "n_detected": detected,
        "event_recall": _safe_div(detected, n_evaluable),
        "missed_events": missed,
        "lead_time_median_h": float(np.median(lead_arr)) if has_leads else float("nan"),
        "lead_time_min_h": float(lead_arr.min()) if has_leads else float("nan"),
        "lead_frac_median": float(np.median(lead_arr) / horizon_h) if has_leads else float("nan"),
        "lead_frac_min": float(lead_arr.min() / horizon_h) if has_leads else float("nan"),
    }


def event_metrics(
    y_pred: Sequence[int] | np.ndarray,
    event_key: Sequence[Hashable] | np.ndarray,
    times_h: Sequence[float] | np.ndarray,
    crossing_h: Mapping[Hashable, float],
    horizon_h: float,
) -> dict[str, Any]:
    """Event-level recall and lead time.

    An event is detected if at least one warning is issued within the
    horizon before its threshold crossing. Lead time is the number of hours
    between the FIRST such warning and the crossing.

    Args:
        y_pred: 0/1 warning per window.
        event_key: Per window, the key of the event it heads toward.
        times_h: Per window, the time in hours.
        crossing_h: Mapping from event key to the hour of its first crossing.
        horizon_h: Prediction horizon N in hours.

    Returns:
        Dictionary with event counts, event_recall (detected / evaluable),
        the keys of missed events, and lead time median and minimum, both in
        hours and as a fraction of N. Events with no evaluable window are
        counted in n_events_unevaluable and excluded from the recall
        denominator.
    """
    prepared = prepare_events(event_key, times_h, crossing_h, horizon_h)
    return event_metrics_from_prepared(y_pred, prepared, len(crossing_h), horizon_h)


# --------------------------------------------------------------------------
# Threshold selection (proposal Step 5)
# --------------------------------------------------------------------------
def select_threshold(
    scores: Sequence[float] | np.ndarray,
    y_true: Sequence[int] | np.ndarray,
    event_key: Sequence[Hashable] | np.ndarray,
    times_h: Sequence[float] | np.ndarray,
    crossing_h: Mapping[Hashable, float],
    horizon_h: float,
    c_fn: float = 10.0,
    c_fp: float = 1.0,
    min_event_recall: float = 0.90,
    max_candidates: int = 500,
) -> dict[str, Any]:
    """Pick the decision threshold on VALIDATION data only.

    Minimises cost = c_fn * FN + c_fp * FP over candidate thresholds, subject
    to event recall >= min_event_recall. A warning is score >= threshold.

    Ties in cost go to the LOWEST threshold (the more sensitive one).
    If no threshold reaches the required event recall, the result has
    feasible=False and holds the threshold with the highest event recall
    (lowest cost among those); the caller must report this.

    Returns:
        Dictionary with threshold, feasible, cost, event_recall and the
        window-level metrics at that threshold.
    """
    score_arr = np.asarray(scores, dtype=float).ravel()
    truth = _as_binary(y_true, "y_true")
    if score_arr.shape != truth.shape:
        raise ValueError("scores and y_true must have the same length")

    unique_scores = np.unique(score_arr)
    if unique_scores.size > max_candidates:
        picks = np.linspace(0, unique_scores.size - 1, max_candidates).astype(int)
        unique_scores = unique_scores[picks]

    prepared = prepare_events(event_key, times_h, crossing_h, horizon_h)
    n_total = len(crossing_h)

    rows: list[tuple[float, float, float]] = []  # (threshold, cost, recall)
    for tau in unique_scores:
        pred = (score_arr >= tau).astype(int)
        cost = window_metrics(truth, pred, c_fn, c_fp)["cost"]
        recall = event_metrics_from_prepared(pred, prepared, n_total, horizon_h)["event_recall"]
        rows.append((float(tau), cost, recall))

    feasible_rows = [r for r in rows if r[2] >= min_event_recall - _EPS]
    if feasible_rows:
        best = min(feasible_rows, key=lambda r: (r[1], r[0]))
        feasible = True
    else:
        valid = [r for r in rows if not np.isnan(r[2])]
        if not valid:
            raise ValueError(
                "No evaluable events: cannot select a threshold. Check "
                "event_key, times_h and crossing_h."
            )
        top_recall = max(r[2] for r in valid)
        best = min((r for r in valid if r[2] >= top_recall - _EPS), key=lambda r: (r[1], r[0]))
        feasible = False

    tau = best[0]
    pred = (score_arr >= tau).astype(int)
    result: dict[str, Any] = {
        "threshold": tau,
        "feasible": feasible,
        "event_recall": best[2],
        "cost": best[1],
    }
    result.update(window_metrics(truth, pred, c_fn, c_fp))
    return result
