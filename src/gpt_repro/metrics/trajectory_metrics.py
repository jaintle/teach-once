"""Trajectory comparison metrics (Sec. V-B).

Five metrics, each ``(pred, gt) -> float``. The first three are
similarity measures already provided by the ``similaritymeasures``
package; the last two are simple Euclidean / angular endpoints.
"""

from __future__ import annotations

import numpy as np
import similaritymeasures as sm


def _validate(pred: np.ndarray, gt: np.ndarray) -> None:
    if pred.ndim != 2 or gt.ndim != 2 or pred.shape[1] != 2 or gt.shape[1] != 2:
        raise ValueError(
            f"pred and gt must be (N, 2); got {pred.shape} and {gt.shape}"
        )


def frechet_distance(pred: np.ndarray, gt: np.ndarray) -> float:
    """Discrete Fréchet distance between two 2D polylines."""
    _validate(pred, gt)
    return float(sm.frechet_dist(pred, gt))


def area_between_curves(pred: np.ndarray, gt: np.ndarray) -> float:
    """Area between two 2D curves (Sec. V-B metric 2)."""
    _validate(pred, gt)
    return float(sm.area_between_two_curves(pred, gt))


def dtw_distance(pred: np.ndarray, gt: np.ndarray) -> float:
    """Dynamic-Time-Warping distance (Sec. V-B metric 3)."""
    _validate(pred, gt)
    val, _ = sm.dtw(pred, gt)
    return float(val)


def final_position_error(pred: np.ndarray, gt: np.ndarray) -> float:
    """Euclidean distance between the two final points."""
    _validate(pred, gt)
    return float(np.linalg.norm(pred[-1] - gt[-1]))


def _approach_direction(traj: np.ndarray) -> np.ndarray:
    d = traj[-1] - traj[-2]
    n = float(np.linalg.norm(d))
    if n < 1e-12:
        return np.array([1.0, 0.0])
    return d / n


def final_orientation_error(pred: np.ndarray, gt: np.ndarray) -> float:
    """Unsigned angle (radians) between the two final approach directions."""
    _validate(pred, gt)
    if pred.shape[0] < 2 or gt.shape[0] < 2:
        return 0.0
    d1 = _approach_direction(pred)
    d2 = _approach_direction(gt)
    dot = float(np.clip(np.dot(d1, d2), -1.0, 1.0))
    return float(np.arccos(dot))


METRIC_FNS = {
    "frechet":       frechet_distance,
    "area":          area_between_curves,
    "dtw":           dtw_distance,
    "final_pos":     final_position_error,
    "final_orient":  final_orientation_error,
}

METRIC_LABELS = {
    "frechet":      "Fréchet",
    "area":         "Area",
    "dtw":          "DTW",
    "final_pos":    "Final pos. err",
    "final_orient": "Final orient. err",
}
