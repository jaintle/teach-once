"""Synthetic 2D multi-reference-frame demonstrations (Sec. V-B).

Each demonstration is a curved trajectory from a *start frame* to a
*goal frame*. A frame is a position plus an in-plane orientation (the
local approach direction). The trajectory respects both frames: it
leaves the start tangent to the start-frame orientation and approaches
the goal tangent to the goal-frame orientation. This is the synthetic
stand-in for the paper's real demonstrations in Sec. V-B.

Each frame is then tracked by a short cross-shaped point set (5 pts per
frame), matching the paper: "only 5 points are tracked w.r.t. each
reference frame, capturing the position but also the local orientation."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class FrameConfig:
    """Pair of 2D reference frames defining a demonstration's boundary conditions."""

    start_pos: np.ndarray          # (2,)
    start_angle: float             # radians
    goal_pos: np.ndarray           # (2,)
    goal_angle: float              # radians

    def __post_init__(self) -> None:
        object.__setattr__(self, "start_pos", np.asarray(self.start_pos, dtype=float))
        object.__setattr__(self, "goal_pos",  np.asarray(self.goal_pos,  dtype=float))


# ---------------------------------------------------------------------------
# Demonstration synthesis
# ---------------------------------------------------------------------------
def _cubic_bezier(p0, p1, p2, p3, t):
    one_t = 1.0 - t
    return (
        one_t[:, None] ** 3 * p0
        + 3 * one_t[:, None] ** 2 * t[:, None] * p1
        + 3 * one_t[:, None] * t[:, None] ** 2 * p2
        + t[:, None] ** 3 * p3
    )


def make_multiframe_demo(
    config: FrameConfig,
    n_points: int = 80,
    noise: float = 0.02,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """Cubic-Bezier trajectory consistent with the 2-frame ``config``.

    Returns ``{"x": (N, 2), "xdot": (N, 2), "t": (N,)}``. The control
    handles are chosen so the trajectory leaves the start frame tangent
    to ``start_angle`` and arrives at the goal tangent to ``goal_angle``.
    """
    if n_points < 3:
        raise ValueError("n_points must be >= 3")
    rng = np.random.default_rng(int(seed))
    p0 = config.start_pos
    p3 = config.goal_pos
    dist = float(np.linalg.norm(p3 - p0))
    handle = 0.4 * dist
    p1 = p0 + handle * np.array([np.cos(config.start_angle), np.sin(config.start_angle)])
    p2 = p3 - handle * np.array([np.cos(config.goal_angle), np.sin(config.goal_angle)])
    t = np.linspace(0.0, 1.0, n_points)
    pos = _cubic_bezier(p0, p1, p2, p3, t)
    if noise > 0:
        pos = pos + float(noise) * rng.standard_normal(pos.shape)
    vel = np.zeros_like(pos)
    vel[0] = (pos[1] - pos[0]) / (t[1] - t[0])
    vel[-1] = (pos[-1] - pos[-2]) / (t[-1] - t[-2])
    vel[1:-1] = (pos[2:] - pos[:-2]) / (t[2:] - t[:-2])[:, None]
    return {"x": pos, "xdot": vel, "t": t}


# ---------------------------------------------------------------------------
# Library of 9 frame configurations
# ---------------------------------------------------------------------------
def make_9_frame_configs(seed: int = 0) -> List[FrameConfig]:
    """Deterministic list of 9 diverse (start, goal) frame configurations."""
    rng = np.random.default_rng(int(seed))
    configs: List[FrameConfig] = []
    for _ in range(9):
        start_pos = rng.uniform(low=[-2.0, -1.5], high=[-0.5, 1.5])
        goal_pos  = rng.uniform(low=[ 0.5, -1.5], high=[ 2.0, 1.5])
        start_angle = rng.uniform(-np.pi / 3, np.pi / 3)
        goal_angle  = rng.uniform(np.pi - np.pi / 3, np.pi + np.pi / 3)
        configs.append(FrameConfig(
            start_pos=start_pos, start_angle=float(start_angle),
            goal_pos=goal_pos,   goal_angle=float(goal_angle),
        ))
    return configs


# ---------------------------------------------------------------------------
# Frame points (Sec. V-B: 5 pts per frame)
# ---------------------------------------------------------------------------
def _frame_cross(center: np.ndarray, angle: float, n_pts: int = 5,
                 radius: float = 0.25) -> np.ndarray:
    if n_pts != 5:
        raise NotImplementedError("frame cross only implemented for n_pts=5")
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    local = np.array([
        [ 0.0,  0.0],         # frame origin
        [ radius,  0.0],      # +x (local)
        [-radius,  0.0],      # -x
        [ 0.0,    radius],    # +y
        [ 0.0,   -radius],    # -y
    ])
    return local @ R.T + center


def get_frame_points(
    config: FrameConfig, n_pts_per_frame: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(S_points, T_points)`` of shape ``(2 * n_pts_per_frame, 2)``.

    Source = canonical frames (start at origin facing +x, goal at the
    start↔goal distance facing -x). Target = the supplied ``config``.
    So :math:`\\phi : S \\to T` is non-trivial whenever the queried
    config differs from the canonical one.
    """
    dist = float(np.linalg.norm(config.goal_pos - config.start_pos))
    S = np.vstack([
        _frame_cross(np.array([0.0, 0.0]), 0.0,    n_pts_per_frame),
        _frame_cross(np.array([dist, 0.0]), np.pi, n_pts_per_frame),
    ])
    T = np.vstack([
        _frame_cross(config.start_pos, config.start_angle, n_pts_per_frame),
        _frame_cross(config.goal_pos,  config.goal_angle,  n_pts_per_frame),
    ])
    return S, T


def make_canonical_demo(config: FrameConfig, n_points: int = 80,
                        noise: float = 0.0, seed: int = 0) -> Dict[str, np.ndarray]:
    """Demonstration in the *canonical* frame matching ``config``'s
    start↔goal distance. Used by single-demo methods (GPT, DMP) as the
    "source" trajectory that is then transported to the target frames."""
    dist = float(np.linalg.norm(config.goal_pos - config.start_pos))
    canonical = FrameConfig(
        start_pos=np.array([0.0, 0.0]), start_angle=0.0,
        goal_pos=np.array([dist, 0.0]), goal_angle=np.pi,
    )
    return make_multiframe_demo(canonical, n_points=n_points, noise=noise, seed=seed)
