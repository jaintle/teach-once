"""Synthetic 2D multi-source single-target demonstrations (Sec. V-C).

Each of K sources has its own position and orientation. All sources share
a single target frame. Each source demo is a letter-C trajectory placed
in the source frame. The target demo is a letter-C in the target frame —
it serves as the ground-truth for evaluating transported rollouts.

Anchor points follow the 5-pt cross convention from Phase 7: one cross
per frame (source or target), capturing both position and orientation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from gpt_repro.policies.demonstrations import make_letter_C_demo


@dataclass(frozen=True)
class SourceConfig:
    """Single source → target frame mapping for multi-source transport."""

    source_pos: np.ndarray   # (2,) centroid of source surface
    source_angle: float      # orientation of source frame (radians)
    target_pos: np.ndarray   # (2,) centroid of shared target
    target_angle: float      # orientation of target frame (radians)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_pos", np.asarray(self.source_pos, dtype=float))
        object.__setattr__(self, "target_pos",  np.asarray(self.target_pos,  dtype=float))


# ---------------------------------------------------------------------------
# Anchor cross helper (5-pt cross identical to Phase 7's _frame_cross)
# ---------------------------------------------------------------------------
def _cross(center: np.ndarray, angle: float, n_pts: int = 5,
           radius: float = 0.25) -> np.ndarray:
    """Return n_pts cross-shaped anchor points centred at `center`.

    Only n_pts == 5 is implemented (the paper's standard cross).
    Two sets of 5 give 10 points, as used in Phase 7 for 2 frames;
    here we use a single 5-pt cross per frame (one source → one target).
    """
    if n_pts != 5:
        raise NotImplementedError("_cross only supports n_pts=5")
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]])
    local = np.array([
        [0.0,     0.0],
        [radius,  0.0],
        [-radius, 0.0],
        [0.0,     radius],
        [0.0,    -radius],
    ])
    return local @ R.T + center


# ---------------------------------------------------------------------------
# Demo synthesis in a given frame
# ---------------------------------------------------------------------------
def _demo_in_frame(center: np.ndarray, angle: float,
                   n_points: int = 80, seed: int = 0) -> Dict[str, np.ndarray]:
    """Letter-C demo rotated and translated to (center, angle) frame."""
    demo = make_letter_C_demo(n_points=n_points, seed=seed)
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]])
    x = demo["x"] @ R.T + center
    xdot = demo["xdot"] @ R.T
    return {"x": x, "xdot": xdot, "t": demo["t"]}


# ---------------------------------------------------------------------------
# Scenario builder
# ---------------------------------------------------------------------------
def make_multisource_scenario(
    n_sources: int = 4,
    seed: int = 0,
    n_points: int = 80,
    target_pos: np.ndarray | None = None,
    target_angle: float | None = None,
) -> Dict:
    """Build a multi-source single-target scenario for Sec. V-C.

    Source positions and angles are derived deterministically from ``seed``
    so that different seeds yield different transport maps — required for
    a meaningful multi-rep benchmark. Base angles cycle through
    {-60°, -30°, 30°, 60°} and are perturbed per seed; source positions
    are placed at seed-determined distances and bearings from the origin.

    Parameters
    ----------
    n_sources : int
        Number of source frames (K).
    seed : int
        Global RNG seed; controls source positions, angles, and demo noise.
    n_points : int
        Trajectory sample count per demo.
    target_pos : (2,) array, optional
        Override target centroid (default seed-derived near [3.0, 0.5]).
    target_angle : float, optional
        Override target orientation in radians (default seed-derived).

    Returns
    -------
    dict with keys:
        ``"source_configs"`` : list[SourceConfig]  — length n_sources
        ``"source_demos"``   : list[dict]          — letter-C per source
        ``"target_demo"``    : dict                — ground-truth target demo
        ``"S_list"``         : list[np.ndarray]    — (5, 2) anchor pts per source
        ``"T"``              : np.ndarray           — (5, 2) shared target anchors
    """
    rng = np.random.default_rng(int(seed))

    if target_pos is None:
        base = np.array([3.0, 0.5])
        target_pos = base + rng.uniform(-0.5, 0.5, 2)
    else:
        target_pos = np.asarray(target_pos, dtype=float)

    if target_angle is None:
        target_angle = float(rng.uniform(-np.pi / 6, np.pi / 6))

    # Base source angles cycle through {-60°,-30°,30°,60°} + seed perturbation
    _base_angles_deg = np.array([-60.0, -30.0, 30.0, 60.0])
    _base_distances  = [2.0, 2.0, 2.0, 2.0]

    source_configs: List[SourceConfig] = []
    for k in range(n_sources):
        base_ang = np.deg2rad(_base_angles_deg[k % len(_base_angles_deg)])
        # Seed-derived bearing for source centroid (polar relative to origin)
        bearing = (k * 2.0 * np.pi / max(n_sources, 4)
                   + float(rng.uniform(-0.3, 0.3)))
        dist = float(rng.uniform(1.5, 2.5))
        source_pos = np.array([dist * np.cos(bearing), dist * np.sin(bearing)])
        source_angle = base_ang + float(rng.uniform(-0.2, 0.2))
        source_configs.append(SourceConfig(
            source_pos=source_pos,
            source_angle=float(source_angle),
            target_pos=target_pos.copy(),
            target_angle=float(target_angle),
        ))

    source_demos = [
        _demo_in_frame(cfg.source_pos, cfg.source_angle,
                       n_points=n_points, seed=int(seed) + k)
        for k, cfg in enumerate(source_configs)
    ]
    target_demo = _demo_in_frame(target_pos, target_angle,
                                 n_points=n_points, seed=int(seed) + 100)

    # Anchor point sets: 5-pt cross per source, shared 5-pt cross for target
    T = _cross(target_pos, target_angle)
    S_list: List[np.ndarray] = [
        _cross(cfg.source_pos, cfg.source_angle) for cfg in source_configs
    ]

    return {
        "source_configs": source_configs,
        "source_demos":   source_demos,
        "target_demo":    target_demo,
        "S_list":         S_list,
        "T":              T,
    }
