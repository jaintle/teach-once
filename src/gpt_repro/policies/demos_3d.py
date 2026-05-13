"""3D demonstration generators for Phase 9.

Provides helpers that produce kinematic demonstrations and scene descriptions
for the 3D reshelving and arm-pose generalization experiments (Sec. V of the
paper extended to 3-D).

All trajectories are cubic Bézier curves in R^3, matching the smooth,
point-attractor demonstrations used in the 2D experiments (Sec. V-A/B).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Low-level Bézier helper
# ---------------------------------------------------------------------------

def _cubic_bezier(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    n: int,
) -> np.ndarray:
    """Evaluate a cubic Bézier curve at ``n`` evenly-spaced parameter values.

    Returns shape (n, 3).
    """
    t = np.linspace(0.0, 1.0, n)[:, None]  # (n, 1)
    return (
        (1 - t) ** 3 * p0
        + 3 * (1 - t) ** 2 * t * p1
        + 3 * (1 - t) * t ** 2 * p2
        + t ** 3 * p3
    )


def _cubic_bezier_dot(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    n: int,
    dt: float,
) -> np.ndarray:
    """First derivative of the cubic Bézier (velocity), shape (n, 3).

    Uses second-order finite differences at the endpoints.
    """
    t = np.linspace(0.0, 1.0, n)[:, None]
    # Analytical derivative of the cubic Bézier w.r.t. parameter t
    dBdt = (
        3 * (1 - t) ** 2 * (p1 - p0)
        + 6 * (1 - t) * t * (p2 - p1)
        + 3 * t ** 2 * (p3 - p2)
    )
    # Scale by dt spacing — parameter goes 0→1 over time-span = (n-1)*dt
    time_span = (n - 1) * dt
    return dBdt / time_span


# ---------------------------------------------------------------------------
# Public API: generic 3D trajectory
# ---------------------------------------------------------------------------

def make_3d_trajectory(
    start: np.ndarray,
    goal: np.ndarray,
    n_points: int = 100,
    noise: float = 0.0,
    seed: int = 0,
    curve_scale: float = 0.3,
    dt: float = 0.02,
) -> dict:
    """Generate a smooth 3-D demonstration trajectory via a cubic Bézier curve.

    Parameters
    ----------
    start : array-like (3,) — start position.
    goal  : array-like (3,) — attractor / goal position.
    n_points : int — number of waypoints.
    noise : float — Gaussian noise std added to positions.
    seed : int — random seed for noise and control-point perturbation.
    curve_scale : float — amplitude of the control-point offsets that create
        a smooth arc (relative to the start-goal chord length).
    dt : float — time step used to compute velocity from the Bézier derivative.

    Returns
    -------
    dict with keys:
        "x"    : np.ndarray (N, 3) — positions.
        "xdot" : np.ndarray (N, 3) — velocities.
        "t"    : np.ndarray (N,)   — time stamps.
    """
    rng = np.random.default_rng(seed)
    start = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)

    chord = goal - start
    length = np.linalg.norm(chord) + 1e-10

    # Perturb interior control points perpendicular to the chord
    perp1 = rng.standard_normal(3)
    perp1 -= np.dot(perp1, chord / length) * (chord / length)
    perp1 /= np.linalg.norm(perp1) + 1e-10

    perp2 = rng.standard_normal(3)
    perp2 -= np.dot(perp2, chord / length) * (chord / length)
    perp2 /= np.linalg.norm(perp2) + 1e-10

    p1 = start + chord / 3.0 + perp1 * length * curve_scale
    p2 = start + 2.0 * chord / 3.0 + perp2 * length * curve_scale

    x = _cubic_bezier(start, p1, p2, goal, n_points)
    xdot = _cubic_bezier_dot(start, p1, p2, goal, n_points, dt)

    if noise > 0.0:
        x = x + rng.standard_normal(x.shape) * noise

    t = np.arange(n_points) * dt
    return {"x": x, "xdot": xdot, "t": t}


# ---------------------------------------------------------------------------
# Reshelving scenario
# ---------------------------------------------------------------------------

def _box_corners(center: np.ndarray, half_size: float = 0.05) -> np.ndarray:
    """Return 8 corner points of an axis-aligned cube, shape (8, 3)."""
    offsets = np.array(
        [
            [-1, -1, -1],
            [-1, -1, +1],
            [-1, +1, -1],
            [-1, +1, +1],
            [+1, -1, -1],
            [+1, -1, +1],
            [+1, +1, -1],
            [+1, +1, +1],
        ],
        dtype=float,
    )
    return center + offsets * half_size


def make_reshelving_demo(seed: int = 0) -> Tuple[dict, dict]:
    """Generate a reshelving demonstration and scene description.

    The task: pick-and-place analog where the end-effector traces a smooth
    3-D arc from an object's location to its goal location.

    Scene layout:
    * ``object_pose`` — 4×4 SE(3) homogeneous matrix for the object.
      Object center at [0.3, 0.0, 0.5]; cube half-size 0.05 m.
    * ``goal_pose``   — 4×4 SE(3) for the goal shelf position.
      Goal center at [0.0, 0.4, 0.7].
    * ``S`` — (8, 3) corner points of the object bounding box (source frame).
    * ``T`` — (8, 3) corner points of the goal bounding box (target frame).

    Returns
    -------
    demo  : dict with "x" (N,3), "xdot" (N,3), "t" (N,).
    scene : dict with "object_pose", "goal_pose", "S", "T".
    """
    obj_center = np.array([0.3, 0.0, 0.5])
    goal_center = np.array([0.0, 0.4, 0.7])

    demo = make_3d_trajectory(
        start=obj_center,
        goal=goal_center,
        n_points=100,
        noise=0.0,
        seed=seed,
        curve_scale=0.3,
    )

    obj_pose = np.eye(4)
    obj_pose[:3, 3] = obj_center

    goal_pose = np.eye(4)
    goal_pose[:3, 3] = goal_center

    S = _box_corners(obj_center, half_size=0.05)
    T = _box_corners(goal_center, half_size=0.05)

    scene = {
        "object_pose": obj_pose,
        "goal_pose": goal_pose,
        "S": S,
        "T": T,
    }
    return demo, scene


def randomize_reshelving_scene(
    base_scene: dict,
    seed: int = 0,
    pos_range: float = 0.15,
    angle_range_deg: float = 30.0,
) -> dict:
    """Return a copy of ``base_scene`` with randomized poses.

    The object and goal centers are perturbed by ±``pos_range`` uniformly,
    and each pose is rotated by a random angle in [0, ``angle_range_deg``]
    around a random axis.

    Parameters
    ----------
    base_scene : dict — output of :func:`make_reshelving_demo`.
    seed : int — random seed.
    pos_range : float — maximum position perturbation in metres.
    angle_range_deg : float — maximum rotation perturbation in degrees.

    Returns
    -------
    dict — same structure as ``base_scene``.
    """
    rng = np.random.default_rng(seed)

    def _random_rotation(rng: np.random.Generator, max_deg: float) -> np.ndarray:
        axis = rng.standard_normal(3)
        axis /= np.linalg.norm(axis) + 1e-10
        angle = rng.uniform(0.0, np.deg2rad(max_deg))
        # Rodrigues formula
        K = np.array(
            [
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0],
            ]
        )
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
        return R

    new_scene = {}
    for key in ("object_pose", "goal_pose"):
        pose = base_scene[key].copy()
        pose[:3, 3] += rng.uniform(-pos_range, pos_range, size=3)
        R_rand = _random_rotation(rng, angle_range_deg)
        pose[:3, :3] = R_rand @ pose[:3, :3]
        new_scene[key] = pose

    obj_center = new_scene["object_pose"][:3, 3]
    goal_center = new_scene["goal_pose"][:3, 3]
    new_scene["S"] = _box_corners(obj_center, half_size=0.05)
    new_scene["T"] = _box_corners(goal_center, half_size=0.05)
    return new_scene


# ---------------------------------------------------------------------------
# Arm-pose scenario
# ---------------------------------------------------------------------------

def _cross_points(center: np.ndarray, arm: float = 0.04) -> np.ndarray:
    """Return 3 axis-aligned cross points around ``center``, shape (3, 3)."""
    return center + np.array(
        [[arm, 0, 0], [0, arm, 0], [0, 0, arm]], dtype=float
    )


def make_armpose_demo(seed: int = 0) -> Tuple[dict, dict]:
    """Generate an arm-pose following demonstration and scene description.

    The end-effector traces an arc from the shoulder toward the wrist,
    mimicking a kinematic arm-following task.

    Scene keypoints:
    * shoulder : [0.0, 0.0, 1.0]
    * elbow    : [0.3, 0.0, 0.8]
    * wrist    : [0.5, 0.0, 0.6]
    * hand     : [0.6, 0.0, 0.5]

    ``S`` is a (12, 3) array of 3-point axis crosses around each keypoint
    (used as source frame for transport). ``T`` duplicates ``S`` by
    default (identity transport baseline) — callers should randomize.

    Returns
    -------
    demo  : dict with "x" (N,3), "xdot" (N,3), "t" (N,).
    scene : dict with "shoulder", "elbow", "wrist", "hand", "S", "T".
    """
    shoulder = np.array([0.0, 0.0, 1.0])
    elbow = np.array([0.3, 0.0, 0.8])
    wrist = np.array([0.5, 0.0, 0.6])
    hand = np.array([0.6, 0.0, 0.5])

    demo = make_3d_trajectory(
        start=shoulder,
        goal=hand,
        n_points=100,
        noise=0.0,
        seed=seed,
        curve_scale=0.2,
    )

    S = np.vstack(
        [
            _cross_points(shoulder),
            _cross_points(elbow),
            _cross_points(wrist),
            _cross_points(hand),
        ]
    )  # (12, 3)
    T = S.copy()

    scene = {
        "shoulder": shoulder,
        "elbow": elbow,
        "wrist": wrist,
        "hand": hand,
        "S": S,
        "T": T,
    }
    return demo, scene


def randomize_armpose_scene(
    base_scene: dict,
    seed: int = 0,
    pos_range: float = 0.08,
    angle_range_deg: float = 25.0,
) -> dict:
    """Return a copy of ``base_scene`` with randomly perturbed keypoints.

    Each keypoint is shifted by ±``pos_range`` uniformly (independent per
    coordinate), and the ``S`` / ``T`` crosses are recomputed accordingly.
    A global rotation (bounded by ``angle_range_deg``) is also applied to
    all keypoints around the shoulder.

    Parameters
    ----------
    base_scene : dict — output of :func:`make_armpose_demo`.
    seed : int — random seed.
    pos_range : float — max position perturbation in metres.
    angle_range_deg : float — max global rotation in degrees.

    Returns
    -------
    dict — same structure as ``base_scene``.
    """
    rng = np.random.default_rng(seed)

    def _small_rotation(rng: np.random.Generator, max_deg: float) -> np.ndarray:
        axis = rng.standard_normal(3)
        axis /= np.linalg.norm(axis) + 1e-10
        angle = rng.uniform(0.0, np.deg2rad(max_deg))
        K = np.array(
            [
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0],
            ]
        )
        return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K

    R = _small_rotation(rng, angle_range_deg)

    shoulder_base = base_scene["shoulder"].copy()
    new_scene = {}
    for key in ("shoulder", "elbow", "wrist", "hand"):
        pt = base_scene[key].copy()
        # Rotate around base shoulder, then perturb
        pt = shoulder_base + R @ (pt - shoulder_base)
        pt += rng.uniform(-pos_range, pos_range, size=3)
        new_scene[key] = pt

    S = np.vstack(
        [
            _cross_points(new_scene["shoulder"]),
            _cross_points(new_scene["elbow"]),
            _cross_points(new_scene["wrist"]),
            _cross_points(new_scene["hand"]),
        ]
    )
    new_scene["S"] = S
    new_scene["T"] = S.copy()
    return new_scene
