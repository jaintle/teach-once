"""Task-specific EE waypoint generators for Franka demos — Phase 14.

Each function returns a (M, 3) numpy array of EE positions in world frame.
Positions are derived from scene dicts so they work for both base scenes and
transported scenes.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Bezier helper
# ---------------------------------------------------------------------------

def _cubic_bezier(p0, p1, p2, p3, n: int) -> np.ndarray:
    """Sample n points on a cubic Bezier curve."""
    t = np.linspace(0.0, 1.0, n)
    b = (
        np.outer((1 - t) ** 3, p0)
        + np.outer(3 * (1 - t) ** 2 * t, p1)
        + np.outer(3 * (1 - t) * t ** 2, p2)
        + np.outer(t ** 3, p3)
    )
    return b  # (n, 3)


# ---------------------------------------------------------------------------
# Reshelving waypoints
# ---------------------------------------------------------------------------

def get_reshelving_waypoints(
    scene: dict,
    n_waypoints: int = 8,
) -> np.ndarray:
    """EE waypoints for a pick-and-place reshelving task.

    Expected scene keys: "object_pose" (3,), "goal_pose" (3,).
    Falls back to default positions if keys are missing.

    Steps:
    1. Pre-grasp  : 0.1 m above object.
    2. Grasp      : at object position.
    3. Lift       : 0.15 m above object.
    4. Arc        : 3 Bezier-interpolated points toward goal.
    5. Pre-place  : 0.1 m above goal.
    6. Place      : at goal position.
    7. Retreat    : 0.1 m above goal.
    """
    obj  = np.asarray(scene.get("object_pose", [0.50, 0.00, 0.63]), dtype=float)
    goal = np.asarray(scene.get("goal_pose",   [0.30, 0.10, 0.75]), dtype=float)

    pre_grasp  = obj  + np.array([0.0, 0.0,  0.10])
    grasp      = obj.copy()
    lift       = obj  + np.array([0.0, 0.0,  0.15])
    pre_place  = goal + np.array([0.0, 0.0,  0.10])
    place      = goal.copy()
    retreat    = goal + np.array([0.0, 0.0,  0.10])

    # Bezier arc from lift to pre_place (3 intermediate points)
    mid_height = max(lift[2], pre_place[2]) + 0.05
    ctrl1 = lift      + np.array([0.0, 0.0,  0.08])
    ctrl2 = pre_place + np.array([0.0, 0.0,  0.08])
    arc = _cubic_bezier(lift, ctrl1, ctrl2, pre_place, n=3)  # (3, 3)

    waypoints = np.array([pre_grasp, grasp, lift])  # (3, 3)
    waypoints = np.vstack([waypoints, arc])          # (6, 3)
    waypoints = np.vstack([waypoints, [place, retreat]])  # (8, 3)
    return waypoints  # (8, 3)


# ---------------------------------------------------------------------------
# Cleaning waypoints
# ---------------------------------------------------------------------------

def get_cleaning_waypoints(
    scene: dict,
    surface_config=None,
    n_strokes: int = 4,
) -> np.ndarray:
    """EE waypoints for a raster-scan surface cleaning task.

    Expected scene keys: "surface_center" (3,), "surface_half_size" (2,)
    giving [half_x, half_y] extent of the surface.

    Each stroke: left edge → right edge at EE height = surface_z + 0.02 m.
    Between strokes: lift 0.05 m then move to next row start.
    Total waypoints ≈ n_strokes * 10.
    """
    center = np.asarray(scene.get("surface_center", [0.50, 0.00, 0.64]), dtype=float)
    half   = np.asarray(scene.get("surface_half_size", [0.12, 0.12]),    dtype=float)

    surface_z  = center[2]
    work_z     = surface_z + 0.02   # EE height during cleaning stroke
    lift_z     = surface_z + 0.07   # EE height during transition

    xs_per_stroke = 8               # samples along x-axis per stroke
    x_vals = np.linspace(center[0] - half[0], center[0] + half[0], xs_per_stroke)
    y_vals = np.linspace(center[1] - half[1], center[1] + half[1], n_strokes)

    waypoints = []
    # Approach first point from above
    waypoints.append([x_vals[0], y_vals[0], lift_z])

    for j, y in enumerate(y_vals):
        # Stroke left-to-right or right-to-left (boustrophedon)
        x_row = x_vals if j % 2 == 0 else x_vals[::-1]
        # Descend to work height
        waypoints.append([x_row[0], y, work_z])
        # Sweep across
        for x in x_row:
            waypoints.append([x, y, work_z])
        # Lift after stroke
        waypoints.append([x_row[-1], y, lift_z])
        # Move to start of next row (if not last)
        if j < n_strokes - 1:
            x_next_start = x_vals[0] if (j + 1) % 2 == 0 else x_vals[-1]
            waypoints.append([x_next_start, y_vals[j + 1], lift_z])

    return np.array(waypoints, dtype=float)


# ---------------------------------------------------------------------------
# Arm-pose waypoints
# ---------------------------------------------------------------------------

def get_armpose_waypoints(
    scene: dict,
    n_waypoints: int = 6,
) -> np.ndarray:
    """EE waypoints tracing a mannequin arm's keypoints.

    Expected scene keys: "shoulder" (3,), "elbow" (3,),
    "wrist" (3,), "hand" (3,).  Falls back to defaults.

    Steps:
    1. Approach shoulder from above (+0.10 m in z).
    2. Touch shoulder.
    3. Arc to elbow.
    4. Move to wrist.
    5. Move to hand.
    6. Retreat upward (+0.10 m from hand).
    """
    shoulder = np.asarray(scene.get("shoulder", [0.35, 0.00, 0.70]), dtype=float)
    elbow    = np.asarray(scene.get("elbow",    [0.47, 0.00, 0.80]), dtype=float)
    wrist    = np.asarray(scene.get("wrist",    [0.57, 0.00, 0.75]), dtype=float)
    hand     = np.asarray(scene.get("hand",     [0.62, 0.00, 0.65]), dtype=float)

    approach  = shoulder + np.array([0.0, 0.0, 0.10])
    retreat   = hand     + np.array([0.0, 0.0, 0.10])

    # Arc mid-point between shoulder and elbow (slightly higher)
    arc_mid   = 0.5 * (shoulder + elbow) + np.array([0.0, 0.0, 0.04])

    waypoints = np.array([
        approach,
        shoulder,
        arc_mid,
        elbow,
        wrist,
        hand,
        retreat,
    ], dtype=float)
    return waypoints  # (7, 3)
