"""Reshelving environment — Phase 9 (3-D kinematic MuJoCo).

Extends :class:`KinematicEndEffectorEnv` with:
* A visual marker geom for the object and the goal position.
* Success detection: end-effector within 0.02 m of the goal.
* ``get_scene_points()`` returning the 8 bounding-box corner points (``S``).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import mujoco
import numpy as np

from gpt_repro.envs.base_env import KinematicEndEffectorEnv


def _build_reshelving_xml(
    obj_pos: np.ndarray,
    goal_pos: np.ndarray,
) -> str:
    """Generate MuJoCo XML with shelf structure, object, goal marker, EE.

    Phase 12: replaced minimal markers with a proper shelf + object visual.
    """
    op = obj_pos
    gp = goal_pos
    # Shelf planks: two horizontal boards around goal height
    shelf_cx, shelf_cy = gp[0], gp[1]
    shelf_z0 = max(gp[2] - 0.15, 0.1)   # lower plank
    shelf_z1 = gp[2] + 0.12              # upper plank
    return f"""
<mujoco model="reshelving">
  <option gravity="0 0 0" timestep="0.02"/>
  <visual>
    <headlight ambient="0.4 0.4 0.4" diffuse="0.8 0.8 0.8" specular="0.1 0.1 0.1"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <quality shadowsize="2048"/>
  </visual>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse="1 1 1" specular="0.3 0.3 0.3"/>
    <light pos="-1 -1 2" dir="1 1 -1" diffuse="0.5 0.5 0.5" specular="0 0 0"/>
    <camera name="fixed" mode="fixed" pos="0.8 -0.6 1.0" zaxis="-0.557 0.743 -0.371"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.72 0.72 0.72 1" pos="0 0 0"/>
    <!-- Shelf planks (grey) -->
    <geom name="shelf_lower" type="box" size="0.30 0.10 0.008"
          pos="{shelf_cx:.4f} {shelf_cy:.4f} {shelf_z0:.4f}" rgba="0.55 0.45 0.35 1"/>
    <geom name="shelf_upper" type="box" size="0.30 0.10 0.008"
          pos="{shelf_cx:.4f} {shelf_cy:.4f} {shelf_z1:.4f}" rgba="0.55 0.45 0.35 1"/>
    <!-- Shelf side supports -->
    <geom name="shelf_left" type="box" size="0.008 0.10 0.15"
          pos="{shelf_cx-0.30:.4f} {shelf_cy:.4f} {(shelf_z0+shelf_z1)/2:.4f}" rgba="0.45 0.35 0.25 1"/>
    <geom name="shelf_right" type="box" size="0.008 0.10 0.15"
          pos="{shelf_cx+0.30:.4f} {shelf_cy:.4f} {(shelf_z0+shelf_z1)/2:.4f}" rgba="0.45 0.35 0.25 1"/>
    <!-- Object (orange box) -->
    <geom name="object_marker" type="box" size="0.05 0.05 0.05"
          pos="{op[0]:.4f} {op[1]:.4f} {op[2]:.4f}" rgba="0.95 0.50 0.10 0.9"/>
    <!-- AprilTag proxy (dark square on object face) -->
    <geom name="apriltag_proxy" type="box" size="0.015 0.015 0.002"
          pos="{op[0]:.4f} {op[1]-0.051:.4f} {op[2]:.4f}" rgba="0.15 0.15 0.15 1"/>
    <!-- Goal slot (semi-transparent green) -->
    <geom name="goal_marker" type="box" size="0.055 0.055 0.055"
          pos="{gp[0]:.4f} {gp[1]:.4f} {gp[2]:.4f}" rgba="0.2 0.85 0.2 0.35"/>
    <!-- End-effector -->
    <body name="ee_body" pos="{op[0]:.4f} {op[1]:.4f} {op[2]:.4f}">
      <joint name="ee_x" type="slide" axis="1 0 0" range="-2 2"/>
      <joint name="ee_y" type="slide" axis="0 1 0" range="-2 2"/>
      <joint name="ee_z" type="slide" axis="0 0 1" range="-2 2"/>
      <geom name="ee_geom" type="capsule" size="0.018 0.030"
            fromto="0 0 -0.030 0 0 0.030" rgba="0.2 0.6 0.9 1"/>
    </body>
  </worldbody>
</mujoco>
"""


def _box_corners(center: np.ndarray, half_size: float = 0.05) -> np.ndarray:
    offsets = np.array(
        [
            [-1, -1, -1], [-1, -1, +1],
            [-1, +1, -1], [-1, +1, +1],
            [+1, -1, -1], [+1, -1, +1],
            [+1, +1, -1], [+1, +1, +1],
        ],
        dtype=float,
    )
    return center + offsets * half_size


class ReshelvingEnv(KinematicEndEffectorEnv):
    """3-D kinematic reshelving environment.

    The agent must move the end-effector from an object position to a
    goal position. Designed as a kinematic analog of the pick-and-place
    task described in Sec. V of the paper.

    Parameters
    ----------
    scene : dict
        Scene dictionary produced by :func:`~gpt_repro.policies.demos_3d.make_reshelving_demo`
        or :func:`~gpt_repro.policies.demos_3d.randomize_reshelving_scene`.
        Must contain keys ``"object_pose"`` (4×4), ``"goal_pose"`` (4×4),
        ``"S"`` (8×3).
    success_thresh : float, optional
        Distance threshold for ``is_success``. Defaults to 0.02 m.
    render_mode : str or None, optional
        See :class:`KinematicEndEffectorEnv`.
    """

    import numpy as _np
    _CAM_LOOKAT    = _np.array([0.15, 0.2, 0.6])
    _CAM_DISTANCE  = 1.3
    _CAM_ELEVATION = 25.0
    _CAM_AZIMUTH   = -40.0

    def __init__(
        self,
        scene: Optional[dict] = None,
        success_thresh: float = 0.02,
        render_mode: Optional[str] = None,
    ) -> None:
        if scene is None:
            from gpt_repro.policies.demos_3d import make_reshelving_demo
            _, scene = make_reshelving_demo(seed=0)

        self._scene = scene
        self._obj_pos: np.ndarray = scene["object_pose"][:3, 3].copy()
        self._goal_pos: np.ndarray = scene["goal_pose"][:3, 3].copy()
        self._success_thresh = success_thresh

        xml = _build_reshelving_xml(self._obj_pos, self._goal_pos)
        super().__init__(xml_string=xml, render_mode=render_mode)

    def _rebuild_model(self, scene: dict) -> None:
        """Regenerate MuJoCo model from new scene.

        Recreates model, data, and renderer so new visuals take effect.
        """
        self._scene = scene
        self._obj_pos = scene["object_pose"][:3, 3].copy()
        self._goal_pos = scene["goal_pose"][:3, 3].copy()
        self._xml_string = _build_reshelving_xml(self._obj_pos, self._goal_pos)
        import mujoco as _mj
        self._model = _mj.MjModel.from_xml_string(self._xml_string)
        self._data = _mj.MjData(self._model)
        _mj.mj_forward(self._model, self._data)
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, dict]:
        if options is None:
            options = {}
        if "init_pos" not in options:
            options["init_pos"] = self._obj_pos.copy()
        return super().reset(seed=seed, options=options)

    def is_success(self) -> bool:
        """Return True if the EE is within ``success_thresh`` of the goal."""
        return bool(
            np.linalg.norm(self.get_ee_pos() - self._goal_pos) < self._success_thresh
        )

    def get_scene_points(self) -> np.ndarray:
        """Return the (8, 3) object bounding-box corner points (source frame ``S``)."""
        return self._scene["S"].copy()
