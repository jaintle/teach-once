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
    """Generate MuJoCo XML with object (red) and goal (green) markers."""
    op = obj_pos
    gp = goal_pos
    return f"""
<mujoco model="reshelving">
  <option gravity="0 0 0" timestep="0.02"/>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.8 0.8 0.8 1" pos="0 0 0"/>
    <!-- Object marker (red cube) -->
    <geom name="object_marker" type="box" size="0.05 0.05 0.05"
          pos="{op[0]:.4f} {op[1]:.4f} {op[2]:.4f}" rgba="0.9 0.2 0.2 0.7"/>
    <!-- Goal marker (green sphere) -->
    <geom name="goal_marker" type="sphere" size="0.04"
          pos="{gp[0]:.4f} {gp[1]:.4f} {gp[2]:.4f}" rgba="0.2 0.8 0.2 0.7"/>
    <!-- End-effector -->
    <body name="ee_body" pos="{op[0]:.4f} {op[1]:.4f} {op[2]:.4f}">
      <joint name="ee_x" type="slide" axis="1 0 0" range="-2 2"/>
      <joint name="ee_y" type="slide" axis="0 1 0" range="-2 2"/>
      <joint name="ee_z" type="slide" axis="0 0 1" range="-2 2"/>
      <geom name="ee_geom" type="sphere" size="0.02" rgba="0.2 0.6 0.9 1"/>
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
