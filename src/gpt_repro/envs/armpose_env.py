"""Arm-pose environment — Phase 9 (3-D kinematic MuJoCo).

Extends :class:`KinematicEndEffectorEnv` with:
* Visual markers (spheres) for shoulder, elbow, wrist, and hand keypoints.
* Success detection: end-effector within 0.03 m of the hand keypoint.
* ``get_scene_points()`` returning the (12, 3) cross points (``S``).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from gpt_repro.envs.base_env import KinematicEndEffectorEnv


def _build_armpose_xml(
    shoulder: np.ndarray,
    elbow: np.ndarray,
    wrist: np.ndarray,
    hand: np.ndarray,
) -> str:
    """Generate MuJoCo XML with keypoint spheres and the EE body."""
    sh, el, wr, ha = shoulder, elbow, wrist, hand
    return f"""
<mujoco model="armpose">
  <option gravity="0 0 0" timestep="0.02"/>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.8 0.8 0.8 1" pos="0 0 0"/>
    <!-- Shoulder (blue) -->
    <geom name="shoulder_marker" type="sphere" size="0.04"
          pos="{sh[0]:.4f} {sh[1]:.4f} {sh[2]:.4f}" rgba="0.2 0.2 0.9 0.7"/>
    <!-- Elbow (yellow) -->
    <geom name="elbow_marker" type="sphere" size="0.035"
          pos="{el[0]:.4f} {el[1]:.4f} {el[2]:.4f}" rgba="0.9 0.8 0.1 0.7"/>
    <!-- Wrist (orange) -->
    <geom name="wrist_marker" type="sphere" size="0.03"
          pos="{wr[0]:.4f} {wr[1]:.4f} {wr[2]:.4f}" rgba="0.9 0.5 0.1 0.7"/>
    <!-- Hand / goal (green) -->
    <geom name="hand_marker" type="sphere" size="0.03"
          pos="{ha[0]:.4f} {ha[1]:.4f} {ha[2]:.4f}" rgba="0.2 0.8 0.2 0.7"/>
    <!-- End-effector -->
    <body name="ee_body" pos="{sh[0]:.4f} {sh[1]:.4f} {sh[2]:.4f}">
      <joint name="ee_x" type="slide" axis="1 0 0" range="-2 2"/>
      <joint name="ee_y" type="slide" axis="0 1 0" range="-2 2"/>
      <joint name="ee_z" type="slide" axis="0 0 1" range="-2 2"/>
      <geom name="ee_geom" type="sphere" size="0.02" rgba="0.2 0.6 0.9 1"/>
    </body>
  </worldbody>
</mujoco>
"""


def _cross_points(center: np.ndarray, arm: float = 0.04) -> np.ndarray:
    return center + np.array([[arm, 0, 0], [0, arm, 0], [0, 0, arm]], dtype=float)


class ArmPoseEnv(KinematicEndEffectorEnv):
    """3-D kinematic arm-pose following environment.

    The agent must move the end-effector from the shoulder toward the hand
    along a kinematic arm chain, following transported demonstrations.

    Parameters
    ----------
    scene : dict
        Scene dictionary produced by :func:`~gpt_repro.policies.demos_3d.make_armpose_demo`
        or :func:`~gpt_repro.policies.demos_3d.randomize_armpose_scene`.
        Must contain keys ``"shoulder"``, ``"elbow"``, ``"wrist"``, ``"hand"``,
        ``"S"`` (12×3).
    success_thresh : float, optional
        Distance threshold for ``is_success``. Defaults to 0.03 m.
    render_mode : str or None, optional
        See :class:`KinematicEndEffectorEnv`.
    """

    def __init__(
        self,
        scene: Optional[dict] = None,
        success_thresh: float = 0.03,
        render_mode: Optional[str] = None,
    ) -> None:
        if scene is None:
            from gpt_repro.policies.demos_3d import make_armpose_demo
            _, scene = make_armpose_demo(seed=0)

        self._scene = scene
        self._shoulder: np.ndarray = np.asarray(scene["shoulder"], dtype=float)
        self._elbow: np.ndarray = np.asarray(scene["elbow"], dtype=float)
        self._wrist: np.ndarray = np.asarray(scene["wrist"], dtype=float)
        self._hand: np.ndarray = np.asarray(scene["hand"], dtype=float)
        self._success_thresh = success_thresh

        xml = _build_armpose_xml(
            self._shoulder, self._elbow, self._wrist, self._hand
        )
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
            options["init_pos"] = self._shoulder.copy()
        return super().reset(seed=seed, options=options)

    def is_success(self) -> bool:
        """Return True if the EE is within ``success_thresh`` of the hand."""
        return bool(
            np.linalg.norm(self.get_ee_pos() - self._hand) < self._success_thresh
        )

    def get_scene_points(self) -> np.ndarray:
        """Return the (12, 3) axis-cross points around the 4 arm keypoints."""
        return self._scene["S"].copy()
