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
    """Generate MuJoCo XML with keypoint spheres, arm links, and EE.

    Phase 12: added arm link capsules, proper lighting, fixed camera.
    """
    sh, el, wr, ha = shoulder, elbow, wrist, hand
    return f"""
<mujoco model="armpose">
  <option gravity="0 0 0" timestep="0.02"/>
  <visual>
    <headlight ambient="0.4 0.4 0.4" diffuse="0.8 0.8 0.8" specular="0.1 0.1 0.1"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <quality shadowsize="2048"/>
  </visual>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse="1 1 1" specular="0.3 0.3 0.3"/>
    <light pos="-1 -1 2" dir="1 1 -1" diffuse="0.5 0.5 0.5" specular="0 0 0"/>
    <camera name="fixed" mode="fixed" pos="0.8 -0.5 1.2" zaxis="-0.615 0.615 -0.492"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.72 0.72 0.72 1" pos="0 0 0"/>
    <!-- Shoulder (cyan) -->
    <geom name="shoulder_marker" type="sphere" size="0.040"
          pos="{sh[0]:.4f} {sh[1]:.4f} {sh[2]:.4f}" rgba="0.0 0.85 0.85 0.9"/>
    <!-- Elbow (magenta) -->
    <geom name="elbow_marker" type="sphere" size="0.030"
          pos="{el[0]:.4f} {el[1]:.4f} {el[2]:.4f}" rgba="0.85 0.0 0.85 0.9"/>
    <!-- Wrist (yellow) -->
    <geom name="wrist_marker" type="sphere" size="0.025"
          pos="{wr[0]:.4f} {wr[1]:.4f} {wr[2]:.4f}" rgba="0.95 0.90 0.0 0.9"/>
    <!-- Hand (blue goal) -->
    <geom name="hand_marker" type="sphere" size="0.020"
          pos="{ha[0]:.4f} {ha[1]:.4f} {ha[2]:.4f}" rgba="0.1 0.2 0.95 0.9"/>
    <!-- Arm link: shoulder→elbow -->
    <geom name="link_sh_el" type="capsule" size="0.010"
          fromto="{sh[0]:.4f} {sh[1]:.4f} {sh[2]:.4f} {el[0]:.4f} {el[1]:.4f} {el[2]:.4f}"
          rgba="0.65 0.65 0.65 0.8"/>
    <!-- Arm link: elbow→wrist -->
    <geom name="link_el_wr" type="capsule" size="0.010"
          fromto="{el[0]:.4f} {el[1]:.4f} {el[2]:.4f} {wr[0]:.4f} {wr[1]:.4f} {wr[2]:.4f}"
          rgba="0.65 0.65 0.65 0.8"/>
    <!-- Arm link: wrist→hand -->
    <geom name="link_wr_ha" type="capsule" size="0.008"
          fromto="{wr[0]:.4f} {wr[1]:.4f} {wr[2]:.4f} {ha[0]:.4f} {ha[1]:.4f} {ha[2]:.4f}"
          rgba="0.65 0.65 0.65 0.8"/>
    <!-- End-effector (red capsule tool) -->
    <body name="ee_body" pos="{sh[0]:.4f} {sh[1]:.4f} {sh[2]:.4f}">
      <joint name="ee_x" type="slide" axis="1 0 0" range="-2 2"/>
      <joint name="ee_y" type="slide" axis="0 1 0" range="-2 2"/>
      <joint name="ee_z" type="slide" axis="0 0 1" range="-2 2"/>
      <geom name="ee_geom" type="capsule" size="0.018 0.030"
            fromto="0 0 -0.030 0 0 0.030" rgba="0.9 0.15 0.15 1"/>
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

    import numpy as _np
    _CAM_LOOKAT    = _np.array([0.3, 0.0, 0.8])
    _CAM_DISTANCE  = 0.81
    _CAM_ELEVATION = 29.5
    _CAM_AZIMUTH   = -45.0

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

    def _rebuild_model(self, scene: dict) -> None:
        """Regenerate MuJoCo model from new scene keypoints.

        Must be called before ``reset()`` to update keypoint visuals.
        Recreates model, data, and renderer (renderer holds old model ref).
        """
        self._scene = scene
        self._shoulder = np.asarray(scene["shoulder"], dtype=float)
        self._elbow = np.asarray(scene["elbow"], dtype=float)
        self._wrist = np.asarray(scene["wrist"], dtype=float)
        self._hand = np.asarray(scene["hand"], dtype=float)
        self._xml_string = _build_armpose_xml(
            self._shoulder, self._elbow, self._wrist, self._hand
        )
        import mujoco as _mj
        self._model = _mj.MjModel.from_xml_string(self._xml_string)
        self._data = _mj.MjData(self._model)
        _mj.mj_forward(self._model, self._data)
        # Recreate renderer (holds reference to old model)
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
