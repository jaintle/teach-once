"""Base kinematic end-effector environment using MuJoCo (Phase 9).

Implements a minimal ``gymnasium.Env`` wrapper around a MuJoCo model that
treats the end-effector as a kinematically controlled 3-D point.

Observation: end-effector position (3,).
Action:      end-effector velocity command (3,).
Dynamics:    pos += action * dt  (pure kinematics, no dynamics).

MuJoCo XML is embedded as a Python string constant so the environment is
fully self-contained (no external .xml files).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import mujoco
import numpy as np

import gymnasium
from gymnasium import spaces


# ---------------------------------------------------------------------------
# Minimal MuJoCo XML
# ---------------------------------------------------------------------------

_BASE_XML = """
<mujoco model="kinematic_ee">
  <option gravity="0 0 0" timestep="0.02"/>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.8 0.8 0.8 1"
          pos="0 0 0"/>
    <body name="ee_body" pos="0 0 0.5">
      <joint name="ee_x" type="slide" axis="1 0 0" range="-2 2"/>
      <joint name="ee_y" type="slide" axis="0 1 0" range="-2 2"/>
      <joint name="ee_z" type="slide" axis="0 0 1" range="-2 2"/>
      <geom name="ee_geom" type="sphere" size="0.02" rgba="0.2 0.6 0.9 1"/>
    </body>
  </worldbody>
</mujoco>
"""


class KinematicEndEffectorEnv(gymnasium.Env):
    """Minimal kinematic end-effector environment backed by MuJoCo.

    The end-effector position is controlled directly via velocity commands:
    ``pos_new = pos_old + action * dt``.

    Parameters
    ----------
    xml_string : str, optional
        MuJoCo XML model string. Defaults to ``_BASE_XML``.
    dt : float, optional
        Control time step. Defaults to 0.02 s.
    render_mode : str or None, optional
        ``"rgb_array"`` for off-screen rendering. Defaults to None.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        xml_string: str = _BASE_XML,
        dt: float = 0.02,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        self._xml_string = xml_string
        self._dt = dt
        self.render_mode = render_mode

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float64
        )

        self._model: Optional[mujoco.MjModel] = None
        self._data: Optional[mujoco.MjData] = None
        self._renderer: Optional[mujoco.Renderer] = None
        self._ee_pos: np.ndarray = np.zeros(3)
        self._build_model()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_model(self) -> None:
        self._model = mujoco.MjModel.from_xml_string(self._xml_string)
        self._data = mujoco.MjData(self._model)
        mujoco.mj_kinematics(self._model, self._data)

    def _get_joint_qpos_adr(self) -> Tuple[int, int, int]:
        """Return qpos address indices for the three slide joints."""
        x_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "ee_x")
        y_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "ee_y")
        z_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "ee_z")
        return (
            self._model.jnt_qposadr[x_id],
            self._model.jnt_qposadr[y_id],
            self._model.jnt_qposadr[z_id],
        )

    # ------------------------------------------------------------------
    # gymnasium.Env interface
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        mujoco.mj_resetData(self._model, self._data)
        init_pos = np.zeros(3)
        if options is not None and "init_pos" in options:
            init_pos = np.asarray(options["init_pos"], dtype=float)
        self.set_ee_pos(init_pos)
        obs = self.get_ee_pos()
        return obs.copy(), {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        new_pos = self.get_ee_pos() + action * self._dt
        self.set_ee_pos(new_pos)
        obs = self.get_ee_pos()
        reward = 0.0
        terminated = False
        truncated = False
        return obs.copy(), reward, terminated, truncated, {}

    def render(self) -> np.ndarray:
        """Render a 64×64 RGB frame, returning a black frame on failure."""
        try:
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self._model, height=64, width=64)
            self._renderer.update_scene(self._data)
            return self._renderer.render()
        except Exception:
            return np.zeros((64, 64, 3), dtype=np.uint8)

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ------------------------------------------------------------------
    # Position accessors
    # ------------------------------------------------------------------

    def get_ee_pos(self) -> np.ndarray:
        """Return current end-effector position as (3,) numpy array."""
        ax, ay, az = self._get_joint_qpos_adr()
        return np.array(
            [self._data.qpos[ax], self._data.qpos[ay], self._data.qpos[az]],
            dtype=float,
        )

    def set_ee_pos(self, pos: np.ndarray) -> None:
        """Set end-effector position directly (kinematics)."""
        pos = np.asarray(pos, dtype=float)
        ax, ay, az = self._get_joint_qpos_adr()
        self._data.qpos[ax] = pos[0]
        self._data.qpos[ay] = pos[1]
        self._data.qpos[az] = pos[2]
        mujoco.mj_kinematics(self._model, self._data)
