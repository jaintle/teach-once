"""Franka Panda kinematic environment — Phase 13.

Wraps a MuJoCo model built by :mod:`~gpt_repro.envs.franka_scene` in a
Gymnasium ``Env``.  IK-based EE control: ``step(target_pos)`` solves
Jacobian-pseudoinverse IK, sets joint positions, runs mj_forward, and
returns the new EE position as observation.

Rendering uses programmatic MjvCamera (CAMERA_FREE mode) — named cameras
defined in XML produce black frames on macOS headless (confirmed in Phase 12).
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import mujoco
import numpy as np
import gymnasium

from gpt_repro.envs.franka_scene import CAMERAS, build_scene_xml, load_scene_model
from gpt_repro.envs.ik_solver import IKSolver


# ---------------------------------------------------------------------------
# Home joint configuration (radians + finger widths)
# ---------------------------------------------------------------------------

Q_HOME: np.ndarray = np.array(
    [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04],
    dtype=np.float64,
)

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class FrankaKinematicEnv(gymnasium.Env):
    """Franka Panda arm environment with IK-based EE position control.

    Observation: EE position (3,) in world frame.
    Action:      target EE position (3,) in world frame.

    The environment is kinematic-only (no physics integration) — we set
    joint positions directly so trajectories from policy transport can be
    replayed without drift.

    Parameters
    ----------
    task : str — "reshelving", "cleaning", or "armpose".
    scene_kwargs : dict — passed to :func:`build_scene_xml`.
    render_mode : str or None — "rgb_array" or None.
    width, height : int — renderer resolution.
    dt : float — MuJoCo timestep (kept in XML for physical consistency).
    control_dt : float — seconds per ``step()`` call (unused kinematically
        but stored for callers that need it).
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        task: str = "reshelving",
        scene_kwargs: Optional[dict] = None,
        render_mode: Optional[str] = None,
        width: int = 720,
        height: int = 480,
        dt: float = 0.002,
        control_dt: float = 0.05,
    ) -> None:
        super().__init__()

        if task not in ("reshelving", "cleaning", "armpose"):
            raise ValueError(
                f"Unknown task {task!r}. Must be 'reshelving', 'cleaning', or 'armpose'."
            )

        self.task = task
        self._render_mode = render_mode
        self._width = width
        self._height = height
        self.dt = dt
        self.control_dt = control_dt

        # Build scene
        sk = scene_kwargs or {}
        xml = build_scene_xml(task, **sk)
        self._model, self._data = load_scene_model(xml)

        # EE site
        self._site_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site"
        )
        if self._site_id < 0:
            raise RuntimeError("attachment_site not found in model.")

        # IK solver
        self._ik = IKSolver(
            self._model,
            self._data,
            ee_site_name="attachment_site",
            max_iter=200,
            tol=1e-3,
            damping=1e-4,
            nullspace_gain=0.5,
        )

        # Home EE position
        self._data.qpos[:] = Q_HOME
        mujoco.mj_forward(self._model, self._data)
        self._ee_home = self._data.site_xpos[self._site_id].copy()

        # Gymnasium spaces
        obs_low = np.full(3, -2.0, dtype=np.float64)
        obs_high = np.full(3,  2.0, dtype=np.float64)
        self.observation_space = gymnasium.spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float64
        )
        # Action = target EE position
        self.action_space = gymnasium.spaces.Box(
            low=obs_low.copy(), high=obs_high.copy(), dtype=np.float64
        )

        # Renderer (lazy-created on first render call)
        self._renderer: Optional[mujoco.Renderer] = None

        # Active camera params: (lookat, distance, elevation, azimuth)
        self._cam_lookat, self._cam_distance, self._cam_elevation, self._cam_azimuth = (
            CAMERAS["front"]
        )

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        # Reset to home
        self._data.qpos[:] = Q_HOME
        self._data.qvel[:] = 0.0
        mujoco.mj_forward(self._model, self._data)

        obs = self._get_obs()
        return obs, {}

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Move EE to target_pos via IK.

        Parameters
        ----------
        action : (3,) target EE position.

        Returns
        -------
        obs, reward, terminated, truncated, info.
        """
        target_pos = np.asarray(action, dtype=np.float64).ravel()[:3]

        # IK solve (warm-start from current joint config)
        q_sol, success = self._ik.solve(target_pos, q_init=self._data.qpos[:7])

        if success:
            self._data.qpos[:7] = q_sol
        # Even if IK failed, we still update to best solution
        else:
            self._data.qpos[:7] = q_sol

        mujoco.mj_forward(self._model, self._data)

        obs = self._get_obs()
        ee_pos = obs
        dist = float(np.linalg.norm(ee_pos - target_pos))
        reward = -dist

        info = {"ik_success": success, "ee_dist_to_target": dist}
        return obs, reward, False, False, info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        return self._data.site_xpos[self._site_id].copy()

    def set_qpos(self, q: np.ndarray) -> None:
        """Directly set arm joint positions (7,) and run forward kinematics."""
        self._data.qpos[:7] = np.asarray(q, dtype=np.float64).ravel()[:7]
        mujoco.mj_forward(self._model, self._data)

    def set_ee_pos(self, target_pos: np.ndarray) -> bool:
        """Solve IK and set joints to reach target_pos.  Return IK success."""
        q_sol, success = self._ik.solve(
            np.asarray(target_pos, dtype=np.float64),
            q_init=self._data.qpos[:7],
        )
        self._data.qpos[:7] = q_sol
        mujoco.mj_forward(self._model, self._data)
        return success

    def get_ee_pos(self) -> np.ndarray:
        """Return current EE position (3,)."""
        return self._data.site_xpos[self._site_id].copy()

    def get_joint_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (q_lower, q_upper) for 7 arm joints."""
        return self._ik.get_joint_limits()

    def get_workspace_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (lo, hi) workspace bounds for EE position clamping.

        Conservative bounds that keep IK success rate high based on
        Phase 13 validation (100% success in [0.28-0.70]x[-0.35-0.35]x[0.45-0.90]).
        """
        lo = np.array([0.25, -0.40, 0.40], dtype=np.float64)
        hi = np.array([0.75,  0.40, 0.95], dtype=np.float64)
        return lo, hi

    @property
    def model(self):
        """Expose MjModel for direct manipulation (e.g. geom RGBA)."""
        return self._model

    @property
    def data(self):
        """Expose MjData for direct manipulation (e.g. qpos read/write)."""
        return self._data

    # ------------------------------------------------------------------
    # Camera control
    # ------------------------------------------------------------------

    def set_camera(self, name: str) -> None:
        """Switch to a named camera preset.

        Parameters
        ----------
        name : str — one of "front", "side", "top".
        """
        if name not in CAMERAS:
            raise ValueError(f"Unknown camera {name!r}. Choose from {list(CAMERAS)}")
        self._cam_lookat, self._cam_distance, self._cam_elevation, self._cam_azimuth = (
            CAMERAS[name]
        )

    def _make_camera(self) -> mujoco.MjvCamera:
        """Build a programmatic MjvCamera from stored params.

        Named cameras (CAMERA_FIXED mode) produce black frames in headless
        MuJoCo 3.x on macOS (confirmed Phase 12). CAMERA_FREE works.
        """
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat = np.array(self._cam_lookat, dtype=np.float64)
        cam.distance = float(self._cam_distance)
        cam.elevation = float(self._cam_elevation)
        cam.azimuth = float(self._cam_azimuth)
        return cam

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self) -> Optional[np.ndarray]:
        """Render current scene to an (H, W, 3) uint8 array."""
        if self._render_mode != "rgb_array":
            return None

        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self._model, self._height, self._width
            )

        cam = self._make_camera()
        self._renderer.update_scene(self._data, camera=cam)
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def __del__(self) -> None:
        if hasattr(self, "_renderer"):
            self.close()
