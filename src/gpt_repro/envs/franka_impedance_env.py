"""Franka Panda physics-based impedance-control environment — Phase 16.

Implements ``FrankaImpedanceEnv``: a MuJoCo physics simulation with
Cartesian impedance control (Sec. IV-D).

Unlike ``FrankaKinematicEnv`` (IK-based), this class:
- Runs full rigid-body dynamics via ``mujoco.mj_step``.
- Applies joint torques computed from a Cartesian impedance law.
- Includes gravity compensation (``qfrc_bias``).

Impedance law (Sec. IV-D):
    F   = K_s @ (x_des - x) + D @ (xdot_des - xdot)
    τ   = J^T @ F + τ_gravity
    τ   = clip(τ, -87, 87)   [Nm]

Critical-damping relationship: D = 2 * sqrt(K_s)  (diagonal case).

Observation space (20D):
    [ee_pos (3), ee_vel (3), q (7), dq (7)]

Action space (9D):
    [x_desired (3), xdot_desired (3), diag_K (3)]

Note: this implementation uses 20D to be consistent.
"""

from __future__ import annotations

from typing import Optional, Tuple

import mujoco
import numpy as np
import gymnasium

from gpt_repro.envs.franka_scene import (
    build_scene_xml,
    load_scene_model,
    CAMERAS,
)
from gpt_repro.utils.seeding import set_global_seed as set_all_seeds

# Default home configuration (7 arm joints + 2 gripper fingers)
Q_HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04])

# Torque limit (Nm) — Franka real hardware limit
TORQUE_LIMIT = 87.0

# Default physics timestep
_DEFAULT_DT = 0.002  # seconds per mj_step

# Number of physics steps per control step (default: 1)
_SUBSTEPS = 1


class FrankaImpedanceEnv(gymnasium.Env):
    """Franka Panda with Cartesian impedance control (physics simulation).

    Parameters
    ----------
    task : str
        One of "reshelving", "cleaning", "armpose".
    scene_kwargs : dict, optional
        Extra keyword arguments forwarded to ``build_scene_xml``.
    render_mode : str, optional
        "rgb_array" or None.
    width, height : int
        Renderer resolution.
    dt : float
        MuJoCo physics timestep (seconds).  Smaller → more stable.
    control_hz : float
        Control frequency (Hz).  Sets ``control_dt = 1/control_hz``.
    ee_site_name : str
        Name of the MuJoCo site for the EE.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        task: str = "reshelving",
        scene_kwargs: Optional[dict] = None,
        render_mode: Optional[str] = "rgb_array",
        width: int = 480,
        height: int = 480,
        dt: float = _DEFAULT_DT,
        control_hz: float = 500.0,
        ee_site_name: str = "attachment_site",
    ):
        super().__init__()
        self.task = task
        self.render_mode = render_mode
        self.width = width
        self.height = height
        self.dt = dt
        self.control_dt = 1.0 / control_hz
        self._ee_site_name = ee_site_name

        # Build scene and load model
        _scene_kw = scene_kwargs or {}
        xml = build_scene_xml(task, **_scene_kw)
        self._model, self._data = load_scene_model(xml)

        # Override MuJoCo timestep
        self._model.opt.timestep = dt

        # Assert we have at least 7 arm actuators
        assert self._model.nu >= 7, (
            f"Expected >= 7 actuators; got {self._model.nu}"
        )

        # Switch to torque control: disable PD actuators
        # Original: gainprm[:,0]=kp (position), biasprm[:,1]=-kp, biasprm[:,2]=-kd
        # Torque mode: gainprm[:,0]=1, biasprm[:,:]=0
        self._model.actuator_gainprm[:7, 0] = 1.0
        self._model.actuator_biasprm[:7, :] = 0.0

        # EE site id
        self._site_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_SITE, ee_site_name
        )
        if self._site_id < 0:
            raise ValueError(f"Site '{ee_site_name}' not found in model.")

        # Gymnasium spaces
        obs_dim = 20  # ee_pos(3) + ee_vel(3) + q(7) + dq(7)
        act_dim = 9   # x_des(3) + xdot_des(3) + diag_K(3)
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float64
        )
        self.action_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(act_dim,), dtype=np.float64
        )

        # Renderer (lazy init)
        self._renderer: Optional[mujoco.Renderer] = None
        if render_mode == "rgb_array":
            self._renderer = mujoco.Renderer(self._model, height=height, width=width)

        # Camera (default: "front")
        self._camera_name = "front"

        # Control step counter
        self._step_count = 0

        # Number of physics sub-steps per control step
        _substeps_float = self.control_dt / self.dt
        self._substeps = max(1, round(_substeps_float))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def model(self):
        return self._model

    @property
    def data(self):
        return self._data

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------

    def set_camera(self, name: str) -> None:
        """Set active render camera.  name ∈ {"front","side","top","quarter"}."""
        if name not in CAMERAS:
            raise ValueError(f"Unknown camera '{name}'; choose from {list(CAMERAS)}")
        self._camera_name = name

    def _make_camera(self) -> mujoco.MjvCamera:
        """Build an MjvCamera from the CAMERAS dict."""
        lookat, dist, elev, azim = CAMERAS[self._camera_name]
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = lookat
        cam.distance = dist
        cam.elevation = elev
        cam.azimuth = azim
        return cam

    # ------------------------------------------------------------------
    # Core physics helpers
    # ------------------------------------------------------------------

    def _jacobian(self) -> np.ndarray:
        """Compute 3×7 position Jacobian for the EE site.

        Uses mujoco.mj_jacSite (Sec. IV-D).
        Returns J of shape (3, 7).
        """
        jacp = np.zeros((3, self._model.nv))
        jacr = np.zeros((3, self._model.nv))
        mujoco.mj_jacSite(self._model, self._data, jacp, jacr, self._site_id)
        return jacp[:, :7]  # (3, 7) — arm joints only

    def _impedance_torques(
        self,
        x_des: np.ndarray,
        xdot_des: np.ndarray,
        K_s: np.ndarray,
        D: np.ndarray,
    ) -> np.ndarray:
        """Compute joint torques from Cartesian impedance law (Sec. IV-D).

        Eq. (Sec. IV-D):
            F   = K_s @ (x_des - x) + D @ (xdot_des - xdot)
            τ   = J^T @ F + qfrc_bias[:7]
            τ   = clip(τ, -87, 87)

        Parameters
        ----------
        x_des    : (3,) desired EE position.
        xdot_des : (3,) desired EE velocity.
        K_s      : (3,3) stiffness matrix.
        D        : (3,3) damping matrix.

        Returns
        -------
        tau : (7,) joint torques, clipped to ±TORQUE_LIMIT.
        """
        J = self._jacobian()  # (3, 7)

        # Current EE state
        x_cur = self._data.site_xpos[self._site_id].copy()  # (3,)
        xdot_cur = J @ self._data.qvel[:7]                  # (3,)

        # Cartesian force
        F = K_s @ (x_des - x_cur) + D @ (xdot_des - xdot_cur)  # (3,)

        # Gravity compensation (Coriolis + gravity in joint space)
        tau_grav = self._data.qfrc_bias[:7].copy()  # (7,)

        # Joint torques
        tau = J.T @ F + tau_grav  # (7,)
        return np.clip(tau, -TORQUE_LIMIT, TORQUE_LIMIT)

    def _get_obs(self) -> np.ndarray:
        """Return 20D observation: [ee_pos(3), ee_vel(3), q(7), dq(7)]."""
        J = self._jacobian()
        ee_pos = self._data.site_xpos[self._site_id].copy()
        ee_vel = J @ self._data.qvel[:7]
        q = self._data.qpos[:7].copy()
        dq = self._data.qvel[:7].copy()
        return np.concatenate([ee_pos, ee_vel, q, dq])

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
        if seed is not None:
            set_all_seeds(seed)

        # Set joints to home configuration
        self._data.qpos[:] = Q_HOME[:self._model.nq]
        self._data.qvel[:] = 0.0
        self._data.ctrl[:] = 0.0
        mujoco.mj_forward(self._model, self._data)

        # Settle under impedance control targeting home EE position
        x_home = self._data.site_xpos[self._site_id].copy()
        K_settle = np.diag([600.0, 600.0, 600.0])
        D_settle = 2.0 * np.sqrt(K_settle)

        for _ in range(200):
            tau = self._impedance_torques(x_home, np.zeros(3), K_settle, D_settle)
            self._data.ctrl[:7] = tau
            mujoco.mj_step(self._model, self._data)

        self._step_count = 0
        return self._get_obs(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Apply one impedance control step.

        Parameters
        ----------
        action : (9,) array — [x_des(3), xdot_des(3), diag_K(3)]

        Returns
        -------
        obs, reward, terminated, truncated, info
        """
        action = np.asarray(action, dtype=np.float64)
        x_des = action[:3]
        xdot_des = action[3:6]
        diag_k = np.abs(action[6:9])  # ensure positive

        K_s = np.diag(diag_k)
        D = 2.0 * np.sqrt(K_s)

        # Compute and apply torques for each physics substep
        for _ in range(self._substeps):
            tau = self._impedance_torques(x_des, xdot_des, K_s, D)
            self._data.ctrl[:7] = tau
            mujoco.mj_step(self._model, self._data)

        self._step_count += 1
        obs = self._get_obs()

        # Reward: negative EE tracking error
        ee_pos = obs[:3]
        reward = -float(np.linalg.norm(ee_pos - x_des))

        # Check for physics divergence
        if np.any(np.isnan(obs)):
            return obs, -1.0, True, False, {"nan": True}

        return obs, reward, False, False, {
            "ee_pos": ee_pos.copy(),
            "x_des": x_des.copy(),
            "tracking_error": -reward,
        }

    def render(self) -> Optional[np.ndarray]:
        """Render current state to RGB array."""
        if self._renderer is None:
            return None
        cam = self._make_camera()
        self._renderer.update_scene(self._data, camera=cam)
        return self._renderer.render()

    def get_ee_pos(self) -> np.ndarray:
        """Return current EE position (3,)."""
        mujoco.mj_forward(self._model, self._data)
        return self._data.site_xpos[self._site_id].copy()

    def get_workspace_bounds(self):
        """Return workspace (lo, hi) bounds — same as kinematic env."""
        lo = np.array([0.25, -0.40, 0.40])
        hi = np.array([0.75,  0.40, 0.95])
        return lo, hi

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
