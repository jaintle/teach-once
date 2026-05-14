"""Jacobian pseudoinverse IK solver for the Franka Panda arm — Phase 13.

Algorithm: damped least-squares (Levenberg-Marquardt) with nullspace projection
for joint centering, standard in redundant manipulator control.

References
----------
- Nakamura & Hanafusa (1986) "Inverse kinematic solutions with singularity
  robustness for robot manipulator control."
- Buss (2004) "Introduction to inverse kinematics with Jacobian transpose,
  pseudoinverse and damped least squares methods."
"""

from __future__ import annotations

from typing import Optional, Tuple

import mujoco
import numpy as np


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------

def interpolate_joint_trajectory(
    q_start: np.ndarray,
    q_end: np.ndarray,
    n_interp: int = 10,
) -> np.ndarray:
    """Linear interpolation in joint space.

    Used by Phase 14 for smooth animation between IK solutions.

    Parameters
    ----------
    q_start, q_end : (7,) joint angles.
    n_interp : number of waypoints including start and end.

    Returns
    -------
    (n_interp, 7) array.
    """
    t = np.linspace(0.0, 1.0, n_interp)
    return np.outer(1 - t, q_start) + np.outer(t, q_end)


# ---------------------------------------------------------------------------
# IK Solver
# ---------------------------------------------------------------------------

class IKSolver:
    """Jacobian pseudoinverse IK with nullspace projection.

    Solves position-only or position+orientation IK for a redundant
    manipulator (Franka Panda, 7 arm DOF).

    Algorithm — Eq. "Jacobian pseudoinverse IK with nullspace projection,
    standard robotics" (Nakamura & Hanafusa 1986; Buss 2004):

        q ← q_init
        for iter in range(max_iter):
            fwd_kinematics(q)
            err = target - ee_pos          # (3,) position-only
            if ‖err‖ < tol: break
            J = jacobian[:3, :7]           # (3, 7) position Jacobian
            JJᵀ = J Jᵀ + λ²I              # damped (3×3)
            J⁺ = Jᵀ (JJᵀ)⁻¹              # (7, 3)
            Δq_null = k_null (q_mid - q)  # joint centering
            N = I - J⁺ J                  # nullspace projector (7×7)
            Δq = J⁺ err + N Δq_null
            q ← clip(q + Δq, q_lo, q_hi)
        return q, ‖err‖ < tol

    Parameters
    ----------
    model : mujoco.MjModel
    data  : mujoco.MjData
    ee_site_name : str — name of the EE site in the XML.
    max_iter : int — max IK iterations.
    tol : float — convergence threshold (metres).
    damping : float — Tikhonov damping λ² (prevents singularity blow-up).
    nullspace_gain : float — gain k_null for joint centering.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        ee_site_name: str = "attachment_site",
        max_iter: int = 200,
        tol: float = 1e-3,
        damping: float = 1e-4,
        nullspace_gain: float = 0.5,
    ) -> None:
        self._model = model
        self._data = data
        self._max_iter = max_iter
        self._tol = tol
        self._damping = damping
        self._nullspace_gain = nullspace_gain

        self._site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, ee_site_name
        )
        if self._site_id < 0:
            raise ValueError(
                f"Site '{ee_site_name}' not found in MuJoCo model. "
                f"Available sites: {[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i) for i in range(model.nsite)]}"
            )

        self._q_lower, self._q_upper = self.get_joint_limits()
        self._q_mid = 0.5 * (self._q_lower + self._q_upper)
        self._nv = model.nv          # total velocity DOF
        # Identify the 7 arm joint velocity DOF indices (skip fingers)
        # Franka joints: joint1-joint7 are DOF 0-6, fingers are 7-8
        self._arm_dof_ids = np.arange(7)

    # ------------------------------------------------------------------
    # Joint limits
    # ------------------------------------------------------------------

    def get_joint_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (q_lower, q_upper) for the 7 arm joints."""
        # First 7 joints are the arm; last 2 are fingers
        lower = self._model.jnt_range[:7, 0].copy()
        upper = self._model.jnt_range[:7, 1].copy()
        return lower, upper

    # ------------------------------------------------------------------
    # Jacobian computation
    # ------------------------------------------------------------------

    def compute_jacobian(self, full: bool = False) -> np.ndarray:
        """Compute position (and optionally rotation) Jacobian for the EE site.

        Uses ``mujoco.mj_jacSite`` to get the (3, nv) position Jacobian
        and (3, nv) rotation Jacobian. Extracts the 7 arm DOF columns.

        Parameters
        ----------
        full : bool — if True, return (6, 7) position+rotation Jacobian.

        Returns
        -------
        (3, 7) or (6, 7) numpy array.
        """
        jacp = np.zeros((3, self._nv))
        jacr = np.zeros((3, self._nv))
        mujoco.mj_jacSite(
            self._model, self._data,
            jacp, jacr,
            self._site_id,
        )
        # Extract arm DOF columns
        J_pos = jacp[:, self._arm_dof_ids]   # (3, 7)
        if full:
            J_rot = jacr[:, self._arm_dof_ids]  # (3, 7)
            return np.vstack([J_pos, J_rot])    # (6, 7)
        return J_pos                             # (3, 7)

    # ------------------------------------------------------------------
    # Orientation error helper
    # ------------------------------------------------------------------

    def quat_error(
        self,
        target_quat: np.ndarray,
        current_mat: np.ndarray,
    ) -> np.ndarray:
        """Compute 3D rotation error vector (axis-angle form).

        Parameters
        ----------
        target_quat : (4,) [w, x, y, z].
        current_mat : (9,) row-major rotation matrix (from site_xmat).

        Returns
        -------
        (3,) rotation error vector.
        """
        q_target = np.asarray(target_quat, dtype=float)
        # Convert current rotation matrix to quaternion
        q_current = np.zeros(4)
        mujoco.mju_mat2Quat(q_current, current_mat)
        # Quaternion difference: q_err = q_target * q_current^{-1}
        q_err = np.zeros(4)
        mujoco.mju_subQuat(q_err, q_target, q_current)
        # Convert to axis-angle (rotation vector)
        # q_err = [cos(θ/2), sin(θ/2)*axis]
        # For small errors, 2 * q_err[1:] ≈ θ*axis
        angle_axis = 2.0 * q_err[1:]   # (3,) approximate axis-angle
        return angle_axis

    # ------------------------------------------------------------------
    # EE position
    # ------------------------------------------------------------------

    def get_ee_pos(self) -> np.ndarray:
        """Return current EE position (3,) from data.site_xpos."""
        return self._data.site_xpos[self._site_id].copy()

    # ------------------------------------------------------------------
    # Main IK solve
    # ------------------------------------------------------------------

    def solve(
        self,
        target_pos: np.ndarray,
        target_quat: Optional[np.ndarray] = None,
        q_init: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, bool]:
        """Solve IK for a target EE position (and optionally orientation).

        Algorithm: Jacobian pseudoinverse IK with nullspace projection
        (Nakamura & Hanafusa 1986; damped LS from Buss 2004).

        Parameters
        ----------
        target_pos : (3,) desired EE position.
        target_quat : (4,) [w,x,y,z] desired orientation, or None.
        q_init : (7,) initial joint angles, or None → uses current data.qpos[:7].

        Returns
        -------
        (q_solution, success) where q_solution is (7,) and success is bool.
        """
        target_pos = np.asarray(target_pos, dtype=float)
        full = target_quat is not None

        # Initialise joint angles
        if q_init is not None:
            q = np.asarray(q_init, dtype=float).copy()
        else:
            q = self._data.qpos[:7].copy()

        err_norm = float("inf")

        for _ in range(self._max_iter):
            # Set arm joints and run forward kinematics
            # Kinematic only — we do NOT integrate physics (mj_step),
            # only update positions/velocities for Jacobian computation.
            self._data.qpos[:7] = q
            mujoco.mj_fwdPosition(self._model, self._data)

            # Compute error
            ee_pos = self._data.site_xpos[self._site_id].copy()
            err_pos = target_pos - ee_pos   # (3,)

            if full:
                err_rot = self.quat_error(
                    target_quat,
                    self._data.site_xmat[self._site_id],
                )
                err = np.concatenate([err_pos, err_rot])   # (6,)
            else:
                err = err_pos                               # (3,)

            err_norm = float(np.linalg.norm(err))
            if err_norm < self._tol:
                break

            # Jacobian
            J = self.compute_jacobian(full=full)   # (3or6, 7)

            # Damped pseudoinverse: J⁺ = Jᵀ (J Jᵀ + λ²I)⁻¹
            n_err = J.shape[0]
            JJT = J @ J.T + self._damping * np.eye(n_err)
            J_pinv = J.T @ np.linalg.inv(JJT)   # (7, 3or6)

            # Nullspace: pull joints toward midrange
            dq_null = self._nullspace_gain * (self._q_mid - q)   # (7,)
            N = np.eye(7) - J_pinv @ J                            # (7, 7)

            # Joint update
            dq = J_pinv @ err + N @ dq_null
            q = q + dq

            # Enforce joint limits
            margin = 5e-3
            q = np.clip(q, self._q_lower + margin, self._q_upper - margin)

        return q, err_norm < self._tol
