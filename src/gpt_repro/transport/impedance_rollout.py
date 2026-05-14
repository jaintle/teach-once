"""GPT impedance rollout — Phase 16.

Wires the GPT pipeline (PolicyTransport + transported stiffness/damping)
into the FrankaImpedanceEnv for physics-based validation.

Functions
---------
transport_and_rollout_impedance
    Full pipeline: transport demo → refit GP DS → run impedance rollout.
get_transported_stiffness
    Helper: transport a single K_s matrix through the transport map.
"""

from __future__ import annotations

from typing import Optional, Type

import numpy as np

from gpt_repro.gp.exact_gp import ExactGPRegressor
from gpt_repro.policies.ds_policy import GPDynamicalSystem
from gpt_repro.transport.policy_transport import PolicyTransport
from gpt_repro.envs.franka_impedance_env import FrankaImpedanceEnv


# ---------------------------------------------------------------------------
# Stiffness transport helper
# ---------------------------------------------------------------------------

def get_transported_stiffness(
    transport: PolicyTransport,
    x: np.ndarray,
    K_s_default: np.ndarray,
) -> np.ndarray:
    """Transport a constant stiffness matrix using the transport Jacobian.

    Applies K̂_s = J·K_s·J^T (Sec. IV-D) evaluated at the mean of x.

    Parameters
    ----------
    transport : PolicyTransport
        A fitted transport object.
    x : (N, d) or (d,) array
        Points at which to evaluate the Jacobian.  The mean is used.
    K_s_default : (d, d) array
        Stiffness in the source frame.

    Returns
    -------
    K_hat : (d, d) transported stiffness matrix, symmetric PSD.
    """
    x_arr = np.atleast_2d(x)
    # Use mean position as the representative point
    x_mean = x_arr.mean(axis=0, keepdims=True)  # (1, d)
    K_batch = K_s_default[None, :, :]            # (1, d, d)
    K_hat = transport.transform_stiffness(x_mean, K_batch)[0]  # (d, d)
    # Symmetrize for numerical safety
    K_hat = 0.5 * (K_hat + K_hat.T)
    return K_hat


# ---------------------------------------------------------------------------
# Main rollout
# ---------------------------------------------------------------------------

def transport_and_rollout_impedance(
    demo: dict,
    S: np.ndarray,
    T: np.ndarray,
    env: FrankaImpedanceEnv,
    gp_cls: Type = ExactGPRegressor,
    K_s_default: Optional[np.ndarray] = None,
    D_default: Optional[np.ndarray] = None,
    attractor_gain: float = 0.5,
    n_steps: int = 300,
    control_hz: float = 500.0,
    gp_n_iter: int = 100,
    seed: int = 0,
    success_threshold: float = 0.05,
    transport_stiffness: bool = True,
) -> dict:
    """Transport a kinematic demo and roll out with impedance control.

    Pipeline (Sec. IV — Eq. 7, 13, IV-D):
    1. Build PolicyTransport from S → T.
    2. Transport EE trajectory x and velocities xdot (Eq. 7, 13).
    3. Transport stiffness K_s (Sec. IV-D, K̂_s = J K_s J^T).
    4. Compute critical damping D = 2·sqrt(K̂_s).
    5. Refit GPDynamicalSystem on transported demo.
    6. Reset impedance env; roll out N steps.

    Parameters
    ----------
    demo : dict
        Keys: "x" (N,3), "xdot" (N,3).  From record_franka_demo.
    S : (M, 3) source reference points.
    T : (M, 3) target reference points.
    env : FrankaImpedanceEnv
        A constructed (not yet reset) impedance env.
    gp_cls : GP class
        Default ExactGPRegressor.
    K_s_default : (3,3) or None
        Source-frame stiffness.  If None, uses 200*I.
    D_default : (3,3) or None
        Source-frame damping.  If None, uses critical-damping rule.
    attractor_gain : float
        Gain added to velocity command toward goal.  Prevents stall.
    n_steps : int
        Number of control steps.
    control_hz : float
        Control frequency (used only for dt computation).
    gp_n_iter : int
        GP training iterations.
    seed : int
        Random seed.
    success_threshold : float
        Meters — used to compute "success" in returned dict.
    transport_stiffness : bool
        If True, transport K_s; otherwise use K_s_default directly.

    Returns
    -------
    dict with keys:
        "x_transported" : (N, 3) transported demo positions.
        "x_rollout"     : (n_steps, 3) impedance rollout positions.
        "x_des_traj"    : (n_steps, 3) desired EE positions.
        "final_error"   : float metres.
        "success"       : bool.
        "K_s_used"      : (3,3) stiffness actually applied.
        "D_used"        : (3,3) damping actually applied.
    """
    np.random.seed(seed)

    x = np.asarray(demo["x"], dtype=np.float64)    # (N, 3)
    xdot = np.asarray(demo["xdot"], dtype=np.float64)  # (N, 3)

    # ------------------------------------------------------------------
    # 1. Build transport map
    # ------------------------------------------------------------------
    transport = PolicyTransport(gp_cls=gp_cls, n_iter_default=gp_n_iter)
    transport.fit(S, T)

    # ------------------------------------------------------------------
    # 2. Transport trajectory and velocities (Eq. 7, 13)
    # ------------------------------------------------------------------
    x_t = transport.transform(x)          # (N, 3)
    xdot_t = transport.transform_velocity(x, xdot)  # (N, 3)

    # ------------------------------------------------------------------
    # 3. Transport stiffness (Sec. IV-D)
    # ------------------------------------------------------------------
    if K_s_default is None:
        K_s_default = np.diag([200.0, 200.0, 200.0])
    K_s_default = np.asarray(K_s_default, dtype=np.float64)

    if transport_stiffness:
        K_s = get_transported_stiffness(transport, x, K_s_default)
    else:
        K_s = K_s_default.copy()

    # Ensure PSD
    eigvals = np.linalg.eigvalsh(K_s)
    if np.any(eigvals < 0):
        K_s = K_s_default.copy()

    if D_default is None:
        D = 2.0 * np.diag(np.sqrt(np.diag(K_s)))
    else:
        D = np.asarray(D_default, dtype=np.float64)

    # ------------------------------------------------------------------
    # 4. Refit GP DS on transported demo
    # ------------------------------------------------------------------
    ds = GPDynamicalSystem(
        gp_cls=gp_cls,
        n_iter_default=gp_n_iter,
    )
    ds.fit(x_t, xdot_t)

    # ------------------------------------------------------------------
    # 5. Reset env and roll out with impedance control
    # ------------------------------------------------------------------
    obs, _ = env.reset(seed=seed)

    x_goal = x_t[-1].copy()
    diag_k = np.diag(K_s)

    rollout_pos = []
    rollout_des = []

    dt = 1.0 / control_hz
    x_cur = obs[:3].copy()

    for step_i in range(n_steps):
        # GP DS velocity prediction
        vel = ds.predict(x_cur[None, :], return_std=False).squeeze()  # (3,)

        # Attractor toward goal
        vel = vel + attractor_gain * (x_goal - x_cur)

        # Desired next position (Euler integration)
        x_des = x_cur + vel * dt

        # Build action: [x_des, xdot_des, diag_K]
        action = np.concatenate([x_des, vel, diag_k])
        obs, _, terminated, _, info = env.step(action)

        if terminated:
            break

        x_cur = obs[:3].copy()
        rollout_pos.append(x_cur.copy())
        rollout_des.append(x_des.copy())

    rollout_pos = np.array(rollout_pos, dtype=np.float64)   # (n_steps, 3)
    rollout_des = np.array(rollout_des, dtype=np.float64)

    final_error = float(np.linalg.norm(rollout_pos[-1] - x_goal)) if len(rollout_pos) > 0 else np.inf

    return {
        "x_transported": x_t,
        "x_rollout": rollout_pos,
        "x_des_traj": rollout_des,
        "final_error": final_error,
        "success": final_error < success_threshold,
        "K_s_used": K_s,
        "D_used": D,
        "x_goal": x_goal,
    }
