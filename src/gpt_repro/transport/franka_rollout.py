"""GPT-Franka rollout adapter — Phase 14.

Wires the GPT pipeline (PolicyTransport + GPDynamicalSystem) into the
Franka kinematic environment.  Three functions:

1. ``record_franka_demo``   — record a waypoint-following kinematic demo.
2. ``transport_and_rollout_franka`` — full Sec. IV-C pipeline for Franka.
3. ``evaluate_franka_generalization`` — repeated evaluation over N scenes.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Type

import mujoco
import numpy as np
from scipy.ndimage import gaussian_filter1d

from gpt_repro.gp.exact_gp import ExactGPRegressor
from gpt_repro.policies.ds_policy import GPDynamicalSystem
from gpt_repro.transport.policy_transport import PolicyTransport
from gpt_repro.envs.franka_env import FrankaKinematicEnv
from gpt_repro.envs.ik_solver import interpolate_joint_trajectory


# ---------------------------------------------------------------------------
# Demo recording
# ---------------------------------------------------------------------------

def record_franka_demo(
    env: FrankaKinematicEnv,
    waypoints: np.ndarray,
    n_interp: int = 5,
) -> dict:
    """Record a kinematic demonstration on the Franka arm by replaying
    EE waypoints through IK.

    For each consecutive pair of waypoints, the function first attempts to
    interpolate ``n_interp`` intermediate IK solutions in joint space (via
    :func:`~gpt_repro.envs.ik_solver.interpolate_joint_trajectory`).
    If any intermediate IK solve fails, it falls back to a direct IK solve
    at the endpoint only.

    Parameters
    ----------
    env : FrankaKinematicEnv
        A reset environment.
    waypoints : (M, 3) array
        EE positions defining the demo path.
    n_interp : int, optional
        Number of interpolated steps between consecutive waypoints.

    Returns
    -------
    dict with keys:
        "x"          : (N, 3)  EE positions.
        "xdot"       : (N, 3)  finite-difference EE velocities.
        "q"          : (N, 7)  joint angles.
        "ik_success" : (N,)    bool success per step.
        "t"          : (N,)    timestamps.
    """
    waypoints = np.asarray(waypoints, dtype=np.float64)
    M = len(waypoints)

    xs: List[np.ndarray] = []
    qs: List[np.ndarray] = []
    ik_successes: List[bool] = []
    fallback_count = 0

    env.reset()

    for seg in range(M - 1):
        wp_start = waypoints[seg]
        wp_end   = waypoints[seg + 1]

        # Solve IK at both endpoints
        q_start, ok_start = env._ik.solve(wp_start, q_init=env._data.qpos[:7])
        q_end,   ok_end   = env._ik.solve(wp_end,   q_init=q_start)

        if ok_start and ok_end:
            # Interpolate in joint space
            q_seg = interpolate_joint_trajectory(q_start, q_end, n_interp=n_interp)
        else:
            # Fallback: single step directly to endpoint
            q_seg = q_end[np.newaxis, :]
            fallback_count += 1

        for q in q_seg:
            env.set_qpos(q)
            xs.append(env.get_ee_pos())
            qs.append(env._data.qpos[:7].copy())
            # Success = endpoint IK was ok for this segment
            ik_successes.append(bool(ok_end))

    # If only one waypoint provided, record it
    if M == 1:
        xs.append(env.get_ee_pos())
        qs.append(env._data.qpos[:7].copy())
        ik_successes.append(True)

    x_arr = np.array(xs, dtype=np.float64)       # (N, 3)
    q_arr = np.array(qs, dtype=np.float64)        # (N, 7)
    ok_arr = np.array(ik_successes, dtype=bool)   # (N,)

    # Finite-difference velocities (pad last with second-to-last)
    if len(x_arr) > 1:
        dx = np.diff(x_arr, axis=0)
        xdot = np.vstack([dx, dx[-1:]])           # (N, 3)
    else:
        xdot = np.zeros_like(x_arr)

    t_arr = np.arange(len(x_arr), dtype=np.float64) * env.control_dt

    if fallback_count:
        print(f"  [record_franka_demo] {fallback_count}/{M-1} segments used direct fallback.")

    return {"x": x_arr, "xdot": xdot, "q": q_arr, "ik_success": ok_arr, "t": t_arr}


# ---------------------------------------------------------------------------
# Full transport → rollout pipeline
# ---------------------------------------------------------------------------

def transport_and_rollout_franka(
    demo: dict,
    S: np.ndarray,
    T: np.ndarray,
    env: FrankaKinematicEnv,
    gp_cls: Type = ExactGPRegressor,
    n_interp: int = 5,
    dt: float = 0.05,
    n_steps: int = 120,
    gp_n_iter: int = 100,
    seed: int = 0,
) -> dict:
    """Transport a Franka demo to a new scene and roll it out with GPT.

    Pipeline (Sec. IV-C: refit f̂ on transported labels):
    1. Fit PolicyTransport(S → T) — Eq. (7) ϕ = γ + ψ∘γ.
    2. Transport demo positions and velocities (Eq. 13).
    3. Refit GPDynamicalSystem on transported (x̂, ẋ̂).
    4. Euler rollout: at each step predict velocity, compute next EE target
       x_next = x + v*dt, clamp to workspace bounds BEFORE IK, call env.step.
       Collect frames (rendered) and raw joint angles.
    5. Joint smoothing (Gaussian, sigma=1.5) applied along time axis — render
       time only; success/error metrics use unsmoothed positions.
    6. Re-render with smoothed joints.

    Parameters
    ----------
    demo : dict — from :func:`record_franka_demo`.
    S : (M, 3) source frame landmarks.
    T : (M, 3) target frame landmarks.
    env : FrankaKinematicEnv — will be reset to first transported waypoint.
    gp_cls : GP regressor class.
    n_interp : int — joint-space interp steps per rollout step.
    dt : float — Euler integration timestep.
    n_steps : int — rollout steps.
    gp_n_iter : int — GP training iterations.
    seed : int — random seed.

    Returns
    -------
    dict:
        "rollout_x"     : (N, 3)              EE positions (unsmoothed).
        "rollout_q"     : (N, 7)              joint angles (smoothed).
        "frames"        : list[np.ndarray]    rendered frames (480×720×3).
        "success"       : bool
        "final_error"   : float               distance to transported goal.
        "ik_fail_rate"  : float               fraction of IK failures.
        "transport"     : PolicyTransport
    """
    x_demo    = np.asarray(demo["x"],    dtype=float)
    xdot_demo = np.asarray(demo["xdot"], dtype=float)
    S_arr = np.asarray(S, dtype=float)
    T_arr = np.asarray(T, dtype=float)

    # 1–2: Fit transport and transport demo ----------------------------------
    transport = PolicyTransport(gp_cls=gp_cls, n_iter_default=gp_n_iter)
    transport.fit(S_arr, T_arr)
    x_t    = transport.transform(x_demo)            # (N, 3)
    xd_t   = transport.transform_velocity(x_demo, xdot_demo)  # (N, 3)

    # 3: Refit GP DS ---------------------------------------------------------
    ds = GPDynamicalSystem(gp_cls=gp_cls, n_iter_default=gp_n_iter)
    ds.fit(x_t, xd_t)

    # Workspace bounds for clamping
    ws_lo, ws_hi = env.get_workspace_bounds()

    # 4: Euler rollout --------------------------------------------------------
    env.reset()
    # Move to first transported waypoint
    init_pos = np.clip(x_t[0], ws_lo, ws_hi)
    env.set_ee_pos(init_pos)

    xs:      List[np.ndarray] = [env.get_ee_pos().copy()]
    qs_raw:  List[np.ndarray] = [env._data.qpos[:7].copy()]
    frames_raw: List[np.ndarray] = []
    ik_fails = 0

    for _ in range(n_steps):
        obs = xs[-1]
        vel_cmd = ds.predict(obs[np.newaxis], return_std=False)
        if vel_cmd.ndim == 2:
            vel_cmd = vel_cmd[0]

        x_next = obs + vel_cmd * dt
        # Clamp BEFORE IK (per spec)
        x_next = np.clip(x_next, ws_lo, ws_hi)

        _, _, _, _, info = env.step(x_next)
        if not info["ik_success"]:
            ik_fails += 1

        xs.append(env.get_ee_pos().copy())
        qs_raw.append(env._data.qpos[:7].copy())

        if env._render_mode == "rgb_array":
            frames_raw.append(env.render())

    rollout_x  = np.array(xs,     dtype=float)  # (N+1, 3)
    q_raw_arr  = np.array(qs_raw, dtype=float)  # (N+1, 7)

    # 5: Smooth joints (Gaussian sigma=1.5 along time axis) -----------------
    q_smooth = gaussian_filter1d(q_raw_arr, sigma=1.5, axis=0)

    # 6: Re-render with smoothed joints --------------------------------------
    frames: List[np.ndarray] = []
    if env._render_mode == "rgb_array":
        for q in q_smooth:
            env.set_qpos(q)
            frame = env.render()
            if frame is not None:
                frames.append(frame)

    # Success / error --------------------------------------------------------
    final_pos  = rollout_x[-1]
    goal_pos   = x_t[-1]
    final_error = float(np.linalg.norm(final_pos - goal_pos))
    success = final_error < 0.1
    ik_fail_rate = ik_fails / max(1, n_steps)

    return {
        "rollout_x":    rollout_x,
        "rollout_q":    q_smooth,
        "frames":       frames,
        "success":      success,
        "final_error":  final_error,
        "ik_fail_rate": ik_fail_rate,
        "transport":    transport,
    }


# ---------------------------------------------------------------------------
# Generalisation evaluation
# ---------------------------------------------------------------------------

def evaluate_franka_generalization(
    base_demo_waypoints: np.ndarray,
    base_scene: dict,
    randomize_fn: Callable[[dict, int], dict],
    n_trials: int = 4,
    seed: int = 0,
    task: str = "reshelving",
    gp_n_iter: int = 100,
    n_steps: int = 120,
) -> dict:
    """Run transport_and_rollout_franka for n_trials randomized scenes.

    Parameters
    ----------
    base_demo_waypoints : (M, 3) EE waypoints for the base demo.
    base_scene : dict with "S" (source landmarks) and any scene geometry.
    randomize_fn : callable(scene, seed) -> new_scene dict (must have "S", "T").
    n_trials : int
    seed : int
    task : str — Franka task name.

    Returns
    -------
    dict:
        "success_rate"      : float
        "mean_final_error"  : float
        "std_final_error"   : float
        "mean_ik_fail_rate" : float
        "all_results"       : list[dict]
    """
    all_results = []

    # Record base demo once
    base_env = FrankaKinematicEnv(task, render_mode=None)
    base_env.reset(seed=seed)
    base_demo = record_franka_demo(base_env, base_demo_waypoints)
    base_env.close()

    for i in range(n_trials):
        trial_seed = seed + i
        new_scene  = randomize_fn(base_scene, seed=trial_seed)
        S = np.asarray(new_scene.get("S", base_scene.get("S", np.zeros((4, 3)))))
        T = np.asarray(new_scene.get("T", S))

        env = FrankaKinematicEnv(task, render_mode="rgb_array",
                                  scene_kwargs=new_scene.get("scene_kwargs", {}))
        result = transport_and_rollout_franka(
            demo=base_demo,
            S=S, T=T,
            env=env,
            gp_n_iter=gp_n_iter,
            n_steps=n_steps,
            seed=trial_seed,
        )
        env.close()
        all_results.append(result)

    successes   = [r["success"]      for r in all_results]
    errors      = [r["final_error"]  for r in all_results]
    fail_rates  = [r["ik_fail_rate"] for r in all_results]

    return {
        "success_rate":      float(np.mean(successes)),
        "mean_final_error":  float(np.mean(errors)),
        "std_final_error":   float(np.std(errors)),
        "mean_ik_fail_rate": float(np.mean(fail_rates)),
        "all_results":       all_results,
    }
