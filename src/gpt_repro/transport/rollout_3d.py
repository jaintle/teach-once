"""3-D rollout pipeline for Phase 9.

Ties together:
1. Recording a demonstration in a :class:`KinematicEndEffectorEnv`.
2. Transporting the demonstration to a new scene via :class:`PolicyTransport`.
3. Re-fitting a :class:`GPDynamicalSystem` on the transported demo.
4. Rolling out the DS inside the env (Euler integration).
5. Evaluating generalisation over many randomised scenes.

All functions operate in R^3 (or any d, since the transport math is
dimension-agnostic).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple, Type

import numpy as np

from gpt_repro.gp.exact_gp import ExactGPRegressor
from gpt_repro.policies.ds_policy import GPDynamicalSystem
from gpt_repro.transport.policy_transport import PolicyTransport
from gpt_repro.envs.base_env import KinematicEndEffectorEnv


# ---------------------------------------------------------------------------
# Demo recording
# ---------------------------------------------------------------------------

def record_demo_3d(
    env: KinematicEndEffectorEnv,
    policy_fn: Callable[[np.ndarray], np.ndarray],
    n_steps: int = 150,
) -> dict:
    """Record an open-loop trajectory by rolling out ``policy_fn`` in ``env``.

    Parameters
    ----------
    env : KinematicEndEffectorEnv
        A reset environment (``env.reset()`` should have been called).
    policy_fn : callable (pos: (3,)) -> action: (3,)
        Maps the current EE position to a velocity command.
    n_steps : int
        Number of simulation steps.

    Returns
    -------
    dict with keys:
        "x"    : (n_steps+1, 3) positions (including initial).
        "xdot" : (n_steps, 3)   velocity commands applied.
        "t"    : (n_steps+1,)   time stamps.
    """
    dt = env._dt
    positions = []
    velocities = []
    obs, _ = env.reset()
    positions.append(obs.copy())
    for _ in range(n_steps):
        action = np.asarray(policy_fn(obs), dtype=float)
        obs, _, _, _, _ = env.step(action)
        positions.append(obs.copy())
        velocities.append(action.copy())

    x_arr = np.array(positions)           # (n_steps+1, 3)
    xdot_arr = np.array(velocities)        # (n_steps, 3)
    t_arr = np.arange(x_arr.shape[0]) * dt
    return {"x": x_arr, "xdot": xdot_arr, "t": t_arr}


# ---------------------------------------------------------------------------
# Full transport → rollout pipeline
# ---------------------------------------------------------------------------

def transport_and_rollout_3d(
    demo: dict,
    S: np.ndarray,
    T: np.ndarray,
    env: KinematicEndEffectorEnv,
    gp_cls: Type = ExactGPRegressor,
    dt: float = 0.02,
    n_steps: int = 150,
    gp_n_iter: int = 100,
    seed: int = 0,
    attractor_gain: float = 0.0,
) -> dict:
    """Transport a 3-D demonstration to a new scene and roll it out.

    Pipeline (cf. Sec. IV of the paper, extended to d=3):

    1. Fit a :class:`PolicyTransport` (Eq. 7: ϕ = γ + ψ ∘ γ) mapping
       source frame ``S`` to target frame ``T``.
    2. Transport the demo waypoints via ϕ.
    3. Transport the demo velocities via :meth:`PolicyTransport.transform_velocity`
       (Eq. 13: ``ẋ̂ = J(x) ẋ``).
    4. Refit a :class:`GPDynamicalSystem` on the transported (x̂, ẋ̂) pairs.
    5. Roll out the DS via Euler integration inside ``env``.

    Parameters
    ----------
    demo : dict
        Demonstration dict with ``"x"`` (N, 3) and ``"xdot"`` (N, 3).
    S : (M, 3) array — source frame landmark positions.
    T : (M, 3) array — target frame landmark positions.
    env : KinematicEndEffectorEnv
        Environment for the rollout.  ``env.reset()`` will be called
        with ``init_pos`` set to the first transported waypoint.
    gp_cls : type, optional
        GP regressor class for the DS. Defaults to :class:`ExactGPRegressor`.
    dt : float, optional
        Euler integration time step. Defaults to 0.02 s.
    n_steps : int, optional
        Number of rollout steps. Defaults to 150.
    gp_n_iter : int, optional
        Training iterations for the GP DS. Defaults to 100.
    seed : int, optional
        Random seed (passed to PolicyTransport / GP fitters).

    Returns
    -------
    dict with keys:
        "rollout_x"   : (n_steps+1, 3) — EE positions during rollout.
        "transported_x" : (N, 3)       — transported demo waypoints.
        "success"     : bool            — ``env.is_success()`` at end.
        "final_error" : float           — distance to goal at end.
        "transport"   : PolicyTransport — fitted transport object.
    """
    rng = np.random.default_rng(seed)

    x_demo = np.asarray(demo["x"], dtype=float)
    xdot_demo = np.asarray(demo["xdot"], dtype=float)
    S_arr = np.asarray(S, dtype=float)
    T_arr = np.asarray(T, dtype=float)

    # -- 1 & 2: Fit PolicyTransport and transport demo positions --------
    transport = PolicyTransport(gp_cls=gp_cls, n_iter_default=gp_n_iter)
    transport.fit(S_arr, T_arr)
    x_transported = transport.transform(x_demo)  # (N, 3)

    # -- 3: Transport velocities (Eq. 13) --------------------------------
    xdot_transported = transport.transform_velocity(x_demo, xdot_demo)  # (N, 3)

    # -- 4: Refit GP DS on transported demo -------------------------------
    ds = GPDynamicalSystem(gp_cls=gp_cls, n_iter_default=gp_n_iter)
    ds.fit(x_transported, xdot_transported)

    # -- 5: Euler rollout in env -----------------------------------------
    x_goal_3d = x_transported[-1]
    init_pos = x_transported[0]
    obs, _ = env.reset(options={"init_pos": init_pos})
    positions = [obs.copy()]
    for _ in range(n_steps):
        vel_cmd = ds.predict(obs, return_std=False)
        if vel_cmd.ndim == 2:
            vel_cmd = vel_cmd[0]
        # Optional attractor term (Sec. III-A)
        if attractor_gain > 0.0:
            vel_cmd = vel_cmd + attractor_gain * (x_goal_3d - obs)
        obs, _, _, _, _ = env.step(vel_cmd)
        positions.append(obs.copy())

    rollout_x = np.array(positions)  # (n_steps+1, 3)

    # -- Success / error -------------------------------------------------
    success = env.is_success() if hasattr(env, "is_success") else False
    final_pos = rollout_x[-1]
    # Determine goal: use env's goal attribute if available
    if hasattr(env, "_goal_pos"):
        final_error = float(np.linalg.norm(final_pos - env._goal_pos))
    elif hasattr(env, "_hand"):
        final_error = float(np.linalg.norm(final_pos - env._hand))
    else:
        final_error = float(np.linalg.norm(final_pos - x_transported[-1]))

    return {
        "rollout_x": rollout_x,
        "transported_x": x_transported,
        "success": success,
        "final_error": final_error,
        "transport": transport,
    }


# ---------------------------------------------------------------------------
# Generalisation evaluation
# ---------------------------------------------------------------------------

def evaluate_generalization_3d(
    base_demo: dict,
    base_scene: dict,
    randomize_fn: Callable[[dict, int], dict],
    n_trials: int = 10,
    seed: int = 0,
    env_cls: Type[KinematicEndEffectorEnv] = None,
    **transport_kwargs: Any,
) -> dict:
    """Evaluate generalisation over randomised scenes.

    For each trial ``i`` in ``range(n_trials)``:
    1. Randomise the scene with ``seed = seed + i``.
    2. Instantiate ``env_cls(scene=new_scene)``.
    3. Call :func:`transport_and_rollout_3d` with the new scene's ``S``/``T``.
    4. Collect success flag and final error.

    Parameters
    ----------
    base_demo : dict
        Demonstration from the base scene.
    base_scene : dict
        Base scene dict (must have ``"S"`` and ``"T"`` arrays).
    randomize_fn : callable (scene, seed) -> scene
        Function that returns a new randomised scene.
    n_trials : int, optional
        Number of random trials. Defaults to 10.
    seed : int, optional
        Base seed (trial ``i`` uses ``seed + i``). Defaults to 0.
    env_cls : type, optional
        Environment class to instantiate. Defaults to
        :class:`~gpt_repro.envs.reshelving_env.ReshelvingEnv`.
    **transport_kwargs
        Extra keyword arguments forwarded to :func:`transport_and_rollout_3d`.

    Returns
    -------
    dict with keys:
        "all_rollouts"   : list[dict] — per-trial result dicts.
        "success_rate"   : float      — fraction of successful trials.
        "mean_error"     : float      — mean final error across trials.
        "std_error"      : float      — std of final errors.
        "final_errors"   : list[float] — per-trial final errors.
    """
    if env_cls is None:
        from gpt_repro.envs.reshelving_env import ReshelvingEnv
        env_cls = ReshelvingEnv

    all_rollouts = []
    for i in range(n_trials):
        trial_seed = seed + i
        new_scene = randomize_fn(base_scene, seed=trial_seed)
        env = env_cls(scene=new_scene)
        S = new_scene["S"]
        T = new_scene.get("T", new_scene["S"])
        result = transport_and_rollout_3d(
            demo=base_demo,
            S=S,
            T=T,
            env=env,
            seed=trial_seed,
            **transport_kwargs,
        )
        env.close()
        all_rollouts.append(result)

    successes = [r["success"] for r in all_rollouts]
    errors = [r["final_error"] for r in all_rollouts]
    return {
        "all_rollouts": all_rollouts,
        "success_rate": float(np.mean(successes)),
        "mean_error": float(np.mean(errors)),
        "std_error": float(np.std(errors)),
        "final_errors": errors,
    }
