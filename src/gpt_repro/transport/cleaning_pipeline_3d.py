"""Full 3D surface cleaning pipeline — Phase 10 (Sec. VI-C analog).

Orchestrates:
1. Surface point cloud generation (source + target).
2. NN-based cloud pairing.
3. Demo generation on the source surface.
4. SVGPPolicyTransport fit on (S_paired, T_paired).
5. Demo transport: positions, velocities, orientations, stiffness.
6. GPDynamicalSystem refit on transported demo.
7. Rollout in SurfaceCleaningEnv.
8. Force norm estimation via Hooke's law proxy.
9. Coverage and mean surface distance computation.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from gpt_repro.policies.surfaces_3d import (
    SurfaceConfig,
    make_surface_pointcloud,
    make_surface_demo,
    pair_surface_clouds,
)
from gpt_repro.transport.policy_transport_svgp import SVGPPolicyTransport
from gpt_repro.policies.ds_policy import GPDynamicalSystem
from gpt_repro.envs.cleaning_env import SurfaceCleaningEnv
from gpt_repro.utils.seeding import set_global_seed


def run_cleaning_pipeline(
    source_config: SurfaceConfig,
    target_config: SurfaceConfig,
    n_source_pts: int = 400,
    n_target_pts: int = 400,
    n_inducing: int = 100,
    n_demo_pts: int = 100,
    seed: int = 0,
    n_iter: int = 300,
    batch_size: int = 256,
    n_steps: int = 150,
    gp_n_iter: int = 100,
) -> dict:
    """Run the full 3D surface cleaning pipeline.

    Pipeline:
    1. Generate source cloud S (n_source_pts pts).
    2. Generate target cloud T (n_target_pts pts).
    3. Pair S and T by nearest-neighbour (pair_surface_clouds).
    4. Generate cleaning demo on source surface (make_surface_demo).
    5. Fit SVGPPolicyTransport on (S_paired, T_paired).
    6. Transport demo: positions (Eq. 7), velocities (Eq. 13),
       orientations (Eq. 15), stiffness matrices (Sec. IV-D).
    7. Refit GPDynamicalSystem on transported (x̂, ẋ̂).
    8. Roll out GPDynamicalSystem in SurfaceCleaningEnv(target_config).
    9. Estimate force norm at each rollout step via Hooke's law proxy.
    10. Compute coverage fraction and mean surface distance.

    Parameters
    ----------
    source_config : SurfaceConfig — source surface.
    target_config : SurfaceConfig — target surface.
    n_source_pts : int — points in source cloud.
    n_target_pts : int — points in target cloud.
    n_inducing : int — SVGP inducing points (paper: 100 for N=400).
    n_demo_pts : int — demo trajectory waypoints.
    seed : int — global seed.
    n_iter : int — SVGP training iterations.
    batch_size : int — SVGP mini-batch size.
    n_steps : int — rollout steps.
    gp_n_iter : int — GPDynamicalSystem training iterations.

    Returns
    -------
    dict with keys:
        "rollout_x"         : (n_steps+1, 3) rollout positions.
        "force_norms"       : (n_steps+1,) Hooke's law force proxy.
        "demo_x"            : (n_demo_pts, 3) original demo positions.
        "transported_x"     : (n_demo_pts, 3) transported demo positions.
        "S"                 : (n_source_pts, 3) source cloud.
        "T"                 : (n_target_pts, 3) target cloud.
        "coverage"          : float in [0, 1].
        "mean_surface_dist" : float mean dist of rollout to target surface.
        "transport"         : SVGPPolicyTransport fitted instance.
        "transported_stiffness" : (n_demo_pts, 3, 3) transported Ks.
    """
    set_global_seed(seed)

    # -- 1. Generate clouds --------------------------------------------------
    S = make_surface_pointcloud(source_config, n_points=n_source_pts, seed=seed)
    T = make_surface_pointcloud(target_config, n_points=n_target_pts, seed=seed)

    # -- 2. Pair clouds via NN ------------------------------------------------
    S_paired, T_paired = pair_surface_clouds(S, T)

    # -- 3. Generate source demo ---------------------------------------------
    demo = make_surface_demo(source_config, n_points=n_demo_pts, seed=seed)
    x_demo = demo["x"]             # (N, 3)
    xdot_demo = demo["xdot"]       # (N, 3)
    Ks_demo = demo["stiffness"]    # (N, 3, 3)
    R_demo = demo["orientation"]   # (N, 3, 3)

    # -- 4. Fit SVGPPolicyTransport -------------------------------------------
    transport = SVGPPolicyTransport(
        n_inducing=n_inducing,
        n_iter_default=n_iter,
        batch_size=batch_size,
    )
    transport.fit(S_paired, T_paired)

    # -- 5. Transport demo ---------------------------------------------------
    x_transported = transport.transform(x_demo)              # (N, 3)
    xdot_transported = transport.transform_velocity(x_demo, xdot_demo)  # (N, 3)

    # Transport orientations (Eq. 15)
    R_transported = transport.transform_orientation(x_demo, R_demo)  # (N, 3, 3)

    # Transport stiffness (Sec. IV-D: K̂_s = J K_s Jᵀ)
    Ks_transported = transport.transform_stiffness(x_demo, Ks_demo)  # (N, 3, 3)

    # Verify finite outputs
    for arr, name in [
        (x_transported, "x_transported"),
        (xdot_transported, "xdot_transported"),
        (Ks_transported, "Ks_transported"),
    ]:
        if not np.all(np.isfinite(arr)):
            raise ValueError(
                f"Non-finite values in {name} after transport. "
                "Check SVGP convergence or reduce n_inducing."
            )

    # -- 6. Refit GPDynamicalSystem ------------------------------------------
    ds = GPDynamicalSystem(n_iter_default=gp_n_iter)
    ds.fit(x_transported, xdot_transported)

    # -- 7. Rollout in target env --------------------------------------------
    env = SurfaceCleaningEnv(target_config, n_surface_pts=n_target_pts)
    init_pos = x_transported[0]
    obs, _ = env.reset(options={"init_pos": init_pos})
    positions = [obs.copy()]

    # Build transported stiffness lookup for force computation
    # We use the transported stiffness profile for force estimation
    # at the demo-indexed steps (best approximation without refit of Ks field).
    # For steps beyond demo length, use the last stiffness value.

    for step_i in range(n_steps):
        vel_cmd = ds.predict(obs, return_std=False)
        if vel_cmd.ndim == 2:
            vel_cmd = vel_cmd[0]
        obs, _, _, _, _ = env.step(vel_cmd)
        positions.append(obs.copy())

    rollout_x = np.array(positions)  # (n_steps+1, 3)

    # -- 8. Force norm estimation --------------------------------------------
    # Estimate forces along rollout using transported Ks profile.
    # Match transported stiffness to rollout steps by nearest-demo-point lookup.
    from scipy.spatial import cKDTree
    demo_tree = cKDTree(x_transported)

    force_norms = np.empty(len(rollout_x))
    for i, pos in enumerate(rollout_x):
        # velocity at this rollout step
        vel = ds.predict(pos, return_std=False)
        if vel.ndim == 2:
            vel = vel[0]
        # Nearest transported demo point for stiffness lookup
        _, idx = demo_tree.query(pos, k=1)
        Ks_i = Ks_transported[idx]
        force_norms[i] = env.get_contact_force_norm(Ks_i, vel)

    # -- 9. Coverage and surface distance ------------------------------------
    coverage = env.coverage_fraction(rollout_x, tol=0.025)

    from scipy.spatial import cKDTree as _KDT
    t_tree = _KDT(T)
    dists, _ = t_tree.query(rollout_x, k=1)
    mean_surface_dist = float(np.mean(dists))

    env.close()

    if not np.all(np.isfinite(force_norms)):
        force_norms = np.where(np.isfinite(force_norms), force_norms, 0.0)

    return {
        "rollout_x": rollout_x,
        "force_norms": force_norms,
        "demo_x": x_demo,
        "transported_x": x_transported,
        "S": S,
        "T": T,
        "coverage": coverage,
        "mean_surface_dist": mean_surface_dist,
        "transport": transport,
        "transported_stiffness": Ks_transported,
    }
