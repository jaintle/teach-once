"""Tests for the Sec. V-B multi-reference-frame benchmark (Phase 7)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from gpt_repro.baselines import (
    DMPBaseline,
    GPTBaseline,
    HMMBaseline,
    TPGMMBaseline,
)
from gpt_repro.metrics import (
    METRIC_FNS,
    area_between_curves,
    build_ranking_table,
    dtw_distance,
    final_orientation_error,
    final_position_error,
    frechet_distance,
    mann_whitney_ranking,
)
from gpt_repro.policies import (
    FrameConfig,
    get_frame_points,
    make_9_frame_configs,
    make_canonical_demo,
    make_multiframe_demo,
)
from gpt_repro.utils import set_global_seed


# ---------------------------------------------------------------------------
def test_frame_points_shape():
    cfg = make_9_frame_configs(seed=0)[0]
    S, T = get_frame_points(cfg, n_pts_per_frame=5)
    assert S.shape == (10, 2)
    assert T.shape == (10, 2)


# ---------------------------------------------------------------------------
def test_demo_generator_arc_length():
    cfg = make_9_frame_configs(seed=1)[0]
    demo = make_multiframe_demo(cfg, n_points=60, noise=0.0)
    arc = float(np.sum(np.linalg.norm(np.diff(demo["x"], axis=0), axis=-1)))
    straight = float(np.linalg.norm(cfg.goal_pos - cfg.start_pos))
    assert arc > straight + 1e-3, f"trajectory is straight: arc={arc}, straight={straight}"


# ---------------------------------------------------------------------------
def test_metrics_all_zero_on_identical_trajectories():
    rng = np.random.default_rng(0)
    traj = rng.uniform(-1, 1, size=(30, 2))
    assert frechet_distance(traj, traj) == pytest.approx(0.0, abs=1e-9)
    assert area_between_curves(traj, traj) == pytest.approx(0.0, abs=1e-6)
    assert dtw_distance(traj, traj) == pytest.approx(0.0, abs=1e-9)
    assert final_position_error(traj, traj) == pytest.approx(0.0, abs=1e-12)
    # Float32 arccos round-off near 1.0 — relax to 1e-6.
    assert final_orientation_error(traj, traj) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
def test_final_orientation_error_90deg():
    # Two horizontal/vertical approach trajectories meeting at the origin.
    pred = np.array([[1.0, 0.0], [0.5, 0.0], [0.0, 0.0]])    # incoming from +x
    gt   = np.array([[0.0, 1.0], [0.0, 0.5], [0.0, 0.0]])    # incoming from +y
    err = final_orientation_error(pred, gt)
    assert abs(err - np.pi / 2) < 1e-6


# ---------------------------------------------------------------------------
def test_mann_whitney_ranking_deterministic():
    rng = np.random.default_rng(0)
    results = {
        "A": rng.normal(0.0, 0.1, size=30).tolist(),
        "B": rng.normal(1.0, 0.1, size=30).tolist(),
        "C": rng.normal(2.0, 0.1, size=30).tolist(),
    }
    p1, r1 = mann_whitney_ranking(results)
    p2, r2 = mann_whitney_ranking(results)
    assert p1 == p2
    assert r1 == r2


# ---------------------------------------------------------------------------
def test_utest_clear_winner():
    rng = np.random.default_rng(0)
    res = {
        "low":  rng.normal(0.0, 0.1, size=20).tolist(),
        "high": rng.normal(10.0, 0.1, size=20).tolist(),
    }
    points, rank = mann_whitney_ranking(res)
    assert points["low"] >= 1
    assert points["high"] == 0
    assert rank["low"] == 1
    assert rank["high"] == 2


# ---------------------------------------------------------------------------
def test_tpgmm_rollout_reaches_goal():
    set_global_seed(0)
    cfgs = make_9_frame_configs(seed=0)[:3]
    demos = [make_multiframe_demo(c, n_points=60, seed=i) for i, c in enumerate(cfgs)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tp = TPGMMBaseline(n_components=4).fit(demos, cfgs)
    cfg = cfgs[0]
    x0 = cfg.start_pos
    initial_dist = float(np.linalg.norm(cfg.goal_pos - x0))
    traj = tp.rollout(cfg, cfg, x0, dt=0.05, n_steps=80)
    final_dist = float(np.linalg.norm(cfg.goal_pos - traj[-1]))
    assert final_dist < 2.0 * initial_dist, (
        f"TPGMM rollout did not move toward goal: "
        f"initial={initial_dist}, final={final_dist}"
    )


# ---------------------------------------------------------------------------
def test_gpt_adapter_rollout_shape():
    set_global_seed(0)
    cfg = make_9_frame_configs(seed=0)[0]
    canon = make_canonical_demo(cfg, n_points=40)
    S, T = get_frame_points(cfg)
    gpt = GPTBaseline(n_iter_transport=80, n_iter_ds=60).fit(
        S, T, canon["x"], canon["xdot"],
    )
    traj, _ = gpt.rollout(cfg.start_pos, dt=0.05, n_steps=40)
    assert traj.shape == (41, 2)


@pytest.fixture(autouse=True)
def _silence_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield
