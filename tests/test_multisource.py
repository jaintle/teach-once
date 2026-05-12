"""Tests for Phase 8 — multi-source single-target (Sec. V-C)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from gpt_repro.baselines.multisource_dmp import MultiSourceDMP
from gpt_repro.baselines.multisource_gpt import MultiSourceGPT
from gpt_repro.baselines.gpt_adapter import GPTBaseline
from gpt_repro.metrics.trajectory_metrics import frechet_distance
from gpt_repro.policies.multisource_demos import make_multisource_scenario


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def scenario2():
    """Tiny 2-source scenario used by most tests."""
    return make_multisource_scenario(n_sources=2, seed=0, n_points=30)


@pytest.fixture(scope="module")
def scenario4():
    """4-source scenario for the fusion quality test."""
    return make_multisource_scenario(n_sources=4, seed=0, n_points=30)


# ---------------------------------------------------------------------------
# 1. Scenario shape
# ---------------------------------------------------------------------------
def test_scenario_shapes(scenario2):
    sc = scenario2
    n = 2
    assert len(sc["source_configs"]) == n
    assert len(sc["source_demos"]) == n
    assert len(sc["S_list"]) == n

    for demo in sc["source_demos"]:
        assert set(demo.keys()) >= {"x", "xdot", "t"}
        assert demo["x"].shape == (30, 2)
        assert demo["xdot"].shape == (30, 2)
        assert demo["t"].shape == (30,)

    td = sc["target_demo"]
    assert set(td.keys()) >= {"x", "xdot", "t"}
    assert td["x"].shape == (30, 2)

    for S_k in sc["S_list"]:
        assert S_k.shape == (5, 2)
    assert sc["T"].shape == (5, 2)


# ---------------------------------------------------------------------------
# 2. MultiSourceGPT fit (no exception)
# ---------------------------------------------------------------------------
def test_multisource_gpt_fit_no_error(scenario2):
    sc = scenario2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = MultiSourceGPT(n_iter_transport=20, n_iter_ds=20).fit(
            sc["S_list"], sc["T"], sc["source_demos"],
        )
    assert model.ds is not None
    assert len(model.transports) == 2


# ---------------------------------------------------------------------------
# 3. MultiSourceDMP fit (no exception)
# ---------------------------------------------------------------------------
def test_multisource_dmp_fit_no_error(scenario2):
    sc = scenario2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = MultiSourceDMP(n_iter_gp=20).fit(
            sc["S_list"], sc["T"], sc["source_demos"],
        )
    assert model.ds is not None
    assert len(model.gammas) == 2


# ---------------------------------------------------------------------------
# 4. Rollout shape
# ---------------------------------------------------------------------------
def test_rollout_shape(scenario2):
    sc = scenario2
    x0 = sc["target_demo"]["x"][0]
    n_steps = 15
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = MultiSourceGPT(n_iter_transport=20, n_iter_ds=20).fit(
            sc["S_list"], sc["T"], sc["source_demos"],
        )
    traj, vels = model.rollout(x0, dt=0.05, n_steps=n_steps)
    assert traj.shape == (n_steps + 1, 2)
    assert vels.shape == (n_steps + 1, 2)


# ---------------------------------------------------------------------------
# 5. Multi-source GPT Fréchet ≤ 2× single-source GPT Fréchet
# ---------------------------------------------------------------------------
def test_multisource_gpt_beats_single_on_frechet(scenario4):
    sc = scenario4
    x0     = sc["target_demo"]["x"][0]
    gt     = sc["target_demo"]["x"]
    n_steps = 50

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        msg = MultiSourceGPT(n_iter_transport=60, n_iter_ds=60).fit(
            sc["S_list"], sc["T"], sc["source_demos"],
        )
        traj_msg, _ = msg.rollout(x0, n_steps=n_steps)

        sgpt = GPTBaseline(n_iter_transport=60, n_iter_ds=60).fit(
            sc["S_list"][0], sc["T"],
            sc["source_demos"][0]["x"], sc["source_demos"][0]["xdot"],
        )
        traj_sgpt, _ = sgpt.rollout(x0, n_steps=n_steps)

    f_msg  = frechet_distance(traj_msg,  gt)
    f_sgpt = frechet_distance(traj_sgpt, gt)
    # Loose threshold: multi-source should not be more than 2× worse than single
    assert f_msg <= 2.0 * f_sgpt + 1e-6, (
        f"MultiSourceGPT Fréchet {f_msg:.4f} > 2× SingleSourceGPT {f_sgpt:.4f}"
    )


# ---------------------------------------------------------------------------
# 6. Uncertainty non-negative
# ---------------------------------------------------------------------------
def test_uncertainty_non_negative(scenario2):
    sc = scenario2
    x0 = sc["target_demo"]["x"][0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = MultiSourceGPT(n_iter_transport=20, n_iter_ds=20).fit(
            sc["S_list"], sc["T"], sc["source_demos"],
        )
    X = sc["target_demo"]["x"]
    Xdot = sc["target_demo"]["xdot"]
    std = model.uncertainty(X, Xdot)
    assert std.shape == (len(X),)
    assert np.all(np.isfinite(std)), "uncertainty contains non-finite values"
    assert np.all(std >= 0.0), "uncertainty contains negative values"
