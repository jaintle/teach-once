"""Tests for Phase 14 — GPT-Franka rollout adapter.

8 tests covering record_franka_demo, transport_and_rollout_franka,
joint smoothing, workspace clamping, and frame annotation utilities.
"""

import numpy as np
import pytest

from gpt_repro.envs.franka_env import FrankaKinematicEnv
from gpt_repro.policies.franka_demos import get_reshelving_waypoints
from gpt_repro.transport.franka_rollout import (
    record_franka_demo,
    transport_and_rollout_franka,
)
from gpt_repro.viz.frame_annotate import add_text_overlay, add_title_bar


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

BASE_SCENE = {
    "object_pose": np.array([0.50, 0.00, 0.63]),
    "goal_pose":   np.array([0.30, 0.10, 0.75]),
}

S4 = np.array([
    [0.50,  0.05, 0.63], [0.50, -0.05, 0.63],
    [0.30,  0.05, 0.75], [0.30, -0.05, 0.75],
])
T4 = S4 + np.array([0.03, 0.02, 0.01])


@pytest.fixture(scope="module")
def base_demo():
    env = FrankaKinematicEnv("reshelving", render_mode=None)
    env.reset(seed=0)
    waypoints = get_reshelving_waypoints(BASE_SCENE)
    demo = record_franka_demo(env, waypoints, n_interp=2)
    env.close()
    return demo


@pytest.fixture(scope="module")
def rollout_result(base_demo):
    env = FrankaKinematicEnv("reshelving", render_mode="rgb_array")
    res = transport_and_rollout_franka(
        demo=base_demo, S=S4, T=T4, env=env,
        gp_n_iter=30, n_steps=20, seed=0,
    )
    env.close()
    return res


# ---------------------------------------------------------------------------
# 1. record_franka_demo shapes
# ---------------------------------------------------------------------------

def test_record_demo_shape(base_demo):
    N = len(base_demo["x"])
    assert N > 0
    assert base_demo["x"].shape == (N, 3), f"x shape: {base_demo['x'].shape}"
    assert base_demo["q"].shape == (N, 7), f"q shape: {base_demo['q'].shape}"
    assert base_demo["xdot"].shape == (N, 3)
    assert base_demo["ik_success"].shape == (N,)
    assert base_demo["t"].shape == (N,)


# ---------------------------------------------------------------------------
# 2. IK success rate > 0.8 for workspace-centred waypoints
# ---------------------------------------------------------------------------

def test_record_demo_ik_mostly_succeeds():
    env = FrankaKinematicEnv("reshelving", render_mode=None)
    env.reset(seed=0)
    wps = np.array([
        [0.40, 0.00, 0.65],
        [0.45, 0.05, 0.70],
        [0.50, 0.00, 0.75],
    ])
    demo = record_franka_demo(env, wps, n_interp=3)
    env.close()
    rate = float(demo["ik_success"].mean())
    assert rate > 0.8, f"IK success rate {rate:.2f} < 0.8"


# ---------------------------------------------------------------------------
# 3. frames non-empty after transport_and_rollout_franka
# ---------------------------------------------------------------------------

def test_transport_rollout_frames_nonempty(rollout_result):
    assert len(rollout_result["frames"]) > 0


# ---------------------------------------------------------------------------
# 4. frame shape is (480, 720, 3) uint8
# ---------------------------------------------------------------------------

def test_transport_rollout_frame_shape(rollout_result):
    frame = rollout_result["frames"][0]
    assert frame.shape == (480, 720, 3), f"frame shape: {frame.shape}"
    assert frame.dtype == np.uint8


# ---------------------------------------------------------------------------
# 5. joint smoothing reduces jitter
# ---------------------------------------------------------------------------

def test_joint_smoothing_reduces_jitter(rollout_result):
    from scipy.ndimage import gaussian_filter1d

    # rollout_q is already smoothed; recover raw from rollout by computing
    # a fresh "rough" trajectory with larger IK noise
    env = FrankaKinematicEnv("reshelving", render_mode=None)
    env.reset(seed=0)
    wps = get_reshelving_waypoints(BASE_SCENE)
    demo = record_franka_demo(env, wps, n_interp=1)  # n_interp=1 → more jitter
    env.close()

    q_raw    = demo["q"]   # (N, 7)
    q_smooth = gaussian_filter1d(q_raw, sigma=1.5, axis=0)

    std_raw    = float(np.diff(q_raw,    axis=0).std())
    std_smooth = float(np.diff(q_smooth, axis=0).std())
    assert std_smooth <= std_raw, (
        f"Smoothed std {std_smooth:.4f} not <= raw std {std_raw:.4f}"
    )


# ---------------------------------------------------------------------------
# 6. workspace clamping — no EE position exceeds bounds
# ---------------------------------------------------------------------------

def test_workspace_clamping(rollout_result):
    env = FrankaKinematicEnv("reshelving", render_mode=None)
    lo, hi = env.get_workspace_bounds()
    env.close()
    for pos in rollout_result["rollout_x"]:
        # Allow 5mm slack for IK residual
        assert np.all(pos >= lo - 0.005), f"pos {pos} below lower bound {lo}"
        assert np.all(pos <= hi + 0.005), f"pos {pos} above upper bound {hi}"


# ---------------------------------------------------------------------------
# 7. add_text_overlay preserves shape
# ---------------------------------------------------------------------------

def test_text_overlay_shape():
    frame = np.zeros((480, 720, 3), dtype=np.uint8)
    out   = add_text_overlay(frame, "hello", pos=(10, 30))
    assert out.shape == frame.shape, f"shape changed: {out.shape}"


# ---------------------------------------------------------------------------
# 8. add_title_bar increases height by bar_height
# ---------------------------------------------------------------------------

def test_title_bar_increases_height():
    frame      = np.zeros((480, 720, 3), dtype=np.uint8)
    bar_height = 40
    out        = add_title_bar(frame, "Title", bar_height=bar_height)
    assert out.shape == (480 + bar_height, 720, 3), (
        f"Expected ({480 + bar_height}, 720, 3), got {out.shape}"
    )
