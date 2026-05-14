"""Tests for Phase 9 MuJoCo environments and 3-D rollout pipeline.

Covers:
1. ReshelvingEnv reset obs shape (3,).
2. ReshelvingEnv step with zero action keeps position.
3. ReshelvingEnv success detection (EE at goal → is_success True).
4. ArmPoseEnv reset obs shape (3,).
5. transport_and_rollout_3d output keys.
6. evaluate_generalization_3d success_rate in [0,1].
"""

from __future__ import annotations

import numpy as np
import pytest

from gpt_repro.envs.reshelving_env import ReshelvingEnv
from gpt_repro.envs.armpose_env import ArmPoseEnv
from gpt_repro.policies.demos_3d import (
    make_reshelving_demo,
    make_armpose_demo,
    randomize_reshelving_scene,
)
from gpt_repro.transport.rollout_3d import (
    transport_and_rollout_3d,
    evaluate_generalization_3d,
)
from gpt_repro.utils.seeding import set_global_seed


# ---------------------------------------------------------------------------
# 1. ReshelvingEnv reset obs shape
# ---------------------------------------------------------------------------

def test_reshelving_env_reset_obs_shape():
    """reset() must return observation of shape (3,)."""
    set_global_seed(0)
    env = ReshelvingEnv()
    obs, info = env.reset(seed=0)
    assert obs.shape == (3,), f"Expected obs shape (3,), got {obs.shape}"
    env.close()


# ---------------------------------------------------------------------------
# 2. ReshelvingEnv step with zero action
# ---------------------------------------------------------------------------

def test_reshelving_env_step():
    """Zero action must not move the end-effector."""
    set_global_seed(0)
    env = ReshelvingEnv()
    obs0, _ = env.reset(seed=0)
    zero_action = np.zeros(3)
    obs1, reward, terminated, truncated, info = env.step(zero_action)
    np.testing.assert_allclose(
        obs1, obs0, atol=1e-10,
        err_msg="EE moved despite zero action",
    )
    env.close()


# ---------------------------------------------------------------------------
# 3. ReshelvingEnv success detection
# ---------------------------------------------------------------------------

def test_reshelving_env_success_detection():
    """Placing EE at goal position must trigger is_success() == True."""
    set_global_seed(0)
    _, scene = make_reshelving_demo(seed=0)
    env = ReshelvingEnv(scene=scene, success_thresh=0.02)
    env.reset(seed=0)
    # Move EE directly to goal
    env.set_ee_pos(scene["goal_pose"][:3, 3])
    assert env.is_success(), "Expected is_success() True when EE is at goal"
    env.close()


# ---------------------------------------------------------------------------
# 4. ArmPoseEnv reset obs shape
# ---------------------------------------------------------------------------

def test_armpose_env_reset_obs_shape():
    """ArmPoseEnv reset() must return observation of shape (3,)."""
    set_global_seed(0)
    env = ArmPoseEnv()
    obs, info = env.reset(seed=0)
    assert obs.shape == (3,), f"Expected obs shape (3,), got {obs.shape}"
    env.close()


# ---------------------------------------------------------------------------
# 5. transport_and_rollout_3d output keys
# ---------------------------------------------------------------------------

def test_transport_rollout_output_keys():
    """transport_and_rollout_3d must return all required keys."""
    set_global_seed(0)
    required_keys = {"rollout_x", "transported_x", "success", "final_error", "transport"}
    demo, scene = make_reshelving_demo(seed=0)
    # Use a tiny demo for speed
    from gpt_repro.policies.demos_3d import make_3d_trajectory
    small_demo = make_3d_trajectory(
        start=np.array([0.3, 0.0, 0.5]),
        goal=np.array([0.0, 0.4, 0.7]),
        n_points=20,
        seed=0,
    )
    env = ReshelvingEnv(scene=scene)
    result = transport_and_rollout_3d(
        demo=small_demo,
        S=scene["S"],
        T=scene["T"],
        env=env,
        gp_n_iter=20,
        n_steps=10,
        seed=0,
    )
    env.close()
    missing = required_keys - set(result.keys())
    assert not missing, f"Missing keys in result: {missing}"
    assert result["rollout_x"].shape[1] == 3, "rollout_x should have 3 columns"


# ---------------------------------------------------------------------------
# 6. evaluate_generalization_3d success_rate in [0, 1]
# ---------------------------------------------------------------------------

def test_evaluate_generalization_returns_rate():
    """evaluate_generalization_3d must return success_rate in [0,1]."""
    set_global_seed(0)
    demo, scene = make_reshelving_demo(seed=0)
    from gpt_repro.policies.demos_3d import make_3d_trajectory
    small_demo = make_3d_trajectory(
        start=np.array([0.3, 0.0, 0.5]),
        goal=np.array([0.0, 0.4, 0.7]),
        n_points=20,
        seed=0,
    )
    result = evaluate_generalization_3d(
        base_demo=small_demo,
        base_scene=scene,
        randomize_fn=randomize_reshelving_scene,
        n_trials=3,
        seed=0,
        env_cls=ReshelvingEnv,
        gp_n_iter=20,
        n_steps=10,
    )
    rate = result["success_rate"]
    assert 0.0 <= rate <= 1.0, f"success_rate {rate} not in [0,1]"
    assert len(result["all_rollouts"]) == 3


# ---------------------------------------------------------------------------
# Phase 12 tests — render resolution, XML content, geom color update
# ---------------------------------------------------------------------------

def test_render_returns_array():
    """All 3 envs return (480, 480, 3) uint8 from render() — Phase 12."""
    set_global_seed(0)
    from gpt_repro.envs.cleaning_env import SurfaceCleaningEnv
    from gpt_repro.policies.surfaces_3d import SurfaceConfig

    envs = [
        ReshelvingEnv(render_mode="rgb_array"),
        ArmPoseEnv(render_mode="rgb_array"),
        SurfaceCleaningEnv(
            SurfaceConfig(kind="flat", center=np.array([0.5, 0.0, 0.5])),
            render_mode="rgb_array",
        ),
    ]
    for env in envs:
        env.reset(seed=0)
        frame = env.render()
        assert frame.shape == (480, 480, 3), (
            f"{type(env).__name__}: expected (480,480,3), got {frame.shape}"
        )
        assert frame.dtype == np.uint8, f"{type(env).__name__}: expected uint8"
        env.close()


def test_reshelving_xml_has_shelf_geom():
    """ReshelvingEnv XML must contain 'shelf' geometry — Phase 12."""
    env = ReshelvingEnv()
    assert "shelf" in env._xml_string, "Expected 'shelf' geom in reshelving XML"
    env.close()


def test_armpose_xml_has_keypoint_spheres():
    """ArmPoseEnv XML must contain 'shoulder' and 'elbow' — Phase 12."""
    env = ArmPoseEnv()
    assert "shoulder" in env._xml_string, "Expected 'shoulder' in armpose XML"
    assert "elbow" in env._xml_string, "Expected 'elbow' in armpose XML"
    env.close()


def test_cleaning_xml_has_point_cloud():
    """SurfaceCleaningEnv XML must have >= 50 sphere geoms — Phase 12."""
    from gpt_repro.envs.cleaning_env import SurfaceCleaningEnv
    from gpt_repro.policies.surfaces_3d import SurfaceConfig

    env = SurfaceCleaningEnv(
        SurfaceConfig(kind="flat", center=np.array([0.5, 0.0, 0.5])),
        n_surface_pts=200,
    )
    # Count sphere geom entries
    count = env._xml_string.count('type="sphere"')
    assert count >= 50, f"Expected >=50 sphere geoms, got {count}"
    env.close()


def test_ee_color_update():
    """Updating model.geom_rgba in SurfaceCleaningEnv must not raise — Phase 12."""
    import mujoco
    from gpt_repro.envs.cleaning_env import SurfaceCleaningEnv
    from gpt_repro.policies.surfaces_3d import SurfaceConfig

    env = SurfaceCleaningEnv(
        SurfaceConfig(kind="flat", center=np.array([0.5, 0.0, 0.5])),
        render_mode="rgb_array",
    )
    env.reset(seed=0)
    ee_id = mujoco.mj_name2id(env._model, mujoco.mjtObj.mjOBJ_GEOM, "ee_geom")
    # Change EE color to red (simulating high force)
    env._model.geom_rgba[ee_id] = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)
    # Should not raise
    _ = env.render()
    env.close()
