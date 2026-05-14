"""Tests for Phase 13 — Franka Panda arm environment + IK solver.

11 tests covering:
  1.  Scene XML builds without error for all 3 tasks.
  2.  Model has correct nq (9), nsite (1).
  3.  env.reset() returns (3,) observation.
  4.  IK converges for a reachable target.
  5.  IK position error < 5mm for reachable target.
  6.  env.step() returns (3,) obs, float reward, bool flags.
  7.  Joint limits are respected after IK solve.
  8.  Camera switching works for all presets.
  9.  render() returns (480, 720, 3) non-black frame.
  10. IK identity test: source==target → solution at source.
  11. interpolate_joint_trajectory returns correct shape & endpoints.
"""

import numpy as np
import pytest

from gpt_repro.envs.franka_scene import build_scene_xml, load_scene_model, CAMERAS
from gpt_repro.envs.franka_env import FrankaKinematicEnv, Q_HOME
from gpt_repro.envs.ik_solver import IKSolver, interpolate_joint_trajectory


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def reshelving_env():
    env = FrankaKinematicEnv("reshelving", render_mode="rgb_array")
    env.reset(seed=0)
    yield env
    env.close()


@pytest.fixture(scope="module")
def ik_solver(reshelving_env):
    return reshelving_env._ik


# ---------------------------------------------------------------------------
# 1. Scene XML builds for all tasks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("task", ["reshelving", "cleaning", "armpose"])
def test_build_scene_xml(task):
    xml = build_scene_xml(task)
    assert isinstance(xml, str)
    assert len(xml) > 100
    assert "panda_with_site.xml" in xml


# ---------------------------------------------------------------------------
# 2. Model nq and nsite
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("task", ["reshelving", "cleaning", "armpose"])
def test_model_nq_nsite(task):
    xml = build_scene_xml(task)
    m, _ = load_scene_model(xml)
    assert m.nq == 9, f"Expected nq=9, got {m.nq}"
    assert m.nsite == 1, f"Expected nsite=1, got {m.nsite}"


# ---------------------------------------------------------------------------
# 3. reset() returns (3,) observation
# ---------------------------------------------------------------------------

def test_reset_obs_shape(reshelving_env):
    obs, info = reshelving_env.reset(seed=42)
    assert obs.shape == (3,), f"Expected (3,), got {obs.shape}"


# ---------------------------------------------------------------------------
# 4. IK converges for a reachable target
# ---------------------------------------------------------------------------

def test_ik_converges(reshelving_env):
    target = np.array([0.45, 0.0, 0.65])
    reshelving_env.reset(seed=0)
    success = reshelving_env.set_ee_pos(target)
    assert success, f"IK did not converge for target {target}"


# ---------------------------------------------------------------------------
# 5. IK position error < 5mm
# ---------------------------------------------------------------------------

def test_ik_position_error(reshelving_env):
    target = np.array([0.40, 0.1, 0.70])
    reshelving_env.reset(seed=0)
    reshelving_env.set_ee_pos(target)
    ee = reshelving_env.get_ee_pos()
    err = float(np.linalg.norm(ee - target))
    assert err < 0.005, f"IK position error {err:.4f}m >= 5mm"


# ---------------------------------------------------------------------------
# 6. step() return types
# ---------------------------------------------------------------------------

def test_step_return_types(reshelving_env):
    reshelving_env.reset(seed=0)
    obs, reward, term, trunc, info = reshelving_env.step([0.4, 0.0, 0.6])
    assert obs.shape == (3,)
    assert isinstance(reward, float)
    assert isinstance(term, bool)
    assert isinstance(trunc, bool)
    assert "ik_success" in info
    assert "ee_dist_to_target" in info


# ---------------------------------------------------------------------------
# 7. Joint limits respected
# ---------------------------------------------------------------------------

def test_joint_limits(reshelving_env, ik_solver):
    q_lo, q_hi = ik_solver.get_joint_limits()
    target = np.array([0.50, -0.20, 0.80])
    reshelving_env.reset(seed=0)
    reshelving_env.set_ee_pos(target)
    q_sol = reshelving_env._data.qpos[:7].copy()
    assert np.all(q_sol >= q_lo - 1e-6), f"Joint below lower limit: {q_sol}"
    assert np.all(q_sol <= q_hi + 1e-6), f"Joint above upper limit: {q_sol}"


# ---------------------------------------------------------------------------
# 8. Camera switching
# ---------------------------------------------------------------------------

def test_camera_switching(reshelving_env):
    for cam_name in CAMERAS:
        reshelving_env.set_camera(cam_name)
        # No exception = OK
    with pytest.raises(ValueError):
        reshelving_env.set_camera("nonexistent_camera")


# ---------------------------------------------------------------------------
# 9. render() returns non-black frame
# ---------------------------------------------------------------------------

def test_render_non_black(reshelving_env):
    reshelving_env.reset(seed=0)
    reshelving_env.set_camera("front")
    frame = reshelving_env.render()
    assert frame is not None
    assert frame.shape == (480, 720, 3), f"Shape mismatch: {frame.shape}"
    assert frame.max() > 0, "Render produced all-black frame"


# ---------------------------------------------------------------------------
# 10. IK identity: warm-start at home, target = current home → zero error
# ---------------------------------------------------------------------------

def test_ik_identity(reshelving_env, ik_solver):
    """Solving IK from home toward home should return near-zero error."""
    import mujoco
    reshelving_env.reset(seed=0)
    ee_home = reshelving_env.get_ee_pos().copy()
    q_sol, ok = ik_solver.solve(ee_home, q_init=Q_HOME[:7])
    # After solving to current position, error should be tiny
    reshelving_env._data.qpos[:7] = q_sol
    mujoco.mj_forward(reshelving_env._model, reshelving_env._data)
    ee_new = reshelving_env.get_ee_pos()
    err = float(np.linalg.norm(ee_new - ee_home))
    assert err < 0.002, f"IK identity error {err:.4f}m too large"


# ---------------------------------------------------------------------------
# 11. interpolate_joint_trajectory shape and endpoints
# ---------------------------------------------------------------------------

def test_interpolate_joint_trajectory():
    q0 = np.zeros(7)
    q1 = np.ones(7)
    traj = interpolate_joint_trajectory(q0, q1, n_interp=11)
    assert traj.shape == (11, 7), f"Expected (11,7), got {traj.shape}"
    np.testing.assert_allclose(traj[0], q0, atol=1e-9)
    np.testing.assert_allclose(traj[-1], q1, atol=1e-9)
    # Midpoint should be 0.5
    np.testing.assert_allclose(traj[5], 0.5 * np.ones(7), atol=1e-9)
