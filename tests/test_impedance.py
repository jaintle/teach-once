"""Tests for Phase 16: FrankaImpedanceEnv and impedance rollout.

6 tests:
1. test_impedance_env_obs_shape
2. test_impedance_env_gravity_compensation
3. test_impedance_env_tracks_nearby_target
4. test_impedance_torques_within_limits
5. test_impedance_rollout_output_keys
6. test_stiffness_transport_in_rollout
"""

import numpy as np
import pytest

from gpt_repro.envs.franka_impedance_env import FrankaImpedanceEnv
from gpt_repro.transport.impedance_rollout import (
    transport_and_rollout_impedance,
    get_transported_stiffness,
)
from gpt_repro.transport.policy_transport import PolicyTransport
from gpt_repro.gp.exact_gp import ExactGPRegressor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def env():
    """Shared impedance env (reshelving, no renderer) — module scope for speed."""
    e = FrankaImpedanceEnv(task="reshelving", render_mode=None, dt=0.002, control_hz=500)
    yield e
    e.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFrankaImpedanceEnv:
    def test_impedance_env_obs_shape(self, env):
        """Observation should have shape (20,): ee_pos(3)+ee_vel(3)+q(7)+dq(7)."""
        obs, info = env.reset(seed=0)
        assert obs.shape == (20,), f"Expected (20,), got {obs.shape}"
        assert not np.any(np.isnan(obs)), "Obs has NaN after reset"

    def test_impedance_env_gravity_compensation(self, env):
        """Arm should stay within 0.10 m of home for 500 steps."""
        obs, _ = env.reset(seed=0)
        x_home = obs[:3].copy()
        K_s = np.diag([400.0, 400.0, 400.0])
        diag_k = np.diag(K_s)

        max_drift = 0.0
        for _ in range(500):
            action = np.concatenate([x_home, np.zeros(3), diag_k])
            obs, _, terminated, _, info = env.step(action)
            if terminated:
                break
            drift = info["tracking_error"]
            max_drift = max(max_drift, drift)

        assert max_drift < 0.10, (
            f"Max position drift under gravity comp = {max_drift:.4f} m > 0.10 m"
        )

    def test_impedance_env_tracks_nearby_target(self, env):
        """Arm should approach a target ~0.06 m away within 0.10 m after 300 steps."""
        obs, _ = env.reset(seed=1)
        x_home = obs[:3].copy()
        x_des = x_home + np.array([0.05, 0.0, 0.03])

        K_s = np.diag([400.0, 400.0, 400.0])
        diag_k = np.diag(K_s)

        x_cur = x_home.copy()
        for _ in range(300):
            action = np.concatenate([x_des, np.zeros(3), diag_k])
            obs, _, terminated, _, _ = env.step(action)
            if terminated:
                break
            x_cur = obs[:3].copy()

        final_error = float(np.linalg.norm(x_cur - x_des))
        assert final_error < 0.10, (
            f"Final tracking error {final_error:.4f} m > 0.10 m"
        )

    def test_impedance_torques_within_limits(self, env):
        """Computed impedance torques must all be within ±87 Nm."""
        obs, _ = env.reset(seed=2)
        x_home = obs[:3].copy()
        # Extreme stiffness and large error → should still clip to 87 Nm
        x_des = x_home + np.array([0.3, 0.2, 0.1])
        K_s = np.diag([500.0, 500.0, 500.0])
        D = 2.0 * np.sqrt(K_s)

        tau = env._impedance_torques(x_des, np.zeros(3), K_s, D)
        assert tau.shape == (7,), f"Expected tau shape (7,), got {tau.shape}"
        assert np.all(np.abs(tau) <= 87.0 + 1e-6), (
            f"Torque exceeded limit: max |τ| = {np.max(np.abs(tau)):.2f} Nm"
        )


class TestImpedanceRollout:
    def test_impedance_rollout_output_keys(self):
        """transport_and_rollout_impedance should return expected keys."""
        # Minimal demo
        np.random.seed(42)
        N = 10
        x_demo = np.linspace([0.35, 0.0, 0.65], [0.55, 0.0, 0.65], N)
        xdot_demo = np.diff(x_demo, axis=0, prepend=x_demo[:1])
        demo = {"x": x_demo, "xdot": xdot_demo}

        # Use 4 points — 3D transport requires N >= d = 3
        S = np.array([
            [0.35, 0.0, 0.65],
            [0.55, 0.0, 0.65],
            [0.45, 0.1, 0.70],
            [0.40, -0.1, 0.60],
        ])
        T = S + np.array([[0.0, 0.1, 0.0]] * 4)

        env = FrankaImpedanceEnv(task="reshelving", render_mode=None, dt=0.002, control_hz=500)
        result = transport_and_rollout_impedance(
            demo, S, T, env,
            gp_cls=ExactGPRegressor,
            K_s_default=np.diag([200.0, 200.0, 200.0]),
            n_steps=20,
            gp_n_iter=30,
            seed=0,
        )
        env.close()

        required_keys = {
            "x_transported", "x_rollout", "x_des_traj",
            "final_error", "success", "K_s_used", "D_used", "x_goal"
        }
        assert required_keys.issubset(result.keys()), (
            f"Missing keys: {required_keys - result.keys()}"
        )
        assert result["x_transported"].shape[1] == 3
        assert result["K_s_used"].shape == (3, 3)
        assert result["D_used"].shape == (3, 3)

    def test_stiffness_transport_in_rollout(self):
        """Transported K_s should be symmetric and PSD."""
        np.random.seed(42)
        S = np.array([[0.35, 0.0, 0.65], [0.55, 0.0, 0.65], [0.45, 0.1, 0.70]])
        T = S + np.array([[0.05, 0.05, 0.0], [0.05, 0.05, 0.0], [0.05, 0.05, 0.0]])

        transport = PolicyTransport(gp_cls=ExactGPRegressor, n_iter_default=30)
        transport.fit(S, T)

        K_s = np.diag([200.0, 200.0, 200.0])
        K_hat = get_transported_stiffness(transport, S, K_s)

        # Should be (3, 3)
        assert K_hat.shape == (3, 3), f"Expected (3,3), got {K_hat.shape}"

        # Should be symmetric
        sym_err = np.max(np.abs(K_hat - K_hat.T))
        assert sym_err < 1e-10, f"K_hat not symmetric: max|K-K^T| = {sym_err}"

        # Should be PSD (all eigenvalues >= 0)
        eigvals = np.linalg.eigvalsh(K_hat)
        assert np.all(eigvals >= -1e-6), (
            f"K_hat not PSD: min eigenvalue = {eigvals.min():.4f}"
        )
