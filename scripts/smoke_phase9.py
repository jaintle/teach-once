#!/usr/bin/env python
"""Smoke test for Phase 9 (3-D extension + MuJoCo environments).

Checks that all Phase 9 components are importable and runnable end-to-end:
1. Import gymnasium / mujoco / imageio and print versions.
2. ReshelvingEnv: 5 steps, check obs shape (3,).
3. ArmPoseEnv: 5 steps, check obs shape (3,).
4. transport_and_rollout_3d on a 20-point demo.
5. Print PASS.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

# ---- 1. Dependency versions ------------------------------------------------
import mujoco
import gymnasium
import imageio

print(f"mujoco     : {mujoco.__version__}")
print(f"gymnasium  : {gymnasium.__version__}")
print(f"imageio    : {imageio.__version__}")

# ---- 2. ReshelvingEnv ------------------------------------------------------
from gpt_repro.envs.reshelving_env import ReshelvingEnv
import numpy as np

env_r = ReshelvingEnv()
obs, _ = env_r.reset(seed=0)
assert obs.shape == (3,), f"ReshelvingEnv reset obs shape: expected (3,), got {obs.shape}"
for _ in range(5):
    obs, _, _, _, _ = env_r.step(np.zeros(3))
assert obs.shape == (3,), f"ReshelvingEnv step obs shape: expected (3,), got {obs.shape}"
env_r.close()
print("ReshelvingEnv: OK")

# ---- 3. ArmPoseEnv ---------------------------------------------------------
from gpt_repro.envs.armpose_env import ArmPoseEnv

env_a = ArmPoseEnv()
obs, _ = env_a.reset(seed=0)
assert obs.shape == (3,), f"ArmPoseEnv reset obs shape: expected (3,), got {obs.shape}"
for _ in range(5):
    obs, _, _, _, _ = env_a.step(np.zeros(3))
assert obs.shape == (3,), f"ArmPoseEnv step obs shape: expected (3,), got {obs.shape}"
env_a.close()
print("ArmPoseEnv: OK")

# ---- 4. transport_and_rollout_3d -------------------------------------------
from gpt_repro.policies.demos_3d import make_3d_trajectory, make_reshelving_demo
from gpt_repro.transport.rollout_3d import transport_and_rollout_3d

demo = make_3d_trajectory(
    start=np.array([0.3, 0.0, 0.5]),
    goal=np.array([0.0, 0.4, 0.7]),
    n_points=20,
    seed=0,
)
_, scene = make_reshelving_demo(seed=0)
env_roll = ReshelvingEnv(scene=scene)
result = transport_and_rollout_3d(
    demo=demo,
    S=scene["S"],
    T=scene["T"],
    env=env_roll,
    gp_n_iter=30,
    n_steps=20,
    seed=0,
)
env_roll.close()

assert "rollout_x" in result
assert result["rollout_x"].shape == (21, 3), (
    f"Expected rollout_x (21,3), got {result['rollout_x'].shape}"
)
assert isinstance(result["success"], bool)
assert isinstance(result["final_error"], float)
print(f"transport_and_rollout_3d: OK  (final_error={result['final_error']:.4f} m)")

# ---- PASS ------------------------------------------------------------------
print("\nPhase 9 smoke test: PASS")
