"""Smoke test for Phase 13 — Franka Panda arm + IK environment.

Checks (per task):
  1. Scene XML loads (nq==9, nsite==1).
  2. env.reset() returns obs of shape (3,).
  3. env.step(target) returns obs (3,) and IK converges for a reachable target.
  4. env.render() returns (480, 720, 3) with max > 0.
  5. Camera switching works for all 3 presets.

Exits with code 0 on success, 1 on failure.
"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))
from gpt_repro.envs.franka_env import FrankaKinematicEnv
from gpt_repro.envs.franka_scene import build_scene_xml, load_scene_model, CAMERAS


def check(condition: bool, msg: str) -> None:
    if not condition:
        print(f"  FAIL: {msg}")
        sys.exit(1)
    print(f"  OK  : {msg}")


def test_scene_xml(task: str) -> None:
    xml = build_scene_xml(task)
    m, d = load_scene_model(xml)
    check(m.nq == 9,   f"[{task}] nq == 9 (got {m.nq})")
    check(m.nsite == 1, f"[{task}] nsite == 1 (got {m.nsite})")


def test_env(task: str) -> None:
    env = FrankaKinematicEnv(task, render_mode="rgb_array", width=720, height=480)

    # reset
    obs, info = env.reset(seed=0)
    check(obs.shape == (3,), f"[{task}] reset obs.shape == (3,)  got {obs.shape}")

    # step with a reachable target
    target = np.array([0.45, 0.05, 0.70])
    obs2, reward, term, trunc, info2 = env.step(target)
    check(obs2.shape == (3,), f"[{task}] step obs.shape == (3,)  got {obs2.shape}")
    check(info2["ik_success"],  f"[{task}] IK converged for target {target}")
    check(info2["ee_dist_to_target"] < 0.01,
          f"[{task}] EE dist < 10mm  (got {info2['ee_dist_to_target']*1000:.2f}mm)")

    # render
    frame = env.render()
    check(frame is not None,           f"[{task}] render() not None")
    check(frame.shape == (480, 720, 3), f"[{task}] render shape (480,720,3)  got {frame.shape}")
    check(frame.max() > 0,             f"[{task}] render max > 0")

    # camera switching
    for cam_name in CAMERAS:
        env.set_camera(cam_name)
        f = env.render()
        check(f.max() > 0, f"[{task}] camera '{cam_name}' non-black")

    env.close()


def main() -> None:
    print("=" * 60)
    print("smoke_phase13.py — Franka + IK environment")
    print("=" * 60)

    for task in ("reshelving", "cleaning", "armpose"):
        print(f"\n--- Task: {task} ---")
        test_scene_xml(task)
        test_env(task)

    # Check output directories exist
    for d in ("reports/figures", "reports/results"):
        check(pathlib.Path(d).is_dir(), f"directory exists: {d}")

    print("\n" + "=" * 60)
    print("smoke_phase13 PASSED")
    print("=" * 60)
    sys.exit(0)


if __name__ == "__main__":
    main()
