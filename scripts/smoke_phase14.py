"""Smoke test for Phase 14 — GPT-Franka rollout.

Checks:
  1. record_franka_demo returns correct shapes.
  2. transport_and_rollout_franka returns non-empty frames.
  3. Frame shape is (480, 720, 3).
  4. GIF saved for a minimal 1-scene reshelving run.
  5. IK fail rate printed.
"""

import pathlib
import sys

import imageio
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))
from gpt_repro.envs.franka_env import FrankaKinematicEnv
from gpt_repro.policies.franka_demos import get_reshelving_waypoints
from gpt_repro.transport.franka_rollout import (
    record_franka_demo,
    transport_and_rollout_franka,
)
from gpt_repro.viz.frame_annotate import add_text_overlay, add_title_bar


def check(cond, msg):
    if not cond:
        print(f"  FAIL: {msg}")
        sys.exit(1)
    print(f"  OK  : {msg}")


def main():
    print("=" * 60)
    print("smoke_phase14.py — GPT-Franka rollout")
    print("=" * 60)

    base_scene = {
        "object_pose": np.array([0.50, 0.00, 0.63]),
        "goal_pose":   np.array([0.30, 0.10, 0.75]),
    }
    waypoints = get_reshelving_waypoints(base_scene)
    check(waypoints.shape[1] == 3, f"waypoints.shape[1] == 3  got {waypoints.shape}")

    # --- record demo ---
    env = FrankaKinematicEnv("reshelving", render_mode="rgb_array", width=720, height=480)
    env.reset(seed=0)
    demo = record_franka_demo(env, waypoints, n_interp=2)
    env.close()

    N = len(demo["x"])
    check(N > 0,              "demo has steps")
    check(demo["x"].shape[1] == 3, f"demo x shape[1]==3  got {demo['x'].shape}")
    check(demo["q"].shape[1] == 7, f"demo q shape[1]==7  got {demo['q'].shape}")
    check(demo["ik_success"].shape == (N,), "ik_success shape matches")

    # --- transport + rollout (tiny) ---
    S = np.array([
        [0.50,  0.05, 0.63], [0.50, -0.05, 0.63],
        [0.30,  0.05, 0.75], [0.30, -0.05, 0.75],
    ])
    T = S + np.array([0.03, 0.03, 0.02])  # small shift

    env2 = FrankaKinematicEnv("reshelving", render_mode="rgb_array", width=720, height=480)
    res = transport_and_rollout_franka(
        demo=demo, S=S, T=T, env=env2,
        gp_n_iter=30, n_steps=20, seed=0,
    )
    env2.close()

    check(len(res["frames"]) > 0, f"frames non-empty  got {len(res['frames'])}")
    check(res["frames"][0].shape == (480, 720, 3),
          f"frame shape (480,720,3)  got {res['frames'][0].shape}")
    check(res["rollout_x"].shape[1] == 3, "rollout_x shape[1]==3")
    check(res["rollout_q"].shape[1] == 7, "rollout_q shape[1]==7")
    check(0.0 <= res["ik_fail_rate"] <= 1.0, "ik_fail_rate in [0,1]")
    print(f"  IK fail rate: {res['ik_fail_rate']*100:.1f}%")

    # --- frame annotation ---
    f0 = res["frames"][0]
    ann = add_text_overlay(f0, "test label", pos=(5, 20))
    check(ann.shape == f0.shape, "add_text_overlay preserves shape")

    titled = add_title_bar(f0, "Test Title", bar_height=32)
    check(titled.shape[0] == f0.shape[0] + 32, "add_title_bar adds bar_height rows")
    check(titled.shape[1] == f0.shape[1], "add_title_bar preserves width")

    # --- save minimal GIF ---
    out_dir = pathlib.Path("reports/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    gif_path = out_dir / "smoke14_reshelving.gif"
    imageio.mimwrite(str(gif_path), res["frames"][:10], fps=10, loop=0)
    check(gif_path.exists() and gif_path.stat().st_size > 0, f"{gif_path.name} saved")
    print(f"  {gif_path.name}: {gif_path.stat().st_size/1024:.0f} KB")

    print("\n" + "=" * 60)
    print("smoke_phase14 PASSED")
    print("=" * 60)
    sys.exit(0)


if __name__ == "__main__":
    main()
