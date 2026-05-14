"""
Animate arm-pose policy-transport rollouts (Phase 12).

Generates a 2×2 tiled GIF (and MP4 if imageio-ffmpeg is installed) showing
GP-transported rollouts across N arm-pose configurations.

CLI
---
  python scripts/animate_armpose.py [--seed 0] [--n_scenes 4] [--fps 15]
                                     [--out_dir reports/figures/]
                                     [--n_steps 80] [--fast]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import imageio  # noqa: E402

from gpt_repro.utils.seeding import set_global_seed  # noqa: E402
from gpt_repro.envs.armpose_env import ArmPoseEnv  # noqa: E402
from gpt_repro.policies.demos_3d import (  # noqa: E402
    make_armpose_demo,
    randomize_armpose_scene,
)
from gpt_repro.transport.rollout_3d import transport_and_rollout_3d  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────

def _tile_2x2(frames_per_env: list[list[np.ndarray]]) -> list[np.ndarray]:
    """Stack 4 frame sequences into 2×2 tiles, padding shorter sequences."""
    n = max(len(f) for f in frames_per_env)
    tiled = []
    for i in range(n):
        cells = [f[min(i, len(f) - 1)] for f in frames_per_env]
        row1 = np.concatenate([cells[0], cells[1]], axis=1)
        row2 = np.concatenate([cells[2], cells[3]], axis=1)
        tiled.append(np.concatenate([row1, row2], axis=0))
    return tiled


def _collect_rollout_frames(
    rollout_x: np.ndarray,
    env: ArmPoseEnv,
    n_steps: int,
) -> list[np.ndarray]:
    """Teleport EE through rollout_x and collect rendered frames."""
    frames: list[np.ndarray] = []
    env.reset(seed=0, options={"init_pos": rollout_x[0]})
    for pos in rollout_x[:n_steps]:
        env.set_ee_pos(pos)
        frame = env.render()
        if frame is None:
            frame = np.ones((480, 480, 3), dtype=np.uint8) * 128
        frames.append(frame)
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Animate arm-pose rollouts")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_scenes", type=int, default=4)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--out_dir", type=str, default="reports/figures/")
    parser.add_argument("--n_steps", type=int, default=80)
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast mode: n_steps=20, gp_n_iter=20 (for smoke test)",
    )
    args = parser.parse_args()

    if args.fast:
        args.n_steps = 20
        gp_n_iter = 20
    else:
        gp_n_iter = 100

    set_global_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Build source demo ──────────────────────────────────────────────────
    demo, source_scene = make_armpose_demo(seed=args.seed)
    print(f"Source demo shape: {demo['x'].shape}")

    # ── 2. Randomise N target scenes and transport ────────────────────────────
    scenes_needed = max(4, args.n_scenes)
    all_frames: list[list[np.ndarray]] = []
    successes = 0

    for i in range(scenes_needed):
        scene = randomize_armpose_scene(
            seed=args.seed + i + 1,
            base_scene=source_scene,
        )
        env = ArmPoseEnv(scene=scene, render_mode="rgb_array")

        try:
            result = transport_and_rollout_3d(
                demo=demo,
                S=source_scene["S"],
                T=scene["T"],
                env=env,
                gp_n_iter=gp_n_iter,
                n_steps=args.n_steps,
                seed=args.seed,
            )
            rollout_x = result["rollout_x"]
            if result["success"]:
                successes += 1
        except Exception as exc:
            print(f"  Scene {i}: transport failed ({exc}), using straight line")
            rollout_x = np.linspace(
                env.get_ee_pos().copy(),
                scene["T"].mean(axis=0),
                args.n_steps,
            )

        frames = _collect_rollout_frames(rollout_x, env, args.n_steps)
        all_frames.append(frames)
        env.close()
        print(f"  Scene {i+1}/{scenes_needed}: {len(frames)} frames")

    # 2×2 tile using first 4 scenes
    tile_frames = _tile_2x2(all_frames[:4])
    gif_frames = tile_frames[::3][:60]

    print(f"Total tiled frames: {len(gif_frames)}")

    # ── 3. Save GIF ───────────────────────────────────────────────────────────
    gif_path = out_dir / "armpose_rollout.gif"
    imageio.mimsave(str(gif_path), gif_frames, fps=args.fps)
    gif_size_kb = gif_path.stat().st_size / 1024
    print(f"GIF saved: {gif_path}  ({gif_size_kb:.0f} KB)")

    if gif_size_kb > 8192:
        print("GIF > 8MB — downscaling ...")
        small = [f[::2, ::2] for f in gif_frames]
        imageio.mimsave(str(gif_path), small, fps=args.fps)
        gif_size_kb = gif_path.stat().st_size / 1024
        print(f"Downscaled GIF: {gif_size_kb:.0f} KB")

    # ── 4. Save MP4 (optional) ────────────────────────────────────────────────
    mp4_path = out_dir / "armpose_rollout.mp4"
    try:
        writer = imageio.get_writer(str(mp4_path), fps=args.fps)
        for f in tile_frames:
            writer.append_data(f)
        writer.close()
        print(f"MP4 saved: {mp4_path}")
    except Exception as e:
        print(f"MP4 skipped ({type(e).__name__}: {e})")

    # ── 5. Summary ────────────────────────────────────────────────────────────
    print(f"\nSuccess rate: {successes}/{scenes_needed}")
    print(f"GIF: {gif_path}  ({gif_size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
