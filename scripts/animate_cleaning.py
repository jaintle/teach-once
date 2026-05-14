"""
Animate surface-cleaning policy-transport rollouts (Phase 12).

Generates a 1×5 tiled GIF (and MP4 if imageio-ffmpeg is installed) showing
GP-transported cleaning rollouts across 5 surface variants, with force
colour-coding: cool blue = low force → warm red = high force.

CLI
---
  python scripts/animate_cleaning.py [--seed 0] [--fps 12]
                                      [--out_dir reports/figures/]
                                      [--n_pts 400] [--n_steps 150]
                                      [--fast]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import imageio  # noqa: E402
import mujoco  # noqa: E402

from gpt_repro.utils.seeding import set_global_seed  # noqa: E402
from gpt_repro.envs.cleaning_env import SurfaceCleaningEnv  # noqa: E402
from gpt_repro.policies.surfaces_3d import SurfaceConfig  # noqa: E402
from gpt_repro.transport.cleaning_pipeline_3d import (  # noqa: E402
    run_cleaning_pipeline,
)


# ── colour helpers ────────────────────────────────────────────────────────────

_COOL = np.array([0.20, 0.50, 0.90, 1.0], dtype=np.float32)   # blue  (low force)
_WARM = np.array([0.90, 0.20, 0.10, 1.0], dtype=np.float32)   # red   (high force)


def _force_color(t: float) -> np.ndarray:
    """Interpolate between cool (t=0) and warm (t=1)."""
    t = float(np.clip(t, 0.0, 1.0))
    return (1.0 - t) * _COOL + t * _WARM


# ─────────────────────────────────────────────────────────────────────────────

def _collect_surface_frames(
    target_config: SurfaceConfig,
    rollout_x: np.ndarray,
    force_norms: np.ndarray,
    n_steps: int,
    seed: int,
) -> list[np.ndarray]:
    """Render frames for one surface; colour-code EE by force norm."""
    env = SurfaceCleaningEnv(
        surface_config=target_config,
        render_mode="rgb_array",
        n_surface_pts=200,
    )
    env.reset(seed=seed, options={"init_pos": rollout_x[0]})

    # Lookup EE geom id
    ee_id = mujoco.mj_name2id(env._model, mujoco.mjtObj.mjOBJ_GEOM, "ee_geom")

    # Normalise force norms for colour mapping
    fn = np.asarray(force_norms, dtype=float)
    fn_max = fn.max() if fn.max() > 1e-8 else 1.0
    fn_norm = fn / fn_max

    frames: list[np.ndarray] = []
    for step_idx, pos in enumerate(rollout_x[:n_steps]):
        env.set_ee_pos(pos)          # directly place EE at waypoint

        # Update EE colour from force norm
        t = float(fn_norm[min(step_idx, len(fn_norm) - 1)])
        env._model.geom_rgba[ee_id] = _force_color(t)

        frame = env.render()
        if frame is None:
            frame = np.ones((480, 480, 3), dtype=np.uint8) * 128
        frames.append(frame)

    env.close()
    return frames


def _tile_1x5(frames_per_surface: list[list[np.ndarray]]) -> list[np.ndarray]:
    """Concatenate 5 frame sequences as a horizontal 1×5 strip."""
    n = max(len(f) for f in frames_per_surface)
    tiled = []
    for i in range(n):
        cells = [f[min(i, len(f) - 1)] for f in frames_per_surface]
        tiled.append(np.concatenate(cells, axis=1))
    return tiled


def main() -> None:
    parser = argparse.ArgumentParser(description="Animate surface cleaning rollouts")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--out_dir", type=str, default="reports/figures/")
    parser.add_argument("--n_pts", type=int, default=400,
                        help="Surface point cloud size")
    parser.add_argument("--n_steps", type=int, default=150)
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast mode: n_pts=50, n_inducing=10, n_steps=30 (for smoke test)",
    )
    args = parser.parse_args()

    if args.fast:
        args.n_pts = 50
        n_inducing = 10
        n_iter = 50
        gp_n_iter = 20
        args.n_steps = 30
    else:
        n_inducing = 100
        n_iter = 300
        gp_n_iter = 100

    set_global_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 5 surface variants ────────────────────────────────────────────────────
    source_config = SurfaceConfig(
        kind="flat", center=np.array([0.5, 0.0, 0.5])
    )
    target_configs = [
        SurfaceConfig(kind="flat",   center=np.array([0.5, 0.0, 0.5])),
        SurfaceConfig(kind="tilted", center=np.array([0.5, 0.0, 0.5]),
                      normal=np.array([0.2, 0.0, 1.0])),
        SurfaceConfig(kind="curved", center=np.array([0.5, 0.0, 0.5])),
        SurfaceConfig(kind="bumpy",  center=np.array([0.5, 0.0, 0.5])),
        SurfaceConfig(kind="tilted", center=np.array([0.5, 0.0, 0.5]),
                      normal=np.array([-0.15, 0.1, 1.0])),
    ]
    surface_labels = ["flat", "tilted", "curved", "bumpy", "tilted2"]

    all_frames: list[list[np.ndarray]] = []
    coverages: list[float] = []

    for idx, (label, tgt_cfg) in enumerate(zip(surface_labels, target_configs)):
        print(f"  Surface {idx+1}/5: {label} ...")
        try:
            result = run_cleaning_pipeline(
                source_config=source_config,
                target_config=tgt_cfg,
                n_source_pts=args.n_pts,
                n_target_pts=args.n_pts,
                n_inducing=n_inducing,
                n_iter=n_iter,
                n_steps=args.n_steps,
                gp_n_iter=gp_n_iter,
                seed=args.seed + idx,
            )
            rollout_x = result["rollout_x"]
            force_norms = result["force_norms"]
            coverage = result.get("coverage", 0.0)
        except Exception as exc:
            print(f"    Pipeline failed ({exc}), using fallback trajectory")
            n = args.n_steps
            start = np.array([0.3, 0.0, 0.5])
            end   = np.array([0.7, 0.2, 0.5])
            rollout_x = np.linspace(start, end, n)
            force_norms = np.zeros(n)
            coverage = 0.0

        coverages.append(coverage)

        frames = _collect_surface_frames(
            target_config=tgt_cfg,
            rollout_x=rollout_x,
            force_norms=force_norms,
            n_steps=args.n_steps,
            seed=args.seed,
        )
        all_frames.append(frames)
        print(f"    {len(frames)} frames, coverage={coverage:.2f}")

    # ── Tile 1×5 ─────────────────────────────────────────────────────────────
    tile_frames = _tile_1x5(all_frames)
    gif_frames = tile_frames[::3][:60]

    print(f"Total tiled frames: {len(gif_frames)}, "
          f"frame size: {gif_frames[0].shape}")

    # ── Save GIF ──────────────────────────────────────────────────────────────
    gif_path = out_dir / "cleaning_rollout.gif"
    imageio.mimsave(str(gif_path), gif_frames, fps=args.fps)
    gif_size_kb = gif_path.stat().st_size / 1024
    print(f"GIF saved: {gif_path}  ({gif_size_kb:.0f} KB)")

    if gif_size_kb > 8192:
        print("GIF > 8MB — downscaling (half resolution) ...")
        small = [f[::2, ::2] for f in gif_frames]
        imageio.mimsave(str(gif_path), small, fps=args.fps)
        gif_size_kb = gif_path.stat().st_size / 1024
        print(f"Downscaled GIF: {gif_size_kb:.0f} KB")

    # ── Save MP4 (optional) ───────────────────────────────────────────────────
    mp4_path = out_dir / "cleaning_rollout.mp4"
    try:
        writer = imageio.get_writer(str(mp4_path), fps=args.fps)
        for f in tile_frames:
            writer.append_data(f)
        writer.close()
        print(f"MP4 saved: {mp4_path}")
    except Exception as e:
        print(f"MP4 skipped ({type(e).__name__}: {e})")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\nCoverage per surface:")
    for label, cov in zip(surface_labels, coverages):
        print(f"  {label:12s}: {cov:.3f}")
    print(f"\nGIF: {gif_path}  ({gif_size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
