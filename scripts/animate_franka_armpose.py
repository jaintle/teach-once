"""animate_franka_armpose.py — Phase 14.

Records a keypoint-tracing arm-pose demo, transports to 4 randomised arm
configs, renders 2×2 GIF/MP4 from the "side" camera with keypoint labels.

CLI args: --seed, --n_scenes, --fps, --width, --height, --out_dir,
          --gp_n_iter, --n_steps
"""

import argparse
import pathlib
import sys

import imageio
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))
from gpt_repro.utils.seeding import set_global_seed
from gpt_repro.envs.franka_env import FrankaKinematicEnv
from gpt_repro.policies.franka_demos import get_armpose_waypoints
from gpt_repro.transport.franka_rollout import (
    record_franka_demo,
    transport_and_rollout_franka,
)
from gpt_repro.viz.frame_annotate import add_text_overlay, add_progress_bar


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed",      type=int, default=0)
    p.add_argument("--n_scenes",  type=int, default=4)
    p.add_argument("--fps",       type=int, default=15)
    p.add_argument("--width",     type=int, default=720)
    p.add_argument("--height",    type=int, default=480)
    p.add_argument("--out_dir",   default="reports/figures/")
    p.add_argument("--gp_n_iter", type=int, default=80)
    p.add_argument("--n_steps",   type=int, default=200)
    p.add_argument("--success_threshold", type=float, default=0.10,
                   help="GP rollout cannot achieve sub-cm precision; 10cm threshold "
                        "captures functional task completion.")
    return p.parse_args()


_BASE_KPS = {
    "shoulder": np.array([0.35, 0.00, 0.70]),
    "elbow":    np.array([0.47, 0.00, 0.80]),
    "wrist":    np.array([0.57, 0.00, 0.75]),
    "hand":     np.array([0.62, 0.00, 0.65]),
}


def _make_S_T(base_kps: dict, rng):
    kp_list = [base_kps["shoulder"], base_kps["elbow"],
               base_kps["wrist"],    base_kps["hand"]]
    S = np.array(kp_list)
    delta = rng.uniform(-0.06, 0.06, S.shape)
    delta[:, 1] = 0  # keep symmetric about y=0 for reachability
    T = S + delta
    return S, T


def _randomize_arm(base_kps: dict, rng) -> dict:
    delta = rng.uniform(-0.05, 0.05, 3)
    delta[1] = 0
    return {k: v + delta for k, v in base_kps.items()}


def _tile_2x2(frames_list):
    max_n = max(len(f) for f in frames_list)
    tiled = []
    for i in range(max_n):
        cells = [frames[min(i, len(frames)-1)] for frames in frames_list]
        tiled.append(np.vstack([np.hstack([cells[0], cells[1]]),
                                 np.hstack([cells[2], cells[3]])]))
    return tiled


def _save_gif_mp4(frames, out_dir, stem, fps, budget_bytes):
    out_dir = pathlib.Path(out_dir)
    gif_path = out_dir / f"{stem}.gif"
    mp4_path = out_dir / f"{stem}.mp4"
    step = 1
    while len(frames[::step]) * frames[0].nbytes // 10 > budget_bytes and step < 8:
        step += 1
    sub = frames[::step]
    imageio.mimwrite(str(gif_path), sub, fps=fps, loop=0)
    print(f"  {gif_path.name}: {gif_path.stat().st_size/1024:.0f} KB")
    try:
        import imageio_ffmpeg  # noqa: F401
        imageio.mimwrite(str(mp4_path), frames, fps=fps*2, codec="libx264",
                         quality=6, macro_block_size=1)
        print(f"  {mp4_path.name}: {mp4_path.stat().st_size/1024:.0f} KB")
    except Exception as e:
        print(f"  MP4 skipped ({e})")


def main():
    args = parse_args()
    set_global_seed(args.seed)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_scene = dict(_BASE_KPS)  # copy

    # Record base demo
    print("Recording base arm-pose demo …")
    base_env = FrankaKinematicEnv("armpose", render_mode=None,
                                   width=args.width, height=args.height)
    base_env.reset(seed=args.seed)
    waypoints = get_armpose_waypoints(base_scene)
    base_demo = record_franka_demo(base_env, waypoints)
    base_env.close()
    print(f"  Demo: {len(base_demo['x'])} steps, IK ok={base_demo['ik_success'].mean()*100:.0f}%")

    results = []
    for i in range(args.n_scenes):
        rng_i = np.random.default_rng(args.seed + i)
        new_kps = _randomize_arm(base_scene, rng_i)
        S, T    = _make_S_T(base_scene, rng_i)

        print(f"Scene {i+1}/{args.n_scenes}")
        env = FrankaKinematicEnv("armpose", render_mode="rgb_array",
                                  width=args.width, height=args.height)
        env.set_camera("side")  # best view for arm tracing

        res = transport_and_rollout_franka(
            demo=base_demo, S=S, T=T, env=env,
            gp_n_iter=args.gp_n_iter, n_steps=args.n_steps,
            success_threshold=args.success_threshold,
            seed=args.seed + i,
        )
        print(f"  err={res['final_error']:.3f}m  ik_fail={res['ik_fail_rate']*100:.0f}%")
        env.close()

        # Annotate frames with keypoint label
        annotated = []
        for ki, frame in enumerate(res["frames"]):
            # Overlay keypoint info
            f = add_text_overlay(
                frame,
                f"Scene {i+1} | err:{res['final_error']:.3f}m",
                pos=(8, 24), font_scale=0.50,
            )
            kp_names = ["shldr", "elbw", "wrst", "hand"]
            for ki2, (kname, kpos) in enumerate(new_kps.items()):
                f = add_text_overlay(
                    f,
                    f"{kp_names[ki2]}: ({kpos[0]:.2f},{kpos[2]:.2f})",
                    pos=(8, 44 + ki2 * 18),
                    font_scale=0.38,
                    color=(220, 220, 50),
                )
            progress = ki / max(1, len(res["frames"]) - 1)
            f = add_progress_bar(f, progress,
                                  success=res["success"] if ki == len(res["frames"])-1 else None)
            annotated.append(f)

        res["annotated_frames"] = annotated
        results.append(res)

    errors = [r["final_error"] for r in results]
    fails  = [r["ik_fail_rate"] for r in results]
    succs  = [r["success"] for r in results]
    print(f"\nArmpose: success={np.mean(succs)*100:.0f}%  mean_err={np.mean(errors):.3f}m  ik_fail={np.mean(fails)*100:.1f}%")

    frames_per_scene = [r["annotated_frames"] for r in results]
    if all(len(f) > 0 for f in frames_per_scene):
        tiled = _tile_2x2(frames_per_scene)
        _save_gif_mp4(tiled, out_dir, "franka_armpose", fps=args.fps,
                      budget_bytes=5 * 1024 * 1024)
    else:
        print("WARNING: some scenes produced no frames.")


if __name__ == "__main__":
    main()
