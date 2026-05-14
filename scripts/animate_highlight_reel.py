"""animate_highlight_reel.py — Phase 14.

Single-run showcase: loads best-trial frames from each of the three tasks
(reshelving/cleaning/armpose), stacks them vertically with title bars,
and saves as a highlight_reel GIF + MP4.

Generates frames independently (does not load cached files) so the script
is self-contained.

CLI args: --seed, --fps (default 10), --out_dir, --gp_n_iter, --n_steps
"""

import argparse
import pathlib
import sys

import imageio
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))
from gpt_repro.utils.seeding import set_global_seed
from gpt_repro.envs.franka_env import FrankaKinematicEnv
from gpt_repro.policies.franka_demos import (
    get_reshelving_waypoints,
    get_cleaning_waypoints,
    get_armpose_waypoints,
)
from gpt_repro.transport.franka_rollout import (
    record_franka_demo,
    transport_and_rollout_franka,
)
from gpt_repro.viz.frame_annotate import add_title_bar

W, H = 720, 480


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed",      type=int, default=0)
    p.add_argument("--fps",       type=int, default=10)
    p.add_argument("--out_dir",   default="reports/figures/")
    p.add_argument("--gp_n_iter", type=int, default=80)
    p.add_argument("--n_steps",   type=int, default=200)
    return p.parse_args()


# ---- Scene configs for each task ----------------------------------------

def _reshelving_config():
    base_scene = {
        "object_pose": np.array([0.50, 0.00, 0.63]),
        "goal_pose":   np.array([0.30, 0.10, 0.75]),
    }
    rng = np.random.default_rng(42)
    obj_p  = base_scene["object_pose"] + rng.uniform(-0.04, 0.04, 3) * [1,1,0]
    goal_p = base_scene["goal_pose"]   + rng.uniform(-0.04, 0.04, 3)
    new_scene = {"object_pose": obj_p, "goal_pose": goal_p}
    S = np.array([
        base_scene["object_pose"] + [ 0.05,  0.05, 0],
        base_scene["object_pose"] + [-0.05,  0.05, 0],
        base_scene["goal_pose"]   + [ 0.05,  0.05, 0],
        base_scene["goal_pose"]   + [-0.05,  0.05, 0],
    ])
    T = S + rng.uniform(-0.04, 0.04, S.shape) * [1,1,0.5]
    return base_scene, new_scene, S, T, get_reshelving_waypoints(base_scene), "front"


def _cleaning_config():
    base_scene = {
        "surface_center":    np.array([0.50, 0.00, 0.64]),
        "surface_half_size": np.array([0.12, 0.12]),
    }
    rng = np.random.default_rng(43)
    shift = rng.uniform(-0.04, 0.04, 3) * [1,1,0.3]
    new_scene = {
        "surface_center":    base_scene["surface_center"]    + shift,
        "surface_half_size": base_scene["surface_half_size"] * rng.uniform(0.85,1.15,2),
    }
    c = base_scene["surface_center"]
    h = base_scene["surface_half_size"]
    S = np.array([
        [c[0]-h[0], c[1]-h[1], c[2]],
        [c[0]+h[0], c[1]-h[1], c[2]],
        [c[0]-h[0], c[1]+h[1], c[2]],
        [c[0]+h[0], c[1]+h[1], c[2]],
    ])
    T = S + shift
    return base_scene, new_scene, S, T, get_cleaning_waypoints(base_scene), "side"


def _armpose_config():
    base_kps = {
        "shoulder": np.array([0.35, 0.00, 0.70]),
        "elbow":    np.array([0.47, 0.00, 0.80]),
        "wrist":    np.array([0.57, 0.00, 0.75]),
        "hand":     np.array([0.62, 0.00, 0.65]),
    }
    rng = np.random.default_rng(44)
    delta = rng.uniform(-0.04, 0.04, 3) * [1,0,1]
    new_kps = {k: v + delta for k, v in base_kps.items()}
    S = np.array(list(base_kps.values()))
    T = S + delta
    return base_kps, new_kps, S, T, get_armpose_waypoints(base_kps), "side"


# ---- Helpers ---------------------------------------------------------------

def _run_task(task, base_scene, new_scene, S, T, waypoints, cam_name, args):
    base_env = FrankaKinematicEnv(task, render_mode=None, width=W, height=H)
    base_env.reset(seed=args.seed)
    demo = record_franka_demo(base_env, waypoints)
    base_env.close()

    env = FrankaKinematicEnv(task, render_mode="rgb_array", width=W, height=H)
    env.set_camera(cam_name)
    res = transport_and_rollout_franka(
        demo=demo, S=S, T=T, env=env,
        gp_n_iter=args.gp_n_iter, n_steps=args.n_steps,
        seed=args.seed,
    )
    env.close()
    return res


def _pad_frames(frames, target_len):
    if len(frames) >= target_len:
        return frames[:target_len]
    return frames + [frames[-1]] * (target_len - len(frames))


def main():
    args = parse_args()
    set_global_seed(args.seed)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        ("reshelving", *_reshelving_config(),
         "Reshelving — GPT generalizes to new object/goal positions"),
        ("cleaning",   *_cleaning_config(),
         "Cleaning — GPT generalizes to shifted surfaces"),
        ("armpose",    *_armpose_config(),
         "Arm-pose — GPT generalizes to new arm configurations"),
    ]

    all_frames = []
    all_titles = []

    for task_name, base_scene, new_scene, S, T, waypoints, cam_name, title in tasks:
        print(f"\n--- {task_name} ---")
        res = _run_task(task_name, base_scene, new_scene, S, T, waypoints, cam_name, args)
        print(f"  err={res['final_error']:.3f}m  ik_fail={res['ik_fail_rate']*100:.0f}%")
        all_frames.append(res["frames"])
        all_titles.append(title)

    # Pad all sequences to same length
    max_n = max(len(f) for f in all_frames if len(f) > 0)
    if max_n == 0:
        print("No frames produced — exiting.")
        return

    for i in range(len(all_frames)):
        if len(all_frames[i]) == 0:
            all_frames[i] = [np.zeros((H, W, 3), dtype=np.uint8)]
        all_frames[i] = _pad_frames(all_frames[i], max_n)

    # Composite: for each timestep, stack 3 rows with title bars
    combined = []
    for fi in range(max_n):
        rows = []
        for task_idx, title in enumerate(all_titles):
            frame = all_frames[task_idx][fi]
            row   = add_title_bar(frame, title, bar_height=32)
            rows.append(row)
        combined.append(np.vstack(rows))

    # Save (dynamic subsampling to stay under 15MB budget)
    gif_path = out_dir / "highlight_reel.gif"
    mp4_path = out_dir / "highlight_reel.mp4"

    step = 1
    while True:
        sub = combined[::step]
        est = len(sub) * combined[0].nbytes // 10
        if est <= 15 * 1024 * 1024 or step >= 8:
            break
        step += 1
    if step > 1:
        print(f"  GIF subsampled every {step} frames ({len(sub)} frames)")
    imageio.mimwrite(str(gif_path), sub, fps=args.fps, loop=0)
    sz = gif_path.stat().st_size
    print(f"\n{gif_path.name}: {sz/1024/1024:.1f} MB")
    if sz > 15 * 1024 * 1024:
        print("  WARNING: GIF exceeds 15MB budget.")

    try:
        import imageio_ffmpeg  # noqa: F401
        imageio.mimwrite(str(mp4_path), combined, fps=args.fps*2, codec="libx264",
                         quality=6, macro_block_size=1)
        print(f"{mp4_path.name}: {mp4_path.stat().st_size/1024:.0f} KB")
    except Exception as e:
        print(f"MP4 skipped ({e})")


if __name__ == "__main__":
    main()
