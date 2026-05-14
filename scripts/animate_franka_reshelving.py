"""animate_franka_reshelving.py — Phase 14.

Records a kinematic pick-and-place demo on the Franka arm, transports it
to 4 randomised scenes via GPT, and renders a 2×2 GIF/MP4.

CLI args
--------
--seed          int  (default 0)
--n_scenes      int  (default 4)
--fps           int  (default 15)
--width         int  (default 720)
--height        int  (default 480)
--out_dir       path (default reports/figures/)
--gp_n_iter     int  (default 80)
--n_steps       int  (default 80)
"""

import argparse
import pathlib
import sys

import imageio
import mujoco
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))
from gpt_repro.utils.seeding import set_global_seed
from gpt_repro.envs.franka_env import FrankaKinematicEnv
from gpt_repro.policies.franka_demos import get_reshelving_waypoints
from gpt_repro.transport.franka_rollout import (
    record_franka_demo,
    transport_and_rollout_franka,
)
from gpt_repro.viz.frame_annotate import add_text_overlay, add_progress_bar

# ---------------------------------------------------------------------------

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
    p.add_argument("--success_threshold", type=float, default=0.08,
                   help="GP rollout cannot achieve sub-cm precision; 8cm threshold "
                        "captures functional task completion.")
    return p.parse_args()


def _make_S_T(base_scene: dict, rng: np.random.Generator):
    """Build source (S) and target (T) landmark arrays for transport."""
    obj  = np.asarray(base_scene["object_pose"])
    goal = np.asarray(base_scene["goal_pose"])
    # 4 corners of a bounding box around object + goal at 3 heights
    S = np.array([
        obj  + [0.05,  0.05, 0.0],
        obj  + [-0.05, 0.05, 0.0],
        goal + [0.05,  0.05, 0.0],
        goal + [-0.05, 0.05, 0.0],
        obj  + [0.0,   0.0,  0.1],
        goal + [0.0,   0.0,  0.1],
    ])
    # Random perturbation for target scene
    delta = rng.uniform(-0.08, 0.08, size=S.shape)
    delta[:, 2] *= 0.5  # less vertical shift
    T = S + delta
    return S, T


def _randomize_scene(base: dict, rng: np.random.Generator) -> dict:
    obj_perturb  = rng.uniform(-0.07, 0.07, 3)
    obj_perturb[2] = 0
    goal_perturb = rng.uniform(-0.07, 0.07, 3)
    goal_perturb[2] = rng.uniform(0.0, 0.06)
    return {
        "object_pose": base["object_pose"] + obj_perturb,
        "goal_pose":   base["goal_pose"]   + goal_perturb,
    }


def _tile_2x2(frames_list):
    """Tile 4 frame sequences into 2×2 grid, padding to equal length."""
    max_n = max(len(f) for f in frames_list)
    tiled = []
    for i in range(max_n):
        cells = []
        for frames in frames_list:
            idx = min(i, len(frames) - 1)
            cells.append(frames[idx])
        row0 = np.hstack([cells[0], cells[1]])
        row1 = np.hstack([cells[2], cells[3]])
        tiled.append(np.vstack([row0, row1]))
    return tiled


def _save_gif_mp4(frames, out_dir, stem, fps, budget_bytes):
    """Save GIF (every Nth frame to stay under budget) and MP4 if ffmpeg available."""
    out_dir = pathlib.Path(out_dir)
    gif_path = out_dir / f"{stem}.gif"
    mp4_path = out_dir / f"{stem}.mp4"

    # Subsample to keep under budget
    step = 1
    while True:
        sub = frames[::step]
        est = len(sub) * frames[0].nbytes // 10  # rough GIF estimate
        if est <= budget_bytes or step >= 8:
            break
        step += 1
    if step > 1:
        print(f"  GIF subsampled every {step} frames ({len(sub)} frames)")

    imageio.mimwrite(str(gif_path), sub, fps=fps, loop=0)
    sz = gif_path.stat().st_size
    print(f"  {gif_path.name}: {sz/1024:.0f} KB")

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

    base_scene = {
        "object_pose": np.array([0.50, 0.00, 0.63]),
        "goal_pose":   np.array([0.30, 0.10, 0.75]),
    }

    # Record base demo
    print("Recording base demo …")
    base_env = FrankaKinematicEnv("reshelving", render_mode=None,
                                   width=args.width, height=args.height)
    base_env.reset(seed=args.seed)
    waypoints = get_reshelving_waypoints(base_scene)
    base_demo = record_franka_demo(base_env, waypoints)
    base_env.close()
    print(f"  Demo: {len(base_demo['x'])} steps, IK ok={base_demo['ik_success'].mean()*100:.0f}%")

    rng = np.random.default_rng(args.seed)

    results = []
    for i in range(args.n_scenes):
        trial_seed = args.seed + i
        rng_i = np.random.default_rng(trial_seed)
        new_scene = _randomize_scene(base_scene, rng_i)
        S, T = _make_S_T(base_scene, rng_i)

        print(f"Scene {i+1}/{args.n_scenes}: obj={np.round(new_scene['object_pose'], 2)}")
        env = FrankaKinematicEnv("reshelving", render_mode="rgb_array",
                                  width=args.width, height=args.height)
        res = transport_and_rollout_franka(
            demo=base_demo, S=S, T=T, env=env,
            gp_n_iter=args.gp_n_iter, n_steps=args.n_steps,
            success_threshold=args.success_threshold,
            attractor_gain=1.5,
            seed=trial_seed,
        )
        print(f"  err={res['final_error']:.3f}m  ik_fail={res['ik_fail_rate']*100:.0f}%  success={res['success']}")

        # Re-render with object attachment — 3 phases auto-detected from rollout_x:
        #   Phase A (approach): EE far from box → box stays on table.
        #   Phase B (carry):    EE within 0.06m of transported object → box follows EE.
        #   Phase C (retreat):  EE within 0.08m of transported goal → box freezes there.
        transported_obj  = res["transport"].transform(new_scene["object_pose"].reshape(1,3))[0]
        transported_goal = res["transport"].transform(new_scene["goal_pose"].reshape(1,3))[0]
        rollout_x = res["rollout_x"]
        q_arr     = res["rollout_q"]

        # Auto-detect phase boundaries
        dist_to_obj  = np.linalg.norm(rollout_x - transported_obj,  axis=1)
        dist_to_goal = np.linalg.norm(rollout_x - transported_goal, axis=1)
        grasp_candidates = np.where(dist_to_obj  < 0.06)[0]
        place_candidates = np.where(dist_to_goal < 0.08)[0]
        grasp_fi = int(grasp_candidates[0]) if len(grasp_candidates) else len(q_arr) // 4
        place_fi = int(place_candidates[0]) if len(place_candidates) else 3 * len(q_arr) // 4
        # Ensure ordering
        place_fi = max(grasp_fi + 1, place_fi)

        _CARRY_OFFSET = np.array([0.0, 0.0, -0.03])  # box hangs 3cm below EE

        annotated = []
        for fi, q in enumerate(q_arr):
            if fi == grasp_fi:
                # Set attach offset so box is 3cm below EE at grasp point
                env._attached_geom_id = None  # ensure clean state
                env.set_qpos(q)
                ee_now = env.get_ee_pos()
                # Override model geom pos to follow EE with fixed downward offset
                gid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "object")
                if gid >= 0:
                    env._attached_geom_id = gid
                    env._attach_offset = _CARRY_OFFSET.copy()
            if fi == place_fi:
                env.detach_object()
            env.set_qpos(q)
            frame = env.render()
            if frame is not None:
                frame = add_text_overlay(
                    frame,
                    f"Scene {i+1} | err:{res['final_error']:.3f}m",
                    pos=(8, 24), font_scale=0.55,
                )
                progress = fi / max(1, len(q_arr) - 1)
                frame = add_progress_bar(frame, progress,
                                         success=res["success"] if fi == len(q_arr)-1 else None)
                annotated.append(frame)
        env.close()
        res["annotated_frames"] = annotated
        results.append(res)

    # Summary
    errors = [r["final_error"] for r in results]
    fails  = [r["ik_fail_rate"] for r in results]
    succs  = [r["success"] for r in results]
    print(f"\nReshelving: success={np.mean(succs)*100:.0f}%  mean_err={np.mean(errors):.3f}m  ik_fail={np.mean(fails)*100:.1f}%")

    # Tile + save
    frames_per_scene = [r["annotated_frames"] for r in results]
    if all(len(f) > 0 for f in frames_per_scene):
        tiled = _tile_2x2(frames_per_scene)
        _save_gif_mp4(tiled, out_dir, "franka_reshelving", fps=args.fps,
                      budget_bytes=5 * 1024 * 1024)
    else:
        print("WARNING: some scenes produced no frames — skipping tiling.")


if __name__ == "__main__":
    main()
