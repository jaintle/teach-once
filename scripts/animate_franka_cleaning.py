"""animate_franka_cleaning.py — Phase 14.

Records a raster-scan cleaning demo on the Franka arm, transports it to
4 surface variants via GPT, and renders a 2×2 GIF/MP4 with:
  - EE colour coded by estimated contact force (Hooke proxy).
  - Camera switch: front for first half, top for second half.

CLI args: --seed, --n_scenes, --fps, --width, --height, --out_dir,
          --gp_n_iter, --n_steps
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
from gpt_repro.policies.franka_demos import get_cleaning_waypoints
from gpt_repro.transport.franka_rollout import (
    record_franka_demo,
    transport_and_rollout_franka,
)
from gpt_repro.viz.frame_annotate import add_text_overlay, colormap_scalar


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed",      type=int, default=0)
    p.add_argument("--n_scenes",  type=int, default=4)
    p.add_argument("--fps",       type=int, default=15)
    p.add_argument("--width",     type=int, default=720)
    p.add_argument("--height",    type=int, default=480)
    p.add_argument("--out_dir",   default="reports/figures/")
    p.add_argument("--gp_n_iter", type=int, default=80)
    p.add_argument("--n_steps",   type=int, default=80)
    return p.parse_args()


def _make_S_T(base: dict, rng):
    """Simple 4-corner transport for surface shift."""
    c   = np.asarray(base["surface_center"])
    h   = np.asarray(base["surface_half_size"])
    S = np.array([
        [c[0]-h[0], c[1]-h[1], c[2]],
        [c[0]+h[0], c[1]-h[1], c[2]],
        [c[0]-h[0], c[1]+h[1], c[2]],
        [c[0]+h[0], c[1]+h[1], c[2]],
        [c[0],      c[1],      c[2]+0.08],
    ])
    shift = rng.uniform(-0.06, 0.06, 3)
    shift[2] *= 0.3
    T = S + shift
    return S, T


def _randomize_surface(base: dict, rng) -> dict:
    shift = rng.uniform(-0.06, 0.06, 3)
    shift[2] *= 0.3
    return {
        "surface_center":    base["surface_center"]    + shift,
        "surface_half_size": base["surface_half_size"] * rng.uniform(0.8, 1.2, 2),
    }


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
    while len(frames[::step]) * frames[0].nbytes // 10 > budget_bytes and step < 4:
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


def _rollout_with_camera_switch_and_force(env, demo, S, T, args, scene_idx):
    """Run transport_and_rollout_franka manually with camera switching and force coloring."""
    from gpt_repro.gp.exact_gp import ExactGPRegressor
    from gpt_repro.policies.ds_policy import GPDynamicalSystem
    from gpt_repro.transport.policy_transport import PolicyTransport
    from scipy.ndimage import gaussian_filter1d

    x_demo    = demo["x"].astype(float)
    xdot_demo = demo["xdot"].astype(float)

    transport = PolicyTransport(gp_cls=ExactGPRegressor, n_iter_default=args.gp_n_iter)
    transport.fit(S, T)
    x_t   = transport.transform(x_demo)
    xd_t  = transport.transform_velocity(x_demo, xdot_demo)

    ds = GPDynamicalSystem(gp_cls=ExactGPRegressor, n_iter_default=args.gp_n_iter)
    ds.fit(x_t, xd_t)

    # Velocity rescaling: compensate for GP attenuation
    _pred_v, _ = ds.predict(x_t, return_std=True)
    demo_v_norm = float(np.linalg.norm(xd_t, axis=1).mean()) + 1e-8
    pred_v_norm = float(np.linalg.norm(_pred_v, axis=1).mean()) + 1e-8
    if pred_v_norm < demo_v_norm * 0.9:
        velocity_scale = float(np.clip(demo_v_norm / pred_v_norm, 1.0, 50.0))
    else:
        velocity_scale = 1.0

    ws_lo, ws_hi = env.get_workspace_bounds()
    env.reset()
    env.set_ee_pos(np.clip(x_t[0], ws_lo, ws_hi))

    xs = [env.get_ee_pos().copy()]
    qs_raw = [env._data.qpos[:7].copy()]
    ik_fails = 0
    n_steps = args.n_steps

    for step_i in range(n_steps):
        obs = xs[-1]
        vel = ds.predict(obs[np.newaxis], return_std=False)
        if vel.ndim == 2:
            vel = vel[0]
        x_next = np.clip(obs + (vel * velocity_scale) * 0.05, ws_lo, ws_hi)
        _, _, _, _, info = env.step(x_next)
        if not info["ik_success"]:
            ik_fails += 1
        xs.append(env.get_ee_pos().copy())
        qs_raw.append(env._data.qpos[:7].copy())

    rollout_x = np.array(xs)
    q_smooth  = gaussian_filter1d(np.array(qs_raw), sigma=1.5, axis=0)
    xdots_est = np.diff(rollout_x, axis=0)
    speed     = np.linalg.norm(np.vstack([xdots_est, xdots_est[-1:]]), axis=1)

    # Re-render with smoothed joints + camera switch + force colour
    frames = []
    half   = len(q_smooth) // 2
    for fi, q in enumerate(q_smooth):
        cam = "front" if fi < half else "top"
        env.set_camera(cam)
        env.set_qpos(q)

        # Force colour via EE speed (Hooke proxy)
        force_val = float(speed[min(fi, len(speed)-1)])
        r, g, b   = colormap_scalar(force_val, vmin=0.0, vmax=0.05, cmap="coolwarm")
        # Try to colour EE geom (search for capsule/sphere geom named "ee" or similar)
        for gname in ["ee", "ee_cap", "end_effector"]:
            gid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, gname)
            if gid >= 0:
                env.model.geom_rgba[gid] = [r/255, g/255, b/255, 1.0]
                break

        frame = env.render()
        if frame is not None:
            frame = add_text_overlay(
                frame,
                f"S{scene_idx+1} {cam} | spd:{force_val:.3f}",
                pos=(8, 24), font_scale=0.50,
            )
            frames.append(frame)

    final_err = float(np.linalg.norm(rollout_x[-1] - x_t[-1]))
    return {
        "frames": frames,
        "final_error": final_err,
        "ik_fail_rate": ik_fails / max(1, n_steps),
        "success": final_err < 0.15,
    }


def main():
    args = parse_args()
    set_global_seed(args.seed)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_scene = {
        "surface_center":    np.array([0.50, 0.00, 0.64]),
        "surface_half_size": np.array([0.12, 0.12]),
    }

    # Record base demo
    print("Recording base cleaning demo …")
    base_env = FrankaKinematicEnv("cleaning", render_mode=None,
                                   width=args.width, height=args.height)
    base_env.reset(seed=args.seed)
    waypoints  = get_cleaning_waypoints(base_scene)
    base_demo  = record_franka_demo(base_env, waypoints)
    base_env.close()
    print(f"  Demo: {len(base_demo['x'])} steps, IK ok={base_demo['ik_success'].mean()*100:.0f}%")

    results = []
    for i in range(args.n_scenes):
        rng_i = np.random.default_rng(args.seed + i)
        new_scene = _randomize_surface(base_scene, rng_i)
        S, T = _make_S_T(base_scene, rng_i)

        print(f"Scene {i+1}/{args.n_scenes}")
        env = FrankaKinematicEnv("cleaning", render_mode="rgb_array",
                                  width=args.width, height=args.height)
        res = _rollout_with_camera_switch_and_force(env, base_demo, S, T, args, i)
        print(f"  err={res['final_error']:.3f}m  ik_fail={res['ik_fail_rate']*100:.0f}%  success={res['success']}")
        env.close()
        results.append(res)

    errors = [r["final_error"] for r in results]
    fails  = [r["ik_fail_rate"] for r in results]
    succs  = [r["success"] for r in results]
    print(f"\nCleaning: success={np.mean(succs)*100:.0f}%  mean_err={np.mean(errors):.3f}m  ik_fail={np.mean(fails)*100:.1f}%")

    frames_per_scene = [r["frames"] for r in results]
    if all(len(f) > 0 for f in frames_per_scene):
        tiled = _tile_2x2(frames_per_scene)
        _save_gif_mp4(tiled, out_dir, "franka_cleaning", fps=args.fps,
                      budget_bytes=8 * 1024 * 1024)
    else:
        print("WARNING: some scenes produced no frames.")


if __name__ == "__main__":
    main()
