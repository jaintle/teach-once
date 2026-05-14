"""animate_cleaning_best.py — single best-scene cleaning animation.

Uses Scene 1 (flat surface, most visible motion).
Quarter-angle camera (arm + surface both visible).
Burns EE path as a red trail on each frame.
Saves reports/figures/franka_cleaning_best.gif.
"""

import pathlib
import sys

import imageio
import mujoco
import numpy as np
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))
from gpt_repro.utils.seeding import set_global_seed
from gpt_repro.envs.franka_env import FrankaKinematicEnv
from gpt_repro.envs.franka_scene import CAMERAS
from gpt_repro.policies.franka_demos import get_cleaning_waypoints
from gpt_repro.transport.franka_rollout import record_franka_demo
from gpt_repro.transport.policy_transport import PolicyTransport
from gpt_repro.policies.ds_policy import GPDynamicalSystem
from gpt_repro.gp.exact_gp import ExactGPRegressor
from gpt_repro.viz.frame_annotate import add_text_overlay, add_progress_bar

SEED       = 0
GP_N_ITER  = 80
N_STEPS    = 200
OUT_PATH   = pathlib.Path("reports/figures/franka_cleaning_best.gif")
FPS        = 20
WIDTH, HEIGHT = 720, 480

BASE_SCENE = {
    "surface_center":    np.array([0.50, 0.00, 0.64]),
    "surface_half_size": np.array([0.12, 0.12]),
}


def _make_S_T_cleaning(base_scene, rng):
    cx, cy, cz = base_scene["surface_center"]
    hw, hd = base_scene["surface_half_size"]
    S = np.array([
        [cx - hw, cy - hd, cz], [cx + hw, cy - hd, cz],
        [cx + hw, cy + hd, cz], [cx - hw, cy + hd, cz],
        [cx,      cy,      cz],
    ])
    delta = rng.uniform(-0.06, 0.06, S.shape); delta[:, 2] *= 0.3
    return S, S + delta


def _burn_trail(frame: np.ndarray, path_2d: list) -> np.ndarray:
    """Burn a thin red trail of previous EE 2D pixel positions onto the frame."""
    if len(path_2d) < 2:
        return frame
    out = frame.copy()
    H, W = out.shape[:2]
    for (px, py) in path_2d:
        px, py = int(round(px)), int(round(py))
        if 0 <= px < W and 0 <= py < H:
            # 2-pixel cross
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    nx, ny = px + dx, py + dy
                    if 0 <= nx < W and 0 <= ny < H:
                        out[ny, nx] = [220, 40, 40]
    return out


def _project_to_pixel(ee_world: np.ndarray, model, data, renderer, width, height):
    """Project a world-frame 3D point to pixel coords using MuJoCo's camera.
    Returns (px, py) or None if behind camera.
    """
    try:
        import mujoco as _mj
        # We don't have easy access to the view matrix here; use a simple
        # orthographic approximation based on camera lookat and azimuth.
        # This is sufficient for a rough trail overlay.
        lookat = np.array([0.45, 0.15, 0.65])  # quarter camera lookat
        # Map x,y world to pixel (approximate linear mapping)
        # Quarter camera: azimuth 225deg → roughly top-left view
        # We project onto (x, y) world plane with a rough scale
        scale_x = width  / 0.8   # world 0.8m -> pixel width
        scale_y = height / 0.6   # world 0.6m -> pixel height
        dx = ee_world[0] - lookat[0]
        dy = ee_world[1] - lookat[1]
        # rotate by -225+180 = -45 deg to get screen coordinates
        angle = np.radians(-45)
        sx = dx * np.cos(angle) - dy * np.sin(angle)
        sy = dx * np.sin(angle) + dy * np.cos(angle)
        px = int(width  / 2 + sx * scale_x)
        py = int(height / 2 - sy * scale_y)
        return (px, py)
    except Exception:
        return None


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    set_global_seed(SEED)

    # Record base demo
    print("Recording base cleaning demo …")
    base_env = FrankaKinematicEnv("cleaning", render_mode=None, width=WIDTH, height=HEIGHT)
    base_env.reset(seed=SEED)
    waypoints = get_cleaning_waypoints(BASE_SCENE)
    base_demo = record_franka_demo(base_env, waypoints)
    base_env.close()
    print(f"  Demo: {len(base_demo['x'])} steps, IK ok={base_demo['ik_success'].mean()*100:.0f}%")

    # Print waypoint coverage
    x_demo = base_demo["x"]
    print(f"  EE x range: [{x_demo[:,0].min():.3f}, {x_demo[:,0].max():.3f}]")
    print(f"  EE y range: [{x_demo[:,1].min():.3f}, {x_demo[:,1].max():.3f}]")
    print(f"  EE z range: [{x_demo[:,2].min():.3f}, {x_demo[:,2].max():.3f}]")
    surf_z = BASE_SCENE["surface_center"][2]
    min_z  = x_demo[:,2].min()
    print(f"  Surface z={surf_z:.3f}, min EE z={min_z:.3f}, gap={min_z - surf_z:.3f}m")

    # Scene 1: no randomization (flat surface, base scene = scene 1)
    rng_0 = np.random.default_rng(SEED)
    S, T  = _make_S_T_cleaning(BASE_SCENE, rng_0)

    # Run inline rollout with quarter camera
    print("Running GPT transport + rollout (Scene 1, quarter camera) …")
    env = FrankaKinematicEnv("cleaning", render_mode="rgb_array", width=WIDTH, height=HEIGHT)
    env.set_camera("quarter")
    ws_lo, ws_hi = env.get_workspace_bounds()

    xdot_demo = base_demo["xdot"].astype(float)
    transport  = PolicyTransport(gp_cls=ExactGPRegressor, n_iter_default=GP_N_ITER)
    transport.fit(S, T)
    x_t  = transport.transform(x_demo)
    xd_t = transport.transform_velocity(x_demo, xdot_demo)

    ds = GPDynamicalSystem(gp_cls=ExactGPRegressor, n_iter_default=GP_N_ITER)
    ds.fit(x_t, xd_t)

    _pred_v, _ = ds.predict(x_t, return_std=True)
    demo_v_norm = float(np.linalg.norm(xd_t, axis=1).mean()) + 1e-8
    pred_v_norm = float(np.linalg.norm(_pred_v, axis=1).mean()) + 1e-8
    velocity_scale = float(np.clip(demo_v_norm / pred_v_norm, 1.0, 50.0)) if pred_v_norm < demo_v_norm * 0.9 else 1.0
    if velocity_scale > 1.0:
        print(f"  Velocity rescale: {velocity_scale:.2f}x")

    x_goal = x_t[-1]
    env.reset()
    env.set_ee_pos(np.clip(x_t[0], ws_lo, ws_hi))

    xs     = [env.get_ee_pos().copy()]
    qs_raw = [env._data.qpos[:7].copy()]

    for _ in range(N_STEPS):
        obs = xs[-1]
        vel = ds.predict(obs[np.newaxis], return_std=False)
        if vel.ndim == 2: vel = vel[0]
        vel = vel + 0.8 * (x_goal - obs)
        x_next = np.clip(obs + (vel * velocity_scale) * 0.05, ws_lo, ws_hi)
        env.step(x_next)
        xs.append(env.get_ee_pos().copy())
        qs_raw.append(env._data.qpos[:7].copy())

    rollout_x = np.array(xs)
    q_smooth  = gaussian_filter1d(np.array(qs_raw), sigma=1.5, axis=0)
    final_err  = float(np.linalg.norm(rollout_x[-1] - x_t[-1]))
    print(f"  err={final_err:.3f}m")

    # Render with trail
    frames     = []
    trail_2d   = []
    for fi, q in enumerate(q_smooth):
        env.set_qpos(q)
        mujoco.mj_forward(env._model, env._data)
        frame = env.render()
        if frame is None:
            continue

        ee_now = env.get_ee_pos()
        px_py  = _project_to_pixel(ee_now, env._model, env._data, env._renderer, WIDTH, HEIGHT)
        if px_py is not None:
            trail_2d.append(px_py)

        frame = _burn_trail(frame, trail_2d[-60:])   # last 60 positions = ~3s trail

        progress = fi / max(1, len(q_smooth) - 1)
        frame = add_text_overlay(frame, f"Cleaning | quarter-view | err:{final_err:.3f}m",
                                  pos=(8, 24), font_scale=0.50)
        frame = add_progress_bar(frame, progress, success=None)
        frames.append(frame)

    env.close()

    if not frames:
        print("ERROR: no frames rendered"); return

    imageio.mimwrite(str(OUT_PATH), frames, fps=FPS, loop=0)
    sz = OUT_PATH.stat().st_size / 1024
    print(f"\nSaved {OUT_PATH.name}: {sz:.0f} KB  ({len(frames)} frames)")


if __name__ == "__main__":
    main()
