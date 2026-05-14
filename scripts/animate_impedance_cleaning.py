"""animate_impedance_cleaning.py — kinematic direct-waypoint cleaning GIF.

Uses FrankaKinematicEnv + direct IK waypoint replay (NOT DS rollout).
Cleaning is a PATH task: direct waypoint replay gives correct surface sweep.
Quarter camera. Light-blue surface, red EE, coverage % overlay.
Saves reports/figures/final_cleaning.gif  720x480, 12fps.
"""

import argparse
import pathlib
import sys

import imageio
import mujoco
import numpy as np
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from gpt_repro.utils.seeding import set_global_seed
from gpt_repro.envs.franka_env import FrankaKinematicEnv, Q_HOME
from gpt_repro.envs.franka_scene import build_scene_xml, load_scene_model, CAMERAS, FRANKA_ASSETS_DIR
from gpt_repro.envs.ik_solver import IKSolver, interpolate_joint_trajectory
from gpt_repro.policies.franka_demos import get_cleaning_waypoints
from gpt_repro.transport.policy_transport import PolicyTransport
from gpt_repro.gp.exact_gp import ExactGPRegressor
from gpt_repro.viz.frame_annotate import add_text_overlay, add_progress_bar

import gymnasium

OUT_PATH   = pathlib.Path("reports/figures/final_cleaning.gif")
FPS        = 12
WIDTH, HEIGHT = 720, 480
GP_N_ITER  = 80
N_INTERP   = 5  # frames between waypoints

BASE_SCENE = {
    "surface_center":    np.array([0.50, 0.00, 0.64]),
    "surface_half_size": np.array([0.12, 0.12]),
}
TARGET_SCENE = {
    "surface_center":    np.array([0.50, 0.08, 0.64]),
    "surface_half_size": np.array([0.12, 0.12]),
}

# Quarter camera
_CUSTOM_CAM = (np.array([0.45, 0.15, 0.65]), 1.6, -35.0, 225.0)


def _make_S_T():
    c_s = BASE_SCENE["surface_center"]
    c_t = TARGET_SCENE["surface_center"]
    h = float(BASE_SCENE["surface_half_size"][0])
    offsets = np.array([
        [ h, 0, 0.02], [-h, 0, 0.02],
        [0,  h, 0.02], [0, -h, 0.02],
        [ h, h, 0.02], [-h,-h, 0.02],
    ])
    return c_s + offsets, c_t + offsets


def _build_env(render_mode):
    """Build cleaning env with light-blue surface."""
    xml = build_scene_xml("cleaning")
    # Light blue surface
    xml = xml.replace('material="surface_mat"', 'rgba="0.5 0.75 1.0 1"')
    model, data = load_scene_model(xml)
    env = FrankaKinematicEnv.__new__(FrankaKinematicEnv)
    env.task = "cleaning"
    env._render_mode = render_mode
    env._width, env._height = WIDTH, HEIGHT
    env.dt, env.control_dt = 0.002, 0.05
    env._model, env._data = model, data
    env._site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    env._ik = IKSolver(model, data, ee_site_name="attachment_site",
                       max_iter=200, tol=1e-3, damping=1e-4, nullspace_gain=0.5)
    env._data.qpos[:] = Q_HOME
    mujoco.mj_forward(model, data)
    env._ee_home = data.site_xpos[env._site_id].copy()
    ob = np.full(3, -2.0); ob2 = np.full(3, 2.0)
    env.observation_space = gymnasium.spaces.Box(low=ob, high=ob2, dtype=np.float64)
    env.action_space      = gymnasium.spaces.Box(low=ob.copy(), high=ob2.copy(), dtype=np.float64)
    env._renderer = None
    env._attached_geom_id = None
    env._attach_offset = None
    env._cam_lookat, env._cam_distance, env._cam_elevation, env._cam_azimuth = _CUSTOM_CAM
    return env


def _make_camera():
    lookat, dist, elev, azim = _CUSTOM_CAM
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance = dist
    cam.elevation = elev
    cam.azimuth = azim
    return cam


def _render_env(env):
    if env._renderer is None:
        env._renderer = mujoco.Renderer(env._model, height=HEIGHT, width=WIDTH)
    env._renderer.update_scene(env._data, camera=_make_camera())
    return env._renderer.render().copy()


def _coverage(ee_positions, scene):
    """Fraction of the surface area covered by EE path."""
    cx, cy, cz = scene["surface_center"]
    hx, hy = scene["surface_half_size"]
    # Count unique 5cm grid cells touched
    cells = set()
    cell_size = 0.05
    for p in ee_positions:
        if abs(p[0]-cx) <= hx and abs(p[1]-cy) <= hy and abs(p[2]-cz) < 0.06:
            ix = int((p[0] - (cx-hx)) / cell_size)
            iy = int((p[1] - (cy-hy)) / cell_size)
            cells.add((ix, iy))
    total = int((2*hx / cell_size + 1) * (2*hy / cell_size + 1))
    return len(cells) / max(1, total)


def main(seed: int = 0):
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    set_global_seed(seed)

    # Get base waypoints
    base_waypoints = get_cleaning_waypoints(BASE_SCENE)
    print(f"Base waypoints: {len(base_waypoints)}")

    S, T = _make_S_T()

    # Transport the waypoints (NOT the full demo trajectory)
    print("Fitting transport...")
    transport = PolicyTransport(gp_cls=ExactGPRegressor, n_iter_default=GP_N_ITER)
    transport.fit(S, T)

    transported_wps = transport.transform(np.asarray(base_waypoints))
    ws_lo = np.array([0.25, -0.40, 0.40])
    ws_hi = np.array([0.75,  0.40, 0.95])
    transported_wps = np.clip(transported_wps, ws_lo, ws_hi)
    print(f"Transported waypoints x=[{transported_wps[:,0].min():.3f}, {transported_wps[:,0].max():.3f}]")
    print(f"                       y=[{transported_wps[:,1].min():.3f}, {transported_wps[:,1].max():.3f}]")

    # Build env
    print("Building environment...")
    env = _build_env(render_mode="rgb_array")
    env._data.qpos[:] = Q_HOME
    mujoco.mj_forward(env._model, env._data)

    # Direct IK waypoint replay with interpolation
    print("Running direct IK waypoint replay...")
    all_qs = []
    all_ee = []
    M = len(transported_wps)
    q_cur = Q_HOME[:7].copy()

    for seg in range(M):
        wp = transported_wps[seg]
        q_wp, ok = env._ik.solve(wp, q_init=q_cur)
        if not ok:
            print(f"  IK failed at waypoint {seg}, using best solution")
        # Interpolate from previous
        if seg == 0:
            all_qs.append(q_wp)
            all_ee.append(wp)
            q_cur = q_wp
        else:
            q_prev = all_qs[-1]
            for alpha in np.linspace(0, 1, N_INTERP + 1)[1:]:
                q_interp = (1 - alpha) * q_prev + alpha * q_wp
                all_qs.append(q_interp)
                # EE pos from FK
                env._data.qpos[:7] = q_interp
                mujoco.mj_forward(env._model, env._data)
                all_ee.append(env._data.site_xpos[env._site_id].copy())
            q_cur = q_wp

    target_scene = {
        "surface_center":    transport.transform(BASE_SCENE["surface_center"].reshape(1,3))[0],
        "surface_half_size": BASE_SCENE["surface_half_size"],
    }
    cov = _coverage(all_ee, target_scene)
    print(f"Surface coverage: {cov*100:.0f}%  ({len(all_ee)} positions)")

    # Render all frames
    print("Rendering frames...")
    frames = []
    for fi, q in enumerate(all_qs):
        env.set_qpos(q)
        frame = _render_env(env)
        progress = fi / max(1, len(all_qs)-1)
        cov_so_far = _coverage(all_ee[:fi+1], target_scene)
        frame = add_text_overlay(frame,
                                  f"Cleaning (GPT)  Coverage: {cov_so_far*100:.0f}%",
                                  pos=(8, 28), font_scale=0.55, color=(255, 220, 80))
        frame = add_progress_bar(frame, progress)
        frames.append(frame)

    env._renderer.close()
    print(f"  Frames: {len(frames)}")

    imageio.mimsave(str(OUT_PATH), frames, fps=FPS, loop=0)
    sz = OUT_PATH.stat().st_size / 1e6
    print(f"Saved {OUT_PATH}  ({sz:.1f} MB, {len(frames)} frames)")
    print(f"\nFinal coverage: {cov*100:.0f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(seed=args.seed)
