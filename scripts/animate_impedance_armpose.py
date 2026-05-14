"""animate_impedance_armpose.py — kinematic direct-waypoint armpose GIF.

Uses FrankaKinematicEnv + direct IK waypoint replay (NOT DS rollout).
Armpose is a PATH task: waypoints = transported keypoints.
15 interpolation frames per waypoint for slow, clear motion.
Keypoints flash white when EE comes within 0.08m.
Side camera. 8fps. Saves reports/figures/final_armpose.gif  720x480.
"""

import argparse
import pathlib
import sys

import imageio
import mujoco
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from gpt_repro.utils.seeding import set_global_seed
from gpt_repro.envs.franka_env import FrankaKinematicEnv, Q_HOME
from gpt_repro.envs.franka_scene import build_scene_xml, load_scene_model, CAMERAS, FRANKA_ASSETS_DIR
from gpt_repro.envs.ik_solver import IKSolver
from gpt_repro.policies.franka_demos import get_armpose_waypoints
from gpt_repro.transport.policy_transport import PolicyTransport
from gpt_repro.gp.exact_gp import ExactGPRegressor
from gpt_repro.viz.frame_annotate import add_text_overlay, add_progress_bar

import gymnasium

OUT_PATH   = pathlib.Path("reports/figures/final_armpose.gif")
FPS        = 8
WIDTH, HEIGHT = 720, 480
GP_N_ITER  = 80
N_INTERP   = 15  # frames between keypoints

BASE_KPS = {
    "shoulder": np.array([0.35, 0.00, 0.70]),
    "elbow":    np.array([0.47, 0.00, 0.80]),
    "wrist":    np.array([0.57, 0.00, 0.75]),
    "hand":     np.array([0.62, 0.00, 0.65]),
}
TARGET_KPS = {
    "shoulder": np.array([0.35, 0.06, 0.70]),
    "elbow":    np.array([0.47, 0.06, 0.80]),
    "wrist":    np.array([0.57, 0.06, 0.75]),
    "hand":     np.array([0.62, 0.06, 0.65]),
}

# Sphere geom names and their default RGBA
_KP_NAMES  = ["kp_shoulder", "kp_elbow", "kp_wrist", "kp_hand"]
_KP_LABELS = ["shoulder",    "elbow",    "wrist",    "hand"]
_KP_DEFAULT_RGBA = {
    "kp_shoulder": np.array([0.0, 0.9, 0.9, 0.9]),
    "kp_elbow":    np.array([0.9, 0.0, 0.9, 0.9]),
    "kp_wrist":    np.array([1.0, 0.9, 0.0, 0.9]),
    "kp_hand":     np.array([0.2, 0.4, 0.9, 0.9]),
}
_KP_TEXT_COLORS = {
    "shoulder": (0, 230, 230),
    "elbow":    (230, 0, 230),
    "wrist":    (255, 230, 0),
    "hand":     (80, 120, 230),
}
FLASH_FRAMES = 5
TOUCH_THRESH = 0.08  # m

_SIDE_CAM = CAMERAS["side"]


def _make_S_T():
    keys = list(BASE_KPS.keys())
    S = np.array([BASE_KPS[k] for k in keys])
    T = np.array([TARGET_KPS[k] for k in keys])
    mid_s = (S[0] + S[1]) * 0.5
    mid_t = (T[0] + T[1]) * 0.5
    return np.vstack([S, mid_s]), np.vstack([T, mid_t])


def _build_env(render_mode):
    """Build armpose env with large keypoint spheres (r=0.07)."""
    xml = build_scene_xml("armpose", large_kp_spheres=True)
    # Even larger spheres: patch to 0.07
    xml = xml.replace('size="0.060"', 'size="0.07"')
    xml = xml.replace('size="0.050"', 'size="0.07"')
    xml = xml.replace('size="0.040"', 'size="0.07"')
    # Connecting "bone" capsules between keypoints (thin grey)
    bones = ""
    pts = [BASE_KPS[k] for k in ["shoulder", "elbow", "wrist", "hand"]]
    for i in range(len(pts)-1):
        p0, p1 = pts[i], pts[i+1]
        mid = (p0 + p1) / 2
        half_len = np.linalg.norm(p1 - p0) / 2
        # Capsule oriented along x (approx — fromto is easier)
        bones += (
            f'    <geom name="bone{i}" type="capsule" fromto='
            f'"{p0[0]:.4f} {p0[1]:.4f} {p0[2]:.4f} {p1[0]:.4f} {p1[1]:.4f} {p1[2]:.4f}" '
            f'size="0.012" rgba="0.55 0.55 0.55 0.7"/>\n'
        )
    xml = xml.replace("  </worldbody>", bones + "  </worldbody>", 1)

    model, data = load_scene_model(xml)
    env = FrankaKinematicEnv.__new__(FrankaKinematicEnv)
    env.task = "armpose"
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
    env._cam_lookat, env._cam_distance, env._cam_elevation, env._cam_azimuth = _SIDE_CAM
    return env


def _make_camera():
    lookat, dist, elev, azim = _SIDE_CAM
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


def main(seed: int = 0):
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    set_global_seed(seed)

    S, T = _make_S_T()

    # Transport keypoints
    print("Fitting transport...")
    transport = PolicyTransport(gp_cls=ExactGPRegressor, n_iter_default=GP_N_ITER)
    transport.fit(S, T)

    base_wps = get_armpose_waypoints(BASE_KPS)
    transported_wps = transport.transform(np.asarray(base_wps))
    ws_lo = np.array([0.25, -0.40, 0.40])
    ws_hi = np.array([0.75,  0.40, 0.95])
    transported_wps = np.clip(transported_wps, ws_lo, ws_hi)

    # Transported keypoint positions (for flash detection)
    t_kps = {k: transport.transform(BASE_KPS[k].reshape(1,3))[0]
             for k in BASE_KPS}
    print("Transported keypoints:")
    for k, v in t_kps.items():
        print(f"  {k}: {np.round(v, 3)}")

    # Build env
    print("Building environment...")
    env = _build_env(render_mode="rgb_array")

    # Update sphere positions to transported ones
    for kname, kpos in t_kps.items():
        gid = mujoco.mj_name2id(env._model, mujoco.mjtObj.mjOBJ_GEOM, kname)
        if gid >= 0:
            env._model.geom_pos[gid] = kpos

    env._data.qpos[:] = Q_HOME
    mujoco.mj_forward(env._model, env._data)

    # Direct IK waypoint replay
    print("Running direct IK waypoint replay...")
    M = len(transported_wps)
    all_qs = []
    all_ee = []
    q_cur = Q_HOME[:7].copy()

    for seg in range(M):
        wp = transported_wps[seg]
        q_wp, ok = env._ik.solve(wp, q_init=q_cur)
        if not ok:
            print(f"  IK warn at waypoint {seg}")
        if seg == 0:
            all_qs.append(q_wp)
            env._data.qpos[:7] = q_wp
            mujoco.mj_forward(env._model, env._data)
            all_ee.append(env._data.site_xpos[env._site_id].copy())
            q_cur = q_wp
        else:
            q_prev = all_qs[-1]
            for alpha in np.linspace(0, 1, N_INTERP + 1)[1:]:
                q_int = (1 - alpha) * q_prev + alpha * q_wp
                all_qs.append(q_int)
                env._data.qpos[:7] = q_int
                mujoco.mj_forward(env._model, env._data)
                all_ee.append(env._data.site_xpos[env._site_id].copy())
            q_cur = q_wp

    # Check which keypoints reached
    kp_reached = {}
    for k, kpos in t_kps.items():
        dists = [np.linalg.norm(np.array(ee) - kpos) for ee in all_ee]
        kp_reached[k] = min(dists) < TOUCH_THRESH
    print("Keypoints reached (within 0.08m):")
    for k, reached in kp_reached.items():
        dists = [np.linalg.norm(np.array(ee) - t_kps[k]) for ee in all_ee]
        print(f"  {k}: {'YES' if reached else 'NO'}  min_dist={min(dists):.3f}m")

    # Render frames with flash effect
    print("Rendering frames...")
    kp_geom_ids = {n: mujoco.mj_name2id(env._model, mujoco.mjtObj.mjOBJ_GEOM, n)
                   for n in _KP_NAMES}
    flash_counter = {n: 0 for n in _KP_NAMES}  # frames remaining in flash

    frames = []
    for fi, (q, ee) in enumerate(zip(all_qs, all_ee)):
        env.set_qpos(q)

        # Flash logic
        for ki, (kname, label) in enumerate(zip(_KP_NAMES, _KP_LABELS)):
            kpos = t_kps[label]
            dist = float(np.linalg.norm(np.array(ee) - kpos))
            gid  = kp_geom_ids[kname]
            if dist < TOUCH_THRESH and gid >= 0:
                flash_counter[kname] = FLASH_FRAMES
            if flash_counter[kname] > 0:
                env._model.geom_rgba[gid] = [1, 1, 1, 1]  # flash white
                flash_counter[kname] -= 1
            else:
                if gid >= 0:
                    env._model.geom_rgba[gid] = _KP_DEFAULT_RGBA[kname]

        mujoco.mj_forward(env._model, env._data)
        frame = _render_env(env)

        # Find nearest keypoint for label
        dists = {k: float(np.linalg.norm(np.array(ee) - t_kps[k])) for k in BASE_KPS}
        nearest = min(dists, key=dists.get)
        near_d  = dists[nearest]
        label_text = f"Reaching: {nearest}" if near_d < 0.12 else "Arm-pose (GPT)"
        col = _KP_TEXT_COLORS.get(nearest, (255,255,255)) if near_d < 0.12 else (255,255,200)

        frame = add_text_overlay(frame, label_text, pos=(8, 28), font_scale=0.55, color=col)
        frame = add_progress_bar(frame, fi / max(1, len(all_qs)-1))
        frames.append(frame)

    env._renderer.close()
    print(f"  Frames: {len(frames)}")

    imageio.mimsave(str(OUT_PATH), frames, fps=FPS, loop=0)
    sz = OUT_PATH.stat().st_size / 1e6
    print(f"Saved {OUT_PATH}  ({sz:.1f} MB, {len(frames)} frames)")
    print(f"\nKeypoints reached: {[k for k, v in kp_reached.items() if v]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(seed=args.seed)
