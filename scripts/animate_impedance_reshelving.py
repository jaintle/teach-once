"""animate_impedance_reshelving.py — kinematic 3-phase reshelving GIF.

Uses FrankaKinematicEnv (NOT impedance), n_steps=500, 3-phase attractor:
  APPROACH (gain=2.5) → CARRY (box follows EE, gain=1.0) → PLACE (gain=0.5).
Bright orange box (0.035m), bright green goal marker.
Custom camera pull-back so both table and shelf are visible.
Saves reports/figures/final_reshelving.gif  720x480, 15fps.
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
from gpt_repro.envs.ik_solver import IKSolver
from gpt_repro.policies.franka_demos import get_reshelving_waypoints
from gpt_repro.transport.franka_rollout import record_franka_demo
from gpt_repro.transport.policy_transport import PolicyTransport
from gpt_repro.policies.ds_policy import GPDynamicalSystem
from gpt_repro.gp.exact_gp import ExactGPRegressor
from gpt_repro.viz.frame_annotate import add_text_overlay, add_progress_bar

import gymnasium

OUT_PATH   = pathlib.Path("reports/figures/final_reshelving.gif")
FPS        = 15
WIDTH, HEIGHT = 720, 480
GP_N_ITER  = 80
N_STEPS    = 500

BASE_SCENE = {
    "object_pose": np.array([0.50, 0.00, 0.63]),
    "goal_pose":   np.array([0.30, 0.10, 0.75]),
}
TARGET_SCENE = {
    "object_pose": np.array([0.48, 0.08, 0.63]),
    "goal_pose":   np.array([0.32, -0.06, 0.75]),
}

# Custom camera: pulled back so both table + shelf visible
_CUSTOM_CAM = (np.array([0.4, 0.2, 0.7]), 1.8, -25.0, 165.0)


def _make_S_T():
    obj_s  = BASE_SCENE["object_pose"]
    goal_s = BASE_SCENE["goal_pose"]
    obj_t  = TARGET_SCENE["object_pose"]
    goal_t = TARGET_SCENE["goal_pose"]
    S = np.array([
        obj_s  + [0.05,  0.05, 0.0], obj_s  + [-0.05, 0.05, 0.0],
        goal_s + [0.05,  0.05, 0.0], goal_s + [-0.05, 0.05, 0.0],
        obj_s  + [0.0,   0.0,  0.1], goal_s + [0.0,   0.0,  0.1],
    ])
    T = np.array([
        obj_t  + [0.05,  0.05, 0.0], obj_t  + [-0.05, 0.05, 0.0],
        goal_t + [0.05,  0.05, 0.0], goal_t + [-0.05, 0.05, 0.0],
        obj_t  + [0.0,   0.0,  0.1], goal_t + [0.0,   0.0,  0.1],
    ])
    return S, T


def _build_env(obj_pos, goal_pos, render_mode):
    """Build FrankaKinematicEnv with bright object + goal marker."""
    xml = build_scene_xml("reshelving", object_pos=tuple(obj_pos))
    # Bigger, brighter orange box
    xml = xml.replace('size="0.025 0.025 0.025"', 'size="0.035 0.035 0.035"')
    xml = xml.replace('material="object_mat"', 'rgba="1.0 0.45 0.0 1"')
    # Brighter green goal slot
    xml = xml.replace('material="goal_mat"', 'rgba="0.1 1.0 0.1 0.9"')
    # Extra sphere at goal
    goal_marker = (
        f'    <geom name="goal_marker" type="sphere" size="0.04"\n'
        f'          pos="{goal_pos[0]:.4f} {goal_pos[1]:.4f} {goal_pos[2]:.4f}" '
        f'rgba="0.1 1.0 0.1 0.7"/>\n'
    )
    xml = xml.replace("  </worldbody>", goal_marker + "  </worldbody>", 1)

    model, data = load_scene_model(xml)
    env = FrankaKinematicEnv.__new__(FrankaKinematicEnv)
    env.task = "reshelving"
    env._render_mode = render_mode
    env._width  = WIDTH
    env._height = HEIGHT
    env.dt = 0.002
    env.control_dt = 0.05
    env._model = model
    env._data  = data
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


def main(seed: int = 0):
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    set_global_seed(seed)

    # Record base demo
    print("Recording base demo...")
    base_env = FrankaKinematicEnv("reshelving", render_mode=None, width=WIDTH, height=HEIGHT)
    base_env.reset(seed=seed)
    base_demo = record_franka_demo(base_env, get_reshelving_waypoints(BASE_SCENE))
    base_env.close()
    print(f"  Demo: {len(base_demo['x'])} steps")

    S, T = _make_S_T()

    # Fit transport + DS
    print("Fitting transport...")
    transport = PolicyTransport(gp_cls=ExactGPRegressor, n_iter_default=GP_N_ITER)
    transport.fit(S, T)
    x_demo = np.asarray(base_demo["x"], dtype=float)
    xd_demo = np.asarray(base_demo["xdot"], dtype=float)
    x_t  = transport.transform(x_demo)
    xd_t = transport.transform_velocity(x_demo, xd_demo)

    ds = GPDynamicalSystem(gp_cls=ExactGPRegressor, n_iter_default=GP_N_ITER)
    ds.fit(x_t, xd_t)

    t_obj  = transport.transform(BASE_SCENE["object_pose"].reshape(1,3))[0]
    t_goal = transport.transform(BASE_SCENE["goal_pose"].reshape(1,3))[0]
    print(f"  Transported obj:  {np.round(t_obj,3)}")
    print(f"  Transported goal: {np.round(t_goal,3)}")

    # Velocity rescaling
    _pv, _ = ds.predict(x_t, return_std=True)
    dv = float(np.linalg.norm(xd_t, axis=1).mean()) + 1e-8
    pv = float(np.linalg.norm(_pv, axis=1).mean()) + 1e-8
    vel_scale = float(np.clip(dv/pv, 1.0, 50.0)) if pv < dv*0.9 else 1.0
    if vel_scale > 1: print(f"  Velocity rescale: {vel_scale:.2f}x")

    ws_lo = np.array([0.25, -0.40, 0.40])
    ws_hi = np.array([0.75,  0.40, 0.95])

    # Build render env
    print("Building environment...")
    env = _build_env(t_obj, t_goal, render_mode="rgb_array")
    obj_id  = mujoco.mj_name2id(env._model, mujoco.mjtObj.mjOBJ_GEOM, "object")
    tag_id  = mujoco.mj_name2id(env._model, mujoco.mjtObj.mjOBJ_GEOM, "object_tag")

    env._data.qpos[:] = Q_HOME
    mujoco.mj_forward(env._model, env._data)
    env.set_ee_pos(np.clip(x_t[0], ws_lo, ws_hi))

    # 3-phase rollout
    PHASE_APPROACH, PHASE_CARRY, PHASE_PLACE = "APPROACH", "CARRY", "PLACE"
    CARRY_THRESH = 0.07
    PLACE_THRESH = 0.09
    CARRY_OFFSET = np.array([0.0, 0.0, -0.03])
    GAINS = {PHASE_APPROACH: 2.5, PHASE_CARRY: 1.0, PHASE_PLACE: 0.5}

    phase = PHASE_APPROACH
    obj_pos = t_obj.copy()
    xs = [env.get_ee_pos().copy()]
    qs_raw = [env._data.qpos[:7].copy()]
    phases = [PHASE_APPROACH]

    for _ in range(N_STEPS):
        obs = xs[-1]
        dist_obj  = float(np.linalg.norm(obs - obj_pos))
        dist_goal = float(np.linalg.norm(obs - t_goal))
        if phase == PHASE_APPROACH and dist_obj < CARRY_THRESH:
            phase = PHASE_CARRY
        if phase == PHASE_CARRY and dist_goal < PLACE_THRESH:
            phase = PHASE_PLACE

        x_tgt = x_t[-1] if phase == PHASE_APPROACH else t_goal
        vel, _ = ds.predict(obs[np.newaxis], return_std=True)
        vel = vel[0] + GAINS[phase] * (x_tgt - obs)
        x_next = np.clip(obs + vel * vel_scale * 0.05, ws_lo, ws_hi)
        env.step(x_next)
        xs.append(env.get_ee_pos().copy())
        qs_raw.append(env._data.qpos[:7].copy())
        phases.append(phase)

        if phase in (PHASE_CARRY, PHASE_PLACE):
            obj_pos = env.get_ee_pos() + CARRY_OFFSET
            if obj_id >= 0: env._model.geom_pos[obj_id] = obj_pos
            if tag_id >= 0: env._model.geom_pos[tag_id] = obj_pos + [0,-0.036,0]

    rollout_x = np.array(xs)
    final_error = float(np.linalg.norm(rollout_x[-1] - t_goal))
    print(f"  Final error: {final_error:.4f} m")
    print(f"  APPROACH={phases.count(PHASE_APPROACH)}, "
          f"CARRY={phases.count(PHASE_CARRY)}, PLACE={phases.count(PHASE_PLACE)}")

    # Smooth + re-render
    q_smooth = gaussian_filter1d(np.array(qs_raw), sigma=1.5, axis=0)
    if obj_id >= 0: env._model.geom_pos[obj_id] = t_obj.copy()
    carry_start = next((i for i, p in enumerate(phases) if p == PHASE_CARRY), len(phases))

    frames = []
    print("Rendering...")
    for fi, q in enumerate(q_smooth):
        env.set_qpos(q)
        if fi >= carry_start:
            op = env.get_ee_pos() + CARRY_OFFSET
            if obj_id >= 0: env._model.geom_pos[obj_id] = op
            if tag_id >= 0: env._model.geom_pos[tag_id] = op + [0,-0.036,0]
        mujoco.mj_forward(env._model, env._data)
        frame = _render_env(env)
        ph = phases[min(fi, len(phases)-1)]
        frame = add_text_overlay(frame, ph, pos=(WIDTH-120, 30), font_scale=0.55,
                                  color=(255, 220, 50))
        frame = add_text_overlay(frame, f"Reshelving (GPT)  err={final_error:.3f}m",
                                  pos=(8, 28), font_scale=0.5, color=(255, 255, 200))
        frame = add_progress_bar(frame, fi / max(1, len(q_smooth)-1))
        frames.append(frame)

    env._renderer.close()
    print(f"  Frames: {len(frames)}")

    imageio.mimsave(str(OUT_PATH), frames, fps=FPS, loop=0)
    sz = OUT_PATH.stat().st_size / 1e6
    print(f"Saved {OUT_PATH}  ({sz:.1f} MB, {len(frames)} frames)")
    reached = final_error < 0.12
    print(f"\nBox reaches shelf: {'YES' if reached else 'NO'}  (dist={final_error:.3f}m)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(seed=args.seed)
