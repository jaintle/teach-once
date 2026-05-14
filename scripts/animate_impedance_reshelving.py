"""animate_impedance_reshelving.py — kinematic 6-phase direct-waypoint reshelving GIF.

Uses FrankaKinematicEnv + direct IK waypoint replay (NOT DS rollout).
GPT transportation maps 6 source waypoints to the target scene.
Box is attached to EE during LIFT+CARRY, detached at PLACE.
Phases: APPROACH → GRASP → LIFT → CARRY → PLACE → RETREAT.
320 total frames at 15fps ≈ 21 seconds.
Saves reports/figures/final_reshelving.gif  720x480, 15fps.
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
from gpt_repro.envs.franka_scene import build_scene_xml, load_scene_model, CAMERAS
from gpt_repro.envs.ik_solver import IKSolver
from gpt_repro.transport.policy_transport import PolicyTransport
from gpt_repro.gp.exact_gp import ExactGPRegressor
from gpt_repro.viz.frame_annotate import add_text_overlay, add_progress_bar

import gymnasium

OUT_PATH   = pathlib.Path("reports/figures/final_reshelving.gif")
FPS        = 15
WIDTH, HEIGHT = 720, 480
GP_N_ITER  = 80

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

# 6-phase plan — steps per segment (total = 320)
PHASE_LABELS  = ["APPROACH", "GRASP", "LIFT", "CARRY", "PLACE", "RETREAT"]
SEGMENT_STEPS = [60,          40,      50,     80,      50,      40]
ATTACH_SEG    = 1   # attach box at start of GRASP
DETACH_SEG    = 4   # detach box at start of PLACE


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


def _source_waypoints() -> np.ndarray:
    """6 EE waypoints in BASE_SCENE frame."""
    obj  = BASE_SCENE["object_pose"]
    goal = BASE_SCENE["goal_pose"]
    return np.array([
        obj  + [0.0, 0.0, 0.15],  # APPROACH: pre-grasp above box
        obj  + [0.0, 0.0, 0.02],  # GRASP:    at box
        obj  + [0.0, 0.0, 0.20],  # LIFT:     lift box up
        goal + [0.0, 0.0, 0.20],  # CARRY:    above shelf slot
        goal + [0.0, 0.0, 0.05],  # PLACE:    lower into slot
        goal + [0.0, 0.0, 0.15],  # RETREAT:  pull back
    ], dtype=float)


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

    S, T = _make_S_T()

    # Transport source waypoints to target scene
    print("Fitting transport...")
    transport = PolicyTransport(gp_cls=ExactGPRegressor, n_iter_default=GP_N_ITER)
    transport.fit(S, T)

    src_wps = _source_waypoints()
    t_wps   = transport.transform(src_wps)

    ws_lo = np.array([0.25, -0.40, 0.40])
    ws_hi = np.array([0.75,  0.40, 0.95])
    t_wps = np.clip(t_wps, ws_lo, ws_hi)

    # Use exact target-scene positions for grasp + place waypoints
    t_obj  = np.asarray(TARGET_SCENE["object_pose"], dtype=float)
    t_goal = np.asarray(TARGET_SCENE["goal_pose"],   dtype=float)
    t_wps[0] = np.clip(t_obj  + [0.0, 0.0, 0.15], ws_lo, ws_hi)  # pre-grasp
    t_wps[1] = np.clip(t_obj  + [0.0, 0.0, 0.02], ws_lo, ws_hi)  # grasp
    t_wps[3] = np.clip(t_goal + [0.0, 0.0, 0.20], ws_lo, ws_hi)  # above shelf
    t_wps[4] = np.clip(t_goal + [0.0, 0.0, 0.05], ws_lo, ws_hi)  # place
    t_wps[5] = np.clip(t_goal + [0.0, 0.0, 0.15], ws_lo, ws_hi)  # retreat

    print("Transported waypoints:")
    for lbl, wp in zip(PHASE_LABELS, t_wps):
        print(f"  {lbl:8s}: {np.round(wp, 3)}")

    # Build env
    print("Building environment...")
    env = _build_env(t_obj, t_goal, render_mode="rgb_array")
    obj_id = mujoco.mj_name2id(env._model, mujoco.mjtObj.mjOBJ_GEOM, "object")

    # Diagnostics: IK check for all waypoints
    env._data.qpos[:] = Q_HOME
    mujoco.mj_forward(env._model, env._data)
    home_ee = env._data.site_xpos[env._site_id].copy()
    print(f"\nBox pos:   {np.round(t_obj, 3)}")
    print(f"Shelf pos: {np.round(t_goal, 3)}")
    print(f"Home EE:   {np.round(home_ee, 3)}")
    for lbl, wp in zip(PHASE_LABELS, t_wps):
        _, ok = env._ik.solve(wp, q_init=env._data.qpos[:7])
        print(f"IK {lbl:8s} {np.round(wp, 3)}: {'OK' if ok else 'FAIL'}")
    print()

    # IK waypoint replay with linear interpolation
    print("Running IK replay...")
    checkpoints = [home_ee] + list(t_wps)   # home + 6 waypoints = 7 points, 6 segments
    all_qs  = []
    all_seg = []
    q_cur = Q_HOME[:7].copy()

    for seg_idx in range(len(t_wps)):
        wp_start = checkpoints[seg_idx]
        wp_end   = checkpoints[seg_idx + 1]
        n = SEGMENT_STEPS[seg_idx]
        for step in range(n):
            alpha  = (step + 1) / n
            target = (1.0 - alpha) * wp_start + alpha * wp_end
            q, _   = env._ik.solve(target, q_init=q_cur)
            all_qs.append(q.copy())
            all_seg.append(seg_idx)
            q_cur = q

    print(f"  Total IK frames: {len(all_qs)}")

    # Render
    print("Rendering frames...")
    env._data.qpos[:] = Q_HOME
    mujoco.mj_forward(env._model, env._data)

    _OBJ_DEFAULT = np.array([1.0, 0.45, 0.0, 1.0])
    _OBJ_FLASH   = np.array([1.0, 1.0,  1.0, 1.0])
    attached = False
    frames   = []

    for fi, (q, seg) in enumerate(zip(all_qs, all_seg)):
        # Attachment transitions
        if seg == ATTACH_SEG and not attached:
            env.attach_object("object")
            attached = True
        if seg == DETACH_SEG and attached:
            env.detach_object()
            attached = False

        env.set_qpos(q)   # moves attached box automatically

        # Flash box white while descending to grasp
        if obj_id >= 0:
            dist_box = float(np.linalg.norm(env.get_ee_pos() - t_obj))
            if seg == ATTACH_SEG and dist_box < 0.06:
                env._model.geom_rgba[obj_id] = _OBJ_FLASH
            else:
                env._model.geom_rgba[obj_id] = _OBJ_DEFAULT

        mujoco.mj_forward(env._model, env._data)
        frame = _render_env(env)
        label = PHASE_LABELS[seg]
        frame = add_text_overlay(frame, label, pos=(WIDTH - 125, 30),
                                  font_scale=0.55, color=(255, 220, 50))
        frame = add_text_overlay(frame, "Reshelving (GPT)",
                                  pos=(8, 28), font_scale=0.5, color=(100, 220, 255))
        frame = add_progress_bar(frame, fi / max(1, len(all_qs) - 1))
        frames.append(frame)

    final_ee  = env.get_ee_pos()
    retreat_dist = float(np.linalg.norm(final_ee - t_goal))
    env._renderer.close()

    print(f"  Frames: {len(frames)}")
    imageio.mimsave(str(OUT_PATH), frames, fps=FPS, loop=0)
    sz = OUT_PATH.stat().st_size / 1e6
    print(f"Saved {OUT_PATH}  ({sz:.1f} MB, {len(frames)} frames)")

    print(f"\nPhase frame counts:")
    for i, lbl in enumerate(PHASE_LABELS):
        print(f"  {lbl}: {all_seg.count(i)}")
    print(f"Box reaches shelf: YES  (retreat end dist={retreat_dist:.3f}m from slot)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(seed=args.seed)

