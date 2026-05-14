"""animate_reshelving_best.py — single best-scene reshelving animation.

Uses Scene 2 (i=1, seed 1) — lowest error 0.077m, success case.
Front camera 720x480. All frames (no subsampling).
Adds small green waypoint markers at pick + place positions.
Saves reports/figures/franka_reshelving_best.gif.
"""

import pathlib
import sys

import imageio
import mujoco
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))
from gpt_repro.utils.seeding import set_global_seed
from gpt_repro.envs.franka_env import FrankaKinematicEnv
from gpt_repro.envs.franka_scene import build_scene_xml, load_scene_model, CAMERAS
from gpt_repro.policies.franka_demos import get_reshelving_waypoints
from gpt_repro.transport.franka_rollout import record_franka_demo, transport_and_rollout_franka
from gpt_repro.viz.frame_annotate import add_text_overlay, add_progress_bar

SEED       = 0
SCENE_IDX  = 1          # Scene 2 (i=1) — the success case
GP_N_ITER  = 80
N_STEPS    = 200
OUT_PATH   = pathlib.Path("reports/figures/franka_reshelving_best.gif")
FPS        = 20
WIDTH, HEIGHT = 720, 480

BASE_SCENE = {
    "object_pose": np.array([0.50, 0.00, 0.63]),
    "goal_pose":   np.array([0.30, 0.10, 0.75]),
}

# Replicate scene-2 randomisation (same RNG as animate_franka_reshelving.py)
def _randomize_scene(base, rng):
    obj_perturb  = rng.uniform(-0.07, 0.07, 3); obj_perturb[2]  = 0
    goal_perturb = rng.uniform(-0.07, 0.07, 3); goal_perturb[2] = rng.uniform(0.0, 0.06)
    return {
        "object_pose": base["object_pose"] + obj_perturb,
        "goal_pose":   base["goal_pose"]   + goal_perturb,
    }

def _make_S_T(base, rng):
    obj  = np.asarray(base["object_pose"])
    goal = np.asarray(base["goal_pose"])
    S = np.array([
        obj  + [0.05,  0.05, 0.0],
        obj  + [-0.05, 0.05, 0.0],
        goal + [0.05,  0.05, 0.0],
        goal + [-0.05, 0.05, 0.0],
        obj  + [0.0,   0.0,  0.1],
        goal + [0.0,   0.0,  0.1],
    ])
    delta = rng.uniform(-0.08, 0.08, size=S.shape)
    delta[:, 2] *= 0.5
    return S, S + delta


def _add_waypoint_marker(xml: str, name: str, pos: np.ndarray, rgba="0.2 0.9 0.2 0.85") -> str:
    """Inject a small green sphere into the worldbody of the XML."""
    sphere_xml = (
        f'    <geom name="{name}" type="sphere" size="0.025"\n'
        f'          pos="{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}" rgba="{rgba}"/>\n'
    )
    return xml.replace("  </worldbody>", sphere_xml + "  </worldbody>", 1)


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    set_global_seed(SEED)

    # Record base demo (render=None for speed)
    print("Recording base demo …")
    base_env = FrankaKinematicEnv("reshelving", render_mode=None, width=WIDTH, height=HEIGHT)
    base_env.reset(seed=SEED)
    waypoints = get_reshelving_waypoints(BASE_SCENE)
    base_demo = record_franka_demo(base_env, waypoints)
    base_env.close()
    print(f"  Demo: {len(base_demo['x'])} steps, IK ok={base_demo['ik_success'].mean()*100:.0f}%")

    # Build scene 2 (i=1)
    trial_seed = SEED + SCENE_IDX
    rng_i = np.random.default_rng(trial_seed)
    new_scene = _randomize_scene(BASE_SCENE, rng_i)
    rng_i = np.random.default_rng(trial_seed)          # reset to get same S/T
    _randomize_scene(BASE_SCENE, rng_i)                # advance rng past scene randomize
    rng_i2 = np.random.default_rng(trial_seed)
    _randomize_scene(BASE_SCENE, rng_i2)               # consumed in _randomize_scene
    S, T = _make_S_T(BASE_SCENE, rng_i2)

    pick_pos  = new_scene["object_pose"].copy()
    place_pos = new_scene["goal_pose"].copy()
    print(f"Scene 2: pick={np.round(pick_pos,3)}  place={np.round(place_pos,3)}")

    # Build env with waypoint markers injected
    from gpt_repro.envs.franka_scene import build_scene_xml, load_scene_model
    xml = build_scene_xml("reshelving",
                           object_pos=tuple(pick_pos))
    xml = _add_waypoint_marker(xml, "marker_pick",  pick_pos  + [0, 0, 0.04])
    xml = _add_waypoint_marker(xml, "marker_place", place_pos + [0, 0, 0.04])

    # Load model into env (rebuild env with custom xml)
    env = FrankaKinematicEnv.__new__(FrankaKinematicEnv)
    from gpt_repro.envs.franka_env import Q_HOME
    from gpt_repro.envs.ik_solver import IKSolver
    import gymnasium
    env.task = "reshelving"
    env._render_mode = "rgb_array"
    env._width  = WIDTH
    env._height = HEIGHT
    env.dt          = 0.002
    env.control_dt  = 0.05
    env._model, env._data = load_scene_model(xml)
    env._site_id = mujoco.mj_name2id(env._model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    env._ik = IKSolver(env._model, env._data, ee_site_name="attachment_site",
                       max_iter=200, tol=1e-3, damping=1e-4, nullspace_gain=0.5)
    env._data.qpos[:] = Q_HOME
    mujoco.mj_forward(env._model, env._data)
    env._ee_home = env._data.site_xpos[env._site_id].copy()
    obs_low  = np.full(3, -2.0); obs_high = np.full(3, 2.0)
    env.observation_space = gymnasium.spaces.Box(low=obs_low, high=obs_high, dtype=np.float64)
    env.action_space      = gymnasium.spaces.Box(low=obs_low.copy(), high=obs_high.copy(), dtype=np.float64)
    env._renderer         = None
    env._attached_geom_id = None
    env._attach_offset    = None
    env._cam_lookat, env._cam_distance, env._cam_elevation, env._cam_azimuth = CAMERAS["front"]

    # Run transport + rollout
    print("Running GPT transport + rollout …")
    res = transport_and_rollout_franka(
        demo=base_demo, S=S, T=T, env=env,
        gp_n_iter=GP_N_ITER, n_steps=N_STEPS,
        success_threshold=0.08, attractor_gain=1.5,
        seed=trial_seed,
    )
    print(f"  err={res['final_error']:.3f}m  success={res['success']}  ik_fail={res['ik_fail_rate']*100:.0f}%")

    # Re-render with 3-phase attachment + markers
    rollout_x = res["rollout_x"]
    q_arr     = res["rollout_q"]
    transported_obj  = res["transport"].transform(new_scene["object_pose"].reshape(1,3))[0]
    transported_goal = res["transport"].transform(new_scene["goal_pose"].reshape(1,3))[0]

    dist_to_obj  = np.linalg.norm(rollout_x - transported_obj,  axis=1)
    dist_to_goal = np.linalg.norm(rollout_x - transported_goal, axis=1)
    grasp_candidates = np.where(dist_to_obj  < 0.06)[0]
    place_candidates = np.where(dist_to_goal < 0.08)[0]
    grasp_fi = int(grasp_candidates[0])  if len(grasp_candidates) else len(q_arr) // 4
    place_fi = int(place_candidates[0])  if len(place_candidates) else 3 * len(q_arr) // 4
    _CARRY_OFFSET = np.array([0, 0, -0.03])

    obj_geom_id = mujoco.mj_name2id(env._model, mujoco.mjtObj.mjOBJ_GEOM, "object")
    obj_tag_id  = mujoco.mj_name2id(env._model, mujoco.mjtObj.mjOBJ_GEOM, "object_tag")
    pick_marker_id  = mujoco.mj_name2id(env._model, mujoco.mjtObj.mjOBJ_GEOM, "marker_pick")
    place_marker_id = mujoco.mj_name2id(env._model, mujoco.mjtObj.mjOBJ_GEOM, "marker_place")

    frames = []
    for fi, q in enumerate(q_arr):
        env.set_qpos(q)
        ee_pos = env.get_ee_pos()

        # Phase A: move marker_pick to transported object position; box stays
        if fi < grasp_fi:
            if obj_geom_id >= 0:
                env._model.geom_pos[obj_geom_id] = transported_obj
            if obj_tag_id >= 0:
                env._model.geom_pos[obj_tag_id]  = transported_obj + [0, -0.026, 0]
            # Make pick marker glow white when EE is close
            if pick_marker_id >= 0:
                dist_now = np.linalg.norm(ee_pos - transported_obj)
                if dist_now < 0.10:
                    env._model.geom_rgba[pick_marker_id] = [1.0, 1.0, 1.0, 0.95]
        # Phase B: box follows EE (carry)
        elif fi < place_fi:
            box_pos = ee_pos + _CARRY_OFFSET
            if obj_geom_id >= 0:
                env._model.geom_pos[obj_geom_id] = box_pos
            if obj_tag_id >= 0:
                env._model.geom_pos[obj_tag_id]  = box_pos + [0, -0.026, 0]
            # Dim pick marker
            if pick_marker_id >= 0:
                env._model.geom_rgba[pick_marker_id] = [0.2, 0.9, 0.2, 0.3]
        # Phase C: box stays at place position; light up place marker
        else:
            if obj_geom_id >= 0:
                env._model.geom_pos[obj_geom_id] = transported_goal + _CARRY_OFFSET
            if obj_tag_id >= 0:
                env._model.geom_pos[obj_tag_id]  = transported_goal + _CARRY_OFFSET + [0, -0.026, 0]
            if place_marker_id >= 0:
                env._model.geom_rgba[place_marker_id] = [1.0, 1.0, 1.0, 0.95]

        mujoco.mj_forward(env._model, env._data)
        frame = env.render()
        if frame is None:
            continue

        phase = "approach" if fi < grasp_fi else ("carry" if fi < place_fi else "place")
        frame = add_text_overlay(frame, f"Reshelving | {phase} | err:{res['final_error']:.3f}m",
                                 pos=(8, 24), font_scale=0.50)
        progress = fi / max(1, len(q_arr) - 1)
        is_last  = (fi == len(q_arr) - 1)
        frame = add_progress_bar(frame, progress, success=res["success"] if is_last else None)
        frames.append(frame)

    env.close()

    if not frames:
        print("ERROR: no frames rendered"); return

    imageio.mimwrite(str(OUT_PATH), frames, fps=FPS, loop=0)
    sz = OUT_PATH.stat().st_size / 1024
    print(f"\nSaved {OUT_PATH.name}: {sz:.0f} KB  ({len(frames)} frames)")


if __name__ == "__main__":
    main()
