"""animate_highlight_reel_best.py — 3-panel side-by-side highlight reel.

Renders one scene per task (best cases) and stitches into 1440x360 (3×480x360).
Title bars above each panel (30px).
Saves reports/figures/highlight_reel_best.gif at 15fps.
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
from gpt_repro.envs.franka_scene import build_scene_xml, load_scene_model, CAMERAS
from gpt_repro.policies.franka_demos import get_reshelving_waypoints, get_cleaning_waypoints, get_armpose_waypoints
from gpt_repro.transport.franka_rollout import record_franka_demo, transport_and_rollout_franka
from gpt_repro.transport.policy_transport import PolicyTransport
from gpt_repro.policies.ds_policy import GPDynamicalSystem
from gpt_repro.gp.exact_gp import ExactGPRegressor
from gpt_repro.viz.frame_annotate import add_text_overlay, add_progress_bar

SEED = 0
GP_N_ITER = 80
N_STEPS   = 200
PANEL_W, PANEL_H = 480, 360
TITLE_H   = 30
OUT_PATH  = pathlib.Path("reports/figures/highlight_reel_best.gif")
FPS       = 15

TITLES = [
    "Reshelving: GPT transports pick-place to new object/goal",
    "Cleaning: GPT transports scan to shifted surface",
    "Arm-pose: GPT transports path to new arm configuration",
]


# ── Shared utils ─────────────────────────────────────────────────────────────

def _resize_frame(frame, w, h):
    """Nearest-neighbour resize (H, W, 3) → (h, w, 3)."""
    src_h, src_w = frame.shape[:2]
    xs = (np.arange(w) * src_w / w).astype(int)
    ys = (np.arange(h) * src_h / h).astype(int)
    return frame[np.ix_(ys, xs)]


def _add_title_bar(frame, text, bar_h=TITLE_H):
    """Prepend a dark title bar to the frame."""
    H, W = frame.shape[:2]
    bar  = np.zeros((bar_h, W, 3), dtype=np.uint8)
    bar[:] = [30, 30, 50]
    from gpt_repro.viz.frame_annotate import add_text_overlay
    bar = add_text_overlay(bar, text, pos=(6, bar_h - 7), font_scale=0.35,
                            color=(220, 220, 220))
    return np.vstack([bar, frame])


# ── Reshelving scene 2 ────────────────────────────────────────────────────────

def _reshelving_frames():
    BASE_SCENE = {
        "object_pose": np.array([0.50, 0.00, 0.63]),
        "goal_pose":   np.array([0.30, 0.10, 0.75]),
    }
    SCENE_IDX = 1   # Scene 2 (success)

    def _randomize(base, rng):
        op = rng.uniform(-0.07, 0.07, 3); op[2] = 0
        gp = rng.uniform(-0.07, 0.07, 3); gp[2] = rng.uniform(0.0, 0.06)
        return {"object_pose": base["object_pose"] + op, "goal_pose": base["goal_pose"] + gp}

    def _make_ST(base, rng):
        obj  = base["object_pose"]; goal = base["goal_pose"]
        S = np.array([obj+[0.05,0.05,0], obj+[-0.05,0.05,0],
                      goal+[0.05,0.05,0], goal+[-0.05,0.05,0],
                      obj+[0,0,0.1], goal+[0,0,0.1]])
        d = rng.uniform(-0.08,0.08,S.shape); d[:,2]*=0.5
        return S, S+d

    print("--- reshelving (scene 2) ---")
    base_env = FrankaKinematicEnv("reshelving", render_mode=None, width=720, height=480)
    base_env.reset(seed=SEED)
    base_demo = record_franka_demo(base_env, get_reshelving_waypoints(BASE_SCENE))
    base_env.close()

    rng_i = np.random.default_rng(SEED + SCENE_IDX)
    new_scene = _randomize(BASE_SCENE, rng_i)
    rng_i2 = np.random.default_rng(SEED + SCENE_IDX)
    _randomize(BASE_SCENE, rng_i2)
    S, T = _make_ST(BASE_SCENE, rng_i2)

    env = FrankaKinematicEnv("reshelving", render_mode="rgb_array", width=720, height=480)
    env.set_camera("front")
    res = transport_and_rollout_franka(
        demo=base_demo, S=S, T=T, env=env,
        gp_n_iter=GP_N_ITER, n_steps=N_STEPS,
        success_threshold=0.08, attractor_gain=1.5, seed=SEED+SCENE_IDX)
    env.close()
    print(f"  err={res['final_error']:.3f}m  success={res['success']}")
    return res["frames"], res["success"], res["final_error"]


# ── Cleaning scene 1 ─────────────────────────────────────────────────────────

def _cleaning_frames():
    BASE_SCENE = {
        "surface_center":    np.array([0.50, 0.00, 0.64]),
        "surface_half_size": np.array([0.12, 0.12]),
    }

    def _make_ST(base, rng):
        cx, cy, cz = base["surface_center"]; hw, hd = base["surface_half_size"]
        S = np.array([[cx-hw,cy-hd,cz],[cx+hw,cy-hd,cz],[cx+hw,cy+hd,cz],
                      [cx-hw,cy+hd,cz],[cx,cy,cz]])
        d = rng.uniform(-0.06,0.06,S.shape); d[:,2]*=0.3
        return S, S+d

    print("--- cleaning (scene 1) ---")
    base_env = FrankaKinematicEnv("cleaning", render_mode=None, width=720, height=480)
    base_env.reset(seed=SEED)
    base_demo = record_franka_demo(base_env, get_cleaning_waypoints(BASE_SCENE))
    base_env.close()

    x_demo    = base_demo["x"].astype(float)
    xdot_demo = base_demo["xdot"].astype(float)
    rng_0 = np.random.default_rng(SEED)
    S, T  = _make_ST(BASE_SCENE, rng_0)

    env = FrankaKinematicEnv("cleaning", render_mode="rgb_array", width=720, height=480)
    env.set_camera("quarter")
    ws_lo, ws_hi = env.get_workspace_bounds()

    transport = PolicyTransport(gp_cls=ExactGPRegressor, n_iter_default=GP_N_ITER)
    transport.fit(S, T)
    x_t  = transport.transform(x_demo)
    xd_t = transport.transform_velocity(x_demo, xdot_demo)
    ds   = GPDynamicalSystem(gp_cls=ExactGPRegressor, n_iter_default=GP_N_ITER)
    ds.fit(x_t, xd_t)

    _pv, _ = ds.predict(x_t, return_std=True)
    dv = float(np.linalg.norm(xd_t,axis=1).mean())+1e-8
    pv = float(np.linalg.norm(_pv, axis=1).mean())+1e-8
    vs = float(np.clip(dv/pv,1,50)) if pv < dv*0.9 else 1.0

    x_goal = x_t[-1]
    env.reset(); env.set_ee_pos(np.clip(x_t[0], ws_lo, ws_hi))
    xs=[env.get_ee_pos().copy()]; qs=[env._data.qpos[:7].copy()]
    for _ in range(N_STEPS):
        obs=xs[-1]; vel=ds.predict(obs[np.newaxis],return_std=False)
        if vel.ndim==2: vel=vel[0]
        vel = vel + 0.8*(x_goal-obs)
        env.step(np.clip(obs+(vel*vs)*0.05, ws_lo, ws_hi))
        xs.append(env.get_ee_pos().copy()); qs.append(env._data.qpos[:7].copy())

    q_smooth = gaussian_filter1d(np.array(qs), sigma=1.5, axis=0)
    final_err = float(np.linalg.norm(np.array(xs)[-1] - x_t[-1]))
    print(f"  err={final_err:.3f}m")

    frames = []
    for fi, q in enumerate(q_smooth):
        env.set_qpos(q); mujoco.mj_forward(env._model, env._data)
        frame = env.render()
        if frame is not None: frames.append(frame)
    env.close()
    return frames, False, final_err


# ── Armpose scene 1 ───────────────────────────────────────────────────────────

def _armpose_frames():
    _BASE_KPS = {
        "shoulder": np.array([0.35, 0.00, 0.70]),
        "elbow":    np.array([0.47, 0.00, 0.80]),
        "wrist":    np.array([0.57, 0.00, 0.75]),
        "hand":     np.array([0.62, 0.00, 0.65]),
    }

    def _randomize(base, rng):
        d=rng.uniform(-0.05,0.05,3); d[1]=0
        return {k: v+d for k,v in base.items()}

    def _make_ST(base, rng):
        S=np.array([base["shoulder"],base["elbow"],base["wrist"],base["hand"]])
        d=rng.uniform(-0.06,0.06,S.shape); d[:,1]=0
        return S, S+d

    print("--- armpose (scene 1) ---")
    base_env = FrankaKinematicEnv("armpose", render_mode=None, width=720, height=480,
                                   scene_kwargs={"large_kp_spheres": True})
    base_env.reset(seed=SEED)
    base_demo = record_franka_demo(base_env, get_armpose_waypoints(_BASE_KPS))
    base_env.close()

    rng_i  = np.random.default_rng(SEED)
    _randomize(_BASE_KPS, rng_i)
    rng_i2 = np.random.default_rng(SEED)
    _randomize(_BASE_KPS, rng_i2)
    S, T = _make_ST(_BASE_KPS, rng_i2)

    env = FrankaKinematicEnv("armpose", render_mode="rgb_array", width=720, height=480,
                              scene_kwargs={"large_kp_spheres": True})
    env.set_camera("side")
    res = transport_and_rollout_franka(
        demo=base_demo, S=S, T=T, env=env,
        gp_n_iter=GP_N_ITER, n_steps=N_STEPS,
        success_threshold=0.10, attractor_gain=1.2, seed=SEED)
    env.close()
    print(f"  err={res['final_error']:.3f}m  success={res['success']}")
    return res["frames"], res["success"], res["final_error"]


# ── Stitch ─────────────────────────────────────────────────────────────────────

def _stitch(frames_a, suc_a, err_a,
            frames_b, suc_b, err_b,
            frames_c, suc_c, err_c):
    max_n = max(len(frames_a), len(frames_b), len(frames_c))
    out = []
    for i in range(max_n):
        fa = frames_a[min(i, len(frames_a)-1)]
        fb = frames_b[min(i, len(frames_b)-1)]
        fc = frames_c[min(i, len(frames_c)-1)]

        # Resize each panel to PANEL_W x PANEL_H
        fa = _resize_frame(fa, PANEL_W, PANEL_H)
        fb = _resize_frame(fb, PANEL_W, PANEL_H)
        fc = _resize_frame(fc, PANEL_W, PANEL_H)

        # Add title bar
        fa = _add_title_bar(fa, TITLES[0])
        fb = _add_title_bar(fb, TITLES[1])
        fc = _add_title_bar(fc, TITLES[2])

        row = np.hstack([fa, fb, fc])
        out.append(row)
    return out


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    set_global_seed(SEED)

    fr_a, suc_a, err_a = _reshelving_frames()
    fr_b, suc_b, err_b = _cleaning_frames()
    fr_c, suc_c, err_c = _armpose_frames()

    if not fr_a or not fr_b or not fr_c:
        print("ERROR: one or more tasks produced no frames"); return

    frames = _stitch(fr_a, suc_a, err_a, fr_b, suc_b, err_b, fr_c, suc_c, err_c)

    # Subsample to stay under 15MB
    step = 1
    while len(frames[::step]) * frames[0].nbytes // 10 > 15*1024*1024 and step < 8:
        step += 1
    sub = frames[::step]

    imageio.mimwrite(str(OUT_PATH), sub, fps=FPS, loop=0)
    sz = OUT_PATH.stat().st_size / 1024
    print(f"\nSaved {OUT_PATH.name}: {sz:.0f} KB  ({len(sub)} frames)")

    try:
        import imageio_ffmpeg  # noqa
        mp4 = OUT_PATH.with_suffix(".mp4")
        imageio.mimwrite(str(mp4), frames, fps=FPS, codec="libx264", quality=6, macro_block_size=1)
        print(f"  {mp4.name}: {mp4.stat().st_size/1024:.0f} KB")
    except Exception as e:
        print(f"  MP4 skipped ({e})")


if __name__ == "__main__":
    main()
