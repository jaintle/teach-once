"""animate_armpose_best.py — single best-scene arm-pose animation.

Uses Scene 1 only (0.052m error, the success case).
Side camera 720x480. Large keypoint spheres (radius 0.06m).
Burns keypoint labels into each frame.
Saves reports/figures/franka_armpose_best.gif.
"""

import pathlib
import sys

import imageio
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))
from gpt_repro.utils.seeding import set_global_seed
from gpt_repro.envs.franka_env import FrankaKinematicEnv
from gpt_repro.policies.franka_demos import get_armpose_waypoints
from gpt_repro.transport.franka_rollout import record_franka_demo, transport_and_rollout_franka
from gpt_repro.viz.frame_annotate import add_text_overlay, add_progress_bar

SEED       = 0
SCENE_IDX  = 0          # Scene 1 (i=0) — the success case (0.052m)
GP_N_ITER  = 80
N_STEPS    = 200
OUT_PATH   = pathlib.Path("reports/figures/franka_armpose_best.gif")
FPS        = 20
WIDTH, HEIGHT = 720, 480

_BASE_KPS = {
    "shoulder": np.array([0.35, 0.00, 0.70]),
    "elbow":    np.array([0.47, 0.00, 0.80]),
    "wrist":    np.array([0.57, 0.00, 0.75]),
    "hand":     np.array([0.62, 0.00, 0.65]),
}

# Label positions (pixel x,y) for each keypoint — side camera 720x480
# Approximate positions: shoulder left, elbow mid, wrist right, hand lower-right
_KP_LABEL_POS = {
    "shoulder": (20,  80),
    "elbow":    (150, 55),
    "wrist":    (280, 70),
    "hand":     (350, 120),
}
_KP_COLORS = {
    "shoulder": (0,   230, 230),
    "elbow":    (230, 0,   230),
    "wrist":    (255, 230, 0),
    "hand":     (80,  120, 230),
}


def _make_S_T(base_kps, rng):
    kp_list = [base_kps["shoulder"], base_kps["elbow"],
               base_kps["wrist"],    base_kps["hand"]]
    S = np.array(kp_list)
    delta = rng.uniform(-0.06, 0.06, S.shape)
    delta[:, 1] = 0
    return S, S + delta


def _randomize_arm(base_kps, rng):
    delta = rng.uniform(-0.05, 0.05, 3); delta[1] = 0
    return {k: v + delta for k, v in base_kps.items()}


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    set_global_seed(SEED)

    # Record base demo (large_kp_spheres=True so spheres are visible in render)
    print("Recording base arm-pose demo …")
    base_env = FrankaKinematicEnv("armpose", render_mode=None,
                                   width=WIDTH, height=HEIGHT,
                                   scene_kwargs={"large_kp_spheres": True})
    base_env.reset(seed=SEED)
    waypoints = get_armpose_waypoints(_BASE_KPS)
    base_demo = record_franka_demo(base_env, waypoints)
    base_env.close()
    print(f"  Demo: {len(base_demo['x'])} steps, IK ok={base_demo['ik_success'].mean()*100:.0f}%")

    # Scene 1 (i=0)
    rng_i = np.random.default_rng(SEED + SCENE_IDX)
    new_kps = _randomize_arm(_BASE_KPS, rng_i)
    rng_i2  = np.random.default_rng(SEED + SCENE_IDX)
    _randomize_arm(_BASE_KPS, rng_i2)   # advance rng
    S, T = _make_S_T(_BASE_KPS, rng_i2)

    print("Scene 1 …")
    env = FrankaKinematicEnv("armpose", render_mode="rgb_array",
                              width=WIDTH, height=HEIGHT,
                              scene_kwargs={"large_kp_spheres": True})
    env.set_camera("side")

    res = transport_and_rollout_franka(
        demo=base_demo, S=S, T=T, env=env,
        gp_n_iter=GP_N_ITER, n_steps=N_STEPS,
        success_threshold=0.10, attractor_gain=1.2,
        seed=SEED + SCENE_IDX,
    )
    print(f"  err={res['final_error']:.3f}m  success={res['success']}  ik_fail={res['ik_fail_rate']*100:.0f}%")
    env.close()

    # Annotate all frames with keypoint labels
    frames = []
    for fi, frame in enumerate(res["frames"]):
        frame = add_text_overlay(frame, f"Arm-pose | err:{res['final_error']:.3f}m  {'✓' if res['success'] else ''}",
                                  pos=(8, 24), font_scale=0.55, color=(255, 255, 255))
        # Keypoint labels
        for kname in ["shoulder", "elbow", "wrist", "hand"]:
            px, py = _KP_LABEL_POS[kname]
            col    = _KP_COLORS[kname]
            frame  = add_text_overlay(frame, kname,
                                       pos=(px, py), font_scale=0.42, color=col)
        progress = fi / max(1, len(res["frames"]) - 1)
        is_last  = (fi == len(res["frames"]) - 1)
        frame = add_progress_bar(frame, progress,
                                  success=res["success"] if is_last else None)
        frames.append(frame)

    if not frames:
        print("ERROR: no frames rendered"); return

    imageio.mimwrite(str(OUT_PATH), frames, fps=FPS, loop=0)
    sz = OUT_PATH.stat().st_size / 1024
    print(f"\nSaved {OUT_PATH.name}: {sz:.0f} KB  ({len(frames)} frames)")


if __name__ == "__main__":
    main()
