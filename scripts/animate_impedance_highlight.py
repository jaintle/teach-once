"""animate_impedance_highlight.py — 3-panel kinematic highlight GIF.

Loads final_reshelving.gif, final_cleaning.gif, final_armpose.gif,
resizes each to 480x360, stitches side-by-side → 1440x360.
Pads shorter GIFs to equal length.
Saves reports/figures/final_highlight.gif
"""

import pathlib
import sys

import imageio
import numpy as np
from PIL import Image

OUT_PATH   = pathlib.Path("reports/figures/final_highlight.gif")
PANEL_W, PANEL_H = 480, 360
FPS = 12

SOURCES = [
    ("Reshelving",  pathlib.Path("reports/figures/final_reshelving.gif")),
    ("Cleaning",    pathlib.Path("reports/figures/final_cleaning.gif")),
    ("Arm-Pose",    pathlib.Path("reports/figures/final_armpose.gif")),
]

LABEL_COLORS = {
    "Reshelving": (255, 200, 60),
    "Cleaning":   (255, 220, 80),
    "Arm-Pose":   (100, 220, 255),
}


def _load_gif(path):
    """Return list of numpy RGB frames (H, W, 3)."""
    reader = imageio.get_reader(str(path))
    frames = [np.array(f)[:, :, :3] for f in reader]
    reader.close()
    return frames


def _resize_frame(frame, w, h):
    img = Image.fromarray(frame.astype(np.uint8))
    img = img.resize((w, h), Image.LANCZOS)
    return np.array(img)


def _add_label(frame, text, color):
    """Draw a simple text label in the top-left corner."""
    try:
        import cv2
        cv2.putText(frame, text, (8, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    except Exception:
        pass
    return frame


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Load and resize all panels
    panels = []
    for label, src in SOURCES:
        if not src.exists():
            print(f"  WARNING: {src} not found, using blank panel.")
            blank = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
            panels.append(([blank], label))
            continue
        raw_frames = _load_gif(src)
        resized = [_resize_frame(f, PANEL_W, PANEL_H) for f in raw_frames]
        print(f"  {label}: {len(resized)} frames from {src.name}")
        panels.append((resized, label))

    # Pad to equal length (repeat last frame)
    max_len = max(len(p[0]) for p in panels)
    padded = []
    for (frames, label) in panels:
        if len(frames) < max_len:
            frames = frames + [frames[-1]] * (max_len - len(frames))
        padded.append((frames[:max_len], label))

    # Stitch frames
    print(f"Stitching {max_len} frames (3 x {PANEL_W}x{PANEL_H})...")
    out_frames = []
    for fi in range(max_len):
        row = []
        for (frames, label) in padded:
            f = frames[fi].copy()
            col = LABEL_COLORS.get(label, (255, 255, 255))
            f = _add_label(f, label, col)
            row.append(f)
        stitched = np.concatenate(row, axis=1)  # (PANEL_H, 3*PANEL_W, 3)
        out_frames.append(stitched)

    imageio.mimsave(str(OUT_PATH), out_frames, fps=FPS, loop=0)
    sz = OUT_PATH.stat().st_size / 1e6
    print(f"Saved {OUT_PATH}  ({sz:.1f} MB, {max_len} frames, "
          f"{3*PANEL_W}x{PANEL_H})")


if __name__ == "__main__":
    main()

OUT_PATH = pathlib.Path("reports/figures/final_highlight.gif")
FPS = 15
W, H = 480, 480   # per panel

TASKS = {
    "reshelving": {
        "camera": "front",
        "color": (255, 255, 100),
        "base_scene": {
            "object_pose": np.array([0.50, 0.00, 0.63]),
            "goal_pose":   np.array([0.30, 0.10, 0.75]),
        },
        "target_scene": {
            "object_pose": np.array([0.48, 0.08, 0.63]),
            "goal_pose":   np.array([0.32, -0.08, 0.75]),
        },
        "K_s_fn": get_reshelving_stiffness,
        "waypoints_fn": get_reshelving_waypoints,
    },
    "cleaning": {
        "camera": "quarter",
        "color": (255, 220, 80),
        "base_scene": {
            "surface_center":    np.array([0.50, 0.00, 0.64]),
            "surface_half_size": np.array([0.12, 0.12]),
        },
        "target_scene": {
            "surface_center":    np.array([0.50, 0.08, 0.64]),
            "surface_half_size": np.array([0.12, 0.12]),
        },
        "K_s_fn": get_cleaning_stiffness,
        "waypoints_fn": get_cleaning_waypoints,
    },
    "armpose": {
        "camera": "side",
        "color": (100, 220, 255),
        "base_scene": {
            "shoulder": np.array([0.35, 0.00, 0.70]),
            "elbow":    np.array([0.47, 0.00, 0.80]),
            "wrist":    np.array([0.57, 0.00, 0.75]),
            "hand":     np.array([0.62, 0.00, 0.65]),
        },
        "target_scene": {
            "shoulder": np.array([0.35, 0.06, 0.70]),
            "elbow":    np.array([0.47, 0.06, 0.80]),
            "wrist":    np.array([0.57, 0.06, 0.75]),
            "hand":     np.array([0.62, 0.06, 0.65]),
        },
        "K_s_fn": get_armpose_stiffness,
        "waypoints_fn": get_armpose_waypoints,
    },
}


def _S_T_for_task(task_name, base_scene, target_scene):
    if task_name == "reshelving":
        obj_s = np.asarray(base_scene["object_pose"])
        goal_s = np.asarray(base_scene["goal_pose"])
        obj_t = np.asarray(target_scene["object_pose"])
        goal_t = np.asarray(target_scene["goal_pose"])
        S = np.array([
            obj_s + [0.05, 0.05, 0.0], obj_s + [-0.05, 0.05, 0.0],
            goal_s + [0.05, 0.05, 0.0], goal_s + [-0.05, 0.05, 0.0],
            obj_s + [0, 0, 0.1], goal_s + [0, 0, 0.1],
        ])
        T = np.array([
            obj_t + [0.05, 0.05, 0.0], obj_t + [-0.05, 0.05, 0.0],
            goal_t + [0.05, 0.05, 0.0], goal_t + [-0.05, 0.05, 0.0],
            obj_t + [0, 0, 0.1], goal_t + [0, 0, 0.1],
        ])
    elif task_name == "cleaning":
        c_s = np.asarray(base_scene["surface_center"])
        c_t = np.asarray(target_scene["surface_center"])
        h = float(base_scene["surface_half_size"][0])
        offsets = np.array([[h,0,.02],[-h,0,.02],[0,h,.02],[0,-h,.02],[h,h,.02],[-h,-h,.02]])
        S = c_s + offsets
        T = c_t + offsets
    else:  # armpose
        keys = ["shoulder", "elbow", "wrist", "hand"]
        S = np.array([base_scene[k] for k in keys])
        T = np.array([target_scene[k] for k in keys])
        mid_s = (S[0] + S[1]) * 0.5
        mid_t = (T[0] + T[1]) * 0.5
        S = np.vstack([S, mid_s])
        T = np.vstack([T, mid_t])
    return S, T


def run_task(task_name, cfg):
    """Return (frames, final_error) for one task. Each frame is W×H."""
    set_global_seed(SEED)
    base_scene = cfg["base_scene"]
    target_scene = cfg["target_scene"]
    K_s = cfg["K_s_fn"]()
    waypoints_fn = cfg["waypoints_fn"]

    # Record demo
    kin_env = FrankaKinematicEnv(task_name, render_mode=None, width=W, height=H)
    kin_env.reset(seed=SEED)
    waypoints = waypoints_fn(base_scene)
    base_demo = record_franka_demo(kin_env, waypoints)
    kin_env.close()

    S, T = _S_T_for_task(task_name, base_scene, target_scene)

    # Fit transport + DS
    transport = PolicyTransport(gp_cls=ExactGPRegressor, n_iter_default=GP_N_ITER)
    transport.fit(S, T)
    x_t = transport.transform(base_demo["x"])
    xdot_t = transport.transform_velocity(base_demo["x"], base_demo["xdot"])
    ds = GPDynamicalSystem(gp_cls=ExactGPRegressor, n_iter_default=GP_N_ITER)
    ds.fit(x_t, xdot_t)

    x_goal = x_t[-1].copy()
    diag_k = np.diag(K_s)
    dt = 1.0 / 500.0

    # Render
    render_env = FrankaImpedanceEnv(
        task_name, render_mode="rgb_array", dt=0.002, control_hz=500, width=W, height=H
    )
    render_env.set_camera(cfg["camera"])
    obs, _ = render_env.reset(seed=SEED)
    x_cur = obs[:3].copy()

    final_err = None
    frames = []
    for step_i in range(N_STEPS + 1):
        frame = render_env.render()
        if frame is not None:
            frame = add_text_overlay(
                frame, f"{task_name.capitalize()}  step={step_i}",
                pos=(8, 25), font_scale=0.5, color=cfg["color"],
            )
            frame = add_progress_bar(frame, step_i / N_STEPS)
            frames.append(frame)

        vel = ds.predict(x_cur[None, :], return_std=False).squeeze()
        vel = vel + 0.5 * (x_goal - x_cur)
        x_des = x_cur + vel * dt
        action = np.concatenate([x_des, vel, diag_k])
        obs, _, terminated, _, _ = render_env.step(action)
        if terminated:
            break
        x_cur = obs[:3].copy()

    render_env.close()
    final_err = float(np.linalg.norm(x_cur - x_goal))
    print(f"  {task_name}: {len(frames)} frames, final_err={final_err:.4f} m")
    return frames, final_err


def stitch_row(frame_lists):
    """Stitch panels side by side, pad shorter lists with last frame."""
    max_len = max(len(fl) for fl in frame_lists)
    rows = []
    for i in range(max_len):
        panels = []
        for fl in frame_lists:
            idx = min(i, len(fl) - 1)
            panels.append(fl[idx])
        row = np.concatenate(panels, axis=1)  # (H, 3*W, 3)
        rows.append(row)
    return rows


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_frames = {}
    all_errors = {}
    for task_name, cfg in TASKS.items():
        print(f"\n[{task_name}]")
        frames, err = run_task(task_name, cfg)
        all_frames[task_name] = frames
        all_errors[task_name] = err

    print("\nStitching 3-panel highlight reel...")
    frame_lists = [all_frames[t] for t in ["reshelving", "cleaning", "armpose"]]
    stitched = stitch_row(frame_lists)

    # Add summary overlay on first frame
    summary = (
        f"GPT Impedance  |  "
        f"reshelv={all_errors['reshelving']:.3f}m  "
        f"clean={all_errors['cleaning']:.3f}m  "
        f"arm={all_errors['armpose']:.3f}m"
    )
    stitched[0] = add_text_overlay(
        stitched[0], summary,
        pos=(10, 30), font_scale=0.55, color=(255, 255, 255),
    )

    print(f"Saving {OUT_PATH} ...")
    imageio.mimsave(str(OUT_PATH), stitched, fps=FPS, loop=0)
    size_mb = OUT_PATH.stat().st_size / 1e6
    print(f"  Saved {OUT_PATH}  ({size_mb:.1f} MB, {len(stitched)} frames)")


if __name__ == "__main__":
    main()
