"""Render static high-resolution scenes for each task — Phase 13.

Saves 1280×720 RGB frames from 3 camera angles for each task.

Outputs
-------
reports/figures/phase13_scene_reshelving.png
reports/figures/phase13_scene_cleaning.png
reports/figures/phase13_scene_armpose.png

CLI args
--------
--seed    int  (default 0)
--out_dir path (default reports/figures/)
"""

import argparse
import pathlib
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))
from gpt_repro.utils.seeding import set_global_seed as set_all_seeds
from gpt_repro.envs.franka_env import FrankaKinematicEnv
from gpt_repro.envs.franka_scene import CAMERAS


# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed",    type=int, default=0)
    p.add_argument("--out_dir", type=str, default="reports/figures/")
    return p.parse_args()


def render_task(task: str, out_path: pathlib.Path, seed: int) -> None:
    """Render a 3-camera strip for one task and save to out_path."""
    env = FrankaKinematicEnv(task, render_mode="rgb_array", width=720, height=480)
    env.reset(seed=seed)

    cam_names = ["front", "side", "top"]
    frames = []
    for cam_name in cam_names:
        env.set_camera(cam_name)
        frame = env.render()
        frames.append(frame)

    env.close()

    # Compose a 1×3 strip
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"Franka Panda scene — task: {task}  (home pose)\n"
        f"Left=front  Centre=side  Right=top",
        fontsize=13, y=1.01,
    )
    for ax, frame, cam_name in zip(axes, frames, cam_names):
        ax.imshow(frame)
        ax.set_title(cam_name, fontsize=11)
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    args = parse_args()
    set_all_seeds(args.seed)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"render_scenes.py  seed={args.seed}  {ts}")

    for task in ("reshelving", "cleaning", "armpose"):
        out_path = out_dir / f"phase13_scene_{task}.png"
        render_task(task, out_path, seed=args.seed)

    print("Done.")


if __name__ == "__main__":
    main()
