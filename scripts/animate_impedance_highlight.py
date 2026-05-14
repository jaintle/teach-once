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

