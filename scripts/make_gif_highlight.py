"""make_gif_highlight.py — 2×3 grid highlight reel.

Layout:
  Header bar (24 px): task column labels
  ┌──────────────────┬──────────────────┬──────────────────┐
  │ DEMO             │ DEMO             │ DEMO             │  ← 340 px
  │ Reshelving       │ Cleaning         │ Arm-pose         │
  ├──────────────────┼──────────────────┼──────────────────┤  ← 2 px divider
  │ TP-GPT           │ TP-GPT           │ TP-GPT           │  ← 340 px
  │ Reshelving       │ Cleaning         │ Arm-pose         │
  └──────────────────┴──────────────────┴──────────────────┘
  Caption bar (20 px)

Each cell: 320 × 340 px  (480 × 510 source half-panel at ⅔ scale — exact,
no distortion: 480×2/3=320, 510×2/3=340).

Total: 960 × (24 + 340 + 2 + 340 + 20) = 960 × 726 px, 10 fps, loop forever.
"""

import pathlib
import numpy as np
import imageio
from PIL import Image, ImageDraw, ImageFont

SRC = {
    'reshelving': pathlib.Path('reports/figures/final_reshelving.gif'),
    'cleaning':   pathlib.Path('reports/figures/final_cleaning.gif'),
    'armpose':    pathlib.Path('reports/figures/final_armpose.gif'),
}
OUT_PATH = pathlib.Path('reports/figures/final_highlight.gif')
FPS      = 10

CELL_W   = 320   # 480 * 2/3
CELL_H   = 340   # 510 * 2/3
TOTAL_W  = CELL_W * 3        # 960
HEADER_H = 24
DIV_H    = 2
CAP_H    = 20
TOTAL_H  = HEADER_H + CELL_H + DIV_H + CELL_H + CAP_H   # 726

ACCENT   = (0, 212, 255)
DIM      = (60, 60, 80)
BG       = (10, 10, 15)
DEMO_COL = (136, 136, 255)   # lavender
TPGPT_COL= (0, 212, 255)     # cyan

TASKS    = ['Reshelving', 'Cleaning', 'Arm-pose']


def _load_gif(path):
    frames = imageio.mimread(str(path), memtest=False)
    out = []
    for f in frames:
        arr = np.asarray(f, dtype=np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr]*3, axis=-1)
        elif arr.shape[2] == 4:
            arr = arr[:, :, :3]
        out.append(arr)
    return out


def _split_and_scale(frame):
    """Split 960×510 frame into left/right 480×510 halves, scale each to 320×340."""
    h, w = frame.shape[:2]
    mid = w // 2
    left  = Image.fromarray(frame[:, :mid]).resize((CELL_W, CELL_H), Image.LANCZOS)
    right = Image.fromarray(frame[:, mid:]).resize((CELL_W, CELL_H), Image.LANCZOS)
    return np.array(left, dtype=np.uint8), np.array(right, dtype=np.uint8)


def _load_font(size=11):
    for p in ['/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
              '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf']:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _make_header():
    """24 px bar: task names centred over each column, cyan dividers."""
    bar = np.full((HEADER_H, TOTAL_W, 3), BG, dtype=np.uint8)
    img = Image.fromarray(bar); draw = ImageDraw.Draw(img)
    font = _load_font(11)
    for i, name in enumerate(TASKS):
        cx = CELL_W * i + CELL_W // 2
        bbox = draw.textbbox((0,0), name, font=font)
        tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
        draw.text((cx - tw//2, HEADER_H//2 - th//2), name,
                  fill=ACCENT, font=font)
    for x in [CELL_W, CELL_W*2]:
        draw.line([(x,0),(x,HEADER_H-1)], fill=DIM, width=1)
    return np.array(img, dtype=np.uint8)


def _make_divider():
    """2 px horizontal divider between DEMO and TP-GPT rows."""
    row = np.full((DIV_H, TOTAL_W, 3), DIM, dtype=np.uint8)
    return row


def _make_caption():
    """20 px caption bar at the bottom."""
    bar = np.full((CAP_H, TOTAL_W, 3), BG, dtype=np.uint8)
    img = Image.fromarray(bar); draw = ImageDraw.Draw(img)
    font = _load_font(9)
    txt = 'Three tasks  ·  Single demonstration each  ·  TP-GPT (Franzese et al. 2024)'
    bbox = draw.textbbox((0,0), txt, font=font)
    tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
    draw.text((TOTAL_W//2 - tw//2, CAP_H//2 - th//2), txt,
              fill=(100, 100, 120), font=font)
    return np.array(img, dtype=np.uint8)


def _col_dividers(content_row):
    """Stamp 1-px cyan column dividers into a 340×960×3 row array."""
    content_row[:, CELL_W-1, :]   = DIM
    content_row[:, CELL_W*2-1, :] = DIM
    return content_row


def main():
    print("Loading source GIFs...")
    gifs = {k: _load_gif(v) for k, v in SRC.items()}
    for k, v in gifs.items():
        print(f"  {k}: {len(v)} frames, shape {v[0].shape}")

    N       = max(len(v) for v in gifs.values())
    header  = _make_header()
    divider = _make_divider()
    caption = _make_caption()

    frames = []
    for fi in range(N):
        demo_cells, tpgpt_cells = [], []
        for key in ('reshelving', 'cleaning', 'armpose'):
            src = gifs[key]
            frame = src[fi % len(src)]
            left, right = _split_and_scale(frame)
            demo_cells.append(left)
            tpgpt_cells.append(right)

        demo_row  = _col_dividers(np.concatenate(demo_cells,  axis=1))
        tpgpt_row = _col_dividers(np.concatenate(tpgpt_cells, axis=1))

        combined = np.vstack([header, demo_row, divider, tpgpt_row, caption])
        assert combined.shape == (TOTAL_H, TOTAL_W, 3), combined.shape
        frames.append(combined)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(OUT_PATH), frames, fps=FPS, loop=0)
    sz = OUT_PATH.stat().st_size / 1e6
    print(f"Saved {OUT_PATH}  ({sz:.1f} MB, {len(frames)} frames, {TOTAL_W}×{TOTAL_H})")
    print(f"Cell scale: 480×510 → {CELL_W}×{CELL_H} (⅔, aspect {CELL_W/CELL_H:.3f} vs {480/510:.3f})")


if __name__ == '__main__':
    main()
