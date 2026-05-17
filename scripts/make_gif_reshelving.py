"""make_gif_reshelving.py — split-panel reshelving GIF.

Left:  DEMO      — source scene (box at default pos, shelf at default height)
Right: TP-GPT →  — transported scene (box moved right, shelf raised)

960 × 510 px (960 × 480 content + 30 px title bar), 15 fps.
Uses matplotlib Agg + PIL — no OpenGL / MuJoCo rendering required.
Transport: SVD linear alignment (Eqs. 8–11, Sec. IV-A).
"""

import argparse
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import imageio
from PIL import Image, ImageDraw, ImageFont

OUT_PATH  = pathlib.Path("reports/figures/final_reshelving.gif")
FPS       = 15
PW, PH    = 480, 480   # per-panel pixels
TITLE_H   = 30

# ── Dark theme ──────────────────────────────────────────────────────────────
BG       = '#0a0a0f'
PANEL_BG = '#0d0d1a'
ACCENT   = (0, 212, 255)
TRAJ_L   = '#8888ff'    # source traj colour
TRAJ_R   = '#00d4ff'    # transported traj colour
ARM_C    = '#c0c8d8'
BOX_C    = '#ff7214'
SHELF_C  = '#4ade80'
TABLE_C  = '#5c3d1e'

PHASE_COLORS = {
    'APPROACH': '#aaaaff', 'GRASP': '#ff9944',
    'LIFT':     '#ffdd00', 'CARRY': '#00d4ff',
    'PLACE':    '#4ade80', 'RETREAT': '#ff8888',
}

# ── Scene configs (MuJoCo Z-up, side view = XZ projection) ──────────────────
SRC_OBJ  = np.array([0.50, 0.00, 0.63])
SRC_GOAL = np.array([0.30, 0.10, 0.75])
TGT_OBJ  = np.array([0.65, 0.00, 0.63])
TGT_GOAL = np.array([0.30, 0.10, 0.92])

XLIM = (-0.20, 0.90)
ZLIM = (0.32, 1.18)
ARM_BASE_XZ = (0.00, 0.33)

PHASE_LABELS  = ['APPROACH', 'GRASP', 'LIFT', 'CARRY', 'PLACE', 'RETREAT']
SEG_STEPS     = [40, 28, 42, 60, 42, 28]   # 240 frames total
ATTACH_SEGS   = {2, 3}                       # LIFT, CARRY → box moves with EE


# ── Helpers ──────────────────────────────────────────────────────────────────

def _waypoints(obj, goal):
    return np.array([
        obj  + [0., 0.,  0.16],   # APPROACH
        obj  + [0., 0.,  0.01],   # GRASP
        obj  + [0., 0.,  0.22],   # LIFT
        goal + [0., 0.,  0.22],   # CARRY
        goal + [0., 0.,  0.04],   # PLACE
        goal + [0., 0.,  0.16],   # RETREAT
    ], dtype=float)


def _svd_transport(S, T):
    """Kabsch SVD alignment — Eqs. (8)–(11)."""
    S, T = np.asarray(S, float), np.asarray(T, float)
    sb, tb = S.mean(0), T.mean(0)
    H = (S - sb).T @ (T - tb)
    U, _, Vt = np.linalg.svd(H)
    A = Vt.T @ U.T
    if np.linalg.det(A) < 0:
        Vt[-1] *= -1
        A = Vt.T @ U.T
    def fn(x):
        x = np.asarray(x, float)
        return (x - sb) @ A.T + tb if x.ndim > 1 else A @ (x - sb) + tb
    sigma = float(np.linalg.norm((S - sb) @ A.T + tb - T, axis=1).mean())
    return fn, sigma


def _make_ST(obj_s, goal_s, obj_t, goal_t):
    offsets = np.array([[.05,.05,0],[-.05,.05,0],[0,0,.10]])
    S = np.vstack([obj_s + offsets, goal_s + offsets])
    T = np.vstack([obj_t + offsets, goal_t + offsets])
    return S, T


def _dense_path(wps):
    """Linearly interpolate waypoints → (N,3) path + segment index list.
    wps has 6 points, SEG_STEPS has 6 entries: segment i goes wps[i]→wps[i+1]
    but segment 5 (RETREAT) goes wps[5] only (last point, no i+1).
    Use 5 segments between 6 waypoints, repeat last waypoint for RETREAT.
    """
    path, segs = [], []
    # 5 interpolated segments (between consecutive waypoint pairs)
    for i in range(len(wps) - 1):
        n = SEG_STEPS[i]
        for k in range(n):
            t = k / n
            path.append((1 - t) * wps[i] + t * wps[i + 1])
            segs.append(i)
    # Final waypoint held for RETREAT duration
    n_retreat = SEG_STEPS[-1]
    for _ in range(n_retreat):
        path.append(wps[-1].copy())
        segs.append(len(SEG_STEPS) - 1)
    return np.array(path), segs


def _arm_curve(bx, bz, ex, ez, n=30):
    """Quadratic Bezier from base to EE (simulates arm arc)."""
    ctrl_x = (bx + ex) / 2
    ctrl_z = max(bz, ez) + 0.10
    t = np.linspace(0, 1, n)
    x = (1-t)**2 * bx + 2*(1-t)*t * ctrl_x + t**2 * ex
    z = (1-t)**2 * bz + 2*(1-t)*t * ctrl_z + t**2 * ez
    return x, z


def _load_font(size=11):
    for p in ['/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
              '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf']:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _add_panel_badge(draw, text, cx, y_center, font, color):
    """Draw a small badge label (DEMO / TP-GPT) in top-left of each panel half."""
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad = 5
    x0, y0 = cx - tw // 2, y_center - th // 2
    draw.rectangle([x0 - pad, y0 - pad, x0 + tw + pad, y0 + th + pad],
                   fill=(10, 10, 20))
    draw.text((x0, y0), text, fill=color, font=font)


def _add_title_bar(content_arr, left_title, right_title):
    """Prepend 30 px title bar with panel titles and vertical divider."""
    H, W = content_arr.shape[:2]
    bar = np.full((TITLE_H, W, 3), (10, 10, 15), dtype=np.uint8)
    combined = np.vstack([bar, content_arr])
    img = Image.fromarray(combined)
    draw = ImageDraw.Draw(img)
    font = _load_font(12)
    half = W // 2
    _add_panel_badge(draw, left_title,  half // 2,          TITLE_H // 2, font, (180, 180, 255))
    _add_panel_badge(draw, right_title, half + half // 2,   TITLE_H // 2, font, ACCENT)
    # Vertical divider
    for y in range(TITLE_H + PH):
        draw.point((half, y), fill=ACCENT)
    return np.array(img, dtype=np.uint8)


# ── Scene drawing helpers ─────────────────────────────────────────────────────

def _draw_static(ax, obj_xz, goal_xz, traj_color, panel_label):
    ax.set_facecolor(PANEL_BG)
    ax.set_xlim(*XLIM)
    ax.set_ylim(*ZLIM)
    ax.set_aspect('equal', adjustable='datalim')
    ax.axis('off')

    # Table surface
    ax.axhspan(0.37, 0.41, xmin=0, xmax=1, color=TABLE_C, alpha=0.85, zorder=1)

    # Shelf bracket at goal position
    gx, gz = goal_xz
    shelf_patch = mpatches.FancyBboxPatch(
        (gx - 0.09, gz - 0.05), 0.18, 0.055,
        boxstyle='round,pad=0.005', fc=SHELF_C, ec='none', alpha=0.9, zorder=3)
    ax.add_patch(shelf_patch)
    ax.fill_betweenx([gz - 0.05, gz + 0.14], gx + 0.075, gx + 0.095,
                     color=SHELF_C, alpha=0.5, zorder=3)

    # Box at source position (static, hidden when attached)
    bx, bz = obj_xz
    box_static = mpatches.Rectangle(
        (bx - 0.035, bz - 0.035), 0.07, 0.07,
        fc=BOX_C, ec='white', lw=0.8, alpha=0.95, zorder=4)
    ax.add_patch(box_static)

    # Arm base
    ax.add_patch(plt.Circle(ARM_BASE_XZ, 0.022, color='#444466', zorder=2))

    # Panel label (top-left corner)
    ax.text(0.04, 0.96, panel_label, transform=ax.transAxes,
            color=traj_color, fontsize=10, fontweight='bold', va='top',
            bbox=dict(fc='#00000099', ec='none', boxstyle='round,pad=0.3'))

    return box_static


def _make_artists(ax, traj_color):
    """Create dynamic artists. Returns dict."""
    traj_line, = ax.plot([], [], color=traj_color, lw=1.4, alpha=0.55, zorder=5)
    arm_line,  = ax.plot([], [], color=ARM_C, lw=2.8, alpha=0.85, zorder=5)
    ee_dot     = ax.scatter([], [], s=100, c='white', zorder=7,
                             edgecolors='white', linewidths=0.6)
    box_moving = mpatches.Rectangle(
        (0, 0), 0.07, 0.07, fc=BOX_C, ec='white', lw=0.8, visible=False, zorder=6)
    ax.add_patch(box_moving)
    phase_txt  = ax.text(0.50, 0.04, '', transform=ax.transAxes,
                          color='white', fontsize=8.5, ha='center', va='bottom',
                          fontweight='bold',
                          bbox=dict(fc='#00000099', ec='none', boxstyle='round,pad=0.25'))
    return dict(traj=traj_line, arm=arm_line, ee=ee_dot,
                box_m=box_moving, phase=phase_txt)


def _update_artists(arts, traj_x, traj_z, fi, path_xz, seg, obj_xz, box_static):
    ex, ez = path_xz[0][fi], path_xz[1][fi]
    phase  = PHASE_LABELS[min(seg, len(PHASE_LABELS) - 1)]
    pcol   = PHASE_COLORS.get(phase, 'white')
    attached = seg in ATTACH_SEGS

    arts['traj'].set_data(traj_x[:fi+1], traj_z[:fi+1])

    ax_bx, ax_bz = _arm_curve(*ARM_BASE_XZ, ex, ez)
    arts['arm'].set_data(ax_bx, ax_bz)

    arts['ee'].set_offsets([[ex, ez]])
    arts['ee'].set_facecolors([pcol])

    box_static.set_visible(not attached)
    if attached:
        arts['box_m'].set_xy((ex - 0.035, ez - 0.035))
        arts['box_m'].set_visible(True)
    else:
        arts['box_m'].set_visible(False)

    arts['phase'].set_text(phase)
    arts['phase'].set_color(pcol)


# ── Main generate function ────────────────────────────────────────────────────

def generate_frames(pw=PW, ph=PH, seed=0):
    np.random.seed(seed)

    src_wps = _waypoints(SRC_OBJ, SRC_GOAL)
    S, T    = _make_ST(SRC_OBJ, SRC_GOAL, TGT_OBJ, TGT_GOAL)
    transport, sigma = _svd_transport(S, T)

    tgt_wps = transport(src_wps)
    ws_lo, ws_hi = np.array([0.05, -0.5, 0.38]), np.array([0.90, 0.5, 1.10])
    tgt_wps = np.clip(tgt_wps, ws_lo, ws_hi)
    # Override grasp + place to exact target positions
    tgt_wps[0] = TGT_OBJ  + [0., 0.,  0.16]
    tgt_wps[1] = TGT_OBJ  + [0., 0.,  0.01]
    tgt_wps[3] = TGT_GOAL + [0., 0.,  0.22]
    tgt_wps[4] = TGT_GOAL + [0., 0.,  0.04]
    tgt_wps[5] = TGT_GOAL + [0., 0.,  0.16]

    src_path, src_segs = _dense_path(src_wps)
    tgt_path, tgt_segs = _dense_path(tgt_wps)

    src_xz = src_path[:, 0], src_path[:, 2]
    tgt_xz = tgt_path[:, 0], tgt_path[:, 2]

    final_dist = float(np.linalg.norm(tgt_path[-1, [0,2]] -
                                       np.array([TGT_GOAL[0], TGT_GOAL[2]])))

    dpi = 100
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(2 * pw / dpi, ph / dpi), dpi=dpi)
    fig.patch.set_facecolor(BG)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.02, wspace=0.03)

    box_l = _draw_static(ax_l, (SRC_OBJ[0], SRC_OBJ[2]),
                          (SRC_GOAL[0], SRC_GOAL[2]), TRAJ_L, 'DEMO')
    box_r = _draw_static(ax_r, (TGT_OBJ[0], TGT_OBJ[2]),
                          (TGT_GOAL[0], TGT_GOAL[2]), TRAJ_R, 'TP-GPT')

    arts_l = _make_artists(ax_l, TRAJ_L)
    arts_r = _make_artists(ax_r, TRAJ_R)

    frames = []
    N = len(src_path)

    for fi in range(N):
        _update_artists(arts_l, src_xz[0], src_xz[1], fi, src_xz, src_segs[fi],
                        (SRC_OBJ[0], SRC_OBJ[2]), box_l)
        _update_artists(arts_r, tgt_xz[0], tgt_xz[1], fi, tgt_xz, tgt_segs[fi],
                        (TGT_OBJ[0], TGT_OBJ[2]), box_r)

        fig.canvas.draw()
        arr = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[:, :, :3].copy()
        frames.append(arr)

    plt.close(fig)
    return frames, sigma, final_dist


def main(seed=0):
    print("Generating reshelving split-panel GIF...")
    frames, sigma, final_dist = generate_frames(seed=seed)

    titled = [_add_title_bar(f, 'SOURCE DEMO', 'TP-GPT RESULT') for f in frames]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(OUT_PATH), titled, fps=FPS, loop=0)
    sz = OUT_PATH.stat().st_size / 1e6
    print(f"Saved {OUT_PATH}  ({sz:.1f} MB, {len(titled)} frames)")
    print(f"Transport σ = {sigma:.4f}")
    print(f"Final EE distance from shelf: {final_dist:.4f} m")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    main(seed=args.seed)
