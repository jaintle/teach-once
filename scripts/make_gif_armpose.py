"""make_gif_armpose.py — split-panel arm-pose GIF.

Left:  DEFAULT CONFIG (demo)   — arm traces keypoints at default positions
Right: NEW CONFIG (TP-GPT)     — arm traces keypoints shifted up & forward

960 × 510 px, 10 fps.
Uses matplotlib Agg + PIL — no OpenGL / MuJoCo required.
Transport: SVD linear alignment (Eqs. 8–11, Sec. IV-A).
Side view (XZ projection) with coloured keypoint spheres.
"""

import argparse
import pathlib
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import imageio
from PIL import Image, ImageDraw, ImageFont

OUT_PATH = pathlib.Path("reports/figures/final_armpose.gif")
FPS      = 10
PW, PH   = 480, 480
TITLE_H  = 30

# ── Dark theme ────────────────────────────────────────────────────────────────
BG       = '#0a0a0f'
PANEL_BG = '#0d0d1a'
ACCENT   = (0, 212, 255)
TRAJ_L   = '#8888ff'
TRAJ_R   = '#00d4ff'
ARM_C    = '#c0c8d8'
TABLE_C  = '#5c3d1e'

# Keypoint colours matching web demo (cyan, magenta, yellow, blue)
KP_COLORS = {
    'shoulder': ('#00e5e5', (0,   229, 229)),
    'elbow':    ('#e500e5', (229,  0,  229)),
    'wrist':    ('#ffe000', (255, 224,   0)),
    'hand':     ('#3366e5', (51,  102, 229)),
}

# ── Scene configs ─────────────────────────────────────────────────────────────
BASE_KPS = {
    'shoulder': np.array([0.35, 0.00, 0.70]),
    'elbow':    np.array([0.47, 0.00, 0.80]),
    'wrist':    np.array([0.57, 0.00, 0.75]),
    'hand':     np.array([0.62, 0.00, 0.65]),
}
# Target: all keypoints raised 0.08 m in z + shifted 0.07 m in y
TARGET_KPS = {
    'shoulder': np.array([0.35, 0.07, 0.78]),
    'elbow':    np.array([0.47, 0.07, 0.88]),
    'wrist':    np.array([0.57, 0.07, 0.83]),
    'hand':     np.array([0.62, 0.07, 0.73]),
}
KP_ORDER = ['shoulder', 'elbow', 'wrist', 'hand']

ARM_BASE = np.array([0.00, 0.00, 0.33])   # Franka base in world (XZ side view)
XLIM = (-0.15, 0.85)
ZLIM = (0.28, 1.08)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _svd_transport(S, T):
    S, T = np.asarray(S, float), np.asarray(T, float)
    sb, tb = S.mean(0), T.mean(0)
    H = (S - sb).T @ (T - tb)
    U, _, Vt = np.linalg.svd(H)
    A = Vt.T @ U.T
    if np.linalg.det(A) < 0:
        Vt[-1] *= -1; A = Vt.T @ U.T
    def fn(x):
        x = np.asarray(x, float)
        return (x - sb) @ A.T + tb if x.ndim > 1 else A @ (x - sb) + tb
    sigma = float(np.linalg.norm((S - sb) @ A.T + tb - T, axis=1).mean())
    return fn, sigma


def _make_waypoints(kps):
    """EE path: approach shoulder → shoulder → elbow → wrist → hand → retreat."""
    sh, el, wr, ha = [kps[k] for k in KP_ORDER]
    arc_mid = 0.5 * (sh + el) + np.array([0., 0., 0.04])
    return np.array([
        sh + [0., 0., 0.12],   # approach
        sh,                     # touch shoulder
        arc_mid,                # arc shoulder→elbow
        el,                     # touch elbow
        wr,                     # touch wrist
        ha,                     # touch hand
        ha + [0., 0., 0.12],   # retreat
    ], dtype=float)


def _dense_path(wps, steps_per_seg=20):
    path, segs = [], []
    for i in range(len(wps) - 1):
        for k in range(steps_per_seg):
            t = k / steps_per_seg
            path.append((1-t) * wps[i] + t * wps[i+1])
            segs.append(i)
    path.append(wps[-1])
    segs.append(len(wps) - 1)
    return np.array(path), segs


def _arm_curve(bx, bz, ex, ez, n=25):
    cx = (bx + ex) / 2
    cz = max(bz, ez) + 0.09
    t  = np.linspace(0, 1, n)
    return ((1-t)**2*bx + 2*(1-t)*t*cx + t**2*ex,
            (1-t)**2*bz + 2*(1-t)*t*cz + t**2*ez)


def _kp_phase_color(seg_id):
    """Return colour for whichever keypoint we're heading toward."""
    kp_targets = ['shoulder', 'shoulder', 'elbow', 'wrist', 'hand', 'hand']
    k = kp_targets[min(seg_id, len(kp_targets)-1)]
    return KP_COLORS[k][0]


def _load_font(size=11):
    for p in ['/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
              '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf']:
        try: return ImageFont.truetype(p, size)
        except OSError: pass
    return ImageFont.load_default()


def _add_title_bar(content_arr, left_title, right_title):
    H, W = content_arr.shape[:2]
    bar  = np.full((TITLE_H, W, 3), (10, 10, 15), dtype=np.uint8)
    combined = np.vstack([bar, content_arr])
    img  = Image.fromarray(combined)
    draw = ImageDraw.Draw(img)
    font = _load_font(12)
    half = W // 2

    def _badge(text, cx, color):
        bbox = draw.textbbox((0,0), text, font=font)
        tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
        x0, y0 = cx - tw//2, TITLE_H//2 - th//2
        draw.rectangle([x0-5, y0-5, x0+tw+5, y0+th+5], fill=(10,10,20))
        draw.text((x0, y0), text, fill=color, font=font)

    _badge(left_title,  half//2,       (180,180,255))
    _badge(right_title, half+half//2,  ACCENT)
    for y in range(TITLE_H + PH):
        draw.point((half, y), fill=ACCENT)
    return np.array(img, dtype=np.uint8)


# ── Drawing helpers ───────────────────────────────────────────────────────────
def _draw_static(ax, kps, traj_c, panel_lbl):
    ax.set_facecolor(PANEL_BG)
    ax.set_xlim(*XLIM); ax.set_ylim(*ZLIM)
    ax.axis('off')

    # Table
    ax.axhspan(0.35, 0.41, color=TABLE_C, alpha=0.8, zorder=1)

    # Arm base
    ax.add_patch(plt.Circle((ARM_BASE[0], ARM_BASE[2]), 0.022,
                              color='#444466', zorder=2))

    # "Bone" connections between keypoints
    pts_xz = [(kps[k][0], kps[k][2]) for k in KP_ORDER]
    for i in range(len(pts_xz)-1):
        ax.plot([pts_xz[i][0], pts_xz[i+1][0]],
                [pts_xz[i][1], pts_xz[i+1][1]],
                color='#555577', lw=3, alpha=0.6, zorder=3, solid_capstyle='round')

    # Keypoint spheres (static, under-glow)
    for name in KP_ORDER:
        kp  = kps[name]
        col = KP_COLORS[name][0]
        ax.add_patch(plt.Circle((kp[0], kp[2]), 0.038, color=col, alpha=0.25, zorder=3))
        ax.add_patch(plt.Circle((kp[0], kp[2]), 0.028, color=col, alpha=0.90, zorder=4))

    # Panel label
    ax.text(0.04, 0.96, panel_lbl, transform=ax.transAxes,
            color=traj_c, fontsize=10, fontweight='bold', va='top',
            bbox=dict(fc='#00000099', ec='none', boxstyle='round,pad=0.3'))


def _make_dynamic_artists(ax, traj_c):
    traj, = ax.plot([], [], color=traj_c, lw=1.5, alpha=0.6, zorder=5)
    arm,  = ax.plot([], [], color=ARM_C,  lw=2.8, alpha=0.85, zorder=5)
    ee    = ax.scatter([], [], s=110, c='white', zorder=7,
                        edgecolors='white', linewidths=0.6)
    # Flash ring for keypoint touch
    ring  = ax.scatter([], [], s=280, facecolors='none',
                        edgecolors='white', linewidths=1.5, alpha=0.0, zorder=6)
    phase = ax.text(0.50, 0.04, '', transform=ax.transAxes,
                     color='white', fontsize=8.5, ha='center', va='bottom',
                     fontweight='bold',
                     bbox=dict(fc='#00000099', ec='none', boxstyle='round,pad=0.2'))
    return dict(traj=traj, arm=arm, ee=ee, ring=ring, phase=phase)


TOUCH_THRESH = 0.07   # m — flash when EE is within this distance of a keypoint

def _is_near_kp(ee_pos, kps):
    """Return name of nearest keypoint if within TOUCH_THRESH, else None."""
    best, best_d = None, TOUCH_THRESH
    for name in KP_ORDER:
        d = np.linalg.norm(ee_pos - kps[name])
        if d < best_d:
            best, best_d = name, d
    return best


def _update_artists(arts, traj_x, traj_z, fi, path, segs, kps, prev_ring_alpha):
    ex, ez = traj_x[fi], traj_z[fi]
    seg    = segs[fi]
    pcol   = _kp_phase_color(seg)

    arts['traj'].set_data(traj_x[:fi+1], traj_z[:fi+1])

    bax, baz = _arm_curve(ARM_BASE[0], ARM_BASE[2], ex, ez)
    arts['arm'].set_data(bax, baz)
    arts['ee'].set_offsets([[ex, ez]]); arts['ee'].set_facecolors([pcol])

    # Flash ring near keypoints
    near = _is_near_kp(path[fi], kps)
    if near is not None:
        ring_col = KP_COLORS[near][0]
        alpha    = min(1.0, 1.0 - np.linalg.norm(path[fi] - kps[near]) / TOUCH_THRESH)
        arts['ring'].set_offsets([[ex, ez]])
        arts['ring'].set_edgecolors([ring_col])
        arts['ring'].set_alpha(alpha)
    else:
        arts['ring'].set_alpha(0.0)

    phase_labels = ['APPROACH', 'SHOULDER', 'ARC', 'ELBOW', 'WRIST', 'HAND', 'RETREAT']
    lbl = phase_labels[min(seg, len(phase_labels)-1)]
    arts['phase'].set_text(lbl); arts['phase'].set_color(pcol)


# ── Main generate function ────────────────────────────────────────────────────
def generate_frames(pw=PW, ph=PH, seed=0):
    np.random.seed(seed)
    warnings.filterwarnings('ignore')

    S = np.array([BASE_KPS[k] for k in KP_ORDER])
    T = np.array([TARGET_KPS[k] for k in KP_ORDER])
    transport, sigma = _svd_transport(S, T)

    src_wps = _make_waypoints(BASE_KPS)
    tgt_kps = {k: transport(TARGET_KPS[k]) for k in KP_ORDER}  # already at target
    tgt_wps = _make_waypoints(TARGET_KPS)

    src_path, src_segs = _dense_path(src_wps)
    tgt_path, tgt_segs = _dense_path(tgt_wps)

    src_x, src_z = src_path[:, 0], src_path[:, 2]
    tgt_x, tgt_z = tgt_path[:, 0], tgt_path[:, 2]

    # Check keypoints reached
    kp_dists = {}
    for k in KP_ORDER:
        dists = np.linalg.norm(tgt_path - TARGET_KPS[k], axis=1)
        kp_dists[k] = float(dists.min())

    dpi = 100
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(2*pw/dpi, ph/dpi), dpi=dpi)
    fig.patch.set_facecolor(BG)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.02, wspace=0.03)

    _draw_static(ax_l, BASE_KPS,   TRAJ_L, 'DEMO')
    _draw_static(ax_r, TARGET_KPS, TRAJ_R, 'TP-GPT')

    arts_l = _make_dynamic_artists(ax_l, TRAJ_L)
    arts_r = _make_dynamic_artists(ax_r, TRAJ_R)

    frames = []
    N = len(src_path)

    for fi in range(N):
        _update_artists(arts_l, src_x, src_z, fi, src_path, src_segs, BASE_KPS,   0)
        _update_artists(arts_r, tgt_x, tgt_z, fi, tgt_path, tgt_segs, TARGET_KPS, 0)
        fig.canvas.draw()
        arr = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[:, :, :3].copy()
        frames.append(arr)

    plt.close(fig)
    return frames, sigma, kp_dists


def main(seed=0):
    print("Generating armpose split-panel GIF...")
    frames, sigma, kp_dists = generate_frames(seed=seed)
    titled = [_add_title_bar(f, 'DEFAULT CONFIG (demo)', 'NEW CONFIG (TP-GPT)')
              for f in frames]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(OUT_PATH), titled, fps=FPS, loop=0)
    sz = OUT_PATH.stat().st_size / 1e6
    print(f"Saved {OUT_PATH}  ({sz:.1f} MB, {len(titled)} frames)")
    print(f"Transport σ = {sigma:.4f}")
    print("Min dist to each keypoint (transported):")
    for k, d in kp_dists.items():
        print(f"  {k}: {d:.4f} m  ({'REACHED' if d < 0.07 else 'MISS'})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    main(seed=args.seed)
