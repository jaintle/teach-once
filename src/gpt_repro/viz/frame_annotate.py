"""Frame annotation utilities — Phase 14.

All functions gracefully degrade:
  cv2 available   → use cv2.putText / cv2.rectangle.
  Pillow available → use PIL.ImageDraw (default on this project).
  Neither         → no-op (frame returned unchanged).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    import cv2 as _cv2  # type: ignore
except ImportError:
    _cv2 = None

try:
    from PIL import Image as _PILImage, ImageDraw as _PILDraw, ImageFont as _PILFont
    _PIL_OK = True
except ImportError:
    _PIL_OK = False

try:
    import matplotlib.cm as _mpl_cm
    import matplotlib.colors as _mpl_colors
    _MPL_OK = True
except ImportError:
    _MPL_OK = False


# ---------------------------------------------------------------------------
# add_text_overlay
# ---------------------------------------------------------------------------

def add_text_overlay(
    frame: np.ndarray,
    text: str,
    pos: Tuple[int, int] = (10, 30),
    color: Tuple[int, int, int] = (255, 255, 255),
    font_scale: float = 0.7,
) -> np.ndarray:
    """Draw text on a copy of ``frame``.

    Tries cv2 first, then Pillow, then returns the frame unchanged.

    Parameters
    ----------
    frame      : (H, W, 3) uint8 numpy array.
    text       : string to draw.
    pos        : (x, y) pixel position of text baseline.
    color      : (R, G, B) uint8 tuple.
    font_scale : approximate font size scaling factor.

    Returns
    -------
    (H, W, 3) uint8 annotated copy.
    """
    out = frame.copy()

    if _cv2 is not None:
        # cv2 uses BGR color order
        bgr = (int(color[2]), int(color[1]), int(color[0]))
        thickness = max(1, int(font_scale * 1.5))
        _cv2.putText(
            out, text, pos,
            _cv2.FONT_HERSHEY_SIMPLEX,
            font_scale, bgr, thickness, _cv2.LINE_AA,
        )
        return out

    if _PIL_OK:
        img = _PILImage.fromarray(out)
        draw = _PILDraw.Draw(img)
        # Estimate font size from scale (Pillow default font is ~10px)
        font_px = max(10, int(font_scale * 16))
        try:
            font = _PILFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_px)
        except (IOError, OSError):
            try:
                font = _PILFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_px)
            except (IOError, OSError):
                font = _PILFont.load_default()
        draw.text(pos, text, fill=tuple(color), font=font)
        return np.array(img, dtype=np.uint8)

    # No annotation backend — return unchanged
    return out


# ---------------------------------------------------------------------------
# add_title_bar
# ---------------------------------------------------------------------------

def add_title_bar(
    frame: np.ndarray,
    title: str,
    bar_height: int = 40,
    bg_color: Tuple[int, int, int] = (30, 30, 30),
    text_color: Tuple[int, int, int] = (240, 240, 240),
) -> np.ndarray:
    """Prepend a solid-colour title bar above ``frame``.

    Parameters
    ----------
    frame      : (H, W, 3) uint8 numpy array.
    title      : text to centre in the bar.
    bar_height : pixels.
    bg_color   : (R, G, B) background colour of the bar.
    text_color : (R, G, B) text colour.

    Returns
    -------
    (H + bar_height, W, 3) uint8 array.
    """
    H, W = frame.shape[:2]
    bar = np.full((bar_height, W, 3), bg_color, dtype=np.uint8)

    # Draw text centred in bar
    x_center = W // 2
    y_center = bar_height // 2

    if _cv2 is not None:
        bgr_text = (int(text_color[2]), int(text_color[1]), int(text_color[0]))
        font = _cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.55
        thickness = 1
        (tw, th), _ = _cv2.getTextSize(title, font, scale, thickness)
        tx = max(0, x_center - tw // 2)
        ty = y_center + th // 2
        _cv2.putText(bar, title, (tx, ty), font, scale, bgr_text, thickness, _cv2.LINE_AA)
    elif _PIL_OK:
        img_bar = _PILImage.fromarray(bar)
        draw = _PILDraw.Draw(img_bar)
        font_px = max(10, bar_height - 16)
        try:
            font = _PILFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_px)
        except (IOError, OSError):
            try:
                font = _PILFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_px)
            except (IOError, OSError):
                font = _PILFont.load_default()
        # Center text
        bbox = draw.textbbox((0, 0), title, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        tx = max(0, x_center - tw // 2)
        ty = max(0, y_center - th // 2)
        draw.text((tx, ty), title, fill=tuple(text_color), font=font)
        bar = np.array(img_bar, dtype=np.uint8)

    return np.vstack([bar, frame])


# ---------------------------------------------------------------------------
# colormap_scalar
# ---------------------------------------------------------------------------

def colormap_scalar(
    value: float,
    vmin: float = 0.0,
    vmax: float = 1.0,
    cmap: str = "coolwarm",
) -> Tuple[int, int, int]:
    """Map a scalar to an (R, G, B) uint8 tuple using a matplotlib colormap.

    Parameters
    ----------
    value : float scalar.
    vmin, vmax : normalization range.
    cmap : matplotlib colormap name.

    Returns
    -------
    (R, G, B) as uint8 ints.
    """
    if _MPL_OK:
        norm  = _mpl_colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
        cm    = _mpl_cm.get_cmap(cmap)
        r, g, b, _ = cm(norm(value))
        return (int(r * 255), int(g * 255), int(b * 255))
    # Fallback: linear blue→red
    t = max(0.0, min(1.0, (value - vmin) / max(vmax - vmin, 1e-9)))
    return (int(t * 255), 0, int((1 - t) * 255))
