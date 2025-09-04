from __future__ import annotations
from typing import Dict, Iterable, List, Tuple, Optional

import numpy as np

# Visual palette and luminance helpers
PALETTE = {
    0: (0, 0, 0),
    1: (0, 0, 255),
    2: (255, 0, 0),
    3: (0, 255, 0),
    4: (255, 255, 0),
    5: (128, 128, 128),
    6: (255, 192, 203),
    7: (255, 165, 0),
    8: (0, 128, 128),
    9: (139, 69, 19),
}


def _default_palette() -> Dict[int, Tuple[int, int, int]]:
    return PALETTE.copy()


def to_luminance(rgb_uint8: np.ndarray) -> np.ndarray:
    img = rgb_uint8.astype(np.float32) / 255.0
    return 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]


def grid_to_luminance(g: np.ndarray) -> np.ndarray:
    h, w = g.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    pal = _default_palette()
    for k, (r, gc, b) in pal.items():
        rgb[g == k] = (r, gc, b)
    return to_luminance(rgb)






def detect_bright_overlays_absolute(
    grid: Iterable[Iterable[int]],
    *,
    p_hi: float = 99.7,
    nms_radius: int = 4,
    context_pad: int = 2,
) -> List[dict]:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("grid must be 2-D")
    h, w = g.shape
    lum = grid_to_luminance(g)
    # local maxima
    padded = np.pad(lum, nms_radius, mode="edge")
    peaks = np.ones_like(lum, dtype=bool)
    for dy in range(-nms_radius, nms_radius + 1):
        for dx in range(-nms_radius, nms_radius + 1):
            if dy == 0 and dx == 0:
                continue
            neigh = padded[nms_radius + dy : nms_radius + dy + h, nms_radius + dx : nms_radius + dx + w]
            peaks &= lum >= neigh
    thr = np.percentile(lum, p_hi)
    ys, xs = np.where(peaks & (lum >= thr))
    overlays: List[dict] = []
    for rank, (py, px) in enumerate(zip(ys, xs), start=1):
        y1, y2 = max(0, py - 1), min(h - 1, py + 1)
        x1, x2 = max(0, px - 1), min(w - 1, px + 1)
        wy1, wy2 = max(0, y1 - context_pad), min(h - 1, y2 + context_pad)
        wx1, wx2 = max(0, x1 - context_pad), min(w - 1, x2 + context_pad)
        window = lum[wy1 : wy2 + 1, wx1 : wx2 + 1]
        boxmask = np.zeros_like(window, dtype=bool)
        boxmask[(y1 - wy1) : (y2 - wy1 + 1), (x1 - wx1) : (x2 - wx1 + 1)] = True
        surround_vals = window[~boxmask]
        surround_mean = float(surround_vals.mean()) if surround_vals.size else float(lum.mean())
        peak_lum = float(lum[py, px])
        contrast = max(0.0, peak_lum - surround_mean)
        overlays.append(
            {
                "overlay_id": rank,
                "center_row": int(py + 1),
                "center_col": int(px + 1),
                "y1": int(y1 + 1),
                "x1": int(x1 + 1),
                "y2": int(y2 + 1),
                "x2": int(x2 + 1),
                "height": int(y2 - y1 + 1),
                "width": int(x2 - x1 + 1),
                "contrast": float(contrast),
                "peak_lum": float(peak_lum),
                "area": int((y2 - y1 + 1) * (x2 - x1 + 1)),
            }
        )
    return overlays
