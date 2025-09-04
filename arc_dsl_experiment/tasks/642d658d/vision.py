from __future__ import annotations
from typing import Dict, Iterable, List, Tuple, Optional

import numpy as np
from functools import lru_cache

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


# Overlay detection

def detect_bright_overlays(
    grid: Iterable[Iterable[int]],
    palette: Optional[Dict[int, Tuple[int, int, int]]] = None,
    *,
    nms_radius: int = 4,
    local_radii: Tuple[int, ...] = (1, 2, 3),
    peak_k: float = 3.4,
    local_k: float = 3.8,
    p_hi: float = 99.7,
    drop_threshold: float = 0.06,
    scale_gamma: float = 1.0,
    max_radius: float = 1.4,
    context_pad: int = 2,
) -> List[dict]:
    def default_palette():
        return _default_palette()

    def robust_zscores(a: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        med = np.median(a)
        mad = np.median(np.abs(a - med)) + eps
        return (a - med) / (1.4826 * mad)

    def local_mean_std(img: np.ndarray, r: int) -> Tuple[np.ndarray, np.ndarray]:
        h, w = img.shape
        pad = r
        arr = np.pad(img, pad, mode="edge")
        arr2 = arr * arr
        ii = arr.cumsum(0).cumsum(1)
        ii2 = arr2.cumsum(0).cumsum(1)

        def rect_sum(ii, y, x, win):
            y2, x2 = y + win - 1, x + win - 1
            a = ii[y - 1, x - 1] if (y > 0 and x > 0) else 0.0
            b = ii[y - 1, x2] if y > 0 else 0.0
            c = ii[y2, x - 1] if x > 0 else 0.0
            d = ii[y2, x2]
            return d + a - b - c

        win = 2 * r + 1
        n = float(win * win)
        means = np.empty((h, w), dtype=np.float32)
        stds = np.empty((h, w), dtype=np.float32)
        for y in range(h):
            for x in range(w):
                py, px = y, x
                s = rect_sum(ii, py, px, win)
                s2 = rect_sum(ii2, py, px, win)
                m = s / n
                v = max(0.0, (s2 / n) - m * m)
                means[y, x] = m
                stds[y, x] = np.sqrt(v)
        return means, np.maximum(stds, 1e-6)

    def local_maxima_mask(arr: np.ndarray, radius: int) -> np.ndarray:
        h, w = arr.shape
        padded = np.pad(arr, radius, mode="edge")
        out = np.ones_like(arr, dtype=bool)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue
                neigh = padded[radius + dy : radius + dy + h, radius + dx : radius + dx + w]
                out &= arr >= neigh
        return out

    def normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
        lo, hi = float(v.min()), float(v.max())
        if hi - lo < eps:
            return np.zeros_like(v, dtype=np.float32)
        return (v - lo) / (hi - lo)

    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("grid must be 2-D")
    h, w = g.shape
    pal = palette if palette is not None else default_palette()
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for k, (r, gc, b) in pal.items():
        rgb[g == k] = (r, gc, b)
    lum = to_luminance(rgb)

    peak_mask_all = local_maxima_mask(lum, radius=nms_radius)
    py, px = np.where(peak_mask_all)
    if len(py) == 0:
        return []

    rz = robust_zscores(lum)
    local_z_max = np.zeros_like(lum, dtype=np.float32)
    for r in local_radii:
        mu, sd = local_mean_std(lum, r)
        zloc = (lum - mu) / sd
        local_z_max = np.maximum(local_z_max, zloc.astype(np.float32))
    lum_hi = np.percentile(lum, p_hi)

    kept = []
    for y, x in zip(py, px):
        if (rz[y, x] >= peak_k and local_z_max[y, x] >= local_k) or (lum[y, x] >= lum_hi):
            kept.append((int(y), int(x)))
    if not kept:
        return []

    comps: List[List[Tuple[int, int]]] = [[] for _ in range(len(kept))]
    for y in range(h):
        for x in range(w):
            best, bd = None, 1e18
            for idx, (pyk, pxk) in enumerate(kept):
                d = (y - pyk) * (y - pyk) + (x - pxk) * (x - pxk)
                if d < bd:
                    bd, best = d, idx
            peak_val = lum[kept[best][0], kept[best][1]]
            if lum[y, x] >= peak_val * (1.0 - drop_threshold):
                comps[best].append((y, x))

    overlays = []
    contrasts = []
    areas = []
    temp = []
    for idx, comp in enumerate(comps):
        if not comp:
            continue
        pyc, pxc = kept[idx]
        dists = [np.hypot(y - pyc, x - pxc) for (y, x) in comp]
        r = max(1.0, 1.0 * float(np.percentile(dists, 75)))
        r = min(r, max_radius)
        half = int(round(r))
        y1, y2 = max(0, pyc - half), min(h - 1, pyc + half)
        x1, x2 = max(0, pxc - half), min(w - 1, pxc + half)
        wy1, wy2 = max(0, y1 - context_pad), min(h - 1, y2 + context_pad)
        wx1, wx2 = max(0, x1 - context_pad), min(w - 1, x2 + context_pad)
        window = lum[wy1 : wy2 + 1, wx1 : wx2 + 1]
        boxmask = np.zeros_like(window, dtype=bool)
        boxmask[(y1 - wy1) : (y2 - wy1 + 1), (x1 - wx1) : (x2 - wx1 + 1)] = True
        surround_vals = window[~boxmask]
        surround_mean = float(surround_vals.mean()) if surround_vals.size else float(lum.mean())
        peak_lum = float(lum[pyc, pxc])
        contrast = max(0.0, peak_lum - surround_mean)
        temp.append(
            {
                "center_rc0": (pyc, pxc),
                "y1x1y2x2_0": (y1, x1, y2, x2),
                "area": len(comp),
                "peak_lum": peak_lum,
                "contrast": contrast,
            }
        )
        contrasts.append(contrast)
        areas.append(len(comp))

    if not temp:
        return []

    def normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
        lo, hi = float(v.min()), float(v.max())
        if hi - lo < eps:
            return np.zeros_like(v, dtype=np.float32)
        return (v - lo) / (hi - lo)

    contrasts = np.array(contrasts, dtype=np.float32)
    areas = np.array(areas, dtype=np.float32)
    scores = 0.8 * normalize(contrasts) + 0.2 * normalize(np.log1p(areas))
    order = np.argsort(-scores)
    for rank, k in enumerate(order, start=1):
        (pyc, pxc) = temp[k]["center_rc0"]
        (y1, x1, y2, x2) = temp[k]["y1x1y2x2_0"]
        overlays.append(
            {
                "overlay_id": rank,
                "center_row": int(pyc + 1),
                "center_col": int(pxc + 1),
                "y1": int(y1 + 1),
                "x1": int(x1 + 1),
                "y2": int(y2 + 1),
                "x2": int(x2 + 1),
                "height": int(y2 - y1 + 1),
                "width": int(x2 - x1 + 1),
                "contrast": float(temp[k]["contrast"]),
                "peak_lum": float(temp[k]["peak_lum"]),
                "area": int(temp[k]["area"]),
            }
        )
    return overlays


# Fast-first helpers

def _grid_key_bytes(g: np.ndarray):
    a = np.asarray(g, dtype=np.uint8)
    return (a.shape[0], a.shape[1], a.tobytes())


@lru_cache(maxsize=256)
def _fast_centers_from_bytes(h: int, w: int, buf: bytes, p_hi: float = 99.7):
    a = np.frombuffer(buf, dtype=np.uint8).reshape(h, w)
    # luminance
    pal = _default_palette()
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for k, (rr, gg, bb) in pal.items():
        rgb[a == k] = (rr, gg, bb)
    img = rgb.astype(np.float32) / 255.0
    lum = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
    # 8-neighbor local maxima
    pad = np.pad(lum, 1, mode="edge")
    is_max = np.ones((h, w), dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            neigh = pad[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
            is_max &= lum >= neigh
    thr = np.percentile(lum, p_hi)
    sel = np.where(is_max & (lum >= thr))
    centers = [(int(r + 1), int(c + 1)) for r, c in zip(sel[0], sel[1])]
    return centers


def fast_uniform_cross_color_if_agree(g: np.ndarray) -> Optional[int]:
    h, w = np.asarray(g, dtype=np.uint8).shape
    buf = np.asarray(g, dtype=np.uint8).tobytes()
    centers = _fast_centers_from_bytes(h, w, buf, 99.7)
    if not centers:
        return None
    colors = []
    for (r, c) in centers:
        vals = []
        rr, cc = r - 1, c - 1
        if rr - 1 >= 0:
            vals.append(int(g[rr - 1, cc]))
        if rr + 1 < g.shape[0]:
            vals.append(int(g[rr + 1, cc]))
        if cc - 1 >= 0:
            vals.append(int(g[rr, cc - 1]))
        if cc + 1 < g.shape[1]:
            vals.append(int(g[rr, cc + 1]))
        if not (len(vals) == 4 and len(set(vals)) == 1 and vals[0] != 0):
            return None
        colors.append(vals[0])
    if not colors:
        return None
    if len(set(colors)) == 1:
        return int(colors[0])
    return None
