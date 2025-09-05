from __future__ import annotations
from typing import Iterable, List, Literal, Optional

import numpy as np
from pattern_mining import gen_schemas_for_triple  # generic 1x3 schema miner


PatternKind = Literal["h3", "v3", "schema_nxn"]

# (No per-grid printing; n×n mining and pretty printers are available in pattern_mining if needed.)


def _to_np_grid(grid: Iterable[Iterable[int]]) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("grid must be 2-D")
    return g


def _emit_overlay(center_r: int, center_c: int, y1: int, x1: int, y2: int, x2: int, overlay_id: int) -> dict:
    h = y2 - y1 + 1
    w = x2 - x1 + 1
    return {
        "overlay_id": int(overlay_id),
        "center_row": int(center_r + 1),
        "center_col": int(center_c + 1),
        "y1": int(y1 + 1),
        "x1": int(x1 + 1),
        "y2": int(y2 + 1),
        "x2": int(x2 + 1),
        "height": int(h),
        "width": int(w),
        # For compatibility with vision overlays, we include numeric fields with neutral defaults
        "contrast": float(0.0),
        "peak_lum": float(0.0),
        "area": int(h * w),
    }


def detect_pattern_overlays(
    grid: Iterable[Iterable[int]],
    *,
    kind: PatternKind = "h3",
    color: int,
    min_repeats: int = 2,
    dedup_centers: bool = True,
    window_size: Optional[int] = None,
) -> List[dict]:
    """
    Detect overlays by simple pattern templates.

    - kind="h3": emit one overlay per center matching (x, color, x) horizontally.
    - kind="v3": emit one overlay per center matching (x, color, x) vertically.
    - kind="schema_nxn": emit one overlay per pixel of the given color with an n×n box (default n=3 unless provided via window_size).
    """
    g = _to_np_grid(grid)
    H, W = g.shape
    overlays: List[dict] = []
    overlay_id = 1

    if kind == "schema_nxn":
        # One overlay per pixel of the given color. Box is n×n clipped to grid.
        n = int(window_size) if window_size is not None else 3
        if n < 1:
            raise ValueError("window_size must be >= 1")
        # Asymmetric extents when n is even: the selected center pixel is included,
        # with window spanning up/left by floor((n-1)/2) and down/right by floor(n/2).
        up_left = (n - 1) // 2
        down_right = n // 2
        for r in range(H):
            for c in range(W):
                if int(g[r, c]) != int(color):
                    continue
                y1 = max(0, r - up_left); x1 = max(0, c - up_left)
                y2 = min(H - 1, r + down_right); x2 = min(W - 1, c + down_right)
                overlays.append(_emit_overlay(r, c, y1, x1, y2, x2, overlay_id))
                overlay_id += 1
        # (Printing of mined patterns has been disabled.)
        return overlays
    elif kind == "h3":
        # Use generic schema mining to detect [X, color, X] horizontal windows
        for r in range(H):
            for c in range(W):
                if int(g[r, c]) != int(color):
                    continue
                if c - 1 < 0 or c + 1 >= W:
                    continue
                triple = (int(g[r, c - 1]), int(g[r, c]), int(g[r, c + 1]))
                schemas = gen_schemas_for_triple(triple)
                # Preserve prior behavior: require non-zero flank color
                if ("X", int(color), "X") in schemas and int(triple[0]) != 0:
                    overlays.append(_emit_overlay(r, c, r, c - 1, r, c + 1, overlay_id))
                    overlay_id += 1
        return overlays
    elif kind == "v3":
        # Use generic schema mining to detect [X, color, X] on vertical triples
        for r in range(H):
            for c in range(W):
                if int(g[r, c]) != int(color):
                    continue
                if r - 1 < 0 or r + 1 >= H:
                    continue
                triple = (int(g[r - 1, c]), int(g[r, c]), int(g[r + 1, c]))
                schemas = gen_schemas_for_triple(triple)
                if ("X", int(color), "X") in schemas and int(triple[0]) != 0:
                    overlays.append(_emit_overlay(r, c, r - 1, c, r + 1, c, overlay_id))
                    overlay_id += 1
        return overlays
    else:
        raise ValueError(f"Unknown pattern kind: {kind}")
