from __future__ import annotations
from typing import Iterable, List, Literal

import numpy as np
from pattern_mining import gen_schemas_for_triple  # generic 1x3 schema miner


PatternKind = Literal["h3", "v3", "cross3"]


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
) -> List[dict]:
    """
    Detect overlays by simple pattern templates.

    - kind="h3": emit one overlay per center matching (x, color, x) horizontally.
    - kind="v3": emit one overlay per center matching (x, color, x) vertically.
    - kind="cross3": emit one overlay per pixel of the given color with a 3x3 box.
    """
    g = _to_np_grid(grid)
    H, W = g.shape
    overlays: List[dict] = []
    overlay_id = 1

    if kind == "cross3":
        # One overlay per pixel of the given color. Box is 3x3 clipped to grid.
        for r in range(H):
            for c in range(W):
                if int(g[r, c]) != int(color):
                    continue
                y1 = max(0, r - 1); x1 = max(0, c - 1)
                y2 = min(H - 1, r + 1); x2 = min(W - 1, c + 1)
                overlays.append(_emit_overlay(r, c, y1, x1, y2, x2, overlay_id))
                overlay_id += 1
        # For this kind, we return immediately (no grouping by repeats)
        return overlays
    elif kind == "h3":
        # Use generic schema mining to detect [X, color, X] horizontal windows
        desired_schema = ("X", int(color), "X")
        for r in range(H):
            for c in range(W):
                if int(g[r, c]) != int(color):
                    continue
                if c - 1 < 0 or c + 1 >= W:
                    continue
                triple = (int(g[r, c - 1]), int(g[r, c]), int(g[r, c + 1]))
                schemas = gen_schemas_for_triple(triple)
                # Preserve prior behavior: require non-zero flank color
                if desired_schema in schemas and int(triple[0]) != 0:
                    overlays.append(_emit_overlay(r, c, r, c - 1, r, c + 1, overlay_id))
                    overlay_id += 1
        return overlays
    elif kind == "v3":
        # One overlay per pixel of the given color that forms vertical (x,color,x) with its neighbors
        for r in range(H):
            for c in range(W):
                if int(g[r, c]) != int(color):
                    continue
                if r-1 < 0 or r+1 >= H:
                    continue
                a, b = int(g[r-1, c]), int(g[r+1, c])
                if a == b and a != 0:
                    overlays.append(_emit_overlay(r, c, r-1, c, r+1, c, overlay_id))
                    overlay_id += 1
        return overlays
    else:
        raise ValueError(f"Unknown pattern kind: {kind}")


