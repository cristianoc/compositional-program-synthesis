from __future__ import annotations
from typing import Iterable, List, Literal, Optional

import numpy as np
from pattern_mining import gen_schemas_for_triple  # generic 1x3 schema miner


PatternKind = Literal["h3", "v3", "window_nxn"]

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
    - kind="window_nxn": emit one overlay per full n×n window across the grid (no center/color requirement). Each overlay carries the window's schema.
    """
    g = _to_np_grid(grid)
    H, W = g.shape
    overlays: List[dict] = []
    overlay_id = 1

    if kind == "window_nxn":
        # Match prior schema_nxn semantics: one overlay per pixel equal to the given color,
        # with an n×n window centered at that pixel and clipped to the grid.
        n = int(window_size) if window_size is not None else 3
        if n < 1:
            raise ValueError("window_size must be >= 1")
        up_left = (n - 1) // 2
        down_right = n // 2

        VARS = (
            "X", "Y", "Z", "U", "V", "W", "Q", "R", "S", "A", "B", "C",
            "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "T"
        )
        def window_schema(win: np.ndarray) -> List[List[object]]:
            vals_to_positions: dict[int, List[tuple[int,int]]] = {}
            nr, nc = win.shape
            for i in range(nr):
                for j in range(nc):
                    v = int(win[i, j])
                    vals_to_positions.setdefault(v, []).append((i, j))
            schema: List[List[object]] = [["*" for _ in range(nc)] for _ in range(nr)]
            next_var = 0
            for v, poss in vals_to_positions.items():
                if len(poss) >= 2:
                    tok = VARS[min(next_var, len(VARS)-1)]
                    next_var += 1
                    for (i, j) in poss:
                        schema[i][j] = tok
                else:
                    (i, j) = poss[0]
                    schema[i][j] = "*"
            return schema

        for r in range(H):
            for c in range(W):
                if int(g[r, c]) != int(color):
                    continue
                y1 = max(0, r - up_left); x1 = max(0, c - up_left)
                y2 = min(H - 1, r + down_right); x2 = min(W - 1, c + down_right)
                ov = _emit_overlay(r, c, y1, x1, y2, x2, overlay_id)
                win = g[y1 : y2 + 1, x1 : x2 + 1]
                ov["window_size"] = int(win.shape[0])
                ov["window"] = win.astype(int).tolist()
                schema = window_schema(win)
                # Ensure the center position within the window is marked as the constant selected color
                ci, cj = int(r - y1), int(c - x1)
                if 0 <= ci < win.shape[0] and 0 <= cj < win.shape[1]:
                    schema[ci][cj] = int(color)
                ov["schema"] = schema
                overlays.append(ov)
                overlay_id += 1
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
