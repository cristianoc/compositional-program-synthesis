from __future__ import annotations
from typing import Iterable, List, Literal, Optional

import numpy as np
from pattern_mining import gen_schemas_for_triple  # generic 1x3 schema miner


PatternKind = Literal["window_nxm", "window_nxm_all"]

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
    kind: PatternKind = "window_nxm",
    color: int,
    min_repeats: int = 2,
    dedup_centers: bool = True,
    window_shape: Optional[tuple[int, int]] = None,
) -> List[dict]:
    """
    Detect overlays by simple pattern templates.

    - kind="window_nxm": emit one overlay per center equal to the selected color with an (n×m) window clipped to the grid. Each overlay carries the window and its per-window schema.
    - kind="window_nxm_all": emit one overlay for every full (n×m) window (centered), irrespective of pixel color; color parameter is ignored.
    """
    g = _to_np_grid(grid)
    H, W = g.shape
    overlays: List[dict] = []
    overlay_id = 1

    if kind == "window_nxm":
        # One overlay per pixel equal to the given color,
        # with an n×m window centered at that pixel and clipped to the grid.
        if window_shape is None:
            hh, ww = 3, 3
        else:
            hh, ww = int(window_shape[0]), int(window_shape[1])
        if hh < 1 or ww < 1:
            raise ValueError("window_shape dims must be >= 1")
        up = (hh - 1) // 2
        down = hh // 2
        left = (ww - 1) // 2
        right = ww // 2

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
                    # For colorless schemas, keep singletons as numeric constants to anchor matches
                    schema[i][j] = int(v)
            return schema

        for r in range(H):
            for c in range(W):
                if int(g[r, c]) != int(color):
                    continue
                y1 = max(0, r - up); x1 = max(0, c - left)
                y2 = min(H - 1, r + down); x2 = min(W - 1, c + right)
                ov = _emit_overlay(r, c, y1, x1, y2, x2, overlay_id)
                win = g[y1 : y2 + 1, x1 : x2 + 1]
                ov["window_h"] = int(win.shape[0])
                ov["window_w"] = int(win.shape[1])
                ov["window_shape"] = [int(win.shape[0]), int(win.shape[1])]
                ov["window"] = win.astype(int).tolist()
                schema = window_schema(win)
                # Ensure the center position within the window is marked as the constant selected color
                ci, cj = int(r - y1), int(c - x1)
                if 0 <= ci < win.shape[0] and 0 <= cj < win.shape[1]:
                    schema[ci][cj] = int(color)
                ov["schema"] = schema
                # Precompute a concrete matched window based on the schema (variables instantiated, '*' omitted)
                # This uses the window itself as the candidate.
                try:
                    from dsl import _schema_match_window  # type: ignore
                    mg = _schema_match_window(schema, win)
                except Exception:
                    mg = None
                if mg is not None:
                    ov["match"] = mg
                overlays.append(ov)
                overlay_id += 1
        return overlays
    elif kind == "window_nxm_all":
        # All full n×m windows, centered at every valid grid position, ignoring pixel color
        if window_shape is None:
            hh, ww = 3, 3
        else:
            hh, ww = int(window_shape[0]), int(window_shape[1])
        if hh < 1 or ww < 1:
            raise ValueError("window_shape dims must be >= 1")
        up = (hh - 1) // 2
        down = hh // 2
        left = (ww - 1) // 2
        right = ww // 2

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
                    # Keep singletons as numeric constants to anchor matches in colorless mode
                    schema[i][j] = int(v)
            return schema

        # Valid centers where a full window fits inside the grid
        rmin, rmax = up, H - 1 - down
        cmin, cmax = left, W - 1 - right
        if rmin <= rmax and cmin <= cmax:
            for r in range(rmin, rmax + 1):
                for c in range(cmin, cmax + 1):
                    y1 = r - up; x1 = c - left
                    y2 = r + down; x2 = c + right
                    ov = _emit_overlay(r, c, y1, x1, y2, x2, overlay_id)
                    win = g[y1 : y2 + 1, x1 : x2 + 1]
                    ov["window_h"] = int(win.shape[0])
                    ov["window_w"] = int(win.shape[1])
                    ov["window_shape"] = [int(win.shape[0]), int(win.shape[1])]
                    ov["window"] = win.astype(int).tolist()
                    schema = window_schema(win)
                    # Do not stamp the center with a constant; no color assumption
                    ov["schema"] = schema
                    # Precompute match for this window and schema
                    try:
                        from dsl import _schema_match_window  # type: ignore
                        mg = _schema_match_window(schema, win)
                    except Exception:
                        mg = None
                    if mg is not None:
                        ov["match"] = mg
                    overlays.append(ov)
                    overlay_id += 1
        return overlays
    else:
        raise ValueError(f"Unknown pattern kind: {kind}")
