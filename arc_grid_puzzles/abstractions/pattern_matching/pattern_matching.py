# -----------------------------------------------------------------------------
# Pattern Matching - Minimal Module
# This module contains only the functions actually used by the experiment:
#  • detect_pattern_matches: Main pattern detection function
#  • _to_np_grid: Grid conversion utility
#  • _emit_pattern_match: Pattern match data structure creation
# -----------------------------------------------------------------------------

from __future__ import annotations
from typing import Iterable, List, Optional, Literal
import numpy as np

PatternKind = Literal["window_nxm"]

def _to_np_grid(grid: Iterable[Iterable[int]]) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("grid must be 2-D")
    return g

def _emit_pattern_match(center_r: int, center_c: int, y1: int, x1: int, y2: int, x2: int, pattern_id: int) -> dict:
    return {
        "y1": int(y1 + 1),
        "x1": int(x1 + 1),
        "y2": int(y2 + 1),
        "x2": int(x2 + 1),
    }

def detect_pattern_matches(
    grid: Iterable[Iterable[int]],
    *,
    kind: PatternKind = "window_nxm",
    color: int,
    min_repeats: int = 2,
    dedup_centers: bool = True,
    window_shape: Optional[tuple[int, int]] = None,
) -> List[dict]:
    """
    Detect pattern matches by simple pattern templates.

    - kind="window_nxm": emit one pattern match per center equal to the selected color with an (n×m) window clipped to the grid. Each pattern match carries the window and its per-window schema.
    """
    g = _to_np_grid(grid)
    H, W = g.shape
    pattern_matches: List[dict] = []
    pattern_id = 1

    if kind == "window_nxm":
        # One pattern match per pixel equal to the given color,
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
                    for i, j in poss:
                        schema[i][j] = tok
            return schema

        for r in range(H):
            for c in range(W):
                if int(g[r, c]) != int(color):
                    continue
                y1 = max(0, r - up)
                y2 = min(H - 1, r + down)
                x1 = max(0, c - left)
                x2 = min(W - 1, c + right)
                win = g[y1:y2+1, x1:x2+1]
                ov = _emit_pattern_match(r, c, y1, x1, y2, x2, pattern_id)
                ov["window"] = win.tolist()
                schema = window_schema(win)
                # Ensure the center position within the window is marked as the constant selected color
                ci, cj = int(r - y1), int(c - x1)
                if 0 <= ci < win.shape[0] and 0 <= cj < win.shape[1]:
                    schema[ci][cj] = int(color)
                ov["schema"] = schema
                # Precompute a concrete matched window based on the schema (variables instantiated, '*' omitted)
                # This uses the window itself as the candidate.
                mg = None  # Schema matching not implemented
                if mg is not None:
                    ov["match"] = mg
                pattern_matches.append(ov)
                pattern_id += 1
        return pattern_matches
    else:
        raise ValueError(f"Unknown pattern kind: {kind}")