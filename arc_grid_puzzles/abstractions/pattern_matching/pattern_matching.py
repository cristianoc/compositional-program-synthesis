# -----------------------------------------------------------------------------
# Pattern Matching - Minimal Module
# This module contains only the functions actually used by the experiment:
#  • detect_pattern_matches: Main pattern detection function
#  • _to_np_grid: Grid conversion utility
#  • _emit_pattern_match: Pattern match data structure creation
# -----------------------------------------------------------------------------

from __future__ import annotations
from typing import Iterable, List
import numpy as np

def _to_np_grid(grid: Iterable[Iterable[int]]) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("grid must be 2-D")
    return g

def _emit_pattern_match(y1: int, x1: int, y2: int, x2: int) -> dict:
    """
    Create a pattern match data structure.
    
    Args:
        y1, x1: Top-left corner of the pattern window (0-indexed)
        y2, x2: Bottom-right corner of the pattern window (0-indexed) 
        
    Returns:
        Dictionary with 1-indexed coordinates for external use
        
    Note:
        The center_r, center_c, and pattern_id parameters were removed because:
        1. They are not used in the returned dictionary
        2. The center can be calculated from the window bounds if needed
        3. This simplifies the function signature and reduces unused parameters
    """
    return {
        "y1": int(y1 + 1),  # Convert to 1-indexed for external use
        "x1": int(x1 + 1),  # Convert to 1-indexed for external use
        "y2": int(y2 + 1),  # Convert to 1-indexed for external use
        "x2": int(x2 + 1),  # Convert to 1-indexed for external use
    }

def detect_pattern_matches(
    grid: Iterable[Iterable[int]],
    *,
    color: int,
    window_shape: tuple[int, int],
) -> List[dict]:
    """
    Detect pattern matches by simple pattern templates.
    
    Args:
        grid: 2D grid of integers
        color: Color to find patterns around (always 4 in practice)
        window_shape: Shape of the window to extract around each color occurrence
        
    Returns:
        List of pattern match dictionaries with window bounds, content, and schema
    """
    g = _to_np_grid(grid)
    H, W = g.shape
    pattern_matches: List[dict] = []

    # Extract window dimensions
    hh, ww = int(window_shape[0]), int(window_shape[1])
    if hh < 1 or ww < 1:
        raise ValueError("window_shape dims must be >= 1")
    
    # Calculate window offsets
    up = (hh - 1) // 2
    down = hh // 2
    left = (ww - 1) // 2
    right = ww // 2

    # Variable names for schema generation
    VARS = (
        "X", "Y", "Z", "U", "V", "W", "Q", "R", "S", "A", "B", "C",
        "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "T"
    )
    
    def window_schema(win: np.ndarray) -> List[List[object]]:
        """Generate a schema for a window by replacing repeated values with variables."""
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

    # Find all occurrences of the target color and extract windows
    for r in range(H):
        for c in range(W):
            if int(g[r, c]) != int(color):
                continue
            y1 = max(0, r - up)
            y2 = min(H - 1, r + down)
            x1 = max(0, c - left)
            x2 = min(W - 1, c + right)
            win = g[y1:y2+1, x1:x2+1]
            ov = _emit_pattern_match(y1, x1, y2, x2)
            ov["window"] = win.tolist()
            schema = window_schema(win)
            # Ensure the center position within the window is marked as the constant selected color
            ci, cj = int(r - y1), int(c - x1)
            if 0 <= ci < win.shape[0] and 0 <= cj < win.shape[1]:
                schema[ci][cj] = int(color)
            ov["schema"] = schema
            pattern_matches.append(ov)
    
    return pattern_matches