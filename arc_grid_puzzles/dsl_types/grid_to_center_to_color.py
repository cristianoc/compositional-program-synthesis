# -----------------------------------------------------------------------------
# Grid to Center to Color Operations
# This module implements core G operations for compositional synthesis:
#  • Center selection operations (Grid -> Center)
#  • Output operations (Center -> Color)
#  • Local structure color rules
# -----------------------------------------------------------------------------

from __future__ import annotations
from typing import List, Tuple, Callable, Dict
import numpy as np
from dsl_types.states import Operation, Grid, Center, Color, OpFailure


def _cross4_vals_any(g: np.ndarray, r: int, c: int):
    H,W = g.shape; vals=[]
    if r-1>=0: vals.append(int(g[r-1,c]))
    if r+1<H:  vals.append(int(g[r+1,c]))
    if c-1>=0: vals.append(int(g[r,c-1]))
    if c+1<W:  vals.append(int(g[r,c+1]))
    return vals


def _mode_int(values):
    from collections import Counter
    if not values: return 0
    cnt = Counter(values); top = max(cnt.values())
    cands = [v for v,c in cnt.items() if c==top]
    return int(min(cands))


def sel_color_uniform_cross_everywhere_mode(x_hat: np.ndarray) -> int:
    g = np.asarray(x_hat, dtype=int); H,W = g.shape; picks=[]
    for r in range(H):
        for c in range(W):
            vals=_cross4_vals_any(g,r,c)
            if vals and len(set(vals))==1 and vals[0]!=0:
                picks.append(vals[0])
    if not picks:
        vals, cnt = np.unique(g[g!=0], return_counts=True)
        return int(vals[np.argmax(cnt)]) if len(vals) else 0
    return _mode_int(picks)


def sel_color_argmax_uniform_cross_color_count(x_hat: np.ndarray) -> int:
    g = np.asarray(x_hat, dtype=int); H,W = g.shape
    counts = {c:0 for c in range(1,10)}
    for r in range(H):
        for c in range(W):
            vals=_cross4_vals_any(g,r,c)
            if vals and len(set(vals))==1 and vals[0]!=0:
                counts[vals[0]] += 1
    best_c = 0; best_n = -1
    for c in range(1,10):
        n = counts[c]
        if n>best_n or (n==best_n and c<best_c):
            best_c, best_n = c, n
    return best_c if best_n>0 else 0


def rule_h3_flank_mode(x_hat: np.ndarray) -> int:
    """Horizontal shape-aware rule: mode of flank colors in [x, c, x] triples.

    Scans all horizontal triples; when left==right!=0, record that flank color.
    Returns the mode (tie -> smallest). If none found, returns global mode of non-zero colors.
    """
    g = np.asarray(x_hat, dtype=int)
    H, W = g.shape
    picks: list[int] = []
    for r in range(H):
        for c in range(1, W-1):
            a, b = int(g[r, c-1]), int(g[r, c+1])
            if a == b and a != 0:
                picks.append(a)
    if not picks:
        vals, cnt = np.unique(g[g!=0], return_counts=True)
        return int(vals[np.argmax(cnt)]) if len(vals) else 0
    return _mode_int(picks)


def rule_v3_flank_mode(x_hat: np.ndarray) -> int:
    """Vertical shape-aware rule: mode of flank colors in vertical [x, c, x] triples.

    Scans all vertical triples; when up==down!=0, record that flank color.
    Returns the mode (tie -> smallest). If none found, returns global mode of non-zero colors.
    """
    g = np.asarray(x_hat, dtype=int)
    H, W = g.shape
    picks: list[int] = []
    for r in range(1, H-1):
        for c in range(W):
            a, b = int(g[r-1, c]), int(g[r+1, c])
            if a == b and a != 0:
                picks.append(a)
    if not picks:
        vals, cnt = np.unique(g[g!=0], return_counts=True)
        return int(vals[np.argmax(cnt)]) if len(vals) else 0
    return _mode_int(picks)


def rule_best_center_cross_mode(x_hat: np.ndarray) -> int:
    """Pick a center color c (1..9) maximizing the number of uniform 4-cross windows.

    For each grid cell equal to c, if its 4-neighborhood (up,down,left,right) is
    uniform and non-zero, record that color. Select the c with the largest number
    of such hits (tie -> smaller c), then return the mode of recorded colors for
    that c (tie -> smaller color). If no hits exist for any c, return 0.
    """
    g = np.asarray(x_hat, dtype=int)
    H, W = g.shape
    best_c = 0
    best_hits = -1
    best_colors: list[int] = []
    for c0 in range(1, 10):
        hits: list[int] = []
        for r in range(H):
            for c in range(W):
                if int(g[r, c]) != c0:
                    continue
                vals = _cross4_vals_any(g, r, c)
                if len(vals)==4 and len(set(vals))==1 and vals[0]!=0:
                    hits.append(int(vals[0]))
        if hits:
            if len(hits) > best_hits or (len(hits) == best_hits and c0 < best_c):
                best_hits = len(hits)
                best_c = c0
                best_colors = hits
    if best_hits <= 0:
        return 0
    return _mode_int(best_colors)


def rule_best_center_flank_mode(x_hat: np.ndarray) -> int:
    """Scan each candidate center color c in 1..9 and collect flank evidence.

    For every grid cell equal to c, if left/right are equal and non-zero, record that
    flank color; likewise for up/down. Pick the c with the most total flank hits
    (tie -> smaller c). Return the mode of recorded flank colors for that c (tie -> min).
    If no evidence at all, return 0.
    """
    g = np.asarray(x_hat, dtype=int)
    H, W = g.shape
    best_c = 0
    best_hits = -1
    best_colors: list[int] = []
    for c0 in range(1, 10):
        hits: list[int] = []
        for r in range(H):
            for c in range(W):
                if int(g[r, c]) != c0:
                    continue
                # horizontal flanks
                if c - 1 >= 0 and c + 1 < W:
                    a, b = int(g[r, c - 1]), int(g[r, c + 1])
                    if a == b and a != 0:
                        hits.append(a)
                # vertical flanks
                if r - 1 >= 0 and r + 1 < H:
                    a, b = int(g[r - 1, c]), int(g[r + 1, c])
                    if a == b and a != 0:
                        hits.append(a)
        if hits:
            if len(hits) > best_hits or (len(hits) == best_hits and c0 < best_c):
                best_hits = len(hits)
                best_c = c0
                best_colors = hits
    if best_hits <= 0:
        return 0
    return _mode_int(best_colors)


# Center-choosers and center-conditioned outputs (for composition)
def _collect_full_33_windows_for_center(g: np.ndarray, c0: int) -> List[np.ndarray]:
    H, W = g.shape
    wins: List[np.ndarray] = []
    for r in range(1, H-1):
        for c in range(1, W-1):
            if int(g[r, c]) != int(c0):
                continue
            wins.append(g[r-1:r+2, c-1:c+2])
    return wins


_REL_CROSS_33 = [(-1,0),(1,0),(0,-1),(0,1)]


def _cross_equal_implied_across_windows(wins: List[np.ndarray]) -> bool:
    if not wins:
        return False
    # For each pair of cross positions, verify equality across all windows
    for i in range(len(_REL_CROSS_33)):
        for j in range(i+1, len(_REL_CROSS_33)):
            dri, dci = _REL_CROSS_33[i]
            drj, dcj = _REL_CROSS_33[j]
            base = int(wins[0][1+dri, 1+dci]) - int(wins[0][1+drj, 1+dcj])
            for wv in wins:
                if int(wv[1+dri, 1+dci]) != int(wv[1+drj, 1+dcj]):
                    return False
    return True


def choose_center_cross_implied_33(x_hat: np.ndarray) -> int:
    g = np.asarray(x_hat, dtype=int)
    best_c = 0
    best_n = -1
    for c0 in range(1,10):
        wins = _collect_full_33_windows_for_center(g, c0)
        if not wins:
            continue
        if _cross_equal_implied_across_windows(wins):
            n = len(wins)
            if n>best_n or (n==best_n and c0<best_c):
                best_n = n
                best_c = c0
    return best_c if best_n>0 else 0


def out_mode_cross_for_center_33(x_hat: np.ndarray, c0: int) -> int:
    g = np.asarray(x_hat, dtype=int)
    wins = _collect_full_33_windows_for_center(g, c0)
    cols: List[int] = []
    for wv in wins:
        vals = [int(wv[1+dr,1+dc]) for (dr,dc) in _REL_CROSS_33]
        if len(set(vals))==1 and vals[0]!=0:
            cols.append(int(vals[0]))
    return _mode_int(cols) if cols else 0


def out_mode_flank_for_center(x_hat: np.ndarray, c0: int) -> int:
    g = np.asarray(x_hat, dtype=int)
    H, W = g.shape
    cols: List[int] = []
    for r in range(H):
        for c in range(W):
            if int(g[r,c]) != c0:
                continue
            if c-1>=0 and c+1<W:
                a,b = int(g[r,c-1]), int(g[r,c+1])
                if a==b and a!=0:
                    cols.append(a)
            if r-1>=0 and r+1<H:
                a,b = int(g[r-1,c]), int(g[r+1,c])
                if a==b and a!=0:
                    cols.append(a)
    return _mode_int(cols) if cols else 0


# Local-structure color rules
COLOR_RULES_BASE: List[Tuple[str, Callable[[np.ndarray], int]]] = [
    ("uniform_cross_everywhere_mode", sel_color_uniform_cross_everywhere_mode),
    ("argmax_uniform_cross_color_count", sel_color_argmax_uniform_cross_color_count),
    ("best_center_flank_mode", rule_best_center_flank_mode),
    ("best_center_cross_mode", rule_best_center_cross_mode),
    ("h3_flank_mode", rule_h3_flank_mode),
    ("v3_flank_mode", rule_v3_flank_mode),
]

