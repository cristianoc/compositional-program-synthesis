# -----------------------------------------------------------------------------
# Overlay Abstraction Experiment Utilities
# This module implements:
#  • Overlay extractor: detect_bright_overlays  (README_clean.md §2.2, “Methods: Overlay extractor”)
#  • Abstraction: PatternOverlayExtractor       (pattern overlays + stored overlays)
#  • Pattern predicate & predictor pipeline     (UniformPatternPredicate; §2.3; §5 “Analysis”)
#  • G core color rules incl. local-structure   (README_clean.md §2.4 “Extensions to G”)
#  • Program enumeration & pretty-print helpers (README_clean.md §4 “Results”)
# For numeric defaults of the detector, see README_clean.md: “Detector defaults (for reproducibility)”.
# -----------------------------------------------------------------------------


from __future__ import annotations
from typing import Dict, Iterable, List, Tuple, Optional, Callable, Type, TypeVar, Generic, Union
import numpy as np
from overlay_patterns import detect_pattern_overlays
# ===================== Typed, compositional DSL =====================
# Minimal typed-DSL scaffolding to make composition explicit and extensible.
# States capture the current representation; Operations convert between states.

class State:  # marker base
    pass


class GridState(State):
    def __init__(self, grid: np.ndarray):
        self.grid = np.asarray(grid, dtype=int)


class OverlayContext(State):
    def __init__(self, grid: np.ndarray, overlays: List[dict], stats: Dict[str, float], kind: Optional[str] = None, color: Optional[int] = None):
        self.grid = np.asarray(grid, dtype=int)
        self.overlays = overlays
        self.stats = stats
        self.kind = kind
        self.color = int(color) if color is not None else None


class ColorState(State):
    def __init__(self, color: int):
        self.color = int(color)


class CenterState(State):
    def __init__(self, grid: np.ndarray, center_color: int):
        self.grid = np.asarray(grid, dtype=int)
        self.center_color = int(center_color)


class MatchesState(State):
    def __init__(self, grid: np.ndarray, matches: List[dict]):
        self.grid = np.asarray(grid, dtype=int)
        self.matches = matches


class OpFailure(Exception):
    pass


InS = TypeVar("InS", bound=State)
OutS = TypeVar("OutS", bound=State)


class Operation(Generic[InS, OutS]):
    input_type: Type[State] = State
    output_type: Type[State] = State

    def accepts(self, state: State) -> bool:
        return isinstance(state, self.input_type)

    def apply(self, state: InS) -> OutS:
        raise NotImplementedError


class Pipeline:
    def __init__(self, ops: List[Operation]):
        self.ops = ops

    def run(self, state: State) -> State:
        cur = state
        for op in self.ops:
            if not op.accepts(cur):
                raise TypeError(f"Operation {op.__class__.__name__} does not accept state {type(cur).__name__}")
            cur = op.apply(cur)  # type: ignore[arg-type]
        return cur
# ===================== Overlay Extractor (pattern-based) =====================
def detect_overlays(
    grid: Iterable[Iterable[int]],
    *,
    kind: str = "window_nxm",
    color: int,
    min_repeats: int = 2,
    window_shape: Optional[tuple[int, int]] = None,
) -> List[dict]:
    # For window_nxm, propagate a configurable window shape (defaults to WINDOW_SHAPE_DEFAULT).
    eff_kind = kind
    eff_shape: Optional[tuple[int, int]] = window_shape if window_shape is not None else (WINDOW_SHAPE_DEFAULT if kind == "window_nxm" else None)
    return detect_pattern_overlays(
        grid,
        kind=eff_kind,  # type: ignore[arg-type]
        color=int(color),
        min_repeats=min_repeats,
        window_shape=eff_shape,
    )
# Pattern kinds considered during search/enumeration
PATTERN_KINDS: List[str] = ["window_nxm"]
# Default window shape for window_nxm (n×m with n,m≥1). Used across detection and pretty-printing.
WINDOW_SHAPE_DEFAULT: tuple[int, int] = (3, 3)
# Optimization: pre-check that a pattern appears in all examples (train+test)
# Optimization: pre-check that a pattern appears in all examples (train+test)
def pattern_present_in_all_examples(task: Dict, kind: str, color: int, *, window_shape: Optional[tuple[int,int]] = None) -> bool:
    # task is ARC-like dict with "train" and "test" splits
    for split in ("train", "test"):
        for ex in task.get(split, []):
            g = ex["input"]
            ovs = detect_overlays(g, kind=kind, color=int(color), window_shape=window_shape)
            if len(ovs) == 0:
                return False
    return True

# ---------------------- Combined schema for window_nxm ----------------------
def _gather_full_windows_for_grid(g: np.ndarray, color: int, *, window_shape: tuple[int,int]) -> List[np.ndarray]:
    """Collect all full n×m windows centered on pixels equal to color.

    Only returns windows fully inside the grid (no clipping)."""
    wins: List[np.ndarray] = []
    H, W = g.shape
    hh, ww = int(window_shape[0]), int(window_shape[1])
    if hh < 1 or ww < 1:
        return wins
    up = (hh - 1) // 2
    down = hh // 2
    left = (ww - 1) // 2
    right = ww // 2
    rmin, rmax = up, H - 1 - down
    cmin, cmax = left, W - 1 - right
    if rmin > rmax or cmin > cmax:
        return wins
    for r in range(rmin, rmax + 1):
        for c in range(cmin, cmax + 1):
            if int(g[r, c]) != int(color):
                continue
            wins.append(g[r-up:r+down+1, c-left:c+right+1].copy())
    return wins

def _gather_full_windows_for_task(task: Dict, color: int, *, window_shape: tuple[int,int]) -> List[np.ndarray]:
    wins: List[np.ndarray] = []
    import numpy as np
    for split in ("train","test"):
        for ex in task.get(split, []):
            g = np.asarray(ex["input"], dtype=int)
            wins.extend(_gather_full_windows_for_grid(g, color, window_shape=window_shape))
    return wins

def combined_window_nxm_schema(task: Dict, color: int, *, window_shape: tuple[int,int]) -> List[List[Union[int, str]]]:
    """Return the consensus schema matrix across all full windows for the given color and n×m."""
    wins = _gather_full_windows_for_task(task, color, window_shape=window_shape)
    hh, ww = int(window_shape[0]), int(window_shape[1])
    if not wins:
        return [["*"] * ww for _ in range(hh)]
    # Determine constants per position across windows
    pos_vals: List[set[int]] = []
    for i in range(hh):
        for j in range(ww):
            vals = {int(win[i, j]) for win in wins}
            pos_vals.append(vals)
    is_const = [len(s) == 1 for s in pos_vals]
    const_val: List[Optional[int]] = [next(iter(s)) if len(s) == 1 else None for s in pos_vals]

    # Equality relation among non-constant positions: equal across all windows
    npos = hh * ww
    adj = [[False] * npos for _ in range(npos)]
    for a in range(npos):
        adj[a][a] = True
    for a in range(npos):
        if is_const[a]:
            continue
        ai, aj = divmod(a, ww)
        for b in range(a + 1, npos):
            if is_const[b]:
                continue
            bi, bj = divmod(b, ww)
            equal_all = True
            for win in wins:
                if int(win[ai, aj]) != int(win[bi, bj]):
                    equal_all = False
                    break
            if equal_all:
                adj[a][b] = adj[b][a] = True

    # Connected components of equal positions (size >= 2 → variable group)
    visited = [False] * npos
    comps: List[List[int]] = []
    for v in range(npos):
        if visited[v] or is_const[v]:
            continue
        stack = [v]
        visited[v] = True
        comp = [v]
        while stack:
            u = stack.pop()
            for w in range(npos):
                if not visited[w] and adj[u][w]:
                    visited[w] = True
                    stack.append(w)
                    comp.append(w)
        if len(comp) >= 2:
            comps.append(sorted(comp))

    # Assemble schema grid
    schema: List[List[Union[int, str]]] = [["*" for _ in range(ww)] for _ in range(hh)]
    for p in range(npos):
        if is_const[p]:
            i, j = divmod(p, ww)
            schema[i][j] = int(const_val[p]) if const_val[p] is not None else "*"
    var_tokens = ("X", "Y", "Z", "U", "V", "W")
    next_var = 0
    for comp in comps:
        tok = var_tokens[min(next_var, len(var_tokens) - 1)]
        next_var += 1
        for p in comp:
            i, j = divmod(p, ww)
            schema[i][j] = tok
    return schema

 
# ===================== Abstraction & Predicates =====================
class PatternOverlayExtractor:
    def __init__(self):
        self.overlays: List[dict] = []
        self.last_stats: Dict[str, float] = {}


# Typed-DSL operations corresponding to the above components

# ===================== A Ops (Overlay Abstraction) =====================
class OpBrightOverlayIdentity(Operation[GridState, OverlayContext]):
    input_type = GridState
    output_type = OverlayContext

    def __init__(self, absx: Optional[PatternOverlayExtractor] = None, kind: str = "window_nxm", color: Optional[int] = None, window_shape: Optional[tuple[int,int]] = None):
        self.absx = absx or PatternOverlayExtractor()
        self.kind = kind
        if color is None:
            raise ValueError("OpBrightOverlayIdentity requires explicit color")
        self.color = int(color)
        self.window_shape = tuple(window_shape) if window_shape is not None else None
        # Base label; parameters may be appended in pretty-printing
        self.label = f"overlay_{self.kind}"

    def apply(self, state: GridState) -> OverlayContext:
        g = state.grid
        # Use configured pattern kind and default shape where applicable
        k = self.kind
        # Select window shape override if provided, else default for kind
        ws: Optional[tuple[int,int]] = None
        if k == "window_nxm":
            ws = self.window_shape if self.window_shape is not None else WINDOW_SHAPE_DEFAULT
        # Use the pattern kind and color for overlays
        ovs = detect_overlays(g.tolist(), kind=k, color=self.color, window_shape=ws)
        H, W = g.shape
        count = len(ovs)
        max_contrast = max([ov["contrast"] for ov in ovs], default=0.0)
        total_area = sum([ov["area"] for ov in ovs]) if ovs else 0
        total_area_frac = float(total_area) / float(H * W) if H * W > 0 else 0.0
        stats = dict(count=count, max_contrast=max_contrast, total_area=total_area, total_area_frac=total_area_frac)
        self.absx.overlays = ovs
        self.absx.last_stats = stats
        return OverlayContext(g, ovs, stats, k, self.color)


class OpUniformPatternPredicate(Operation[OverlayContext, ColorState]):
    input_type = OverlayContext
    output_type = ColorState
    label = "uniform_pattern_predicate"

    def apply(self, state: OverlayContext) -> ColorState:
        # Predicts a single output color from pattern overlays.
        # Semantics by kind:
        #  - window_nxm: center-neighborhood evidence with special cases for degenerate shapes
        # If no kind-specific evidence is found, returns 0 (no guess).
        g = state.grid
        from collections import Counter
        flank_colors: List[int] = []
        kind = getattr(state, "kind", None)
        # Color is optional (colorless overlays supported); predicate does not require it
        sel_color = getattr(state, "color", None)
        # Collect evidence depending on kind
        for ov in state.overlays:
            r, c = ov["center_row"] - 1, ov["center_col"] - 1
            if kind == "window_nxm":
                # Previous centerless rule: derive evidence from the window center neighborhood only.
                # For center region (rows CR, cols CC):
                #  - odd×odd: 4-way cross around true center
                #  - even×even: central 2×2 block
                #  - odd×even: central 1×2 block
                #  - even×odd: central 2×1 block.
                win = ov.get("window")
                if win is None:
                    y1,x1,y2,x2 = ov["y1"]-1, ov["x1"]-1, ov["y2"]-1, ov["x2"]-1
                    win_arr = g[y1:y2+1, x1:x2+1].copy()
                    win = win_arr.astype(int).tolist()
                h_eff = int(len(win)) if hasattr(win, "__len__") else 0
                if h_eff <= 0:
                    continue
                w_eff = int(len(win[0])) if h_eff>0 else 0
                if any(len(row) != w_eff for row in win):
                    continue
                # Temporary special-cases to mirror 1×3 (horizontal) and 3×1 (vertical) semantics for degenerate shapes
                if h_eff == 1 and w_eff == 3:
                    a, b = int(win[0][0]), int(win[0][2])
                    if a == b and a != 0:
                        flank_colors.append(a)
                    continue
                if h_eff == 3 and w_eff == 1:
                    a, b = int(win[0][0]), int(win[2][0])
                    if a == b and a != 0:
                        flank_colors.append(a)
                    continue
                if h_eff % 2 == 1 and w_eff % 2 == 1:
                    ci = h_eff // 2; cj = w_eff // 2
                    if ci-1 < 0 or ci+1 >= h_eff or cj-1 < 0 or cj+1 >= w_eff:
                        continue
                    vals = [int(win[ci-1][cj]), int(win[ci+1][cj]), int(win[ci][cj-1]), int(win[ci][cj+1])]
                    if len(set(vals)) == 1 and vals[0] != 0:
                        flank_colors.append(int(vals[0]))
                elif h_eff % 2 == 0 and w_eff % 2 == 0:
                    i0 = h_eff // 2 - 1; i1 = h_eff // 2
                    j0 = w_eff // 2 - 1; j1 = w_eff // 2
                    vals = [int(win[i0][j0]), int(win[i0][j1]), int(win[i1][j0]), int(win[i1][j1])]
                    if len(set(vals)) == 1 and vals[0] != 0:
                        flank_colors.append(int(vals[0]))
                elif h_eff % 2 == 1 and w_eff % 2 == 0:
                    ci = h_eff // 2
                    j0 = w_eff // 2 - 1; j1 = w_eff // 2
                    vals = [int(win[ci][j0]), int(win[ci][j1])]
                    if len(set(vals)) == 1 and vals[0] != 0:
                        flank_colors.append(int(vals[0]))
                elif h_eff % 2 == 0 and w_eff % 2 == 1:
                    cj = w_eff // 2
                    i0 = h_eff // 2 - 1; i1 = h_eff // 2
                    vals = [int(win[i0][cj]), int(win[i1][cj])]
                    if len(set(vals)) == 1 and vals[0] != 0:
                        flank_colors.append(int(vals[0]))
            elif kind == "window_nxm_all":
                # Same evidence logic as window_nxm, using the window content only
                win = ov.get("window")
                if win is None:
                    y1,x1,y2,x2 = ov["y1"]-1, ov["x1"]-1, ov["y2"]-1, ov["x2"]-1
                    win_arr = g[y1:y2+1, x1:x2+1].copy()
                    win = win_arr.astype(int).tolist()
                h_eff = int(len(win)) if hasattr(win, "__len__") else 0
                if h_eff <= 0:
                    continue
                w_eff = int(len(win[0])) if h_eff>0 else 0
                if any(len(row) != w_eff for row in win):
                    continue
                if h_eff == 1 and w_eff == 3:
                    a, b = int(win[0][0]), int(win[0][2])
                    if a == b and a != 0:
                        flank_colors.append(a)
                    continue
                if h_eff == 3 and w_eff == 1:
                    a, b = int(win[0][0]), int(win[2][0])
                    if a == b and a != 0:
                        flank_colors.append(a)
                    continue
                if h_eff % 2 == 1 and w_eff % 2 == 1:
                    ci = h_eff // 2; cj = w_eff // 2
                    if ci-1 < 0 or ci+1 >= h_eff or cj-1 < 0 or cj+1 >= w_eff:
                        continue
                    vals = [int(win[ci-1][cj]), int(win[ci+1][cj]), int(win[ci][cj-1]), int(win[ci][cj+1])]
                    if len(set(vals)) == 1 and vals[0] != 0:
                        flank_colors.append(int(vals[0]))
                elif h_eff % 2 == 0 and w_eff % 2 == 0:
                    i0 = h_eff // 2 - 1; i1 = h_eff // 2
                    j0 = w_eff // 2 - 1; j1 = w_eff // 2
                    vals = [int(win[i0][j0]), int(win[i0][j1]), int(win[i1][j0]), int(win[i1][j1])]
                    if len(set(vals)) == 1 and vals[0] != 0:
                        flank_colors.append(int(vals[0]))
                elif h_eff % 2 == 1 and w_eff % 2 == 0:
                    ci = h_eff // 2
                    j0 = w_eff // 2 - 1; j1 = w_eff // 2
                    vals = [int(win[ci][j0]), int(win[ci][j1])]
                    if len(set(vals)) == 1 and vals[0] != 0:
                        flank_colors.append(int(vals[0]))
                elif h_eff % 2 == 0 and w_eff % 2 == 1:
                    cj = w_eff // 2
                    i0 = h_eff // 2 - 1; i1 = h_eff // 2
                    vals = [int(win[i0][cj]), int(win[i1][cj])]
                    if len(set(vals)) == 1 and vals[0] != 0:
                        flank_colors.append(int(vals[0]))
        if flank_colors:
            cnt = Counter(flank_colors); top = max(cnt.values())
            cands = [k for k,v in cnt.items() if v==top]
            return ColorState(int(min(cands)))
        # No evidence from required pattern; return 0 (no fallback to cross-only)
        return ColorState(0)


def _schema_match_window(schema: List[List[Union[int, str]]], win: np.ndarray) -> Optional[List[List[Optional[int]]]]:
    nr, nc = win.shape
    if nr != len(schema) or (nr > 0 and nc != len(schema[0])):
        return None
    var_map: Dict[str, int] = {}
    out: List[List[Optional[int]]] = [[None for _ in range(nc)] for _ in range(nr)]
    for i in range(nr):
        for j in range(nc):
            s = schema[i][j]
            v = int(win[i, j])
            if isinstance(s, int):
                if v != int(s):
                    return None
                out[i][j] = v
            elif isinstance(s, str):
                if s == "*":
                    out[i][j] = None
                else:
                    if s not in var_map:
                        var_map[s] = v
                    elif var_map[s] != v:
                        return None
                    out[i][j] = v
            else:
                return None
    return out


class OpSchemaMatchAcrossGrid(Operation[OverlayContext, MatchesState]):
    input_type = OverlayContext
    output_type = MatchesState

    label = "schema_match_across_grid"

    def __init__(self, limit_schemas: Optional[int] = 4):
        self.limit_schemas = limit_schemas

    def apply(self, state: OverlayContext) -> MatchesState:
        g = np.asarray(state.grid, dtype=int)
        H, W = g.shape
        matches: List[dict] = []
        # Deduplicate schemas by shape+string
        seen = set()
        picked: List[dict] = []
        for ov in state.overlays:
            schema = ov.get("schema")
            if not schema:
                continue
            key = (tuple(len(row) for row in schema), tuple(tuple(row) for row in schema))
            if key in seen:
                continue
            seen.add(key)
            picked.append(ov)
            if self.limit_schemas is not None and len(picked) >= int(self.limit_schemas):
                break
        for ov in picked:
            schema = ov.get("schema")
            if not schema:
                continue
            nr = len(schema); nc = len(schema[0]) if nr>0 else 0
            up = (nr - 1) // 2
            down = nr // 2
            left = (nc - 1) // 2
            right = nc // 2
            rmin, rmax = up, H - 1 - down
            cmin, cmax = left, W - 1 - right
            if rmin > rmax or cmin > cmax:
                continue
            for r in range(rmin, rmax + 1):
                for c in range(cmin, cmax + 1):
                    y1 = r - up; x1 = c - left
                    y2 = r + down; x2 = c + right
                    win = g[y1:y2+1, x1:x2+1]
                    mg = _schema_match_window(schema, win)
                    if mg is not None:
                        matches.append({
                            "y1": int(y1+1), "x1": int(x1+1), "y2": int(y2+1), "x2": int(x2+1),
                            "match": mg,
                            "schema": schema,
                        })
        return MatchesState(g, matches)


class OpUniformColorFromMatches(Operation[MatchesState, ColorState]):
    input_type = MatchesState
    output_type = ColorState

    label = "uniform_color_from_matches"

    def apply(self, state: MatchesState) -> ColorState:
        # Aggregate all non-None matched cells across all matches and pick the mode (excluding 0 by default)
        vals: List[int] = []
        for m in state.matches:
            grid = m.get("match", [])
            for row in grid:
                for v in row:
                    if v is None:
                        continue
                    vv = int(v)
                    if vv != 0:
                        vals.append(vv)
        if not vals:
            return ColorState(0)
        return ColorState(_mode_int(vals))


class OpFilterCrosslikeSchemas(Operation[OverlayContext, OverlayContext]):
    input_type = OverlayContext
    output_type = OverlayContext

    label = "filter_crosslike_schemas"

    def _is_crosslike(self, schema: List[List[Union[int, str]]]) -> bool:
        nr = len(schema)
        nc = len(schema[0]) if nr>0 else 0
        if nr == 0 or nc == 0:
            return False
        # Helper to get comparable token at (i,j)
        def tok(i,j):
            v = schema[i][j]
            return v
        # 1x3: ends equal and not '*'
        if nr == 1 and nc == 3:
            a, b = tok(0,0), tok(0,2)
            return (a == b) and (a != "*")
        # 3x1: ends equal and not '*'
        if nr == 3 and nc == 1:
            a, b = tok(0,0), tok(2,0)
            return (a == b) and (a != "*")
        # odd×odd: 4-way cross equal and not '*'
        if nr % 2 == 1 and nc % 2 == 1:
            ci, cj = nr//2, nc//2
            positions = [(ci-1,cj),(ci+1,cj),(ci,cj-1),(ci,cj+1)]
            tks = [tok(i,j) for (i,j) in positions]
            return len(set(tks)) == 1 and tks[0] != "*"
        # odd×even: central 1×2 equal and not '*'
        if nr % 2 == 1 and nc % 2 == 0:
            ci = nr//2; j0 = nc//2 - 1; j1 = nc//2
            a, b = tok(ci, j0), tok(ci, j1)
            return (a == b) and (a != "*")
        # even×odd: central 2×1 equal and not '*'
        if nr % 2 == 0 and nc % 2 == 1:
            cj = nc//2; i0 = nr//2 - 1; i1 = nr//2
            a, b = tok(i0, cj), tok(i1, cj)
            return (a == b) and (a != "*")
        # even×even: central 2×2 all equal and not '*'
        if nr % 2 == 0 and nc % 2 == 0:
            i0 = nr//2 - 1; i1 = nr//2
            j0 = nc//2 - 1; j1 = nc//2
            tks = [tok(i0,j0), tok(i0,j1), tok(i1,j0), tok(i1,j1)]
            return len(set(tks)) == 1 and tks[0] != "*"
        return False

    def apply(self, state: OverlayContext) -> OverlayContext:
        ovs = []
        for ov in state.overlays:
            schema = ov.get("schema")
            if not schema:
                continue
            if self._is_crosslike(schema):
                ovs.append(ov)
        # Keep stats consistent
        g = state.grid
        H,W = g.shape
        count = len(ovs)
        max_contrast = max([ov.get("contrast", 0.0) for ov in ovs], default=0.0)
        total_area = sum([ov.get("area", 0) for ov in ovs]) if ovs else 0
        total_area_frac = float(total_area) / float(H*W) if H*W>0 else 0.0
        stats = dict(count=count, max_contrast=max_contrast, total_area=total_area, total_area_frac=total_area_frac)
        return OverlayContext(state.grid, ovs, stats, state.kind, getattr(state, 'color', None))


class OpFilterSchemasWithConstant(Operation[OverlayContext, OverlayContext]):
    input_type = OverlayContext
    output_type = OverlayContext

    label = "filter_schemas_with_constant"

    def __init__(self, exclude_zero: bool = True):
        self.exclude_zero = bool(exclude_zero)

    def apply(self, state: OverlayContext) -> OverlayContext:
        ovs = []
        for ov in state.overlays:
            schema = ov.get("schema")
            if not schema:
                continue
            has_const = False
            for row in schema:
                for s in row:
                    if isinstance(s, int):
                        if self.exclude_zero and int(s) == 0:
                            continue
                        has_const = True
                        break
                if has_const:
                    break
            if has_const:
                ovs.append(ov)
        g = state.grid
        H,W = g.shape
        count = len(ovs)
        max_contrast = max([ov.get("contrast", 0.0) for ov in ovs], default=0.0)
        total_area = sum([ov.get("area", 0) for ov in ovs]) if ovs else 0
        total_area_frac = float(total_area) / float(H*W) if H*W>0 else 0.0
        stats = dict(count=count, max_contrast=max_contrast, total_area=total_area, total_area_frac=total_area_frac)
        return OverlayContext(state.grid, ovs, stats, state.kind, getattr(state, 'color', None))


class OpFilterSchemasWithConstantValue(Operation[OverlayContext, OverlayContext]):
    input_type = OverlayContext
    output_type = OverlayContext

    def __init__(self, const_value: int, exclude_zero: bool = True):
        self.const_value = int(const_value)
        self.exclude_zero = bool(exclude_zero)
        self.label = f"filter_schema_const({self.const_value})"

    def apply(self, state: OverlayContext) -> OverlayContext:
        target = int(self.const_value)
        if self.exclude_zero and target == 0:
            return OverlayContext(state.grid, [], dict(count=0, max_contrast=0.0, total_area=0, total_area_frac=0.0), state.kind, getattr(state, 'color', None))
        ovs = []
        for ov in state.overlays:
            schema = ov.get("schema")
            if not schema:
                continue
            has_target = False
            for row in schema:
                for s in row:
                    if isinstance(s, int) and int(s) == target:
                        has_target = True
                        break
                if has_target:
                    break
            if has_target:
                ovs.append(ov)
        g = state.grid
        H,W = g.shape
        count = len(ovs)
        max_contrast = max([ov.get("contrast", 0.0) for ov in ovs], default=0.0)
        total_area = sum([ov.get("area", 0) for ov in ovs]) if ovs else 0
        total_area_frac = float(total_area) / float(H*W) if H*W>0 else 0.0
        stats = dict(count=count, max_contrast=max_contrast, total_area=total_area, total_area_frac=total_area_frac)
        return OverlayContext(state.grid, ovs, stats, state.kind, getattr(state, 'color', None))


class OpFilterSchemasCenterConstant(Operation[OverlayContext, OverlayContext]):
    input_type = OverlayContext
    output_type = OverlayContext

    def __init__(self, const_value: int):
        self.const_value = int(const_value)
        self.label = f"filter_center_const({self.const_value})"

    def _center_positions(self, nr: int, nc: int):
        # Return list of center coordinates depending on parity
        if nr % 2 == 1 and nc % 2 == 1:
            return [(nr//2, nc//2)]
        if nr % 2 == 1 and nc % 2 == 0:
            return [(nr//2, nc//2 - 1), (nr//2, nc//2)]
        if nr % 2 == 0 and nc % 2 == 1:
            return [(nr//2 - 1, nc//2), (nr//2, nc//2)]
        # even×even → 2×2 center
        return [(nr//2 - 1, nc//2 - 1), (nr//2 - 1, nc//2), (nr//2, nc//2 - 1), (nr//2, nc//2)]

    def apply(self, state: OverlayContext) -> OverlayContext:
        v = int(self.const_value)
        ovs: List[dict] = []
        for ov in state.overlays:
            sc = ov.get("schema")
            if not sc:
                continue
            nr = len(sc); nc = len(sc[0]) if nr>0 else 0
            ok = True
            for (i,j) in self._center_positions(nr, nc):
                s = sc[i][j]
                if not (isinstance(s, int) and int(s) == v):
                    ok = False
                    break
            if ok:
                ovs.append(ov)
        g = state.grid
        H,W = g.shape
        count = len(ovs)
        max_contrast = max([ov.get("contrast", 0.0) for ov in ovs], default=0.0)
        total_area = sum([ov.get("area", 0) for ov in ovs]) if ovs else 0
        total_area_frac = float(total_area) / float(H*W) if H*W>0 else 0.0
        stats = dict(count=count, max_contrast=max_contrast, total_area=total_area, total_area_frac=total_area_frac)
        return OverlayContext(state.grid, ovs, stats, state.kind, getattr(state, 'color', None))


class OpUniformColorPerSchemaThenMode(Operation[MatchesState, ColorState]):
    input_type = MatchesState
    output_type = ColorState

    label = "uniform_color_per_schema_then_mode"

    def __init__(self, cross_only: bool = True):
        self.cross_only = bool(cross_only)

    def _cross_positions(self, nr: int, nc: int):
        if nr == 1 and nc == 3:
            return [(0,0),(0,2)]
        if nr == 3 and nc == 1:
            return [(0,0),(2,0)]
        ci, cj = nr//2, nc//2
        if nr%2==1 and nc%2==1:
            return [(ci-1,cj),(ci+1,cj),(ci,cj-1),(ci,cj+1)]
        if nr%2==1 and nc%2==0:
            return [(ci, cj-1),(ci, cj)]
        if nr%2==0 and nc%2==1:
            return [(ci-1, cj),(ci, cj)]
        return [(ci-1, cj-1),(ci-1, cj),(ci, cj-1),(ci, cj)]

    def apply(self, state: MatchesState) -> ColorState:
        from collections import defaultdict, Counter
        by_schema: Dict[str, List[List[List[Optional[int]]]]] = defaultdict(list)
        for m in state.matches:
            sc = m.get("schema")
            mg = m.get("match")
            if sc is None or mg is None:
                continue
            key = str((tuple(len(row) for row in sc), tuple(tuple(row) for row in sc)))
            by_schema[key].append(mg)
        schema_modes: List[int] = []
        for mg_list in by_schema.values():
            vals: List[int] = []
            for mg in mg_list:
                nr = len(mg); nc = len(mg[0]) if nr>0 else 0
                if nr==0 or nc==0: continue
                if self.cross_only:
                    for (i,j) in self._cross_positions(nr, nc):
                        v = mg[i][j]
                        if v is None or v==0: continue
                        vals.append(int(v))
                else:
                    for row in mg:
                        for v in row:
                            if v is None or v==0: continue
                            vals.append(int(v))
            if vals:
                c = Counter(vals)
                top = max(c.values())
                mode_vals = [k for k,v in c.items() if v==top]
                schema_modes.append(int(min(mode_vals)))
        if not schema_modes:
            return ColorState(0)
        # final mode across schema modes
        from collections import Counter
        c2 = Counter(schema_modes)
        top = max(c2.values())
        mode_vals = [k for k,v in c2.items() if v==top]
        return ColorState(int(min(mode_vals)))


class OpUniformColorFromSchemaConstantsOnly(Operation[MatchesState, ColorState]):
    input_type = MatchesState
    output_type = ColorState

    label = "uniform_from_schema_constants"

    def apply(self, state: MatchesState) -> ColorState:
        vals: List[int] = []
        for m in state.matches:
            schema = m.get("schema")
            mg = m.get("match")
            if schema is None or mg is None:
                continue
            nr = len(schema)
            nc = len(schema[0]) if nr>0 else 0
            for i in range(nr):
                for j in range(nc):
                    s = schema[i][j]
                    if isinstance(s, int):
                        v = mg[i][j]
                        if v is None:
                            continue
                        vv = int(v)
                        if vv != 0:
                            vals.append(vv)
        if not vals:
            return ColorState(0)
        return ColorState(_mode_int(vals))


class OpUniformColorFromMatchesExcludeGlobal(Operation[MatchesState, ColorState]):
    input_type = MatchesState
    output_type = ColorState

    label = "uniform_from_matches_excl_global"

    def __init__(self, cross_only: bool = False):
        self.cross_only = bool(cross_only)

    def _cross_positions(self, nr: int, nc: int):
        if nr == 1 and nc == 3:
            return [(0,0),(0,2)]
        if nr == 3 and nc == 1:
            return [(0,0),(2,0)]
        ci, cj = nr//2, nc//2
        if nr%2==1 and nc%2==1:
            return [(ci-1,cj),(ci+1,cj),(ci,cj-1),(ci,cj+1)]
        if nr%2==1 and nc%2==0:
            return [(ci, cj-1),(ci, cj)]
        if nr%2==0 and nc%2==1:
            return [(ci-1, cj),(ci, cj)]
        return [(ci-1, cj-1),(ci-1, cj),(ci, cj-1),(ci, cj)]

    def apply(self, state: MatchesState) -> ColorState:
        g = state.grid
        # Global nonzero mode of grid
        nz = g[g!=0]
        g_mode = None
        if nz.size > 0:
            vals, cnt = np.unique(nz, return_counts=True)
            g_mode = int(vals[int(np.argmax(cnt))])
        vals_out: List[int] = []
        for m in state.matches:
            mg = m.get("match")
            schema = m.get("schema")
            if mg is None:
                continue
            if self.cross_only and schema is not None:
                nr = len(schema); nc = len(schema[0]) if nr>0 else 0
                for (i,j) in self._cross_positions(nr, nc):
                    v = mg[i][j]
                    if v is None:
                        continue
                    vv = int(v)
                    if vv == 0 or (g_mode is not None and vv == g_mode):
                        continue
                    vals_out.append(vv)
            else:
                for row in mg:
                    for v in row:
                        if v is None:
                            continue
                        vv = int(v)
                        if vv == 0 or (g_mode is not None and vv == g_mode):
                            continue
                        vals_out.append(vv)
        if not vals_out:
            return ColorState(0)
        return ColorState(_mode_int(vals_out))

# Composed program body used in abstraction space: PatternOverlayExtractor |> UniformPatternPredicate |> OutputAgreedColor.
def predict_window_nxm_uniform_color(grid: List[List[int]], color: int) -> int:
    # Typed pipeline: Grid -> OverlayContext -> Color
    gstate = GridState(np.asarray(grid, dtype=int))
    pipeline = Pipeline([
        OpBrightOverlayIdentity(kind="window_nxm", color=color),
        OpUniformPatternPredicate(),
    ])
    out = pipeline.run(gstate)
    assert isinstance(out, ColorState)
    return int(out.color)

def predict_with_pattern_kind(grid: List[List[int]], kind: str, color: int) -> int:
    gstate = GridState(np.asarray(grid, dtype=int))
    pipeline = Pipeline([
        OpBrightOverlayIdentity(kind=kind, color=color),
        OpUniformPatternPredicate(),
    ])
    out = pipeline.run(gstate)
    assert isinstance(out, ColorState)
    return int(out.color)

class OpBrightOverlayAllWindows(Operation[GridState, OverlayContext]):
    input_type = GridState
    output_type = OverlayContext

    def __init__(self, absx: Optional[PatternOverlayExtractor] = None, window_shape: Optional[tuple[int,int]] = None):
        self.absx = absx or PatternOverlayExtractor()
        self.kind = "window_nxm_all"
        self.color = None  # colorless
        self.window_shape = tuple(window_shape) if window_shape is not None else None
        self.label = "overlay_window_nxm_all"

    def apply(self, state: GridState) -> OverlayContext:
        g = state.grid
        ws: Optional[tuple[int,int]] = self.window_shape if self.window_shape is not None else WINDOW_SHAPE_DEFAULT
        ovs = detect_overlays(g.tolist(), kind=self.kind, color=0, window_shape=ws)
        H, W = g.shape
        count = len(ovs)
        max_contrast = max([ov["contrast"] for ov in ovs], default=0.0)
        total_area = sum([ov["area"] for ov in ovs]) if ovs else 0
        total_area_frac = float(total_area) / float(H * W) if H * W > 0 else 0.0
        stats = dict(count=count, max_contrast=max_contrast, total_area=total_area, total_area_frac=total_area_frac)
        self.absx.overlays = ovs
        self.absx.last_stats = stats
        return OverlayContext(g, ovs, stats, self.kind, None)


class OpFilterWindowsCenterValue(Operation[OverlayContext, OverlayContext]):
    input_type = OverlayContext
    output_type = OverlayContext

    def __init__(self, const_value: int):
        self.const_value = int(const_value)
        self.label = f"filter_windows_center({self.const_value})"

    def apply(self, state: OverlayContext) -> OverlayContext:
        target = int(self.const_value)
        ovs: List[dict] = []
        for ov in state.overlays:
            win = ov.get("window")
            if win is None:
                continue
            nr = len(win); nc = len(win[0]) if nr>0 else 0
            if nr==0 or nc==0:
                continue
            ci, cj = nr//2, nc//2
            try:
                if int(win[ci][cj]) == target:
                    ovs.append(ov)
            except Exception:
                continue
        g = state.grid
        H,W = g.shape
        count = len(ovs)
        max_contrast = max([ov.get("contrast", 0.0) for ov in ovs], default=0.0)
        total_area = sum([ov.get("area", 0) for ov in ovs]) if ovs else 0
        total_area_frac = float(total_area) / float(H*W) if H*W>0 else 0.0
        stats = dict(count=count, max_contrast=max_contrast, total_area=total_area, total_area_frac=total_area_frac)
        return OverlayContext(state.grid, ovs, stats, state.kind, getattr(state, 'color', None))


def _build_combined_schema_from_windows(wins: List[np.ndarray]) -> List[List[Union[int, str]]]:
    if not wins:
        return [["*"]]
    hh, ww = wins[0].shape
    # Constants per position across windows
    pos_vals: List[set[int]] = []
    for i in range(hh):
        for j in range(ww):
            vals = {int(win[i, j]) for win in wins}
            pos_vals.append(vals)
    is_const = [len(s) == 1 for s in pos_vals]
    const_val: List[Optional[int]] = [next(iter(s)) if len(s) == 1 else None for s in pos_vals]
    # Equality relation among non-constant positions
    npos = hh * ww
    adj = [[False] * npos for _ in range(npos)]
    for a in range(npos):
        adj[a][a] = True
    for a in range(npos):
        if is_const[a]:
            continue
        ai, aj = divmod(a, ww)
        for b in range(a + 1, npos):
            if is_const[b]:
                continue
            bi, bj = divmod(b, ww)
            equal_all = True
            for win in wins:
                if int(win[ai, aj]) != int(win[bi, bj]):
                    equal_all = False
                    break
            if equal_all:
                adj[a][b] = adj[b][a] = True
    # Connected components → variable groups
    visited = [False] * npos
    comps: List[List[int]] = []
    for v in range(npos):
        if visited[v] or is_const[v]:
            continue
        stack = [v]
        visited[v] = True
        comp = [v]
        while stack:
            u = stack.pop()
            for w in range(npos):
                if not visited[w] and adj[u][w]:
                    visited[w] = True
                    stack.append(w)
                    comp.append(w)
        if len(comp) >= 2:
            comps.append(sorted(comp))
    # Assemble schema grid
    schema: List[List[Union[int, str]]] = [["*" for _ in range(ww)] for _ in range(hh)]
    for p in range(npos):
        if is_const[p]:
            i, j = divmod(p, ww)
            schema[i][j] = int(const_val[p]) if const_val[p] is not None else "*"
    var_tokens = ("X", "Y", "Z", "U", "V", "W")
    next_var = 0
    for comp in comps:
        tok = var_tokens[min(next_var, len(var_tokens) - 1)]
        next_var += 1
        for p in comp:
            i, j = divmod(p, ww)
            schema[i][j] = tok
    return schema


class OpMatchCombinedSchemaAcrossGrid(Operation[OverlayContext, MatchesState]):
    input_type = OverlayContext
    output_type = MatchesState

    def __init__(self, cross_only: bool = False):
        self.cross_only = bool(cross_only)
        self.label = "match_combined_schema"

    def apply(self, state: OverlayContext) -> MatchesState:
        # Build combined schema from current overlays’ windows, then match it across the grid
        wins: List[np.ndarray] = []
        for ov in state.overlays:
            win = ov.get("window")
            if win is None:
                continue
            wins.append(np.array(win, dtype=int))
        if not wins:
            return MatchesState(state.grid, [])
        schema = _build_combined_schema_from_windows(wins)
        g = state.grid
        H, W = g.shape
        nr, nc = len(schema), len(schema[0]) if len(schema)>0 else 0
        up = (nr - 1) // 2
        down = nr // 2
        left = (nc - 1) // 2
        right = nc // 2
        rmin, rmax = up, H - 1 - down
        cmin, cmax = left, W - 1 - right
        matches: List[dict] = []
        for r in range(rmin, rmax + 1):
            for c in range(cmin, cmax + 1):
                y1 = r - up; x1 = c - left
                y2 = r + down; x2 = c + right
                win = g[y1:y2+1, x1:x2+1]
                mg = _schema_match_window(schema, win)
                if mg is not None:
                    matches.append({
                        "y1": int(y1+1), "x1": int(x1+1), "y2": int(y2+1), "x2": int(x2+1),
                        "match": mg, "schema": schema,
                    })
        return MatchesState(g, matches)


class OpMatchFixedSchema(Operation[GridState, MatchesState]):
    input_type = GridState
    output_type = MatchesState

    def __init__(self, schema: List[List[Union[int, str]]], label: Optional[str] = None):
        self.schema = schema
        self.label = label or "match_fixed_schema"

    def apply(self, state: GridState) -> MatchesState:
        g = np.asarray(state.grid, dtype=int)
        H, W = g.shape
        schema = self.schema
        nr, nc = len(schema), (len(schema[0]) if len(schema)>0 else 0)
        up, down = (nr-1)//2, nr//2
        left, right = (nc-1)//2, nc//2
        matches: List[dict] = []
        if nr == 0 or nc == 0:
            return MatchesState(g, matches)
        rmin, rmax = up, H - 1 - down
        cmin, cmax = left, W - 1 - right
        for r in range(rmin, rmax + 1):
            for c in range(cmin, cmax + 1):
                y1, x1 = r - up, c - left
                y2, x2 = r + down, c + right
                win = g[y1:y2+1, x1:x2+1]
                mg = _schema_match_window(schema, win)
                if mg is not None:
                    matches.append({
                        "y1": int(y1+1), "x1": int(x1+1), "y2": int(y2+1), "x2": int(x2+1),
                        "match": mg, "schema": schema,
                    })
        return MatchesState(g, matches)


def build_intersected_universal_schemas_for_task(task: Dict, *, window_shape: tuple[int,int]=(3,3), center_value: int = 4, splits: tuple[str,...] = ("train","test")) -> Dict[tuple[int,int], List[List[Union[int,str]]]]:
    """For each position (ri,rj) in the window, build the universal schema intersection across all inputs
    for the cohort of windows whose (ri,rj) equals center_value. Only positions present in every input
    are returned."""
    ri_max, rj_max = int(window_shape[0]), int(window_shape[1])
    per_pos_schemas: Dict[tuple[int,int], List[List[List[Union[int,str]]]]] = {}
    present_all: Dict[tuple[int,int], bool] = {}
    for ri in range(ri_max):
        for rj in range(rj_max):
            per_pos_schemas[(ri,rj)] = []
            present_all[(ri,rj)] = True
    for split in splits:
        for ex in task.get(split, []):
            g = np.array(ex["input"], dtype=int)
            ov = OpBrightOverlayAllWindows(window_shape=window_shape).apply(GridState(g))
            wins_by_pos: Dict[tuple[int,int], List[np.ndarray]] = { (ri,rj): [] for ri in range(ri_max) for rj in range(rj_max) }
            for o in ov.overlays:
                w = np.array(o.get("window"), dtype=int)
                for ri in range(ri_max):
                    for rj in range(rj_max):
                        try:
                            if int(w[ri, rj]) == int(center_value):
                                wins_by_pos[(ri,rj)].append(w)
                        except Exception:
                            pass
            for pos, wins in wins_by_pos.items():
                if not wins:
                    present_all[pos] = False
                else:
                    per_pos_schemas[pos].append(_build_combined_schema_from_windows(wins))
    out: Dict[tuple[int,int], List[List[Union[int,str]]]] = {}
    for pos, ok in present_all.items():
        if not ok:
            continue
        schemas_here = per_pos_schemas[pos]
        if schemas_here:
            out[pos] = intersect_universal_schemas(schemas_here)
    return out


def intersect_universal_schemas(schemas: List[List[List[Union[int, str]]]]) -> List[List[Union[int, str]]]:
    """Intersect per-grid universal schemas into a cross-grid universal schema.

    Semantics: a constraint (constant at a position, or equality between positions)
    is kept only if it holds in every input schema. Constants must agree across all
    inputs to remain constants; otherwise, if equality between two positions holds
    in every input (either both positions are the same constant in that input or
    belong to the same variable group there), we mark them as a variable group here.
    All other positions become '*'.
    """
    if not schemas:
        return [["*"]]
    hh = len(schemas[0])
    ww = len(schemas[0][0]) if hh > 0 else 0
    # Sanity: all shapes equal
    for sc in schemas:
        if len(sc) != hh or (hh > 0 and len(sc[0]) != ww):
            raise ValueError("All schemas must share the same shape for intersection")

    npos = hh * ww
    # Determine positions that are constants with the same value across all schemas
    is_const_global = [True] * npos
    const_value_global: List[Optional[int]] = [None] * npos
    for p in range(npos):
        i, j = divmod(p, ww)
        vals: List[Optional[int]] = []
        ok = True
        for sc in schemas:
            s = sc[i][j]
            if isinstance(s, int):
                vals.append(int(s))
            else:
                ok = False
                break
        if ok and vals and all(v == vals[0] for v in vals):
            const_value_global[p] = int(vals[0])
            is_const_global[p] = True
        else:
            const_value_global[p] = None
            is_const_global[p] = False

    # Equality relation among non-constant positions that holds in every schema
    eq_all = [[False] * npos for _ in range(npos)]
    for a in range(npos):
        eq_all[a][a] = True
    for a in range(npos):
        if is_const_global[a]:
            continue
        ai, aj = divmod(a, ww)
        for b in range(a + 1, npos):
            if is_const_global[b]:
                continue
            bi, bj = divmod(b, ww)
            equal_everywhere = True
            for sc in schemas:
                sa = sc[ai][aj]
                sb = sc[bi][bj]
                # If either is '*', this input does not guarantee equality
                if isinstance(sa, str) and sa == "*":
                    equal_everywhere = False
                    break
                if isinstance(sb, str) and sb == "*":
                    equal_everywhere = False
                    break
                # Both constants: must be equal numerically in this input
                if isinstance(sa, int) and isinstance(sb, int):
                    if int(sa) != int(sb):
                        equal_everywhere = False
                        break
                # Both variables: must be the same token in this input schema
                elif isinstance(sa, str) and isinstance(sb, str):
                    if sa != sb:
                        equal_everywhere = False
                        break
                else:
                    # One const, one var → not universally equal in this input
                    equal_everywhere = False
                    break
            if equal_everywhere:
                eq_all[a][b] = eq_all[b][a] = True

    # Connected components among non-constant positions with equality everywhere
    visited = [False] * npos
    comps: List[List[int]] = []
    for p in range(npos):
        if visited[p] or is_const_global[p]:
            continue
        stack = [p]
        visited[p] = True
        comp = [p]
        while stack:
            u = stack.pop()
            for v in range(npos):
                if not visited[v] and eq_all[u][v]:
                    visited[v] = True
                    stack.append(v)
                    comp.append(v)
        if len(comp) >= 2:
            comps.append(sorted(comp))

    # Assemble intersected schema
    out: List[List[Union[int, str]]] = [["*" for _ in range(ww)] for _ in range(hh)]
    for p in range(npos):
        if is_const_global[p] and const_value_global[p] is not None:
            i, j = divmod(p, ww)
            out[i][j] = int(const_value_global[p])
    var_tokens = ("X", "Y", "Z", "U", "V", "W")
    next_var = 0
    for comp in comps:
        tok = var_tokens[min(next_var, len(var_tokens) - 1)]
        next_var += 1
        for p in comp:
            i, j = divmod(p, ww)
            out[i][j] = tok
    # Runtime self-check: every equality encoded in 'out' must be supported by all inputs
    def _eq_in_schema(sc, a, b):
        ai, aj = divmod(a, ww); bi, bj = divmod(b, ww)
        sa, sb = sc[ai][aj], sc[bi][bj]
        if isinstance(sa, str) and sa == "*": return False
        if isinstance(sb, str) and sb == "*": return False
        if isinstance(sa, int) and isinstance(sb, int): return int(sa) == int(sb)
        if isinstance(sa, str) and isinstance(sb, str): return sa == sb
        return False
    violations: List[str] = []
    # Check constants
    for p in range(npos):
        if isinstance(out[p//ww][p%ww], int):
            v = int(out[p//ww][p%ww])
            for sc in schemas:
                s = sc[p//ww][p%ww]
                if not (isinstance(s, int) and int(s) == v):
                    violations.append(f"const mismatch at pos {p} expected {v} got {s}")
                    break
    # Check variable equalities
    pos_by_tok: Dict[str, List[int]] = {}
    for p in range(npos):
        s = out[p//ww][p%ww]
        if isinstance(s, str) and s != "*":
            pos_by_tok.setdefault(s, []).append(p)
    for tok, poss in pos_by_tok.items():
        for i in range(len(poss)):
            for j in range(i+1, len(poss)):
                a, b = poss[i], poss[j]
                for sc in schemas:
                    if not _eq_in_schema(sc, a, b):
                        violations.append(f"equality mismatch for token {tok} between {a} and {b}")
                        break
    if violations:
        print("[warn] intersect_universal_schemas: semantic check failed:")
        for msg in violations[:8]:
            print(" -", msg)
    return out


def validate_universal_schema_on_windows(schema: List[List[Union[int,str]]], wins: List[np.ndarray]) -> Dict[str, Union[int, List[dict]]]:
    """Validate that 'schema' matches every window in 'wins'. Returns a report."""
    violations: List[dict] = []
    ok = 0
    for idx, win in enumerate(wins):
        mg = _schema_match_window(schema, win)
        if mg is None:
            violations.append({"index": idx, "window": win.tolist()})
        else:
            ok += 1
    return {"total": len(wins), "ok": ok, "violations": violations}


def debug_check_universal_intersection(task: Dict, *, window_shape: tuple[int,int]=(3,3), center_value: int = 4, splits: tuple[str,...] = ("train","test")) -> None:
    """Build per-grid universal schemas for windows with center == center_value over given splits
    (e.g., ("train", "test")), intersect them, and validate that the intersection holds on every
    center==value window of every grid across those splits."""
    # Collect per-grid cohorts and schemas
    per_grid_wins: List[List[np.ndarray]] = []
    per_grid_schemas: List[List[List[Union[int,str]]]] = []
    for split in splits:
        for ex in task.get(split, []):
            g = np.array(ex["input"], dtype=int)
            ov = OpBrightOverlayAllWindows(window_shape=window_shape).apply(GridState(g))
            wins: List[np.ndarray] = []
            for o in ov.overlays:
                w = np.array(o.get("window"), dtype=int)
                nr, nc = w.shape
                if nr>0 and nc>0 and int(w[nr//2, nc//2]) == int(center_value):
                    wins.append(w)
            per_grid_wins.append(wins)
            per_grid_schemas.append(_build_combined_schema_from_windows(wins))
    inter = intersect_universal_schemas(per_grid_schemas)
    # Validate intersection schema against every cohort window
    any_fail = False
    for gi, wins in enumerate(per_grid_wins, start=1):
        rep = validate_universal_schema_on_windows(inter, wins)
        if rep["ok"] != rep["total"]:
            any_fail = True
            print(f"[warn] universal semantics failed on grid {gi}: {rep['ok']}/{rep['total']} windows passed")
            for v in rep["violations"][:3]:
                print("  - window (sample):")
                for row in v["window"]:
                    print("    ", row)
    if not any_fail:
        print("Universal semantics validated: intersection schema holds for all cohort windows across inputs.")
    # Print the intersection schema for visibility
    print("Intersection schema:")
    for row in inter:
        print(" ", "[" + ", ".join(str(x) for x in row) + "]")

def predict_with_pattern_kind_shape(grid: List[List[int]], kind: str, color: int, *, window_shape: Optional[tuple[int,int]]) -> int:
    gstate = GridState(np.asarray(grid, dtype=int))
    pipeline = Pipeline([
        OpBrightOverlayIdentity(kind=kind, color=color, window_shape=window_shape),
        OpUniformPatternPredicate(),
    ])
    out = pipeline.run(gstate)
    assert isinstance(out, ColorState)
    return int(out.color)

# Local-structure color rules
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
    

# ===================== G Ops (Core composition: choose -> out) =====================
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

def choose_center_best_flank(x_hat: np.ndarray) -> int:
    g = np.asarray(x_hat, dtype=int)
    H, W = g.shape
    best_c = 0
    best_hits = -1
    for c0 in range(1,10):
        hits = 0
        for r in range(H):
            for c in range(W):
                if int(g[r,c]) != c0:
                    continue
                if c-1>=0 and c+1<W:
                    a,b = int(g[r,c-1]), int(g[r,c+1])
                    if a==b and a!=0:
                        hits+=1
                if r-1>=0 and r+1<H:
                    a,b = int(g[r-1,c]), int(g[r+1,c])
                    if a==b and a!=0:
                        hits+=1
        if hits>best_hits or (hits==best_hits and c0<best_c):
            best_hits = hits; best_c = c0
    return best_c if best_hits>0 else 0

def choose_center_best_cross(x_hat: np.ndarray) -> int:
    g = np.asarray(x_hat, dtype=int)
    H, W = g.shape
    best_c = 0
    best_hits = -1
    for c0 in range(1,10):
        wins = _collect_full_33_windows_for_center(g, c0)
        hits = 0
        for wv in wins:
            vals = [int(wv[1+dr,1+dc]) for (dr,dc) in _REL_CROSS_33]
            if len(set(vals))==1 and vals[0]!=0:
                hits+=1
        if hits>best_hits or (hits==best_hits and c0<best_c):
            best_hits = hits; best_c = c0
    return best_c if best_hits>0 else 0

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

def _compose_center_to_output(name: str, choose_fn: Callable[[np.ndarray], int], out_fn: Callable[[np.ndarray, int], int]) -> Tuple[str, Callable[[np.ndarray], int]]:
    def f(x: np.ndarray) -> int:
        c = int(choose_fn(x))
        return int(out_fn(x, c)) if c!=0 else 0
    return (name, f)

def _compose_choose_first(name: str, c1: Callable[[np.ndarray], int], c2: Callable[[np.ndarray], int]) -> Tuple[str, Callable[[np.ndarray], int]]:
    def f(x: np.ndarray) -> int:
        v = int(c1(x))
        return v if v!=0 else int(c2(x))
    return (name, f)
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

# ---- Typed operation wrappers for G composition (choose -> out) ----
class OpChooseCenterCrossImplied33(Operation[GridState, CenterState]):
    input_type = GridState
    output_type = CenterState

    label = "choose_cross_implied_33"

    def apply(self, state: GridState) -> CenterState:
        c = int(choose_center_cross_implied_33(state.grid))
        if c == 0:
            raise OpFailure("choose_cross_implied_33 failed: no center")
        return CenterState(state.grid, c)


class OpChooseCenterBestFlank(Operation[GridState, CenterState]):
    input_type = GridState
    output_type = CenterState

    label = "choose_best_flank"

    def apply(self, state: GridState) -> CenterState:
        c = int(choose_center_best_flank(state.grid))
        if c == 0:
            raise OpFailure("choose_best_flank failed: no center")
        return CenterState(state.grid, c)


class OpChooseCenterBestCross(Operation[GridState, CenterState]):
    input_type = GridState
    output_type = CenterState

    label = "choose_best_cross"

    def apply(self, state: GridState) -> CenterState:
        c = int(choose_center_best_cross(state.grid))
        if c == 0:
            raise OpFailure("choose_best_cross failed: no center")
        return CenterState(state.grid, c)


class OpOutCrossModeForCenter33(Operation[CenterState, ColorState]):
    input_type = CenterState
    output_type = ColorState

    label = "out_cross_mode_33"

    def apply(self, state: CenterState) -> ColorState:
        y = int(out_mode_cross_for_center_33(state.grid, state.center_color))
        if y == 0:
            raise OpFailure("out_cross_mode_33 failed: no color")
        return ColorState(y)


class OpOutFlankModeForCenter(Operation[CenterState, ColorState]):
    input_type = CenterState
    output_type = ColorState

    label = "out_flank_mode"

    def apply(self, state: CenterState) -> ColorState:
        y = int(out_mode_flank_for_center(state.grid, state.center_color))
        if y == 0:
            raise OpFailure("out_flank_mode failed: no color")
        return ColorState(y)

# Explicit op registries (for documentation and composition seeding)
# G ops: typed, fixed arity
G_TYPED_OPS: List[Operation] = [
    OpChooseCenterCrossImplied33(),
    OpChooseCenterBestFlank(),
    OpChooseCenterBestCross(),
    OpOutCrossModeForCenter33(),
    OpOutFlankModeForCenter(),
]

# A ops: two-stage pipeline types (GridState->OverlayContext, OverlayContext->ColorState)
# Overlay extractor is parameterized by kind, color, and optional shape; predicate is fixed.
A_OP_TYPE_SUMMARY: List[Tuple[str, str, str]] = [
    ("overlay_window_nxm", "GridState", "OverlayContext"),
    ("uniform_pattern_predicate", "OverlayContext", "ColorState"),
]
# ---- G Core: base rules ----
COLOR_RULES_BASE: List[Tuple[str, Callable[[np.ndarray], int]]] = [
    ("uniform_cross_everywhere_mode", sel_color_uniform_cross_everywhere_mode),
    ("argmax_uniform_cross_color_count", sel_color_argmax_uniform_cross_color_count),
    ("best_center_flank_mode", rule_best_center_flank_mode),
    ("best_center_cross_mode", rule_best_center_cross_mode),
    ("h3_flank_mode", rule_h3_flank_mode),
    ("v3_flank_mode", rule_v3_flank_mode),
]

# ---- G Core: simple compositional rules ----
def _compose_first_nonzero(name: str, *fs: Callable[[np.ndarray], int]) -> Tuple[str, Callable[[np.ndarray], int]]:
    def f(x: np.ndarray) -> int:
        for fn in fs:
            v = int(fn(x))
            if v != 0:
                return v
        return 0
    return (name, f)

def _compose_mode_nonzero(name: str, *fs: Callable[[np.ndarray], int]) -> Tuple[str, Callable[[np.ndarray], int]]:
    def f(x: np.ndarray) -> int:
        vals = [int(fn(x)) for fn in fs]
        vals = [v for v in vals if v != 0]
        if not vals:
            return 0
        return _mode_int(vals)
    return (name, f)

def _build_composed_rules(base: List[Tuple[str, Callable[[np.ndarray], int]]]) -> List[Tuple[str, Callable[[np.ndarray], int]]]:
    # Limit to the four most structural base rules to keep the space small
    pick_names = {
        "uniform_cross_everywhere_mode",
        "argmax_uniform_cross_color_count",
        "best_center_flank_mode",
        "best_center_cross_mode",
        "h3_flank_mode",
        "v3_flank_mode",
    }
    core = [(n, f) for (n, f) in base if n in pick_names]
    comps: List[Tuple[str, Callable[[np.ndarray], int]]] = []
    for i, (n1, f1) in enumerate(core):
        for j, (n2, f2) in enumerate(core):
            if i == j:
                continue
            comps.append(_compose_first_nonzero(f"first_nonzero({n1},{n2})", f1, f2))
            comps.append(_compose_mode_nonzero(f"mode_nonzero({n1},{n2})", f1, f2))
    # A couple of three-way compositions
    name_map = {n: f for (n, f) in core}
    def add_tri_first(a,b,c):
        if a in name_map and b in name_map and c in name_map:
            comps.append(_compose_first_nonzero(f"first_nonzero({a},{b},{c})", name_map[a], _compose_first_nonzero("_tmp", name_map[b], name_map[c])[1]))
    def add_tri_mode(a,b,c):
        if a in name_map and b in name_map and c in name_map:
            comps.append(_compose_mode_nonzero(f"mode_nonzero({a},{b},{c})", name_map[a], _compose_mode_nonzero("_tmp", name_map[b], name_map[c])[1]))
    add_tri_first("h3_flank_mode", "v3_flank_mode", "uniform_cross_everywhere_mode")
    add_tri_mode("h3_flank_mode", "v3_flank_mode", "argmax_uniform_cross_color_count")
    # Center-to-output compositions
    choose_map = {
        "choose_cross_implied_33": choose_center_cross_implied_33,
        "choose_best_flank": choose_center_best_flank,
        "choose_best_cross": choose_center_best_cross,
    }
    out_map = {
        "out_cross_mode_33": out_mode_cross_for_center_33,
        "out_flank_mode": out_mode_flank_for_center,
    }
    # Direct two-stage compositions
    for cn, cf in choose_map.items():
        for on, of in out_map.items():
            comps.append(_compose_center_to_output(f"compose({cn}->{on})", cf, of))
    # Fallback choose then output
    combos = [
        ("choose_first(crossImplied,bestCross)", choose_center_cross_implied_33, choose_center_best_cross),
        ("choose_first(bestCross,bestFlank)", choose_center_best_cross, choose_center_best_flank),
    ]
    for nm, c1, c2 in combos:
        choose_fn = _compose_choose_first(nm, c1, c2)[1]
        comps.append(_compose_center_to_output(f"compose({nm}->out_cross_mode_33)", choose_fn, out_mode_cross_for_center_33))
    return comps

# Build composed rules once.
COLOR_RULES_COMPOSED: List[Tuple[str, Callable[[np.ndarray], int]]] = _build_composed_rules(COLOR_RULES_BASE)

# Expose G as composed-only (function-style) for backward compatibility with older reporting.
COLOR_RULES: List[Tuple[str, Callable[[np.ndarray], int]]] = COLOR_RULES_COMPOSED

# ---- Typed composition engine ----
def _enumerate_typed_programs(
    task: Dict,
    ops: List[Operation],
    *,
    max_depth: int = 2,
    min_depth: int = 2,
    start_type: Type[State] = GridState,
    end_type: Type[State] = ColorState,
) -> List[Tuple[str, List[Operation]]]:
    # Build train pairs once
    train_pairs = [
        (np.array(ex["input"], dtype=int), int(ex["output"][0][0])) for ex in task["train"]
    ]

    def accepts_chain(seq: List[Operation]) -> bool:
        if not seq:
            return False
        if not (seq[0].input_type is start_type or issubclass(start_type, seq[0].input_type)):
            return False
        for a, b in zip(seq, seq[1:]):
            if not (b.input_type is a.output_type or issubclass(a.output_type, b.input_type)):
                return False
        return (seq[-1].output_type is end_type) or issubclass(seq[-1].output_type, end_type)

    def label_of(op: Operation) -> str:
        return getattr(op, "label", op.__class__.__name__)

    winners: List[Tuple[str, List[Operation]]] = []
    # Simple DFS over sequences up to max_depth
    from itertools import product
    # Expand sequences by chaining type-compatible ops
    def extend(seq: List[Operation]) -> List[List[Operation]]:
        outs: List[List[Operation]] = []
        last = seq[-1]
        for op in ops:
            if op.input_type is last.output_type or issubclass(last.output_type, op.input_type):
                outs.append(seq + [op])
        return outs

    # Start with ops that accept start_type
    seeds = [op for op in ops if op.input_type is start_type or issubclass(start_type, op.input_type)]
    frontier: List[List[Operation]] = [[op] for op in seeds]
    for depth in range(1, max_depth + 1):
        next_frontier: List[List[Operation]] = []
        for seq in frontier:
            if depth >= min_depth and accepts_chain(seq) and (seq[-1].output_type is end_type or issubclass(seq[-1].output_type, end_type)):
                # Evaluate on train
                ok = True
                for x, y in train_pairs:
                    state: State = GridState(x)
                    try:
                        for op in seq:
                            if not op.accepts(state):
                                raise OpFailure(f"type mismatch: {type(state).__name__} -> {op.__class__.__name__}")
                            state = op.apply(state)  # type: ignore[arg-type]
                        assert isinstance(state, ColorState)
                        if int(state.color) != y:
                            ok = False
                            break
                    except Exception:
                        ok = False
                        break
                if ok:
                    name = " |> ".join(label_of(op) for op in seq)
                    winners.append((name, seq))
            if depth < max_depth:
                next_frontier.extend(extend(seq))
        frontier = next_frontier
    return winners
# ===================== Enumeration & Printing =====================
# Enumerates programs that are correct on ALL training examples (README_clean.md §3–§4).
def enumerate_programs_for_task(task: Dict, num_preops: int = 200, seed: int = 11):
    # G core via typed composition engine (choose -> out), but keep node count from COLOR_RULES for continuity.
    # G typed ops (choose -> out)
    winners_g = _enumerate_typed_programs(task, G_TYPED_OPS, max_depth=2, min_depth=2, start_type=GridState, end_type=ColorState)
    programs_G = []
    for name, _ in winners_g:
        # Present in legacy style for G composed programs
        if name.startswith("choose_"):
            parts = name.split(" |> ")
            if len(parts) == 2 and parts[1].startswith("out_"):
                programs_G.append(f"compose({parts[0]}->{parts[1]})")
            else:
                programs_G.append(name)
        else:
            programs_G.append(name)
    total_G = len(COLOR_RULES)

    # Abstractions: enumerate both (overlay->predicate) and (overlay_all->schema_match->uniform_from_matches)
    colors = list(range(1, 10))
    shapes: List[tuple[int,int]] = [(1,3), (3,1), WINDOW_SHAPE_DEFAULT]
    # Node count heuristic: seeds = (window_nxm × colors × shapes) + (window_nxm_all × shapes)
    total_ABS = len(PATTERN_KINDS) * len(colors) * len(shapes) + len(shapes)
    abs_ops: List[Operation] = [
        OpUniformPatternPredicate(),
        OpFilterCrosslikeSchemas(),
        OpFilterSchemasWithConstant(),
        OpSchemaMatchAcrossGrid(),
        OpUniformColorFromMatches(),
        OpUniformColorPerSchemaThenMode(),
        OpUniformColorFromSchemaConstantsOnly(),
        OpUniformColorFromMatchesExcludeGlobal(),
    ]
    # Pre-instantiate overlay ops per (shape, color) for kind=window_nxm
    for shape in shapes:
        for c in colors:
            if not pattern_present_in_all_examples(task, "window_nxm", c, window_shape=shape):
                continue
            abs_ops.append(OpBrightOverlayIdentity(kind="window_nxm", color=c, window_shape=shape))
    # Add colorless overlay ops per shape
    for shape in shapes:
        # Only include if at least one full window exists in all examples (shape fits)
        ok_all = True
        for split in ("train", "test"):
            for ex in task.get(split, []):
                g = np.array(ex["input"], dtype=int)
                H,W = g.shape
                hh, ww = int(shape[0]), int(shape[1])
                up, down = (hh-1)//2, hh//2
                left, right = (ww-1)//2, ww//2
                if H <= up+down or W <= left+right:
                    ok_all = False
                    break
            if not ok_all:
                break
        if ok_all:
            abs_ops.append(OpBrightOverlayAllWindows(window_shape=shape))
    # Add filter-by-constant ops (c ∈ 1..9)
    for c in range(1, 10):
        abs_ops.append(OpFilterSchemasWithConstantValue(c))
    # Add center-constant filters (c ∈ 1..9)
    for c in range(1, 10):
        abs_ops.append(OpFilterSchemasCenterConstant(c))
    # Add a schema matcher variant with no limit to pick up all unique schemas
    abs_ops.append(OpSchemaMatchAcrossGrid(limit_schemas=None))
    # Add universal fixed-schema matchers derived from task (train+test) for 3x3 and center_value=4
    try:
        uni_schemas = build_intersected_universal_schemas_for_task(task, window_shape=(3,3), center_value=4, splits=("train","test"))
        for (ri,rj), schema in uni_schemas.items():
            label = f"match_universal_pos({ri},{rj})"
            abs_ops.append(OpMatchFixedSchema(schema, label=label))
    except Exception:
        pass
    # Enumerate up to depth 3 to allow schema-matching chains
    winners_abs = _enumerate_typed_programs(task, abs_ops, max_depth=4, min_depth=2, start_type=GridState, end_type=ColorState)
    programs_ABS = []
    for name, seq in winners_abs:
        # Pretty-print sequences in a readable, parameterized style
        if len(seq) == 2 and isinstance(seq[0], OpBrightOverlayIdentity) and isinstance(seq[1], OpUniformPatternPredicate):
            ov: OpBrightOverlayIdentity = seq[0]  # type: ignore[assignment]
            shape = ov.window_shape if ov.window_shape is not None else (WINDOW_SHAPE_DEFAULT if ov.kind == "window_nxm" else None)
            extra = f", window_shape={shape}" if shape is not None else ""
            programs_ABS.append(
                f"PatternOverlayExtractor(kind={ov.kind}, color={ov.color}{extra}) |> UniformPatternPredicate |> OutputAgreedColor"
            )
        elif len(seq) == 3 and isinstance(seq[0], OpBrightOverlayAllWindows) and isinstance(seq[1], OpSchemaMatchAcrossGrid) and isinstance(seq[2], OpUniformColorFromMatches):
            ov: OpBrightOverlayAllWindows = seq[0]  # type: ignore[assignment]
            shape = ov.window_shape if ov.window_shape is not None else WINDOW_SHAPE_DEFAULT
            programs_ABS.append(
                f"PatternOverlayExtractor(kind=window_nxm_all, window_shape={shape}) |> SchemaMatchAcrossGrid |> UniformColorFromMatches"
            )
        elif len(seq) == 4 and isinstance(seq[0], OpBrightOverlayAllWindows) and isinstance(seq[1], OpFilterSchemasWithConstant) and isinstance(seq[2], OpSchemaMatchAcrossGrid) and isinstance(seq[3], OpUniformColorPerSchemaThenMode):
            ov: OpBrightOverlayAllWindows = seq[0]  # type: ignore[assignment]
            shape = ov.window_shape if ov.window_shape is not None else WINDOW_SHAPE_DEFAULT
            programs_ABS.append(
                f"PatternOverlayExtractor(kind=window_nxm_all, window_shape={shape}) |> FilterSchemasWithConstant |> SchemaMatchAcrossGrid |> UniformColorPerSchemaThenMode"
            )
        elif len(seq) == 4 and isinstance(seq[0], OpBrightOverlayAllWindows) and isinstance(seq[1], OpFilterCrosslikeSchemas) and isinstance(seq[2], OpSchemaMatchAcrossGrid) and isinstance(seq[3], OpUniformColorPerSchemaThenMode):
            ov: OpBrightOverlayAllWindows = seq[0]  # type: ignore[assignment]
            shape = ov.window_shape if ov.window_shape is not None else WINDOW_SHAPE_DEFAULT
            programs_ABS.append(
                f"PatternOverlayExtractor(kind=window_nxm_all, window_shape={shape}) |> FilterCrosslikeSchemas |> SchemaMatchAcrossGrid |> UniformColorPerSchemaThenMode"
            )
        elif len(seq) == 4 and isinstance(seq[0], OpBrightOverlayAllWindows) and isinstance(seq[1], OpFilterSchemasWithConstantValue) and isinstance(seq[2], OpSchemaMatchAcrossGrid) and isinstance(seq[3], (OpUniformColorFromSchemaConstantsOnly, OpUniformColorPerSchemaThenMode, OpUniformColorFromMatchesExcludeGlobal)):
            ov: OpBrightOverlayAllWindows = seq[0]  # type: ignore[assignment]
            fop: OpFilterSchemasWithConstantValue = seq[1]  # type: ignore[assignment]
            shape = ov.window_shape if ov.window_shape is not None else WINDOW_SHAPE_DEFAULT
            if isinstance(seq[3], OpUniformColorFromSchemaConstantsOnly):
                agg = "UniformFromSchemaConstants"
            elif isinstance(seq[3], OpUniformColorPerSchemaThenMode):
                agg = "UniformColorPerSchemaThenMode"
            else:
                agg = "UniformFromMatchesExclGlobal"
            programs_ABS.append(
                f"PatternOverlayExtractor(kind=window_nxm_all, window_shape={shape}) |> FilterSchemasWithConstantValue({fop.const_value}) |> SchemaMatchAcrossGrid |> {agg}"
            )
        elif len(seq) == 2 and isinstance(seq[0], OpMatchFixedSchema) and isinstance(seq[1], (OpUniformColorFromMatchesExcludeGlobal, OpUniformColorPerSchemaThenMode, OpUniformColorFromSchemaConstantsOnly, OpUniformColorFromMatches)):
            m0 = seq[0]
            aggname = seq[1].__class__.__name__
            programs_ABS.append(f"{getattr(m0,'label','match_fixed_schema')} |> {aggname}")
        else:
            # Fallback: use labels
            programs_ABS.append(name)

    return {"G": {"nodes": total_G, "programs": programs_G},
            "ABS": {"nodes": total_ABS, "programs": sorted(set(programs_ABS))}}
# Pretty-prints the programs and node counts (README_clean.md §4).
def print_programs_for_task(task: Dict, num_preops: int = 200, seed: int = 11):
    res = enumerate_programs_for_task(task, num_preops=num_preops, seed=seed)
    print("=== Node counts ===")
    print(f"G core nodes: {res['G']['nodes']}")
    print(f"Overlay+predicate nodes: {res['ABS']['nodes']}")
    print("\n=== Programs found (G core) ===")
    if res["G"]["programs"]:
        for s in res["G"]["programs"]: print("-", s)
    else:
        print("(none)")
    print("\n=== Programs found (overlay abstraction + pattern check) ===")
    if res["ABS"]["programs"]:
        for s in res["ABS"]["programs"]: print("-", s)
    else:
        print("(none)")
    return res
def measure_spaces(task: Dict, num_preops: int = 200, seed: int = 11):
    import time
    train_pairs = [(np.array(ex["input"], dtype=int), int(ex["output"][0][0])) for ex in task["train"]]
    # G (no pre-ops)
    t0=time.perf_counter(); valid_G=[]; tried=0; tries_first=None; found=False
    for cn, cf in COLOR_RULES:
        tried+=1; ok=True
        for x,y in train_pairs:
            if int(cf(x)) != y: ok=False; break
        if ok:
            valid_G.append(cn)
            if not found: tries_first=tried; found=True
    t1=time.perf_counter()
    # ABS (typed enumeration including schema matching chains)
    t2=time.perf_counter()
    colors = list(range(1,10))
    shapes: List[tuple[int,int]] = [(1,3), (3,1), WINDOW_SHAPE_DEFAULT]
    # Build ABS ops as in enumerate_programs_for_task
    abs_ops: List[Operation] = [
        OpUniformPatternPredicate(), OpFilterCrosslikeSchemas(), OpFilterSchemasWithConstant(), OpSchemaMatchAcrossGrid(), OpUniformColorFromMatches(), OpUniformColorPerSchemaThenMode()
    ]
    for shape in shapes:
        for c in colors:
            if not pattern_present_in_all_examples(task, "window_nxm", c, window_shape=shape):
                continue
            abs_ops.append(OpBrightOverlayIdentity(kind="window_nxm", color=c, window_shape=shape))
    for shape in shapes:
        ok_all = True
        for split in ("train", "test"):
            for ex in task.get(split, []):
                g = np.array(ex["input"], dtype=int)
                H,W = g.shape
                hh, ww = int(shape[0]), int(shape[1])
                up, down = (hh-1)//2, hh//2
                left, right = (ww-1)//2, ww//2
                if H <= up+down or W <= left+right:
                    ok_all = False
                    break
            if not ok_all:
                break
        if ok_all:
            abs_ops.append(OpBrightOverlayAllWindows(window_shape=shape))
    winners_abs = _enumerate_typed_programs(task, abs_ops, max_depth=4, min_depth=2, start_type=GridState, end_type=ColorState)
    t3=time.perf_counter()
    nodes_abs = len(PATTERN_KINDS)*9*3 + 3  # add 3 colorless overlay seeds
    return {
        "G":{"nodes": len(COLOR_RULES), "programs_found": len(valid_G),
             "tries_to_first": tries_first, "time_sec": t1-t0},
        "ABS":{"nodes": nodes_abs, "programs_found": len(winners_abs),
               "tries_to_first": None, "time_sec": t3-t2},
    } 
