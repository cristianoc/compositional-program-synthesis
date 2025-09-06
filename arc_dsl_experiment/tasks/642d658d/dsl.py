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
    kind: str = "h3",
    color: int,
    min_repeats: int = 2,
    window_shape: Optional[tuple[int, int]] = None,
) -> List[dict]:
    # For window_nxm, propagate a configurable window shape (defaults to WINDOW_SHAPE_DEFAULT).
    ws = window_shape if window_shape is not None else (WINDOW_SHAPE_DEFAULT if kind == "window_nxm" else None)
    return detect_pattern_overlays(
        grid,
        kind=kind,  # type: ignore[arg-type]
        color=int(color),
        min_repeats=min_repeats,
        window_shape=ws,
    )
# Pattern kinds considered during search/enumeration
PATTERN_KINDS: List[str] = ["h3", "v3", "window_nxm"]
# Default window shape for window_nxm (n×m with n,m≥1). Used across detection and pretty-printing.
WINDOW_SHAPE_DEFAULT: tuple[int, int] = (3, 3)
# Optimization: pre-check that a pattern appears in all examples (train+test)
# Optimization: pre-check that a pattern appears in all examples (train+test)
def pattern_present_in_all_examples(task: Dict, kind: str, color: int) -> bool:
    # task is ARC-like dict with "train" and "test" splits
    for split in ("train", "test"):
        for ex in task.get(split, []):
            g = ex["input"]
            ovs = detect_overlays(g, kind=kind, color=int(color))
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

def combined_window_nxm_schema_string(task: Dict, color: int, *, window_shape: tuple[int,int]) -> str:
    schema = combined_window_nxm_schema(task, color, window_shape=window_shape)
    return "[" + ", ".join("[" + ", ".join(str(x) for x in row) + "]" for row in schema) + "]"
# ===================== Abstraction & Predicates =====================
def _cross_vals(g: np.ndarray, r1: int, c1: int) -> List[int]:
    r, c = r1-1, c1-1  # caller passes 1-based
    H, W = g.shape; vals=[]
    if r-1>=0: vals.append(int(g[r-1,c]))
    if r+1<H:  vals.append(int(g[r+1,c]))
    if c-1>=0: vals.append(int(g[r,c-1]))
    if c+1<W:  vals.append(int(g[r,c+1]))
    return vals
class PatternOverlayExtractor:
    def __init__(self):
        self.overlays: List[dict] = []
        self.last_stats: Dict[str, float] = {}


# Typed-DSL operations corresponding to the above components

class PreOpPalette(Operation[GridState, GridState]):
    input_type = GridState
    output_type = GridState

    def __init__(self, name: str, f: Callable[[np.ndarray], np.ndarray]):
        self.name = name
        self.f = f

    def apply(self, state: GridState) -> GridState:
        return GridState(self.f(state.grid))


class OpBrightOverlayIdentity(Operation[GridState, OverlayContext]):
    input_type = GridState
    output_type = OverlayContext

    def __init__(self, absx: Optional[PatternOverlayExtractor] = None, kind: str = "window_nxm", color: Optional[int] = None):
        self.absx = absx or PatternOverlayExtractor()
        self.kind = kind
        if color is None:
            raise ValueError("OpBrightOverlayIdentity requires explicit color")
        self.color = int(color)

    def apply(self, state: GridState) -> OverlayContext:
        g = state.grid
        # Use the configured pattern kind and color for overlays
        ovs = detect_overlays(g.tolist(), kind=self.kind, color=self.color)
        H, W = g.shape
        count = len(ovs)
        max_contrast = max([ov["contrast"] for ov in ovs], default=0.0)
        total_area = sum([ov["area"] for ov in ovs]) if ovs else 0
        total_area_frac = float(total_area) / float(H * W) if H * W > 0 else 0.0
        stats = dict(count=count, max_contrast=max_contrast, total_area=total_area, total_area_frac=total_area_frac)
        self.absx.overlays = ovs
        self.absx.last_stats = stats
        return OverlayContext(g, ovs, stats, self.kind, self.color)


class OpUniformPatternPredicate(Operation[OverlayContext, ColorState]):
    input_type = OverlayContext
    output_type = ColorState

    def apply(self, state: OverlayContext) -> ColorState:
        # Predicts a single output color from pattern overlays.
        # Semantics by kind:
        #  - h3: for each overlay center at selected color, check horizontal flanks (x,c,x);
        #        collect the flank color x when uniform and nonzero, then return the mode (tie→min).
        #  - v3: analogous, but on vertical flanks above/below the center.
        # If no kind-specific evidence is found, returns 0 (no guess).
        g = state.grid
        from collections import Counter
        flank_colors: List[int] = []
        kind = getattr(state, "kind", None)
        sel_color = getattr(state, "color", None)
        if sel_color is None:
            raise ValueError("OverlayContext missing required color")
        # Collect evidence depending on kind
        for ov in state.overlays:
            r, c = ov["center_row"] - 1, ov["center_col"] - 1
            if kind in ("h3", "v3"):
                if int(g[r, c]) != int(sel_color):
                    continue
            if kind == "h3":
                if c-1>=0 and c+1<g.shape[1]:
                    a, b = int(g[r, c-1]), int(g[r, c+1])
                    if a == b and a != 0:
                        flank_colors.append(a)
            elif kind == "v3":
                if r-1>=0 and r+1<g.shape[0]:
                    a, b = int(g[r-1, c]), int(g[r+1, c])
                    if a == b and a != 0:
                        flank_colors.append(a)
            
            elif kind == "window_nxm":
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

# Saved for later: schema-driven predicate variant for window_nxm.
class OpSchemaDrivenPatternPredicate(Operation[OverlayContext, ColorState]):
    input_type = OverlayContext
    output_type = ColorState

    def apply(self, state: OverlayContext) -> ColorState:
        g = state.grid
        from collections import Counter, defaultdict
        flank_colors: List[int] = []
        kind = getattr(state, "kind", None)
        if kind != "window_nxm":
            return ColorState(0)
        for ov in state.overlays:
            win = ov.get("window")
            if win is None:
                y1,x1,y2,x2 = ov["y1"]-1, ov["x1"]-1, ov["y2"]-1, ov["x2"]-1
                win_arr = g[y1:y2+1, x1:x2+1].copy()
                win = win_arr.astype(int).tolist()
            schema = ov.get("schema")
            if schema is None:
                continue
            nr = len(schema); nc = len(schema[0]) if nr>0 else 0
            if nr==0 or any(len(row)!=nc for row in schema):
                continue
            if len(win)!=nr or any(len(row)!=nc for row in win):
                continue
            groups: dict[str, list[tuple[int,int]]] = defaultdict(list)
            for i in range(nr):
                for j in range(nc):
                    tok = schema[i][j]
                    if isinstance(tok, str) and tok != '*':
                        groups[tok].append((i,j))
            for coords in groups.values():
                vals = [int(win[i][j]) for (i,j) in coords]
                if vals and len(set(vals))==1 and vals[0]!=0:
                    flank_colors.append(int(vals[0]))
        if flank_colors:
            cnt = Counter(flank_colors); top = max(cnt.values())
            cands = [k for k,v in cnt.items() if v==top]
            return ColorState(int(min(cands)))
        return ColorState(0)
def read_overlay_cross_colors(g: np.ndarray, overlays: List[dict]) -> List[Optional[int]]:
    out: List[Optional[int]] = []
    for ov in overlays:
        r = ov["center_row"]; c = ov["center_col"]
        vals = _cross_vals(g, r, c)
        if len(vals)==4 and len(set(vals))==1 and vals[0]!=0:
            out.append(vals[0])
        else:
            out.append(None)
    return out
def overlays_have_uniform_cross_color(g: np.ndarray, overlays: List[dict]):
    colors = read_overlay_cross_colors(g, overlays)
    if not colors: return (False, None, {"why":"no_overlays"})
    if any(c is None for c in colors): return (False, None, {"why":"non_uniform_cross"})
    if len(set(colors))!=1: return (False, None, {"why":"disagree"})
    return (True, colors[0], {"count": len(colors)})
def bright_overlay_cross_mode(g: np.ndarray, overlays: List[dict]) -> int:
    from collections import Counter
    colors=[]
    for ov in overlays:
        r = ov["center_row"]; c = ov["center_col"]
        vals = _cross_vals(g, r, c)
        if len(vals)==4 and len(set(vals))==1 and vals[0]!=0:
            colors.append(vals[0])
    if not colors: return 0
    cnt = Counter(colors); m = max(cnt.values())
    cands = [k for k,v in cnt.items() if v==m]
    return int(min(cands))
# Composed program body used in abstraction space: PatternOverlayExtractor |> UniformPatternPredicate |> OutputAgreedColor.
def predict_bright_overlay_uniform_cross(grid: List[List[int]], color: int) -> int:
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
# ===================== Core G: preops & color rules =====================
def identity(x: np.ndarray) -> np.ndarray: return x
# Pre-ops: only identity (no palette permutations needed in pattern-only system)
def build_preops_for_dataset(train_pairs: List[Tuple[np.ndarray,int]], num_preops: int = 200, seed: int = 11):
    preops: List[Tuple[str, Callable[[np.ndarray], np.ndarray]]] = [("identity", identity)]
    return preops
# Global/simple color rules
def rule_argmin_hist(x: np.ndarray) -> int:
    vals, cnt = np.unique(x[x!=0], return_counts=True)
    if len(vals)==0: return 0
    return int(vals[np.argmin(cnt)])
def rule_max_id(x: np.ndarray) -> int:
    vals = np.unique(x[x!=0])
    return int(vals.max()) if len(vals) else 0
# Local-structure color rules
def _ring8_vals(g: np.ndarray, r: int, c: int):
    H,W = g.shape; vals=[]
    for dr in (-1,0,1):
        for dc in (-1,0,1):
            if dr==0 and dc==0: continue
            rr,cc = r+dr, c+dc
            if 0<=rr<H and 0<=cc<W: vals.append(int(g[rr,cc]))
    return vals
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
def sel_color_uniform_ring_mode(x_hat: np.ndarray) -> int:
    g = np.asarray(x_hat, dtype=int); H,W = g.shape; picks=[]
    for r in range(H):
        for c in range(W):
            vals=_ring8_vals(g,r,c)
            if vals and len(set(vals))==1 and vals[0]!=0:
                picks.append(vals[0])
    if not picks:
        vals, cnt = np.unique(g[g!=0], return_counts=True)
        return int(vals[np.argmax(cnt)]) if len(vals) else 0
    return _mode_int(picks)
COLOR_RULES: List[Tuple[str, Callable[[np.ndarray], int]]] = [
    ("argmin_hist", rule_argmin_hist),
    ("max_id", rule_max_id),
    ("uniform_cross_everywhere_mode", sel_color_uniform_cross_everywhere_mode),
    ("argmax_uniform_cross_color_count", sel_color_argmax_uniform_cross_color_count),
]
# ===================== Enumeration & Printing =====================
# Enumerates programs that are correct on ALL training examples (README_clean.md §3–§4).
def enumerate_programs_for_task(task: Dict, num_preops: int = 200, seed: int = 11):
    train_pairs = [(np.array(ex["input"], dtype=int), int(ex["output"][0][0])) for ex in task["train"]]
    # G core (no pre-ops)
    total_G = len(COLOR_RULES); valid_G: List[str] = []
    for cn, cf in COLOR_RULES:
        ok=True
        for x,y in train_pairs:
            if int(cf(x)) != y:
                ok=False; break
        if ok: valid_G.append(cn)
    # Overlay + predicate across all pattern kinds and colors (no pre-ops)
    colors = list(range(1,10))
    total_ABS = len(PATTERN_KINDS) * len(colors)
    valid_ABS: List[Tuple[str,int]] = []
    for kind in PATTERN_KINDS:
        for c in colors:
            # Optimization: skip candidates that cannot work because pattern missing in some input
            if not pattern_present_in_all_examples(task, kind, c):
                continue
            ok=True
            for x,y in train_pairs:
                if predict_with_pattern_kind(x.tolist(), kind, c) != y:
                    ok=False; break
            if ok:
                valid_ABS.append((kind, c))
    programs_G = [f"{cn}" for cn in valid_G]
    programs_ABS = []
    for (kind, c) in valid_ABS:
        extra = ""
        if kind in ("h3", "v3"):
            extra = f", pattern=[X, {int(c)}, X]"
        elif kind == "window_nxm":
            # Include window shape only (centerless windows already carry per-window schema)
            extra = f", window_shape={WINDOW_SHAPE_DEFAULT}"
        programs_ABS.append(
            f"PatternOverlayExtractor(kind={kind}, color={c}{extra}) |> UniformPatternPredicate |> OutputAgreedColor"
        )
    return {"G":{"nodes": total_G, "programs": programs_G},
            "ABS":{"nodes": total_ABS, "programs": programs_ABS}}
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
    # ABS (all pattern kinds and colors; no pre-ops)
    t2=time.perf_counter(); valid_ABS: List[Tuple[str,int]]=[]; tried2=0; tries_first2=None; found2=False
    colors = list(range(1,10))
    for kind in PATTERN_KINDS:
        for c in colors:
            # Optimization: skip candidates that cannot work because pattern missing in some input
            if not pattern_present_in_all_examples(task, kind, c):
                continue
            tried2+=1; ok=True
            for x,y in train_pairs:
                if predict_with_pattern_kind(x.tolist(), kind, c) != y: ok=False; break
            if ok:
                valid_ABS.append((kind, c))
                if not found2: tries_first2=tried2; found2=True
    t3=time.perf_counter()
    return {
        "G":{"nodes": len(COLOR_RULES), "programs_found": len(valid_G),
             "tries_to_first": tries_first, "time_sec": t1-t0},
        "ABS":{"nodes": len(PATTERN_KINDS)*9, "programs_found": len(valid_ABS),
               "tries_to_first": tries_first2, "time_sec": t3-t2},
    } 
