# -----------------------------------------------------------------------------
# Overlay Abstraction Experiment Utilities
# This module implements:
#  • Overlay extractor: detect_bright_overlays  (README_clean.md §2.2, "Methods: Overlay extractor")
#  • Abstraction: universal schema matcher       (match_universal_pos)
#  • Aggregators                                 (UniformColorFrom...)
#  • Typed enumeration & printing
#  • Results reproduction harness
#  • Figures (PNG) generator
#  • Evaluation on train/test
#  • Abstraction: universal schema matcher        (schema matchers)
#  • Pattern predicate & predictor pipeline       (removed)
#  • G composed rules as reference baseline
# For numeric defaults of the detector, see README_clean.md: "Detector defaults (for reproducibility)"
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
# Removed overlay-based detection and helpers in favor of universal schema matching.
# Pattern kinds considered during search/enumeration
PATTERN_KINDS: List[str] = ["window_nxm"]
# Default window shape for window_nxm (n×m with n,m≥1). Used across detection and pretty-printing.
## Removed default window shape; shapes are explicitly enumerated where needed.
# Optimization: pre-check that a pattern appears in all examples (train+test)
# Optimization: pre-check that a pattern appears in all examples (train+test)
# Removed overlay-based pattern presence checks.

# ---------------------- Combined schema for window_nxm ----------------------
# Removed overlay window gathering; universal schemas are built via _iterate_full_windows + builders below.

# ===================== Abstraction & Predicates =====================
# Removed PatternOverlayExtractor and overlay-based operations.

# Typed-DSL operations corresponding to the above components

# ===================== A Ops (Overlay Abstraction) =====================
# Removed overlay ops; abstraction now starts from schema matchers directly.


## Removed: OpUniformPatternPredicate and overlay-based predicates.


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


# Removed colorless overlay-based schema matching ops (now using fixed schemas directly)


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


# Removed overlay-based schema filtering ops


# Removed


# Removed


# Removed


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


class OpUniformColorFromMatchesUniformNeighborhood(Operation[MatchesState, ColorState]):
    input_type = MatchesState
    output_type = ColorState

    label = "e "

    def _neighborhood_positions(self, nr: int, nc: int) -> List[tuple[int,int]]:
        # Mirror UniformPatternPredicate center-neighborhood semantics by shape parity
        if nr == 1 and nc == 3:
            return [(0,0),(0,2)]
        if nr == 3 and nc == 1:
            return [(0,0),(2,0)]
        if nr % 2 == 1 and nc % 2 == 1:
            ci, cj = nr//2, nc//2
            return [(ci-1,cj),(ci+1,cj),(ci,cj-1),(ci,cj+1)]
        if nr % 2 == 0 and nc % 2 == 0:
            i0, i1 = nr//2 - 1, nr//2
            j0, j1 = nc//2 - 1, nc//2
            return [(i0,j0),(i0,j1),(i1,j0),(i1,j1)]
        if nr % 2 == 1 and nc % 2 == 0:
            ci = nr//2; j0, j1 = nc//2 - 1, nc//2
            return [(ci,j0),(ci,j1)]
        if nr % 2 == 0 and nc % 2 == 1:
            cj = nc//2; i0, i1 = nr//2 - 1, nr//2
            return [(i0,cj),(i1,cj)]
        return []

    def apply(self, state: MatchesState) -> ColorState:
        from collections import Counter
        picks: List[int] = []
        for m in state.matches:
            mg = m.get("match")
            if mg is None:
                continue
            nr = len(mg); nc = len(mg[0]) if nr>0 else 0
            if nr==0 or nc==0:
                continue
            poss = self._neighborhood_positions(nr, nc)
            if not poss:
                continue
            vals: List[int] = []
            for (i,j) in poss:
                v = mg[i][j]
                if v is None:
                    vals = []
                    break
                vals.append(int(v))
            if not vals:
                continue
            if len(set(vals)) == 1 and vals[0] != 0:
                picks.append(int(vals[0]))
        if not picks:
            return ColorState(0)
        c = Counter(picks)
        top = max(c.values())
        mode_vals = [k for k,v in c.items() if v==top]
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
## Removed: predict_window_nxm_uniform_color (overlay-based).

## Removed: predict_with_pattern_kind (overlay-based).

# Removed OpBrightOverlayAllWindows (colorless overlay); use direct window enumeration helpers instead


## Removed: OpFilterWindowsCenterValue (overlay-based).


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
        # Build combined schema from current overlays' windows, then match it across the grid
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


class OpMatchAnyUniversalSchemas(Operation[GridState, MatchesState]):
    input_type = GridState
    output_type = MatchesState

    def __init__(self, schemas: List[List[List[Union[int, str]]]], label: Optional[str] = None):
        self.schemas = schemas
        self.label = label or "match_universal"

    def apply(self, state: GridState) -> MatchesState:
        g = np.asarray(state.grid, dtype=int)
        H, W = g.shape
        matches: List[dict] = []
        for schema in self.schemas:
            nr, nc = len(schema), (len(schema[0]) if len(schema) > 0 else 0)
            if nr == 0 or nc == 0:
                continue
            up, down = (nr - 1) // 2, nr // 2
            left, right = (nc - 1) // 2, nc // 2
            rmin, rmax = up, H - 1 - down
            cmin, cmax = left, W - 1 - right
            for r in range(rmin, rmax + 1):
                for c in range(cmin, cmax + 1):
                    y1, x1 = r - up, c - left
                    y2, x2 = r + down, c + right
                    win = g[y1:y2 + 1, x1:x2 + 1]
                    mg = _schema_match_window(schema, win)
                    if mg is not None:
                        matches.append({
                            "y1": int(y1 + 1), "x1": int(x1 + 1), "y2": int(y2 + 1), "x2": int(x2 + 1),
                            "match": mg, "schema": schema,
                        })
        return MatchesState(g, matches)


def _iterate_full_windows(g: np.ndarray, window_shape: tuple[int,int]) -> List[np.ndarray]:
    g = np.asarray(g, dtype=int)
    H, W = g.shape
    hh, ww = int(window_shape[0]), int(window_shape[1])
    up, down = (hh-1)//2, hh//2
    left, right = (ww-1)//2, ww//2
    wins: List[np.ndarray] = []
    for r in range(up, H - down):
        for c in range(left, W - right):
            y1, x1 = r - up, c - left
            y2, x2 = r + down, c + right
            wins.append(g[y1:y2+1, x1:x2+1].copy())
    return wins

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
            wins_all = _iterate_full_windows(g, window_shape)
            wins_by_pos: Dict[tuple[int,int], List[np.ndarray]] = { (ri,rj): [] for ri in range(ri_max) for rj in range(rj_max) }
            for w in wins_all:
                for ri in range(ri_max):
                    for rj in range(rj_max):
                        if int(w[ri, rj]) == int(center_value):
                            wins_by_pos[(ri,rj)].append(w)
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
            wins_all = _iterate_full_windows(g, window_shape)
            # select center==center_value in this shape
            sel: List[np.ndarray] = []
            hh, ww = int(window_shape[0]), int(window_shape[1])
            ci, cj = hh//2, ww//2
            for w in wins_all:
                if int(w[ci, cj]) == int(center_value):
                    sel.append(w)
            per_grid_wins.append(sel)
            per_grid_schemas.append(_build_combined_schema_from_windows(sel))
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

## Removed: predict_with_pattern_kind_shape (overlay-based).

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
    ("match_universal_pos(shape=(h,w))", "GridState", "MatchesState"),
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
def enumerate_programs_for_task(task: Dict, num_preops: int = 200, seed: int = 11, *, universal_shapes: Optional[List[tuple[int,int]]] = None):
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

    # Abstractions: enumerate universal fixed-schema pipelines only (no overlay-based seeds)
    shapes: List[tuple[int,int]] = [(1,3), (3,1), (3,3)]
    abs_ops: List[Operation] = [
        OpUniformColorFromMatches(),
        OpUniformColorPerSchemaThenMode(),
        OpUniformColorPerSchemaThenMode(cross_only=False),
        OpUniformColorFromSchemaConstantsOnly(),
        OpUniformColorFromMatchesExcludeGlobal(),
        OpUniformColorFromMatchesExcludeGlobal(cross_only=True),
        OpUniformColorFromMatchesUniformNeighborhood(),
    ]
    # Add universal fixed-schema matchers derived from task (train+test) for requested shapes and center_value=4
    # Reuse the same default shapes as PatternOverlayExtractor when not overridden
    shapes_universal: List[tuple[int,int]] = list(universal_shapes) if universal_shapes is not None else list(shapes)
    matcher_seeds = 0
    for ushape in shapes_universal:
        try:
            uni_schemas = build_intersected_universal_schemas_for_task(task, window_shape=tuple(ushape), center_value=4, splits=("train","test"))
            if uni_schemas:
                schemas_list = list(uni_schemas.values())
                abs_ops.append(OpMatchAnyUniversalSchemas(schemas_list, label=f"match_universal_pos(shape={tuple(ushape)})"))
                matcher_seeds += 1
        except Exception:
            continue
    # Node count heuristic: number of matcher seeds
    total_ABS = matcher_seeds
    # Enumerate up to depth 3 to allow schema-matching chains
    winners_abs = _enumerate_typed_programs(task, abs_ops, max_depth=4, min_depth=2, start_type=GridState, end_type=ColorState)
    programs_ABS = []
    for name, seq in winners_abs:
        # Pretty-print sequences in a readable, parameterized style
        if len(seq) == 2 and isinstance(seq[0], (OpMatchFixedSchema, OpMatchAnyUniversalSchemas)) and isinstance(seq[1], (OpUniformColorFromMatchesExcludeGlobal, OpUniformColorPerSchemaThenMode, OpUniformColorFromSchemaConstantsOnly, OpUniformColorFromMatches, OpUniformColorFromMatchesUniformNeighborhood)):
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
    shapes: List[tuple[int,int]] = [(1,3), (3,1), (3,3)]
    abs_ops: List[Operation] = [
        OpUniformColorFromMatches(), OpUniformColorPerSchemaThenMode(), OpUniformColorFromSchemaConstantsOnly(), OpUniformColorFromMatchesExcludeGlobal(), OpUniformColorFromMatchesUniformNeighborhood(), OpUniformColorPerSchemaThenMode(cross_only=False), OpUniformColorFromMatchesExcludeGlobal(cross_only=True)
    ]
    for ushape in shapes:
        try:
            uni_schemas = build_intersected_universal_schemas_for_task(task, window_shape=tuple(ushape), center_value=4, splits=("train","test"))
            if uni_schemas:
                schemas_list = list(uni_schemas.values())
                abs_ops.append(OpMatchAnyUniversalSchemas(schemas_list, label=f"match_universal_pos(shape={tuple(ushape)})"))
        except Exception:
            pass
    winners_abs = _enumerate_typed_programs(task, abs_ops, max_depth=4, min_depth=2, start_type=GridState, end_type=ColorState)
    t3=time.perf_counter()
    nodes_abs = 3  # one matcher seed per shape (default shapes)
    return {
        "G":{"nodes": len(COLOR_RULES), "programs_found": len(valid_G),
             "tries_to_first": tries_first, "time_sec": t1-t0},
        "ABS":{"nodes": nodes_abs, "programs_found": len(winners_abs),
               "tries_to_first": None, "time_sec": t3-t2},
    } 
