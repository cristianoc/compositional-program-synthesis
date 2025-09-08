# -----------------------------------------------------------------------------
# Grid to Matches Operations
# This module implements operations that convert Grid to Matches:
#  • Schema matching operations (OpMatchFixedSchema, OpMatchAnyUniversalSchemas)
#  • Universal schema building and intersection utilities
#  • Pattern position selection based on structural complexity
# -----------------------------------------------------------------------------

from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
from dsl_types.states import Operation, Grid, Matches


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
    const_val: List[int] = [next(iter(s)) if len(s) == 1 else -1 for s in pos_vals]  # -1 placeholder for non-const
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
            schema[i][j] = const_val[p]
    var_tokens = ("X", "Y", "Z", "U", "V", "W")
    next_var = 0
    for comp in comps:
        tok = var_tokens[min(next_var, len(var_tokens) - 1)]
        next_var += 1
        for p in comp:
            i, j = divmod(p, ww)
            schema[i][j] = tok
    return schema


def select_best_pattern_position(uni_schemas: Dict[tuple[int,int], List[List[Union[int,str]]]]) -> tuple[tuple[int,int], List[List[Union[int,str]]]]:
    """Select the pattern position with highest structural complexity from a set of universal schemas."""
    from collections import Counter
    
    def analyze_schema_structure(schema):
        """Analyze structural complexity of a schema."""
        if not schema or not schema[0]:
            return 0
        
        nr, nc = len(schema), len(schema[0])
        constraint_positions = 0
        variable_counts = Counter()
        
        for i in range(nr):
            for j in range(nc):
                cell = schema[i][j]
                if cell != '*':  # Not a wildcard
                    constraint_positions += 1
                    if isinstance(cell, str) and cell.isalpha():  # Variable
                        variable_counts[cell] += 1
        
        # Calculate structure score
        variable_relationships = sum(count - 1 for count in variable_counts.values() if count > 1)
        variable_diversity = len(variable_counts)
        relationship_bonus = variable_relationships * 2  # High weight for repeated variables
        diversity_bonus = variable_diversity
        
        return constraint_positions + relationship_bonus + diversity_bonus
    
    # Score all positions
    scored_positions = []
    for pos, schema in uni_schemas.items():
        score = analyze_schema_structure(schema)
        scored_positions.append((pos, schema, score))
    
    # Return the position with highest structural complexity
    if scored_positions:
        scored_positions.sort(key=lambda x: x[2], reverse=True)
        best_pos, best_schema, best_score = scored_positions[0]
        return best_pos, best_schema
    else:
        # Fallback to first available position
        first_pos = next(iter(uni_schemas.keys()))
        return first_pos, uni_schemas[first_pos]


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
    const_value_global: List[int] = [-1] * npos  # -1 placeholder for non-const positions
    for p in range(npos):
        i, j = divmod(p, ww)
        vals: List[int] = []
        ok = True
        for sc in schemas:
            s = sc[i][j]
            if isinstance(s, int):
                vals.append(s)
            else:
                ok = False
                break
        if ok and vals and all(v == vals[0] for v in vals):
            const_value_global[p] = vals[0]  # vals[0] is guaranteed to be int here
            is_const_global[p] = True
        else:
            const_value_global[p] = -1  # placeholder value
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
        if is_const_global[p]:
            i, j = divmod(p, ww)
            out[i][j] = const_value_global[p]
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


def validate_universal_schema_on_windows(schema: List[List[Union[int,str]]], wins: List[np.ndarray]) -> Dict[str, Any]:
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


class OpMatchFixedSchema(Operation[Grid, Matches]):
    input_type = Grid
    output_type = Matches

    def __init__(self, schema: List[List[Union[int, str]]], label: Optional[str] = None):
        self.schema = schema
        self.label = label or "match_fixed_schema"

    def apply(self, state: Grid) -> Matches:
        g = np.asarray(state.grid, dtype=int)
        H, W = g.shape
        schema = self.schema
        nr, nc = len(schema), (len(schema[0]) if len(schema)>0 else 0)
        up, down = (nr-1)//2, nr//2
        left, right = (nc-1)//2, nc//2
        matches: List[dict] = []
        if nr == 0 or nc == 0:
            return Matches(g, matches)
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
        return Matches(g, matches)


class OpMatchAnyUniversalSchemas(Operation[Grid, Matches]):
    input_type = Grid
    output_type = Matches

    def __init__(self, schemas: List[List[List[Union[int, str]]]], label: Optional[str] = None):
        self.schemas = schemas
        self.label = label or "match_universal"

    def apply(self, state: Grid) -> Matches:
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
        return Matches(g, matches)
