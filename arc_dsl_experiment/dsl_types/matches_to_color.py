# -----------------------------------------------------------------------------
# Matches to Color Operations
# This module implements operations that convert Matches to Color:
#  • Various strategies for extracting colors from pattern matches
#  • Cross-position filtering and neighborhood analysis
#  • Global color exclusion and schema-based color extraction
#
# OPERATIONS IN THIS FILE:
# 1. OpUniformColorFromMatches - Simple mode of matched colors
# 2. OpUniformColorPerSchemaThenMode - Mode per schema, then overall mode
# 3. OpUniformColorFromMatchesUniformNeighborhood - Mode of uniform neighborhoods
# 4. OpUniformColorFromSchemaConstantsOnly - Extract colors from schema constants
# 5. OpUniformColorFromMatchesExcludeGlobal - Mode excluding background colors
# -----------------------------------------------------------------------------

from __future__ import annotations
from typing import List, Dict, Optional
import numpy as np
from dsl_types.states import Operation, Matches, Color


def _mode_int(values):
    from collections import Counter
    if not values: return 0
    cnt = Counter(values); top = max(cnt.values())
    cands = [v for v,c in cnt.items() if c==top]
    return int(min(cands))


class OpUniformColorFromMatches(Operation[Matches, Color]):
    input_type = Matches
    output_type = Color

    label = "uniform_color_from_matches"

    def apply(self, state: Matches) -> Color:
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
            return Color(0)
        return Color(_mode_int(vals))


class OpUniformColorPerSchemaThenMode(Operation[Matches, Color]):
    input_type = Matches
    output_type = Color

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

    def apply(self, state: Matches) -> Color:
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
            return Color(0)
        # final mode across schema modes
        from collections import Counter
        c2 = Counter(schema_modes)
        top = max(c2.values())
        mode_vals = [k for k,v in c2.items() if v==top]
        return Color(int(min(mode_vals)))


class OpUniformColorFromMatchesUniformNeighborhood(Operation[Matches, Color]):
    input_type = Matches
    output_type = Color

    label = "uniform_color_from_matches_uniform_neighborhood"

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

    def apply(self, state: Matches) -> Color:
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
            return Color(0)
        c = Counter(picks)
        top = max(c.values())
        mode_vals = [k for k,v in c.items() if v==top]
        return Color(int(min(mode_vals)))


class OpUniformColorFromSchemaConstantsOnly(Operation[Matches, Color]):
    input_type = Matches
    output_type = Color

    label = "uniform_from_schema_constants"

    def apply(self, state: Matches) -> Color:
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
            return Color(0)
        return Color(_mode_int(vals))


class OpUniformColorFromMatchesExcludeGlobal(Operation[Matches, Color]):
    input_type = Matches
    output_type = Color

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

    def apply(self, state: Matches) -> Color:
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
            return Color(0)
        return Color(_mode_int(vals_out))


# =============================================================================
# OPERATION REGISTRY
# =============================================================================

# Registry of all Matches -> Color operations in this module
MATCHES_TO_COLOR_OPERATIONS = [
    OpUniformColorFromMatches,
    OpUniformColorPerSchemaThenMode,
    OpUniformColorFromMatchesUniformNeighborhood,
    OpUniformColorFromSchemaConstantsOnly,
    OpUniformColorFromMatchesExcludeGlobal,
]

# Validation: ensure registry matches actual operations
def _validate_registry():
    """Validate that the registry contains all Operation classes in this module."""
    import inspect
    import sys
    
    # Get all Operation classes defined in this module
    current_module = sys.modules[__name__]
    actual_operations = []
    for name, obj in inspect.getmembers(current_module):
        if (inspect.isclass(obj) and 
            hasattr(obj, '__bases__') and 
            any('Operation' in str(base) for base in obj.__bases__)):
            actual_operations.append(obj)
    
    # Check registry matches
    registry_set = set(MATCHES_TO_COLOR_OPERATIONS)
    actual_set = set(actual_operations)
    
    if registry_set != actual_set:
        missing_from_registry = actual_set - registry_set
        extra_in_registry = registry_set - actual_set
        raise ValueError(f"Registry mismatch! Missing: {missing_from_registry}, Extra: {extra_in_registry}")
    
    return True

# Validate on import
_validate_registry()
