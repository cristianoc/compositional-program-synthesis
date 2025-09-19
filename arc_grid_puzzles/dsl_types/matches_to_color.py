# -----------------------------------------------------------------------------
# Matches to Color Operations
# This module contains operations that convert Matches to Color.
# Only the operations actually used by the experiment are included.
# -----------------------------------------------------------------------------

from __future__ import annotations
from typing import List, Dict, Optional, Union
import numpy as np
from dsl_types.states import Operation, Matches, Color
from utils import mode_int, cross_positions


class OpUniformColorPerSchemaThenMode(Operation[Matches, Color]):
    input_type = Matches
    output_type = Color

    label = "uniform_color_per_schema_then_mode"

    def __init__(self, cross_only: bool = False):
        self.cross_only = cross_only

    def apply(self, state: Matches) -> Color:
        # For each match, compute the mode of non-zero values in its schema
        schema_modes: List[int] = []
        for m in state.matches:
            schema = m.get("schema", [])
            vals: List[int] = []
            for row in schema:
                for cell in row:
                    if isinstance(cell, int) and cell != 0:
                        vals.append(cell)
            if vals:
                schema_modes.append(mode_int(vals))
        
        if not schema_modes:
            return Color(0)
        
        # Return the mode of schema modes
        from collections import Counter
        c2 = Counter(schema_modes)
        top = max(c2.values())
        mode_vals = [k for k,v in c2.items() if v==top]
        return Color(int(min(mode_vals)))


class OpUniformColorFromMatchesUniformNeighborhood(Operation[Matches, Color]):
    input_type = Matches
    output_type = Color

    label = "uniform_color_from_matches_uniform_neighborhood"

    def _neighborhood_positions(self, nr: int, nc: int) -> List[tuple[int, int]]:
        # Mirror UniformPatternPredicate center-neighborhood semantics by shape parity
        if nr == 1 and nc == 3:
            return [(0, 0), (0, 2)]
        if nr == 3 and nc == 1:
            return [(0, 0), (2, 0)]
        if nr % 2 == 1 and nc % 2 == 1:
            ci, cj = nr // 2, nc // 2
            return [(ci - 1, cj), (ci + 1, cj), (ci, cj - 1), (ci, cj + 1)]
        if nr % 2 == 0 and nc % 2 == 0:
            i0, i1 = nr // 2 - 1, nr // 2
            j0, j1 = nc // 2 - 1, nc // 2
            return [(i0, j0), (i0, j1), (i1, j0), (i1, j1)]
        if nr % 2 == 1 and nc % 2 == 0:
            ci = nr // 2
            j0, j1 = nc // 2 - 1, nc // 2
            return [(ci, j0), (ci, j1)]
        if nr % 2 == 0 and nc % 2 == 1:
            cj = nc // 2
            i0, i1 = nr // 2 - 1, nr // 2
            return [(i0, cj), (i1, cj)]
        return []

    def apply(self, state: Matches) -> Color:
        from collections import Counter
        picks: List[int] = []
        for m in state.matches:
            mg = m.get("match")
            if mg is None:
                continue
            nr = len(mg)
            nc = len(mg[0]) if nr > 0 else 0
            if nr == 0 or nc == 0:
                continue
            poss = self._neighborhood_positions(nr, nc)
            if not poss:
                continue
            vals: List[int] = []
            for (i, j) in poss:
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
        mode_vals = [k for k, v in c.items() if v == top]
        return Color(int(min(mode_vals)))


class OpUniformColorFromMatchesExcludeGlobal(Operation[Matches, Color]):
    input_type = Matches
    output_type = Color

    label = "uniform_from_matches_excl_global"

    def __init__(self, cross_only: bool = False):
        self.cross_only = cross_only

    def apply(self, state: Matches) -> Color:
        # Extract colors from match positions, excluding global/wildcard positions
        vals: List[int] = []
        for m in state.matches:
            schema = m.get("schema", [])
            match_data = m.get("match", [])
            
            if not schema or not match_data:
                continue
                
            # Find positions that are not wildcards in the schema
            for i, row in enumerate(schema):
                for j, cell in enumerate(row):
                    if not self._is_wildcard_position(schema, i, j):
                        if (i < len(match_data) and 
                            j < len(match_data[i]) and 
                            match_data[i][j] is not None):
                            val = int(match_data[i][j])
                            if val != 0:
                                vals.append(val)
        
        if not vals:
            return Color(0)
        return Color(mode_int(vals))

    def _is_wildcard_position(self, schema: List[List[Union[int, str]]], row_idx: int, cell_idx: int) -> bool:
        """Check if a position in the schema is a wildcard ('*')."""
        if row_idx < len(schema) and cell_idx < len(schema[row_idx]):
            return schema[row_idx][cell_idx] == "*"
        return True


# =============================================================================
# OPERATION REGISTRY
# =============================================================================

# Registry of all heuristic Matches -> Color operations in this module
HEURISTIC_MATCHES_TO_COLOR_OPERATIONS = [
    OpUniformColorPerSchemaThenMode,
    OpUniformColorFromMatchesUniformNeighborhood,
    OpUniformColorFromMatchesExcludeGlobal,
]

# Validation: ensure registry matches actual operations
def _validate_registry():
    """Validate that the registry contains all Operation classes in this module."""
    import inspect
    import sys
    
    # Get all Operation classes defined in this module
    current_module = sys.modules[__name__]
    operation_classes = []
    for name, obj in inspect.getmembers(current_module):
        if (inspect.isclass(obj) and 
            issubclass(obj, Operation) and 
            obj != Operation and
            obj.__module__ == __name__):
            operation_classes.append(obj)
    
    # Check that registry contains exactly these classes
    registry_set = set(HEURISTIC_MATCHES_TO_COLOR_OPERATIONS)
    defined_set = set(operation_classes)
    
    if registry_set != defined_set:
        missing_in_registry = defined_set - registry_set
        extra_in_registry = registry_set - defined_set
        raise ValueError(f"Registry mismatch: missing={missing_in_registry}, extra={extra_in_registry}")
    
    return True

# Validate on import
_validate_registry()