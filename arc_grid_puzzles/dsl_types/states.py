# -----------------------------------------------------------------------------
# DSL State Classes
# This module defines the core state classes for the typed DSL:
#  • Grid: Raw grid data
#  • Matches: Pattern match results (see matches.py for details)
#  • Color: Predicted output color
#  • Center: Grid with identified center color
#  • Overlay: Legacy pattern-based state (unused)
#  • Operation framework and Pipeline
# -----------------------------------------------------------------------------

from __future__ import annotations
from typing import Dict, List, Type, TypeVar, Generic, Union, Any, Optional
import numpy as np


class State:  # marker base
    pass


class Grid(State):
    def __init__(self, grid: np.ndarray):
        self.grid = np.asarray(grid, dtype=int)


class Overlay(State):
    def __init__(self, grid: np.ndarray, overlays: List[dict], stats: Dict[str, float], kind: Optional[str] = None, color: Optional[int] = None):
        self.grid = np.asarray(grid, dtype=int)
        self.overlays = overlays
        self.stats = stats
        self.kind = kind
        self.color = int(color) if color is not None else None


class Color(State):
    def __init__(self, color: int):
        self.color = int(color)


class Center(State):
    def __init__(self, grid: np.ndarray, center_color: int):
        self.grid = np.asarray(grid, dtype=int)
        self.center_color = int(center_color)


class Matches(State):
    """
    State containing pattern match results.
    
    Matches are dictionaries with the following structure:
    {
        "y1": int, "x1": int, "y2": int, "x2": int,  # Match location (1-indexed)
        "match": List[List[Optional[int]]],           # Matched grid content
        "schema": List[List[Union[int, str]]],        # Pattern schema that matched
        # Additional metadata...
    }
    
    See matches.py for detailed documentation and utilities.
    """
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
