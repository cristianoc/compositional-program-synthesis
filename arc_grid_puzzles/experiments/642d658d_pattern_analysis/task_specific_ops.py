# -----------------------------------------------------------------------------
# Task-Specific Operations for 642d658d
# These operations are specific to the patterns found in task 642d658d
# and should not be part of the general DSL.
# -----------------------------------------------------------------------------

from __future__ import annotations
from typing import List, Tuple
import numpy as np
from dsl_types.states import Operation, Grid, Center, Color, OpFailure

# Import the specific functions from the main DSL
from dsl_types.grid_to_center_to_color import (
    choose_center_cross_implied_33,
    out_mode_cross_for_center_33, 
    out_mode_flank_for_center
)

# Task-specific operations for 642d658d
class OpChooseCenterCrossImplied33(Operation[Grid, Center]):
    """Task-specific center selection for 642d658d cross patterns."""
    input_type = Grid
    output_type = Center
    label = "choose_cross_implied_33"

    def apply(self, state: Grid) -> Center:
        c = int(choose_center_cross_implied_33(state.grid))
        if c == 0:
            raise OpFailure("choose_cross_implied_33 failed: no center")
        return Center(state.grid, c)


class OpOutCrossModeForCenter33(Operation[Center, Color]):
    """Task-specific color output for 642d658d cross patterns."""
    input_type = Center
    output_type = Color
    label = "out_cross_mode_33"

    def apply(self, state: Center) -> Color:
        y = int(out_mode_cross_for_center_33(state.grid, state.center_color))
        if y == 0:
            raise OpFailure("out_cross_mode_33 failed: no color")
        return Color(y)


class OpOutFlankModeForCenter(Operation[Center, Color]):
    """Task-specific color output for 642d658d flank patterns."""
    input_type = Center
    output_type = Color
    label = "out_flank_mode"

    def apply(self, state: Center) -> Color:
        y = int(out_mode_flank_for_center(state.grid, state.center_color))
        if y == 0:
            raise OpFailure("out_flank_mode failed: no color")
        return Color(y)


# Task-specific operations list for 642d658d
G_TYPED_OPS_642D658D = [
    OpChooseCenterCrossImplied33(),
    OpOutCrossModeForCenter33(),
    OpOutFlankModeForCenter(),
]
