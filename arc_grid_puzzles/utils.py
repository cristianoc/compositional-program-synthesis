#!/usr/bin/env python3
"""
Common Utilities
Shared utility functions used across the codebase.
"""

from collections import Counter
from typing import List, Tuple


def mode_int(values):
    """Find the mode (most frequent value) in a list of integers.
    
    Returns the smallest value if there's a tie.
    Returns 0 if the list is empty.
    """
    if not values: 
        return 0
    cnt = Counter(values)
    top = max(cnt.values())
    cands = [v for v, c in cnt.items() if c == top]
    return int(min(cands))


def cross_positions(nr: int, nc: int) -> List[Tuple[int, int]]:
    """Calculate cross positions for a given grid shape.
    
    Returns list of (row, col) tuples representing cross positions.
    """
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
