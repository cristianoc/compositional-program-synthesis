#!/usr/bin/env python3
"""
Common Utilities
Shared utility functions used across the codebase.
"""

from collections import Counter


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
