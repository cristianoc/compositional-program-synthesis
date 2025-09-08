# -----------------------------------------------------------------------------
# Pattern Matching - Consolidated Module
# This module contains all pattern matching functionality:
#  • Pattern mining for 1x3 windows and n×n signatures
#  • Overlay detection and schema generation
#  • Match data structures and utilities
# -----------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    Iterable,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)
from collections import Counter
import numpy as np

# =============================================================================
# PATTERN MINING CORE
# =============================================================================

# Types & Constants
Symbol = Hashable
Schema = Tuple[Symbol, Symbol, Symbol]  # variables are strings 'X','Y','Z'
Window = Tuple[Symbol, Symbol, Symbol]

VAR_TOKENS: Tuple[str, str, str] = ("X", "Y", "Z")
VAR_SET = set(VAR_TOKENS)

# Extended variable tokens for n×n grids
VAR_TOKENS_GRID: Tuple[str, ...] = ("X", "Y", "Z", "U", "V", "W", "Q", "R", "S", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "T")

# Data Models
@dataclass(frozen=True)
class SchemaResult:
    schema: Schema
    supp: int
    rsupp: float

@dataclass(frozen=True)
class CommonSchema:
    schema: Schema
    supp1: int
    rsupp1: float
    supp2: int
    rsupp2: float

# Core Utils
def is_var(tok: Symbol) -> bool:
    return isinstance(tok, str) and tok in VAR_SET

def canonicalize(schema: Sequence[Symbol]) -> Schema:
    """Rename variables by first occurrence order -> X, Y, Z.
    Constants remain as-is.
    """
    mapping: Dict[str, str] = {}
    next_idx = 0
    out: List[Symbol] = []
    for a in schema:
        if is_var(a):
            # narrow type for type-checker: variables are strings
            assert isinstance(a, str)
            key: str = a
            if key not in mapping:
                mapping[key] = VAR_TOKENS[next_idx]
                next_idx += 1
            out.append(mapping[key])
        else:
            out.append(a)
    return (out[0], out[1], out[2])

def extract_windows(G: Sequence[Sequence[Symbol]]) -> List[Window]:
    W: List[Window] = []
    for row in G:
        if len(row) < 3:
            continue
        for c in range(len(row) - 2):
            W.append((row[c], row[c + 1], row[c + 2]))
    return W

# Schema Enumeration
def _blocks_for_mask(mask: Tuple[int, ...]) -> List[List[Tuple[int, ...]]]:
    """Enumerate partitions of positions in mask. Partitions are lists of blocks
    (each block is a tuple of positions). For size k ≤ 3 we hardcode structurally.
    Order of blocks determines variable names (X for first, etc.).
    """
    m = tuple(sorted(mask))
    if len(m) == 0:
        return [[()]]  # sentinel: no vars; caller ignores the empty block
    if len(m) == 1:
        return [[(m[0],)]]
    if len(m) == 2:
        i, j = m
        return [
            [(i,), (j,)],  # different variables X, Y
            [(i, j)],  # same variable X
        ]
    if len(m) == 3:
        i, j, k = m
        return [
            [(i,), (j,), (k,)],  # X, Y, Z
            [(i, j), (k,)],  # X shared on i,j; Y on k
            [(i, k), (j,)],  # X shared on i,k; Y on j
            [(j, k), (i,)],  # X shared on j,k; Y on i
            [(i, j, k)],  # all share X
        ]
    raise ValueError("Mask larger than 3 not supported for 1x3 windows.")

def gen_schemas_for_triple(t: Window) -> List[Schema]:
    """Generate all schemas that triple t SATISFIES.

    We iterate over masks of positions turned into variables, and over valid
    partitions of those positions where equality is consistent with t.
    """
    c = list(t)
    schemas: set[Schema] = set()
    positions = (0, 1, 2)

    # iterate over all masks (8 subsets)
    for m in range(0, 1 << 3):
        mask = tuple(i for i in positions if (m >> i) & 1)
        # enumerate partitions for this mask
        for blocks in _blocks_for_mask(mask):
            # verify equality constraints align with t
            ok = True
            for blk in blocks:
                if len(blk) <= 1:
                    continue
                base = c[blk[0]]
                for idx in blk[1:]:
                    if c[idx] != base:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue

            # build schema
            a: List[Symbol] = [None, None, None]
            # keep constants for positions not in mask
            for i in positions:
                if i not in mask:
                    a[i] = c[i]
            # assign variables for blocks
            for b_idx, blk in enumerate(blocks):
                if len(blk) == 0:
                    continue  # sentinel for empty mask
                var_tok = VAR_TOKENS[b_idx]
                for i in blk:
                    a[i] = var_tok
            schemas.add(canonicalize(a))

    # Use a deterministic key that compares mixed types via string form
    return sorted(schemas, key=lambda S: tuple(str(x) for x in S))

# Interestingness
InterestingFn = Callable[[Schema, Dict[str, Any]], bool]

def has_constant(schema: Schema) -> bool:
    return any(not is_var(x) for x in schema)

def has_equality(schema: Schema) -> bool:
    a, b, c = schema
    return (is_var(a) and a == b) or (is_var(b) and b == c) or (is_var(a) and a == c)

def default_interesting(schema: Schema, stats: Dict[str, Any]) -> bool:
    """Default: at least one constant AND at least one variable equality."""
    return has_constant(schema) and has_equality(schema)

# Mining Pipeline
def _structure_flags(schema: Schema) -> Dict[str, bool]:
    a, b, c = schema
    return {
        "palindrome": is_var(a) and a == c,
        "adj_left": is_var(a) and a == b,
        "adj_right": is_var(b) and b == c,
        "all_same": is_var(a) and a == b and b == c,
    }

def mine_schemas(
    G: Sequence[Sequence[Symbol]],
    interesting_fn: InterestingFn = default_interesting,
) -> List[SchemaResult]:
    """Mine schemas from grid G using the interestingness predicate.

    Returns list sorted by decreasing support, then lexicographically by schema.
    """
    windows = extract_windows(G)
    N = len(windows)
    if N == 0:
        return []

    counts: Counter[Schema] = Counter()

    for t in windows:
        for S in gen_schemas_for_triple(t):
            counts[S] += 1

    # build stats for predicate (can be expanded later)
    stats = {
        "N": N,
    }

    results: List[SchemaResult] = []
    for S, n in counts.items():
        if interesting_fn(S, stats):
            results.append(SchemaResult(S, n, n / N))

    # Stable sort by support desc, then schema lexicographically via string key
    results.sort(key=lambda r: (-r.supp, tuple(str(x) for x in r.schema)))
    return results

def intersect(
    R1: Iterable[SchemaResult], R2: Iterable[SchemaResult]
) -> List[CommonSchema]:
    """Intersection of mined schemas by exact identity (canonicalized).

    Returns list with supports/relative supports from both sides, sorted by the
    sum of supports descending.
    """
    m1 = {r.schema: r for r in R1}
    m2 = {r.schema: r for r in R2}
    common: List[CommonSchema] = []
    for S in sorted(
        set(m1.keys()) & set(m2.keys()), key=lambda S: tuple(str(x) for x in S)
    ):
        r1, r2 = m1[S], m2[S]
        common.append(CommonSchema(S, r1.supp, r1.rsupp, r2.supp, r2.rsupp))
    common.sort(key=lambda x: -(x.supp1 + x.supp2))
    return common

# Convenience
def format_schema(schema: Schema) -> str:
    """Stringify a schema like [X, yellow, X]."""
    return "[{}]".format(", ".join(str(tok) for tok in schema))

# =============================================================================
# N×N MINING
# =============================================================================

SignatureCell = Union[int, str]
SignatureGrid = List[List[SignatureCell]]

def mine_nxn_signature(
    G: Sequence[Sequence[int]],
    *,
    center_color: int,
    window_size: int,
    require_full_window: bool = True,
) -> Optional[List[List[SignatureCell]]]:
    if window_size % 2 == 0 or window_size < 1:
        raise ValueError("window_size must be odd and >= 1")
    g = np.asarray(G, dtype=int)
    H, W = g.shape
    r2 = window_size // 2
    windows: List[np.ndarray] = []
    for r in range(r2, H - r2):
        for c in range(r2, W - r2):
            if int(g[r, c]) != int(center_color):
                continue
            win = g[r - r2 : r + r2 + 1, c - r2 : c + r2 + 1]
            if require_full_window and (win.shape != (window_size, window_size)):
                continue
            windows.append(win.copy())
    if not windows:
        return None
    sig: List[List[SignatureCell]] = [
        ["*" for _ in range(window_size)] for _ in range(window_size)
    ]
    for i in range(window_size):
        for j in range(window_size):
            vals = {int(win[i, j]) for win in windows}
            if len(vals) == 1:
                sig[i][j] = int(next(iter(vals)))
    return sig

def format_nxn_signature(sig: List[List[SignatureCell]]) -> str:
    def cell(x: SignatureCell) -> str:
        return str(x)

    rows = ["[" + ", ".join(cell(x) for x in row) + "]" for row in sig]
    return "[" + ", ".join(rows) + "]"

def format_nxn_pretty(sig: List[List[SignatureCell]]) -> str:
    def cell(x: SignatureCell) -> str:
        return str(x)

    lines = ["[ " + " ".join(cell(x) for x in row) + " ]" for row in sig]
    return "\n" + "\n".join(lines)

def _var_token_for_index(idx: int) -> str:
    """Return a variable token for the given index, scaling beyond base tokens.

    Tokens cycle through VAR_TOKENS_GRID and add a numeric suffix when needed.
    Example: X, Y, Z, ..., T, X1, Y1, Z1, ...
    """
    if idx < 0:
        idx = 0
    base_len = len(VAR_TOKENS_GRID)
    suffix_num = idx // base_len
    base_tok = VAR_TOKENS_GRID[idx % base_len]
    return base_tok if suffix_num == 0 else f"{base_tok}{suffix_num}"

def mine_nxn_schema_with_vars(
    G: Sequence[Sequence[int]],
    *,
    center_color: int,
    window_size: int,
    require_full_window: bool = True,
) -> Optional[List[List[SignatureCell]]]:
    g = np.asarray(G, dtype=int)
    H, W = g.shape
    if window_size % 2 == 0 or window_size < 1:
        raise ValueError("window_size must be odd and >= 1")
    r2 = window_size // 2
    windows: List[np.ndarray] = []
    for r in range(r2, H - r2):
        for c in range(r2, W - r2):
            if int(g[r, c]) != int(center_color):
                continue
            win = g[r - r2 : r + r2 + 1, c - r2 : c + r2 + 1]
            if require_full_window and (win.shape != (window_size, window_size)):
                continue
            windows.append(win.copy())
    if not windows:
        return None
    # Determine constants per position across windows
    pos_vals: List[set[int]] = []
    for i in range(window_size):
        for j in range(window_size):
            vals = {int(win[i, j]) for win in windows}
            pos_vals.append(vals)
    is_const = [len(s) == 1 for s in pos_vals]
    const_val = [next(iter(s)) if len(s) == 1 else None for s in pos_vals]

    # Equality relation across positions for non-constants: always equal across all windows
    npos = window_size * window_size
    adj = [[False] * npos for _ in range(npos)]
    for a in range(npos):
        adj[a][a] = True

    def idx(i: int, j: int) -> int:
        return i * window_size + j

    # Check pairwise equality across all windows for non-constant positions
    for a in range(npos):
        if is_const[a]:
            continue
        ai, aj = divmod(a, window_size)
        for b in range(a + 1, npos):
            if is_const[b]:
                continue
            bi, bj = divmod(b, window_size)
            equal_all = True
            for win in windows:
                if int(win[ai, aj]) != int(win[bi, bj]):
                    equal_all = False
                    break
            if equal_all:
                adj[a][b] = adj[b][a] = True

    # Build components among non-constants using adjacency
    visited = [False] * npos
    components: List[List[int]] = []
    for v in range(npos):
        if visited[v] or is_const[v]:
            continue
        if not any(adj[v]):
            continue
        # BFS/DFS
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
            components.append(sorted(comp))

    # Assign variable tokens to components in scan order
    sig: List[List[SignatureCell]] = [
        ["*" for _ in range(window_size)] for _ in range(window_size)
    ]
    # Fill constants
    for p in range(npos):
        if is_const[p]:
            i, j = divmod(p, window_size)
            sig[i][j] = int(const_val[p])  # type: ignore[arg-type]
    # Map position to var token
    var_token_map: Dict[int, str] = {}
    next_var_idx = 0
    for comp in components:
        tok = _var_token_for_index(next_var_idx)
        next_var_idx += 1
        for p in comp:
            i, j = divmod(p, window_size)
            var_token_map[p] = tok
            sig[i][j] = tok
    # Leave remaining non-constant singles as '*'
    return sig

# =============================================================================
# OVERLAY DETECTION
# =============================================================================

PatternKind = Literal["window_nxm"]

def _to_np_grid(grid: Iterable[Iterable[int]]) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("grid must be 2-D")
    return g

def _emit_overlay(center_r: int, center_c: int, y1: int, x1: int, y2: int, x2: int, overlay_id: int) -> dict:
    h = y2 - y1 + 1
    w = x2 - x1 + 1
    return {
        "overlay_id": int(overlay_id),
        "center_row": int(center_r + 1),
        "center_col": int(center_c + 1),
        "y1": int(y1 + 1),
        "x1": int(x1 + 1),
        "y2": int(y2 + 1),
        "x2": int(x2 + 1),
        "height": int(h),
        "width": int(w),
        # For compatibility with vision overlays, we include numeric fields with neutral defaults
        "contrast": float(0.0),
        "peak_lum": float(0.0),
        "area": int(h * w),
    }

def detect_pattern_overlays(
    grid: Iterable[Iterable[int]],
    *,
    kind: PatternKind = "window_nxm",
    color: int,
    min_repeats: int = 2,
    dedup_centers: bool = True,
    window_shape: Optional[tuple[int, int]] = None,
) -> List[dict]:
    """
    Detect overlays by simple pattern templates.

    - kind="window_nxm": emit one overlay per center equal to the selected color with an (n×m) window clipped to the grid. Each overlay carries the window and its per-window schema.
    """
    g = _to_np_grid(grid)
    H, W = g.shape
    overlays: List[dict] = []
    overlay_id = 1

    if kind == "window_nxm":
        # One overlay per pixel equal to the given color,
        # with an n×m window centered at that pixel and clipped to the grid.
        if window_shape is None:
            hh, ww = 3, 3
        else:
            hh, ww = int(window_shape[0]), int(window_shape[1])
        if hh < 1 or ww < 1:
            raise ValueError("window_shape dims must be >= 1")
        up = (hh - 1) // 2
        down = hh // 2
        left = (ww - 1) // 2
        right = ww // 2

        VARS = (
            "X", "Y", "Z", "U", "V", "W", "Q", "R", "S", "A", "B", "C",
            "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "T"
        )
        def window_schema(win: np.ndarray) -> List[List[object]]:
            vals_to_positions: dict[int, List[tuple[int,int]]] = {}
            nr, nc = win.shape
            for i in range(nr):
                for j in range(nc):
                    v = int(win[i, j])
                    vals_to_positions.setdefault(v, []).append((i, j))
            schema: List[List[object]] = [["*" for _ in range(nc)] for _ in range(nr)]
            next_var = 0
            for v, poss in vals_to_positions.items():
                if len(poss) >= 2:
                    tok = VARS[min(next_var, len(VARS)-1)]
                    next_var += 1
                    for (i, j) in poss:
                        schema[i][j] = tok
                else:
                    (i, j) = poss[0]
                    # For colorless schemas, keep singletons as numeric constants to anchor matches
                    schema[i][j] = int(v)
            return schema

        for r in range(H):
            for c in range(W):
                if int(g[r, c]) != int(color):
                    continue
                y1 = max(0, r - up); x1 = max(0, c - left)
                y2 = min(H - 1, r + down); x2 = min(W - 1, c + right)
                ov = _emit_overlay(r, c, y1, x1, y2, x2, overlay_id)
                win = g[y1 : y2 + 1, x1 : x2 + 1]
                ov["window_h"] = int(win.shape[0])
                ov["window_w"] = int(win.shape[1])
                ov["window_shape"] = [int(win.shape[0]), int(win.shape[1])]
                ov["window"] = win.astype(int).tolist()
                schema = window_schema(win)
                # Ensure the center position within the window is marked as the constant selected color
                ci, cj = int(r - y1), int(c - x1)
                if 0 <= ci < win.shape[0] and 0 <= cj < win.shape[1]:
                    schema[ci][cj] = int(color)
                ov["schema"] = schema
                # Precompute a concrete matched window based on the schema (variables instantiated, '*' omitted)
                # This uses the window itself as the candidate.
                try:
                    from dsl import _schema_match_window  # type: ignore
                    mg = _schema_match_window(schema, win)
                except Exception:
                    mg = None
                if mg is not None:
                    ov["match"] = mg
                overlays.append(ov)
                overlay_id += 1
        return overlays
    else:
        raise ValueError(f"Unknown pattern kind: {kind}")

# =============================================================================
# MATCH DATA STRUCTURES
# =============================================================================

class Match:
    """
    A single pattern match result containing:
    - Location: (y1, x1, y2, x2) coordinates of the matched region
    - Content: The actual matched grid content
    - Schema: The pattern schema that matched
    - Metadata: Additional match information
    """
    
    def __init__(self, y1: int, x1: int, y2: int, x2: int, 
                 match: List[List[Optional[int]]], 
                 schema: List[List[Union[int, str]]],
                 **kwargs):
        self.y1 = int(y1)  # Top-left row (1-indexed)
        self.x1 = int(x1)  # Top-left column (1-indexed) 
        self.y2 = int(y2)  # Bottom-right row (1-indexed)
        self.x2 = int(x2)  # Bottom-right column (1-indexed)
        self.match = match  # The matched grid content
        self.schema = schema  # The pattern schema that matched
        self.metadata = kwargs  # Additional match metadata
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert match to dictionary format for compatibility."""
        return {
            "y1": self.y1,
            "x1": self.x1, 
            "y2": self.y2,
            "x2": self.x2,
            "match": self.match,
            "schema": self.schema,
            **self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Match':
        """Create match from dictionary format."""
        return cls(
            y1=data["y1"],
            x1=data["x1"],
            y2=data["y2"], 
            x2=data["x2"],
            match=data["match"],
            schema=data["schema"],
            **{k: v for k, v in data.items() if k not in ["y1", "x1", "y2", "x2", "match", "schema"]}
        )
    
    def __repr__(self) -> str:
        return f"Match(y1={self.y1}, x1={self.x1}, y2={self.y2}, x2={self.x2})"

class MatchesCollection:
    """
    A collection of pattern matches with utilities for:
    - Filtering matches by criteria
    - Extracting colors from matches
    - Validating match consistency
    """
    
    def __init__(self, matches: List[Match]):
        self.matches = list(matches)
    
    def filter_by_schema(self, schema: List[List[Union[int, str]]]) -> 'MatchesCollection':
        """Filter matches that used a specific schema."""
        filtered = [m for m in self.matches if m.schema == schema]
        return MatchesCollection(filtered)
    
    def filter_by_position(self, y1: int, x1: int, y2: int, x2: int) -> 'MatchesCollection':
        """Filter matches within a specific region."""
        filtered = [m for m in self.matches 
                   if m.y1 >= y1 and m.x1 >= x1 and m.y2 <= y2 and m.x2 <= x2]
        return MatchesCollection(filtered)
    
    def extract_colors(self, exclude_wildcards: bool = True) -> List[int]:
        """Extract all colors from matches, optionally excluding wildcard positions."""
        colors = []
        for match in self.matches:
            for row in match.match:
                for cell in row:
                    if cell is not None:
                        if not exclude_wildcards or not self._is_wildcard_position(match.schema, row, cell):
                            colors.append(int(cell))
        return colors
    
    def _is_wildcard_position(self, schema: List[List[Union[int, str]]], row_idx: int, cell_idx: int) -> bool:
        """Check if a position in the schema is a wildcard."""
        if row_idx < len(schema) and cell_idx < len(schema[row_idx]):
            return schema[row_idx][cell_idx] == "*"
        return False
    
    def __len__(self) -> int:
        return len(self.matches)
    
    def __iter__(self):
        return iter(self.matches)
    
    def __getitem__(self, idx):
        return self.matches[idx]

# Match format documentation
MATCH_FORMAT_DOC = """
Match Dictionary Format:
{
    "y1": int,      # Top-left row (1-indexed)
    "x1": int,      # Top-left column (1-indexed)
    "y2": int,      # Bottom-right row (1-indexed) 
    "x2": int,      # Bottom-right column (1-indexed)
    "match": List[List[Optional[int]]],  # Matched grid content
    "schema": List[List[Union[int, str]]],  # Pattern schema that matched
    # Additional metadata fields...
}

Example Match:
{
    "y1": 2, "x1": 3, "y2": 4, "x2": 5,
    "match": [[1, 2, 3], [2, 4, 2], [3, 2, 1]],
    "schema": [["X", 4, "X"], ["*", "X", "*"], ["Y", "*", "Y"]]
}

This represents a 3x3 match at position (2,3) to (4,5) where:
- Position (0,1) must be color 4 (constant)
- Positions (0,0) and (0,2) must be the same color (variable X)
- Positions (2,0) and (2,2) must be the same color (variable Y)  
- Positions marked "*" can be any color
"""

# =============================================================================
# DEMO
# =============================================================================

def demo() -> None:
    """Tiny demo."""
    G1 = [
        ["red", "yellow", "red", "blue"],
        ["blue", "yellow", "blue", "green"],
    ]
    G2 = [
        ["red", "yellow", "red"],
        ["red", "yellow", "red"],
    ]
    r1 = mine_schemas(G1)
    r2 = mine_schemas(G2)
    print("G1 interesting schemas:")
    for r in r1:
        print(r.supp, f"{format_schema(r.schema)}", f"rsupp={r.rsupp:.2f}")
    print("\nG2 interesting schemas:")
    for r in r2:
        print(r.supp, f"{format_schema(r.schema)}", f"rsupp={r.rsupp:.2f}")
    print("\nCommon:")
    for c in intersect(r1, r2):
        print(
            c.supp1 + c.supp2,
            f"{format_schema(c.schema)}",
            f"(G1 {c.supp1}/{c.rsupp1:.2f}, G2 {c.supp2}/{c.rsupp2:.2f})",
        )

if __name__ == "__main__":
    demo()
