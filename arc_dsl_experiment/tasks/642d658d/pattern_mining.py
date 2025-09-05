"""ARG-AGI pattern mining for 1x3 windows.

This module extracts horizontal 1x3 windows from a grid and mines abstract
"schemas" (patterns with constants and equality-linked variables).

Default interestingness: schema has (a) at least one constant and (b) at least
one variable equality (same variable appearing at ≥2 positions). Example:
    [X, yellow, X]

The API allows overriding the interestingness predicate.

Notes
-----
• Variables are tokens 'X','Y','Z'. If your grid uses these as *constants*,
  set VAR_TOKENS to different strings and pass is_var accordingly.
• Grids are lists of lists (rows), with arbitrary hashable symbols (e.g. str).
• All results are canonicalized so variable names follow first occurrence order
  left→right (X then Y then Z).

High-level API
--------------
- mine_schemas(grid, interesting_fn=default_interesting) -> list[SchemaResult]
- intersect(results1, results2) -> list[CommonSchema]

Lower-level helpers are exposed for custom workflows.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Hashable, Iterable, List, Sequence, Tuple, Union, Optional
from collections import Counter
import math

# ---------------------------- Types & Constants -----------------------------
Symbol = Hashable
Schema = Tuple[Symbol, Symbol, Symbol]  # variables are strings 'X','Y','Z'
Window = Tuple[Symbol, Symbol, Symbol]

VAR_TOKENS: Tuple[str, str, str] = ("X", "Y", "Z")
VAR_SET = set(VAR_TOKENS)

# ------------------------------ Data Models --------------------------------
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

# ------------------------------ Core Utils ---------------------------------

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

# ------------------------- Schema Enumeration ------------------------------

# For a mask (subset of {0,1,2}), we enumerate valid partitions (blocks of
# equal-variable positions). We only allow a block if the triple has equal
# symbols at those positions, so the resulting schema is satisfied by the triple.

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
            [(i,), (j,)],   # different variables X, Y
            [(i, j)],       # same variable X
        ]
    if len(m) == 3:
        i, j, k = m
        return [
            [(i,), (j,), (k,)],  # X, Y, Z
            [(i, j), (k,)],      # X shared on i,j; Y on k
            [(i, k), (j,)],      # X shared on i,k; Y on j
            [(j, k), (i,)],      # X shared on j,k; Y on i
            [(i, j, k)],         # all share X
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

# -------------------------- Interestingness --------------------------------

InterestingFn = Callable[[Schema, Dict[str, Any]], bool]


def has_constant(schema: Schema) -> bool:
    return any(not is_var(x) for x in schema)


def has_equality(schema: Schema) -> bool:
    a, b, c = schema
    return (is_var(a) and a == b) or (is_var(b) and b == c) or (is_var(a) and a == c)


def default_interesting(schema: Schema, stats: Dict[str, Any]) -> bool:
    """Default: at least one constant AND at least one variable equality."""
    return has_constant(schema) and has_equality(schema)

# --------------------------- Mining Pipeline --------------------------------
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
    for S in sorted(set(m1.keys()) & set(m2.keys()), key=lambda S: tuple(str(x) for x in S)):
        r1, r2 = m1[S], m2[S]
        common.append(
            CommonSchema(S, r1.supp, r1.rsupp, r2.supp, r2.rsupp)
        )
    common.sort(key=lambda x: -(x.supp1 + x.supp2))
    return common

# ---------------------------- Convenience -----------------------------------
def format_schema(schema: Schema) -> str:
    """Stringify a schema like [X, yellow, X]."""
    return "[{}]".format(
        ", ".join(str(tok) for tok in schema)
    )


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

# ------------------------------ 3x3 Mining -----------------------------------
# We mine a single 3x3 signature over all full 3x3 windows centered on a given color.
# Rule: for each of the 9 positions, if all windows agree on the same constant value,
# keep that constant; otherwise emit the wildcard symbol "*" (meaning "anything").

SignatureCell = Union[int, str]
Signature3x3 = List[List[SignatureCell]]


def mine_3x3_signature(
    G: Sequence[Sequence[int]],
    *,
    center_color: int,
    require_full_window: bool = True,
) -> Optional[Signature3x3]:
    import numpy as np
    g = np.asarray(G, dtype=int)
    H, W = g.shape
    # collect full 3x3 windows centered on the specified color
    windows: List[np.ndarray] = []
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if int(g[r, c]) != int(center_color):
                continue
            win = g[r - 1 : r + 2, c - 1 : c + 2]
            if require_full_window and (win.shape != (3, 3)):
                continue
            windows.append(win.copy())
    if not windows:
        return None
    # build consensus with wildcard
    sig: Signature3x3 = [["*" for _ in range(3)] for _ in range(3)]
    for i in range(3):
        for j in range(3):
            vals = {int(w[i, j]) for w in windows}
            if len(vals) == 1:
                v = next(iter(vals))
                sig[i][j] = int(v)
            else:
                sig[i][j] = "*"
    return sig


def format_3x3_signature(sig: Signature3x3) -> str:
    # Renders like [[a,b,c],[d,e,f],[g,h,i]] with * as wildcard
    def cell(x: SignatureCell) -> str:
        return str(x)
    rows = ["[" + ", ".join(cell(x) for x in row) + "]" for row in sig]
    return "[" + ", ".join(rows) + "]"

def format_3x3_pretty(sig: Signature3x3) -> str:
    # Multi-line 2D rendering without commas for readability
    def cell(x: SignatureCell) -> str:
        return str(x)
    lines = ["[ " + " ".join(cell(x) for x in row) + " ]" for row in sig]
    return "\n" + "\n".join(lines)

# Extended: 3x3 schema with variables (X, Y, ...) for positions that are always equal
_VAR_TOKENS_3X3: Tuple[str, ...] = ("X", "Y", "Z", "U", "V", "W", "Q", "R", "S")


def mine_3x3_schema_with_vars(
    G: Sequence[Sequence[int]],
    *,
    center_color: int,
    require_full_window: bool = True,
) -> Optional[Signature3x3]:
    import numpy as np
    g = np.asarray(G, dtype=int)
    H, W = g.shape
    windows: List[np.ndarray] = []
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if int(g[r, c]) != int(center_color):
                continue
            win = g[r - 1 : r + 2, c - 1 : c + 2]
            if require_full_window and (win.shape != (3, 3)):
                continue
            windows.append(win.copy())
    if not windows:
        return None
    # Determine constants per position across windows
    pos_vals: List[set[int]] = []
    for i in range(3):
        for j in range(3):
            vals = {int(win[i, j]) for win in windows}
            pos_vals.append(vals)
    is_const = [len(s) == 1 for s in pos_vals]
    const_val = [next(iter(s)) if len(s) == 1 else None for s in pos_vals]

    # Equality relation across positions for non-constants: always equal across all windows
    npos = 9
    adj = [[False] * npos for _ in range(npos)]
    for a in range(npos):
        adj[a][a] = True
    def idx(i: int, j: int) -> int: return i * 3 + j
    # Check pairwise equality across all windows for non-constant positions
    for a in range(npos):
        if is_const[a]:
            continue
        ai, aj = divmod(a, 3)
        for b in range(a + 1, npos):
            if is_const[b]:
                continue
            bi, bj = divmod(b, 3)
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
    sig: Signature3x3 = [["*" for _ in range(3)] for _ in range(3)]
    # Fill constants
    for p in range(npos):
        if is_const[p]:
            i, j = divmod(p, 3)
            sig[i][j] = int(const_val[p])  # type: ignore[arg-type]
    # Map position to var token
    var_token_map: Dict[int, str] = {}
    next_var_idx = 0
    for comp in components:
        tok = _VAR_TOKENS_3X3[min(next_var_idx, len(_VAR_TOKENS_3X3) - 1)]
        next_var_idx += 1
        for p in comp:
            i, j = divmod(p, 3)
            var_token_map[p] = tok
            sig[i][j] = tok
    # Leave remaining non-constant singles as '*'
    return sig


def format_3x3_schema(sig: Signature3x3) -> str:
    return format_3x3_signature(sig)

def format_3x3_schema_pretty(sig: Signature3x3) -> str:
    return format_3x3_pretty(sig)
