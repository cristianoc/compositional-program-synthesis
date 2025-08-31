# Scaling analysis experiment
# We vary:
#   - Lx: length of the X-program (number of x-only ops)
#   - K: number of cross-ops (add_first_to_second) inserted among the X ops
#   - b_first_size: size of the X-ops DSL (branching factor on X). We grow it by adding extra ops.
#
# Methods compared:
#   - Global BFS (full DSL)
#   - Compositional+ (synthesize X, then enumerate placements for K cross-ops)
#
# We report nodes expanded and wall time, and compute speedups.
#
# Notes:
# - We keep Y-only op `inc1_second` in the DSL but do not use it in targets.
# - Positions for cross-ops are deterministically spaced to avoid stochastic variance.
#
# Plots:
#   1) Nodes vs K for Lx=8, b_first_size=4 (hardest setting), both methods.
#   2) Geometric-mean speedup (Global/Comp) across Lx and b, by K.


import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, List, Tuple, Dict

import math
import pandas as pd
import matplotlib.pyplot as plt
# import caas_jupyter_tools as cj  # Removed - not available


State = Tuple[int, int]


@dataclass(frozen=True)
class Op:
    name: str
    func: Callable[[State], State]


# Base primitives
def inc1_first(s: State) -> State:
    x, y = s
    return (x + 1, y)


def inc2_first(s: State) -> State:
    x, y = s
    return (x + 2, y)


def double_first(s: State) -> State:
    x, y = s
    return (2 * x, y)


def triple_first(s: State) -> State:
    x, y = s
    return (3 * x, y)


def inc1_second(s: State) -> State:
    x, y = s
    return (x, y + 1)


def add_first_to_second(s: State) -> State:
    x, y = s
    return (x, y + x)


def make_dsl_first(b_first_size: int) -> List[Op]:
    # Grow X-ops set deterministically
    base = [Op("inc1_first", inc1_first), Op("double_first", double_first)]
    if b_first_size >= 3:
        base.append(Op("inc2_first", inc2_first))
    if b_first_size >= 4:
        base.append(Op("triple_first", triple_first))
    return base[:b_first_size]  # safety


def make_dsl_global(b_first_size: int) -> List[Op]:
    return make_dsl_first(b_first_size) + [
        Op("inc1_second", inc1_second),
        Op("add_first_to_second", add_first_to_second),
    ]


def run_program(prog: List[Op], s: State) -> State:
    for op in prog:
        s = op.func(s)
    return s


def signature_on_examples(prog: List[Op], xs: List[State]) -> Tuple[State, ...]:
    return tuple(run_program(prog, x) for x in xs)


@dataclass
class SynthesisResult:
    program: List[Op]
    time_sec: float
    nodes_expanded: int
    depth_found: int
    success: bool


def synthesize_global_bfs(spec, dsl: List[Op], max_depth: int) -> SynthesisResult:
    xs = [x for x, _ in spec]
    start = time.perf_counter()
    queue = deque([[]])
    nodes = 0
    seen_by_depth: Dict[int, set] = {0: {signature_on_examples([], xs)}}

    for depth in range(max_depth + 1):
        level_count = len(queue)
        for _ in range(level_count):
            prog = queue.popleft()
            nodes += 1
            if all(run_program(prog, x) == y for x, y in spec):
                end = time.perf_counter()
                return SynthesisResult(
                    program=prog,
                    time_sec=end - start,
                    nodes_expanded=nodes,
                    depth_found=len(prog),
                    success=True,
                )
            if len(prog) == max_depth:
                continue
            for op in dsl:
                new_prog = prog + [op]
                sig = signature_on_examples(new_prog, xs)
                d = len(new_prog)
                if d not in seen_by_depth:
                    seen_by_depth[d] = set()
                if sig in seen_by_depth[d]:
                    continue
                seen_by_depth[d].add(sig)
                queue.append(new_prog)
    end = time.perf_counter()
    return SynthesisResult(
        program=[],
        time_sec=end - start,
        nodes_expanded=nodes,
        depth_found=-1,
        success=False,
    )


# Build a target program with X-part of length Lx and K cross-ops
def build_target(Lx: int, K: int, dsl_first: List[Op]) -> List[Op]:
    # Deterministic X-part: r increments then d doubles where r+d=Lx and d is Lx//3 rounded, r = Lx-d
    # Keep to ops guaranteed available: inc1_first and double_first (always present)
    d = max(1, Lx // 3)
    r = Lx - d
    x_prog = [Op("inc1_first", inc1_first)] * r + [Op("double_first", double_first)] * d

    # Place K cross-ops at spaced positions across 0..Lx
    positions = [
        int(round((i + 1) * Lx / (K + 1))) for i in range(K)
    ]  # 1..Lx inclusive-ish
    # Convert to insert indices (after that many X-ops)
    positions = [min(Lx, max(0, p)) for p in positions]

    # Interleave
    program = []
    cursor = 0
    for i in range(Lx + K):
        # if we should place a cross-op now
        if positions and cursor == positions[0]:
            program.append(Op("add_first_to_second", add_first_to_second))
            positions.pop(0)
        else:
            # take next X op
            program.append(x_prog[cursor])
            cursor += 1
    return program


# Generate deterministic train/test inputs
train_inputs = [(-4, 0), (-1, 2), (0, 3), (1, -2), (2, -1), (3, 4), (5, -3), (-2, 5)]
test_inputs = [(-7, 1), (-5, -4), (4, 1), (7, -2)]


def make_spec(program: List[Op]):
    train_outputs = [run_program(program, s) for s in train_inputs]
    test_outputs = [run_program(program, s) for s in test_inputs]
    train_spec = list(zip(train_inputs, train_outputs))
    test_spec = list(zip(test_inputs, test_outputs))
    return train_spec, test_spec


# Compositional+ with K crosses: synthesize X, then enumerate all K placements among len(px)+1 slots
def compositional_plus(train_spec, K: int, dsl_first: List[Op], max_depth_first: int):
    # 1) synthesize X from projected spec
    proj_first = [(x[0], y[0]) for x, y in train_spec]
    # lift to pair space with y=0
    xs2 = [(x, 0) for x, _ in proj_first]
    ys2 = [(y, 0) for _, y in proj_first]
    pair_spec = list(zip(xs2, ys2))
    res_x = synthesize_global_bfs(pair_spec, dsl_first, max_depth=max_depth_first)
    nodes = res_x.nodes_expanded
    start = time.perf_counter()
    if not res_x.success:
        end = time.perf_counter()
        return SynthesisResult([], res_x.time_sec + (end - start), nodes, -1, False)

    px = res_x.program
    slots = len(px) + 1  # insertion slots

    # Enumerate multisets of size K over [0..slots-1] (combinations with repetition)
    # We generate in nondecreasing order so insertion yields correct sequencing.
    def combinations_with_replacement(n, k):
        # returns tuples of length k with values in 0..n-1 nondecreasing
        if k == 0:
            yield tuple()
            return

        # simple recursive generator
        def rec(start, rem, acc):
            if rem == 0:
                yield tuple(acc)
                return
            for v in range(start, n):
                acc.append(v)
                yield from rec(v, rem - 1, acc)
                acc.pop()

        yield from rec(0, k, [])

    best = None
    for combo in combinations_with_replacement(slots, K):
        # build candidate by inserting K cross-ops at the chosen slots
        cand = []
        at = {i: 0 for i in range(slots)}
        # Count how many crosses at each slot
        for t in combo:
            at[t] += 1
        # Interleave
        for i in range(slots):
            # insert crosses scheduled before taking px[i] (for i<slots-1)
            for _ in range(at[i]):
                cand.append(Op("add_first_to_second", add_first_to_second))
            if i < len(px):
                cand.append(px[i])
        nodes += 1
        if all(run_program(cand, x) == y for x, y in train_spec):
            best = cand
            end = time.perf_counter()
            return SynthesisResult(
                best, res_x.time_sec + (end - start), nodes, len(cand), True
            )
    end = time.perf_counter()
    return SynthesisResult([], res_x.time_sec + (end - start), nodes, -1, False)


# Run grid
Lx_values = [4, 6, 8]
K_values = [0, 1, 2, 3]
b_sizes = [2, 3, 4]

records = []

for Lx in Lx_values:
    for b in b_sizes:
        dsl_first = make_dsl_first(b)
        dsl_global = make_dsl_global(b)
        for K in K_values:
            target_prog = build_target(Lx, K, dsl_first)
            train_spec, test_spec = make_spec(target_prog)
            max_depth_global = len(target_prog)  # ground-truth depth
            # Allow first-synth to use slightly more than necessary
            max_depth_first = Lx

            # Global
            g = synthesize_global_bfs(
                train_spec, dsl_global, max_depth=max_depth_global
            )
            g_ok = g.success and all(
                run_program(g.program, x) == y for x, y in test_spec
            )

            # Compositional+
            c = compositional_plus(
                train_spec, K, dsl_first, max_depth_first=max_depth_first
            )
            c_ok = c.success and all(
                run_program(c.program, x) == y for x, y in test_spec
            )

            records.append(
                {
                    "Lx": Lx,
                    "K": K,
                    "b_first_size": b,
                    "target_len": len(target_prog),
                    "global_nodes": g.nodes_expanded,
                    "global_time": g.time_sec,
                    "global_found": g.success,
                    "global_ok_test": g_ok,
                    "comp_nodes": c.nodes_expanded,
                    "comp_time": c.time_sec,
                    "comp_found": c.success,
                    "comp_ok_test": c_ok,
                }
            )

df = pd.DataFrame.from_records(records)
print("\n=== Scaling Results (grid) ===")
print(df.to_string(index=False))

# Compute speedups (only where both succeed)
mask = df["global_found"] & df["comp_found"]
df_speed = df[mask].copy()
df_speed["nodes_speedup"] = df_speed["global_nodes"] / df_speed["comp_nodes"]
df_speed["time_speedup"] = df_speed["global_time"] / df_speed["comp_time"]


# Geometric mean speedup per K
def gmean(series):
    vals = [v for v in series if v > 0]
    if not vals:
        return float("nan")
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


gm_by_K = (
    df_speed.groupby("K")
    .agg(
        gmean_nodes_speedup=("nodes_speedup", gmean),
        gmean_time_speedup=("time_speedup", gmean),
        runs=("K", "count"),
    )
    .reset_index()
)

print("\n=== Geometric-mean speedups by K ===")
print(gm_by_K.to_string(index=False))

# Plot 1: Nodes vs K for hardest setting (Lx=8, b=4)
subset = df[(df["Lx"] == 8) & (df["b_first_size"] == 4)]
plt.figure()
plt.plot(subset["K"], subset["global_nodes"], label="Global BFS")
plt.plot(subset["K"], subset["comp_nodes"], label="Compositional+")
plt.title("Nodes vs K (Lx=8, b_first_size=4)")
plt.xlabel("K (number of cross-ops)")
plt.ylabel("nodes expanded")
plt.yscale("log")
plt.legend()
plt.tight_layout()
plt.savefig("nodes_vs_k_scaling.png", dpi=150, bbox_inches='tight')
plt.close()  # Close the figure to free memory
print("Plot saved: nodes_vs_k_scaling.png")

# Plot 2: Geometric-mean speedup by K
plt.figure()
plt.plot(gm_by_K["K"], gm_by_K["gmean_nodes_speedup"], label="Nodes speedup")
plt.plot(gm_by_K["K"], gm_by_K["gmean_time_speedup"], label="Time speedup")
plt.title("Geometric-mean speedup vs K (Global / Compositional+)")
plt.xlabel("K")
plt.ylabel("speedup (×)")
plt.yscale("log")
plt.legend()
plt.tight_layout()
plt.savefig("speedup_vs_k.png", dpi=150, bbox_inches='tight')
plt.close()  # Close the figure to free memory
print("Plot saved: speedup_vs_k.png")

# Print a concise textual summary
print("=== Summary ===")
print("Total runs:", len(df))
ok_both = df_speed.shape[0]
print("Both methods succeeded on:", ok_both, "runs")
print("\nWorst-case (hard subset Lx=8, b=4):")
for k in K_values:
    row = subset[subset["K"] == k].iloc[0]
    print(f"K={k}: Global nodes={row['global_nodes']}, Comp nodes={row['comp_nodes']}")

print("\nGeometric-mean speedups by K:")
for _, row in gm_by_K.iterrows():
    print(
        f"K={int(row['K'])}: nodes×≈{row['gmean_nodes_speedup']:.2f}, time×≈{row['gmean_time_speedup']:.2f} (n={int(row['runs'])})"
    )
