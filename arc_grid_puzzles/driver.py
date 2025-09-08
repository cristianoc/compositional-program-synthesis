# -----------------------------------------------------------------------------
# Generic Driver for Pattern Abstraction Experiments
# This module provides:
#  • Generic enumeration engine for typed operations
#  • Program composition and evaluation utilities
#  • Results collection and reporting
#  • Task-agnostic pattern abstraction framework
# -----------------------------------------------------------------------------

from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Callable, Type, Any
import numpy as np
import time
from dsl_types.states import Operation, State, Grid, Color, Pipeline
from dsl_types.grid_to_matches import (
    OpMatchAnyUniversalSchemas, 
    build_intersected_universal_schemas_for_task,
    select_best_pattern_position
)
from dsl_types.matches_to_color import (
    MATCHES_TO_COLOR_OPERATIONS,
    OpUniformColorPerSchemaThenMode,
    OpUniformColorFromMatchesExcludeGlobal
)
from dsl_types.grid_to_center_to_color import G_TYPED_OPS, COLOR_RULES_BASE


def _enumerate_typed_programs(
    task: Dict,
    ops: List[Operation],
    *,
    max_depth: int = 2,
    min_depth: int = 2,
    start_type: Type[State] = Grid,
    end_type: Type[State] = Color,
) -> List[Tuple[str, List[Operation]]]:
    # Build train pairs once
    train_pairs = [
        (np.array(ex["input"], dtype=int), int(ex["output"][0][0])) for ex in task["train"]
    ]

    def accepts_chain(seq: List[Operation]) -> bool:
        if not seq:
            return False
        if not (seq[0].input_type is start_type or issubclass(start_type, seq[0].input_type)):
            return False
        for a, b in zip(seq, seq[1:]):
            if not (b.input_type is a.output_type or issubclass(a.output_type, b.input_type)):
                return False
        return (seq[-1].output_type is end_type) or issubclass(seq[-1].output_type, end_type)

    def label_of(op: Operation) -> str:
        return getattr(op, "label", op.__class__.__name__)

    winners: List[Tuple[str, List[Operation]]] = []
    # Simple DFS over sequences up to max_depth
    from itertools import product
    # Expand sequences by chaining type-compatible ops
    def extend(seq: List[Operation]) -> List[List[Operation]]:
        outs: List[List[Operation]] = []
        last = seq[-1]
        for op in ops:
            if op.input_type is last.output_type or issubclass(last.output_type, op.input_type):
                outs.append(seq + [op])
        return outs

    # Start with ops that accept start_type
    seeds = [op for op in ops if op.input_type is start_type or issubclass(start_type, op.input_type)]
    frontier: List[List[Operation]] = [[op] for op in seeds]
    for depth in range(1, max_depth + 1):
        next_frontier: List[List[Operation]] = []
        for seq in frontier:
            if depth >= min_depth and accepts_chain(seq) and (seq[-1].output_type is end_type or issubclass(seq[-1].output_type, end_type)):
                # Evaluate on train
                ok = True
                for x, y in train_pairs:
                    state: State = Grid(x)
                    try:
                        for op in seq:
                            if not op.accepts(state):
                                raise TypeError(f"type mismatch: {type(state).__name__} -> {op.__class__.__name__}")
                            state = op.apply(state)  # type: ignore[arg-type]
                        assert isinstance(state, Color)
                        if int(state.color) != y:
                            ok = False
                            break
                    except Exception:
                        ok = False
                        break
                if ok:
                    name = " |> ".join(label_of(op) for op in seq)
                    winners.append((name, seq))
            if depth < max_depth:
                next_frontier.extend(extend(seq))
        frontier = next_frontier
    return winners


def enumerate_programs_for_task(
    task: Dict, 
    num_preops: int = 200, 
    seed: int = 11, 
    *, 
    universal_shapes: Optional[List[tuple[int,int]]] = None
) -> Dict[str, Any]:
    """Enumerate programs for a task using both G core and pattern abstraction approaches."""
    import time
    
    # G core via typed composition engine (choose -> out), but keep node count from COLOR_RULES for continuity.
    # G typed ops (choose -> out)
    t0 = time.perf_counter()
    winners_g = _enumerate_typed_programs(task, G_TYPED_OPS, max_depth=2, min_depth=2, start_type=Grid, end_type=Color)
    t1 = time.perf_counter()
    programs_G = []
    for name, _ in winners_g:
        # Present in legacy style for G composed programs
        if name.startswith("choose_"):
            parts = name.split(" |> ")
            if len(parts) == 2 and parts[1].startswith("out_"):
                programs_G.append(f"compose({parts[0]}->{parts[1]})")
            else:
                programs_G.append(name)
        else:
            programs_G.append(name)
    total_G = len(COLOR_RULES_BASE)

    # Abstractions: enumerate universal fixed-schema pipelines only (no overlay-based seeds)
    shapes: List[tuple[int,int]] = [(1,3), (3,1), (3,3)]
    abs_ops: List[Operation] = []
    
    # Add all operations from registry
    for op_class in MATCHES_TO_COLOR_OPERATIONS:
        abs_ops.append(op_class())
    
    # Add parameterized variants
    abs_ops.extend([
        OpUniformColorPerSchemaThenMode(cross_only=False),
        OpUniformColorFromMatchesExcludeGlobal(cross_only=True),
    ])
    # Add universal fixed-schema matchers derived from task (train+test) for requested shapes and center_value=4
    # Reuse the same default shapes as PatternOverlayExtractor when not overridden
    shapes_universal: List[tuple[int,int]] = list(universal_shapes) if universal_shapes is not None else list(shapes)
    matcher_seeds = 0
    for ushape in shapes_universal:
        try:
            uni_schemas = build_intersected_universal_schemas_for_task(task, window_shape=ushape, center_value=4, splits=("train","test"))
            if uni_schemas:
                # Select best pattern position based on structural complexity
                best_pos, best_schema = select_best_pattern_position(uni_schemas)
                abs_ops.append(OpMatchAnyUniversalSchemas([best_schema], label=f"match_universal_pos(shape={tuple(ushape)},pos={best_pos})"))
                matcher_seeds += 1
        except Exception:
            continue
    # Node count heuristic: number of matcher seeds
    total_ABS = matcher_seeds
    # Enumerate up to depth 3 to allow schema-matching chains
    t2 = time.perf_counter()
    winners_abs = _enumerate_typed_programs(task, abs_ops, max_depth=4, min_depth=2, start_type=Grid, end_type=Color)
    t3 = time.perf_counter()
    programs_ABS = []
    
    # Evaluate each found program on test cases
    test_pairs = [(np.array(ex["input"], dtype=int), int(ex["output"][0][0])) for ex in task["test"]]
    
    for name, seq in winners_abs:
        # Pretty-print sequences in a readable, parameterized style
        if len(seq) == 2 and isinstance(seq[0], OpMatchAnyUniversalSchemas) and isinstance(seq[1], tuple(MATCHES_TO_COLOR_OPERATIONS)):
            m0 = seq[0]
            agg = seq[1]
            aggname = agg.__class__.__name__
            
            # Add parameter info for parameterized aggregators to avoid duplicates
            if isinstance(agg, OpUniformColorFromMatchesExcludeGlobal) and getattr(agg, 'cross_only', False):
                aggname += "(cross_only=True)"
            elif isinstance(agg, OpUniformColorPerSchemaThenMode) and not getattr(agg, 'cross_only', True):
                aggname += "(cross_only=False)"
            
            prog_name = f"{getattr(m0,'label','match_fixed_schema')} |> {aggname}"
        else:
            # Fallback: use labels
            prog_name = name
        
        # Evaluate on test
        test_correct = 0
        test_preds = []
        for x, y in test_pairs:
            try:
                state: State = Grid(x)
                for op in seq:
                    state = op.apply(state)  # type: ignore[arg-type]
                assert isinstance(state, Color)
                pred = int(state.color)
                test_preds.append(pred)
                if pred == y:
                    test_correct += 1
            except Exception:
                test_preds.append(-1)  # Error marker
        
        test_status = "✓" if test_correct == len(test_pairs) else "✗"
        # Single programs list with test indicators (eliminates redundancy)
        programs_ABS.append(f"{prog_name} [{test_status} {test_correct}/{len(test_pairs)} test]")

    return {"G": {"nodes": total_G, "programs": programs_G, "programs_found": len(programs_G), "time_sec": (t1 - t0)},
            "ABS": {"nodes": total_ABS, "programs": sorted(set(programs_ABS)), "programs_found": len(winners_abs), "time_sec": (t3 - t2), "program_sequences": winners_abs}}


def print_programs_for_task(task: Dict, num_preops: int = 200, seed: int = 11):
    """Pretty-prints the programs and node counts."""
    res = enumerate_programs_for_task(task, num_preops=num_preops, seed=seed)
    print("=== Node counts ===")
    print(f"G core nodes: {res['G']['nodes']}")
    print(f"Overlay+predicate nodes: {res['ABS']['nodes']}")
    print("\n=== Programs found (G core) ===")
    if res["G"]["programs"]:
        for s in res["G"]["programs"]: print("-", s)
    else:
        print("(none)")
    print("\n=== Programs found (overlay abstraction + pattern check) ===")
    if res["ABS"]["programs"]:
        for s in res["ABS"]["programs"]: print("-", s)
    else:
        print("(none)")
    return res


def measure_spaces(task: Dict, num_preops: int = 200, seed: int = 11):
    """Measure the search spaces for both G and abstraction approaches."""
    import time
    train_pairs = [(np.array(ex["input"], dtype=int), int(ex["output"][0][0])) for ex in task["train"]]
    # G (no pre-ops)
    t0=time.perf_counter(); valid_G=[]; tried=0; tries_first=None; found=False
    for cn, cf in COLOR_RULES_BASE:
        tried+=1; ok=True
        for x,y in train_pairs:
            if int(cf(x)) != y: ok=False; break
        if ok:
            valid_G.append(cn)
            if not found: tries_first=tried; found=True
    t1=time.perf_counter()
    # ABS (typed enumeration including schema matching chains)
    t2=time.perf_counter()
    shapes: List[tuple[int,int]] = [(1,3), (3,1), (3,3)]
    abs_ops: List[Operation] = []
    
    # Add all operations from registry
    for op_class in MATCHES_TO_COLOR_OPERATIONS:
        abs_ops.append(op_class())
    
    # Add parameterized variants
    abs_ops.extend([
        OpUniformColorPerSchemaThenMode(cross_only=False),
        OpUniformColorFromMatchesExcludeGlobal(cross_only=True),
    ])
    for ushape in shapes:
        try:
            uni_schemas = build_intersected_universal_schemas_for_task(task, window_shape=ushape, center_value=4, splits=("train","test"))
            if uni_schemas:
                # Select best pattern position based on structural complexity
                best_pos, best_schema = select_best_pattern_position(uni_schemas)
                abs_ops.append(OpMatchAnyUniversalSchemas([best_schema], label=f"match_universal_pos(shape={tuple(ushape)},pos={best_pos})"))
        except Exception:
            pass
    winners_abs = _enumerate_typed_programs(task, abs_ops, max_depth=4, min_depth=2, start_type=Grid, end_type=Color)
    t3=time.perf_counter()
    nodes_abs = 3  # one matcher seed per shape (default shapes)
    return {
        "G":{"nodes": len(COLOR_RULES_BASE), "programs_found": len(valid_G),
             "tries_to_first": tries_first, "time_sec": t1-t0},
        "ABS":{"nodes": nodes_abs, "programs_found": len(winners_abs),
               "tries_to_first": None, "time_sec": t3-t2},
    }
