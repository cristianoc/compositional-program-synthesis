"""
Experiment driver utilities for ARC grid puzzles.

Provides shared functionality for:
- Task discovery and loading
- Compatibility checks for single-color outputs
- Pattern abstraction extraction and interestingness analysis
- Solver invocation using the program search engine

Intended to be imported by experiment scripts (e.g., arc_agi_evaluation).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple
import json
import time


# Import project-local modules (assumes project root on sys.path by caller)
from program_search import enumerate_programs_for_task, DEFAULT_UNIVERSAL_SHAPES
from dsl_types.grid_to_matches import (
    build_intersected_universal_schemas_for_task,
    select_best_pattern_position,
    schema_is_admissible_mixed,
    schema_constraints_summary,
    score_schema_structure,
)


# -----------------------------
# Task I/O and discovery
# -----------------------------

def load_task(task_path: str | Path) -> Dict[str, Any]:
    p = Path(task_path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def discover_all_task_paths(base_tasks_dir: str | Path | None = None) -> List[Path]:
    """Discover all ARC-AGI task JSONs in both datasets (training + evaluation)."""
    if base_tasks_dir is None:
        # experiments/driver.py -> project_root/ then tasks/
        base_tasks_dir = Path(__file__).parent.parent / "tasks"
    base = Path(base_tasks_dir)

    paths: List[Path] = []
    for dataset in ("arc_agi_1", "arc_agi_2"):
        for split in ("training", "evaluation"):
            d = base / dataset / split
            if d.exists():
                paths.extend(sorted(d.glob("*.json")))
    return paths


# -----------------------------
# Compatibility & Abstractions
# -----------------------------

def is_task_compatible(task: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Check if task is compatible with single-color output prediction (format [[N]]).
    Returns (is_compatible, reason).
    """
    all_examples = task.get("train", []) + task.get("test", [])
    for example in all_examples:
        output = example.get("output", [])
        if not (
            isinstance(output, list)
            and len(output) == 1
            and isinstance(output[0], list)
            and len(output[0]) == 1
            and isinstance(output[0][0], int)
        ):
            if isinstance(output, list) and output and isinstance(output[0], list):
                output_size = f"{len(output)}x{len(output[0])}"
            else:
                output_size = f"format: {type(output).__name__}"
            return False, f"Output size {output_size} (expected 1x1 single color)"
    return True, "Compatible: all outputs are single colors"




def find_task_abstractions(
    task: Dict[str, Any], *,
    shapes: List[tuple[int, int]] | None = None,
    colors_to_search: List[int] | None = None,
    splits: tuple[str, ...] = ("train", "test"),
) -> Tuple[bool, Dict[str, Any]]:
    """Extract best universal schemas per shape and center color.

    Returns (holds, result_info), where 'holds' is True iff at least one admissible schema is found.
    'admissible' currently means the schema has both constants and variables.
    """
    if shapes is None:
        shapes = list(DEFAULT_UNIVERSAL_SHAPES)
    if colors_to_search is None:
        colors_to_search = list(range(10))

    start_time = time.time()
    is_compatible, compatibility_reason = is_task_compatible(task)

    admissible_schemas: List[Dict[str, Any]] = []
    pattern_analysis: Dict[str, Any] = {}

    for shape in shapes:
        shape_admissible: List[Dict[str, Any]] = []
        shape_analysis: Dict[str, Any] = {}

        for center_color in colors_to_search:
            try:
                uni_schemas = build_intersected_universal_schemas_for_task(
                    task, window_shape=shape, center_value=center_color, splits=splits
                )
                if uni_schemas:
                    best_pos, best_schema = select_best_pattern_position(uni_schemas)
                    admissible = schema_is_admissible_mixed(best_schema)
                    struct_score = score_schema_structure(best_schema)

                    if admissible:
                        shape_admissible.append(
                            {
                                "center_color": center_color,
                                "shape": list(shape),
                                "position": list(best_pos),
                                "schema": best_schema,
                                "structure_score": struct_score,
                                "admissible": True,
                            }
                        )

                    shape_analysis[f"color_{center_color}"] = {
                        "universal_schemas": {str(k): v for k, v in uni_schemas.items()},
                        "best_position": list(best_pos),
                        "best_schema": best_schema,
                        "admissible_mixed": admissible,
                        "structure_score": struct_score,
                        "constraints": schema_constraints_summary(best_schema),
                    }
            except Exception as e:  # pragma: no cover - robust experiment wrapper
                shape_analysis[f"color_{center_color}"] = {"error": str(e), "is_interesting": False}

        admissible_schemas.extend(shape_admissible)
        pattern_analysis[f"shape_{shape}"] = shape_analysis

    elapsed_time = time.time() - start_time
    holds = len(admissible_schemas) > 0
    return holds, {
        "elapsed_time": elapsed_time,
        "admissible_schemas": admissible_schemas,
        "pattern_analysis": pattern_analysis,
        "holds": holds,
        "incompatible": not is_compatible,
        "reason": compatibility_reason,
    }


def analyze_task_patterns(task_path: str | Path) -> Tuple[bool, Dict[str, Any]]:
    """Analyze abstractions in a task file (wrapper around find_task_abstractions)."""
    task = load_task(task_path)
    return find_task_abstractions(task)


# -----------------------------
# Solver
# -----------------------------

def is_task_solvable(task_path: str | Path, *, universal_shapes: List[tuple[int, int]] | None = None) -> Tuple[bool, Dict[str, Any]]:
    """Check if a task can be solved with current ABS operations.

    Returns (is_solvable, result_info)
    """
    start_time = time.time()
    task = load_task(task_path)

    is_compatible, compatibility_reason = is_task_compatible(task)
    if not is_compatible:
        elapsed_time = time.time() - start_time
        return False, {
            "elapsed_time": elapsed_time,
            "num_programs_found": 0,
            "programs": [],
            "total_operations_tested": 0,
            "success": False,
            "incompatible": True,
            "reason": compatibility_reason,
        }

    result = enumerate_programs_for_task(
        task,
        seed=11,
        universal_shapes=universal_shapes or list(DEFAULT_UNIVERSAL_SHAPES),
    )
    elapsed_time = time.time() - start_time

    abs_result = result.get("ABS", {})
    programs = abs_result.get("programs", [])
    is_solvable = len(programs) > 0

    return is_solvable, {
        "elapsed_time": elapsed_time,
        "num_programs_found": len(programs),
        "programs": programs[:3],
        "total_operations_tested": abs_result.get("nodes", 0),
        "success": is_solvable,
        "incompatible": False,
        "reason": compatibility_reason,
    }


def run_solver_over_dataset() -> Dict[str, Any]:
    """Run solver across all tasks and produce an aggregated result structure.

    Matches the structure of arc_agi_all_results.json.
    """
    all_tasks = discover_all_task_paths()

    results: Dict[str, Any] = {
        "tasks_tested": len(all_tasks),
        "solvable_tasks": [],
        "unsolvable_tasks": [],
    }

    for task_path in all_tasks:
        ok, info = is_task_solvable(task_path)
        task_entry = {
            "task_id": task_path.stem,
            "task_file": task_path.name,
            "dataset": f"{task_path.parent.parent.name}/{task_path.parent.name}",
            "is_required": False,
            **info,
        }
        if ok:
            results["solvable_tasks"].append(task_entry)
        else:
            results["unsolvable_tasks"].append(task_entry)

    # Add summary counts similar to existing file
    results.update(
        {
            "solvable_count": len(results["solvable_tasks"]),
            "unsolvable_count": len(results["unsolvable_tasks"]),
        }
    )
    return results
