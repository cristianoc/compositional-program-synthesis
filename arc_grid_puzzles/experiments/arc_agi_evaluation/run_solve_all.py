#!/usr/bin/env python3
"""
Run solver across ARC-AGI tasks and save results to arc_agi_all_results.json.

Usage:
  python run_solve_all.py [--dataset {all,arc_agi_1,arc_agi_2}] [--split {all,training,evaluation}] [--shapes 1x3 3x1 3x3]
"""

import sys
import json
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments import driver as driver_mod


def _parse_shapes(shape_tokens):
    if not shape_tokens:
        return None
    shapes = []
    for tok in shape_tokens:
        try:
            r, c = tok.lower().split("x", 1)
            shapes.append((int(r), int(c)))
        except Exception:
            raise SystemExit(f"Invalid shape token '{tok}'. Use e.g. 1x3 3x1 3x3")
    return shapes


def _filter_tasks(paths, dataset: str, split: str):
    def keep(p: Path) -> bool:
        ds_ok = (dataset == "all") or (p.parent.parent.name == dataset)
        sp_ok = (split == "all") or (p.parent.name == split)
        return ds_ok and sp_ok
    return [p for p in paths if keep(p)]


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="ARC-AGI solver across tasks")
    ap.add_argument("--dataset", choices=["all", "arc_agi_1", "arc_agi_2"], default="all", help="Dataset to analyze")
    ap.add_argument("--split", choices=["all", "training", "evaluation"], default="all", help="Split to analyze")
    ap.add_argument("--shapes", nargs="*", help="Override shapes as tokens like 1x3 3x1 3x3")
    ap.add_argument("--task-id", help="Solve only a single task by file stem (e.g., 642d658d)")
    ap.add_argument("--limit", type=int, help="Limit number of tasks after filtering")
    args = ap.parse_args()

    shapes = _parse_shapes(args.shapes)

    print("ARC-AGI Solver Experiment")
    target_desc = f"dataset={args.dataset}, split={args.split}"
    if shapes:
        target_desc += f", shapes={shapes}"
    print(f"Running solver ({target_desc})")
    print("=" * 60)

    all_paths = driver_mod.discover_all_task_paths()
    sel_paths = _filter_tasks(all_paths, args.dataset, args.split)
    if args.task_id:
        sel_paths = [p for p in sel_paths if p.stem == args.task_id]
    if args.limit is not None:
        sel_paths = sel_paths[: max(0, int(args.limit))]

    results = {
        "tasks_tested": len(sel_paths),
        "solvable_tasks": [],
        "unsolvable_tasks": [],
    }

    for task_path in sel_paths:
        ok, info = driver_mod.is_task_solvable(task_path, universal_shapes=shapes or None)
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

    results.update(
        {
            "solvable_count": len(results["solvable_tasks"]),
            "unsolvable_count": len(results["unsolvable_tasks"]),
        }
    )

    out_path = Path(__file__).parent / "arc_agi_all_results.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSolved: {results.get('solvable_count', 0)} / {results.get('tasks_tested', 0)}")
    print(f"Unsolved: {results.get('unsolvable_count', 0)}")
    print(f"\n💾 Results saved to: {out_path}")


if __name__ == "__main__":
    main()
