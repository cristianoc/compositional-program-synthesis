#!/usr/bin/env python3
"""
ARC-AGI Abstraction Check over all tasks.
Builds universal schemas and reports whether the abstraction holds
(i.e., at least one admissible schema exists) across ARC-AGI 1 & 2.

Usage:
  python run_pattern_analysis_all.py [--dataset {all,arc_agi_1,arc_agi_2}] [--split {all,training,evaluation}] [--shapes 1x3 3x1 3x3]
"""

import sys
import json
import random
from pathlib import Path
from typing import Dict, List, Any

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments import driver as driver_mod



def _parse_shapes(shape_tokens: List[str] | None) -> List[tuple[int,int]] | None:
    if not shape_tokens:
        return None
    shapes: List[tuple[int,int]] = []
    for tok in shape_tokens:
        try:
            r, c = tok.lower().split("x", 1)
            shapes.append((int(r), int(c)))
        except Exception:
            raise SystemExit(f"Invalid shape token '{tok}'. Use e.g. 1x3 3x1 3x3")
    return shapes


def _filter_tasks(paths: List[Path], dataset: str, split: str) -> List[Path]:
    def keep(p: Path) -> bool:
        ds_ok = (dataset == "all") or (p.parent.parent.name == dataset)
        sp_ok = (split == "all") or (p.parent.name == split)
        return ds_ok and sp_ok
    return [p for p in paths if keep(p)]


def analyze_all_tasks(dataset: str = "all", split: str = "all", shapes: List[tuple[int,int]] | None = None, limit: int | None = None, task_id: str | None = None) -> Dict[str, Any]:
    """Analyze abstractions in ARC-AGI tasks and report whether they hold (no interestingness scoring)."""
    # Discover and filter tasks
    all_test_tasks = driver_mod.discover_all_task_paths()
    all_test_tasks = _filter_tasks(all_test_tasks, dataset, split)
    if task_id:
        all_test_tasks = [p for p in all_test_tasks if p.stem == task_id]
    if limit is not None:
        all_test_tasks = all_test_tasks[: max(0, int(limit))]
    
    results = {
        'tasks_analyzed': 0,
        'tasks_where_abstraction_holds': [],
        'tasks_where_abstraction_not_holds': [],
        'summary': {}
    }
    
    print(f"Analyzing patterns in {len(all_test_tasks)} ARC-AGI tasks (after filters)")
    print("=" * 60)
    
    for i, task_path in enumerate(all_test_tasks, 1):
        if not task_path.exists():
            print(f"⚠️  [{i:2d}/{len(all_test_tasks)}] Task not found: {task_path.name}")
            continue
        
        # No required tasks when testing all tasks
        is_required = False
        status_mark = "📝"
        
        print(f"{status_mark} [{i:2d}/{len(all_test_tasks)}] Analyzing {task_path.name}...", end=' ')
        
        if shapes is None:
            holds, result_info = driver_mod.analyze_task_patterns(str(task_path))
        else:
            # Load and analyze with explicit shapes override
            task = driver_mod.load_task(str(task_path))
            holds, result_info = driver_mod.find_task_abstractions(task, shapes=shapes)
        
        task_result = {
            'task_id': task_path.stem,
            'task_file': task_path.name,
            'dataset': f"{task_path.parent.parent.name}/{task_path.parent.name}",
            'is_required': is_required,
            **result_info
        }
        
        if holds:
            results['tasks_where_abstraction_holds'].append(task_result)
            print(f"✅ HOLDS ({len(result_info['admissible_schemas'])} admissible schemas, {result_info['elapsed_time']:.2f}s)")
        else:
            results['tasks_where_abstraction_not_holds'].append(task_result)
            if result_info.get('incompatible', False):
                print(f"❌ INCOMPATIBLE ({result_info['elapsed_time']:.2f}s)")
                print(f"    Reason: {result_info.get('reason', 'Unknown')}")
            else:
                # Stay silent for trivial patterns as requested
                print(f"✓ Analyzed ({result_info['elapsed_time']:.2f}s)")
        
        results['tasks_analyzed'] += 1
    
    # Calculate summary statistics
    total_analyzed = results['tasks_analyzed']
    holds_count = len(results['tasks_where_abstraction_holds'])
    incompatible_count = sum(1 for task in results['tasks_where_abstraction_not_holds'] if task.get('incompatible', False))
    compatible_analyzed = total_analyzed - incompatible_count
    
    results['summary'] = {
        'total_analyzed': total_analyzed,
        'holds_count': holds_count,
        'not_holds_count': total_analyzed - holds_count,
        'incompatible_count': incompatible_count,
        'compatible_analyzed': compatible_analyzed,
        'holds_rate': holds_count / total_analyzed if total_analyzed > 0 else 0,
        'holds_rate_compatible_only': holds_count / compatible_analyzed if compatible_analyzed > 0 else 0,
    }
    
    return results


def main():
    """Main experiment runner."""
    import argparse
    ap = argparse.ArgumentParser(description="ARC-AGI abstraction-holds analysis across tasks")
    ap.add_argument("--dataset", choices=["all", "arc_agi_1", "arc_agi_2"], default="all", help="Dataset to analyze")
    ap.add_argument("--split", choices=["all", "training", "evaluation"], default="all", help="Split to analyze")
    ap.add_argument("--shapes", nargs="*", help="Override shapes as tokens like 1x3 3x1 3x3")
    ap.add_argument("--task-id", help="Analyze only a single task by file stem (e.g., 642d658d)")
    ap.add_argument("--limit", type=int, help="Limit number of tasks after filtering")
    ap.add_argument("--full", action="store_true", help="Also write the full detailed results JSON (large)")
    args = ap.parse_args()

    shapes = _parse_shapes(args.shapes)

    print("ARC-AGI Pattern Analysis Experiment")
    target_desc = f"dataset={args.dataset}, split={args.split}"
    if shapes:
        target_desc += f", shapes={shapes}"
    if args.task_id:
        target_desc += f", task_id={args.task_id}"
    print(f"Analyzing patterns ({target_desc})")
    print("=" * 60)
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Analyze patterns in all tasks
    results = analyze_all_tasks(dataset=args.dataset, split=args.split, shapes=shapes, limit=args.limit, task_id=args.task_id)
    
    # Print summary
    summary = results['summary']
    print(f"\n{'='*60}")
    print("SUMMARY")
    print("=" * 60)
    print(f"Tasks analyzed: {summary['total_analyzed']}")
    print(f"Tasks with single-color outputs: {summary['compatible_analyzed']}")
    print(f"Tasks with multi-color outputs: {summary['incompatible_count']}")
    print(f"Tasks where abstraction holds: {summary['holds_count']}")
    print(f"Tasks where abstraction does not hold: {summary['not_holds_count']}")
    print(f"Overall holds rate: {summary['holds_rate']:.1%}")
    print(f"Holds rate (single-color tasks only): {summary['holds_rate_compatible_only']:.1%}")
    
    # Show tasks with interesting patterns
    if results['tasks_where_abstraction_holds']:
        print("\n🔍 TASKS WHERE ABSTRACTION HOLDS:")
        for task in results['tasks_where_abstraction_holds']:
            required_mark = "🎯" if task['is_required'] else ""
            print(f"  {required_mark} {task['task_file']} ({task['dataset']}) - {len(task['admissible_schemas'])} admissible schemas")
    
    # Show breakdown by dataset
    print("\n📊 BREAKDOWN BY DATASET:")
    datasets = {}
    for task in results['tasks_where_abstraction_holds'] + results['tasks_where_abstraction_not_holds']:
        dataset = task['dataset']
        if dataset not in datasets:
            datasets[dataset] = {'total': 0, 'compatible': 0, 'holds': 0}
        datasets[dataset]['total'] += 1
        if not task.get('incompatible', False):
            datasets[dataset]['compatible'] += 1
        if task.get('holds') or task.get('has_interesting_patterns') is True:
            datasets[dataset]['holds'] += 1
    
    for dataset, stats in sorted(datasets.items()):
        single_color_rate = stats['compatible'] / stats['total'] * 100 if stats['total'] > 0 else 0
        holds_rate = stats['holds'] / stats['total'] * 100 if stats['total'] > 0 else 0
        print(f"  {dataset}: {stats['total']} tasks, {stats['compatible']} single-color ({single_color_rate:.1f}%), {stats['holds']} holds ({holds_rate:.1f}%)")
    
    # Optionally save detailed results (large)
    if args.full:
        results_file = Path(__file__).parent / "arc_agi_pattern_analysis_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Detailed pattern analysis results saved to: {results_file}")
    else:
        print("\nℹ️ Skipping detailed results (use --full to write the large JSON)")

    # Save compact summary (overall + per-dataset) for version control
    summary_payload = {
        'summary': summary,
        'dataset_breakdown': datasets,
    }
    summary_file = Path(__file__).parent / "arc_agi_pattern_analysis_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary_payload, f, indent=2)
    print(f"💾 Summary saved to: {summary_file}")
    
    return results


if __name__ == "__main__":
    main()
