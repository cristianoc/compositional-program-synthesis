#!/usr/bin/env python3
"""
Test ARC-AGI samples including the specific 642d658d tasks.
"""

import os
import sys
import json
import time
import random
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from program_search import enumerate_programs_for_task


def load_task(task_path: str) -> Dict[str, Any]:
    """Load a task from JSON file."""
    with open(task_path, 'r') as f:
        return json.load(f)


def is_task_compatible(task: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Check if task is compatible with our single-color prediction system.
    
    Returns:
        (is_compatible, reason)
    """
    # Check all outputs in train and test sets
    all_examples = task.get('train', []) + task.get('test', [])
    
    for example in all_examples:
        output = example.get('output', [])
        
        # Output must be a single-cell containing one color value
        if not (isinstance(output, list) and 
                len(output) == 1 and 
                isinstance(output[0], list) and 
                len(output[0]) == 1 and 
                isinstance(output[0][0], int)):
            # Show the actual dimensions of the incompatible output
            if isinstance(output, list) and len(output) > 0 and isinstance(output[0], list):
                output_size = f"{len(output)}x{len(output[0])}"
            else:
                output_size = f"format: {type(output).__name__}"
            return False, f"Output size {output_size} (expected 1x1 single color)"
    
    return True, "Compatible: all outputs are single colors"


def is_task_solvable(task_path: str, max_time: float = 30.0) -> Tuple[bool, Dict[str, Any]]:
    """
    Test if a task can be solved with current operations.
    
    Returns:
        (is_solvable, result_info)
    """
    try:
        start_time = time.time()
        
        # Load the task
        task = load_task(task_path)
        
        # First check if task is compatible with our system
        is_compatible, compatibility_reason = is_task_compatible(task)
        
        if not is_compatible:
            elapsed_time = time.time() - start_time
            return False, {
                'elapsed_time': elapsed_time,
                'num_programs_found': 0,
                'programs': [],
                'total_operations_tested': 0,
                'success': False,
                'incompatible': True,
                'reason': compatibility_reason
            }
        
        # Run program search with timeout
        result = enumerate_programs_for_task(
            task, 
            seed=11,
            universal_shapes=[(1,3), (3,1), (3,3)]  # Use same shapes as 642d658d
        )
        
        elapsed_time = time.time() - start_time
        
        # Check if any programs were found
        abs_result = result.get('ABS', {})
        program_sequences = abs_result.get('program_sequences', [])
        programs = abs_result.get('programs', [])
        is_solvable = len(programs) > 0
        
        result_info = {
            'elapsed_time': elapsed_time,
            'num_programs_found': len(programs),
            'programs': programs[:3],  # First 3 programs
            'total_operations_tested': abs_result.get('nodes', 0),
            'success': is_solvable,
            'incompatible': False,
            'reason': compatibility_reason
        }
        
        return is_solvable, result_info
        
    except Exception as e:
        return False, {
            'error': str(e),
            'elapsed_time': time.time() - start_time,
            'success': False,
            'incompatible': False,
            'reason': 'Error during execution'
        }


def test_all_tasks() -> Dict[str, Any]:
    """Test ALL ARC-AGI tasks to get comprehensive compatibility statistics."""
    base_path = Path(__file__).parent.parent / "tasks"
    
    # Get all task files from both datasets
    all_test_tasks = []
    
    # ARC-AGI 1 tasks
    arc_agi_1_train = list((base_path / "arc_agi_1" / "training").glob("*.json"))
    arc_agi_1_eval = list((base_path / "arc_agi_1" / "evaluation").glob("*.json"))
    
    # ARC-AGI 2 tasks  
    arc_agi_2_train = list((base_path / "arc_agi_2" / "training").glob("*.json"))
    arc_agi_2_eval = list((base_path / "arc_agi_2" / "evaluation").glob("*.json"))
    
    all_test_tasks = arc_agi_1_train + arc_agi_1_eval + arc_agi_2_train + arc_agi_2_eval
    
    # Sort for consistent ordering
    all_test_tasks.sort()
    
    results = {
        'tasks_tested': 0,
        'solvable_tasks': [],
        'unsolvable_tasks': [],
        'summary': {}
    }
    
    print(f"Testing {len(all_test_tasks)} ARC-AGI tasks from all datasets")
    print("=" * 60)
    
    for i, task_path in enumerate(all_test_tasks, 1):
        if not task_path.exists():
            print(f"⚠️  [{i:2d}/{len(all_test_tasks)}] Task not found: {task_path.name}")
            continue
        
        # No required tasks when testing all tasks
        is_required = False
        status_mark = "📝"
        
        print(f"{status_mark} [{i:2d}/{len(all_test_tasks)}] Testing {task_path.name}...", end=' ')
        
        is_solvable, result_info = is_task_solvable(str(task_path))
        
        task_result = {
            'task_id': task_path.stem,
            'task_file': task_path.name,
            'dataset': f"{task_path.parent.parent.name}/{task_path.parent.name}",
            'is_required': is_required,
            **result_info
        }
        
        if is_solvable:
            results['solvable_tasks'].append(task_result)
            print(f"✓ SOLVED ({result_info['num_programs_found']} programs, {result_info['elapsed_time']:.2f}s)")
            if result_info.get('programs'):
                print(f"    Programs: {result_info['programs']}")
        else:
            results['unsolvable_tasks'].append(task_result)
            if result_info.get('incompatible', False):
                print(f"❌ INCOMPATIBLE ({result_info['elapsed_time']:.2f}s)")
                print(f"    Reason: {result_info.get('reason', 'Unknown')}")
            else:
                print(f"✗ UNSOLVED ({result_info['elapsed_time']:.2f}s)")
                if 'error' in result_info:
                    print(f"    Error: {result_info['error']}")
        
        results['tasks_tested'] += 1
    
    # Calculate summary statistics
    total_tested = results['tasks_tested']
    solvable_count = len(results['solvable_tasks'])
    incompatible_count = sum(1 for task in results['unsolvable_tasks'] if task.get('incompatible', False))
    compatible_tested = total_tested - incompatible_count
    
    results['summary'] = {
        'total_tested': total_tested,
        'solvable_count': solvable_count,
        'unsolvable_count': total_tested - solvable_count,
        'incompatible_count': incompatible_count,
        'compatible_tested': compatible_tested,
        'success_rate': solvable_count / total_tested if total_tested > 0 else 0,
        'success_rate_compatible_only': solvable_count / compatible_tested if compatible_tested > 0 else 0,
    }
    
    return results


def main():
    """Main experiment runner."""
    print("ARC-AGI Complete Task Experiment")
    print("Testing ALL tasks to determine compatibility with our system")
    print("=" * 60)
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Test all tasks
    results = test_all_tasks()
    
    # Print summary
    summary = results['summary']
    print(f"\n{'='*60}")
    print("SUMMARY")
    print("=" * 60)
    print(f"Tasks tested: {summary['total_tested']}")
    print(f"Compatible with our system: {summary['compatible_tested']}")
    print(f"Incompatible (wrong output format): {summary['incompatible_count']}")
    print(f"Solvable (among compatible): {summary['solvable_count']}")
    print(f"Unsolvable (among compatible): {summary['compatible_tested'] - summary['solvable_count']}")
    print(f"Overall success rate: {summary['success_rate']:.1%}")
    print(f"Success rate (compatible tasks only): {summary['success_rate_compatible_only']:.1%}")
    
    # Show solvable tasks
    if results['solvable_tasks']:
        print(f"\n✅ SOLVABLE TASKS:")
        for task in results['solvable_tasks']:
            required_mark = "🎯" if task['is_required'] else ""
            print(f"  {required_mark} {task['task_file']} ({task['dataset']}) - {task['num_programs_found']} programs")
    
    # Show breakdown by dataset
    print(f"\n📊 BREAKDOWN BY DATASET:")
    datasets = {}
    for task in results['solvable_tasks'] + results['unsolvable_tasks']:
        dataset = task['dataset']
        if dataset not in datasets:
            datasets[dataset] = {'total': 0, 'compatible': 0, 'solvable': 0}
        datasets[dataset]['total'] += 1
        if not task.get('incompatible', False):
            datasets[dataset]['compatible'] += 1
        if task['success']:
            datasets[dataset]['solvable'] += 1
    
    for dataset, stats in sorted(datasets.items()):
        compat_rate = stats['compatible'] / stats['total'] * 100 if stats['total'] > 0 else 0
        solve_rate = stats['solvable'] / stats['compatible'] * 100 if stats['compatible'] > 0 else 0
        print(f"  {dataset}: {stats['total']} tasks, {stats['compatible']} compatible ({compat_rate:.1f}%), {stats['solvable']} solved ({solve_rate:.1f}%)")
    
    # Save detailed results
    results_file = Path(__file__).parent / "arc_agi_all_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {results_file}")
    
    return results


if __name__ == "__main__":
    main()
