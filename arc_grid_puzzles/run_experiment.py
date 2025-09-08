# -----------------------------------------------------------------------------
# Generic Experiment Runner for Pattern Abstraction
# This script:
#  • Loads any task.json and enumerates program spaces (G vs. Abstraction)
#  • Prints programs found & node counts/timings
#  • Renders annotated images (universal matches + predicted color)
# Usage: python run_experiment.py <task_path>
# Example: python run_experiment.py experiments/642d658d/task.json
# -----------------------------------------------------------------------------

import json, numpy as np, os, argparse
from typing import Optional, Any, Union
from importlib import reload
import sys
from pathlib import Path

# Import from the reorganized structure
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from program_search import enumerate_programs_for_task
from dsl_types.states import Grid, Pipeline
from dsl_types.grid_to_matches import OpMatchAnyUniversalSchemas


def main():
    parser = argparse.ArgumentParser(description='Run pattern abstraction experiment on ARC task')
    parser.add_argument('task_path', help='Path to task.json file')
    parser.add_argument('--output-dir', help='Output directory for results (default: same as task directory)')
    args = parser.parse_args()
    
    task_path = Path(args.task_path)
    if not task_path.exists():
        print(f"Error: Task file not found: {task_path}")
        sys.exit(1)
    
    task = json.loads(task_path.read_text())
    train_pairs = [(np.array(ex["input"], dtype=int), int(ex["output"][0][0])) for ex in task["train"]]

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = task_path.parent
    
    output_dir.mkdir(parents=True, exist_ok=True)

    # Enumerate abstraction operations
    SHAPES = [(1,3),(3,1),(2,3),(3,3),(5,5)]
    res_once = enumerate_programs_for_task(task, num_preops=200, seed=11, universal_shapes=SHAPES)
    # Print and persist simple combined JSON
    programs_path = output_dir / "programs_found.json"
    try:
        # Get node counts from the enumeration result
        abs_nodes = res_once['ABS']['nodes']
        g_nodes = res_once['G']['nodes']
        abs_time = res_once['ABS']['time']
        g_time = res_once['G']['time']
        
        # Create combined result
        combined_result = {
            "ABS": {
                "nodes": abs_nodes,
                "time": abs_time,
                "program_sequences": res_once["ABS"].get("program_sequences", [])
            },
            "G": {
                "nodes": g_nodes,
                "time": g_time,
                "program_sequences": res_once["G"].get("program_sequences", [])
            }
        }
        
        programs_path.write_text(json.dumps(combined_result, indent=2))
        print(f"Wrote {programs_path}")
        
        # Print summary
        print(f"\n=== Program Search Results ===")
        print(f"Abstraction operations: {abs_nodes} nodes explored in {abs_time:.2f}s")
        print(f"G operations: {g_nodes} nodes explored in {g_time:.2f}s")
        
        # Print found programs
        abs_programs = res_once["ABS"].get("program_sequences", [])
        g_programs = res_once["G"].get("program_sequences", [])
        
        if abs_programs:
            print(f"\nFound {len(abs_programs)} abstraction programs:")
            for name, seq in abs_programs:
                print(f"  {name}: {[op.__class__.__name__ for op in seq]}")
        
        if g_programs:
            print(f"\nFound {len(g_programs)} G programs:")
            for name, seq in g_programs:
                print(f"  {name}: {[op.__class__.__name__ for op in seq]}")
                
    except Exception as e:
        print(f"Error processing results: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()