#!/usr/bin/env python3
"""
Simple top-level experiment runner for compositional program synthesis.

Usage:
    python run_experiment.py arc_grid_puzzles experiments/642d658d/task.json
    python run_experiment.py program_synthesis
"""

import sys
import subprocess
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_experiment.py <experiment_type> <args...>")
        print("Available experiments:")
        print("  arc_grid_puzzles <task_path>     - Run ARC grid puzzle experiment")
        print("  program_synthesis <args>         - Run program synthesis experiment")
        print()
        print("Examples:")
        print("  python run_experiment.py arc_grid_puzzles experiments/642d658d/task.json")
        print("  python run_experiment.py program_synthesis")
        sys.exit(1)
    
    experiment_type = sys.argv[1]
    args = sys.argv[2:]
    
    if experiment_type == "arc_grid_puzzles":
        if not args:
            print("Error: arc_grid_puzzles requires task path")
            print("Usage: python run_experiment.py arc_grid_puzzles <task_path>")
            sys.exit(1)
        
        task_path = args[0]
        script_path = Path(__file__).parent / "arc_grid_puzzles" / "run_experiment.py"
        
        if not script_path.exists():
            print(f"Error: Experiment script not found: {script_path}")
            sys.exit(1)
        
        # Run the ARC grid puzzle experiment
        cmd = [sys.executable, str(script_path), task_path] + args[1:]
        subprocess.run(cmd)
        
    elif experiment_type == "program_synthesis":
        script_path = Path(__file__).parent / "program_synthesis" / "scaling.py"
        
        if not script_path.exists():
            print(f"Error: Experiment script not found: {script_path}")
            sys.exit(1)
        
        # Run the program synthesis experiment
        cmd = [sys.executable, str(script_path)] + args
        subprocess.run(cmd)
        
    else:
        print(f"Error: Unknown experiment type: {experiment_type}")
        print("Available experiments: arc_grid_puzzles, program_synthesis")
        sys.exit(1)

if __name__ == "__main__":
    main()