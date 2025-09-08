#!/usr/bin/env python3
import sys
from pathlib import Path

# Get the directory where this script is located
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent

# Add project root to path
sys.path.insert(0, str(project_root))

# Get experiment name from script name
script_name = Path(__file__).name
experiment_name = script_name.replace('run_', '').replace('.py', '')

# Set up args and run with absolute paths
task_path = script_dir / "tasks" / f"{experiment_name}.json"
output_dir = script_dir / f"{experiment_name}_pattern_analysis"

# Add task-specific operations to the DSL for this experiment
experiment_dir = script_dir / f"{experiment_name}_pattern_analysis"
sys.path.insert(0, str(experiment_dir))

# Import task-specific operations
from task_specific_ops import G_TYPED_OPS_642D658D

# Patch the run_experiment module to use task-specific operations
import run_experiment
original_enumerate = run_experiment.enumerate_programs_for_task

def patched_enumerate_programs_for_task(task, **kwargs):
    kwargs['g_operations'] = G_TYPED_OPS_642D658D
    return original_enumerate(task, **kwargs)

run_experiment.enumerate_programs_for_task = patched_enumerate_programs_for_task

sys.argv = ["run_experiment.py", str(task_path), "--output-dir", str(output_dir)]
from run_experiment import main
main()