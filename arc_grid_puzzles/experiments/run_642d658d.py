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

sys.argv = ["run_experiment.py", str(task_path), "--output-dir", str(output_dir)]
from run_experiment import main
main()