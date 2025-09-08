# ARC Grid Puzzle Experiments

This directory contains experiments for ARC (Abstraction and Reasoning Corpus) grid puzzle tasks using compositional abstractions.

## Directory Structure

```
experiments/
├── tasks/                    # Dataset: ARC task definitions
│   └── 642d658d.json       # Task data (input/output examples)
├── 642d658d_pattern_analysis/  # Experiment results
│   ├── programs_found.json
│   ├── images/
│   └── README.md
└── run_642d658d.py         # Self-discoverable experiment runner
```

## Running Experiments

Each experiment has a self-discoverable runner script that can be called from anywhere:

```bash
# Run pattern analysis on task 642d658d (from any directory)
python experiments/run_642d658d.py

# Or from the experiments directory
cd experiments
python run_642d658d.py

# For a new experiment "my_task", create:
# python experiments/run_my_task.py
```

## Key Benefits

- **Separation of Concerns**: Tasks (data) vs Experiments (analysis)
- **Multiple Experiments**: Can run different analyses on the same task
- **Clean Organization**: No mixing of datasets with results
- **Self-Discoverable**: No command-line arguments needed

## Creating New Experiments

1. **Add task data**: Put `task.json` in `tasks/task_name.json`
2. **Create runner**: Copy `run_642d658d.py` to `run_task_name.py`
3. **Run experiment**: `python run_task_name.py`

The runner automatically:
- Finds the task in `tasks/task_name.json`
- Creates results in `task_name_pattern_analysis/`
- Generates all outputs (programs, stats, images)

## Available Experiments

- **642d658d** - Universal pattern matching with cross patterns
