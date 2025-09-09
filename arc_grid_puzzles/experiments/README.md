# ARC Grid Puzzle Experiments

This directory contains experiments for ARC (Abstraction and Reasoning Corpus) grid puzzle tasks using compositional abstractions.

## Directory Structure

```
experiments/
├── 642d658d_pattern_analysis/  # Experiment results
│   ├── programs_found.json
│   ├── images/
│   └── README.md
└── run_642d658d.py         # Self-discoverable experiment runner (uses global ARC‑AGI task)
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

## Dataset-wide Evaluation

The `arc_agi_evaluation` suite provides dataset-wide runners:

- Abstraction-holds over all tasks:
  ```bash
  cd experiments/arc_agi_evaluation
  python run_pattern_analysis_all.py \
    --dataset all --split all \
    --shapes 1x3 3x1 3x3 \
    --limit 50
  ```
  Outputs: `arc_agi_pattern_analysis_results.json`

- Solver run over all tasks:
  ```bash
  cd experiments/arc_agi_evaluation
  python run_solve_all.py \
    --dataset arc_agi_2 --split evaluation \
    --shapes 1x3 3x1 3x3 \
    --limit 50
  ```
  Outputs: `arc_agi_all_results.json`

For full CLI options and defaults, see `experiments/arc_agi_evaluation/README.md`.
