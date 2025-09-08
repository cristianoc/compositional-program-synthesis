# ARC-AGI Dataset Evaluation

This directory contains experiments for evaluating the current compositional program synthesis system on the ARC-AGI datasets. It now supports both solver-oriented and abstraction-holds analysis, sharing functionality through a common driver.

## Files

- **`run_pattern_analysis_all.py`** – Abstraction-holds analysis over ALL tasks
  - Builds window-based universal schemas and reports whether the abstraction holds (i.e., admissible schemas exist) across ARC-AGI 1 & 2 tasks
  - Generates `arc_agi_pattern_analysis_results.json` with details (counts of admissible schemas and per-shape diagnostics)

- **`run_solve_all.py`** – Solver run over ALL tasks
  - Attempts to solve all tasks compatible with single-color outputs (`[[N]]`)
  - Aggregates results and saves to `arc_agi_all_results.json`

## Usage

From the project root directory:

Pattern analysis:
```bash
cd experiments/arc_agi_evaluation
python run_pattern_analysis_all.py --dataset all --split all --shapes 1x3 3x1 3x3 --limit 50
```

Solver run:
```bash
cd experiments/arc_agi_evaluation
python run_solve_all.py --dataset all --split all --shapes 1x3 3x1 3x3 --limit 50
```

## Flags and Defaults

- --dataset: all | arc_agi_1 | arc_agi_2 (default: all)
- --split: all | training | evaluation (default: all)
- --shapes: one or more shapes as tokens like 1x3 3x1 3x3
  - default: `program_search.DEFAULT_UNIVERSAL_SHAPES` (currently [(1,3), (3,1), (3,3)])
- --limit: integer, optional; limit number of tasks after filtering
- --task-id: optional; solve/analyze a single task by filename stem (e.g., 642d658d)

Outputs:
- Pattern analysis: `arc_agi_pattern_analysis_results.json`
- Solver results: `arc_agi_all_results.json`

## Task Compatibility

The solver is designed for tasks that output a single color value (format `[[N]]`). The solver runner automatically:
- Validates task compatibility before attempting program synthesis
- Reports incompatible tasks separately
- Calculates success rates only among compatible tasks

## Results Summary

Both runners save detailed JSON results in this directory for further analysis.

Shared logic (task discovery, compatibility, abstraction extraction, solver invocation) is implemented in `experiments/driver.py`.
