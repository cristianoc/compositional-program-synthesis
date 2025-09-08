# Compositional Program Synthesis for ARC Grid Puzzles

This repository provides a clean implementation of pattern-based program synthesis for ARC-style grid puzzles. It enumerates pattern matchers and simple heuristic aggregators to predict outputs.

## Quick Start

```bash
# Run the main experiment (self-discoverable, no args needed)
cd experiments
python run_642d658d.py
```

This will:
1. Load task data from `experiments/tasks/642d658d.json`
2. Enumerate universal pattern schemas across train+test
3. Build matcher→aggregator pipelines and evaluate them on train
4. Write results to `experiments/642d658d_pattern_analysis/` (including `programs_found.json` and mosaics)

## How It Works

### Pattern Detection and Universal Schemas
- Patterns are detected around a fixed center color (4) using sliding windows with shapes:
  - `(1,3), (3,1), (2,3), (3,3), (5,5)`
- For each shape and position, a universal schema is intersected across train+test examples.
- The most informative position is selected per shape via structural complexity.

### Program Search
A program is a short pipeline: `match_universal_pos(...) |> aggregator`

Heuristic aggregators currently used:
- `uniform_color_per_schema_then_mode`
- `uniform_color_from_matches_uniform_neighborhood` (restored)
- `uniform_from_matches_excl_global`

Programs must achieve 100% accuracy on training examples to be considered winners.

## Current Results (642d658d)

Examples of successful programs (see `experiments/642d658d_pattern_analysis/programs_found.json`):
- `match_universal_pos(shape=(1, 3),pos=(0, 1)) |> uniform_color_from_matches_uniform_neighborhood`
- `match_universal_pos(shape=(3, 1),pos=(1, 0)) |> uniform_color_from_matches_uniform_neighborhood`
- `match_universal_pos(shape=(2, 3),pos=(0, 1)) |> uniform_from_matches_excl_global`
- `match_universal_pos(shape=(3, 3),pos=(1, 1)) |> uniform_from_matches_excl_global`
