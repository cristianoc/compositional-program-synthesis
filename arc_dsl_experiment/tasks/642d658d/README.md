# ARC Task 642d658d (Pattern-only)

## Task at a glance

| Train 0 Input | Train 0 Output |
|---|---|
| ![](images/train_0_in.png) | ![](images/train_0_out.png) |

| Train 1 Input | Train 1 Output |
|---|---|
| ![](images/train_1_in.png) | ![](images/train_1_out.png) |

| Train 2 Input | Train 2 Output |
|---|---|
| ![](images/train_2_in.png) | ![](images/train_2_out.png) |

| Test 0 Input |
|---|
| ![](images/test_0_in.png) |

## Universal matcher mosaic
- Mosaic view with all train and test examples across universal matcher shapes:
  - Columns: shapes with found programs (1×3, 3×1, 3×3, 5×5)
  - Each panel shows the input grid with yellow rectangles where `match_universal_pos(shape=...)` finds matches that are consistent with the used aggregator; the right panel shows the predicted color.

![](images/overlay_mosaic.png)

## Abstract
We use universal fixed-schema pipelines: for each shape, we compute intersected universal schemas from train+test and match them everywhere with a single op, then aggregate matched values to predict the color. The visualization shows only matches that would be retained by the aggregator used for each shape, avoiding pre-aggregation hits that get discarded later.

## 1. Methods (universal-only)

- `match_universal_pos(shape=(h,w))` matches all intersected universal schemas for the given shape across all positions (single op; no per-position parameters).
- Aggregators convert matches to a color: mode, per-schema mode, exclude-global, and uniform-neighborhood.

## 2. Enumeration spaces

- G core: composed color rules only (no pre-ops). Nodes = number of composed rules (currently 70; see repro output).
- Abstraction: universal schema matchers. Nodes = number of matcher seeds (see repro output).
- Single-pass enumeration: ABS is enumerated once and includes universal fixed-schema matchers (one per shape in SHAPES), labeled `match_universal_pos(shape=(h,w))`. Each universal matcher tries all positions for that shape (no per-position parameters). Aggregators then map matches to a color.

## 2.1 Operations and Types

G ops (core composition)

| Operation (label) | Input type | Output type | Description |
|---|---|---|---|
| `choose_cross_implied_33` | `GridState` | `CenterState` | Choose center color maximizing 3×3 cross-equality constraint. |
| `choose_best_flank` | `GridState` | `CenterState` | Choose center color maximizing flank-agreement hits. |
| `choose_best_cross` | `GridState` | `CenterState` | Choose center color maximizing uniform 4-neighborhood hits. |
| `out_cross_mode_33` | `CenterState` | `ColorState` | Output mode color from 3×3 cross around chosen centers. |
| `out_flank_mode` | `CenterState` | `ColorState` | Output mode flank color around chosen centers. |

Abstraction (A) ops

| Operation (label) | Input type | Output type | Notes |
|---|---|---|---|
| `match_universal_pos(shape=(h,w))` | `GridState` | `MatchesState` | Matches all intersected universal schemas for the given shape across all positions (single op; no per-position parameters). |
| `OpUniformColorFrom*` | `MatchesState` | `ColorState` | Multiple aggregator variants: OpUniformColorFromMatches, OpUniformColorFromMatchesExcludeGlobal(cross_only=True/False), OpUniformColorPerSchemaThenMode(cross_only=True/False), OpUniformColorFromMatchesUniformNeighborhood, etc.

These tables reflect the operations used in enumeration. See `dsl.py` for implementation details.

Removed/changed functionality
- Overlay-based `PatternOverlayExtractor` + `UniformPatternPredicate` programs have been removed from enumeration; the same functionality is handled by universal schema matchers with aggregators.

## 3. Results

- Programs found (abstraction): See repro output for current found programs with test performance indicators (✓/✗).
- Example programs:
  - `match_universal_pos(shape=(1, 3)) |> OpUniformColorFromMatchesUniformNeighborhood [✓ 1/1 test]`
  - `match_universal_pos(shape=(3, 1)) |> OpUniformColorFromMatchesUniformNeighborhood [✓ 1/1 test]`
  - `match_universal_pos(shape=(3, 3)) |> OpUniformColorFromMatchesExcludeGlobal(cross_only=True) [✓ 1/1 test]`

- Node counts (this run): See repro output for current counts.

## Code layout

- `overlay_patterns.py`: legacy overlay detector; not used in universal-only runs.
- `pattern_mining.py`: generic 1×3 schema miner (used for exploratory analysis; not required for window_nxm detection).
- `dsl.py`: pipeline wiring, enumeration/printing of programs, universal schema intersection helpers, fixed-schema matcher, and predicate for `window_nxm`.

## 4. Aggregator neighborhood semantics

- Uniform-neighborhood aggregator uses shape-parity neighborhoods: odd×odd → cross; even×even → central 2×2; odd×even → 1×2; even×odd → 2×1. 1×3 and 3×1 reduce to flank checks.

## 5. Test performance indicators

All found programs include test performance indicators (✓ for pass, ✗ for fail) showing how many test cases each program correctly predicts. This helps distinguish between programs that work on training vs. test data.

## Reproducing

Requirements:
- Python 3.10+ with `numpy`

Run:
```bash
python3 repro.py
```

Artifacts:
- Image: `images/overlay_mosaic.png`
- Stats: `programs_found.json` (programs with test indicators), `repro_stats.json` (node counts, timing)

Notes:
- Code is pattern-only.
 - To include universal schema matchers for specific shapes in enumeration, pass them explicitly:
  ```python
  dsl.enumerate_programs_for_task(task, universal_shapes=[(1,3),(3,1),(3,3),(5,5)])
  ```
