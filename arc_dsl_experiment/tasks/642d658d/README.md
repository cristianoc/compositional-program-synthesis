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

## Pattern overlays in action
- Mosaic view with all train and test examples across pattern kinds:
  - Columns: H3, V3, WINDOW (centerless n×n)
  - Each panel shows the input grid with overlay boxes on the left and a single-cell output on the right.

![](images/overlay_mosaic.png)

## Abstract
The `PatternOverlayExtractor` emits overlays for three kinds: `h3` (horizontal `[X, c, X]` with center color `c`), `v3` (vertical `[X, c, X]`), and `window_nxn` (every full `n×n` window, centerless, each carrying its per-window schema). The `UniformPatternPredicate` then reads evidence consistent with the selected pattern to output a single color. For `h3`/`v3`, evidence is the nonzero flank color agreed at each center. For `window_nxn`, evidence comes from uniform, nonzero center neighborhoods inside each window (odd `n`: cross; even `n`: central 2×2). Program search enumerates pattern kinds × colors.

## 1. Methods (pattern-only)

- `PatternOverlayExtractor(kind=..., color=c)` with kinds and detection rules:
  - `h3` (horizontal `[X, c, X]`): emits one overlay per row position whose 3-length window satisfies the generic schema `[X, c, X]` with a nonzero flank color. Detection uses `pattern_mining.gen_schemas_for_triple` to confirm the pattern.
  - `v3` (vertical `[X, c, X]`): analogous on columns.
  - `window_nxn` (centerless `n×n` windows): emits one overlay per full `n×n` window on the grid. Each overlay includes the raw window and a per-window equality schema.

- `UniformPatternPredicate` (kind-aware evidence → final color):
  - `h3`: At each overlay center, if left and right flanks exist and are equal and nonzero, collect that flank color. Return the mode across centers (tie → min). If no such evidence, falls back to the most frequent valid cross color around overlay centers.
  - `v3`: Same using above/below flanks.
  - `window_nxn`: For each window, if the center neighborhood is uniform and non-zero, collect that color (odd `n`: four-cross; even `n`: central 2×2). Final prediction is the mode (tie → min). No center color is required.

- Program schema (abstraction space):
```
PatternOverlayExtractor(kind=..., color=...) |> UniformPatternPredicate |> OutputAgreedColor
```

## 2. Enumeration spaces

- G core: color rules only (no pre-ops). Nodes = number of rules (here 4).
- Abstraction: pattern kinds × colors (1–9), no pre-ops. Nodes = 3 × 9 = 27.

## 3. Results

- Programs found (abstraction):
  - `PatternOverlayExtractor(kind=h3, color=c) |> UniformPatternPredicate |> OutputAgreedColor`
  - `PatternOverlayExtractor(kind=v3, color=c) |> UniformPatternPredicate |> OutputAgreedColor`
  - `PatternOverlayExtractor(kind=window_nxn, color=c) |> UniformPatternPredicate |> OutputAgreedColor`

- Node counts (this run):
  - G core: 4
  - Abstraction: 27

## Code layout

- `overlay_patterns.py`: overlay detector implementing kinds `h3`, `v3`, `window_nxn` (supports `window_size`; emits full windows with per-window schema).
- `pattern_mining.py`: generic 1×3 schema miner used by `h3` and `v3` detection.
- `dsl.py`: pipeline wiring, enumeration/printing of programs, and kind-aware predicate for `h3`, `v3`, and `window_nxn`.

## 4. Window size semantics (`window_nxn`)

- `WINDOW_SIZE_DEFAULT`: Global default `n` used by detection and printing; any `n ≥ 1`.
- Full-window requirement: Only full `n×n` windows entirely inside the grid are emitted for `window_nxn`.
- Odd vs even neighborhoods: odd `n` uses a four-way cross; even `n` uses the central 2×2.
- Edge cases: `n=1` yields no evidence; larger `n` reduces available windows on small grids.

## 5. Printed program schemas

For `window_nxn`, programs include `window_size=n`. Each overlay already carries its per-window schema (`schema` field), so no global consensus string is printed.

## Reproducing

Requirements:
- Python 3.10+ with `numpy`

Run:
```bash
python3 repro.py
```

Artifacts:
- Image: `images/overlay_mosaic.png`
- Stats: `repro_stats.json` (node counts, programs, timing), `pattern_stats.json` (per-example overlay details)

Notes:
- Code is pattern-only.
