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
  - Columns: H3, V3, WINDOW (centerless n×m)
  - Each panel shows the input grid with overlay boxes on the left and a single-cell output on the right.

![](images/overlay_mosaic.png)

## Abstract
The `PatternOverlayExtractor` emits overlays for one kind: `window_nxm` (every full `n×m` window, centerless, each carrying its per-window schema). The `UniformPatternPredicate` reads evidence from the window’s center neighborhood (odd×odd: cross; even×even: central 2×2; odd×even: 1×2; even×odd: 2×1). Program search enumerates color parameters per window shape.

## 1. Methods (pattern-only)

- `PatternOverlayExtractor(kind=..., color=c)` with kinds and detection rules:
  - `window_nxm` (centerless `n×m` windows): emits one overlay per full `n×m` window on the grid. Each overlay includes the raw window and a per-window equality schema.

- `UniformPatternPredicate` (kind-aware evidence → final color):
  - `window_nxm`: For each window, if the center neighborhood is uniform and non-zero, collect that color (odd×odd: cross; even×even: 2×2; odd×even: 1×2; even×odd: 2×1). Final prediction is the mode (tie → min). Degenerate shapes 1×3 and 3×1 behave like horizontal/vertical flank checks.

- Program schema (abstraction space):
```
PatternOverlayExtractor(kind=..., color=...) |> UniformPatternPredicate |> OutputAgreedColor
```

## 2. Enumeration spaces

- G core: composed color rules only (no pre-ops). Nodes = number of composed rules (currently 70; see repro output).
 - Abstraction: pattern kinds × colors (1–9), plus colorless window mining. Nodes = 3 × 9 + 3 = 30.
 - Single-pass enumeration: ABS is enumerated once and includes all three `window_nxm` shape instantiations: `(1,3)`, `(3,1)`, and the default window.
 - Typed composition seeds: the only operations that accept `GridState` are the chooser ops for G and the overlay extractors for ABS (`OpBrightOverlayIdentity(kind=window_nxm, window_shape∈{(1,3),(3,1),default}, color∈1..9)` and `OpBrightOverlayAllWindows(window_shape∈{(1,3),(3,1),default})`), followed by kind-appropriate predicates or schema-matching.

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
| `overlay_window_nxm` | `GridState` | `OverlayContext` | Parameterized by `color ∈ {1..9}` and `window_shape ∈ {(1,3),(3,1),default}`. |
| `uniform_pattern_predicate` | `OverlayContext` | `ColorState` | Center-neighborhood evidence aggregation by shape parity. |
| `overlay_window_nxm_all` | `GridState` | `OverlayContext` | Colorless: enumerates all full windows for a shape; no center or target color assumed. |
| `schema_match_across_grid` | `OverlayContext` | `MatchesState` | Matches stored schemas across the entire grid; returns matched subgrids with wildcards omitted. |
| `uniform_color_from_matches` | `MatchesState` | `ColorState` | Aggregates non-None, non-zero values from matches and returns mode. |

These tables reflect the explicit op registries defined in `dsl.py` (`G_TYPED_OPS` and `A_OP_TYPE_SUMMARY`).

### Colorless window mining
- Detector kind `window_nxm_all` enumerates every full `n×m` window without selecting a color or assuming a semantic center. This path is now included in enumeration via the schema-matching pipeline (`overlay_window_nxm_all |> schema_match_across_grid |> uniform_color_from_matches`).

## 3. Results

- Programs found (abstraction):
  - `PatternOverlayExtractor(kind=window_nxm, color=c, window_shape=(1,3)) |> UniformPatternPredicate |> OutputAgreedColor`
  - `PatternOverlayExtractor(kind=window_nxm, color=c, window_shape=(3,1)) |> UniformPatternPredicate |> OutputAgreedColor`
  - `PatternOverlayExtractor(kind=window_nxm, color=c, window_shape=(n,m)) |> UniformPatternPredicate |> OutputAgreedColor`

- Node counts (this run):
  - G core: 70 (composition only)
  - Abstraction: 27

## Code layout

- `overlay_patterns.py`: overlay detector implementing `window_nxm` (supports `window_shape`; emits full windows with per-window schema).
- `pattern_mining.py`: generic 1×3 schema miner (used for exploratory analysis; not required for window_nxm detection).
- `dsl.py`: pipeline wiring, enumeration/printing of programs, and predicate for `window_nxm`.

## 4. Window size semantics (`window_nxm`)

- `WINDOW_SHAPE_DEFAULT`: Global default `n` used by detection and printing; any `n ≥ 1`.
- Full-window requirement: Only full `n×m` windows entirely inside the grid are emitted for `window_nxm`.
- Odd vs even neighborhoods: odd `n` uses a four-way cross; even `n` uses the central 2×2.
- Edge cases: `n=1` yields no evidence; larger `n` reduces available windows on small grids.

## 5. Printed program schemas

For `window_nxm`, programs include `window_shape=n`. Each overlay already carries its per-window schema (`schema` field), so no global consensus string is printed.

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
