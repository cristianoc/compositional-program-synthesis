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
- Left: grid with overlays for each pattern kind.
  - `overlay_train_i.png`: h3 (horizontal `(x,c,x)` centers, for chosen color `c`)
  - `overlay_train_i_v.png`: v3 (vertical `(x,c,x)` centers, for chosen color `c`)
  - `overlay_train_i_x.png`: cross3 (one overlay per pixel of color `c`; shown as a cross)
- Right: predicted color.

| Train 1 (GT=2) | Train 2 (GT=3) |
|---|---|
| ![](images/overlay_train_1.png) | ![](images/overlay_train_2.png) |

| Train 3 (GT=8) | Test input (pred=2) |
|---|---|
| ![](images/overlay_train_3.png) | ![](images/overlay_test.png) |

## Abstract
The `PatternOverlayExtractor` emits overlays for three explicit pattern kinds that are now color-parameterized: `h3` (interpreted as horizontal `(x,c,x)` with center color `c`), `v3` (vertical `(x,c,x)`), and `cross3` (one overlay per pixel of color `c`). A `UniformPatternPredicate` reads the local structure consistent with the pattern kind and outputs a single color. Program search enumerates pattern kinds × colors. On this task, multiple (kind, color) settings yield a correct solution.

## 1. Methods (pattern-only)

- `PatternOverlayExtractor(kind=..., color=c)` with kinds:
  - `h3`: one overlay per center matching `(x,c,x)` horizontally (center color is `c`)
  - `v3`: one overlay per center matching `(x,c,x)` vertically (center color is `c`)
  - `cross3`: one overlay per pixel of color `c` (drawn as a cross in the renderer)

- `UniformPatternPredicate` (kind-aware):
  - For `h3`, uses horizontal flanks; for `v3`, vertical flanks; for `cross3`, uniform cross color; falls back to cross-mode if needed.

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
  - `PatternOverlayExtractor(kind=cross3, color=c) |> UniformPatternPredicate |> OutputAgreedColor`

- Node counts (this run):
  - G core: 4
  - Abstraction: 27

## Code layout

- `overlay_patterns.py`: overlay detector implementing kinds `h3`, `v3`, `cross3`.
- `pattern_mining.py`: generic 1×3 schema miner used by `h3` detection.
- `dsl.py`: pipeline wiring, enumeration, and printing of programs.

## Reproducing

Requirements:
- Python 3.10+ with `numpy`, `matplotlib`

Run:
```bash
python3 repro.py
```

Artifacts:
- Images: `images/overlay_train_*.png`, `images/overlay_train_*_v.png`, `images/overlay_train_*_x.png`, `images/overlay_test*.png`
- Stats: `repro_stats.json` (node counts, programs, timing), `pattern_stats.json` (per-example overlay details)

Notes:
- Code is pattern-only.

