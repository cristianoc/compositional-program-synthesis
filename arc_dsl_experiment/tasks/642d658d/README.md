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
  - `overlay_train_i.png`: h3_yellow (horizontal `(x,4,x)` centers)
  - `overlay_train_i_v.png`: v3_yellow (vertical `(x,4,x)` centers)
  - `overlay_train_i_x.png`: cross3_yellow (one overlay per yellow center; shown as a cross)
- Right: predicted color.

| Train 1 (GT=2) | Train 2 (GT=3) |
|---|---|
| ![](images/overlay_train_1.png) | ![](images/overlay_train_2.png) |

| Train 3 (GT=8) | Test input (pred=2) |
|---|---|
| ![](images/overlay_train_3.png) | ![](images/overlay_test.png) |

## Abstract
The `PatternOverlayExtractor` emits overlays for three explicit patterns: h3_yellow `(x,4,x)` horizontally, v3_yellow `(x,4,x)` vertically, and cross3_yellow (one overlay per yellow pixel). A `UniformPatternPredicate` then reads the local structure consistent with the pattern kind and outputs a single color. Program search enumerates these three pattern kinds. On this task, all three pattern kinds yield a correct solution.

## 1. Methods (pattern-only)

- `PatternOverlayExtractor(kind=...)` with kinds:
  - `h3_yellow`: one overlay per center matching `(x,4,x)` horizontally
  - `v3_yellow`: one overlay per center matching `(x,4,x)` vertically
  - `cross3_yellow`: one overlay per yellow pixel (drawn as a cross in the renderer)

- `UniformPatternPredicate` (kind-aware):
  - For `h3_yellow`, uses horizontal flanks; for `v3_yellow`, vertical flanks; for `cross3_yellow`, uniform cross color; falls back to cross-mode if needed.

- Program schema (abstraction space):
```
PatternOverlayExtractor(kind=...) |> UniformPatternPredicate |> OutputAgreedColor
```

## 2. Enumeration spaces

- G core: color rules only (no pre-ops). Nodes = number of rules (here 4).
- Abstraction: pattern kinds only (no pre-ops). Nodes = 3.

## 3. Results

- Programs found (abstraction):
  - `PatternOverlayExtractor(kind=h3_yellow) |> UniformPatternPredicate |> OutputAgreedColor`
  - `PatternOverlayExtractor(kind=v3_yellow) |> UniformPatternPredicate |> OutputAgreedColor`
  - `PatternOverlayExtractor(kind=cross3_yellow) |> UniformPatternPredicate |> OutputAgreedColor`

- Node counts (this run):
  - G core: 4
  - Abstraction: 3

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

