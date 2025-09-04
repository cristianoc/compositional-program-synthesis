# ARC Task 642d658d

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

## Overlay detector in action
- Yellow boxes show detected bright overlays (centers used by `UniformCrossPattern`).
- Right panels show the predicted 1×1 color for each grid.

| Train 1 (GT=2) | Train 2 (GT=3) |
|---|---|
| ![](images/overlay_train_1.png) | ![](images/overlay_train_2.png) |

| Train 3 (GT=8) | Test input (pred=2) |
|---|---|
| ![](images/overlay_train_3.png) | ![](images/overlay_test.png) |

## Abstract
We investigated whether a salience (visual prominence)-driven overlay abstraction improves program search for an ARC puzzle compared to a core DSL ("G") that operates with palette pre-ops and global color rules. We integrated an overlay extractor (`detect_bright_overlays`) as an identity abstraction that stores overlays, composed it with a `UniformCrossPattern` predicate, and compared against G (including several new local-structure primitives). On the full 3-train task, G found no solutions (1.70 s, 1,005 nodes). The overlay pipeline found solutions immediately (1st candidate) but at higher per-node cost (24.9 s, 201 nodes). A fast-first path reduced abstraction time substantially on a 2-train subset (24.8 s → 8.4 s for 201 nodes) but didn’t help the full set due to predicate failure triggering the fallback. We recommend merging the overlay abstraction + predicate into the DSL, plus caching and vectorization to cut per-node cost.

## 1. Introduction
ARC puzzles often require recognizing small, high-contrast markers and reading local structure around them (e.g., the color that surrounds each marker in a cross). The baseline core DSL (G) relies on palette permutations (preops) and global color selectors, which can be too weak when the task depends on salient anchors rather than global summaries. We test whether an overlay abstraction—which is identity on the grid but stores salient overlays—enables compact, human-interpretable programs that solve the task.

## 2. Methods

### 2.1 Core DSL “G”
- Search objects: `(preop, color_rule)` programs that map grid → scalar color (1×1 output).
- Preops: identity + N random palette permutations (we used N=50 & 200; 600 attempted).
- Color rules (this run = 5 total): baseline global rules (e.g., `max_id`, `argmin_hist`) plus local-structure aware primitives we added:
  - Peaks-based cross mode: `uniform_cross_at_peaks_mode`
  - Everywhere cross mode / argmax by color / uniform ring mode (variants added; exact set active = 5 total rules in these measurements).

Design limitation: even with these additions, G’s rules do not select salient anchors; they aggregate structure indiscriminately over the grid (or weakly at local maxima) and miss the consistent “read at marker centers” behavior.

### 2.2 Overlay extractor
`detect_bright_overlays(grid, …)` converts the grid to luminance (Rec.709) via palette, finds local maxima (non‑maximum suppression (NMS) radius=4) with global robust z‑score and local local center–surround z‑score thresholds, then grows compact components and fits small peak-centered boxes with a salience score (contrast-dominant). It returns overlay dicts with 1-based centers, box corners, peak luminance, contrast, area.

### 2.3 Abstraction and predicate
- Abstraction: `BrightOverlayIdentity`
  - Identity on the grid; stores overlays and minimal stats (count, max_contrast, total_area, …).
  - It includes a precise applicability check. Let `H×W` be the grid size and let the extractor
produce a list of overlays, each with `area` (in pixels), `peak_lum` (0..1 luminance at
the center) and `contrast = peak_lum − surround_mean`, where `surround_mean` is computed in a
padded window of size `[y1−context_pad : y2+context_pad, x1−context_pad : x2+context_pad]`
excluding the overlay box. Define summary stats:

- `count` = number of overlays
- `max_contrast` = max over overlays of `contrast`
- `total_area` = sum of overlay `area`
- `total_area_frac` = `total_area / (H·W)`

The abstraction is considered **applicable** iff all hold (defaults in parentheses):

1. `count ≥ min_count` (**1**)
2. `max_contrast ≥ min_contrast` (**0.08**)
3. `total_area ≥ min_total_area` (**1**)
4. If `min_total_area_frac > 0`, then `total_area_frac ≥ min_total_area_frac` (**0.0** disables this test)

Implementation sketch:
```python
def applies(self) -> bool:
    c  = self.last_stats.get('count', 0)
    mx = self.last_stats.get('max_contrast', 0.0)
    ta = self.last_stats.get('total_area', 0)
    gaf = self.last_stats.get('total_area_frac', 0.0)
    if c < self.min_count or mx < self.min_contrast or ta < self.min_total_area:
        return False
    if self.min_total_area_frac > 0.0 and gaf < self.min_total_area_frac:
        return False
    return True
```

- Pattern check (predicate): `UniformCrossPattern`
  - At each overlay center, read the 4-neighborhood (↑↓←→). The predicate succeeds iff (a) each center’s cross is uniform and non-zero, and (b) all overlays agree on the same color. The predictor then emits that agreed color.
- Program schema (abstraction space):
```
<preop>
  |> BrightOverlayIdentity
  |> UniformCrossPattern
  |> OutputAgreedColor  <!-- outputs the agreed color -->
```

### 2.4 Extensions to G
We added local-structure color rules to G (grid → scalar color) to test whether G can catch up without overlays:
- `uniform_cross_at_peaks_mode` (8-nbr luminance peaks → uniform cross → mode)
- `uniform_cross_everywhere_mode` (all anchors)
- `argmax_uniform_cross_color_count` (count per color)
- `uniform_ring_mode` (8-nbr ring uniformity)

Despite these, G lacks the salient-anchor selection step the overlay extractor provides.

### 2.5 Fast-first path & caching
To reduce abstraction cost per node:
- Fast-first selector: 8-nbr local maxima above a high luminance percentile (pₕᵢ≈99.7). If all selected centers have uniform crosses and agree on the color → return immediately; else fallback to full overlay detection.
- Tiny cache: LRU cache keyed by grid bytes for the fast centers.

### 2.6 Typed DSL and composition
We expose the abstraction pipeline as a small, typed DSL to make composition explicit and extensible.

- State types:
  - `GridState`: holds a grid (`np.ndarray[int]`).
  - `OverlayContext`: holds the same grid plus `overlays` and summary `stats`.
  - `ColorState`: holds the final scalar color (`int`).

- Operation interface:
  - `Operation[InState, OutState]` with `accepts(state)` and `apply(state)`.
  - A `Pipeline([op1, op2, ...])` runs ops sequentially; each op declares the input/output state types it accepts/produces.

- Built-in operations (used here):
  - `PreOpPalette(GridState→GridState)`: optional palette relabeling (used by G; can precede abstraction when desired).
  - `OpBrightOverlayIdentity(GridState→OverlayContext)`: runs `detect_bright_overlays`, stores overlays and stats.
  - `OpUniformCrossPattern(OverlayContext→ColorState)`: checks uniform 4-neighborhood at overlay centers and emits the agreed color (falls back to mode if needed).

- Example composition (abstraction space):
```python
from dsl import Pipeline, GridState, OpBrightOverlayIdentity, OpUniformCrossPattern

pipeline = Pipeline([
    OpBrightOverlayIdentity(),
    OpUniformCrossPattern(),
])
out = pipeline.run(GridState(grid))  # out is a ColorState
color = out.color
```

This structure makes it straightforward to add new overlay-level primitives (e.g., ring checks, object graphs) by implementing new `Operation` classes with the appropriate input/output state types and inserting them into the `Pipeline`.

#### Operation catalog (current)
- `PreOpPalette: GridState → GridState`
  - Applies a palette permutation or identity to the grid.
- `OpBrightOverlayIdentity: GridState → OverlayContext`
  - Runs `detect_bright_overlays`; attaches `overlays` and summary `stats` while preserving the grid.
- `OpUniformCrossPattern: OverlayContext → ColorState`
  - Checks uniform 4-neighborhood at overlay centers; emits agreed color (fallback to mode if needed).

Implementation note: The visual routines (palette, luminance, overlay detection, fast-first helpers) are factored into `vision.py`. The typed DSL and enumeration live in `dsl.py` and import from `vision.py`.

### 2.7 Why separate “vision” from “logic” (perception vs. reasoning)
We factor perceptual operations (in `vision.py`) from symbolic/logic operations (in `dsl.py`) to mirror the distinction between early visual processing and downstream reasoning.

- **Perception (vision.py):** luminance, salience, non‑maximum suppression, local contrast, and overlay extraction. These are low‑level, metric operations on arrays; they provide anchors/object‑like tokens and summary stats.
- **Reasoning (dsl.py):** typed, symbolic predicates and programs over those tokens (e.g., `UniformCrossPattern: OverlayContext → ColorState`), plus search/enumeration. These are discrete/logical and compose via the typed pipeline.

Benefits:
- **Invariance and safety:** vision code centralizes palette/luminance handling so logic stays label‑agnostic and robust to color relabeling.
- **Composability:** typed boundaries (`GridState → OverlayContext → ColorState`) make it easy to add new perceptual extractors or new logical predicates independently.
- **Performance & caching:** perceptual results (overlays/stats) can be cached/reused across logical variants; logic can short‑circuit without re‑running detection.
- **Scientific alignment:** mirrors current views of the visual cortex (perceptual feature extraction, grouping) feeding higher‑level reasoning modules, while allowing iterative abstraction–refinement across the interface.

## 3. Experimental Setup
- Task: The 3-train ARC puzzle (1 test). Train outputs are single colors.
- Spaces:
  - G core: `(preop × color_rule)`; with 200 preops → 201 × 5 = 1,005 nodes.
  - Abstraction: `(preop × overlay-predicate)`; with 200 preops → 201 nodes.
- Seeds: `seed=11`.
- Environment: Python / NumPy; same machine for all runs.
- Time metric: wall-clock of full enumeration over candidates; includes applying preops and evaluating on all train examples.

## 4. Results

### 4.1 Program strings (abstraction space, full 3-train, 200 preops)
- `identity |> BrightOverlayIdentity |> UniformCrossPattern |> OutputAgreedColor  <!-- outputs the agreed color -->`
- `perm_192 |> BrightOverlayIdentity |> UniformCrossPattern |> OutputAgreedColor  <!-- outputs the agreed color -->`

Interpretation: the overlay detector is luminance-based; some palette permutations preserve the “bright marker” behavior so the predicate still reads the same color.

**Note on `perm_192`:** In our search (seed=11, 200 preops), `perm_192` is a palette permutation
that maps every color used in the provided grids to itself. Therefore the permuted grids are
identical to the originals, and the program with `perm_192` is behaviorally the same as `identity`.


### 4.2 Node counts and timings

Full 3-train (N=200 preops)

| Space | Nodes Tried | Programs Found | Tries to First | Time (s) |
|---|---:|---:|---:|---:|
| G core | 1,005 | 0 | — | 1.70 |
| Overlay + predicate | 201 | 2 | 1 | 24.88 |

2-train subset (N=200 preops) — showing the fast-first gain

| Space | Nodes Tried | Programs Found | Tries to First | Time (s) |
|---|---:|---:|---:|---:|
| Overlay + predicate (before fast-first) | 201 | 2 | 1 | 24.76 |
| Overlay + predicate (with fast-first) | 201 | 17 | 1 | 8.39 |

2-train subset (N=50 preops) — scaling snapshot

| Space | Nodes Tried | Programs Found | Tries to First | Time (s) |
|---|---:|---:|---:|---:|
| G core | 255 | 0 | — | 0.38 |
| Overlay + predicate | 51 | 1 | 1 | 7.29 |

Summary:
- Abstraction space is smaller and finds a solution at the first candidate (identity). Its per-node evaluation is ~70–140 ms, dominated by salience detection + predicate checks.
- G is fast per node (~1–2 ms) but even with local rules, it finds no solution on the full 3-train.
- Fast-first path helps when the cheap selector succeeds; on the full set it often falls back (hence no time win there), but on a subset it yields a 3× speedup.

## 5. Analysis
Why G fails: Even with local rules, G lacks salience targeting—it evaluates uniform crosses at many anchors or at naive peaks, diluting the signal. The task apparently hinges on specific bright markers that the overlay extractor reliably isolates, after which the uniform-cross agreement is trivial to read out.

Why the overlay abstraction succeeds:
- Robust luminance + center–surround picks out the meaningful overlay centers (peaks with contrast), invariant to many palette permutations.
- The `UniformCrossPattern` then enforces strong consistency: every overlay must have the same uniform cross color.
- The composed program is short and interpretable.

Fast-first behavior: On grids where the high-percentile peak selector isolates the same centers that the full detector would, the fast path is accurate and cheap. On the 3-train set, at least one grid violates the fast predicate (noisy peaks or non-uniform crosses), so we fall back to the full extractor—hence no aggregate time win.

## 6. Limitations & Threats to Validity
- Single task / single seed: although representative, broader benchmarking is needed.
- Timing variability: Python loops in `_local_mean_std` and component growth cause non-trivial overhead; numbers depend on grid size and hardware.
- Palette dependence: while luminance helps, adversarial permutations could degrade peak selection; we partially observed palette invariance but didn’t quantify it.
- Predicate specificity: “uniform cross + global agreement” may be too strict for some tasks; a library of predicates is desirable.

## 7. Future Work
1. Caching overlays per (preop, grid): memoize the full extractor (not just fast centers) to reuse results across predicates; expect multi-× wins.
2. Vectorization: replace Python loops in local stats and component growth with NumPy; consider smaller non‑maximum suppression (NMS) radius or separable box sums.
3. Predicate library: add variants (e.g., majority-in-3×3, axis-consistent colors, ring uniformity) with short-circuiting and cheap prefilters.
4. G upgrades: keep the added local rules, plus an explicit anchor-selection primitive (e.g., “apply rule only at high-contrast peaks”), narrowing the gap to overlays.
5. Ablations: isolate contributions of non‑maximum suppression (NMS) radius, z-thresholds, and context padding to both accuracy and cost.
6. Batching / JIT: consider Numba or small C extensions for the detector’s hot loops.

## 8. Conclusion
Overlay-driven salience, composed with a simple `UniformCrossPattern` predicate, solves the puzzle immediately with compact programs. The trade-off is higher per-node compute versus G’s cheap but expressively weaker search. A fast-first path shows promising speedups when it hits; with caching/vectorization, the abstraction route is a strong candidate for default inclusion. Meanwhile, augmenting G with anchor-aware primitives would further close the gap without requiring full overlays in all cases.

## Glossary

- **Salience (visual prominence):** How much a pixel/region stands out from its surroundings.
- **Non‑maximum suppression (NMS):** Keeps only local maxima by suppressing nearby lower peaks.
- **Robust z‑score:** A z‑score computed using median and MAD (less sensitive to outliers).
- **Center–surround z‑score:** Local contrast computed by comparing a center pixel to its neighborhood window.
- **Predicate / Pattern check:** A Boolean test over the grid and/or overlays; here, it requires each overlay's 4‑neighborhood to be uniform and all overlays to agree on the same color.
- **OutputAgreedColor:** Final step that outputs the single color decided by the pattern check.


**Detector defaults (for reproducibility):**

- `nms_radius = 4`
- `local_radii = (1, 2, 3)`
- `peak_k = 3.4` (global robust z‑score threshold)
- `local_k = 3.8` (max local center–surround z‑score threshold)
- `p_hi = 99.7` (high‑tail luminance percentile)
- `drop_threshold = 0.06` (component growth: keep if `lum ≥ peak·(1 − drop_threshold)`)
- `scale_gamma = 1.0` (sets overlay half‑size from component spread)
- `max_radius = 1.4` (cap before rounding; ≈ 3×3 boxes when ≤ 1.5)
- `context_pad = 2` (pixels around each box to estimate surround)

## Reproducing results (overlay abstraction experiment)

Requirements:
- Python 3.10+ with `numpy`, `matplotlib` installed

Steps:
1. From this folder, install deps (no venv required):
   ```bash
   python3 -m pip install --user numpy matplotlib
   ```
2. Run the reproduction script:
   ```bash
   python3 repro.py
   ```
3. Expected outputs:
   - Console:
     - Node counts: `G core nodes: 1005`, `Overlay+predicate nodes: 201`
     - Programs found in abstraction space: `identity`, `perm_192`
     - Stats dict similar to: `{'G': {'nodes': 1005, 'programs_found': 0, ...}, 'ABS': {'nodes': 201, 'programs_found': 2, 'tries_to_first': 1, ...}}`
   - Images under `images/`:
     - `overlay_train_1.png`, `overlay_train_2.png`, `overlay_train_3.png`
     - `overlay_test.png`

Notes:
- `repro.py` uses the local `dsl.py` and `task.json` in this directory.
- Timings depend on hardware; node counts and found program strings should match.
 - If you see an "externally-managed-environment" (PEP 668) error, either use a virtualenv, or rerun the install with:
   ```bash
   python3 -m pip install --user --break-system-packages numpy matplotlib
   ```

Tracked outputs:
- `images/overlay_train_*.png`, `images/overlay_test.png`
- `repro_stats.json` (node counts, programs_found, tries_to_first, timing)

