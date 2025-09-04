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

