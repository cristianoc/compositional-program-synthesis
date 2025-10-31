# Experiment: Invariant — “Output is one extracted object (cropped), recolored, and rotated”

**Goal.** Demonstrate how adding an invariant and carrying meta-information across the A1→A2 abstraction boundary shrinks the state space and makes search trivial on a tailored dataset, while keeping the domain fixed (grids) and only constraining the *functions* that operate on them.

---

## TL;DR

- **Invariant:** pick one object from the input grid → **crop** → **recolor** (uniformly) → **rotate** by 90°×k → (optionally) map colors back to original via A1 inverse.
- **DSL Extension:** Added meta-carrying helpers that expose A1 palette maps, A2 canonical component order, and slots for `{component_index, recolor_rule, rotation_k}`. New `apply_invariant_using_meta(...)` executes the invariant in that meta space.
- **Datasets:** Created two *tailored* datasets (24 pairs each), where the invariant holds:
  - **DS1:** recolor = `most_frequent`, rotation `k=1` (90°)
  - **DS2:** recolor = `least_frequent`, rotation `k=2` (180°)
- **State-Space Reduction (per dataset, summed over all pairs):**  
  Global (G): try every component × 2 recolors × 4 rotations → 704–816 candidates → **Expected tries ~15–18**  
  Using abstraction:
  - A2: **192** candidates → **4.5** expected tries  
  - A2 + learned recolor: **96** → **2.5**  
  - A2 + recolor + rotation (learned): **24** → **1.0**  
  - **Overall reduction:** **~97%** from G → final.
- **Accuracy on tailored test sets:** **100%** with learned `{recolor_rule, k}`.

Artifacts saved to the workspace:
- Datasets:  
  - `invariant_ds_mf_k1.npz` (DS1)  
  - `invariant_ds_lf_k2.npz` (DS2)
- JSON datasets (for easy inspection):  
  - `DS1_mostfreq_k1.json`  
  - `DS2_leastfreq_k2.json`
- Metrics bundle: `abstraction_meta_metrics.json`
- Invariant operator module: `invariant_ops.py`

---

## Formal Invariant (input→output)

Let a grid be a function \(x:\{0..H-1\}\times\{0..W-1\}\to\Sigma\) with background `0`. Let \(\mathcal{C}(x)\) be 4-connected non-background components.

Operators:
- `crop_C(x)`: minimal bbox around component \(C\), others set to 0.
- `recolor_ρ(g)`: map all non-zero to a target color per rule ρ (e.g., most/least frequent color in input).
- `R_k(g)`: 90° rotation repeated `k∈{0,1,2,3}` times.

**Invariant:**  
There exist \(C\in\mathcal{C}(x)\), \(k\in\{0,1,2,3\}\), and a recolor rule ρ such that
\(
y = R_k\big(\; \text{recolor}_ρ(\;\text{crop}_C(x)\;) \big).
\)

---

## DSL Extensions (meta-carrying abstraction)

The core DSL (`dsl.py`) already exposes:
- **A1** palette canonicalization (with `orig_for_can` / `can_for_orig`),
- **A2** canonical object ordering (list of components).

We extend it with a thin meta layer and an invariant executor:

```python
# === A1→A2 with meta ===
def A1A2_with_meta(x):
    x_hat, m1 = alpha1_palette(x)
    _,     m2 = alpha2_objorder(x_hat)
    meta = {
        "orig_for_can": m1["orig_for_can"],   # canonical→original
        "can_for_orig": m1["can_for_orig"],   # original→canonical
        "order": m2["order"],                 # canonical list of components
        # slots to be learned/selected:
        "selected_component_index": None,
        "recolor_rule": None,  # "most_frequent" | "least_frequent" | etc.
        "rotation_k": None,    # 0..3
    }
    return x_hat, meta

# === Execute invariant inside abstraction space ===
def apply_invariant_using_meta(x_hat, meta, comp_index, recolor_rule, k):
    meta["selected_component_index"] = comp_index
    meta["recolor_rule"] = recolor_rule
    meta["rotation_k"]   = k

    comps = meta["order"]
    if not comps:
        return np.zeros((1,1), dtype=int)
    comp = comps[comp_index % len(comps)]

    (r0,c0,r1,c1) = comp["bbox"]
    H, W = r1-r0+1, c1-c0+1
    obj = np.zeros((H,W), dtype=int)
    for (r,c) in comp["pixels"]:
        obj[r-r0, c-c0] = int(x_hat[r,c])

    # recolor rule on canonical input
    if recolor_rule == "most_frequent":
        hist = color_hist_nonzero(x_hat); m = max(hist.values())
        cands = [c for c,v in hist.items() if v==m]; target = min(cands) if cands else 0
    elif recolor_rule == "least_frequent":
        hist = color_hist_nonzero(x_hat)
        if not hist: target = 0
        else:
            m = min(hist.values()); cands = [c for c,v in hist.items() if v==m]; target = min(cands)
    else:
        target = 1

    obj[obj!=0] = target
    y_hat = np.rot90(obj, k % 4)

    # map back to original via A1 inverse
    invmap = meta["orig_for_can"]
    y = np.zeros_like(y_hat)
    for r in range(y_hat.shape[0]):
        for c in range(y_hat.shape[1]):
            v = int(y_hat[r,c])
            y[r,c] = 0 if v==0 else invmap.get(v, v)
    return y
```

> Note: The above extension mirrors the minimal code I used in the notebook; it assumes the presence of `alpha1_palette`, `alpha2_objorder`, and `color_hist_nonzero` from your `dsl.py`.

### Convenience “operator family” (separate file)

I also added a small operator module (`invariant_ops.py`) to make the invariant callable as a single function:

```python
def invariant_extract_recolor_rotate(
    x, select_rule="canonical_first",
    recolor_rule="most_frequent", k=1,
    map_back_to_original=True
):  # returns the output grid
    # 1) A1 palette → 2) A2 ordering → 3) pick comp → 4) crop → 5) recolor → 6) rotate → 7) map back
    ...
```

This is used both for dataset generation and quick evaluation.

---

## Tailored Datasets

**Generator idea.** Create an input with 3–5 rectangular objects on a 12×12 background. The output is produced by running the invariant with a *fixed* recolor rule and rotation, so a unique program is correct for the entire dataset.

- **DS1:** recolor `most_frequent`, rotation `k=1` (90°)
- **DS2:** recolor `least_frequent`, rotation `k=2` (180°)
- Size: 24 pairs each.

Artifacts (already generated by the notebook):
- `invariant_ds_mf_k1.npz` (DS1)  
- `invariant_ds_lf_k2.npz` (DS2)  
- `DS1_mostfreq_k1.json`, `DS2_leastfreq_k2.json` (human-readable dumps)

---

## State-Space Accounting

For an input with `N` components (after A1→A2), the candidate counts are:

- **Global (G):** `N × 2 recolor × 4 rotations = 8N`
- **A2 (canonical-first component):** `2 × 4 = 8`
- **A2 + learned recolor:** `4` (rotations only)
- **A2 + learned recolor + learned rotation:** `1`

With a single valid program inside each set, the expected number of tries with random ordering is \((M+1)/2\).

### Measured on DS1 (most_frequent, 90°)

| Metric | Value |
|---|---|
| Pairs (train/test) | 24 (12 / 12) |
| Learned config | recolor = `most_frequent`, k = `1` |
| Accuracy | train = **1.000**, test = **1.000** |
| Total candidates | G = **704**, A2 = **192**, A2+color = **96**, A2+color+rot = **24** |
| Expected tries (avg/pair) | G = **15.17**, A2 = **4.50**, A2+color = **2.50**, A2+color+rot = **1.00** |
| Reductions | G→A2 = **72.7%**, A2→A2+color = **50.0%**, A2+color→A2+color+rot = **75.0%**, Overall G→final = **96.6%** |

### Measured on DS2 (least_frequent, 180°)

| Metric | Value |
|---|---|
| Pairs (train/test) | 24 (12 / 12) |
| Learned config | recolor = `least_frequent`, k = `2` |
| Accuracy | train = **1.000**, test = **1.000** |
| Total candidates | G = **816**, A2 = **192**, A2+color = **96**, A2+color+rot = **24** |
| Expected tries (avg/pair) | G = **17.50**, A2 = **4.50**, A2+color = **2.50**, A2+color+rot = **1.00** |
| Reductions | G→A2 = **76.5%**, A2→A2+color = **50.0%**, A2+color→A2+color+rot = **75.0%**, Overall G→final = **97.1%** |

---

## Repro Instructions

1) **Files used/generated**
   - `dsl.py` (your original file)  
   - `invariant_ops.py` (added operator)  
   - `invariant_ds_mf_k1.npz`, `invariant_ds_lf_k2.npz` (datasets)  
   - `abstraction_meta_metrics.json` (summary metrics)

2) **Quick evaluate the invariant family on a dataset**
```python
from invariant_ops import invariant_extract_recolor_rotate
import numpy as np

data = np.load("invariant_ds_mf_k1.npz")
x0 = data["x0"]; y0 = data["y0"]
y_pred = invariant_extract_recolor_rotate(x0, recolor_rule="most_frequent", k=1, map_back_to_original=True)
assert (y_pred == y0).all()
```

3) **Measure global vs abstraction-guided**
```python
# Pseudocode outline used in the notebook:
def count_components_after_A1A2(x):
    x_hat, _ = alpha1_palette(x)
    _, meta = alpha2_objorder(x_hat)
    return len(meta["order"])

def explore_global(x):          return count_components_after_A1A2(x) * 2 * 4
def explore_with_A2(x):         return 2 * 4
def explore_with_A2_color(x):   return 4
def explore_with_full_meta(x):  return 1
# Learn recolor+k on train, evaluate on test with apply_invariant_using_meta(...)
```

---

## Notes & Next Steps

- The invariant datasets are intentionally simple (rectangles). We can enrich shape variety (L/T/“plus”), add noise, or vary crop rules to probe robustness.
- For non-tailored corpora (e.g., your `ONE_LARGE_DATASET`), the same accounting still shows **search-size** benefits even when the invariant is occasionally violated; the accuracy just won’t be 100%.
- The meta slots `{component_index, recolor_rule, rotation_k}` provide a clean hook for learning/search at the abstraction level, while keeping the **domain** fixed.

---

## Artifacts (Workspace Paths)

- `invariant_ops.py`  
- `invariant_ds_mf_k1.npz`  
- `invariant_ds_lf_k2.npz`  
- `DS1_mostfreq_k1.json`  
- `DS2_leastfreq_k2.json`  
- `abstraction_meta_metrics.json`
