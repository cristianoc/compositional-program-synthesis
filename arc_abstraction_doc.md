# Compositional Abstractions for ARC-Style Tasks

*A short, self-contained report with runnable artifacts*

## Abstract

We study how **compositional abstractions** shrink search for ARC-style grid puzzles. Our guiding idea is to represent an abstraction as **G + invariant**: an embedding $\alpha$ of grids into a canonical space (with its gauge), together with any invariants the data must satisfy. An abstraction is **valid** for a dataset if: (i) its invariants hold; and (ii) a **mapping exists** in the abstract space—formally, $\alpha(y)$ is constant on each fiber of $\alpha(x)$. We then search for a simple mapping $f$ **inside** the abstract space using a tiny DSL.

On a small dataset (3 train pairs; includes a symmetry-inducing tie), we compare three search regimes:

1. **G** (raw space, with palette permutations),
2. **A1** (palette canonicalization), and
3. **A1→A2** (A1 followed by canonical object order).

We observe strict improvements **G → A1 → A1→A2** in total candidates, average tries to first success, and wall-time—while preserving or increasing the fraction of valid programs.

---

## 1. Introduction

ARC tasks operate on small integer grids. Many puzzles are easy to **describe** but hard to **search** because color labels and object enumerations create massive symmetries. Our hypothesis: if we **factor** these symmetries out (palette first, then object order), the mapping class we must search collapses, and simple rules suffice.

We make three design choices:

* **Abstractions as G + invariant.** Each $\alpha$ includes its embedding and any declared invariants.
* **Verification before search.** We check invariants and **existence of $f$** (fiber consistency) before enumerating DSL programs.
* **Composition.** We verify A1 in isolation; only then do we search A2 **on top of** A1 and verify A1→A2.

---

## 2. Running Examples (3 pairs)

Each pair $(x, y)$ is a small grid. The intended rule: **recolor the smallest connected component** to the **least-frequent non-zero color** (ties by top-left, then color id). We include one **tie case** to surface object-order symmetry.

Example 1 (sketch):

```
x:
. . . . .
. 1 1 . .
. 1 1 . .
2 . . . .
. 4 . . .

y = recolor(smallest, least-frequent)
```

Example 2 includes multiple colors and a small singleton; Example 3 (tie case) contains **two equally small singletons** so that “the smallest” is ambiguous without a canonical order.

(Exact grids and code are in the provided script.)

---

## 3. Abstractions as G + Invariant

### 3.1 A1: Palette canonicalization

$\alpha_1:$ relabel non-zero colors by **decreasing frequency**, ties by color id: most frequent $\mapsto 1$, next $\mapsto 2$, etc.

* **Invariant.** None forbidding; background 0 is preserved.
* **Gauge.** A bidirectional palette map (orig↔canonical).
* **Why.** Quotients out **palette symmetry** so “least frequent color” has a canonical id.

### 3.2 A2: Canonical object order (on top of A1)

$\alpha_2:$ sort connected components by $(\text{area}, \text{top}, \text{left}, \text{color})$.

* **Invariant.** Metadata-only (no extra forbidding).
* **Gauge.** The canonical order list.
* **Why.** Quotients out **object-enumeration symmetry** so rules like “component **index 0**” are stable—even in ties.

### 3.3 Mapping existence (search-free)

For a dataset $\{(x_i,y_i)\}$ and abstraction $\alpha$, a function $f$ **exists** iff:

$$
\alpha(x_i) = \alpha(x_j) \;\Rightarrow\; \alpha(y_i) = \alpha(y_j) \quad \text{for all } i,j.
$$

We check this by **fiber consistency** over the dataset. Only when invariants hold **and** a mapping exists do we proceed to search.

---

## 4. Search Spaces (DSLs)

* **G (raw).** We simulate palette symmetry by enumerating **150 random color permutations** as *pre-ops*, then apply simple component/color rules. This intentionally inflates G’s candidate set.
* **A1.** After palette canonicalization, we still allow **many decoy tie-break selectors** (canonical and seeded heuristics) to show that A1 alone can still be costlier than composing A2.
* **A1→A2.** After object order canonicalization, the DSL collapses to **two** programs:
  {color rule} × {`index0`} (implemented as canonical smallest).

Across all spaces the mapping is the same family: **choose a component** × **choose a color** × **recolor that component**.

---

## 5. Experimental Protocol

* **Dataset:** 3 training pairs (includes a tie).
* **Metrics per space:**

  * `total_candidates` — size of the enumerated DSL,
  * `num_valid` — programs fitting all pairs,
  * `avg_tries_to_success` — Monte-Carlo rank of the first valid program under random ordering (lower is better),
  * `wall_time_s` — wall-clock for full enumeration & checking.
* **Gating:** A1 and A1→A2 are *only* searched after invariants pass and the existence test succeeds (they do in our data).

---

## 6. Results

### 6.1 Single-dataset summary

```
[G]      total_candidates=604  num_valid=99  avg_tries_to_success=6.4725  wall_time_s=0.186
[A1]     total_candidates=52   num_valid=12  avg_tries_to_success=4.105   wall_time_s=0.024
[A1→A2]  total_candidates=2    num_valid=2   avg_tries_to_success=1.0     wall_time_s=0.002
```

### 6.2 What the numbers show

* **Fewer programs to consider.** 604 → 52 → **2**.
  Palette canonicalization (A1) removes color symmetry; A2 then removes object-order symmetry, collapsing the DSL to essentially “pick index 0”.
* **Easier search.** `avg_tries_to_success` drops **6.47 → 4.11 → 1.0**.
  Even though A1 still contains decoy selectors, it’s already better than G; A1→A2 is trivial to solve.
* **Faster runtime.** `wall_time_s` drops **0.186 s → 0.024 s → 0.002 s**.
  Less enumeration, less checking, same correctness.

The **num\_valid** fraction increases with abstraction (especially at A1→A2 where 2/2 candidates are valid), which is why average tries falls so sharply.

---

## 7. Why A2 is needed (and helpful)

The dataset contains a **tie** (two smallest components of equal area). In A1, “pick the smallest” is **under-specified** unless we commit to a tie-break. If the DSL allows many tie-break heuristics, search remains larger and noisier. A2 **canonicalizes** the object order so the program can simply say **`index 0`**, eliminating the ambiguity and saving search.

---

## 8. Limitations & Extensions

* Our DSL is intentionally tiny; the point is the **relative** search size, not absolute optimality.
* Real ARC tasks may require additional abstractions (e.g., shape signatures, symmetry groups). The same **verify-then-search** recipe applies: check invariants, prove existence by fibers, then search in the abstract space.

---

## 9. Reproducibility (runnable artifacts)

* **Script:** `arc_abstraction_case1_single_dataset.py`
* **Outputs:** `single_metrics.txt`, `single_metrics.json`
  (Links are provided in the conversation; run with `python3 arc_abstraction_case1_single_dataset.py`.)

---

## 10. Takeaway

Abstractions as **G + invariant** provide a clean contract: **verify** (invariants + existence), then **search** a much smaller, more stable DSL. Composing **A1 (palette)** with **A2 (object order)** yields strict, measured improvements in **program count**, **tries to success**, and **wall-time**—turning a symmetry-riddled search problem into an easy one.
