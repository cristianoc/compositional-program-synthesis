## Compositional Reasoning via Abstraction–Refinement (ARC-AGI)

> Initial design document / README for a new research repo.

### Why
Current test-time reasoning often attacks the whole task at once, searching a huge state space whose cost grows (often exponentially) with problem complexity. This project explores **compositional reasoning**: repeatedly solve a **simpler abstraction** of the task, then **refine** back to the concrete grid world.  
Primary sandbox: **ARC-AGI** tasks.

---

### TL;DR (approach)
1. Start from grids (concrete domain **G**).  
2. Lift to an abstract domain **A** with an embedding `e: A → G` and a mapping of examples from **G** to **A**.  
3. Solve the problem in **A**; map the solution back to **G** via `e`.  
4. Refine recursively `A → A₂ → …`, forming a tree of abstraction refinements.  
5. Score evidence that an abstraction is correct; branch and vote/select across plausible refinements to hedge against mistakes.

**Intuition:** repeated correct abstractions can shrink effective search exponentially; branching provides recovery when an abstraction is wrong.

---

### Background & baseline
- **Task:** given a few input/output grid pairs and a test input, produce the test output.  
- **Non-compositional baseline:** search for a function `f: G → G` (via enumerative DSL, constraints, LLM-guided search, or neural edit policies) such that `f(x_i) = y_i` (under the task's agreement relation), then apply `f(x_test)`.

---

### Formalization (notation)
| Symbol | Meaning |
|---|---|
| **G** | Concrete grid domain (finite palettes, shapes, positions). |
| **A** | Abstract domain. |
| `e: A → G` | Embedding (renderer) from abstract to concrete. |
| Mapping | For each training pair `(x, y)` and test input `x*`, find `(x̂, ŷ)` and `x̂*` in **A** such that `e(x̂) ≈ x` and `e(ŷ) ≈ y` (≈ = task agreement). |
| `f_A: A → A` | Abstract solver mapping `x̂` to `ŷ`; predict `ŷ* = f_A(x̂*)`, then output `e(ŷ*)`. |

*A simple, consistent mapping—and its simplicity—are used as evidence that the abstraction suits the task.*

---

### Scope (what this document does **not** prescribe)
How to **obtain** abstractions is intentionally left open. This README defines the problem, interfaces, and evaluation framing only. Possible mechanisms—enumeration from a fixed library, LLM-guided proposals, meta-learned proposers, etc.—are future work.

- A fixed library can bootstrap experiments but limits expressivity.  
- The intended direction is **test-time discovery** of abstractions per puzzle, i.e., search over abstraction hypotheses with evidence-guided branching.  
- The orchestrator treats abstraction discovery itself as search.

---

### Examples of abstractions `A` and embeddings `e`  *(illustrative only)*
These illustrate the interface `(A, e, mappings)`; they are **not** a prescribed library.

- **Palette reduction:** a color never appears → `A` is `G` without that color; `e` is identity on remaining colors.  
- **Object graph:** `A` contains connected components and relations (adjacency, symmetry); `e` renders objects to pixels.  
- **Downsampling / tiling:** `A` compresses blocks (e.g., 2×2→1×1) with an invertible render back to `G`.  
- **Symmetry quotienting:** factor out rotations/reflections; `e` re-applies a canonical pose.  
- **Masking irrelevant regions:** `A` excludes proven-irrelevant cells; `e` fills them via a carry-over rule.

Abstractions can be **parametric** (which color, which symmetry, etc.) and **composed**.

---

### Abstraction–refinement tree
Each node is a task hypothesis:

```text
Node = {
  A: abstract domain,
  e: embedding A→G,
  m: mapping of concrete examples/test into A,
  S_A: solver on A,
  score: evidence that (A, e, m) is valid for this task,
}
```

Edges represent further abstraction `A → A₂` or refinement choices (e.g., symmetry, palette subset). We run bounded best-first / beam search over the tree. Leaves emit candidate predictions in **G** via `e`.

#### Evidence & scoring
- **Training consistency:** how well the abstract hypothesis explains `(x_i, y_i)` after embedding.  
- **Compression / MDL:** shorter descriptions of `A`, `e`, and `f_A` score higher.  
- **Stability checks:** invariants (counts, symmetries) preserved across examples.  
- **Robustness:** small perturbations don't break the mapping.  
- **Prior:** prefer lower-capacity abstractions first.

#### Selection & recovery
- Maintain a **K-best** frontier; emit multiple candidate outputs.  
- **Vote** (e.g., majority weighted by score) and/or validate with auxiliary constraints (object counts, boundary rules).  
- **Fallback:** if abstractions underperform, revert to the non-compositional baseline.

---

### Algorithms (sketch)
#### Orchestrator pseudocode
```python
def solve_arc(task):
    frontier = init_frontier_with_baseline(task)
    while budget_not_exceeded():
        node = pop_best(frontier)
        if is_goal(node):
            emit(node.predict())
            continue
        for child in expand(node):               # propose/refine abstractions
            if consistent(child, task.train):
                child.score = evidence(child, task.train)
                frontier.push(child)
    return select_among_emitted(voting=True, constraints=True)
```

#### `expand(node)`: proposing abstractions
This document deliberately leaves the proposal mechanism unspecified. In the envisioned track, abstraction discovery occurs **at test time** as a search over hypotheses guided by evidence (training consistency, MDL, priors).

#### Solvers on `A`
- Lightweight program synthesis / enumerative DSL over `A`.  
- Constraint solving (ILP/SAT) on object relations.  
- Learned policies that operate on `A`'s symbols/objects (not pixels).

---

### Context & related work
This project touches several neighboring lines of work; we aim to interoperate with (not replace) them. Importantly, we avoid committing to a fixed abstraction library and instead emphasize instance-wise, test-time adaptation—aligning with trends in test-time compute and structured search.

#### ARC-AGI benchmark and community
- ARC frames reasoning as generalization from few examples under human-centric priors. Our setup follows ARC's spirit (program search + abstraction) and is compatible with ARC-AGI-1/2 harnesses and emerging agentic variants.  
- Community reports emphasize program synthesis and neurosymbolic hybrids; the abstraction–refinement layer is designed to sit above whichever concrete solver you use.

#### Abstraction–refinement & formal methods
- Inspired by **CEGAR** in model checking: maintain a small abstract model and refine when evidence demands it.  
- Difference here: ARC lacks ground-truth counterexamples at test time, so we replace counterexamples with **evidence scores** and hedge via branching + voting.

#### Program synthesis & library learning
- Enumerative/constraint-based synthesis and DSL design remain strong baselines; in this framing they operate **within `A`** once an abstraction is posited.  
- Library learning (e.g., DreamCoder-style) may help, but is optional.

#### Test-time compute & LLM reasoning
- CoT/Self-Consistency, Tree/Graph/Program-of-Thoughts can be used **inside `A`**. The contribution here is to shape the state space before invoking these methods and to **score hypotheses** for compute allocation.

---

### Guarantees & limits
- **Shrinkage intuition:** if each correct abstraction reduces branching by factor `r < 1`, depth `d` yields `r^d` effective search.  
- **Failure mode:** a wrong abstraction (e.g., assuming a color is absent that appears in the test output) can trap a single path—hence **branching, voting, and fallback** are essential.

---

### Implementation plan
**Phase 0 – Baseline**  
- Provide a simple ARC program-search baseline in **G** for comparison.

**Phase 1 – Interfaces & scaffolding**  
- Define types for `Grid (G)`, `AbstractDomain (A)`, `Embedding (e)`, `Mapping (m)`, `AbstractSolver`, `Node`, and scoring hooks. No commitment to any particular abstraction acquisition strategy.

**Phase 2 – Orchestrator skeleton**  
- Implement beam/best-first search over abstraction hypotheses. Provide a pluggable `propose(node)` hook (enumerative, LLM-guided, learned, etc.).

**Phase 3 – Evaluation harness**  
- Metrics: success rate, compute budget, nodes expanded, abstraction depth, solves unique to compositional mode. Ablations: with/without branching, voting, fallback.

**(Optional) Phase X – Demo probes**  
- Small, non-authoritative example proposers for experimentation, kept separate from the core.

---

### Repository layout (proposed)
```text
compositional-arc/
  README.md                # this file
  data/
    arc/                   # ARC tasks (train/dev)
  core/
    domains.py             # G, A, e, m interfaces & utilities
    hypotheses/            # optional task-local hypotheses (empty by default)
    solver/
      baseline.py          # non-compositional baseline in G
      abstract.py          # abstract-solver interfaces (placeholders)
      constraints.py       # validators/invariants
    orchestrator.py        # search over abstraction/refinement tree
    scoring.py             # evidence/MDL/prior hooks
  experiments/
    eval.py                # harness & metrics
    configs/               # beam sizes, budgets, ablations
  docs/
    figures/               # later: diagrams & examples
  LICENSE
```

---

### Interfaces (typing sketch)
```python
class Grid: ...                      # concrete grid in G
class AbstractDomain: ...            # parametrized domain A

class Embedding:                     # e: A→G
    def render(self, a: 'A') -> Grid: ...

class Mapping:                       # m: G→A (for examples)
    def lift(self, g: Grid) -> 'A': ...

class AbstractSolver:                # solver S_A
    def fit(self, pairs_A): ...
    def predict(self, a_test): ...

class Node:
    A: AbstractDomain
    e: Embedding
    m: Mapping
    solver: AbstractSolver
    score: float
```

---

### Examples (intuition only)
- Missing color: detect unused color → drop it in `A` → solve in smaller palette → embed back (identity on remaining colors).  
- Object duplication with symmetry: factor out rotation in `A`, reason over object graphs, then render with chosen pose.  
- Blockwise scaling: downsample to detect macro-pattern, synthesize macro logic, upsample back.

---

### Open questions
- How best to discover abstractions `(m)` robustly from 1–5 examples?  
- What priors on abstraction families work across ARC tasks?  
- How to quantify evidence: MDL vs. Bayesian model evidence vs. discriminative scoring?  
- How to combine LLM-proposed abstractions with verifiable checks?

---

### Getting started *(placeholder)*
```bash
pip install -e .
python experiments/eval.py --config configs/baseline.yaml
python experiments/eval.py --config configs/comp.yaml
```

---

### Further reading & references *(selection)*
