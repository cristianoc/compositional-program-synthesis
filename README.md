# Compositional Reasoning via Abstraction–Refinement (ARC‑AGI)

> Initial design document / README for a new research repo

## Why
Current "test‑time inference" reasoning methods tend to attack the whole problem at once, effectively searching a very large state space. As problem complexity grows, search cost often grows exponentially. This project explores **compositional reasoning**: shrinking the state space by solving successively more **abstract** versions of the problem and then refining back to the concrete task.

We use **ARC‑AGI** tasks as the primary sandbox.

---

## TL;DR (approach)
1. **Start with grids** (domain **G**).
2. **Lift** to a simpler **abstract domain A** with an **embedding** $(e: A \to G)$ and a **mapping** from concrete examples to A.
3. **Solve** the abstract problem in A; then **embed** the solution back to G.
4. **Refine recursively**: A → A₂ → …, building a **tree of abstraction refinements**.
5. **Score evidence** that an abstraction is correct from example consistency; **branch** and **vote**/select across plausible refinements to hedge against wrong abstractions.

This promises exponential shrinkage of the effective search space under repeated, correct abstractions, while providing recovery via branching when an abstraction is wrong.

---

## Background and baseline
- **Task**: Given a few input/output grid pairs and a test input, produce the test output.
- **Non‑compositional baseline**: Search a function $f: G \to G$ within a program/hypothesis space so that $f(x_i) = y_i$ (up to a task‑specific notion of agreement), then apply $f(x_{test})$.
- We treat this as **program search** (e.g., enumerative DSL, constraint solving, LLM‑guided search, or neural policies producing edits/sequences).

---

## Formalization
- **G**: concrete domain of grids (finite palettes, shapes, positions).
- **A**: abstract domain with **embedding** $e: A \to G$.
- **Mapping** of examples: for each training pair $(x, y)$ and test input $x_{test}$, find **abstract examples** $(x_A, y_A)$ and $x_{A,test}$ s.t. $e(x_A) \approx x$, $e(y_A) \approx y$. ("≈" denotes the task's agreement relation.)
- **Abstract solver**: find $f_A: A \to A$ with $f_A(x_A) = y_A$ then predict $y_{A,test} = f_A(x_{A,test})$, and output $e(y_{A,test})$.

**Existence of a consistent mapping** (and its simplicity) is **evidence** that the abstraction is probably correct for the task.

---

## Scope: what this document *does not* prescribe
We **intentionally do not specify how abstractions are obtained**. This README defines the problem, interfaces, and evaluation framing only. Possible mechanisms—enumeration from a fixed library, LLM‑guided proposal, meta‑learned proposers, or other strategies—are **left for future work**.

- A fixed **library of abstractions** can be used as a temporary scaffold, but it limits expressivity to what is baked in.
- The intended direction is **test‑time discovery of abstractions for each puzzle**, i.e., searching the abstraction hypothesis space once the instance is known.
- The orchestrator treats abstraction discovery itself as **search**, using evidence scores and branching to remain recoverable.

---

## Examples of abstractions (A) and embeddings (e)
*Illustrative only.* These examples communicate the **interface** (A, e, mappings) and are **not** a prescribed library.
- **Palette reduction**: a color never appears → A is G without that color; $e$ is identity on remaining colors.
- **Object graph**: A contains connected components and relations (adjacency, symmetry); $e$ renders objects to pixels.
- **Downsampling / tiling**: A compresses blocks (e.g., 2×2 to 1×1) with an invertible render back to G.
- **Symmetry quotienting**: A factors out rotations/reflections; e re‑applies a chosen canonical pose.
- **Masking irrelevant regions**: A excludes proven‑irrelevant cells; e fills them via a carry‑over rule.

Abstractions can be **parametric** (e.g., which color is absent, which symmetry) and **composed**.

---

## Abstraction–refinement tree
We build a **tree** where each node is a hypothesis:
```
Node = {
  A: abstract domain,
  e: embedding A→G,
  m: mapping of concrete examples/test into A,
  S_A: solver on A,
  score: evidence that (A, e, m) is valid for this task,
}
```
Edges represent **further abstraction** (A → A₂) or **refinement choices** (e.g., which symmetry, which palette subset). We conduct **bounded best‑first / beam search** over this tree. Leaves produce candidate predictions in G via $e$.

### Evidence and scoring
- **Training consistency**: how perfectly the abstract hypothesis explains $(x_i,y_i)$ after embedding.
- **Compression/MDL**: shorter descriptions of A, e, and $f_A$ score higher.
- **Stability checks**: invariants (counts, symmetries) preserved across examples.
- **Robustness**: small perturbations to examples don’t break mapping.
- **Prior**: prefer low‑capacity abstractions first.

### Selection and recovery
- Maintain a **K‑best frontier**; generate multiple candidate outputs.
- **Vote** (majority, weighted by score) or **validate** using auxiliary constraints (e.g., object counts, boundary rules) to select a final output.
- **Fallback**: if abstractions underperform, revert to the baseline non‑compositional solver.

---

## Algorithms (sketch)

### Orchestrator pseudocode
```python
 def solve_arc(task):
     frontier = init_frontier_with_baseline(task)
     while budget_not_exceeded():
         node = pop_best(frontier)
         if is_goal(node):
             emit(node.predict())
         for child in expand(node):  # propose/refine abstractions
             if consistent(child, task.train):
                 child.score = evidence(child, task.train)
                 frontier.push(child)
     return select_among_emitted(voting=True, constraints=True)
```

### Expand(node): proposing abstractions
This document **deliberately leaves the proposal mechanism unspecified**. In the envisioned research track, **abstraction discovery occurs at test time** as a search over hypotheses guided by evidence (training consistency, MDL, priors). Implementations may explore different mechanisms in subsequent work.

### Solvers on A
- Lightweight program synthesis / enumerative DSL over A.
- Constraint solving (ILP/SAT) on object relations.
- Learned policies that operate on A’s symbols/objects instead of pixels.

---

## Context & Related Work
This project touches several neighboring lines of work; we aim to interoperate with (not replace) them. Importantly, we **avoid committing to a fixed abstraction library** and instead emphasize **instance‑wise, test‑time adaptation**—aligning with trends in test‑time compute and structured search.

### ARC‑AGI benchmark and community
- **ARC‑AGI** frames reasoning as generalization from few examples under human‑centric priors. Our setup follows ARC’s spirit (program search and abstraction) and is compatible with ARC‑AGI‑1/2 evaluation harnesses and emerging agentic variants.
- Community reports emphasize **program synthesis** and **neurosymbolic** hybrids as promising directions; our abstraction–refinement layer is designed to sit *above* whichever concrete solver you use.

### Abstraction–refinement and formal methods
- Our search over abstraction hypotheses is inspired by **CEGAR** from model checking: maintain a small abstract model and refine when evidence demands it.
- Differences: ARC lacks ground‑truth counterexamples at test time, so we replace counterexamples with **evidence scores** and hedge via **branching + voting** across multiple abstractions.

### Program synthesis & library learning
- **Enumerative/constraint‑based synthesis** and DSL design remain strong baselines. In our framing, these operate **within A** once an abstraction is posited.
- **Library learning** (e.g., DreamCoder) shows benefits of learned abstractions; in our agenda, such libraries are optional probes—not core dependencies.

### Test‑time compute & reasoning search with LLMs
- **CoT/Self‑Consistency, Tree/Graph of Thoughts, Program‑of‑Thoughts** can be used *inside* A. Our contribution is **shaping the state space** before invoking these methods, and scoring hypotheses to allocate compute.

---

## Guarantees and limits
- **Exponential shrinkage intuition**: if each correct abstraction reduces branching by factor $r<1$, depth $d$ yields $r^d$ effective search.
- **Failure mode**: a **wrong abstraction** (e.g., assuming a color absent that appears in the test output) can trap a single path. Hence **branching**, **voting**, and **fallback** are essential.

---

## Implementation plan
**Phase 0 – Baseline**
- Provide a simple ARC program‑search baseline in G for comparison.

**Phase 1 – Interfaces & scaffolding**
- Define types for Grid (G), AbstractDomain (A), Embedding (e), Mapping (m), AbstractSolver, Node, and scoring hooks. No commitment to any particular abstraction acquisition strategy.

**Phase 2 – Orchestrator skeleton**
- Implement beam/best‑first search over **abstraction hypotheses**. Provide a pluggable `propose(node)` hook that can be implemented later (enumerative, LLM‑guided, learned, etc.).

**Phase 3 – Evaluation harness**
- Metrics: success rate, compute budget, nodes expanded, abstraction depth, solves unique to compositional mode. Ablations comparing with/without branching, voting, fallback.

**(Optional) Phase X – Demo probes**
- Small, *non‑authoritative* example proposers for experimentation, kept separate from the core.

---

## Repository layout (proposed)
```
compositional-arc/
  README.md                # this file
  data/
    arc/                   # ARC tasks (train/dev)
  core/
    domains.py             # G, A, e, m interfaces & utilities
    hypotheses/            # optional: task-local hypothesis instances (empty by default)
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

## Interfaces (typing sketch)
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

## Examples (intuition only)
- **Missing color**: detect unused color → drop it in A → solve in smaller palette → embed back (identity on remaining colors).
- **Object duplication with symmetry**: factor out rotation in A, reason over object graphs, then render with chosen canonical pose.
- **Blockwise scaling**: downsample to detect macro‑pattern, synthesize macro logic, upsample back.

---

## Open questions
- How best to **discover** abstractions (m) robustly from 1–5 examples?
- What **priors** on abstraction families work across ARC tasks?
- How to quantify **evidence**: MDL vs. Bayesian model evidence vs. discriminative scoring?
- How to combine **LLM‑proposed** abstractions with verifiable checks?

---

## Getting started (placeholder)
```bash
pip install -e .
python experiments/eval.py --config configs/baseline.yaml
python experiments/eval.py --config configs/comp.yaml
```

---

## Further reading & references (selection)
- Chollet, F. (2019). *On the Measure of Intelligence* — introduces ARC.
- ARC Prize (2024–2025). *Technical reports & blogs on ARC‑AGI‑1/2, evaluation, and approaches.*
- Clarke, Grumberg, Jha, Lu, Veith (2000–2003). *Counterexample‑Guided Abstraction Refinement (CEGAR).* Foundational abstraction–refinement.
- Ellis et al. (2020–2021). *DreamCoder.* Library learning for program synthesis; learning abstractions to speed search.
- Wang et al. (2022). *Self‑Consistency improves Chain‑of‑Thought.* Voting across sampled reasoning.
- Yao et al. (2023). *Tree of Thoughts.* Structured search over partial solutions.
- Besta et al. (2023–2024). *Graph of Thoughts; Demystifying Chains, Trees, and Graphs of Thoughts.* Generalizes structured reasoning search.
- Chen et al. (2022/2023). *Program‑of‑Thoughts (PoT).* Delegate computation to executors; a natural solver for our abstract domains.
- Snell et al. (2024/2025). *Scaling Test‑Time Compute.* Adaptive compute allocation and verifier‑guided search.
- Selected ARC‑specific methods (2024–2025): LLM‑guided output search and validators; ARC‑AGI‑2 design notes and community reviews.

## Contributing
Issues and PRs welcome. Please include a minimal failing example or task ID when reporting bugs.

## License
MIT (tentative).

