## Compositional Program Synthesis via Abstraction–Refinement

> A new take on compositional program synthesis: repeatedly solve simpler abstractions of the task, then refine back to concrete domains. Think CEGAR but for program synthesis from examples. Two toy instantiations on ARC-AGI and inductive programming show meaningful speedups.

### Why
**Goal:** Use composable abstractions to obtain exponential reduction of the state space.

**Key insight:** Instead of attacking the whole search space at once, lift problems to abstract domains, solve there, then embed back. Branching + voting handles abstraction failures.

**Domains:** ARC-AGI grid puzzles and inductive programming.

---

### How it works
Starting from a concrete problem domain **G** (grids, programs, structured data):

1. **Lift** problems to abstract domains **A** with embedding `e: A → G`  
2. **Solve** in the simpler abstract space  
3. **Embed** back to concrete domain **G** via `e`  
4. **Refine** recursively `A → A₂ → …` when needed  
5. **Branch + vote** across abstraction hypotheses to handle failures

**Result:** Exponential search space reduction when abstractions are good; graceful recovery when they fail.

**Key insight:** An abstraction is plausible when there exists a mapping from the training examples and test input into the abstract domain. The existence of this mapping indicates the abstraction is consistent with the task.

*Examples:*
- *ARC:* if every training grid is 7×12, there exists a mapping of the examples and test input into the abstract shape class “7×12”; that existence indicates the size abstraction is consistent with the task.
- *Inductive:* if all training pairs satisfy “even-in ⇒ even-out,” there exists a mapping of each pair (x,y) into a parity abstraction that records (parity(x), parity(y)) ∈ {0,1}²; that existence indicates a parity abstraction is consistent with the task property.

Note: Consistency is necessary but not sufficient. Different abstractions can admit such mappings on the same training pairs, and a consistent abstraction can still fail if its constraint does not hold for the (unknown) test output (e.g., train grids are 7×12 but the test grid is 9×12; train outputs are even but the test output is odd).

---

### Results

Two toy instantiations show meaningful speedups:

#### 1. ARC-AGI Grid Puzzles
Eliminates symmetries in grid puzzles through two abstraction levels:
- **A1**: Palette canonicalization (quotient out color symmetries)
- **A2**: Object ordering canonicalization (quotient out enumeration symmetries)

**Result**: 2404→2 programs (-99.917%), 138× speedup.

#### 2. Inductive Programming (Flash Fill style)
Two-phase approach for program synthesis on product domains:
- **Phase 1**: Cross-free factorization (solve coordinates independently) 
- **Phase 2**: Interface refinement (add minimal cross-operation coupling)

**Result**: 4-59× fewer nodes, 8-184× speedup over global search.

These results suggest the approach is worth exploring further.

---

### Background & baseline
- **Task:** given a few input/output example pairs `(x_i, y_i)` and a test input `x_test`, synthesize a program that produces the test output `y_test`.  
- **Non-compositional baseline:** search for a function `f: G → G` (via enumerative DSL, constraints, LLM-guided search, or neural edit policies) such that `f(x_i) = y_i` for all training examples, then apply `f(x_test)`.

---

### Scope (what this document does **not** prescribe)
How to **obtain** abstractions is intentionally left open. This README defines the problem, interfaces, and evaluation framing only. Possible mechanisms—enumeration from a fixed library, LLM-guided proposals, meta-learned proposers, etc.—can vary by domain and application.

- A fixed library can bootstrap experiments but limits expressivity.  
- The intended direction is **test-time discovery** of abstractions per task, i.e., search over abstraction hypotheses with evidence-guided branching.  
- The orchestrator treats abstraction discovery itself as search.

---

### Examples

**Grid puzzles:**
- **Missing color**: detect unused color → drop it in `A` → solve in smaller palette → embed back
- **Object symmetry**: factor out rotation/reflection → reason over object graphs → render with canonical pose
- **Downsampling**: compress blocks (2×2→1×1) → synthesize macro logic → upsample back

**Programming:**
- **Type abstraction**: ignore data types → work with shape/structure → instantiate concrete types
- **Cross-operation factorization**: solve coordinates independently → add minimal coupling → full program
- **String processing**: factor out character-level ops → reason over tokens → expand to characters

Abstractions can be parametric and composed into hierarchies.

---

### Abstraction–refinement tree
Each node is a task hypothesis:

```text
Node = {
  A: abstract domain,
  e: embedding A → G,
  m: mapping of concrete examples/test into A,
  S_A: solver on A,
  score: evidence that (A, e, m) is valid for this task,
}
```

Edges represent further abstraction `A → A₂` or refinement choices (e.g., symmetry, palette subset). We run bounded best-first / beam search over the tree. Leaves emit candidate predictions in **G** via `e`.

#### Evidence & scoring
- **Training consistency:** how well the abstract hypothesis explains `(x_i, y_i)` after embedding.  
- **Compression / MDL:** shorter descriptions of `A`, `e`, and `S_A` score higher.  
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
def solve_task(task):
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

**CEGAR connection:** Think CEGAR (Counter-Example Guided Abstraction Refinement) but for program synthesis from examples. Instead of counterexamples, we use evidence scores and branching + voting to handle abstraction failures.

**Program synthesis:** Compatible with existing synthesis approaches—the abstraction layer sits above your concrete solver, whether that's enumerative search, constraints, or LLMs.



---

### Guarantees & limits
- **Shrinkage intuition:** if each correct abstraction reduces search space by factor `r < 1`, depth `d` yields `r^d` relative search cost.  
- **Failure mode:** a wrong abstraction (e.g., assuming an element is absent that appears in the test output) can trap a single path—hence **branching, voting, and fallback** are essential.

---

### Repository layout (current)
```text
compositional-program-synthesis/
  README.md                           # this file
  arc_dsl_experiment/
    dsl.py                           # domain-specific language implementation
    challenging_metrics.json        # metrics data for challenging tasks
    challenging_metrics.txt         # human-readable metrics summary
    compositional_abstractions.tex  # LaTeX paper: "Compositional Abstractions for ARC-Style Tasks"
    compositional_abstractions.pdf  # compiled paper
    README.md                       # documentation for DSL experiments
  program_synthesis/
    scaling.py                      # scaling analysis and performance evaluation
    compositional_synthesis.tex    # LaTeX paper: "Compositional Synthesis via Exact Interface Abstraction"
    compositional_synthesis.pdf    # compiled paper
    nodes_vs_k_scaling.png         # visualization: node count vs. parameter k
    speedup_vs_k.png              # visualization: speedup analysis
    README.md                     # documentation for synthesis experiments
```
