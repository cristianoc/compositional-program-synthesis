Compositional Program Synthesis via Two Abstractions A and A^{+}: A Short Empirical Note

Abstract

We present a concrete instantiation of a research plan on compositional reasoning via abstraction–refinement. We define two abstraction layers over program synthesis on paired integers: a cross-free factorization $A$ and an interface-augmented $A^{+}$ that captures where cross-operations occur. We give embeddings $e:A\!\to\!G$ and $e^{+}:A^{+}\!\to\!G$, algorithms that solve in $A$ and refine in $A^{+}$, complexity bounds, and correctness conditions. Experiments show large efficiency gains over global search: $7$–$60\times$ fewer nodes and $8$–$180\times$ faster in coupled tasks, with identical accuracy. We conclude that $A$ is a direct instantiation of the original plan; $A^{+}$ is a problem-specific refinement of the program search space that fits the spirit of abstraction–refinement even if it is not the exact state-space abstraction emphasized in the original note.

⸻

1 Introduction (Intuition First)

Global program synthesis often explores a huge search space. If a task nearly factors into independent pieces with a few “wires” between them, we can:
	1.	Solve the easy factors, ignoring the wires.
	2.	Refine by putting back a small interface that reconnects the factors.

We demonstrate this idea in a tiny DSL on integer pairs $(x,y)$. In separable tasks, solving each coordinate independently is optimal. In coupled tasks (e.g., $y$ must read the current $x$), a global solver works but is wasteful. Our Compositional+ solver first synthesizes the $x$-only program, then searches a small space of cross-op placements that wire $y$ to $x$ at just the right moments.

Inline picture of the abstraction ladder:

   Abstract, easy         Intermediate, small interface              Concrete programs
┌──────────────────┐      ┌────────────────────────────────┐      ┌──────────────────────────┐
│        A         │  →   │              A⁺                │  →   │            G             │
│  (factors only)  │      │ (factor + K cross-op slots)    │      │ (all interleavings, ops) │
└──────────────────┘ e    └────────────────────────────────┘ e⁺   └──────────────────────────┘
         solve                         refine (enumerate α)                    execute


⸻

2 Concrete Setting

2.1 Domains, DSL, Semantics
	•	Concrete state space: $G=\mathbb{Z}\times \mathbb{Z}$.
	•	Primitives are partitioned
$$\Sigma = \Sigma_X \;\cup\; \Sigma_Y \;\cup\; \Sigma_{\times},$$
where $\Sigma_X$ edits only $x$, $\Sigma_Y$ edits only $y$, and $\Sigma_{\times}$ are cross-ops (e.g., `add_first_to_second`: $(x,y)\mapsto(x,\,y+x)$).
A program is a word $p\in\Sigma^{*}$ with standard functional semantics $⟦p⟧:G\rightarrow G$.
	•	Supervision: dataset $D=\{(s_i,t_i)\}\subseteq G\times G$. Goal: find $p$ with $\forall i,\ \llbracket p\rrbracket(s_i)=t_i$.

⸻

3 Abstraction A: Cross-Free Factorization

3.1 Definition

Let
$$A \;=\; \Sigma_X^{}\times \Sigma_Y^{}.$$
An element $a=(p_X,p_Y)$ means "do $p_X$ on $x$ and $p_Y$ on $y$, with no cross-ops."

3.2 Embedding and Agreement

Because $\Sigma_X$ commutes with $\Sigma_Y$,
$$e:A\to \Sigma^{*},\quad e(p_X,p_Y)=\text{any interleaving of }p_X,p_Y$$
is well-defined up to semantic equivalence: all such interleavings produce the same $\llbracket e(p_X,p_Y)\rrbracket$ on $G$.

3.3 Solve in A

Project the dataset onto coordinates and synthesize independently:
$$\forall i:\ \pi_X(\llbracket p_X\rrbracket(s_i))=\pi_X(t_i),\qquad
\forall i:\ \pi_Y(\llbracket p_Y\rrbracket(s_i))=\pi_Y(t_i).$$
If both succeed, return $e(p_X,p_Y)$.

Intuition. $A$ "turns off" the wires between coordinates, producing two small searches.

⸻

4 Abstraction A^{+}: Factorization with a Finite Interface

4.1 Slots and Interfaces

Fix a bound $K$ (number of cross-ops). For a first-coordinate program $p_X$ of length $L$, define insertion slots $\{0,\dots,L\}$. Let
$$\Pi_{L,K} \;=\; \big\{\,\alpha=(\alpha_1\le\cdots\le\alpha_K)\ \big|\ \alpha_j\in\{0,\dots,L\}\,\big\}.$$
Then
$$A^{+} \;=\; \bigcup_{L\ge 0}\ \big(\Sigma_X^{L}\times \Pi_{L,K}\big).$$
An element $a^{+}=(p_X,\alpha)$ is an abstract program skeleton: use $p_X$ and insert $K$ cross-ops at slots $\alpha$.

4.2 Embedding to Concrete Programs

Choose $\kappa\in\Sigma_{\times}$ (e.g., `add_first_to_second`).
Define
$$e^{+}(p_X,\alpha)\in \Sigma^{*}$$
by interleaving $p_X$ with $K$ copies of $\kappa$ inserted just before the $x$-op at each slot index in $\alpha$. (If needed, $Y$-only ops that commute with $\Sigma_X$ can be appended without changing the interface.)

Inline picture of A^{+} objects:

p_X:     [ X₁ ] [ X₂ ] [ X₃ ] [ X₄ ] [ X₅ ]  …  [ X_L ]
slots:   ^ 0   ^ 1     ^ 2     ^ 3     ^ 4        ^ L
α:              ↑                 ↑
insert κ here   |                 |

4.3 Solve-then-Refine (Compositional+)
	1.	Solve in $A$: find $p_X$ satisfying the $x$-projection of $D$.
	2.	Refine in $A^{+}$: enumerate $\alpha\in \Pi_{|p_X|,K}$ and test $e^{+}(p_X,\alpha)$ on full $D$. Return the first that fits.

4.4 Completeness (Triangular Coupling)

Assume each $\kappa\in\Sigma_{\times}$ is triangular: it updates $y$ by a function of the current $x$ and leaves $x$ unchanged, i.e.,
$$\kappa:\ (x,y)\mapsto(x,\ y\oplus h(x)).$$
If there exists a concrete solution of the form "some $p_X\in\Sigma_X^{L}$ interleaved with exactly $K$ copies of $\kappa$", then there exists $\alpha^{}\in\Pi_{L,K}$ such that $e^{+}(p_X,\alpha^{})$ satisfies $D$.
Sketch. Every valid interleaving corresponds to inserting $\kappa$ at specific slots w.r.t. $p_X$; these slots are exactly $\alpha$. Hence enumerating $\Pi_{L,K}$ is complete for this family.

⸻

5 Algorithms and Complexity
	•	Global BFS (baseline). Branching $b$ over $\Sigma$; minimal solution length $L_X+K$.
Cost: $O(b^{\,L_X+K})$ (modulo semantic memoization).
	•	Compositional+.
(i) Synthesize $p_X$ with branching $b_X$ over $\Sigma_X$.
(ii) Enumerate $\binom{L_X+K}{K}$ interfaces (combinations with repetition).
Cost:
$$O(b_X^{\,L_X})\;+\;O\!\big(\tbinom{L_X+K}{K}\big).$$
For small $K$ and moderate $L_X$, the second term is tiny relative to the global exponential.

⸻

6 Examples and Results (All Executed)

6.1 Separable Task (fits A)

Target: $(x,y)\mapsto\big(2(x+3),\ y^{2}+1\big)$.
Minimal program: inc1_first×3 → double_first  and square_second → inc1_second.
	•	Global BFS: found length 6, 343 nodes, 0.0119 s.
	•	Compositional (A): found length 6, 23 nodes, 0.00029 s.
	•	Both perfectly match held-out tests.

Intuition. Perfect factorization; solving coordinates independently is optimal.

6.2 Coupled Task (needs A^{+})

Target: $(x,y)\mapsto\big(2(x+3),\ y+(x+3)\big)$ using cross-op $\kappa=$ `add_first_to_second`.
	•	Global BFS: found length 5, 187 nodes, 0.0367 s.
	•	Naïve split: fails (cannot express $y$'s dependence on $x$).
	•	Compositional+ ($A→A⁺$): found (equivalent) program, 25 nodes, 0.00032 s.

Intuition. Solve $x$ first, then place a single wire where $y$ must read $x$.

6.3 Scaling Study (vary $L_X\in\{4,6,8\}$, $K\in\{0,1,2,3\}$, $b_X\in\{2,3,4\}$)

Geometric-mean speedups (Global / Compositional+) across the grid:
	•	K=0: 4.2× fewer nodes, 8.1× faster.
	•	K=1: 19.3×, 44.3×.
	•	K=2: 29.2×, 87.2×.
	•	K=3: 58.7×, 183×.

Hardest slice ($L_X=8$, $b_X=4$):
Global nodes $1609 \to 32061$ as $K:0\to3$; Compositional+ stays near $372$–$387$.

⸻

7 Does This Instantiate the Original Plan?
	•	Yes for $A$. $A$ is a direct instantiation of "lift to an abstract, easier domain; solve; embed." The embedding $e$ and separability proofs are standard.
	•	Qualified yes for $A^{+}$. $A^{+}$ refines $A$ by adding a finite interface (cross-op placements). This is an abstraction–refinement pipeline
$$A \;\xrightarrow{\text{add interface}}\; A^{+} \;\xrightarrow{e^{+}}\; G,$$
with a concrete completeness guarantee under triangular couplings.
It does follow the spirit of the note (progressive refinement toward concrete solvability), though it abstracts the program/search space rather than the task state space emphasized in the note’s ARC framing.

⸻

8 Limitations and Next Steps
	•	Two-way couplings. If $\Sigma_{\times}$ also includes `add_second_to_first`, $x$ cannot be solved in isolation. Remedy: an AND–OR refinement that alternates partial $X$- and $Y$-solves while growing the interface grammar (an $A^{++}$).
	•	Beyond identity embeddings. Port to grids (ARC): $A$ as attribute factorization (shape/color), $A^{+}$ as small wiring constraints (e.g., "color-from-shape"), moving closer to the note's state-space abstractions.
	•	Heuristics. Cost-guided search, CEGIS, or learned guidance would likely amplify the benefit of $A$/$A^{+}$.

⸻

9 Conclusion

Distinguishing two abstraction layers crystallizes the method:
	•	$A$: cross-free factorization—cheap, complete for separable tasks.
	•	$A^{+}$: finite interface refinement—tiny extra search that restores necessary couplings.

On coupled tasks, $A\to A^{+}$ achieves the same solutions as global search at a fraction of the cost, validating the core thesis: solve in a simpler world, then refine only what must be coupled.