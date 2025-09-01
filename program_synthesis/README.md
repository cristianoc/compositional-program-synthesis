# Compositional Program Synthesis

This directory contains the implementation and documentation for compositional program synthesis via two abstractions A and A⁺.

## Files

- **`compositional_synthesis.pdf`** - Main research paper (compiled LaTeX)
- **`compositional_synthesis.tex`** - LaTeX source for the paper
- **`scaling.py`** - Python implementation of the scaling experiments
- **`nodes_vs_k_scaling.png`** - Visualization of node exploration vs. coupling parameter K
- **`speedup_vs_k.png`** - Speedup analysis across different configurations

## Quick Start

### Viewing the Paper
Open `compositional_synthesis.pdf` to read the full research note.

### Running Experiments
```bash
python scaling.py
```

### Compiling LaTeX
```bash
pdflatex compositional_synthesis.tex
```

## Method Overview

The approach uses two abstraction levels:
1. **A (Cross-free factorization)**: Solve coordinates independently, ignoring cross-operations
2. **A⁺ (Interface refinement)**: Add minimal cross-operation interfaces to restore necessary couplings

This achieves 4-59× fewer explored nodes and 8-184× speedup over global search methods.
