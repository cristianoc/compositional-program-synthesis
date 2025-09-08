# Compositional Abstractions for ARC-Style Tasks

This directory contains the implementation and documentation for compositional abstractions that eliminate symmetries in ARC-style grid puzzles.

## Files

- **`compositional_abstractions.pdf`** - Main research paper (compiled LaTeX)
- **`compositional_abstractions.tex`** - LaTeX source for the paper
- **`dsl.py`** - Python implementation of the DSL and abstraction experiments
- **`challenging_metrics.json`** - Experimental results in JSON format
- **`challenging_metrics.txt`** - Human-readable experimental results

## Quick Start

### Viewing the Paper
Open `compositional_abstractions.pdf` to read the full research note.

### Running Experiments
```bash
python dsl.py
```

### Compiling LaTeX
```bash
pdflatex compositional_abstractions.tex
```

## Method Overview

The approach uses two compositional abstractions:
1. **A₁ (Palette canonicalization)**: Relabel non-zero colors by decreasing frequency to quotient out palette symmetry
2. **A₂ (Canonical object order)**: Sort connected components by (area, top, left, color) to quotient out object-enumeration symmetry

This achieves dramatic search space reductions:
- **G→A₁**: 2404→172 programs (-92.85%)
- **A₁→A₂**: 172→2 programs (-98.84%)
- **Overall**: 2404→2 programs (-99.917%) with 138× speedup

## Results Summary

| Method | Total Candidates | Valid Programs | Avg Tries to Success | Wall Time (s) |
|--------|------------------|----------------|---------------------|---------------|
| G      | 2404            | 441            | 5.405               | 3.884         |
| A₁     | 172             | 4              | 35.573              | 0.197         |
| A₁→A₂  | 2               | 2              | 1.000               | 0.028         |

The composed abstractions (A₁→A₂) achieve near-zero search cost with 100% validity rate.
