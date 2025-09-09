#!/usr/bin/env python3
"""
Benchmark parallelism for abstraction-holds analysis.

Runs a subset of tasks with different -j settings and reports wall times.
"""

from __future__ import annotations

import time
import multiprocessing as mp
from pathlib import Path
import sys

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.arc_agi_evaluation.run_pattern_analysis_all import analyze_all_tasks, _filter_tasks  # type: ignore
from experiments.driver import discover_all_task_paths


def main():
    # Candidate jobs: 1, physical cores (heuristic: half of logical), logical cores
    logical = mp.cpu_count() or 1
    physical = max(1, logical // 2)
    candidates = []
    for j in (1, physical, logical):
        if j not in candidates:
            candidates.append(j)

    # Build a subset once (default: first 400 tasks across datasets)
    all_paths = discover_all_task_paths()
    all_paths = _filter_tasks(all_paths, dataset="all", split="all")  # type: ignore[arg-type]
    subset = all_paths[:400]
    print(f"Benchmarking with {len(subset)} tasks; candidates: {candidates}")

    timings = {}
    for j in candidates:
        t0 = time.perf_counter()
        # Analyze with no per-task prints and avoid writing files
        analyze_all_tasks(jobs=j, limit=len(subset), quiet=True)  # type: ignore[arg-type]
        t1 = time.perf_counter()
        timings[j] = t1 - t0
        print(f"-j {j}: {timings[j]:.2f}s")

    # Recommend best
    best_j = min(timings, key=timings.get)
    print("\nRecommendation:")
    print(f"Use -j {best_j} (fastest on this benchmark)")


if __name__ == "__main__":
    main()

