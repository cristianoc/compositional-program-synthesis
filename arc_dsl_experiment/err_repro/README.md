# ERR (Extract–Recolor–Rotate) Repro Bundle

This bundle reproduces the "output is one object, recolored, rotated" invariant experiments.

## Contents
- `dsl.py` — your original DSL (used for A1/A2 and helpers)
- `invariant_ops.py` — invariant operator built on top of A1/A2
- `scripts/`
  - `01_generate_datasets.py` — builds DS1 (most_frequent,k=1) and DS2 (least_frequent,k=2)
  - `02_print_datasets.py` — prints datasets as raw numbers and writes JSON dumps
  - `03_eval_invariant_family.py` — evaluates invariant configs on a dataset
  - `04_measure_with_meta.py` — measures global vs abstraction-guided search on a 50/50 split
- `data/`
  - `invariant_ds_mf_k1.npz`, `invariant_ds_lf_k2.npz` — tailored datasets
  - `DS1_mostfreq_k1.json`, `DS2_leastfreq_k2.json` — numeric dumps
  - `abstraction_meta_metrics.json` — global vs abstraction metrics (from 04)
  - `invariant_report.json` — small earlier report
  - optional: `challenging_metrics.txt`, `challenging_metrics.json` — from `dsl.run()` if present
  - sample npy triplets: `x_0.npy`, `y_true_0.npy`, `y_pred_0.npy`, `x_1.npy`, `y_true_1.npy`, `y_pred_1.npy`

## How to run
Use Python 3 with NumPy and Matplotlib.

1. Generate datasets:
   ```bash
   python scripts/01_generate_datasets.py
   ```
2. Print datasets (to console) and write JSON:
   ```bash
   python scripts/02_print_datasets.py
   ```
3. Evaluate invariant family on DS1 or DS2:
   ```bash
   python scripts/03_eval_invariant_family.py --dataset data/invariant_ds_mf_k1.npz
   ```
4. Measure global vs. abstractions with a 50/50 split:
   ```bash
   python scripts/04_measure_with_meta.py --dataset data/invariant_ds_mf_k1.npz
   python scripts/04_measure_with_meta.py --dataset data/invariant_ds_lf_k2.npz
   ```

The expected numbers for the provided datasets:
- DS1: G=704, A2=192, A2+color=96, A2+color+rot=24; Reductions ~72.7%, 50.0%, 75.0%, 96.6%
- DS2: G=816, A2=192, A2+color=96, A2+color+rot=24; Reductions ~76.5%, 50.0%, 75.0%, 97.1%
