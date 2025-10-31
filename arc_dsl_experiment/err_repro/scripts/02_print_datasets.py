
import numpy as np, argparse, json
from pathlib import Path

np.set_printoptions(threshold=np.inf, linewidth=160)

def load_pairs(npz_path):
    data = np.load(npz_path)
    idxs = sorted(int(k[1:]) for k in data.keys() if k.startswith("x"))
    return [(data[f"x{i}"], data[f"y{i}"]) for i in idxs]

def print_dataset(name, pairs):
    print(f"\n===== {name} =====")
    print(f"num_pairs = {len(pairs)}\n")
    for i,(x,y) in enumerate(pairs):
        print(f"-- {name} Pair {i} — Input (shape={x.shape}) --"); print(x)
        print(f"-- {name} Pair {i} — Output (shape={y.shape}) --"); print(y); print()

if __name__ == "__main__":
    ds_dir = Path(__file__).resolve().parent.parent / "data"
    for fname in ["invariant_ds_mf_k1.npz","invariant_ds_lf_k2.npz"]:
        pairs = load_pairs(ds_dir/fname)
        print_dataset(fname, pairs)
