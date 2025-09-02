
import numpy as np, argparse, importlib.util, sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("invariant_ops", str(Path(__file__).resolve().parent.parent / "invariant_ops.py"))
inv = importlib.util.module_from_spec(spec); sys.modules["invariant_ops"] = inv
spec.loader.exec_module(inv)

def load_pairs(npz_path):
    data = np.load(npz_path)
    idxs = sorted(int(k[1:]) for k in data if k.startswith("x"))
    return [(data[f"x{i}"], data[f"y{i}"]) for i in idxs]

def eval_family(pairs):
    configs=[("most_frequent",0),("most_frequent",1),("most_frequent",2),("most_frequent",3),
             ("least_frequent",0),("least_frequent",1),("least_frequent",2),("least_frequent",3)]
    for rr,k in configs:
        ok=sum(int(np.array_equal(inv.invariant_extract_recolor_rotate(x, 'canonical_first', rr, k, True), y))
               for (x,y) in pairs)
        print(f"rule={rr:14s}  k={k}  matches={ok}/{len(pairs)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()
    pairs = load_pairs(args.dataset)
    eval_family(pairs)
