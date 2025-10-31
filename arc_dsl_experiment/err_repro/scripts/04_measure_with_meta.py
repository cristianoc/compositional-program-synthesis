
import numpy as np, argparse, importlib.util, sys, json
from pathlib import Path

# load core dsl and build meta functions inline
spec = importlib.util.spec_from_file_location("dsl_core", str(Path(__file__).resolve().parent.parent / "dsl.py"))
dsl = importlib.util.module_from_spec(spec); sys.modules["dsl_core"] = dsl
spec.loader.exec_module(dsl)

def A1A2_with_meta(x):
    x_hat, m1 = dsl.alpha1_palette(x)
    _,   m2 = dsl.alpha2_objorder(x_hat)
    return x_hat, {
        "orig_for_can": m1["orig_for_can"],
        "can_for_orig": m1["can_for_orig"],
        "order": m2["order"],
    }

def apply_invariant_using_meta(x_hat, meta, comp_index, recolor_rule, k):
    comps = meta["order"]
    comp = comps[comp_index % len(comps)] if comps else None
    if comp is None: return np.zeros((1,1), dtype=int)
    (r0,c0,r1,c1) = comp["bbox"]; H,W = r1-r0+1, c1-c0+1
    obj = np.zeros((H,W), dtype=int)
    for (r,c) in comp["pixels"]:
        obj[r-r0, c-c0] = int(x_hat[r,c])
    # recolor
    hist = dsl.color_hist_nonzero(x_hat)
    if recolor_rule=="most_frequent":
        m = max(hist.values()); cand=[c for c,v in hist.items() if v==m]; target=min(cand)
    elif recolor_rule=="least_frequent":
        m = min(hist.values()); cand=[c for c,v in hist.items() if v==m]; target=min(cand)
    else: target=1
    obj[obj!=0]=target
    y_hat = np.rot90(obj, k%4)
    inv = meta["orig_for_can"]
    y = np.zeros_like(y_hat)
    for r in range(y_hat.shape[0]):
        for c in range(y_hat.shape[1]):
            v = int(y_hat[r,c]); y[r,c] = 0 if v==0 else inv.get(v, v)
    return y

def load_pairs(npz_path):
    data = np.load(npz_path)
    idxs = sorted(int(k[1:]) for k in data if k.startswith("x"))
    return [(data[f"x{i}"], data[f"y{i}"]) for i in idxs]

COLOR_RULES = ["most_frequent","least_frequent"]
ROTATIONS = [0,1,2,3]

def explore_counts(x):
    x_hat, meta = A1A2_with_meta(x)
    ncomp = len(meta["order"]) if meta["order"] else 0
    R0 = ncomp * len(COLOR_RULES) * len(ROTATIONS)
    R1 = len(COLOR_RULES) * len(ROTATIONS)
    R2 = len(ROTATIONS)
    R3 = 1
    return R0,R1,R2,R3

def solve_with_config(x, rr, kk):
    x_hat, meta = A1A2_with_meta(x)
    return apply_invariant_using_meta(x_hat, meta, comp_index=0, recolor_rule=rr, k=kk)

def infer_best_config(train_pairs):
    best=None; best_ok=-1
    for rr in COLOR_RULES:
        for kk in ROTATIONS:
            ok=sum(int(np.array_equal(solve_with_config(x, rr, kk), y)) for (x,y) in train_pairs)
            if ok>best_ok: best_ok=ok; best=(rr,kk)
    return best, best_ok

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--train_frac", type=float, default=0.5)
    args = parser.parse_args()
    pairs = load_pairs(args.dataset)
    n=len(pairs); n_train=int(round(n*args.train_frac))
    train = pairs[:n_train]; test = pairs[n_train:]
    # counts
    R0=R1=R2=R3=0
    for (x,_) in pairs:
        r0,r1,r2,r3 = explore_counts(x)
        R0+=r0; R1+=r1; R2+=r2; R3+=r3
    # learn config
    (rr,kk), ok_train = infer_best_config(train)
    ok_test = sum(int(np.array_equal(solve_with_config(x, rr, kk), y)) for (x,y) in test)
    acc_train = ok_train/len(train) if train else 0.0
    acc_test  = ok_test/len(test) if test else 0.0
    # expected tries (avg per pair)
    def meanE(Ms): return float(np.mean([(m+1)/2 for m in Ms]))
    Ms=[explore_counts(x)[0] for (x,_) in pairs]; E_G=meanE(Ms)
    Ms=[explore_counts(x)[1] for (x,_) in pairs]; E_A2=meanE(Ms)
    Ms=[explore_counts(x)[2] for (x,_) in pairs]; E_A2c=meanE(Ms)
    Ms=[explore_counts(x)[3] for (x,_) in pairs]; E_A2cr=meanE(Ms)
    out = {
        "dataset": Path(args.dataset).name,
        "n_pairs": n,
        "train_pairs": len(train),
        "test_pairs": len(test),
        "learned_recolor_rule": rr,
        "learned_rotation_k": kk,
        "train_accuracy": acc_train,
        "test_accuracy": acc_test,
        "total_candidates": {"G":R0,"A2":R1,"A2+color":R2,"A2+color+rot":R3},
        "expected_tries": {"G":E_G,"A2":E_A2,"A2+color":E_A2c,"A2+color+rot":E_A2cr},
        "reductions": {
            "G→A2": 1 - R1/R0,
            "A2→A2+color": 1 - R2/R1,
            "A2+color→A2+color+rot": 1 - R3/R2,
            "G→A2+color+rot": 1 - R3/R0
        }
    }
    print(json.dumps(out, indent=2))
    # save
    out_path = Path(__file__).resolve().parent.parent / "data" / f"metrics_{Path(args.dataset).stem}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print("Saved metrics to", out_path)
