
import numpy as np
import importlib.util, sys

# import dsl.py as dsl_mod
spec = importlib.util.spec_from_file_location("dsl_mod", "/mnt/data/dsl.py")
dsl_mod = importlib.util.module_from_spec(spec)
sys.modules["dsl_mod"] = dsl_mod
spec.loader.exec_module(dsl_mod)

def crop_component_grid(g, comp):
    (r0,c0,r1,c1) = comp["bbox"]
    H, W = r1-r0+1, c1-c0+1
    out = np.zeros((H,W), dtype=int)
    for (r,c) in comp["pixels"]:
        out[r-r0, c-c0] = int(g[r,c])
    return out

def recolor_uniform(grid, color):
    out = grid.copy()
    out[out!=0] = int(color)
    return out

def rotate_k(grid, k):
    return np.rot90(grid, k=k)

def most_frequent_color_nonzero(g):
    hist = dsl_mod.color_hist_nonzero(g)
    if not hist: return 0
    maxc = max(hist.values())
    cands = [c for c,v in hist.items() if v==maxc]
    return min(cands)

def least_frequent_color_nonzero(g):
    hist = dsl_mod.color_hist_nonzero(g)
    if not hist: return 0
    minc = min(hist.values())
    cands = [c for c,v in hist.items() if v==minc]
    return min(cands)

def select_component_canonical_first(x_hat):
    _, meta = dsl_mod.alpha2_objorder(x_hat)
    order = meta["order"]
    return order[0] if order else None

def invariant_extract_recolor_rotate(x, select_rule="canonical_first", recolor_rule="most_frequent", k=1, map_back_to_original=True):
    x_hat, meta1 = dsl_mod.alpha1_palette(x)
    x_hat2, meta2 = dsl_mod.alpha2_objorder(x_hat)
    comp = select_component_canonical_first(x_hat2) if select_rule=="canonical_first" else None
    if comp is None:
        return np.zeros((1,1), dtype=int)
    obj = crop_component_grid(x_hat2, comp)
    if recolor_rule=="most_frequent":
        target = most_frequent_color_nonzero(x_hat2)
    elif recolor_rule=="least_frequent":
        target = least_frequent_color_nonzero(x_hat2)
    else:
        target = 1
    y = recolor_uniform(obj, target)
    y = rotate_k(y, k % 4)
    if map_back_to_original:
        inv = meta1["orig_for_can"]
        out = np.zeros_like(y)
        for r in range(y.shape[0]):
            for c in range(y.shape[1]):
                v = int(y[r,c])
                out[r,c] = 0 if v==0 else inv.get(v, v)
        return out
    return y

if __name__ == "__main__":
    # quick smoke test
    x = np.array([[0,1,1],[0,0,0],[2,2,0]])
    y = invariant_extract_recolor_rotate(x, "canonical_first", "most_frequent", k=1, map_back_to_original=True)
    print("Input:\n", x)
    print("Output:\n", y)
