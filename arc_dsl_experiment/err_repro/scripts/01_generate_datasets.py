
import numpy as np, random, argparse
from pathlib import Path
import importlib.util, sys

# Load invariant ops
spec = importlib.util.spec_from_file_location("invariant_ops", str(Path(__file__).resolve().parent.parent / "invariant_ops.py"))
inv = importlib.util.module_from_spec(spec); sys.modules["invariant_ops"] = inv
spec.loader.exec_module(inv)

rng = random.Random(42)
np.random.seed(42)

def place_rect(grid, color, h, w, rng, max_tries=100):
    H, W = grid.shape
    for _ in range(max_tries):
        r0 = rng.randrange(0, H - h + 1)
        c0 = rng.randrange(0, W - w + 1)
        sub = grid[r0:r0+h, c0:c0+w]
        if np.all(sub == 0):
            grid[r0:r0+h, c0:c0+w] = color
            return True
    return False

def make_input_grid(H=12, W=12, num_objects=3, color_pool=range(1,10), rng=random.Random()):
    g = np.zeros((H,W), dtype=int)
    used_colors = set(); attempts = 0
    while len(used_colors) < num_objects and attempts < num_objects*10:
        attempts += 1
        h = rng.randint(1, 4); w = rng.randint(1, 5)
        color_candidates = [c for c in color_pool if c not in used_colors]
        if not color_candidates: break
        color = rng.choice(color_candidates)
        if place_rect(g, color, h, w, rng): used_colors.add(color)
    return g

def build_dataset(out_npz, out_json, n=24, H=12, W=12, num_objects_range=(3,5), recolor_rule="most_frequent", k=1):
    pairs = []
    rng = random.Random(42)
    for _ in range(n):
        num_objs = rng.randint(*num_objects_range)
        x = make_input_grid(H=H, W=W, num_objects=num_objs, rng=rng)
        y = inv.invariant_extract_recolor_rotate(x, select_rule="canonical_first", recolor_rule=recolor_rule, k=k, map_back_to_original=True)
        pairs.append((x,y))
    # save
    data = {}
    for i,(x,y) in enumerate(pairs):
        data[f"x{i}"] = x; data[f"y{i}"] = y
    np.savez(out_npz, **data)
    # json
    import json
    Path(out_json).write_text(json.dumps([{"x":x.tolist(), "y":y.tolist()} for (x,y) in pairs], indent=2))

if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent.parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    build_dataset(out_dir/"invariant_ds_mf_k1.npz", out_dir/"DS1_mostfreq_k1.json", recolor_rule="most_frequent", k=1)
    build_dataset(out_dir/"invariant_ds_lf_k2.npz", out_dir/"DS2_leastfreq_k2.json", recolor_rule="least_frequent", k=2)
    print("Datasets written to", out_dir)
