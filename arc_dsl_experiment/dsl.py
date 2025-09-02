
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List, Tuple, Dict
import numpy as np, random, time, json

def segment_components(g):
    H, W = g.shape
    visited = np.zeros_like(g, dtype=bool)
    comps = []; dirs = [(1,0),(-1,0),(0,1),(0,-1)]
    for r in range(H):
        for c in range(W):
            col = int(g[r,c])
            if col == 0 or visited[r,c]: continue
            q=[(r,c)]; visited[r,c]=True; pixels=[]
            min_r=min_c=r; max_r=max_c=c
            while q:
                rr,cc=q.pop(); pixels.append((rr,cc))
                min_r=min(min_r,rr); max_r=max(max_r,rr)
                min_c=min(min_c,cc); max_c=max(max_c,cc)
                for dr,dc in dirs:
                    nr,nc=rr+dr, cc+dc
                    if 0<=nr<H and 0<=nc<W and not visited[nr,nc] and int(g[nr,nc])==col:
                        visited[nr,nc]=True; q.append((nr,nc))
            comps.append({"color":col,"pixels":pixels,"area":len(pixels),"bbox":(min_r,min_c,max_r,max_c),"anchor":(min_r,min_c)})
    return comps

def color_hist_nonzero(g):
    vals = g[g!=0]
    if vals.size==0: return {}
    u,c = np.unique(vals, return_counts=True)
    return {int(ui): int(ci) for ui,ci in zip(u,c)}

def canonical_component_key(comp):
    return (comp['area'], comp['bbox'][0], comp['bbox'][1], comp['color'])

def alpha1_palette(g):
    """
    A1 palette canonicalization: relabel nonzero colors by
      (1) descending frequency,
      (2) tie → lexicographic first occurrence (row, col),
      (3) tie → lexicographic order of the color's binary mask string (row-major).
    Background 0 is fixed. Returns (x_hat, meta) where meta has
    "orig_for_can" and "can_for_orig" bijections.
    """
    H, W = g.shape
    hist = color_hist_nonzero(g)  # {color: count}, excludes 0
    if not hist:
        # nothing to do
        return g.copy(), {"orig_for_can": {}, "can_for_orig": {}}

    # First occurrence (row, col) per color (row-major scan)
    first = {}
    for r in range(H):
        for c in range(W):
            v = int(g[r, c])
            if v == 0:
                continue
            if v not in first:
                first[v] = (r, c)

    # Binary mask string per color (row-major)
    masks = {}
    # To avoid quadratic work, we build masks in one pass
    # by appending to lists, then join once.
    masks_lists = {v: [] for v in hist.keys()}
    for r in range(H):
        for c in range(W):
            v = int(g[r, c])
            for col in hist.keys():
                masks_lists[col].append('1' if col == v else '0')
    for col, lst in masks_lists.items():
        masks[col] = ''.join(lst)

    # Sort colors by the invariant total order
    colors = list(hist.keys())
    colors.sort(key=lambda col: (-hist[col], first[col], masks[col]))

    can_for_orig = {}
    orig_for_can = {}
    for k, orig in enumerate(colors, start=1):
        can_for_orig[orig] = k
        orig_for_can[k] = orig

    # Relabel grid, keeping background 0
    x_hat = np.zeros_like(g)
    for r in range(H):
        for c in range(W):
            v = int(g[r, c])
            x_hat[r, c] = 0 if v == 0 else can_for_orig[v]

    return x_hat, {"orig_for_can": orig_for_can, "can_for_orig": can_for_orig}

def remap_with_can_for_orig(g, can_for_orig: Dict[int,int]):
    out = np.zeros_like(g)
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            v = int(g[r,c])
            out[r,c] = 0 if v==0 else can_for_orig.get(v, v)
    return out

def alpha2_objorder(x_hat):
    comps = segment_components(x_hat)
    comps_sorted = sorted(comps, key=canonical_component_key)
    return x_hat, {"order": comps_sorted}

def ground_truth_transform(x):
    comps = segment_components(x)
    if not comps: return x.copy()
    s = sorted(comps, key=canonical_component_key)[0]
    hist = color_hist_nonzero(x); m=min(hist.values()); cand=[c for c,h in hist.items() if h==m]
    target = max(cand)
    y = x.copy()
    for (r,c) in s["pixels"]:
        y[r,c] = target
    return y

def place_rect(g, r, c, h, w, color):
    H,W = g.shape
    r2 = min(H, r+h); c2 = min(W, c+w)
    g[r:r2, c:c2] = color

def empty_with_padding(g, rr, cc):
    H,W = g.shape
    if g[rr,cc] != 0: return False
    for dr,dc in [(1,0),(-1,0),(0,1),(0,-1)]:
        nr, nc = rr+dr, cc+dc
        if 0<=nr<H and 0<=nc<W and g[nr,nc]!=0: return False
    return True

def gen_pair(H, W, C, tie=False, seed=0):
    rng = random.Random(seed)
    g = np.zeros((H,W), dtype=int)
    colors = list(range(1, C+1))

    # 1) Distractors first (so singletons won't be overwritten or attached)
    for _ in range(rng.randint(8, 16)):
        col = rng.choice(colors)
        h = rng.randint(2, max(2, H//3))
        w = rng.randint(2, max(2, W//3))
        r = rng.randint(0, max(0, H-h))
        c = rng.randint(0, max(0, W-w))
        place_rect(g, r, c, h, w, col)

    for _ in range(rng.randint(5, 10)):
        col = rng.choice(colors)
        if rng.random()<0.5 and H>=2:
            rr = rng.randint(0, H-2); cc = rng.randint(0, W-1)
            if g[rr,cc]==0 and g[rr+1,cc]==0:
                g[rr,cc]=col; g[rr+1,cc]=col
        else:
            rr = rng.randint(0, H-1); cc = rng.randint(0, W-2)
            if g[rr,cc]==0 and g[rr,cc+1]==0:
                g[rr,cc]=col; g[rr,cc+1]=col

    # 2) Protected singletons last, with 4-neighbor empty padding
    singles = 2 if tie else 1
    placed=0; attempts=0
    while placed<singles and attempts<5000:
        rr = rng.randint(0, H-1); cc = rng.randint(0, W-1)
        if empty_with_padding(g, rr, cc):
            # choose a color whose global count is currently not the minimum to make it change on recolor
            hist = color_hist_nonzero(g)
            available = list(hist.keys()) if hist else colors[:]
            col = rng.choice(available)
            g[rr,cc] = col
            placed+=1
        attempts+=1

    x = g
    y = ground_truth_transform(x)
    return x, y

def generate_dataset(n_pairs=12, H_range=(18,24), W_range=(18,24), C_range=(7,12), tie_ratio=0.5, seed=4242):
    rng = random.Random(seed)
    pairs = []
    for i in range(n_pairs):
        H = rng.randint(H_range[0], H_range[1])
        W = rng.randint(W_range[0], W_range[1])
        C = rng.randint(C_range[0], C_range[1])
        tie = (rng.random() < tie_ratio)
        x,y = gen_pair(H,W,C, tie=tie, seed=rng.randint(0,10**9))
        pairs.append((x,y))
    return pairs

ONE_LARGE_DATASET = generate_dataset()

def sel_color_max_id(x_hat):
    used = set(int(v) for v in np.unique(x_hat) if v!=0)
    return max(used) if used else 0

def sel_color_argmin_hist(x_hat):
    hist = color_hist_nonzero(x_hat)
    if not hist: return 0
    m = min(hist.values()); cand = [c for c,h in hist.items() if h==m]
    return max(cand)

COLOR_RULES = [("max_id", sel_color_max_id), ("argmin_hist", sel_color_argmin_hist)]

def sel_comp_smallest_canonical(x_hat):
    comps=segment_components(x_hat)
    if not comps: return None
    return sorted(comps, key=canonical_component_key)[0]

def sel_comp_smallest_unstable(x_hat):
    comps=segment_components(x_hat)
    if not comps: return None
    seed = hash(x_hat.tobytes()) & 0xFFFFFFFF
    rng = random.Random(seed); rng.shuffle(comps)
    m = min(c["area"] for c in comps)
    for c in comps:
        if c["area"]==m: return c
    return comps[0]

def recolor_component(g, comp, new_color):
    y=g.copy()
    if comp is None: return y
    for (r,c) in comp["pixels"]:
        y[r,c] = new_color
    return y

def rotate_90(g):
    return np.rot90(g, k=-1)

def preop_identity(g): return g

def make_preop_color_perm(perm_map: Dict[int,int]):
    def pre(g):
        out = np.zeros_like(g)
        for r in range(g.shape[0]):
            for c in range(g.shape[1]):
                v=int(g[r,c])
                out[r,c] = perm_map.get(v, v)
        return out
    return pre

def random_palette_permutation(g, rng):
    colors = sorted(set(int(v) for v in np.unique(g) if v!=0))
    perm = colors[:]; rng.shuffle(perm)
    return {c:p for c,p in zip(colors, perm)}

def build_preops_for_dataset(dataset, num_preops=600, seed=11):
    rng = random.Random(seed)
    bases=[x for (x,_) in dataset]
    preops=[("identity", preop_identity)]
    for i in range(num_preops):
        base = rng.choice(bases)
        perm = random_palette_permutation(base, rng)
        preops.append((f"perm_{i+1}", make_preop_color_perm(perm)))
    return preops

def measure_G(dataset, trials=600, seed=1):
    preops = build_preops_for_dataset(dataset, num_preops=600, seed=11)
    comp_rules = [("smallest_unstable", sel_comp_smallest_unstable),
                  ("smallest_canonical", sel_comp_smallest_canonical)]
    cands = []
    for pre_name, pre_f in preops:
        for cn, cf in COLOR_RULES:
            for sn, sf in comp_rules:
                cands.append((pre_name, cn, sn, pre_f, cf, sf))
    total=len(cands); valid=[]
    t0=time.time()
    for (pre_name, cn, sn, pre_f, cf, sf) in cands:
        ok=True
        for x,y in dataset:
            x2 = pre_f(x)
            comp = sf(x2); color = cf(x2)
            y_pred = recolor_component(x2, comp, color)
            y2 = pre_f(y)
            if not np.array_equal(y_pred, y2):
                ok=False; break
        if ok: valid.append((pre_name,cn,sn))
    t1=time.time()
    avg=None
    if valid:
        rng=random.Random(seed); ranks=[]
        valid_set=set(valid)
        for _ in range(trials):
            order=list(range(total)); rng.shuffle(order)
            rank=None
            for idx,pos in enumerate(order, start=1):
                pre_name,cn,sn,*_ = cands[pos]
                if (pre_name,cn,sn) in valid_set: rank=idx; break
            ranks.append(rank if rank is not None else total)
        avg=sum(ranks)/len(ranks)
    return {"total_candidates":total,"num_valid":len(valid),"avg_tries_to_success":avg,"wall_time_s":t1-t0,"sample_valid":valid[:10]}

def build_A1_selectors():
    sels=[("smallest_canonical", sel_comp_smallest_canonical),
          ("smallest_unstable", sel_comp_smallest_unstable)]
    key_names=["area_left_top_color","area_color_top_left","bbox_area_top_left_color","anchor_lex_color"]
    def make_sel_smallest_key(order_key_name):
        def f(x_hat):
            comps=segment_components(x_hat)
            if not comps: return None
            if order_key_name=="area_left_top_color":
                key=lambda c:(c["area"], c["bbox"][1], c["bbox"][0], c["color"])
            elif order_key_name=="area_color_top_left":
                key=lambda c:(c["area"], c["color"], c["bbox"][0], c["bbox"][1])
            elif order_key_name=="bbox_area_top_left_color":
                key=lambda c:(((c["bbox"][2]-c["bbox"][0]+1)*(c["bbox"][3]-c["bbox"][1]+1)), c["bbox"][0], c["bbox"][1], c["color"])
            elif order_key_name=="anchor_lex_color":
                key=lambda c:(c["area"], c["anchor"][0], c["anchor"][1], c["color"])
            else:
                key=lambda c:(c["area"], c["bbox"][0], c["bbox"][1], c["color"])
            return sorted(comps, key=key)[0]
        f.__name__ = f"sel_smallest_{order_key_name}"
        return f
    for kn in key_names:
        f=make_sel_smallest_key(kn); sels.append((f.__name__, f))
    def make_sel_smallest_seed(seed_val):
        def f(x_hat):
            comps=segment_components(x_hat)
            if not comps: return None
            rng = random.Random(seed_val); rng.shuffle(comps)
            m=min(c["area"] for c in comps)
            for c in comps:
                if c["area"]==m: return c
            return comps[0]
        f.__name__ = f"sel_smallest_seed_{seed_val}"
        return f
    for s in range(80):
        f=make_sel_smallest_seed(s); sels.append((f.__name__, f))
    return sels

def measure_A1(dataset, trials=600, seed=1):
    sels=build_A1_selectors()
    cands=[(cn,sn, cf,sf) for (cn,cf) in COLOR_RULES for (sn,sf) in sels]
    total=len(cands); valid=[]
    t0=time.time()
    for cn,sn, cf,sf in cands:
        ok=True
        for x,y in dataset:
            ax, meta = alpha1_palette(x)
            ay_fixed = remap_with_can_for_orig(y, meta["can_for_orig"])
            comp=sf(ax); color=cf(ax)
            y_pred_raw=recolor_component(ax, comp, color)
            if not np.array_equal(y_pred_raw, ay_fixed):
                ok=False; break
        if ok: valid.append((cn,sn))
    t1=time.time()
    avg=None
    if valid:
        rng=random.Random(seed); ranks=[]
        valid_set=set(valid)
        for _ in range(trials):
            order=list(range(total)); rng.shuffle(order)
            rank=None
            for idx,pos in enumerate(order, start=1):
                cn,sn,*_=cands[pos]
                if (cn,sn) in valid_set: rank=idx; break
            ranks.append(rank if rank is not None else total)
        avg=sum(ranks)/len(ranks)
    return {"total_candidates": total,"num_valid": len(valid),"avg_tries_to_success": avg,"wall_time_s": t1-t0,"sample_valid": valid[:10]}

def measure_A12(dataset, trials=600, seed=1):
    comp_rules=[("index0", lambda a: sel_comp_smallest_canonical(a))]
    cands=[(cn,sn, cf,sf) for (cn,cf) in COLOR_RULES for (sn,sf) in comp_rules]
    total=len(cands); valid=[]
    t0=time.time()
    for cn,sn, cf,sf in cands:
        ok=True
        for x,y in dataset:
            ax, meta = alpha1_palette(x)
            a12,_ = alpha2_objorder(ax)
            ay_fixed = remap_with_can_for_orig(y, meta["can_for_orig"])
            b12,_ = alpha2_objorder(ay_fixed)
            comp=sf(a12); color=cf(a12)
            y_pred_raw=recolor_component(a12, comp, color)
            y_pred,_ = alpha2_objorder(y_pred_raw)
            if not np.array_equal(y_pred, b12):
                ok=False; break
        if ok: valid.append((cn,sn))
    t1=time.time()
    avg=None
    if valid:
        rng=random.Random(seed); ranks=[]
        valid_set=set(valid)
        for _ in range(trials):
            order=list(range(total)); rng.shuffle(order)
            rank=None
            for idx,pos in enumerate(order, start=1):
                cn,sn,*_=cands[pos]
                if (cn,sn) in valid_set: rank=idx; break
            ranks.append(rank if rank is not None else total)
        avg=sum(ranks)/len(ranks)
    return {"total_candidates": total,"num_valid": len(valid),"avg_tries_to_success": avg,"wall_time_s": t1-t0,"sample_valid": valid[:10]}

def run():
    np.random.seed(0); random.seed(0)
    ds = ONE_LARGE_DATASET
    resG  = measure_G(ds, trials=600, seed=1)
    resA1 = measure_A1(ds, trials=600, seed=1)
    resA12= measure_A12(ds, trials=600, seed=1)
    lines = []
    lines.append("=== Challenging Single-Dataset Metrics (G → A1 → A1→A2, fixed A1 gauge, protected singles) ===")
    for name,res in [("G",resG),("A1",resA1),("A1→A2",resA12)]:
        lines.append(f"[{name}] total_candidates={res['total_candidates']}  num_valid={res['num_valid']}  avg_tries_to_success={res['avg_tries_to_success']}  wall_time_s={res['wall_time_s']:.3f}")
    out = "\\n".join(lines)
    print(out)
    with open("challenging_metrics.txt","w",encoding="utf-8") as f:
        f.write(out)
    with open("challenging_metrics.json","w",encoding="utf-8") as f:
        json.dump({"G":resG,"A1":resA1,"A1->A2":resA12}, f, indent=2)

def example_recolor_rotate():
    """
    Example showing invariant: output grid is one object from input grid, recolored and rotated.
    
    Invariant representation:
    1. Select object using canonical ordering (smallest by area→top→left→color)
    2. Recolor to target color (e.g., max color ID or min frequency color)  
    3. Apply 90-degree rotation
    4. Output is the transformed grid
    """
    # Create example input: 4x4 grid with two objects
    x = np.array([
        [0, 1, 0, 2],
        [0, 1, 0, 2], 
        [0, 0, 0, 0],
        [3, 3, 3, 0]
    ])
    
    print("Input grid:")
    print(x)
    print()
    
    # Apply A1 abstraction (palette canonicalization)
    x_a1, meta = alpha1_palette(x)
    print("After A1 (palette canonicalization):")
    print(x_a1)
    print(f"Color mapping: {meta['can_for_orig']}")
    print()
    
    # Apply A2 abstraction (object ordering)  
    x_a12, obj_meta = alpha2_objorder(x_a1)
    print("After A2 (object ordering):")
    print("Objects in canonical order:", [f"area={c['area']}, color={c['color']}" for c in obj_meta['order']])
    print()
    
    # Select object (smallest canonical)
    selected_obj = sel_comp_smallest_canonical(x_a12)
    print(f"Selected object: area={selected_obj['area']}, color={selected_obj['color']}")
    print()
    
    # Recolor to target (max color ID)
    target_color = sel_color_max_id(x_a12)
    y_recolored = recolor_component(x_a12, selected_obj, target_color)
    print(f"After recoloring to target color {target_color}:")
    print(y_recolored)
    print()
    
    # Apply rotation
    y_rotated = rotate_90(y_recolored)
    print("After 90-degree rotation (final output):")
    print(y_rotated)
    print()
    
    # Convert back to original color space
    y_final = remap_with_can_for_orig(y_rotated, meta['orig_for_can'])
    print("Mapped back to original colors:")
    print(y_final)

if __name__=="__main__":
    print("=== Example: Recolor and Rotate Invariant ===")
    example_recolor_rotate()
    print("\n" + "="*50 + "\n")
    run()
