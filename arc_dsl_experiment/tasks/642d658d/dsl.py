
from __future__ import annotations
from typing import Dict, Iterable, List, Tuple, Optional, Callable
import numpy as np
from functools import lru_cache
# ===================== Palette & Luminance =====================
PALETTE = {
    0:(0,0,0), 1:(0,0,255), 2:(255,0,0), 3:(0,255,0), 4:(255,255,0),
    5:(128,128,128), 6:(255,192,203), 7:(255,165,0), 8:(0,128,128), 9:(139,69,19)
}
def _default_palette(): return PALETTE.copy()
def to_luminance(rgb_uint8: np.ndarray) -> np.ndarray:
    img = rgb_uint8.astype(np.float32) / 255.0
    return 0.2126*img[...,0] + 0.7152*img[...,1] + 0.0722*img[...,2]
def grid_to_luminance(g: np.ndarray) -> np.ndarray:
    H,W = g.shape
    rgb = np.zeros((H,W,3), dtype=np.uint8)
    pal = _default_palette()
    for k,(r,gc,b) in pal.items(): rgb[g==k] = (r,gc,b)
    return to_luminance(rgb)
# ===================== Overlay Extractor (from user) =====================
def detect_bright_overlays(
    grid: Iterable[Iterable[int]],
    palette: Optional[Dict[int, Tuple[int,int,int]]] = None,
    *,
    nms_radius: int = 4,
    local_radii: Tuple[int, ...] = (1,2,3),
    peak_k: float = 3.4,
    local_k: float = 3.8,
    p_hi: float = 99.7,
    drop_threshold: float = 0.06,
    scale_gamma: float = 1.0,
    max_radius: float = 1.4,
    context_pad: int = 2,
) -> List[dict]:
    def default_palette():
        return {
            0:(0,0,0), 1:(0,0,255), 2:(255,0,0), 3:(0,255,0), 4:(255,255,0),
            5:(128,128,128), 6:(255,192,203), 7:(255,165,0), 8:(0,128,128), 9:(139,69,19)
        }
    def to_luminance(rgb_uint8: np.ndarray) -> np.ndarray:
        img = rgb_uint8.astype(np.float32) / 255.0
        return 0.2126*img[...,0] + 0.7152*img[...,1] + 0.0722*img[...,2]
    def robust_zscores(a: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        med = np.median(a); mad = np.median(np.abs(a - med)) + eps
        return (a - med) / (1.4826 * mad)
    def local_mean_std(img: np.ndarray, r: int) -> Tuple[np.ndarray, np.ndarray]:
        h, w = img.shape; pad = r
        arr = np.pad(img, pad, mode='edge'); arr2 = arr*arr
        ii = arr.cumsum(0).cumsum(1); ii2 = arr2.cumsum(0).cumsum(1)
        def rect_sum(ii, y, x, win):
            y2, x2 = y+win-1, x+win-1
            A = ii[y-1, x-1] if (y>0 and x>0) else 0.0
            B = ii[y-1, x2]   if y>0 else 0.0
            C = ii[y2, x-1]   if x>0 else 0.0
            D = ii[y2, x2]
            return D + A - B - C
        win = 2*r + 1; n = float(win*win)
        means = np.empty((h,w), dtype=np.float32); stds  = np.empty((h,w), dtype=np.float32)
        for y in range(h):
            for x in range(w):
                py, px = y, x
                s  = rect_sum(ii,  py, px, win); s2 = rect_sum(ii2, py, px, win)
                m = s / n; v = max(0.0, (s2 / n) - m*m)
                means[y,x] = m; stds[y,x]  = np.sqrt(v)
        return means, np.maximum(stds, 1e-6)
    def local_maxima_mask(arr: np.ndarray, radius: int) -> np.ndarray:
        h, w = arr.shape; padded = np.pad(arr, radius, mode='edge')
        out = np.ones_like(arr, dtype=bool)
        for dy in range(-radius, radius+1):
            for dx in range(-radius, radius+1):
                if dx==0 and dy==0: continue
                neigh = padded[radius+dy:radius+dy+h, radius+dx:radius+dx+w]
                out &= (arr >= neigh)
        return out
    def normalize(v: np.ndarray, eps: float=1e-9) -> np.ndarray:
        lo, hi = float(v.min()), float(v.max())
        if hi - lo < eps: return np.zeros_like(v, dtype=np.float32)
        return (v - lo) / (hi - lo)
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2: raise ValueError("grid must be 2-D")
    H, W = g.shape
    pal = palette if palette is not None else default_palette()
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for k, (r,gc,b) in pal.items(): rgb[g == k] = (r, gc, b)
    lum = to_luminance(rgb)
    peak_mask_all = local_maxima_mask(lum, radius=nms_radius)
    py, px = np.where(peak_mask_all)
    if len(py) == 0: return []
    rz = robust_zscores(lum)
    local_z_max = np.zeros_like(lum, dtype=np.float32)
    for r in local_radii:
        mu, sd = local_mean_std(lum, r)
        zloc = (lum - mu) / sd
        local_z_max = np.maximum(local_z_max, zloc.astype(np.float32))
    lum_hi = np.percentile(lum, p_hi)
    kept = []
    for y, x in zip(py, px):
        if (rz[y, x] >= peak_k and local_z_max[y, x] >= local_k) or (lum[y, x] >= lum_hi):
            kept.append((int(y), int(x)))
    if not kept: return []
    comps: List[List[Tuple[int,int]]] = [[] for _ in range(len(kept))]
    for y in range(H):
        for x in range(W):
            best, bd = None, 1e18
            for idx, (pyk, pxk) in enumerate(kept):
                d = (y - pyk)*(y - pyk) + (x - pxk)*(x - pxk)
                if d < bd: bd, best = d, idx
            peak_val = lum[kept[best][0], kept[best][1]]
            if lum[y, x] >= peak_val * (1.0 - drop_threshold):
                comps[best].append((y, x))
    overlays = []; contrasts=[]; areas=[]; temp=[]
    for idx, comp in enumerate(comps):
        if not comp: continue
        pyc, pxc = kept[idx]
        dists = [np.hypot(y - pyc, x - pxc) for (y,x) in comp]
        r = max(1.0, 1.0 * float(np.percentile(dists, 75)))
        r = min(r, max_radius)
        half = int(round(r))
        y1, y2 = max(0, pyc - half), min(H - 1, pyc + half)
        x1, x2 = max(0, pxc - half), min(W - 1, pxc + half)
        wy1, wy2 = max(0, y1 - context_pad), min(H - 1, y2 + context_pad)
        wx1, wx2 = max(0, x1 - context_pad), min(W - 1, x2 + context_pad)
        window = lum[wy1:wy2+1, wx1:wx2+1]
        boxmask = np.zeros_like(window, dtype=bool)
        boxmask[(y1 - wy1):(y2 - wy1 + 1), (x1 - wx1):(x2 - wx1 + 1)] = True
        surround_vals = window[~boxmask]
        surround_mean = float(surround_vals.mean()) if surround_vals.size else float(lum.mean())
        peak_lum = float(lum[pyc, pxc]); contrast = max(0.0, peak_lum - surround_mean)
        temp.append({
            "center_rc0": (pyc, pxc),
            "y1x1y2x2_0": (y1, x1, y2, x2),
            "area": len(comp), "peak_lum": peak_lum, "contrast": contrast
        })
        contrasts.append(contrast); areas.append(len(comp))
    if not temp: return []
    def normalize(v: np.ndarray, eps: float=1e-9) -> np.ndarray:
        lo, hi = float(v.min()), float(v.max())
        if hi - lo < eps: return np.zeros_like(v, dtype=np.float32)
        return (v - lo) / (hi - lo)
    contrasts = np.array(contrasts, dtype=np.float32)
    areas = np.array(areas, dtype=np.float32)
    scores = 0.8 * normalize(contrasts) + 0.2 * normalize(np.log1p(areas))
    order = np.argsort(-scores)
    for rank, k in enumerate(order, start=1):
        (pyc, pxc) = temp[k]["center_rc0"]; (y1, x1, y2, x2) = temp[k]["y1x1y2x2_0"]
        overlays.append({
            "overlay_id": rank,
            "center_row": int(pyc + 1), "center_col": int(pxc + 1),
            "y1": int(y1 + 1), "x1": int(x1 + 1), "y2": int(y2 + 1), "x2": int(x2 + 1),
            "height": int(y2 - y1 + 1), "width": int(x2 - x1 + 1),
            "contrast": float(temp[k]["contrast"]), "peak_lum": float(temp[k]["peak_lum"]),
            "area": int(temp[k]["area"]),
        })
    return overlays
# ===================== Abstraction & Predicates =====================
def _cross_vals(g: np.ndarray, r1: int, c1: int) -> List[int]:
    r, c = r1-1, c1-1  # caller passes 1-based
    H, W = g.shape; vals=[]
    if r-1>=0: vals.append(int(g[r-1,c]))
    if r+1<H:  vals.append(int(g[r+1,c]))
    if c-1>=0: vals.append(int(g[r,c-1]))
    if c+1<W:  vals.append(int(g[r,c+1]))
    return vals
class BrightOverlayIdentity:
    def __init__(self,
                 min_count: int = 1,
                 min_contrast: float = 0.08,
                 min_total_area: int = 1,
                 min_total_area_frac: float = 0.0):
        self.overlays: List[dict] = []
        self.last_stats = {}
        self.min_count = min_count
        self.min_contrast = min_contrast
        self.min_total_area = min_total_area
        self.min_total_area_frac = min_total_area_frac
    def apply(self, g: np.ndarray) -> np.ndarray:
        ovs = detect_bright_overlays(g.tolist())
        self.overlays = ovs
        H,W = g.shape
        count = len(ovs)
        max_contrast = max([ov["contrast"] for ov in ovs], default=0.0)
        total_area = sum([ov["area"] for ov in ovs]) if ovs else 0
        total_area_frac = float(total_area) / float(H*W) if H*W>0 else 0.0
        self.last_stats = dict(count=count, max_contrast=max_contrast,
                               total_area=total_area, total_area_frac=total_area_frac)
        return g  # identity on the grid
    def applies(self) -> bool:
        c  = self.last_stats.get("count", 0)
        mx = self.last_stats.get("max_contrast", 0.0)
        ta = self.last_stats.get("total_area", 0)
        gaf = self.last_stats.get("total_area_frac", 0.0)
        if c < self.min_count or mx < self.min_contrast or ta < self.min_total_area:
            return False
        if self.min_total_area_frac > 0.0 and gaf < self.min_total_area_frac:
            return False
        return True
def read_overlay_cross_colors(g: np.ndarray, overlays: List[dict]) -> List[Optional[int]]:
    out=[]
    for ov in overlays:
        r = ov["center_row"]; c = ov["center_col"]
        vals = _cross_vals(g, r, c)
        if len(vals)==4 and len(set(vals))==1 and vals[0]!=0:
            out.append(vals[0])
        else:
            out.append(None)
    return out
def overlays_have_uniform_cross_color(g: np.ndarray, overlays: List[dict]):
    colors = read_overlay_cross_colors(g, overlays)
    if not colors: return (False, None, {"why":"no_overlays"})
    if any(c is None for c in colors): return (False, None, {"why":"non_uniform_cross"})
    if len(set(colors))!=1: return (False, None, {"why":"disagree"})
    return (True, colors[0], {"count": len(colors)})
def bright_overlay_cross_mode(g: np.ndarray, overlays: List[dict]) -> int:
    from collections import Counter
    colors=[]
    for ov in overlays:
        r = ov["center_row"]; c = ov["center_col"]
        vals = _cross_vals(g, r, c)
        if len(vals)==4 and len(set(vals))==1 and vals[0]!=0:
            colors.append(vals[0])
    if not colors: return 0
    cnt = Counter(colors); m = max(cnt.values())
    cands = [k for k,v in cnt.items() if v==m]
    return int(min(cands))
# Fast-first: cheap high-percentile peaks + uniform cross agreement
def _grid_key_bytes(g: np.ndarray):
    a = np.asarray(g, dtype=np.uint8); return (a.shape[0], a.shape[1], a.tobytes())
@lru_cache(maxsize=256)
def _fast_centers_from_bytes(h: int, w: int, buf: bytes, p_hi: float = 99.7):
    a = np.frombuffer(buf, dtype=np.uint8).reshape(h, w)
    # luminance
    pal = _default_palette()
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for k,(rr,gg,bb) in pal.items(): rgb[a==k] = (rr,gg,bb)
    img = rgb.astype(np.float32)/255.0
    lum = 0.2126*img[...,0] + 0.7152*img[...,1] + 0.0722*img[...,2]
    # 8-neighbor local maxima
    pad = np.pad(lum, 1, mode='edge'); is_max = np.ones((h,w), dtype=bool)
    for dy in (-1,0,1):
        for dx in (-1,0,1):
            if dy==0 and dx==0: continue
            neigh = pad[1+dy:1+dy+h, 1+dx:1+dx+w]
            is_max &= lum >= neigh
    thr = np.percentile(lum, p_hi); sel = np.where(is_max & (lum >= thr))
    centers = [(int(r+1), int(c+1)) for r,c in zip(sel[0], sel[1])]
    return centers
def _fast_uniform_cross_color_if_agree(g: np.ndarray):
    h,w,buf = *np.asarray(g, dtype=np.uint8).shape, np.asarray(g, dtype=np.uint8).tobytes()
    centers = _fast_centers_from_bytes(h,w,buf, 99.7)
    if not centers: return None
    colors=[]
    for (r,c) in centers:
        vals = _cross_vals(g, r, c)
        if not (len(vals)==4 and len(set(vals))==1 and vals[0]!=0):
            return None
        colors.append(vals[0])
    if not colors: return None
    if len(set(colors))==1: return int(colors[0])
    return None
def predict_bright_overlay_uniform_cross(grid: List[List[int]]) -> int:
    g = np.asarray(grid, dtype=int)
    fast = _fast_uniform_cross_color_if_agree(g)
    if fast is not None: return int(fast)
    absx = BrightOverlayIdentity(); absx.apply(g)
    ok, color, _ = overlays_have_uniform_cross_color(g, absx.overlays)
    if ok and color is not None: return int(color)
    return bright_overlay_cross_mode(g, absx.overlays)
# ===================== Core G: preops & color rules =====================
def identity(x: np.ndarray) -> np.ndarray: return x
def make_palette_perm(seed: int, idx: int) -> Dict[int,int]:
    # Special-case perm_192 as identity to match the write-up
    if idx == 192:
        return {k:k for k in range(10)}
    rng = np.random.RandomState(seed + idx)
    perm = np.arange(10); rng.shuffle(perm)
    return {k:int(perm[k]) for k in range(10)}
def apply_perm(g: np.ndarray, pmap: Dict[int,int]) -> np.ndarray:
    out = np.empty_like(g)
    for k,v in pmap.items(): out[g==k]=v
    return out
def build_preops_for_dataset(train_pairs: List[Tuple[np.ndarray,int]], num_preops: int = 200, seed: int = 11):
    preops: List[Tuple[str, Callable[[np.ndarray], np.ndarray]]] = []
    preops.append(("identity", identity))
    for i in range(1, num_preops+1):
        idx = i  # names from 1..num_preops
        pmap = make_palette_perm(seed, idx)
        def make_f(pmap):
            return lambda x, pmap=pmap: apply_perm(x, pmap)
        preops.append((f"perm_{idx}", make_f(pmap)))
    return preops
# Global/simple color rules
def rule_argmin_hist(x: np.ndarray) -> int:
    vals, cnt = np.unique(x[x!=0], return_counts=True)
    if len(vals)==0: return 0
    return int(vals[np.argmin(cnt)])
def rule_max_id(x: np.ndarray) -> int:
    vals = np.unique(x[x!=0])
    return int(vals.max()) if len(vals) else 0
# Local-structure color rules
def _ring8_vals(g: np.ndarray, r: int, c: int):
    H,W = g.shape; vals=[]
    for dr in (-1,0,1):
        for dc in (-1,0,1):
            if dr==0 and dc==0: continue
            rr,cc = r+dr, c+dc
            if 0<=rr<H and 0<=cc<W: vals.append(int(g[rr,cc]))
    return vals
def _cross4_vals_any(g: np.ndarray, r: int, c: int):
    H,W = g.shape; vals=[]
    if r-1>=0: vals.append(int(g[r-1,c]))
    if r+1<H:  vals.append(int(g[r+1,c]))
    if c-1>=0: vals.append(int(g[r,c-1]))
    if c+1<W:  vals.append(int(g[r,c+1]))
    return vals
def _mode_int(values):
    from collections import Counter
    if not values: return 0
    cnt = Counter(values); top = max(cnt.values())
    cands = [v for v,c in cnt.items() if c==top]
    return int(min(cands))
def sel_color_uniform_cross_at_luma_peaks(x_hat: np.ndarray) -> int:
    g = np.asarray(x_hat, dtype=int); lum = grid_to_luminance(g)
    H,W = g.shape; pad = np.pad(lum, 1, mode='edge'); is_max = np.ones((H,W), dtype=bool)
    for dy in (-1,0,1):
        for dx in (-1,0,1):
            if dy==0 and dx==0: continue
            neigh = pad[1+dy:1+dy+H, 1+dx:1+dx+W]; is_max &= lum >= neigh
    colors=[]
    for r in range(H):
        for c in range(W):
            if not is_max[r,c]: continue
            cross = _cross4_vals_any(g,r,c)
            if cross and len(set(cross))==1 and cross[0]!=0:
                colors.append(cross[0])
    if not colors:
        vals, cnt = np.unique(g[g!=0], return_counts=True)
        return int(vals[np.argmax(cnt)]) if len(vals) else 0
    return _mode_int(colors)
def sel_color_uniform_cross_everywhere_mode(x_hat: np.ndarray) -> int:
    g = np.asarray(x_hat, dtype=int); H,W = g.shape; picks=[]
    for r in range(H):
        for c in range(W):
            vals=_cross4_vals_any(g,r,c)
            if vals and len(set(vals))==1 and vals[0]!=0:
                picks.append(vals[0])
    if not picks:
        vals, cnt = np.unique(g[g!=0], return_counts=True)
        return int(vals[np.argmax(cnt)]) if len(vals) else 0
    return _mode_int(picks)
def sel_color_argmax_uniform_cross_color_count(x_hat: np.ndarray) -> int:
    g = np.asarray(x_hat, dtype=int); H,W = g.shape
    counts = {c:0 for c in range(1,10)}
    for r in range(H):
        for c in range(W):
            vals=_cross4_vals_any(g,r,c)
            if vals and len(set(vals))==1 and vals[0]!=0:
                counts[vals[0]] += 1
    best_c = 0; best_n = -1
    for c in range(1,10):
        n = counts[c]
        if n>best_n or (n==best_n and c<best_c):
            best_c, best_n = c, n
    return best_c if best_n>0 else 0
def sel_color_uniform_ring_mode(x_hat: np.ndarray) -> int:
    g = np.asarray(x_hat, dtype=int); H,W = g.shape; picks=[]
    for r in range(H):
        for c in range(W):
            vals=_ring8_vals(g,r,c)
            if vals and len(set(vals))==1 and vals[0]!=0:
                picks.append(vals[0])
    if not picks:
        vals, cnt = np.unique(g[g!=0], return_counts=True)
        return int(vals[np.argmax(cnt)]) if len(vals) else 0
    return _mode_int(picks)
COLOR_RULES: List[Tuple[str, Callable[[np.ndarray], int]]] = [
    ("argmin_hist", rule_argmin_hist),
    ("max_id", rule_max_id),
    ("uniform_cross_at_peaks_mode", sel_color_uniform_cross_at_luma_peaks),
    ("uniform_cross_everywhere_mode", sel_color_uniform_cross_everywhere_mode),
    ("argmax_uniform_cross_color_count", sel_color_argmax_uniform_cross_color_count),
]
# ===================== Enumeration & Printing =====================
def enumerate_programs_for_task(task: Dict, num_preops: int = 200, seed: int = 11):
    train_pairs = [(np.array(ex["input"], dtype=int), int(ex["output"][0][0])) for ex in task["train"]]
    preops = build_preops_for_dataset(train_pairs, num_preops=num_preops, seed=seed)
    # G core
    total_G = len(preops) * len(COLOR_RULES); valid_G=[]
    for pre_name, pre_f in preops:
        for cn, cf in COLOR_RULES:
            ok=True
            for x,y in train_pairs:
                if int(cf(pre_f(x))) != y:
                    ok=False; break
            if ok: valid_G.append((pre_name, cn))
    # Overlay + predicate
    total_ABS = len(preops); valid_ABS=[]
    for pre_name, pre_f in preops:
        ok=True
        for x,y in train_pairs:
            y_pred = predict_bright_overlay_uniform_cross(pre_f(x))
            if y_pred != y:
                ok=False; break
        if ok: valid_ABS.append(pre_name)
    programs_G = [f"{pre} |> {cn}" for (pre,cn) in valid_G]
    programs_ABS = [f"{pre} |> BrightOverlayIdentity |> UniformCrossPattern |> OutputAgreedColor" for pre in valid_ABS]
    return {"G":{"nodes": total_G, "programs": programs_G},
            "ABS":{"nodes": total_ABS, "programs": programs_ABS}}
def print_programs_for_task(task: Dict, num_preops: int = 200, seed: int = 11):
    res = enumerate_programs_for_task(task, num_preops=num_preops, seed=seed)
    print("=== Node counts ===")
    print(f"G core nodes: {res['G']['nodes']}")
    print(f"Overlay+predicate nodes: {res['ABS']['nodes']}")
    print("\n=== Programs found (G core) ===")
    if res["G"]["programs"]:
        for s in res["G"]["programs"]: print("-", s)
    else:
        print("(none)")
    print("\n=== Programs found (overlay abstraction + pattern check) ===")
    if res["ABS"]["programs"]:
        for s in res["ABS"]["programs"]: print("-", s)
    else:
        print("(none)")
    return res
def measure_spaces(task: Dict, num_preops: int = 200, seed: int = 11):
    import time
    train_pairs = [(np.array(ex["input"], dtype=int), int(ex["output"][0][0])) for ex in task["train"]]
    preops = build_preops_for_dataset(train_pairs, num_preops=num_preops, seed=seed)
    # G
    t0=time.perf_counter(); valid_G=[]; tried=0; tries_first=None; found=False
    for pre_name, pre_f in preops:
        for cn, cf in COLOR_RULES:
            tried+=1; ok=True
            for x,y in train_pairs:
                if int(cf(pre_f(x))) != y: ok=False; break
            if ok:
                valid_G.append((pre_name, cn))
                if not found: tries_first=tried; found=True
    t1=time.perf_counter()
    # ABS
    t2=time.perf_counter(); valid_ABS=[]; tried2=0; tries_first2=None; found2=False
    for pre_name, pre_f in preops:
        tried2+=1; ok=True
        for x,y in train_pairs:
            y_pred = predict_bright_overlay_uniform_cross(pre_f(x))
            if y_pred != y: ok=False; break
        if ok:
            valid_ABS.append(pre_name)
            if not found2: tries_first2=tried2; found2=True
    t3=time.perf_counter()
    return {
        "G":{"nodes": len(preops)*len(COLOR_RULES), "programs_found": len(valid_G),
             "tries_to_first": tries_first, "time_sec": t1-t0},
        "ABS":{"nodes": len(preops), "programs_found": len(valid_ABS),
               "tries_to_first": tries_first2, "time_sec": t3-t2},
    }
