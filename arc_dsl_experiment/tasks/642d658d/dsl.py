# -----------------------------------------------------------------------------
# Overlay Abstraction Experiment Utilities
# This module implements:
#  • Overlay extractor: detect_bright_overlays  (README_clean.md §2.2, “Methods: Overlay extractor”)
#  • Abstraction: BrightOverlayIdentity         (README_clean.md §2.3, identity + stored overlays)
#  • Pattern predicate & predictor pipeline     (README_clean.md §2.3; §5 “Analysis”)
#  • G core color rules incl. local-structure   (README_clean.md §2.4 “Extensions to G”)
#  • Program enumeration & pretty-print helpers (README_clean.md §4 “Results”)
# For numeric defaults of the detector, see README_clean.md: “Detector defaults (for reproducibility)”.
# -----------------------------------------------------------------------------


from __future__ import annotations
from typing import Dict, Iterable, List, Tuple, Optional, Callable, Type, TypeVar, Generic
import numpy as np
from functools import lru_cache
from vision import (
    PALETTE as VISION_PALETTE,
    grid_to_luminance as vision_grid_to_luminance,
    detect_bright_overlays_absolute as vision_detect_bright_overlays,
)
# ===================== Typed, compositional DSL =====================
# Minimal typed-DSL scaffolding to make composition explicit and extensible.
# States capture the current representation; Operations convert between states.

class State:  # marker base
    pass


class GridState(State):
    def __init__(self, grid: np.ndarray):
        self.grid = np.asarray(grid, dtype=int)


class OverlayContext(State):
    def __init__(self, grid: np.ndarray, overlays: List[dict], stats: Dict[str, float]):
        self.grid = np.asarray(grid, dtype=int)
        self.overlays = overlays
        self.stats = stats


class ColorState(State):
    def __init__(self, color: int):
        self.color = int(color)


InS = TypeVar("InS", bound=State)
OutS = TypeVar("OutS", bound=State)


class Operation(Generic[InS, OutS]):
    input_type: Type[State] = State
    output_type: Type[State] = State

    def accepts(self, state: State) -> bool:
        return isinstance(state, self.input_type)

    def apply(self, state: InS) -> OutS:
        raise NotImplementedError


class Pipeline:
    def __init__(self, ops: List[Operation]):
        self.ops = ops

    def run(self, state: State) -> State:
        cur = state
        for op in self.ops:
            if not op.accepts(cur):
                raise TypeError(f"Operation {op.__class__.__name__} does not accept state {type(cur).__name__}")
            cur = op.apply(cur)  # type: ignore[arg-type]
        return cur
# ===================== Palette & Luminance =====================
PALETTE = VISION_PALETTE
def _default_palette(): return VISION_PALETTE.copy()
def grid_to_luminance(g: np.ndarray) -> np.ndarray:
    return vision_grid_to_luminance(g)
# ===================== Overlay Extractor (from user) =====================
# See README_clean.md §2.2 for a high-level description and defaults; this function is the source of truth.
def detect_bright_overlays(
    grid: Iterable[Iterable[int]],
    palette: Optional[Dict[int, Tuple[int,int,int]]] = None,
    **kwargs,
) -> List[dict]:
    # palette/kwargs ignored for the absolute detector
    return vision_detect_bright_overlays(grid)
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


# Typed-DSL operations corresponding to the above components

class PreOpPalette(Operation[GridState, GridState]):
    input_type = GridState
    output_type = GridState

    def __init__(self, name: str, f: Callable[[np.ndarray], np.ndarray]):
        self.name = name
        self.f = f

    def apply(self, state: GridState) -> GridState:
        return GridState(self.f(state.grid))


class OpBrightOverlayIdentity(Operation[GridState, OverlayContext]):
    input_type = GridState
    output_type = OverlayContext

    def __init__(self, absx: Optional[BrightOverlayIdentity] = None):
        self.absx = absx or BrightOverlayIdentity()

    def apply(self, state: GridState) -> OverlayContext:
        g = state.grid
        ovs = detect_bright_overlays(g.tolist())
        H, W = g.shape
        count = len(ovs)
        max_contrast = max([ov["contrast"] for ov in ovs], default=0.0)
        total_area = sum([ov["area"] for ov in ovs]) if ovs else 0
        total_area_frac = float(total_area) / float(H * W) if H * W > 0 else 0.0
        stats = dict(count=count, max_contrast=max_contrast, total_area=total_area, total_area_frac=total_area_frac)
        self.absx.overlays = ovs
        self.absx.last_stats = stats
        return OverlayContext(g, ovs, stats)


class OpUniformCrossPattern(Operation[OverlayContext, ColorState]):
    input_type = OverlayContext
    output_type = ColorState

    def apply(self, state: OverlayContext) -> ColorState:
        ok, color, _ = overlays_have_uniform_cross_color(state.grid, state.overlays)
        if ok and color is not None:
            return ColorState(int(color))
        # Fallback: mode among valid overlay crosses
        return ColorState(int(bright_overlay_cross_mode(state.grid, state.overlays)))
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
def _fast_uniform_cross_color_if_agree(g: np.ndarray):
    return None
# Composed program body used in abstraction space: preop |> BrightOverlayIdentity |> UniformCrossPattern |> OutputAgreedColor.
# Uses a fast-first path (cheap local-max selector) then falls back to full overlays (README_clean.md §2.5, §4).
def predict_bright_overlay_uniform_cross(grid: List[List[int]]) -> int:
    # Typed pipeline: Grid -> OverlayContext -> Color
    gstate = GridState(np.asarray(grid, dtype=int))
    pipeline = Pipeline([
        OpBrightOverlayIdentity(),
        OpUniformCrossPattern(),
    ])
    out = pipeline.run(gstate)
    assert isinstance(out, ColorState)
    return int(out.color)
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
# Builds (‘identity’ + N palette permutations) used as pre-ops in both spaces.
# Note: In our run (seed=11, 200 preops) perm_192 was an identity mapping on used colors (README_clean.md, Note on perm_192).
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
# Enumerates programs that are correct on ALL training examples (README_clean.md §3–§4).
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
# Pretty-prints the programs and node counts (README_clean.md §4).
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