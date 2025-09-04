# -----------------------------------------------------------------------------
# Reproduction script for “Overlay abstraction vs. core G” experiment
# This script:
#  • Loads task.json and enumerates program spaces (G vs. Abstraction) and checks ALL train
#  • Prints programs found & node counts/timings                            (README_clean.md §4)
#  • Renders annotated images (overlays + predicted color)                  (README_clean.md Figures)
# No behavior changes are made; this is the exact harness used for results.
# -----------------------------------------------------------------------------


import json, time, numpy as np
from importlib import reload
import sys
from pathlib import Path

# Ensure local imports resolve to this folder's dsl.py
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import dsl
dsl = reload(dsl)

TASK_PATH = str(HERE / "task.json")

def main():
    task = json.loads(Path(TASK_PATH).read_text())
    train_pairs = [(np.array(ex["input"], dtype=int), int(ex["output"][0][0])) for ex in task["train"]]

    # Enumerate and print programs (composed form)
    # Pretty print programs found in both spaces (README_clean.md §4):
    res = dsl.print_programs_for_task(task, num_preops=200, seed=11)

    # Measure timing and counts (single pass, rounded to reduce noise)
    def measure(task, num_preops=200, seed=11):
        preops = dsl.build_preops_for_dataset(train_pairs, num_preops=num_preops, seed=seed)

        # G core
        color_rules = dsl.COLOR_RULES
        t0 = time.perf_counter()
        valid_G = []
        tried = 0; tries_first_G=None
        for pre_name, pre_f in preops:
            for cn, cf in color_rules:
                tried += 1
                ok = True
                for x,y in train_pairs:  # <-- CHECK on ALL training examples (selection criterion)
                    if int(cf(pre_f(x))) != y:
                        ok=False; break
                if ok:
                    valid_G.append((pre_name, cn))
                    if tries_first_G is None: tries_first_G = tried
        t1 = time.perf_counter()

        # Abstraction
        t2 = time.perf_counter()
        valid_ABS = []
        tried2 = 0; tries_first_ABS=None
        for pre_name, pre_f in preops:
            tried2 += 1
            ok=True
            for x,y in train_pairs:  # <-- CHECK on ALL training examples (selection criterion)
                y_pred = dsl.predict_bright_overlay_uniform_cross(pre_f(x))
                if y_pred != y:
                    ok=False; break
            if ok:
                valid_ABS.append(pre_name)
                if tries_first_ABS is None: tries_first_ABS = tried2
        t3 = time.perf_counter()

        return {
            "G": {"nodes": len(preops)*len(color_rules), "programs_found": len(valid_G), "tries_to_first": tries_first_G, "time_sec": round(t1-t0, 2)},
            "ABS":{"nodes": len(preops), "programs_found": len(valid_ABS), "tries_to_first": tries_first_ABS, "time_sec": round(t3-t2, 2)},
        }

    stats = measure(task, num_preops=200, seed=11)
    print("\n=== STATS (200 preops) ===")
    print(stats)
    # Quick evaluation of other pattern kinds (v3, cross3) on train
    def eval_kind(kind: str):
        res = []
        for x,y in train_pairs:
            yp = dsl.predict_with_pattern_kind(x.tolist(), kind)
            res.append(int(yp==y))
        return sum(res), len(res)

    k_v3_ok, k_total = eval_kind("v3_yellow")
    k_cross_ok, _ = eval_kind("cross3_yellow")
    print(f"\n=== Pattern kind eval (train acc) ===\n v3: {k_v3_ok}/{k_total}\n cross3: {k_cross_ok}/{k_total}")
    # Test-set predictions (no GT in ARC test; we just report)
    gtest = np.array(task["test"][0]["input"], dtype=int)
    pred_v3 = dsl.predict_with_pattern_kind(gtest.tolist(), "v3_yellow")
    pred_cross3 = dsl.predict_with_pattern_kind(gtest.tolist(), "cross3_yellow")
    print(f"Test predictions: v3={pred_v3} cross3={pred_cross3}")

    # Verify overlay counts for h3_yellow vs count of yellow pixels
    def count_yellow(g):
        return int((np.array(g)==4).sum())
    for split in ("train","test"):
        exs = task[split]
        for i, ex in enumerate(exs, start=1):
            g = ex["input"]
            n_y = count_yellow(g)
            ovs_h3y = dsl.detect_overlays(g, kind="h3_yellow")
            print(f"{split}[{i}] yellow={n_y} overlays_h3_yellow={len(ovs_h3y)}")
    # Emit a tracked artifact with stable ordering
    out_stats = HERE / "repro_stats.json"
    with open(out_stats, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, sort_keys=True)
    print(f"Wrote {out_stats}")

    # Emit detailed stats for overlays/predicate per example
    def per_example_details(task):
        details = {"train": [], "test": []}
        # absolute detector to sync with OCaml; plus compare with pattern abstraction
        for ex in task["train"]:
            g = ex["input"]
            ovs = dsl.detect_overlays(g, kind="cross3_yellow")
            # pattern overlays
            ovs_h3 = dsl.detect_overlays(g, kind="h3_yellow")
            ovs_v3 = dsl.detect_overlays(g, kind="v3_yellow")
            ovs_cross3 = dsl.detect_overlays(g, kind="cross3_yellow")
            entries = []
            H, W = len(g), len(g[0])
            for ov in ovs:
                r, c = ov["center_row"], ov["center_col"]
                vals = [g[r-2][c-1], g[r][c-1], g[r-1][c-2], g[r-1][c]] if (1<=r-1<H-1 and 1<=c-1<W-1) else []
                col = vals[0] if len(vals)==4 and len(set(vals))==1 and vals[0]!=0 else None
                entries.append(((r, c), col))
            entries.sort(key=lambda x: (x[0][0], x[0][1]))
            centers = [[r,c] for (r,c),_ in entries]
            cols = [col for _,col in entries if col is not None]
            # subset/equality checks for centers
            def centers_of(ovs):
                return sorted([[ov["center_row"], ov["center_col"]] for ov in ovs])
            centers_abs = centers_of(ovs)
            centers_h3 = centers_of(ovs_h3)
            centers_v3 = centers_of(ovs_v3)
            centers_cross3 = centers_of(ovs_cross3)
            # filter ABS centers by local pattern predicate
            def is_h3_center(r, c):
                rr, cc = r-1, c-1
                if 0<=cc-1 and cc+1<W and 0<=rr<H:
                    a, b, c3 = g[rr][cc-1], g[rr][cc], g[rr][cc+1]
                    return a == b == c3 != 0
                return False
            def is_v3_center(r, c):
                rr, cc = r-1, c-1
                if 0<=rr-1 and rr+1<H and 0<=cc<W:
                    a, b, c3 = g[rr-1][cc], g[rr][cc], g[rr+1][cc]
                    return a == b == c3 != 0
                return False
            def is_square3_center(r, c):
                rr, cc = r-1, c-1
                if 0<=rr-1 and rr+1<H and 0<=cc-1 and cc+1<W:
                    block = [g[rr-1][cc-1], g[rr-1][cc], g[rr-1][cc+1],
                             g[rr][cc-1],   g[rr][cc],   g[rr][cc+1],
                             g[rr+1][cc-1], g[rr+1][cc], g[rr+1][cc+1]]
                    return len(set(block))==1 and block[0]!=0
                return False
            abs_h3 = sorted([xy for xy in centers_abs if is_h3_center(*xy)])
            abs_v3 = sorted([xy for xy in centers_abs if is_v3_center(*xy)])
            abs_square3 = sorted([xy for xy in centers_abs if is_square3_center(*xy)])
            def subset(a,b):
                sb = set(map(tuple,a))
                bb = set(map(tuple,b))
                return sb.issubset(bb)
            details["train"].append({
                "centers": centers,
                "uniform_cross_colors": cols,
                "pred": (min(cols) if cols else 0),
                "gt": ex["output"][0][0],
                "centers_abs": centers_abs,
                "centers_h3": centers_h3,
                "centers_v3": centers_v3,
                "centers_cross3": centers_cross3,
                "abs_h3": abs_h3,
                "abs_v3": abs_v3,
                "abs_square3": abs_square3,
                "h3_equal_abs_h3": centers_h3 == abs_h3,
                "v3_equal_abs_v3": centers_v3 == abs_v3,
                "cross3_equal_abs_cross3": centers_cross3 == abs_square3,  # compare against 3x3 box-filtered abs
            })
        for ex in task["test"]:
            g = ex["input"]
            ovs = dsl.detect_overlays(g, kind="cross3_yellow")
            ovs_h3 = dsl.detect_overlays(g, kind="h3_yellow")
            ovs_v3 = dsl.detect_overlays(g, kind="v3_yellow")
            ovs_cross3 = dsl.detect_overlays(g, kind="cross3_yellow")
            entries = []
            H, W = len(g), len(g[0])
            for ov in ovs:
                r, c = ov["center_row"], ov["center_col"]
                vals = [g[r-2][c-1], g[r][c-1], g[r-1][c-2], g[r-1][c]] if (1<=r-1<H-1 and 1<=c-1<W-1) else []
                col = vals[0] if len(vals)==4 and len(set(vals))==1 and vals[0]!=0 else None
                entries.append(((r, c), col))
            entries.sort(key=lambda x: (x[0][0], x[0][1]))
            centers = [[r,c] for (r,c),_ in entries]
            cols = [col for _,col in entries if col is not None]
            def centers_of(ovs):
                return sorted([[ov["center_row"], ov["center_col"]] for ov in ovs])
            centers_abs = centers_of(ovs)
            centers_h3 = centers_of(ovs_h3)
            centers_v3 = centers_of(ovs_v3)
            centers_cross3 = centers_of(ovs_cross3)
            def is_h3_center(r, c):
                rr, cc = r-1, c-1
                if 0<=cc-1 and cc+1<W and 0<=rr<H:
                    a, b, c3 = g[rr][cc-1], g[rr][cc], g[rr][cc+1]
                    return a == b == c3 != 0
                return False
            def is_v3_center(r, c):
                rr, cc = r-1, c-1
                if 0<=rr-1 and rr+1<H and 0<=cc<W:
                    a, b, c3 = g[rr-1][cc], g[rr][cc], g[rr+1][cc]
                    return a == b == c3 != 0
                return False
            def is_square3_center(r, c):
                rr, cc = r-1, c-1
                if 0<=rr-1 and rr+1<H and 0<=cc-1 and cc+1<W:
                    block = [g[rr-1][cc-1], g[rr-1][cc], g[rr-1][cc+1],
                             g[rr][cc-1],   g[rr][cc],   g[rr][cc+1],
                             g[rr+1][cc-1], g[rr+1][cc], g[rr+1][cc+1]]
                    return len(set(block))==1 and block[0]!=0
                return False
            abs_h3 = sorted([xy for xy in centers_abs if is_h3_center(*xy)])
            abs_v3 = sorted([xy for xy in centers_abs if is_v3_center(*xy)])
            abs_square3 = sorted([xy for xy in centers_abs if is_square3_center(*xy)])
            def subset(a,b):
                sb = set(map(tuple,a))
                bb = set(map(tuple,b))
                return sb.issubset(bb)
            details["test"].append({
                "centers": centers,
                "uniform_cross_colors": cols,
                "centers_abs": centers_abs,
                "centers_h3": centers_h3,
                "centers_v3": centers_v3,
                "centers_cross3": centers_cross3,
                "abs_h3": abs_h3,
                "abs_v3": abs_v3,
                "abs_square3": abs_square3,
                "h3_equal_abs_h3": centers_h3 == abs_h3,
                "v3_equal_abs_v3": centers_v3 == abs_v3,
                "cross3_equal_abs_cross3": centers_cross3 == abs_square3,
            })
        return details

    py_details = per_example_details(task)
    with open(HERE / "pattern_stats.json", "w", encoding="utf-8") as f:
        json.dump(py_details, f, separators=(",", ":"), sort_keys=False)
        f.write("\n")
    print("Wrote", HERE / "pattern_stats.json")

    # Render pictures (train + test)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.lines import Line2D

    PALETTE = {
        0:(0,0,0), 1:(0,0,255), 2:(255,0,0), 3:(0,255,0), 4:(255,255,0),
        5:(128,128,128), 6:(255,192,203), 7:(255,165,0), 8:(0,128,128), 9:(139,69,19)
    }
    def grid_to_rgb(g, palette=PALETTE):
        g = np.asarray(g, dtype=int)
        H, W = g.shape
        img = np.zeros((H, W, 3), dtype=np.uint8)
        for k, (r, gg, b) in palette.items():
            img[g == k] = (r, gg, b)
        return img

    def render_grid_with_overlays(g, pred_color, title, out_path, kind="h3_yellow"):
        rgb = grid_to_rgb(g)
        overlays = dsl.detect_overlays(g.tolist(), kind=kind)
        H, W = g.shape
        fig, axes = plt.subplots(1, 2, figsize=(12, 8), dpi=150)
        fig.canvas.toolbar_visible = False
        fig.canvas.header_visible = False
        fig.canvas.footer_visible = False
        ax = axes[0]
        ax.imshow(rgb, interpolation='nearest')
        ax.set_title(title, fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
        overlays_sorted = sorted(overlays, key=lambda ov: (ov["center_row"], ov["center_col"]))
        for ov in overlays_sorted:
            if kind == "cross3_yellow":
                rr = ov["center_row"] - 1
                cc = ov["center_col"] - 1
                # vertical arm (3 cells): from rr-1 to rr+1 at column cc
                ax.add_line(Line2D([cc, cc], [rr - 1 - 0.5, rr + 1 + 0.5], color='yellow', linewidth=2.0))
                # horizontal arm (3 cells): from cc-1 to cc+1 at row rr
                ax.add_line(Line2D([cc - 1 - 0.5, cc + 1 + 0.5], [rr, rr], color='yellow', linewidth=2.0))
            else:
                y1,x1,y2,x2 = ov["y1"]-1, ov["x1"]-1, ov["y2"]-1, ov["x2"]-1
                rect = Rectangle((x1 - 0.5, y1 - 0.5), (x2 - x1 + 1), (y2 - y1 + 1), fill=False, linewidth=1.5, edgecolor='yellow')
                ax.add_patch(rect)
        ax.set_xlim(-0.5, W-0.5); ax.set_ylim(H-0.5, -0.5)

        ax2 = axes[1]
        tile = np.full((20, 20, 3), PALETTE.get(int(pred_color), (0,0,0)), dtype=np.uint8)
        ax2.imshow(tile, interpolation='nearest')
        ax2.set_title(f"Predicted Color: {int(pred_color)}", fontsize=12)
        ax2.set_xticks([]); ax2.set_yticks([])
        plt.tight_layout(); plt.savefig(out_path, bbox_inches="tight", metadata={}); plt.close(fig); return out_path

    # Train pics
    images_dir = HERE / "images"
    images_dir.mkdir(exist_ok=True)

    for i, ex in enumerate(task["train"], start=1):
        g = np.array(ex["input"], dtype=int)
        pred = dsl.predict_bright_overlay_uniform_cross(g.tolist())
        render_grid_with_overlays(
            g,
            pred,
            f"Train {i}: h3_yellow (GT={int(ex['output'][0][0])})",
            str(images_dir / f"overlay_train_{i}.png"),
            kind="h3_yellow",
        )
        render_grid_with_overlays(
            g,
            pred,
            f"Train {i}: v3_yellow (GT={int(ex['output'][0][0])})",
            str(images_dir / f"overlay_train_{i}_v.png"),
            kind="v3_yellow",
        )
        render_grid_with_overlays(
            g,
            pred,
            f"Train {i}: cross3_yellow (GT={int(ex['output'][0][0])})",
            str(images_dir / f"overlay_train_{i}_x.png"),
            kind="cross3_yellow",
        )

    # Test pic
    gtest = np.array(task["test"][0]["input"], dtype=int)
    predt = dsl.predict_bright_overlay_uniform_cross(gtest.tolist())
    render_grid_with_overlays(gtest, predt, "Test: h3_yellow", str(images_dir / "overlay_test.png"), kind="h3_yellow")
    render_grid_with_overlays(gtest, predt, "Test: v3_yellow", str(images_dir / "overlay_test_v.png"), kind="v3_yellow")
    render_grid_with_overlays(gtest, predt, "Test: cross3_yellow", str(images_dir / "overlay_test_x.png"), kind="cross3_yellow")

if __name__ == "__main__":
    main()