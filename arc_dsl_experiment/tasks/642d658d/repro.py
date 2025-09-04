# -----------------------------------------------------------------------------
# Reproduction script for “Overlay abstraction vs. core G” experiment
# This script:
#  • Loads task.json and builds pre-ops (palette permutations + identity)   (README_clean.md §3)
#  • Enumerates program spaces (G vs. Abstraction) and checks ALL train     (README_clean.md §3–4)
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

    # Measure timing and counts
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
            "G": {"nodes": len(preops)*len(color_rules), "programs_found": len(valid_G), "tries_to_first": tries_first_G, "time_sec": t1-t0},
            "ABS":{"nodes": len(preops), "programs_found": len(valid_ABS), "tries_to_first": tries_first_ABS, "time_sec": t3-t2},
        }

    stats = measure(task, num_preops=200, seed=11)
    print("\n=== STATS (200 preops) ===")
    print(stats)
    # Emit a tracked artifact with stable ordering
    out_stats = HERE / "repro_stats.json"
    with open(out_stats, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, sort_keys=True)
    print(f"Wrote {out_stats}")

    # Emit detailed stats for overlays/predicate per example
    def per_example_details(task):
        details = {"train": [], "test": []}
        # absolute detector to sync with OCaml
        for ex in task["train"]:
            g = ex["input"]
            ovs = dsl.detect_bright_overlays(g)
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
            details["train"].append({
                "centers": centers,
                "uniform_cross_colors": cols,
                "pred": (min(cols) if cols else 0),
                "gt": ex["output"][0][0],
            })
        for ex in task["test"]:
            g = ex["input"]
            ovs = dsl.detect_bright_overlays(g)
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
            details["test"].append({
                "centers": centers,
                "uniform_cross_colors": cols,
            })
        return details

    py_details = per_example_details(task)
    with open(HERE / "python_stats.json", "w", encoding="utf-8") as f:
        json.dump(py_details, f, separators=(",", ":"), sort_keys=False)
        f.write("\n")
    print("Wrote", HERE / "python_stats.json")

    # Render pictures (train + test)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

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

    def render_grid_with_overlays(g, pred_color, title, out_path):
        rgb = grid_to_rgb(g)
        overlays = dsl.detect_bright_overlays(g.tolist())
        H, W = g.shape
        fig, axes = plt.subplots(1, 2, figsize=(12, 8), dpi=150)
        fig.canvas.toolbar_visible = False
        fig.canvas.header_visible = False
        fig.canvas.footer_visible = False
        ax = axes[0]
        ax.imshow(rgb, interpolation='nearest')
        ax.set_title(title, fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
        # deterministic draw order: sort overlays by (center_row, center_col)
        overlays_sorted = sorted(overlays, key=lambda ov: (ov["center_row"], ov["center_col"]))
        for ov in overlays_sorted:
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
            f"Train {i}: overlays (GT={int(ex['output'][0][0])})",
            str(images_dir / f"overlay_train_{i}.png"),
        )

    # Test pic
    gtest = np.array(task["test"][0]["input"], dtype=int)
    predt = dsl.predict_bright_overlay_uniform_cross(gtest.tolist())
    render_grid_with_overlays(
        gtest, predt, "Test: overlays", str(images_dir / "overlay_test.png")
    )

if __name__ == "__main__":
    main()