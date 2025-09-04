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

    # Render pictures (train + test)
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
        ax = axes[0]
        ax.imshow(rgb, interpolation='nearest')
        ax.set_title(title, fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
        for ov in overlays:
            y1,x1,y2,x2 = ov["y1"]-1, ov["x1"]-1, ov["y2"]-1, ov["x2"]-1
            rect = Rectangle((x1 - 0.5, y1 - 0.5), (x2 - x1 + 1), (y2 - y1 + 1), fill=False, linewidth=1.5, edgecolor='yellow')
            ax.add_patch(rect)
        ax.set_xlim(-0.5, W-0.5); ax.set_ylim(H-0.5, -0.5)

        ax2 = axes[1]
        tile = np.full((20, 20, 3), PALETTE.get(int(pred_color), (0,0,0)), dtype=np.uint8)
        ax2.imshow(tile, interpolation='nearest')
        ax2.set_title(f"Predicted Color: {int(pred_color)}", fontsize=12)
        ax2.set_xticks([]); ax2.set_yticks([])
        plt.tight_layout(); plt.savefig(out_path, bbox_inches="tight"); plt.close(fig); return out_path

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