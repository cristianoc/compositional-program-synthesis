import json
import os
from typing import List, Tuple

import numpy as np


def get_arc_colormap() -> Tuple[List[Tuple[float, float, float]], List[str]]:
    # ARC standard palette for colors 0-9 (RGB in 0-1 range)
    # 0: black, 1: blue, 2: red, 3: green, 4: yellow, 5: gray, 6: pink,
    # 7: orange, 8: teal, 9: maroon — approximate choices for clarity
    rgb = [
        (0.0, 0.0, 0.0),       # 0 black
        (0.2, 0.4, 1.0),       # 1 blue
        (1.0, 0.2, 0.2),       # 2 red
        (0.2, 0.7, 0.2),       # 3 green
        (1.0, 0.9, 0.2),       # 4 yellow
        (0.6, 0.6, 0.6),       # 5 gray
        (1.0, 0.6, 0.8),       # 6 pink
        (1.0, 0.6, 0.2),       # 7 orange
        (0.2, 0.8, 0.8),       # 8 teal
        (0.6, 0.0, 0.2),       # 9 maroon
    ]
    names = ["black","blue","red","green","yellow","gray","pink","orange","teal","maroon"]
    return rgb, names


def save_grid_png(grid: List[List[int]], out_path: str, scale: int = 20) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
    except Exception as exc:
        raise RuntimeError("matplotlib is required to render images") from exc

    rgb, _ = get_arc_colormap()
    cmap = ListedColormap(rgb)

    arr = np.array(grid, dtype=np.int32)
    h, w = arr.shape
    dpi = 100
    fig_w = max(1, w * scale / dpi)
    fig_h = max(1, h * scale / dpi)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.imshow(arr, cmap=cmap, vmin=0, vmax=9, interpolation="nearest")
    ax.set_axis_off()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def make_overview(train_pairs: List[Tuple[List[List[int]], List[List[int]]]], test_inputs: List[List[List[int]]], out_path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
    except Exception as exc:
        raise RuntimeError("matplotlib is required to render images") from exc

    rgb, _ = get_arc_colormap()
    cmap = ListedColormap(rgb)

    # Compose a simple overview with first train pair and first test input if present
    ncols = 3
    fig, axes = plt.subplots(1, ncols, figsize=(9, 3), dpi=150)

    def show(ax, grid, title: str):
        arr = np.array(grid, dtype=np.int32)
        ax.imshow(arr, cmap=cmap, vmin=0, vmax=9, interpolation="nearest")
        ax.set_title(title)
        ax.set_axis_off()

    if len(train_pairs) > 0:
        show(axes[0], train_pairs[0][0], "train[0] input")
        show(axes[1], train_pairs[0][1], "train[0] output")
    else:
        axes[0].set_axis_off()
        axes[1].set_axis_off()

    if len(test_inputs) > 0:
        show(axes[2], test_inputs[0], "test[0] input")
    else:
        axes[2].set_axis_off()

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    task_path = os.path.join(here, "task.json")
    images_dir = os.path.join(here, "images")
    os.makedirs(images_dir, exist_ok=True)

    with open(task_path, "r") as f:
        data = json.load(f)

    # Render training pairs
    train_pairs = []
    for i, pair in enumerate(data.get("train", [])):
        inp = pair["input"]
        out = pair["output"]
        train_pairs.append((inp, out))
        save_grid_png(inp, os.path.join(images_dir, f"train_{i}_in.png"))
        save_grid_png(out, os.path.join(images_dir, f"train_{i}_out.png"))

    # Render test inputs
    test_inputs = []
    for i, pair in enumerate(data.get("test", [])):
        inp = pair["input"]
        test_inputs.append(inp)
        save_grid_png(inp, os.path.join(images_dir, f"test_{i}_in.png"))

    # Overview image
    make_overview(train_pairs, test_inputs, os.path.join(images_dir, "overview.png"))


if __name__ == "__main__":
    main()


