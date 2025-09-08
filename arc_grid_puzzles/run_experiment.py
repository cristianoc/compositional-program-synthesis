# -----------------------------------------------------------------------------
# Generic Experiment Runner for Pattern Abstraction
# This script:
#  • Loads any task.json and enumerates program spaces (G vs. Abstraction)
#  • Prints programs found & node counts/timings
#  • Renders annotated images (universal matches + predicted color)
# Usage: python run_experiment.py <task_path>
# Example: python run_experiment.py experiments/642d658d/task.json
# -----------------------------------------------------------------------------

import json, numpy as np, os, argparse
from typing import Optional, Any, Union
from importlib import reload
import sys
from pathlib import Path

# Import from the reorganized structure
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from program_search import enumerate_programs_for_task
from dsl_types.states import Grid, Pipeline
from dsl_types.grid_to_matches import OpMatchAnyUniversalSchemas


def main():
    parser = argparse.ArgumentParser(description='Run pattern abstraction experiment on ARC task')
    parser.add_argument('task_path', help='Path to task.json file')
    parser.add_argument('--output-dir', help='Output directory for results (default: same as task directory)')
    args = parser.parse_args()
    
    task_path = Path(args.task_path)
    if not task_path.exists():
        print(f"Error: Task file not found: {task_path}")
        sys.exit(1)
    
    task = json.loads(task_path.read_text())
    train_pairs = [(np.array(ex["input"], dtype=int), int(ex["output"][0][0])) for ex in task["train"]]

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = task_path.parent
    
    output_dir.mkdir(parents=True, exist_ok=True)

    # Enumerate abstraction operations
    SHAPES = [(1,3),(3,1),(2,3),(3,3),(5,5)]
    res_once = enumerate_programs_for_task(task, num_preops=200, seed=11, universal_shapes=SHAPES)
    # Print and persist simple combined JSON
    programs_path = output_dir / "programs_found.json"
    try:
        # Get node counts from the enumeration result
        abs_nodes = res_once['ABS']['nodes']
        print("=== Node counts ===")
        print(f"Pattern matching nodes: {abs_nodes}")

        print("\n=== Programs found (pattern matching) ===")
        abs_progs = res_once['ABS']['programs']
        if abs_progs:
            for sname in abs_progs:
                print("-", sname)
        else:
            print("(none)")

        # Save JSON without program sequences and timing (which aren't JSON serializable or change every run)
        json_output = {}
        for k, v in res_once.items():
            json_output[k] = {}
            for subk, subv in v.items():
                if subk not in ["program_sequences", "time_sec"]:
                    json_output[k][subk] = subv
        
        with open(programs_path, "w", encoding="utf-8") as f:
            json.dump(json_output, f, indent=2, sort_keys=True)
        print("Wrote", programs_path)
    except Exception as e:
        print("[warn] failed to write programs_found.json:", e)

    # Print single-pass stats from enumeration result
    print("\n=== STATS (single-pass) ===")
    print({
        "ABS": {"nodes": res_once['ABS']['nodes'], "programs_found": len(res_once['ABS']['programs']), "time_sec": res_once['ABS'].get('time_sec')},
    })
    # Print intersected universal schemas per shape (train+test)
    try:
        print("\n=== Intersected universal schemas (train+test) ===")
        from dsl_types.grid_to_matches import build_intersected_universal_schemas_for_task
        for ushape in SHAPES:
            uni = build_intersected_universal_schemas_for_task(task, window_shape=tuple(ushape), center_value=4, splits=("train","test"))
            if not uni:
                print(f"shape {ushape}: none")
                continue
            print(f"shape {ushape}: {len(uni)} positions")
            for pos, sc in sorted(uni.items()):
                print(f"pos {pos}")
                cells = [str(x) for row in sc for x in row]
                w = max((len(c) for c in cells), default=1)
                for row in sc:
                    print(" " + "[" + ", ".join(f"{str(x):>{w}}" for x in row) + "]")
                print()
    except Exception as e:
        print(f"[warn] failed to print universal schemas: {e}")

    # Mosaic visualization using actual found programs

     # Render pictures (universal matches) without matplotlib for better determinism
    os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
    
    # Import shared image utilities
    from image_utils import PALETTE, YELLOW, grid_to_rgb, upsample, save_png, draw_rect_outline, draw_text, add_border

    # Universal matcher mosaic (replacement for overlay mosaic)
    try:
        images_dir = HERE / "images"
        images_dir.mkdir(exist_ok=True)

        def build_panel_body_universal(g, shape: tuple[int,int], schemas_list: list[list[list[Union[int, str]]]], pred_color: int) -> np.ndarray:
            # Draw base grid and rectangles for each match
            g = np.asarray(g, dtype=int)
            SCALE = 16
            base = upsample(grid_to_rgb(g), scale=SCALE)
            # collect matches using matcher only
            pipe_match = Pipeline([OpMatchAnyUniversalSchemas(schemas_list, label=f"match_universal_pos(shape={shape})")])
            mstate = pipe_match.run(Grid(g))
            matches = getattr(mstate, 'matches', [])
            # Show all pattern matches (structural complexity selection provides quality control)
            for ov in sorted(matches, key=lambda ov: (ov["y1"], ov["x1"])):
                y1,x1,y2,x2 = ov["y1"]-1, ov["x1"]-1, ov["y2"]-1, ov["x2"]-1
                draw_rect_outline(base, y1, x1, y2, x2, color=YELLOW, scale=SCALE)
            # Compose predicted color panel at right, same as before
            Hs = base.shape[0]
            spacer_w = 8
            spacer = np.full((Hs + 16, spacer_w, 3), 255, dtype=np.uint8)
            in_w = base.shape[1]
            out_w = max(in_w, SCALE)
            if in_w < out_w:
                pad = np.full((Hs, out_w - in_w, 3), 255, dtype=np.uint8)
                base_panel = np.concatenate([base, pad], axis=1)
            else:
                base_panel = base
            out_panel = np.full((Hs, out_w, 3), 255, dtype=np.uint8)
            out_panel[0:SCALE, 0:SCALE, :] = PALETTE.get(int(pred_color), (0,0,0))
            base_panel_b = add_border(base, 8)
            out_panel_b = add_border(out_panel, 8)
            bw = base_panel_b.shape[1]
            ow = out_panel_b.shape[1]
            target_w = max(bw, ow)
            if bw < target_w:
                extra = np.full((base_panel_b.shape[0], target_w - bw, 3), 255, dtype=np.uint8)
                base_panel_b = np.concatenate([base_panel_b, extra], axis=1)
            if ow < target_w:
                extra = np.full((out_panel_b.shape[0], target_w - ow, 3), 255, dtype=np.uint8)
                out_panel_b = np.concatenate([out_panel_b, extra], axis=1)
            # Match heights
            bh = base_panel_b.shape[0]
            oh = out_panel_b.shape[0]
            target_h = max(bh, oh)
            if bh < target_h:
                extra = np.full((target_h - bh, base_panel_b.shape[1], 3), 255, dtype=np.uint8)
                base_panel_b = np.concatenate([base_panel_b, extra], axis=0)
            if oh < target_h:
                extra = np.full((target_h - oh, out_panel_b.shape[1], 3), 255, dtype=np.uint8)
                out_panel_b = np.concatenate([out_panel_b, extra], axis=0)
            spacer = np.full((target_h, spacer_w, 3), 255, dtype=np.uint8)
            return np.concatenate([base_panel_b, spacer, out_panel_b], axis=1)

        def render_mosaic_universal_all_examples(task, out_path: str, found_programs):
            # Extract unique shapes and their programs from found sequences
            shape_to_program = {}  # shape -> (matcher, aggregator) tuple
            shape_to_schemas = {}   # shape -> schemas list for visualization
            
            for prog_name, seq in found_programs:
                if len(seq) == 2 and isinstance(seq[0], OpMatchAnyUniversalSchemas):
                    # Parse shape from matcher's label
                    label = getattr(seq[0], 'label', '')
                    if 'shape=' in label:
                        try:
                            shape_str = label.split('shape=')[1].split(')')[0] + ')'
                            shape = eval(shape_str)
                            # Use first found program for each shape (avoid duplicates)
                            if shape not in shape_to_program:
                                shape_to_program[shape] = (seq[0], seq[1])
                                # Extract schemas from the matcher (already properly typed)
                                matcher = seq[0]
                                if isinstance(matcher, OpMatchAnyUniversalSchemas):
                                    shape_to_schemas[shape] = matcher.schemas
                                else:
                                    shape_to_schemas[shape] = []
                        except:
                            pass
            
            # Use all available shapes (now that we select best pattern per shape)
            shapes = sorted(shape_to_program.keys())
            
            # Simple prediction function that directly uses the found program
            def predict_for_grid(g, shape):
                if shape not in shape_to_program:
                    return 0
                matcher, aggregator = shape_to_program[shape]
                pipe = Pipeline([matcher, aggregator])
                try:
                    out = pipe.run(Grid(np.asarray(g, dtype=int)))
                    return int(getattr(out, 'color', 0))
                except:
                    return 0

            # Build rows
            rows = []
            row_labels = []
            kinds = [(f"{h}x{w}", (h,w)) for (h,w) in shapes]
            for idx, ex in enumerate(task["train"], start=1):
                g = np.array(ex["input"], dtype=int)
                pan = []
                for _, shape in kinds:
                    pred = predict_for_grid(g, shape)
                    schemas_list = shape_to_schemas[shape]
                    pan.append(build_panel_body_universal(g, shape, schemas_list, pred))
                rows.append(pan); row_labels.append(f"TRAIN {idx}")
            for idx, ex in enumerate(task["test"], start=1):
                g = np.array(ex["input"], dtype=int)
                pan = []
                for _, shape in kinds:
                    pred = predict_for_grid(g, shape)
                    schemas_list = shape_to_schemas[shape]
                    pan.append(build_panel_body_universal(g, shape, schemas_list, pred))
                rows.append(pan); row_labels.append(f"TEST {idx}")

            # Equalize panel widths per column
            ncols = len(kinds)
            nrows = len(rows)
            col_widths = [0]*ncols
            row_heights = [0]*nrows
            for r in range(nrows):
                for c in range(ncols):
                    h, w, _ = rows[r][c].shape
                    col_widths[c] = max(col_widths[c], w)
                    row_heights[r] = max(row_heights[r], h)
            # Pad panels to column widths and row heights
            for r in range(nrows):
                for c in range(ncols):
                    img = rows[r][c]
                    h, w, _ = img.shape
                    target_h = row_heights[r]; target_w = col_widths[c]
                    if w < target_w:
                        extra = np.full((h, target_w - w, 3), 255, dtype=np.uint8)
                        img = np.concatenate([img, extra], axis=1)
                    if h < target_h:
                        extra = np.full((target_h - h, img.shape[1], 3), 255, dtype=np.uint8)
                        img = np.concatenate([img, extra], axis=0)
                    rows[r][c] = img

            # Column headers
            header_h = 5*2 + 6
            col_header = np.full((header_h, sum(col_widths) + (ncols-1)*8 + 150, 3), 255, dtype=np.uint8)
            x = 150
            for (label, _), w in zip(kinds, col_widths):
                draw_text(col_header, x + 8, 2, label, color=(0,0,0), scale=2)
                x += w + 8

            # Build each row image with left label area
            mosaic_rows = [col_header]
            for r in range(nrows):
                left = np.full((row_heights[r], 150, 3), 255, dtype=np.uint8)
                draw_text(left, 8, 8, row_labels[r], color=(0,0,0), scale=2)
                row_imgs = [left]
                for c in range(ncols):
                    if c>0:
                        row_imgs.append(np.full((row_heights[r], 8, 3), 255, dtype=np.uint8))
                    row_imgs.append(rows[r][c])
                mosaic_rows.append(np.concatenate(row_imgs, axis=1))

            mosaic = np.concatenate(mosaic_rows, axis=0)
            save_png(out_path, mosaic)
            return out_path

        # Get found programs from enumeration results - use actual sequences (before JSON filtering)
        found_abs_programs = res_once["ABS"].get("program_sequences", [])
        
        images_dir = output_dir / "images"
        images_dir.mkdir(exist_ok=True)
        render_mosaic_universal_all_examples(task, str(images_dir / "pattern_mosaic.png"), found_abs_programs)
        print("Wrote", images_dir / "pattern_mosaic.png")
    except Exception as e:
        print("[warn] failed to render universal mosaic:", e)

if __name__ == "__main__":
    main()
