#!/usr/bin/env python3
import sys
from pathlib import Path

# Get the directory where this script is located
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent

# Add project root to path
sys.path.insert(0, str(project_root))

"""Runner for the 642d658d task experiment.

Uses the global ARC-AGI dataset task explicitly to avoid ambiguity.
"""

# Get experiment name from script name
script_name = Path(__file__).name
experiment_name = script_name.replace('run_', '').replace('.py', '')

# Hardcoded global dataset path (first encountered in global discovery order)
task_path = project_root / "tasks" / "arc_agi_1" / "evaluation" / f"{experiment_name}.json"
output_dir = script_dir / f"{experiment_name}_pattern_analysis"

"""Runner for the 642d658d task experiment.

Keeps task-specific ops and assets co-located under
experiments/642d658d_pattern_analysis/ for reproducibility.
"""

# Add task-specific operations to the path and import them
experiment_dir = script_dir / f"{experiment_name}_pattern_analysis"
sys.path.insert(0, str(experiment_dir))
from task_specific_ops import enumerate_g_programs  # noqa: E402

# Import program search directly (no need for run_experiment)
from program_search import enumerate_programs_for_task
import json

def patched_main():
    """Custom main that combines G ops and ABS results."""
    # Use the paths already set up at the top of the script
    # task_path and output_dir are already defined globally
    
    if not task_path.exists():
        print(f"Error: Task file not found: {task_path}")
        return
    
    # Load task JSON
    task = json.loads(task_path.read_text())
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run G ops enumeration
    print("=== Enumerating G programs ===")
    g_results = enumerate_g_programs(task)
    
    # Run ABS enumeration
    print("=== Enumerating ABS programs ===")
    SHAPES = [(1,3),(3,1),(2,3),(3,3),(5,5)]
    abs_results = enumerate_programs_for_task(task, num_preops=200, seed=11, universal_shapes=SHAPES)
    
    # Combine results
    combined_results = {
        "G": g_results,
        "ABS": abs_results["ABS"]
    }
    
    # Print combined results
    print("=== Node counts ===")
    print(f"G core nodes: {g_results['nodes']}")
    print(f"Pattern matching nodes: {abs_results['ABS']['nodes']}")
    
    print("\n=== Programs found (G core) ===")
    if g_results['programs']:
        for sname in g_results['programs']:
            print("-", sname)
    else:
        print("(none)")
    
    print("\n=== Programs found (pattern matching) ===")
    if abs_results['ABS']['programs']:
        for sname in abs_results['ABS']['programs']:
            print("-", sname)
    else:
        print("(none)")
    
    # Print intersected universal schemas per shape (train+test) in nice 2D format
    try:
        print("\n=== Intersected universal schemas (train+test) ===")
        from dsl_types.grid_to_matches import build_intersected_universal_schemas_for_task, format_schema_as_text
        for ushape in SHAPES:
            uni_schemas = build_intersected_universal_schemas_for_task(task, window_shape=ushape, center_value=4, splits=("train","test"))
            if uni_schemas:
                print(f"shape {ushape}: {len(uni_schemas)} positions")
                for pos, schema in uni_schemas.items():
                    print(f"pos {pos}")
                    print(format_schema_as_text(schema))
    except Exception as e:
        print("[warn] failed to print universal schemas:", e)
    
    # Save combined results
    programs_path = output_dir / "programs_found.json"
    try:
        json_output = {}
        for k, v in combined_results.items():
            json_output[k] = {}
            for subk, subv in v.items():
                if subk not in ["program_sequences", "time_sec"]:
                    json_output[k][subk] = subv
        
        with open(programs_path, "w", encoding="utf-8") as f:
            json.dump(json_output, f, indent=2, sort_keys=True)
        print("Wrote", programs_path)
    except Exception as e:
        print("[warn] failed to write programs_found.json:", e)
    
    # Print combined stats
    print("\n=== STATS (single-pass) ===")
    print({
        "G": {"nodes": g_results['nodes'], "programs_found": len(g_results['programs']), "time_sec": g_results.get('time_sec')},
        "ABS": {"nodes": abs_results['ABS']['nodes'], "programs_found": len(abs_results['ABS']['programs']), "time_sec": abs_results['ABS'].get('time_sec')},
    })
    
    # Generate images
    try:
        print("\n=== Generating images ===")
        from dsl_types.grid_to_matches import (
            build_intersected_universal_schemas_for_task,
            select_best_pattern_position,
            OpMatchAnyUniversalSchemas,
        )
        import numpy as np
        
        # Create images directory
        images_dir = output_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        # Generate proper visual mosaic using the original function
        found_abs_programs = abs_results["ABS"].get("program_sequences", [])
        mosaic_path = images_dir / "pattern_mosaic.png"
        
        # Import pattern matching for original mosaic approach
        from abstractions.pattern_matching.pattern_matching import detect_pattern_matches
        
        # Import shared image utilities
        from image_utils import (
            PALETTE, YELLOW, grid_to_rgb, upsample, save_png, 
            draw_rect_outline, draw_text, render_grid_image, render_single_cell_image, add_border
        )
        
        def build_panel_body(g, pred_color, shape: tuple[int,int], schema) -> np.ndarray:
            # Use original detect_overlays approach (matching original repro.py)
            g = np.asarray(g, dtype=int)
            SCALE = 8
            base = upsample(grid_to_rgb(g), scale=SCALE)
            
            # For mosaic rendering, relax the "center==4" constraint by matching universal schemas anywhere
            overlays = []
            if schema is not None:
                try:
                    from dsl_types.states import Grid as GridState
                    matcher = OpMatchAnyUniversalSchemas([schema])
                    ms = matcher.apply(GridState(g))
                    for m in ms.matches:
                        overlays.append({
                            "y1": m["y1"], "x1": m["x1"], "y2": m["y2"], "x2": m["x2"],
                        })
                except Exception:
                    overlays = []
            if not overlays:
                # Fallback to original centered detection
                overlays = detect_pattern_matches(g.tolist(), color=4, window_shape=shape)
            
            # Show overlays (matching original approach)
            for ov in sorted(overlays, key=lambda ov: (ov["y1"], ov["x1"])):
                y1 = ov["y1"] - 1
                x1 = ov["x1"] - 1
                y2 = ov["y2"] - 1
                x2 = ov["x2"] - 1
                draw_rect_outline(base, y1, x1, y2, x2, color=YELLOW, scale=SCALE)
            # Compose predicted color panel at right, same as original
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
            base_panel_b = add_border(base_panel, 8)
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
            # Show overlays for all enumerated shapes
            shapes = [(1,3),(3,1),(2,3),(3,3),(5,5)]

            # Pre-compute best universal schema per shape (train+test) for mosaic overlays
            shape_to_schema = {}
            for shp in shapes:
                try:
                    uni = build_intersected_universal_schemas_for_task(task, window_shape=shp, center_value=4, splits=("train","test"))
                    if uni:
                        best_pos, best_schema = select_best_pattern_position(uni)
                        shape_to_schema[shp] = best_schema
                except Exception:
                    continue

            # Choose aggregator: use original predict_with_pattern_kind approach
            def predict_for_grid(g, shape):
                # Use original overlay detection approach for prediction
                overlays = detect_pattern_matches(g.tolist(), color=4, window_shape=shape)
                if not overlays:
                    return 0
                
                # Simple prediction: take the mode color from all overlay centers
                colors = []
                for ov in overlays:
                    # Use the detection anchor position (where pattern was found)
                    y1 = ov["y1"] - 1
                    x1 = ov["x1"] - 1
                    # The detection anchor is at the top-left of the bounding box
                    anchor_y, anchor_x = y1, x1
                    if 0 <= anchor_y < len(g) and 0 <= anchor_x < len(g[0]):
                        colors.append(g[anchor_y][anchor_x])
                
                if not colors:
                    return 0
                
                # Return the most common color (mode)
                from collections import Counter
                color_counts = Counter(colors)
                return color_counts.most_common(1)[0][0]

            # Build rows (matching original structure)
            rows = []
            row_labels = []
            def _label_for_shape(shp: tuple[int,int]) -> str:
                r, c = shp
                return f"{r}x{c}"
            kinds = [(_label_for_shape(s), s) for s in shapes]
            for idx, ex in enumerate(task["train"], start=1):
                g = np.array(ex["input"], dtype=int)
                pan = []
                for _, shape in kinds:
                    pred = predict_for_grid(g, shape)
                    pan.append(build_panel_body(g, pred, shape, shape_to_schema.get(shape)))
                rows.append(pan); row_labels.append(f"TRAIN {idx}")
            for idx, ex in enumerate(task["test"], start=1):
                g = np.array(ex["input"], dtype=int)
                pan = []
                for _, shape in kinds:
                    pred = predict_for_grid(g, shape)
                    pan.append(build_panel_body(g, pred, shape, shape_to_schema.get(shape)))
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
        
        # Generate the proper visual mosaic
        render_mosaic_universal_all_examples(task, str(mosaic_path), found_abs_programs)
        print("Wrote", mosaic_path)
        
        # Generate individual train/test images using same custom rendering as mosaic
        for i, example in enumerate(task["train"]):
            input_img = np.array(example["input"], dtype=int)
            output_color = int(example["output"][0][0])

            # Save input image using shared utilities
            input_path = images_dir / f"train_{i}_in.png"
            input_scaled = render_grid_image(input_img, scale=16)
            save_png(str(input_path), input_scaled)

            # Save output image using shared utilities (single cell, scaled up to be visible)
            output_path = images_dir / f"train_{i}_out.png"
            output_scaled = render_single_cell_image(output_color, scale=64)
            save_png(str(output_path), output_scaled)

        # Generate test input images using shared utilities
        for i, example in enumerate(task["test"]):
            input_img = np.array(example["input"], dtype=int)
            input_path = images_dir / f"test_{i}_in.png"
            input_scaled = render_grid_image(input_img, scale=16)
            save_png(str(input_path), input_scaled)
        
        print("Generated all images")
        
    except Exception as e:
        print("[warn] failed to generate images:", e)

if __name__ == "__main__":
    patched_main()
