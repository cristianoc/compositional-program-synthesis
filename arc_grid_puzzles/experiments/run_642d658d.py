#!/usr/bin/env python3
import sys
from pathlib import Path

# Get the directory where this script is located
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent

# Add project root to path
sys.path.insert(0, str(project_root))

# Get experiment name from script name
script_name = Path(__file__).name
experiment_name = script_name.replace('run_', '').replace('.py', '')

# Set up args and run with absolute paths
task_path = script_dir / "tasks" / f"{experiment_name}.json"
output_dir = script_dir / f"{experiment_name}_pattern_analysis"

# Add task-specific operations to the DSL for this experiment
experiment_dir = script_dir / f"{experiment_name}_pattern_analysis"
sys.path.insert(0, str(experiment_dir))

# Import task-specific operations
from task_specific_ops import G_TYPED_OPS_642D658D, enumerate_g_programs

# Patch the run_experiment module to combine G and ABS results
import run_experiment
import json
from pathlib import Path

def patched_main():
    """Custom main that combines G ops and ABS results."""
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description='Run pattern abstraction experiment on ARC task')
    parser.add_argument('task_path', help='Path to task.json file')
    parser.add_argument('--output-dir', help='Output directory for results (default: same as task directory)')
    args = parser.parse_args()
    
    task_path = Path(args.task_path)
    if not task_path.exists():
        print(f"Error: Task file not found: {task_path}")
        return
    
    task = json.loads(task_path.read_text())
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = task_path.parent
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run G ops enumeration
    print("=== Enumerating G programs ===")
    g_results = enumerate_g_programs(task)
    
    # Run ABS enumeration
    print("=== Enumerating ABS programs ===")
    SHAPES = [(1,3),(3,1),(2,3),(3,3),(5,5)]
    abs_results = run_experiment.enumerate_programs_for_task(task, num_preops=200, seed=11, universal_shapes=SHAPES)
    
    # Combine results
    combined_results = {
        "G": g_results,
        "ABS": abs_results["ABS"]
    }
    
    # Print combined results
    print("=== Node counts ===")
    print(f"G core nodes: {g_results['nodes']}")
    print(f"Overlay+predicate nodes: {abs_results['ABS']['nodes']}")
    
    print("\n=== Programs found (G core) ===")
    if g_results['programs']:
        for sname in g_results['programs']:
            print("-", sname)
    else:
        print("(none)")
    
    print("\n=== Programs found (overlay abstraction + pattern check) ===")
    if abs_results['ABS']['programs']:
        for sname in abs_results['ABS']['programs']:
            print("-", sname)
    else:
        print("(none)")
    
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
    
    # Print intersected universal schemas per shape (train+test)
    try:
        print("\n=== Intersected universal schemas (train+test) ===")
        from dsl_types.grid_to_matches import build_intersected_universal_schemas_for_task
        for ushape in SHAPES:
            uni_schemas = build_intersected_universal_schemas_for_task(task, window_shape=ushape, center_value=4, splits=("train","test"))
            if uni_schemas:
                print(f"shape {ushape}: {len(uni_schemas)} positions")
                for pos, schema in uni_schemas.items():
                    print(f"pos {pos}")
                    print("", schema)
    except Exception as e:
        print("[warn] failed to print universal schemas:", e)
    
    # Generate images
    try:
        print("\n=== Generating images ===")
        from dsl_types.grid_to_matches import build_intersected_universal_schemas_for_task
        from dsl_types.matches_to_color import OpUniformColorFromMatches
        from dsl_types.states import Grid, Pipeline
        import numpy as np
        
        # Create images directory
        images_dir = output_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        # Generate overlay mosaic
        mosaic_data = []
        for ushape in SHAPES:
            uni_schemas = build_intersected_universal_schemas_for_task(task, window_shape=ushape, center_value=4, splits=("train","test"))
            if uni_schemas:
                # Use first schema for this shape
                pos, schema = next(iter(uni_schemas.items()))
                matcher = run_experiment.OpMatchAnyUniversalSchemas([schema], label=f"match_universal_pos(shape={tuple(ushape)},pos={pos})")
                aggregator = OpUniformColorFromMatches()
                pipeline = Pipeline([matcher, aggregator])
                
                # Test on first training example
                train_input = np.array(task["train"][0]["input"], dtype=int)
                try:
                    result = pipeline.apply(Grid(train_input))
                    mosaic_data.append((ushape, result.color))
                except:
                    mosaic_data.append((ushape, 0))
        
        # Create mosaic image (simplified)
        mosaic_img = np.zeros((len(mosaic_data), 1), dtype=int)
        for i, (shape, color) in enumerate(mosaic_data):
            mosaic_img[i, 0] = color
        
        # Save mosaic
        mosaic_path = images_dir / "overlay_mosaic.png"
        import matplotlib.pyplot as plt
        plt.figure(figsize=(2, len(mosaic_data)))
        plt.imshow(mosaic_img, cmap='tab10', vmin=0, vmax=9)
        plt.title("Overlay Mosaic")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(mosaic_path, dpi=100, bbox_inches='tight')
        plt.close()
        print("Wrote", mosaic_path)
        
        # Generate individual train/test images
        for i, example in enumerate(task["train"]):
            input_img = np.array(example["input"], dtype=int)
            output_color = int(example["output"][0][0])
            
            # Save input image
            input_path = images_dir / f"train_{i}_in.png"
            plt.figure(figsize=(4, 4))
            plt.imshow(input_img, cmap='tab10', vmin=0, vmax=9)
            plt.title(f"Train {i} Input")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(input_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            # Save output image
            output_img = np.full_like(input_img, output_color)
            output_path = images_dir / f"train_{i}_out.png"
            plt.figure(figsize=(4, 4))
            plt.imshow(output_img, cmap='tab10', vmin=0, vmax=9)
            plt.title(f"Train {i} Output")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            plt.close()
        
        # Generate test input images
        for i, example in enumerate(task["test"]):
            input_img = np.array(example["input"], dtype=int)
            input_path = images_dir / f"test_{i}_in.png"
            plt.figure(figsize=(4, 4))
            plt.imshow(input_img, cmap='tab10', vmin=0, vmax=9)
            plt.title(f"Test {i} Input")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(input_path, dpi=100, bbox_inches='tight')
            plt.close()
        
        print("Generated all images")
        
    except Exception as e:
        print("[warn] failed to generate images:", e)

# Replace the run_experiment main with our custom one
run_experiment.main = patched_main

sys.argv = ["run_experiment.py", str(task_path), "--output-dir", str(output_dir)]
from run_experiment import main
main()