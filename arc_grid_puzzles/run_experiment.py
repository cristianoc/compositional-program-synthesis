# -----------------------------------------------------------------------------
# Generic Experiment Runner for Pattern Abstraction
# This script:
#  • Loads any task.json and enumerates program spaces (G vs. Abstraction)
#  • Prints programs found & node counts/timings
#  • Renders annotated images (universal matches + predicted color)
# Usage: python run_experiment.py <task_path>
# Example: python run_experiment.py experiments/642d658d/task.json
# -----------------------------------------------------------------------------

import json, time, numpy as np, os, argparse
from typing import Optional, Any, Union
from importlib import reload
import sys
from pathlib import Path

# Import from the reorganized structure
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from driver import enumerate_programs_for_task
from dsl_types.states import Grid, Pipeline
from dsl_types.grid_to_matches import OpMatchAnyUniversalSchemas
from dsl_types.grid_to_center_to_color import G_TYPED_OPS


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

    # Enumerate once: both G and ABS (ABS includes all universal shapes)
    SHAPES = [(1,3),(3,1),(2,3),(3,3),(5,5)]
    res_once = enumerate_programs_for_task(task, num_preops=200, seed=11, universal_shapes=SHAPES)
    # Print and persist simple combined JSON
    programs_path = output_dir / "programs_found.json"
    try:
        # G nodes from generic G_TYPED_OPS
        g_nodes = len(G_TYPED_OPS)
        abs_nodes = res_once['ABS']['nodes']
        print("=== Node counts ===")
        print(f"G core nodes: {g_nodes}")
        print(f"Overlay+predicate nodes: {abs_nodes}")

        print("\n=== Programs found (G core) ===")
        g_progs = res_once['G']['programs']
        if g_progs:
            for sname in g_progs:
                print("-", sname)
        else:
            print("(none)")

        print("\n=== Programs found (overlay abstraction + pattern check) ===")
        abs_progs = res_once['ABS']['programs']
        if abs_progs:
            for sname in abs_progs:
                print("-", sname)
        else:
            print("(none)")

        # Save JSON without program sequences (which aren't JSON serializable)
        json_output = {k: v for k, v in res_once.items()}
        if "ABS" in json_output and "program_sequences" in json_output["ABS"]:
            json_output["ABS"] = {k: v for k, v in json_output["ABS"].items() if k != "program_sequences"}
        
        with open(programs_path, "w", encoding="utf-8") as f:
            json.dump(json_output, f, indent=2, sort_keys=True)
        print("Wrote", programs_path)
    except Exception as e:
        print("[warn] failed to write programs_found.json:", e)

    # Use single-pass stats from enumeration result
    stats = {
        "G": {"nodes": len(G_TYPED_OPS), "programs_found": len(res_once['G']['programs']), "time_sec": res_once['G'].get('time_sec')},
        "ABS": {"nodes": res_once['ABS']['nodes'], "programs_found": len(res_once['ABS']['programs']), "time_sec": res_once['ABS'].get('time_sec')},
    }
    print("\n=== STATS (single-pass) ===")
    print(stats)
    # Prepare capture for schema output
    def sprint(msg: str = ""):
        print(msg)

    # Print intersected universal schemas per shape (train+test)
    try:
        sprint("\n=== Intersected universal schemas (train+test) ===")
        from dsl_types.grid_to_matches import build_intersected_universal_schemas_for_task
        for ushape in SHAPES:
            uni = build_intersected_universal_schemas_for_task(task, window_shape=tuple(ushape), center_value=4, splits=("train","test"))
            if not uni:
                sprint(f"shape {ushape}: none")
                continue
            sprint(f"shape {ushape}: {len(uni)} positions")
            for pos, sc in sorted(uni.items()):
                sprint(f"pos {pos}")
                cells = [str(x) for row in sc for x in row]
                w = max((len(c) for c in cells), default=1)
                for row in sc:
                    sprint(" " + "[" + ", ".join(f"{str(x):>{w}}" for x in row) + "]")
                sprint()
    except Exception as e:
        sprint(f"[warn] failed to print universal schemas: {e}")

    # Mosaic visualization using actual found programs

     # Render pictures (universal matches) without matplotlib for better determinism
    os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
    PALETTE = {
        0:(0,0,0), 1:(0,0,255), 2:(255,0,0), 3:(0,255,0), 4:(255,255,0),
        5:(128,128,128), 6:(255,192,203), 7:(255,165,0), 8:(0,128,128), 9:(139,69,19)
    }
    YELLOW = (255,255,0)

    def grid_to_rgb(g, palette=PALETTE):
        g = np.asarray(g, dtype=int)
        H, W = g.shape
        img = np.zeros((H, W, 3), dtype=np.uint8)
        for k, (r, gg, b) in palette.items():
            img[g == k] = (r, gg, b)
        return img

    def upsample(img: np.ndarray, scale: int) -> np.ndarray:
        return np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)

    def save_png(path: str, rgb: np.ndarray):
        import struct, binascii
        H, W, C = rgb.shape
        assert C == 3
        # Build raw bytes with no filter per row
        raw = b''.join(b"\x00" + rgb[i].tobytes() for i in range(H))

        # Deterministic zlib stream with stored (uncompressed) deflate blocks
        def adler32(data: bytes) -> int:
            MOD = 65521
            a = 1
            b = 0
            for byte in data:
                a = (a + byte) % MOD
                b = (b + a) % MOD
            return (b << 16) | a

        def zlib_stored(data: bytes) -> bytes:
            # zlib header: 0x78 0x01 (CMF=0x78, FLG=0x01) satisfies 31-check and FLEVEL=0
            header = b"\x78\x01"
            out = [header]
            i = 0
            n = len(data)
            while i < n:
                chunk = data[i : i + 65535]
                i += len(chunk)
                bfinal = 1 if i >= n else 0
                # Stored block header: 3 bits (BFINAL, BTYPE=00) → at byte boundary => byte is 0x01 for final else 0x00
                out.append(bytes([bfinal]))
                L = len(chunk)
                out.append(L.to_bytes(2, 'little'))
                out.append((0xFFFF - L).to_bytes(2, 'little'))
                out.append(chunk)
            out.append(adler32(data).to_bytes(4, 'big'))
            return b''.join(out)

        # PNG chunks
        def chunk(typ: bytes, data: bytes) -> bytes:
            return (
                struct.pack(">I", len(data))
                + typ
                + data
                + struct.pack(">I", binascii.crc32(typ + data) & 0xFFFFFFFF)
            )

        sig = b"\x89PNG\r\n\x1a\n"
        ihdr = struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0)
        idat = zlib_stored(raw)
        png = sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")
        with open(path, "wb") as f:
            f.write(png)

    def draw_rect_outline(img: np.ndarray, y1: int, x1: int, y2: int, x2: int, color=(255,255,0), scale: int = 16):
        # draw a 1-pixel outline in scaled space
        yy1, xx1, yy2, xx2 = y1*scale, x1*scale, (y2+1)*scale-1, (x2+1)*scale-1
        img[yy1:yy1+1, xx1:xx2+1, :] = color
        img[yy2:yy2+1, xx1:xx2+1, :] = color
        img[yy1:yy2+1, xx1:xx1+1, :] = color
        img[yy1:yy2+1, xx2:xx2+1, :] = color

    def _font_3x5():
        # 3x5 pixel font for A-Z, 0-9 and space
        F = {
            ' ': ["000","000","000","000","000"],
            'A': ["010","101","111","101","101"],
            'C': ["011","100","100","100","011"],
            'D': ["110","101","101","101","110"],
            'E': ["111","100","110","100","111"],
            'G': ["011","100","101","101","011"],
            'H': ["101","101","111","101","101"],
            'I': ["111","010","010","010","111"],
            'L': ["100","100","100","100","111"],
            'N': ["101","111","111","111","101"],
            'O': ["010","101","101","101","010"],
            'P': ["110","101","110","100","100"],
            'R': ["110","101","110","101","101"],
            'S': ["011","100","010","001","110"],
            'T': ["111","010","010","010","010"],
            'U': ["101","101","101","101","111"],
            'V': ["101","101","101","101","010"],
            'Y': ["101","101","010","010","010"],
            'G': ["011","100","101","101","011"],
            'M': ["101","111","101","101","101"],
            'K': ["101","101","110","101","101"],
            'X': ["101","101","010","101","101"],
            '0': ["111","101","101","101","111"],
            '1': ["010","110","010","010","111"],
            '2': ["111","001","111","100","111"],
            '3': ["111","001","111","001","111"],
            '4': ["101","101","111","001","001"],
            '5': ["111","100","111","001","111"],
            '6': ["111","100","111","101","111"],
            '7': ["111","001","010","010","010"],
            '8': ["111","101","111","101","111"],
            '9': ["111","101","111","001","111"],
        }
        return F

    def _draw_text(img: np.ndarray, x: int, y: int, text: str, color=(0,0,0), scale: int = 2):
        F = _font_3x5()
        cx = x
        H, W, _ = img.shape
        for ch in text.upper():
            pat = F.get(ch, F[' '])
            for r, row in enumerate(pat):
                for c, bit in enumerate(row):
                    if bit == '1':
                        yy = y + r*scale
                        xx = cx + c*scale
                        if 0 <= yy < H and 0 <= xx < W:
                            img[yy:yy+scale, xx:xx+scale, :] = color
            cx += (3*scale + scale)  # glyph width + 1 space

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
            def add_border(img: np.ndarray, m: int = 8) -> np.ndarray:
                h, w, _ = img.shape
                bordered = np.full((h + 2*m, w + 2*m, 3), 255, dtype=np.uint8)
                bordered[m:m+h, m:m+w, :] = img
                return bordered
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
                _draw_text(col_header, x + 8, 2, label, color=(0,0,0), scale=2)
                x += w + 8

            # Build each row image with left label area
            mosaic_rows = [col_header]
            for r in range(nrows):
                left = np.full((row_heights[r], 150, 3), 255, dtype=np.uint8)
                _draw_text(left, 8, 8, row_labels[r], color=(0,0,0), scale=2)
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
        render_mosaic_universal_all_examples(task, str(images_dir / "overlay_mosaic.png"), found_abs_programs)
        print("Wrote", images_dir / "overlay_mosaic.png")
    except Exception as e:
        print("[warn] failed to render universal mosaic:", e)

if __name__ == "__main__":
    main()
