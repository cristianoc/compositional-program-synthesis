# -----------------------------------------------------------------------------
# Reproduction script for “Overlay abstraction vs. core G” experiment
# This script:
#  • Loads task.json and enumerates program spaces (G vs. Abstraction) and checks ALL train
#  • Prints programs found & node counts/timings                            (README_clean.md §4)
#  • Renders annotated images (overlays + predicted color)                  (README_clean.md Figures)
# No behavior changes are made; this is the exact harness used for results.
# -----------------------------------------------------------------------------


import json, time, numpy as np, os
from typing import Optional, Any
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

def _best_color_for_kind(task, kind: str):
    train_pairs = [(np.array(ex["input"], dtype=int), int(ex["output"][0][0])) for ex in task["train"]]
    best_c, best_ok = None, -1
    for c in range(1,10):
        ok = 0
        for x,y in train_pairs:
            yp = dsl.predict_with_pattern_kind(x.tolist(), kind, c)
            ok += int(yp == y)
        if ok > best_ok:
            best_ok = ok; best_c = c
    return int(best_c) if best_c is not None else 1


def main():
    task = json.loads(Path(TASK_PATH).read_text())
    train_pairs = [(np.array(ex["input"], dtype=int), int(ex["output"][0][0])) for ex in task["train"]]
    # choose best colors per kind on train
    def _best_color_for_shape(shape):
        from importlib import reload
        import dsl as _dsl
        reload(_dsl)
        setattr(_dsl, 'WINDOW_SHAPE_DEFAULT', shape)
        return _best_color_for_kind(task, "window_nxm")

    # Snapshot the base default shape before any overrides
    BASE_DEFAULT_SHAPE = tuple(dsl.WINDOW_SHAPE_DEFAULT)
    COLOR_1x3 = _best_color_for_shape((1,3))
    COLOR_3x1 = _best_color_for_shape((3,1))
    COLOR_Wn = _best_color_for_shape(BASE_DEFAULT_SHAPE)

    # Enumerate programs for three window shapes: 1x3, 3x1, default (no per-shape printing here)
    # We will print a single combined, top-level structure once below.
    def run_enumeration_for_shape(shape_name: str, shape: tuple[int,int]):
        from importlib import reload
        import dsl as _dsl
        reload(_dsl)
        setattr(_dsl, 'WINDOW_SHAPE_DEFAULT', shape)
        res = _dsl.enumerate_programs_for_task(task, num_preops=200, seed=11)
        return shape_name, res

    programs_all = {}
    for nm, sh in (("1x3", (1,3)), ("3x1", (3,1)), ("window", BASE_DEFAULT_SHAPE)):
        name, res = run_enumeration_for_shape(nm, sh)  # type: ignore[arg-type]
        programs_all[name] = res
    # Print a single combined, top-level structure (like before) and write combined JSON
    programs_path = HERE / "programs_found.json"
    try:
        # Print node counts once
        # G nodes are the same across shapes; ABS nodes represent 3 instantiations × 9 colors
        any_shape = next(iter(programs_all.values()))
        g_nodes = any_shape['G']['nodes']
        abs_nodes_single = any_shape['ABS']['nodes']
        abs_nodes_total = abs_nodes_single * 3
        print("=== Node counts ===")
        print(f"G core nodes: {g_nodes}")
        print(f"Overlay+predicate nodes: {abs_nodes_total}")

        # Print programs once
        print("\n=== Programs found (G core) ===")
        g_progs = any_shape['G']['programs']
        if g_progs:
            for sname in g_progs:
                print("-", sname)
        else:
            print("(none)")

        print("\n=== Programs found (overlay abstraction + pattern check) ===")
        # Merge ABS programs from the three shapes
        abs_all = []
        for shape_name, res in programs_all.items():
            abs_all.extend(res['ABS']['programs'])
        if abs_all:
            for sname in abs_all:
                print("-", sname)
        else:
            print("(none)")

        # Persist combined JSON
        with open(programs_path, "w", encoding="utf-8") as f:
            json.dump(programs_all, f, indent=2, sort_keys=True)
        print("Wrote", programs_path)
    except Exception as e:
        print("[warn] failed to write programs_found.json:", e)

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

        # Abstraction (use best color for schemanxn found on train)
        t2 = time.perf_counter()
        valid_ABS = []
        tried2 = 0; tries_first_ABS=None
        for pre_name, pre_f in preops:
            tried2 += 1
            ok=True
            for x,y in train_pairs:  # <-- CHECK on ALL training examples (selection criterion)
                y_pred = dsl.predict_bright_overlay_uniform_cross(pre_f(x), COLOR_Wn)
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
    # Quick evaluation of shapes on train
    def eval_kind(kind: str):
        best_c = _best_color_for_kind(task, kind)
        res = []
        for x,y in train_pairs:
            yp = dsl.predict_with_pattern_kind(x.tolist(), kind, best_c)
            res.append(int(yp==y))
        return sum(res), len(res), best_c

    def eval_shape(shape):
        from importlib import reload
        import dsl as _dsl
        reload(_dsl)
        setattr(_dsl, 'WINDOW_SHAPE_DEFAULT', shape)
        res = []
        for x,y in train_pairs:
            yp = _dsl.predict_with_pattern_kind(x.tolist(), "window_nxm", COLOR_Wn)
            res.append(int(yp==y))
        return sum(res), len(res)
    k_13_ok, k_total = eval_shape((1,3))
    k_31_ok, _ = eval_shape((3,1))
    print(f"\n=== Pattern eval (train acc) ===\n 1x3: {k_13_ok}/{k_total}\n 3x1: {k_31_ok}/{k_total}")
    # Test-set predictions (no GT in ARC test; we just report)
    gtest = np.array(task["test"][0]["input"], dtype=int)
    from importlib import reload
    reload(dsl)
    setattr(dsl, 'WINDOW_SHAPE_DEFAULT', (1,3))
    pred_13 = dsl.predict_with_pattern_kind(gtest.tolist(), "window_nxm", COLOR_1x3)
    reload(dsl)
    setattr(dsl, 'WINDOW_SHAPE_DEFAULT', (3,1))
    pred_31 = dsl.predict_with_pattern_kind(gtest.tolist(), "window_nxm", COLOR_3x1)
    reload(dsl)
    pred_wn = dsl.predict_with_pattern_kind(gtest.tolist(), "window_nxm", COLOR_Wn)
    print(f"Test predictions: 1x3={pred_13} 3x1={pred_31} window_nxm={pred_wn}")
    # Prepare capture for schema output
    schema_lines: list[str] = []
    def sprint(msg: str = ""):
        print(msg)
        schema_lines.append(msg)

    # Print a few window_nxm schemas for inspection (all train examples)
    try:
        sprint("\n=== Sample window_nxm schemas (train) ===")
        def _print_schema_aligned(schema):
            # Determine max cell width across schema for alignment
            cells = [str(x) for row in schema for x in row]
            w = max((len(c) for c in cells), default=1)
            for row in schema:
                line = "[" + ", ".join(f"{str(x):>{w}}" for x in row) + "]"
                sprint(" " + line)
        for idx, ex in enumerate(task["train"], start=1):
            g_ex = np.array(ex["input"], dtype=int)
            ovs_ex = dsl.detect_overlays(g_ex.tolist(), kind="window_nxm", color=COLOR_Wn)
            sprint(f"-- train[{idx}] overlays={len(ovs_ex)} --")
            for ov in ovs_ex[:min(4, len(ovs_ex))]:
                oid = int(ov["overlay_id"])
                cr, cc = int(ov["center_row"]), int(ov["center_col"])
                shape = ov.get("window_shape")
                if not shape:
                    shape = [ov.get("window_h"), ov.get("window_w")]
                sprint(f"overlay_id {oid:<2} center ({cr:>2}, {cc:>2}) shape {tuple(int(x) for x in shape)}")
                schema = ov.get("schema", [])
                _print_schema_aligned(schema)
                sprint()
    except Exception as e:
        sprint(f"[warn] failed to print schemas: {e}")
    # Print a few window_nxm schemas for test examples as well
    try:
        sprint("\n=== Sample window_nxm schemas (test) ===")
        for idx, ex in enumerate(task["test"], start=1):
            g_ex = np.array(ex["input"], dtype=int)
            ovs_ex = dsl.detect_overlays(g_ex.tolist(), kind="window_nxm", color=COLOR_Wn)
            sprint(f"-- test[{idx}] overlays={len(ovs_ex)} --")
            for ov in ovs_ex[:min(4, len(ovs_ex))]:
                oid = int(ov["overlay_id"])
                cr, cc = int(ov["center_row"]), int(ov["center_col"])
                shape = ov.get("window_shape")
                if not shape:
                    shape = [ov.get("window_h"), ov.get("window_w")]
                sprint(f"overlay_id {oid:<2} center ({cr:>2}, {cc:>2}) shape {tuple(int(x) for x in shape)}")
                schema = ov.get("schema", [])
                # reuse the train printer
                cells = [str(x) for row in schema for x in row]
                w = max((len(c) for c in cells), default=1)
                for row in schema:
                    sprint(" " + "[" + ", ".join(f"{str(x):>{w}}" for x in row) + "]")
                sprint()
    except Exception as e:
        sprint(f"[warn] failed to print test schemas: {e}")
    # Print combined schema across full windows (aligned)
    try:
        sprint("\n=== Combined schema (window_nxm) ===")
        comb = dsl.combined_window_nxm_schema(task, COLOR_Wn, window_shape=dsl.WINDOW_SHAPE_DEFAULT)
        # align
        cells = [str(x) for row in comb for x in row]
        w = max((len(c) for c in cells), default=1)
        for row in comb:
            sprint(" " + "[" + ", ".join(f"{str(x):>{w}}" for x in row) + "]")
        # add a blank line after combined schema
        sprint()
    except Exception as e:
        sprint(f"[warn] failed to print combined schema: {e}")

    # Save captured schema output to file
    out_path = HERE / "schema_print.txt"
    try:
        out_path.write_text("\n".join(schema_lines) + "\n", encoding="utf-8")
        print("Saved schema print to", out_path)
    except Exception as e:
        print("[warn] failed to write schema_print.txt:", e)

    # Verify overlay counts for 1x3 (same as count of target color centers)
    def count_color(g, color):
        return int((np.array(g)==int(color)).sum())
    for split in ("train","test"):
        exs = task[split]
        for i, ex in enumerate(exs, start=1):
            g = ex["input"]
            n_c = count_color(g, COLOR_1x3)
            ovs_13 = dsl.detect_overlays(g, kind="window_nxm", color=COLOR_1x3, window_shape=(1,3))
            print(f"{split}[{i}] color={COLOR_1x3} count={n_c} overlays_1x3={len(ovs_13)}")
    # Emit a tracked artifact with stable ordering
    out_stats = HERE / "repro_stats.json"
    with open(out_stats, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, sort_keys=True)
    print(f"Wrote {out_stats}")

    # Emit simplified per-example stats (pattern centers only)
    def per_example_details(task):
        details = {"train": [], "test": []}
        def centers_of(ovs):
            return sorted([[ov["center_row"], ov["center_col"]] for ov in ovs])
        for ex in task["train"]:
            g = ex["input"]
            centers_h3 = centers_of(dsl.detect_overlays(g, kind="window_nxm", color=COLOR_1x3, window_shape=(1,3)))
            centers_v3 = centers_of(dsl.detect_overlays(g, kind="window_nxm", color=COLOR_3x1, window_shape=(3,1)))
            centers_crossn = centers_of(dsl.detect_overlays(g, kind="window_nxm", color=COLOR_Wn, window_shape=dsl.WINDOW_SHAPE_DEFAULT))
            details["train"].append({
                "target_color": ex["output"][0][0],
                "centers_h3": centers_h3,
                "centers_v3": centers_v3,
                "centers_crossn": centers_crossn,
            })
        for ex in task["test"]:
            g = ex["input"]
            centers_h3 = centers_of(dsl.detect_overlays(g, kind="window_nxm", color=COLOR_1x3, window_shape=(1,3)))
            centers_v3 = centers_of(dsl.detect_overlays(g, kind="window_nxm", color=COLOR_3x1, window_shape=(3,1)))
            centers_crossn = centers_of(dsl.detect_overlays(g, kind="window_nxm", color=COLOR_Wn, window_shape=dsl.WINDOW_SHAPE_DEFAULT))
            details["test"].append({
                "centers_h3": centers_h3,
                "centers_v3": centers_v3,
                "centers_crossn": centers_crossn,
            })
        return details

    py_details = per_example_details(task)
    with open(HERE / "pattern_stats.json", "w", encoding="utf-8") as f:
        json.dump(py_details, f, separators=(",", ":"), sort_keys=False)
        f.write("\n")
    print("Wrote", HERE / "pattern_stats.json")

    # Render pictures (train + test) without matplotlib for better determinism
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

    def draw_rect_outline(img: np.ndarray, y1: int, x1: int, y2: int, x2: int, color=(255,255,0), scale: int = 16):
        # draw a 1-pixel outline in scaled space
        yy1, xx1, yy2, xx2 = y1*scale, x1*scale, (y2+1)*scale-1, (x2+1)*scale-1
        img[yy1:yy1+1, xx1:xx2+1, :] = color
        img[yy2:yy2+1, xx1:xx2+1, :] = color
        img[yy1:yy2+1, xx1:xx1+1, :] = color
        img[yy1:yy2+1, xx2:xx2+1, :] = color

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

    def render_grid_with_overlays(g, pred_color, title, out_path, kind="window_nxm", color: int = 1):
        g = np.asarray(g, dtype=int)
        overlays = dsl.detect_overlays(g.tolist(), kind=kind, color=color, window_shape=window_shape)
        SCALE = 16
        base = upsample(grid_to_rgb(g), scale=SCALE)
        # draw overlays
        for ov in sorted(overlays, key=lambda ov: (ov["center_row"], ov["center_col"])):
            y1,x1,y2,x2 = ov["y1"]-1, ov["x1"]-1, ov["y2"]-1, ov["x2"]-1
            draw_rect_outline(base, y1, x1, y2, x2, color=YELLOW, scale=SCALE)
        # Compose a visible single cell for predicted color to the right
        Hs = base.shape[0]
        spacer_w = 8
        spacer = np.full((Hs + 16, spacer_w, 3), 255, dtype=np.uint8)
        # Equalize panel widths: input and output panels each get max width
        in_w = base.shape[1]
        out_w = max(in_w, SCALE)
        # Pad input panel to out_w if needed (usually already >= SCALE)
        if in_w < out_w:
            pad = np.full((Hs, out_w - in_w, 3), 255, dtype=np.uint8)
            base_panel = np.concatenate([base, pad], axis=1)
        else:
            base_panel = base
        # Build output panel with width out_w; draw single predicted cell at left
        out_panel = np.full((Hs, out_w, 3), 255, dtype=np.uint8)
        out_panel[0:SCALE, 0:SCALE, :] = PALETTE.get(int(pred_color), (0,0,0))
        # Add margins around both panels
        def add_border(img: np.ndarray, m: int = 8) -> np.ndarray:
            h, w, _ = img.shape
            bordered = np.full((h + 2*m, w + 2*m, 3), 255, dtype=np.uint8)
            bordered[m:m+h, m:m+w, :] = img
            return bordered
        base_panel_b = add_border(base_panel, 8)
        out_panel_b = add_border(out_panel, 8)
        # Ensure both bordered panels have identical width (pad right with white if needed)
        bw = base_panel_b.shape[1]
        ow = out_panel_b.shape[1]
        target_w = max(bw, ow)
        if bw < target_w:
            extra = np.full((base_panel_b.shape[0], target_w - bw, 3), 255, dtype=np.uint8)
            base_panel_b = np.concatenate([base_panel_b, extra], axis=1)
        if ow < target_w:
            extra = np.full((out_panel_b.shape[0], target_w - ow, 3), 255, dtype=np.uint8)
            out_panel_b = np.concatenate([out_panel_b, extra], axis=1)
        # Match heights (in case borders changed heights slightly)
        bh = base_panel_b.shape[0]
        oh = out_panel_b.shape[0]
        target_h = max(bh, oh)
        if bh < target_h:
            extra = np.full((target_h - bh, base_panel_b.shape[1], 3), 255, dtype=np.uint8)
            base_panel_b = np.concatenate([base_panel_b, extra], axis=0)
        if oh < target_h:
            extra = np.full((target_h - oh, out_panel_b.shape[1], 3), 255, dtype=np.uint8)
            out_panel_b = np.concatenate([out_panel_b, extra], axis=0)
        composed = np.concatenate([base_panel_b, spacer, out_panel_b], axis=1)
        # Add header with two lines: title; and "INPUT" left, "PREDICTED COLOR: N" right above columns
        header_lines = 2
        line_h = 5 * 2 + 4  # font height*scale + padding
        header_h = header_lines * line_h
        header = np.full((header_h, composed.shape[1], 3), 255, dtype=np.uint8)
        # First line: title (uppercase simplified)
        _draw_text(header, 4, 2, title, color=(0,0,0), scale=2)
        # Second line: PREDICTED COLOR right
        right_x = base_panel_b.shape[1] + spacer_w + 4
        _draw_text(header, right_x, 2 + line_h, f"PREDICTED COLOR {int(pred_color)}", color=(0,0,0), scale=2)
        composed = np.concatenate([header, composed], axis=0)
        save_png(out_path, composed)
        return out_path

    # Internal: build panel without header (for mosaics)
    def build_panel_body(g, pred_color, kind="window_nxm", color: int = 1, window_shape: Optional[tuple[int,int]] = None) -> np.ndarray:
        g = np.asarray(g, dtype=int)
        overlays = dsl.detect_overlays(g.tolist(), kind=kind, color=color, window_shape=window_shape)
        SCALE = 16
        base = upsample(grid_to_rgb(g), scale=SCALE)
        for ov in sorted(overlays, key=lambda ov: (ov["center_row"], ov["center_col"])):
            y1,x1,y2,x2 = ov["y1"]-1, ov["x1"]-1, ov["y2"]-1, ov["x2"]-1
            draw_rect_outline(base, y1, x1, y2, x2, color=YELLOW, scale=SCALE)
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

    # Build a single mosaic image with all train + test rows and columns for kinds
    def render_mosaic_all_examples(task, COLOR_1x3: int, COLOR_3x1: int, COLOR_Wn: int, out_path: str):
        import numpy as np
        base_shape = tuple(dsl.WINDOW_SHAPE_DEFAULT)
        kinds: list[tuple[str, tuple[int,int], int]] = [("1x3", (1,3), COLOR_1x3), ("3x1", (3,1), COLOR_3x1), ("WINDOW", base_shape, COLOR_Wn)]
        rows = []
        row_labels = []
        # Collect panels per row (train examples then test)
        for idx, ex in enumerate(task["train"], start=1):
            g = np.array(ex["input"], dtype=int)
            pan = []
            for _, shape, kc in kinds:
                # Set shape for prediction
                from importlib import reload
                import dsl as _dsl
                reload(_dsl)
                setattr(_dsl, "WINDOW_SHAPE_DEFAULT", shape)
                pred = _dsl.predict_with_pattern_kind(g.tolist(), "window_nxm", kc)
                pan.append(build_panel_body(g, pred, kind="window_nxm", color=kc, window_shape=shape))
            rows.append(pan); row_labels.append(f"TRAIN {idx}")
        for idx, ex in enumerate(task["test"], start=1):
            g = np.array(ex["input"], dtype=int)
            pan = []
            for _, shape, kc in kinds:
                # Set shape for prediction
                from importlib import reload
                import dsl as _dsl
                reload(_dsl)
                setattr(_dsl, "WINDOW_SHAPE_DEFAULT", shape)
                pred = _dsl.predict_with_pattern_kind(g.tolist(), "window_nxm", kc)
                pan.append(build_panel_body(g, pred, kind="window_nxm", color=kc, window_shape=shape))
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
        for (label, _, _), w in zip(kinds, col_widths):
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

    # Optionally render individual pictures (off by default; mosaic is always rendered)
    GENERATE_INDIVIDUAL = False
    # Train pics
    images_dir = HERE / "images"
    images_dir.mkdir(exist_ok=True)
    if GENERATE_INDIVIDUAL:
        for i, ex in enumerate(task["train"], start=1):
            g = np.array(ex["input"], dtype=int)
            pred = dsl.predict_bright_overlay_uniform_cross(g.tolist(), COLOR_Wn)
            render_grid_with_overlays(
                g,
                pred,
                f"Train {i}: 1x3 (GT={int(ex['output'][0][0])})",
                str(images_dir / f"overlay_train_{i}.png"),
                kind="window_nxm", color=COLOR_1x3, window_shape=(1,3),
            )
            render_grid_with_overlays(
                g,
                pred,
                f"Train {i}: 3x1 (GT={int(ex['output'][0][0])})",
                str(images_dir / f"overlay_train_{i}_v.png"),
                kind="window_nxm", color=COLOR_3x1, window_shape=(3,1),
            )
            render_grid_with_overlays(
                g,
                pred,
                f"Train {i}: window_nxm (color={COLOR_Wn}) (GT={int(ex['output'][0][0])})",
                str(images_dir / f"overlay_train_{i}_x.png"),
                kind="window_nxm", color=COLOR_Wn,
            )

        # Test pic
        gtest = np.array(task["test"][0]["input"], dtype=int)
        predt = dsl.predict_bright_overlay_uniform_cross(gtest.tolist(), COLOR_Wn)
        render_grid_with_overlays(gtest, predt, f"Test: 1x3", str(images_dir / "overlay_test.png"), kind="window_nxm", color=COLOR_1x3, window_shape=(1,3))
        render_grid_with_overlays(gtest, predt, f"Test: 3x1", str(images_dir / "overlay_test_v.png"), kind="window_nxm", color=COLOR_3x1, window_shape=(3,1))
        render_grid_with_overlays(gtest, predt, f"Test: window_nxm (color={COLOR_Wn})", str(images_dir / "overlay_test_x.png"), kind="window_nxm", color=COLOR_Wn)

    # Composite mosaic (all train + test in one image for all kinds)
    try:
        render_mosaic_all_examples(task, COLOR_1x3, COLOR_3x1, COLOR_Wn, str(images_dir / "overlay_mosaic.png"))
        print("Wrote", images_dir / "overlay_mosaic.png")
    except Exception as e:
        print("[warn] failed to render mosaic:", e)
    # Composite mosaic (all train + test in one image for all kinds)
    try:
        render_mosaic_all_examples(task, COLOR_1x3, COLOR_3x1, COLOR_Wn, str(images_dir / "overlay_mosaic.png"))
        print("Wrote", images_dir / "overlay_mosaic.png")
    except Exception as e:
        print("[warn] failed to render mosaic:", e)

if __name__ == "__main__":
    main()
