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
    COLOR_H3 = _best_color_for_kind(task, "h3")
    COLOR_V3 = _best_color_for_kind(task, "v3")
    COLOR_Wn = _best_color_for_kind(task, "window_nxn")

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
    # Quick evaluation of other pattern kinds (v3, window_nxn) on train
    def eval_kind(kind: str):
        best_c = _best_color_for_kind(task, kind)
        res = []
        for x,y in train_pairs:
            yp = dsl.predict_with_pattern_kind(x.tolist(), kind, best_c)
            res.append(int(yp==y))
        return sum(res), len(res), best_c

    k_v3_ok, k_total, k_v3_c = eval_kind("v3")
    k_w_ok, _, k_w_c = eval_kind("window_nxn")
    print(f"\n=== Pattern kind eval (train acc) ===\n v3: {k_v3_ok}/{k_total} (color={k_v3_c})\n window_nxn: {k_w_ok}/{k_total} (color={k_w_c})")
    # Test-set predictions (no GT in ARC test; we just report)
    gtest = np.array(task["test"][0]["input"], dtype=int)
    pred_v3 = dsl.predict_with_pattern_kind(gtest.tolist(), "v3", COLOR_V3)
    pred_wn = dsl.predict_with_pattern_kind(gtest.tolist(), "window_nxn", COLOR_Wn)
    print(f"Test predictions: v3={pred_v3} window_nxn={pred_wn}")
    # Prepare capture for schema output
    schema_lines: list[str] = []
    def sprint(msg: str = ""):
        print(msg)
        schema_lines.append(msg)

    # Print a few window_nxn schemas for inspection (all train examples)
    try:
        sprint("\n=== Sample window_nxn schemas (train) ===")
        def _print_schema_aligned(schema):
            # Determine max cell width across schema for alignment
            cells = [str(x) for row in schema for x in row]
            w = max((len(c) for c in cells), default=1)
            for row in schema:
                line = "[" + ", ".join(f"{str(x):>{w}}" for x in row) + "]"
                sprint(" " + line)
        for idx, ex in enumerate(task["train"], start=1):
            g_ex = np.array(ex["input"], dtype=int)
            ovs_ex = dsl.detect_overlays(g_ex.tolist(), kind="window_nxn", color=COLOR_Wn)
            sprint(f"-- train[{idx}] overlays={len(ovs_ex)} --")
            for ov in ovs_ex[:min(4, len(ovs_ex))]:
                oid = int(ov["overlay_id"])
                cr, cc = int(ov["center_row"]), int(ov["center_col"])
                n = ov.get("window_size")
                sprint(f"overlay_id {oid:<2} center ({cr:>2}, {cc:>2}) n {int(n)}")
                schema = ov.get("schema", [])
                _print_schema_aligned(schema)
                sprint()
    except Exception as e:
        sprint(f"[warn] failed to print schemas: {e}")
    # Print a few window_nxn schemas for test examples as well
    try:
        sprint("\n=== Sample window_nxn schemas (test) ===")
        for idx, ex in enumerate(task["test"], start=1):
            g_ex = np.array(ex["input"], dtype=int)
            ovs_ex = dsl.detect_overlays(g_ex.tolist(), kind="window_nxn", color=COLOR_Wn)
            sprint(f"-- test[{idx}] overlays={len(ovs_ex)} --")
            for ov in ovs_ex[:min(4, len(ovs_ex))]:
                oid = int(ov["overlay_id"])
                cr, cc = int(ov["center_row"]), int(ov["center_col"])
                n = ov.get("window_size")
                sprint(f"overlay_id {oid:<2} center ({cr:>2}, {cc:>2}) n {int(n)}")
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
        sprint("\n=== Combined schema (window_nxn) ===")
        comb = dsl.combined_window_nxn_schema(task, COLOR_Wn, window_size=dsl.WINDOW_SIZE_DEFAULT)
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

    # Verify overlay counts for h3_yellow vs count of yellow pixels
    def count_color(g, color):
        return int((np.array(g)==int(color)).sum())
    for split in ("train","test"):
        exs = task[split]
        for i, ex in enumerate(exs, start=1):
            g = ex["input"]
            n_c = count_color(g, COLOR_H3)
            ovs_h3 = dsl.detect_overlays(g, kind="h3", color=COLOR_H3)
            print(f"{split}[{i}] color={COLOR_H3} count={n_c} overlays_h3={len(ovs_h3)}")
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
            centers_h3 = centers_of(dsl.detect_overlays(g, kind="h3", color=COLOR_H3))
            centers_v3 = centers_of(dsl.detect_overlays(g, kind="v3", color=COLOR_V3))
            centers_crossn = centers_of(dsl.detect_overlays(g, kind="window_nxn", color=COLOR_Wn))
            details["train"].append({
                "target_color": ex["output"][0][0],
                "centers_h3": centers_h3,
                "centers_v3": centers_v3,
                "centers_crossn": centers_crossn,
            })
        for ex in task["test"]:
            g = ex["input"]
            centers_h3 = centers_of(dsl.detect_overlays(g, kind="h3", color=COLOR_H3))
            centers_v3 = centers_of(dsl.detect_overlays(g, kind="v3", color=COLOR_V3))
            centers_crossn = centers_of(dsl.detect_overlays(g, kind="window_nxn", color=COLOR_Wn))
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

    def render_grid_with_overlays(g, pred_color, title, out_path, kind="h3", color: int = 1):
        g = np.asarray(g, dtype=int)
        overlays = dsl.detect_overlays(g.tolist(), kind=kind, color=color)
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

    # Train pics
    images_dir = HERE / "images"
    images_dir.mkdir(exist_ok=True)

    for i, ex in enumerate(task["train"], start=1):
        g = np.array(ex["input"], dtype=int)
        pred = dsl.predict_bright_overlay_uniform_cross(g.tolist(), COLOR_Wn)
        render_grid_with_overlays(
            g,
            pred,
            f"Train {i}: h3 (color={COLOR_H3}) (GT={int(ex['output'][0][0])})",
            str(images_dir / f"overlay_train_{i}.png"),
            kind="h3", color=COLOR_H3,
        )
        render_grid_with_overlays(
            g,
            pred,
            f"Train {i}: v3 (color={COLOR_V3}) (GT={int(ex['output'][0][0])})",
            str(images_dir / f"overlay_train_{i}_v.png"),
            kind="v3", color=COLOR_V3,
        )
        render_grid_with_overlays(
            g,
            pred,
            f"Train {i}: window_nxn (color={COLOR_Wn}) (GT={int(ex['output'][0][0])})",
            str(images_dir / f"overlay_train_{i}_x.png"),
            kind="window_nxn", color=COLOR_Wn,
        )

    # Test pic
    gtest = np.array(task["test"][0]["input"], dtype=int)
    predt = dsl.predict_bright_overlay_uniform_cross(gtest.tolist(), COLOR_Wn)
    render_grid_with_overlays(gtest, predt, f"Test: h3 (color={COLOR_H3})", str(images_dir / "overlay_test.png"), kind="h3", color=COLOR_H3)
    render_grid_with_overlays(gtest, predt, f"Test: v3 (color={COLOR_V3})", str(images_dir / "overlay_test_v.png"), kind="v3", color=COLOR_V3)
    render_grid_with_overlays(gtest, predt, f"Test: window_nxn (color={COLOR_Wn})", str(images_dir / "overlay_test_x.png"), kind="window_nxn", color=COLOR_Wn)

if __name__ == "__main__":
    main()
