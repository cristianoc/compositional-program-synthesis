#!/usr/bin/env python3
import json
import subprocess
from pathlib import Path
import sys
import time


def main() -> int:
    here = Path(__file__).resolve().parent

    # 1) Generate Python stats (time wall-clock)
    t0 = time.perf_counter()
    subprocess.run([sys.executable, str(here / "repro.py")], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t1 = time.perf_counter()
    py_secs = t1 - t0

    # 2) Generate OCaml stats (time wall-clock for exec)
    ocaml_dir = here / "ocaml"
    subprocess.run(["dune", "build"], cwd=ocaml_dir, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t2 = time.perf_counter()
    subprocess.run(["dune", "exec", "overlay_arc"], cwd=ocaml_dir, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t3 = time.perf_counter()
    ml_secs = t3 - t2

    # 3) Load and compare
    py = json.loads((here / "python_stats.json").read_text())
    oc = json.loads((here / "ocaml_stats.json").read_text())

    mismatches = []
    for split in ("train", "test"):
        for idx, (a, b) in enumerate(zip(py[split], oc[split])):
            if a["centers"] != b["centers"]:
                mismatches.append((split, idx, "centers", a["centers"], b["centers"]))
            if a["uniform_cross_colors"] != b["uniform_cross_colors"]:
                mismatches.append((split, idx, "uniform_cross_colors", a["uniform_cross_colors"], b["uniform_cross_colors"]))
            if split == "train" and a.get("gt") != b.get("gt"):
                mismatches.append((split, idx, "gt", a.get("gt"), b.get("gt")))

    if not mismatches:
        print(f"OK: Python and OCaml stats match (centers, colors, gt).  py={py_secs:.2f}s ocaml={ml_secs:.2f}s")
        return 0

    print(f"Mismatch found (showing up to 5).  py={py_secs:.2f}s ocaml={ml_secs:.2f}s")
    for split, idx, field, av, bv in mismatches[:5]:
        print(f" {split}[{idx}] {field}:\n  py={av}\n  oc={bv}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())


