#!/usr/bin/env python3
import json
import subprocess
from pathlib import Path
import sys


def main() -> int:
    here = Path(__file__).resolve().parent

    # 1) Generate Python stats
    subprocess.run([sys.executable, str(here / "repro.py")], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # 2) Generate OCaml stats
    ocaml_dir = here / "ocaml"
    subprocess.run(["dune", "build"], cwd=ocaml_dir, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["dune", "exec", "overlay_arc"], cwd=ocaml_dir, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

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
        print("OK: Python and OCaml stats match (centers, colors, gt).")
        return 0

    print("Mismatch found (showing up to 5):")
    for split, idx, field, av, bv in mismatches[:5]:
        print(f" {split}[{idx}] {field}:\n  py={av}\n  oc={bv}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())


