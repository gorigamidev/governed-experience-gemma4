#!/usr/bin/env python3
"""Autopsy step 7, candidate C (aux files in the patch): rebuild each patch without the auxiliary files.

For every attempt whose patch ships an auxiliary file (the same AUX_RE as metrics.py), drop those
`diff --git` sections and keep the rest, so lab.eval.verify_patches can re-run the official hidden
tests on the stripped patch. One directory per run (same task can appear in two runs).

Usage: python experiments/autopsy/strip_aux.py --out-root build/autopsy-aux
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import metrics

ROOT = Path(__file__).resolve().parents[2]
SECTION_RE = re.compile(r"^diff --git ", re.M)


def split_sections(patch: str) -> list[str]:
    parts = [p for p in SECTION_RE.split(patch) if p.strip()]
    header = patch[: SECTION_RE.search(patch).start()] if SECTION_RE.search(patch) else ""
    return header, parts


def strip(patch: str) -> tuple[str, list[str]]:
    header, parts = split_sections(patch)
    keep, dropped = [], []
    for part in parts:
        path = part.split(" b/", 1)[1].split("\n", 1)[0].strip() if " b/" in part else part.splitlines()[0]
        (dropped if metrics.AUX_RE.search(path.split("/")[-1]) else keep).append(part)
    text = header + "".join(("diff --git " + p) for p in keep)
    return text, dropped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=ROOT / "build/autopsy-aux", type=Path)
    args = ap.parse_args()
    manifest = {}
    for task, run, bucket in metrics.scope_attempts():
        _, _, _, _, aux, _ = metrics.metrics(task, run)
        if not aux:
            continue
        patch_path = ROOT / "data/eval" / run / "patches" / f"{task}.patch"
        text, dropped = strip(patch_path.read_text())
        out_dir = args.out_root / run / "patches"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{task}.patch").write_text(text if text.endswith("\n") else text + "\n")
        manifest.setdefault(run, {})[task] = {"aux": aux, "dropped": len(dropped), "bucket": bucket}
        print(f"{run:26} {task:16} {bucket:16} dropped {len(dropped)}: {', '.join(aux)}")
    (args.out_root / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"{sum(len(v) for v in manifest.values())} patches -> {args.out_root}")


if __name__ == "__main__":
    main()
