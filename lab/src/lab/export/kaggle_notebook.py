"""Wrap the validated build/submission.zip into a CPU-only Kaggle notebook that writes it back byte for byte.

Code competitions score the `submission.zip` a notebook leaves in /kaggle/working; Kaggle runs the agent
itself on 4x L4. Embedding the archive (base64, ~20 KB) keeps the kernel free of extra datasets and makes
the submitted bytes exactly the ones the official validator accepted locally.

Usage: PYTHONPATH=lab/src python3 -m lab.export.kaggle_notebook [--slug gemma4-agent-submission]
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
COMPETITION = "gemma-4-developer-agent"
OWNER = "gorigamidev"

CELL = '''import base64
import hashlib
import zipfile
from pathlib import Path

# Built from commit {commit} of github.com/gorigamidev/gemma4challenge-kaggle (lab/export/package.py,
# checked with the official adk_submission validator before embedding).
EXPECTED_SHA256 = "{sha256}"
ARCHIVE_B64 = "{b64}"

data = base64.b64decode(ARCHIVE_B64)
digest = hashlib.sha256(data).hexdigest()
assert digest == EXPECTED_SHA256, f"archive corrupted: {{digest}}"

out = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
target = out / "submission.zip"
target.write_bytes(data)
with zipfile.ZipFile(target) as zf:
    names = zf.namelist()
    assert "agent.yaml" in names, "agent.yaml must be at the archive root"
print(f"wrote {{target}} ({{len(data):,}} bytes, sha256 {{digest[:12]}}), {{len(names)}} files:")
print("\\n".join(names))
'''


def with_cell_ids(cells: list[dict]) -> list[dict]:
    """nbformat 4.5+ requires a unique id per cell; deterministic ids keep the same source -> same notebook."""
    return [{"id": f"cell-{i}", **c} for i, c in enumerate(cells)]


def git_commit() -> str:
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "submission"], capture_output=True,
                           text=True).stdout.strip()
    return head.stdout.strip() + ("-dirty" if dirty else "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=ROOT / "build/submission.zip", type=Path)
    ap.add_argument("--out", default=ROOT / "build/kaggle_kernel", type=Path)
    ap.add_argument("--slug", default="gemma4-agent-submission")
    args = ap.parse_args()

    data = args.zip.read_bytes()
    with zipfile.ZipFile(args.zip) as zf:
        assert "agent.yaml" in zf.namelist()
    sha256 = hashlib.sha256(data).hexdigest()
    commit = git_commit()
    cell = CELL.format(commit=commit, sha256=sha256, b64=base64.b64encode(data).decode())

    args.out.mkdir(parents=True, exist_ok=True)
    notebook = {
        "cells": with_cell_ids([{"cell_type": "markdown", "metadata": {}, "source": [
                      f"# {args.slug}\n\nWrites `submission.zip` (built from commit `{commit}`, "
                      f"sha256 `{sha256[:12]}`). Kaggle runs the agent on the hidden tasks."]},
                  {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                   "source": cell.splitlines(keepends=True)}]),
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    (args.out / "submission.ipynb").write_text(json.dumps(notebook, indent=1))
    meta = {"id": f"{OWNER}/{args.slug}", "title": args.slug.replace("-", " "), "code_file": "submission.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": False,
            "enable_tpu": False, "enable_internet": False, "dataset_sources": [], "kernel_sources": [],
            "competition_sources": [COMPETITION], "model_sources": []}
    (args.out / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    print(f"kernel {meta['id']} -> {args.out} | commit {commit} | sha256 {sha256[:12]} | {len(data):,} bytes")


if __name__ == "__main__":
    main()
