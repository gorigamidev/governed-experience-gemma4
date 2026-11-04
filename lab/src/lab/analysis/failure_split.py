"""First, automatic pass of the failure autopsy (experiments/autopsy/METHOD.md) -> data/analysis/failure_split.md.

Splits every task-run of the given runs into one coarse bucket from its result, its patch and the tail of its
verification output: resolved, timeout (with/without patch), context overflow, empty patch, patch misses the gold
file, right file but an import/collection error in the tests, right file but the tests fail. It is a triage rule,
not a diagnosis: the manual pass reads the traces and assigns the codebook codes.

Resolved comes from `verified_task_results.jsonl` (local Docker) when the run has it, else from the harness.
Reserve tasks (doc 03 §14.24) are excluded by default: design changes must not be drawn from the evaluation set.

Usage: PYTHONPATH=lab/src lab/.venv/bin/python -m lab.analysis.failure_split [--runs dir1,dir2] [--include-reserve]
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
RESERVE = frozenset(
    "fastapi_14258 fastapi_14372 fastapi_15280 fastapi_15588 rich_2725 rich_3043 rich_3063 rich_3067 rich_3105 "
    "rich_3180 rich_3278 rich_3506 rich_3676 rich_3934 rich_3944 rich_4077".split())
# New-scorer runs (harness of 30 Sep) on the 31B. Older runs (v1-v5) were scored by another harness.
DEFAULT_RUNS = ("data/eval/kaggle-v7-top-a", "data/eval/kaggle-v7-top-b", "data/eval/kaggle-thinking-top",
                "data/eval/kaggle-thinking-top-think", "data/eval/lora-31b-v1/effect/top-mix-1k",
                "data/eval/diag-analyzer-flat")
ORDER = ("resolved", "timeout, with patch", "timeout, no patch", "context overflow", "finished, empty patch",
         "patch misses the gold file", "right file, import/collection error in tests", "right file, tests fail")


def gold_files(patch: str) -> set[str]:
    return set(re.findall(r"^\+\+\+ b/(.+)$", patch, re.M))


def bucket(resolved: bool, error: str, patch: str, gold: set[str], test_tail: str) -> str:
    if resolved:
        return "resolved"
    if "timeout" in error.lower():
        return "timeout, with patch" if patch.strip() else "timeout, no patch"
    if "ContextWindow" in error:
        return "context overflow"
    if not patch.strip():
        return "finished, empty patch"
    if not gold & gold_files(patch):
        return "patch misses the gold file"
    if re.search(r"ImportError|ModuleNotFoundError|error during collection|SyntaxError|NameError", test_tail):
        return "right file, import/collection error in tests"
    return "right file, tests fail"


def split(run_dirs: list[Path], include_reserve: bool) -> list[dict]:
    tasks = {json.loads(ln)["instance_id"]: json.loads(ln)
             for ln in (ROOT / "data/competition/tasks.jsonl").read_text().splitlines()}
    rows = []
    for d in run_dirs:
        results = d / "task_results.jsonl"
        if not results.exists():
            continue
        verified = d / "verified_task_results.jsonl"
        ver = {json.loads(ln)["instance_id"]: json.loads(ln) for ln in verified.read_text().splitlines()} \
            if verified.exists() else {}
        for ln in results.read_text().splitlines():
            r = json.loads(ln)
            iid = r["instance_id"]
            if iid in RESERVE and not include_reserve:
                continue
            v = ver.get(iid, {})
            patches = sorted((d / "patches").glob(f"{iid}*")) if (d / "patches").exists() else []
            patch = patches[0].read_text() if patches else ""
            rows.append({"run": str(d.relative_to(ROOT)), "task": iid, "reserve": iid in RESERVE,
                         "resolved_source": "docker" if v else "harness",
                         "bucket": bucket(bool(v.get("resolved", r.get("resolved"))), str(r.get("error") or ""), patch,
                                          gold_files(tasks[iid]["patch"]), v.get("test_output_tail") or "")})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=",".join(DEFAULT_RUNS))
    ap.add_argument("--include-reserve", action="store_true", help="only to describe, never to design changes")
    ap.add_argument("--out", type=Path, default=ROOT / "data/analysis/failure_split.md")
    a = ap.parse_args()
    rows = split([ROOT / r for r in a.runs.split(",") if r], a.include_reserve)
    counts = Counter(r["bucket"] for r in rows)
    total = len(rows)
    lines = ["# Failure split (first pass of the autopsy)", "",
             f"Runs: {', '.join('`' + r + '`' for r in a.runs.split(','))}. Reserve tasks "
             f"{'included (description only)' if a.include_reserve else 'excluded'}. Method: `experiments/autopsy/METHOD.md`.",
             "", f"Task-runs: {total}", "", "| Bucket | Runs | Share |", "|---|---|---|"]
    lines += [f"| {b} | {counts.get(b, 0)} | {counts.get(b, 0) / total:.0%} |" for b in ORDER if total]
    lines += ["", "## Per task-run", "", "| Run | Task | Bucket | Resolved from |", "|---|---|---|---|"]
    lines += [f"| {r['run']} | {r['task']} | {r['bucket']} | {r['resolved_source']} |"
              for r in sorted(rows, key=lambda r: (r["bucket"], r["task"], r["run"]))]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:8 + len(ORDER) + 1]))


if __name__ == "__main__":
    main()
