"""Doc 03 §14 item 52, phase A: would the `focus` chunk have covered the first edit, and how many calls would it save?

For every GPU pool trace of the `bf-gov*` family with a successful first edit: take the literal terms of the model's
first two searches before that edit, run `focus` on the base snapshot and check whether the edit's `old_string` is in
the returned code. Saving = searches and reads before the first edit, minus the one `focus` call.

Usage: PYTHONPATH=lab/src lab/.venv/bin/python -m lab.analysis.focus_sim [--out data/analysis/focus_sim.md]
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import re
import statistics
import tarfile
import tempfile
from collections import defaultdict
from pathlib import Path

from lab.analysis.pool_report import AGENT_TOOLS, ROOT
from lab.ingest.load_traces import _is_error

FOCUS = ROOT / "lab/src/lab/export/skill_addons/focus/scripts/focus.py"
SEARCH = re.compile(r"\b(?:git\s+grep|grep|rg)\b((?:\s+-{1,2}[\w=-]+)*)\s+(?:\"([^\"]+)\"|'([^']+)'|([^\s|;&]+))")
LOOK = re.compile(r"\b(cat|sed|head|tail|nl)\b")
MIN_TASK_COVER, MIN_SAVING = 0.5, 4


def load_focus():
    spec = importlib.util.spec_from_file_location("focus_skill", FOCUS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def before_first_edit(trace: dict) -> tuple[list[dict], dict | None]:
    """Charged calls before the first successful edit, and that edit's arguments (None if there is none)."""
    calls = []
    for step in trace.get("steps", []):
        for tc in step.get("tool_calls") or []:
            name = tc.get("function_name")
            if name == "submit_patch" or name in AGENT_TOOLS:
                continue
            if name in ("edit_file", "write_file") and not _is_error((step.get("observation") or {}).get("content")):
                return calls, {"tool": name, **(tc.get("arguments") or {})}
            calls.append(tc)
    return calls, None


def terms_of(calls: list[dict]) -> list[str]:
    terms = []
    for tc in calls:
        cmd = str((tc.get("arguments") or {}).get("command") or "")
        if tc.get("function_name") != "run_command":
            continue
        for m in SEARCH.finditer(cmd):
            t = next(g for g in m.groups()[1:] if g)
            if t not in terms:
                terms.append(t)
        if len(terms) >= 2:
            break
    return terms[:2]


def is_look(tc: dict) -> bool:
    if tc.get("function_name") == "read_file":
        return True
    cmd = str((tc.get("arguments") or {}).get("command") or "")
    return tc.get("function_name") == "run_command" and bool(SEARCH.search(cmd) or LOOK.search(cmd))


def snapshot(task: str, cache: Path) -> Path:
    root = cache / task
    if not root.exists():
        with tarfile.open(ROOT / f"data/competition/snapshots/{task}.tgz") as tf:
            tf.extractall(root, filter="data")
    return root


def simulate(cache: Path) -> list[dict]:
    focus = load_focus()
    rows = []
    for path in sorted(glob.glob(str(ROOT / "data/eval/pool-*-gpu/results/bf-gov*/traces/trace_*.json"))):
        task = Path(path).stem.removeprefix("trace_")
        session = Path(path).parts[-5]
        calls, edit = before_first_edit(json.loads(Path(path).read_text()))
        if edit is None:
            continue
        terms = terms_of(calls)
        covered = False
        if terms and edit["tool"] == "edit_file" and edit.get("old_string"):
            out = focus.focus(", ".join(terms), str(snapshot(task, cache)))
            code = "\n".join(r["code"] for r in out.get("regions", []))
            covered = edit["old_string"].strip() in code
        looks = sum(is_look(tc) for tc in calls)
        rows.append({"session": session, "arm": Path(path).parts[-3], "task": task, "terms": terms,
                     "calls_before": len(calls), "looks": looks, "covered": covered, "saving": looks - 1})
    return rows


def report(rows: list[dict]) -> tuple[list[str], bool]:
    by_task = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r["covered"])
    task_cover = {t: sum(v) > len(v) / 2 for t, v in by_task.items()}
    savings = [r["saving"] for r in rows if r["covered"]]
    share = sum(task_cover.values()) / len(task_cover) if task_cover else 0.0
    med = statistics.median(savings) if savings else 0
    go = share >= MIN_TASK_COVER and med >= MIN_SAVING
    lines = [f"# focus simulation (doc 03 §14 item 52, phase A)", "",
             f"- Traces with a first successful edit: {len(rows)}; covered {sum(r['covered'] for r in rows)}; "
             f"with search terms {sum(bool(r['terms']) for r in rows)}",
             f"- Tasks covered by majority: {sum(task_cover.values())}/{len(task_cover)} = {share:.0%} "
             f"(rule >= {MIN_TASK_COVER:.0%})",
             f"- Saving on covered traces: median {med} (rule >= {MIN_SAVING}); calls before first edit, median "
             f"{statistics.median([r['calls_before'] for r in rows]) if rows else 0}",
             f"- **Phase A: {'GO' if go else 'NO GO'}**", "", "| task | covered/traces | median saving if covered |",
             "|---|---|---|"]
    for t in sorted(by_task):
        s = [r["saving"] for r in rows if r["task"] == t and r["covered"]]
        lines.append(f"| {t} | {sum(by_task[t])}/{len(by_task[t])} | {statistics.median(s) if s else '-'} |")
    return lines, go


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "data/analysis/focus_sim.md")
    args = ap.parse_args()
    with tempfile.TemporaryDirectory() as d:
        rows = simulate(Path(d))
    lines, _ = report(rows)
    args.out.write_text("\n".join(lines) + "\n")
    args.out.with_suffix(".jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
