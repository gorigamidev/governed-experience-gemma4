"""When the agent edits, relative to its call budget (lineage for doc 03 §14.38): in base-family GPU traces, at which
call the first successful edit lands, how many traces never edit, how each group ends, when the gold file was first
read and what the calls before the first edit were.

The base prompt already says "by your 12th tool call you must have edited a source file", so the `early` bucket is
also the compliance of that rule. Rows come from `pool_report.task_row`, so `first_edit_call` (submit_patch not
counted), `edits_after_budget` (edit refused with BudgetExceeded) and `resolved` (Docker verdict on verifiable tasks)
mean the same as in the pool reports. Runs are the GPU base family of `long_turns`.

Usage: PYTHONPATH=lab/src python3 -m lab.analysis.edit_timing [--eval data/eval] [--out data/analysis/edit_timing.json]
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from lab.analysis.command_timeouts import SKIP
from lab.analysis.long_turns import BASE_RUNS
from lab.analysis.pool_report import AGENT_TOOLS, LATE_OR_NEVER, ROOT, _jsonl, bucket, task_row
from lab.eval.verifiable import load_verifiable

DEFAULT_CALLS = 28
BUCKETS = ("early", "mid", "late", "refused_only", "never")


def ending(row: dict, max_calls: int) -> str:
    if "session timeout" in row["error"]:
        return "time"
    if row["tool_calls"] >= max_calls:
        return "calls"
    return "error" if row["error"] else "submit"


def command_kind(cmd: str) -> str:
    if re.search(r"\bpytest\b", cmd):
        return "pytest"
    if re.search(r"python3? -c|<<", cmd):
        return "python_inline"
    if re.search(r"python3?\s+\S+\.py", cmd):
        return "python_script"
    if re.search(r"\b(grep|rg|find|ls|cat|sed|head|tail|wc|git)\b", cmd):
        return "shell_read"
    return "other"


def pre_edit(trace: dict, first_edit_call: int | None, gold: set[str]) -> dict:
    """Call of the first `read_file` on a gold file, and the kinds of calls made before the first edit."""
    calls = 0
    gold_read = None
    kinds: Counter = Counter()
    for step in trace.get("steps", []):
        for tc in step.get("tool_calls") or []:
            name = tc.get("function_name")
            if name == "submit_patch" or name in AGENT_TOOLS:
                continue
            calls += 1
            args = tc.get("arguments") or {}
            if gold_read is None and name == "read_file" and any(g in json.dumps(args) for g in gold):
                gold_read = calls
            if first_edit_call is None or calls < first_edit_call:
                kinds[command_kind(str(args.get("command", ""))) if name == "run_command" else name] += 1
    return {"gold_read": gold_read, "kinds": kinds}


def run_dirs(eval_dir: Path) -> list[Path]:
    out = []
    for f in sorted(eval_dir.glob("**/task_results.jsonl")):
        run = str(f.parent.relative_to(eval_dir))
        if any(s in run for s in SKIP) or not run.startswith(BASE_RUNS) or not (f.parent / "traces").is_dir():
            continue
        out.append(f.parent)
    return out


def _group(rows: list[dict]) -> dict:
    n = len(rows) or 1
    read = sorted(r["gold_read"] for r in rows if r["gold_read"] is not None)
    kinds: Counter = Counter()
    for r in rows:
        kinds.update(r["kinds"])
    return {"traces": len(rows), "resolved": sum(r["resolved"] for r in rows),
            "verifiable": sum(r["verifiable"] for r in rows),
            "resolved_verifiable": sum(r["resolved"] for r in rows if r["verifiable"]),
            "endings": dict(Counter(r["ending"] for r in rows)),
            "gold_read": len(read), "gold_read_call_p50": read[len(read) // 2] if read else None,
            "pre_edit_calls_per_trace": {k: round(v / n, 1) for k, v in kinds.most_common()}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", type=Path, default=Path("data/eval"))
    ap.add_argument("--out", type=Path, default=Path("data/analysis/edit_timing.json"))
    a = ap.parse_args()
    verifiable = load_verifiable() or {}
    tasks = {json.loads(line)["instance_id"]: json.loads(line)
             for line in (ROOT / "data/competition/tasks.jsonl").read_text().splitlines() if line.strip()}
    rows = []
    for d in run_dirs(a.eval):
        meta = d / "run_meta.json"
        max_calls = (json.loads(meta.read_text()).get("budgets", {}).get("max_tool_calls", DEFAULT_CALLS)
                     if meta.exists() else DEFAULT_CALLS)
        harness, docker = _jsonl(d / "task_results.jsonl"), _jsonl(d / "verified_task_results.jsonl")
        for iid, h in harness.items():
            path = d / "traces" / f"trace_{iid.replace('/', '__')}.json"
            if not path.exists():
                continue
            r = task_row(iid, h, docker, verifiable, d)
            gold = set(re.findall(r"^\+\+\+ b/(.+)$", tasks[iid]["patch"], re.M))
            r.update(run=str(d.relative_to(a.eval)), bucket=bucket(r, max_calls), ending=ending(r, max_calls),
                     **pre_edit(json.loads(path.read_text()), r["first_edit_call"], gold))
            rows.append(r)
    by_task = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)
    mixed = {t: v for t, v in by_task.items()
             if any(r["bucket"] == "early" for r in v) and any(r["bucket"] in LATE_OR_NEVER for r in v)}
    stuck = {t: f"{sum(r['bucket'] in LATE_OR_NEVER for r in v)}/{len(v)}" for t, v in sorted(by_task.items())
             if len(v) >= 3 and sum(r["bucket"] in LATE_OR_NEVER for r in v) / len(v) >= .5}
    edits = sorted(r["first_edit_call"] for r in rows if r["first_edit_call"] is not None)
    report = {
        "traces": len(rows), "runs": sorted({r["run"] for r in rows}), "tasks": len(by_task),
        "edit_by_12_compliance": f"{sum(r['bucket'] == 'early' for r in rows)}/{len(rows)}",
        "first_edit_call_hist": dict(sorted(Counter(edits).items())),
        "buckets": {b: _group([r for r in rows if r["bucket"] == b]) for b in BUCKETS},
        "within_task": {
            "tasks": len(mixed),
            "early": _group([r for v in mixed.values() for r in v if r["bucket"] == "early"]),
            "late_or_never": _group([r for v in mixed.values() for r in v if r["bucket"] in LATE_OR_NEVER]),
            "tasks_where_an_early_run_resolves": sum(any(r["resolved"] for r in v if r["bucket"] == "early")
                                                     for v in mixed.values())},
        "tasks_mostly_late_or_never": stuck,
        "share_tasks_mostly_late_or_never": f"{len(stuck)}/{sum(len(v) >= 3 for v in by_task.values())}",
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
