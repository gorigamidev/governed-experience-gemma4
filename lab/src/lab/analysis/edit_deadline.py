"""Why "edit by call 12" is followed in about half the base traces (doc 03 §14.38): what the model can see of its
call count, and how late editing depends on the task statement.

- The harness reports the budget only through `budget_warning` in tool responses, and only near the end. For each
  trace, the call of the first warning, and whether the first edit after call 12 lands before it, within 3 calls of
  it, or later.
- Statement length per task (characters of `problem_statement`), split in thirds over the tasks with traces, and
  whether it carries code or a traceback. Rows per third: late-or-never rate and resolved; resolved by edit bucket
  within short and longer statements (verifiable runs only).
- Mentions of the deadline in the model's own text ("12" next to "call").

Rows come from `pool_report.task_row` and `edit_timing.bucket`, over the runs of `edit_timing.run_dirs`.

Usage: PYTHONPATH=lab/src python3 -m lab.analysis.edit_deadline [--eval data/eval] [--out data/analysis/edit_deadline.json]
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from lab.analysis.edit_timing import DEFAULT_CALLS, bucket, run_dirs
from lab.analysis.pool_report import AGENT_TOOLS, LATE_OR_NEVER, ROOT, _jsonl, task_row
from lab.eval.verifiable import load_verifiable

CODE_RE = re.compile(r"```|Traceback|>>>")
DEADLINE_RE = re.compile(r"\b12(th)?\b.{0,40}\bcall|\bcall.{0,40}\b12(th)?\b", re.I)


def warning_call(trace: dict) -> int | None:
    """Tool call after which the first `budget_warning` came back."""
    calls = 0
    for step in trace.get("steps", []):
        for tc in step.get("tool_calls") or []:
            if tc.get("function_name") != "submit_patch" and tc.get("function_name") not in AGENT_TOOLS:
                calls += 1
        if "budget_warning" in str(step.get("observation") or ""):
            return calls
    return None


def after_warning(first_edit: int | None, warned: int | None) -> str | None:
    if first_edit is None or first_edit <= 12 or warned is None:
        return None
    if first_edit < warned:
        return "before_warning"
    return "within_3_after" if first_edit <= warned + 3 else "later"


def group(bucket_name: str) -> str:
    if bucket_name == "early":
        return "early"
    return "late_or_never" if bucket_name in LATE_OR_NEVER else "mid"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", type=Path, default=Path("data/eval"))
    ap.add_argument("--out", type=Path, default=Path("data/analysis/edit_deadline.json"))
    a = ap.parse_args()
    verifiable = load_verifiable() or {}
    tasks = {json.loads(line)["instance_id"]: json.loads(line)
             for line in (ROOT / "data/competition/tasks.jsonl").read_text().splitlines() if line.strip()}
    rows = []
    for d in run_dirs(a.eval):
        harness, docker = _jsonl(d / "task_results.jsonl"), _jsonl(d / "verified_task_results.jsonl")
        for iid, h in harness.items():
            path = d / "traces" / f"trace_{iid.replace('/', '__')}.json"
            if not path.exists():
                continue
            trace = json.loads(path.read_text())
            r = task_row(iid, h, docker, verifiable, d)
            warned = warning_call(trace)
            r.update(group=group(bucket(r, DEFAULT_CALLS)), warned=warned,
                     after_warning=after_warning(r["first_edit_call"], warned),
                     deadline_mentions=sum(bool(DEADLINE_RE.search(s.get("message") or ""))
                                           for s in trace.get("steps", []) if s.get("source") == "agent"))
            rows.append(r)
    by_task = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)
    order = sorted(by_task, key=lambda t: len(tasks[t]["problem_statement"]))
    k = len(order) // 3
    thirds = {"short": order[:k], "middle": order[k:2 * k], "long": order[2 * k:]}
    short = set(thirds["short"])

    def third(ts: list[str]) -> dict:
        rs = [r for t in ts for r in by_task[t]]
        return {"tasks": len(ts), "max_chars": max(len(tasks[t]["problem_statement"]) for t in ts), "runs": len(rs),
                "late_or_never": sum(r["group"] == "late_or_never" for r in rs),
                "resolved": sum(r["resolved"] for r in rs),
                "tasks_with_code": sum(bool(CODE_RE.search(tasks[t]["problem_statement"])) for t in ts)}

    def resolved_by_group(rs: list[dict]) -> dict:
        v = [r for r in rs if r["verifiable"]]
        return {g: f"{sum(r['resolved'] for r in v if r['group'] == g)}/{sum(r['group'] == g for r in v)}"
                for g in ("early", "mid", "late_or_never")}

    report = {
        "traces": len(rows), "tasks": len(by_task),
        "first_warning_call": dict(sorted(Counter(str(r["warned"]) for r in rows).items())),
        "edit_after_12_vs_warning": dict(Counter(r["after_warning"] for r in rows if r["after_warning"])),
        "deadline_mentions": {g: sum(r["deadline_mentions"] for r in rows if r["group"] == g)
                              for g in ("early", "mid", "late_or_never")},
        "statement_thirds": {name: third(ts) for name, ts in thirds.items()},
        "resolved_by_group": {"short": resolved_by_group([r for r in rows if r["task"] in short]),
                              "middle_and_long": resolved_by_group([r for r in rows if r["task"] not in short])},
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
