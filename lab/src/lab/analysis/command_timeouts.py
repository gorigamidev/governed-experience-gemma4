"""Command hangs in agent traces (doc 03 §14.35): how often `run_command` hits the per-command limit, on which tasks,
and how long the commands that do finish take. Lineage of the `timeout 90` habit.

Copies of the same run (`-raw`, `_parts`) and local or rehearsal runs are skipped.

Usage: PYTHONPATH=lab/src python3 -m lab.analysis.command_timeouts [--eval data/eval] [--out data/analysis/command_timeouts.json]
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

SKIP = ("preflight", "local", "lora", "-raw", "_parts")
INLINE_RE = re.compile(r"python3? -c|<<")
WRAPPED_RE = re.compile(r"\btimeout\s+\d+")
SCRIPT_RE = re.compile(r"python3?\s+\S+\.py")


def command_kind(cmd: str) -> str:
    if "pytest" in cmd:
        return "pytest"
    if INLINE_RE.search(cmd):
        return "inline"
    return "script" if SCRIPT_RE.search(cmd) else "other"


def trace_commands(trace: dict) -> list[dict]:
    """Each `run_command` call: kind, whether it hit the limit, and its duration when the trace records it."""
    out = []
    for step in trace.get("steps", []):
        obs = step.get("observation") or {}
        if not isinstance(obs, dict):
            continue
        for call in step.get("tool_calls") or []:
            if call.get("function_name") != "run_command":
                continue
            cmd = (call.get("arguments") or {}).get("command", "")
            content = str(obs.get("content", ""))
            hung = "TimeoutExceeded" in content
            t0, t1 = (step.get("extra") or {}).get("elapsed_s"), (obs.get("extra") or {}).get("elapsed_s")
            out.append({"kind": command_kind(cmd), "hung": hung, "wrapped": bool(WRAPPED_RE.search(cmd)),
                        "cut": '"exit_code": 124' in content,
                        "seconds": t1 - t0 if t0 is not None and t1 is not None else None})
    return out


def quantile(xs: list[float], q: float) -> float:
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", type=Path, default=Path("data/eval"))
    ap.add_argument("--out", type=Path, default=Path("data/analysis/command_timeouts.json"))
    a = ap.parse_args()
    traces = hung_traces = 0
    kinds, tasks, per_task = Counter(), Counter(), Counter()
    finished: list[float] = []
    for path in sorted(a.eval.glob("**/traces/trace_*.json")):
        if any(s in str(path) for s in SKIP):
            continue
        cmds = trace_commands(json.loads(path.read_text()))
        task = path.stem.removeprefix("trace_")
        traces += 1
        per_task[task] += 1
        hung = [c for c in cmds if c["hung"]]
        hung_traces += bool(hung)
        if hung:
            tasks[task] += 1
        kinds.update(c["kind"] for c in hung)
        finished += [c["seconds"] for c in cmds if not c["hung"] and c["seconds"] is not None]
    finished.sort()
    report = {
        "traces": traces, "traces_with_hang": hung_traces, "hangs_by_kind": dict(kinds),
        "tasks_with_hang": {t: f"{n}/{per_task[t]}" for t, n in tasks.most_common()},
        "finished_commands": len(finished),
        "finished_seconds": {"p50": quantile(finished, .5), "p90": quantile(finished, .9),
                             "p99": quantile(finished, .99), "max": finished[-1],
                             "over_60": sum(x > 60 for x in finished), "over_90": sum(x > 90 for x in finished)},
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
