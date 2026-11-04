"""Tool calls the server returned as reasoning (forum 753521): a model turn with no parsed tool call whose text
holds a Gemma 4 tool call (`call:name{...}`). The forum MRE says vLLM 0.19.1, non-streaming, extracts tool calls
only from content, so a call emitted before the thought channel closes stays in the reasoning and never runs.

Each such turn is classified as `in_thought` (a complete call, the MRE case), `truncated` (the closing
`<tool_call|>` is missing, as when the turn hits the output cap) or `malformed` (the call does not parse as
`call:name{`). The cost is measured as the seconds until the next executed tool call. Every trace outside
SKIP is read and grouped by model (GPU w4a16 is the scorer's model; TPU bf16 runs on another server) and,
on GPU, by whether the run belongs to the base family of `long_turns`.

Usage: PYTHONPATH=lab/src python3 -m lab.analysis.dropped_tool_calls [--eval data/eval]
       [--out data/analysis/dropped_tool_calls.json]
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from lab.analysis.command_timeouts import SKIP
from lab.analysis.long_turns import BASE_RUNS

CALL_RE = re.compile(r"call[:,]\s*\w+\s*\{")
WELL_FORMED_RE = re.compile(r"call:\w+\{")
CLOSE = "<tool_call|>"


def _ts(step: dict) -> float | None:
    return (step.get("extra") or {}).get("timestamp")


def kind(message: str) -> str | None:
    if not CALL_RE.search(message):
        # e.g. `[call 1/28] run_command{...}<tool_call|>`: the closing token without the `call:` prefix
        return "malformed" if CLOSE in message and re.search(r"\w+\{", message) else None
    if CLOSE not in message:
        return "truncated"
    return "in_thought" if WELL_FORMED_RE.search(message) else "malformed"


def drops(trace: dict) -> list[dict]:
    """Agent turns with no parsed tool call that carry one in their text, with what follows them."""
    steps = trace.get("steps", [])
    out, calls = [], 0
    for i, step in enumerate(steps):
        calls += len(step.get("tool_calls") or [])
        if step.get("source") != "agent" or step.get("tool_calls"):
            continue
        k = kind(step.get("message") or "")
        if k is None:
            continue
        nxt = steps[i + 1] if i + 1 < len(steps) else {}
        resumed = next((s for s in steps[i + 1:] if s.get("tool_calls")), None)
        lost = None
        if resumed and _ts(resumed) is not None and _ts(step) is not None:
            lost = round(_ts(resumed) - _ts(step), 1)
        out.append({"kind": k, "after_call": calls, "event": (step.get("extra") or {}).get("event_type"),
                    "nudged": (nxt.get("extra") or {}).get("event_type") == "continuation_nudge",
                    "seconds_to_next_call": lost})
    return out


def _results(run_dir: Path) -> dict[str, dict]:
    path = run_dir / "task_results.jsonl"
    if not path.exists():
        return {}
    rows = (json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return {r["instance_id"]: r for r in rows}


def _group(rows: list[dict]) -> dict:
    hit = [r for r in rows if r["drops"]]
    all_drops = [d for r in rows for d in r["drops"]]
    lost = sorted(d["seconds_to_next_call"] for d in all_drops if d["seconds_to_next_call"] is not None)
    return {"traces": len(rows), "agent_turns": sum(r["agent_turns"] for r in rows),
            "drops": dict(Counter(d["kind"] for d in all_drops)), "traces_with_drop": len(hit),
            "nudged": sum(d["nudged"] for d in all_drops),
            "seconds_to_next_call": lost,
            "resolved_with_drop": sum(bool(r["resolved"]) for r in hit),
            "cases": [{"run": r["run"], "task": r["task"], **d} for r in hit for d in r["drops"]]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", type=Path, default=Path("data/eval"))
    ap.add_argument("--out", type=Path, default=Path("data/analysis/dropped_tool_calls.json"))
    a = ap.parse_args()
    groups: dict[str, list[dict]] = {"gpu_base": [], "gpu_other": [], "tpu_bf16": []}
    results: dict[Path, dict] = {}
    for path in sorted(a.eval.glob("**/traces/trace_*.json")):
        if any(s in str(path) for s in SKIP):
            continue
        run_dir = path.parent.parent
        run = str(run_dir.relative_to(a.eval))
        trace = json.loads(path.read_text())
        steps = trace.get("steps", [])
        models = {s.get("model_name") for s in steps if s.get("model_name")}
        task = path.stem.removeprefix("trace_")
        if run_dir not in results:
            results[run_dir] = _results(run_dir)
        row = {"run": run, "task": task, "drops": drops(trace),
               "agent_turns": sum(s.get("source") == "agent" for s in steps),
               "resolved": results[run_dir].get(task, {}).get("resolved")}
        if any("bf16" in m for m in models):
            groups["tpu_bf16"].append(row)
        elif any("w4a16" in m for m in models):
            groups["gpu_base" if run.startswith(BASE_RUNS) else "gpu_other"].append(row)
    report = {k: _group(v) for k, v in groups.items()}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1))
    for k, g in report.items():
        print(k, {x: g[x] for x in ("traces", "agent_turns", "drops", "traces_with_drop", "nudged",
                                    "seconds_to_next_call", "resolved_with_drop")})
    print("->", a.out)


if __name__ == "__main__":
    main()
