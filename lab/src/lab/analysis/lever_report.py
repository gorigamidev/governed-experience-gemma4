"""Report for the four edit-timing levers (doc 03 §14.40): the per-arm columns of `pool_report` plus each lever's
own compliance metric, and the readings written before the TPU pool.

- (a) `bf-gov-ct`: share of tool-call turns whose text (the turn itself or the text turn just before it) carries
  `Step N/28`, and share of those whose N is within 1 of the real call number.
- (b) `bf-gov-ti`: on short statements (at most SHORT characters), median first edit and share of traces that look at
  tests (`grep ... tests`) or `git log` before the first edit.
- (c) `bf-gov-ec`: median calls between the first read of the first edited file and that edit.
- (d) `bf-gov-22`: call of the first `budget_warning` in every trace with at least 12 calls. Also reported, without
  deciding, as the harness counts: ADK rejects a call that lacks a mandatory parameter before the tool runs, and
  swegemma does not charge it to the budget (correction of 1 Nov, doc 03 §14.40).

Descriptive for every arm: tool calls the parser dropped (`dropped_tool_calls`); a counter that breaks the call
syntax harms even when it complies (preflight 20261028-132016).

Every lever is read against the base arm on the same tasks; the TPU pool is exploratory and decides nothing for the
scorer (§14.40). Usage: PYTHONPATH=lab/src python3 -m lab.analysis.lever_report \
    --arms bf-gov-to=DIR,bf-gov-ct=DIR,bf-gov-ti=DIR,bf-gov-ec=DIR,bf-gov-22=DIR [--out FILE]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from lab.analysis.command_timeouts import quantile
from lab.analysis.dropped_tool_calls import drops
from lab.analysis.edit_deadline import warning_call
from lab.analysis.pool_report import AGENT_TOOLS, REJECTED, ROOT, _jsonl, arm_summary, harness_warning_call, task_row
from lab.eval.verifiable import load_verifiable

BASE = "bf-gov-to"
LEVERS = {"bf-gov-ct": "a", "bf-gov-ti": "b", "bf-gov-ec": "c", "bf-gov-22": "d"}
ORDER = ("bf-gov-22", "bf-gov-ti", "bf-gov-ct", "bf-gov-ec")  # tie order of §14.40: structure first, then b, a, c
MIN_COMMON = 16
MAX_INFRA = 2
SHORT = 243
COUNTER_RE = re.compile(r"\bStep (\d+)/\d+")
LOOKED_RE = re.compile(r"git log|grep[^|]*\btests?\b")
EDITS = ("edit_file", "write_file")


def _calls(trace: dict) -> list[tuple[int, dict, str, int]]:
    """(call number, tool call, text of that turn and the text-only turn before it, turn index) per counted call."""
    out, n, prev_text = [], 0, ""
    for turn, step in enumerate(trace.get("steps", [])):
        if step.get("source") != "agent":
            continue
        text = step.get("message") or ""
        tcs = step.get("tool_calls") or []
        if not tcs:
            prev_text = text
            continue
        for tc in tcs:
            if tc.get("function_name") == "submit_patch" or tc.get("function_name") in AGENT_TOOLS:
                continue
            n += 1
            out.append((n, tc, prev_text + "\n" + text, turn))
        prev_text = ""
    return out


def counter(trace: dict) -> tuple[int, int, int]:
    """(turns, turns with a counter, counters within 1 of the real call)."""
    turns = tagged = right = 0
    seen = set()
    for n, _, text, turn in _calls(trace):
        if turn in seen:
            continue
        seen.add(turn)
        turns += 1
        m = COUNTER_RE.search(text)
        if m:
            tagged += 1
            right += abs(int(m.group(1)) - n) <= 1
    return turns, tagged, right


def _path(tc: dict) -> str:
    args = tc.get("arguments") or {}
    return str(args.get("filepath") or args.get("file_path") or args.get("path") or "")


def first_edit(trace: dict) -> tuple[int | None, str]:
    for n, tc, _, _ in _calls(trace):
        if tc.get("function_name") in EDITS:
            return n, _path(tc)
    return None, ""


def looked_before_edit(trace: dict) -> bool:
    edit_at, _ = first_edit(trace)
    for n, tc, _, _ in _calls(trace):
        if edit_at is not None and n >= edit_at:
            break
        if tc.get("function_name") == "run_command" and LOOKED_RE.search(str((tc.get("arguments") or {}).get("command"))):
            return True
    return False


def read_to_edit_gap(trace: dict) -> int | None:
    edit_at, path = first_edit(trace)
    if edit_at is None or not path:
        return None
    name = path.removeprefix("/workspace/")
    for n, tc, _, _ in _calls(trace):
        if n >= edit_at:
            break
        args = json.dumps(tc.get("arguments") or {})
        if tc.get("function_name") == "read_file" and name in args or (
                tc.get("function_name") == "run_command" and name in args and re.search(r"\b(cat|sed|head|grep)\b", args)):
            return edit_at - n
    return None


def lever_metrics(traces: list[tuple[str, dict]], short: set[str]) -> dict:
    turns = tagged = right = 0
    for _, tr in traces:
        a, b, c = counter(tr)
        turns, tagged, right = turns + a, tagged + b, right + c
    short_tr = [tr for t, tr in traces if t in short]
    short_edits = sorted(e for e in (first_edit(tr)[0] for tr in short_tr) if e is not None)
    gaps = sorted(g for g in (read_to_edit_gap(tr) for _, tr in traces) if g is not None)
    warnings = [warning_call(tr) for _, tr in traces if len(_calls(tr)) >= 12]
    charged = [harness_warning_call(tr) for _, tr in traces if len(_calls(tr)) >= 12]
    return {"counter_turns": turns, "counter_tagged": tagged, "counter_right": right,
            "short_traces": len(short_tr), "short_first_edit_p50": quantile(short_edits, .5) if short_edits else None,
            "short_looked": sum(looked_before_edit(tr) for tr in short_tr),
            "gap_p50": quantile(gaps, .5) if gaps else None,
            "warned_traces": len(warnings), "warned_at_12": sum(w == 12 for w in warnings),
            "warned_at_12_harness": sum(w == 12 for w in charged), "rejected_calls": sum(
                REJECTED in str(st.get("observation") or "") for _, tr in traces for st in tr.get("steps", [])),
            "dropped_calls": sum(len(drops(tr)) for _, tr in traces)}


def complies(lever: str, m: dict, base: dict) -> bool:
    if lever == "a":
        return (m["counter_turns"] > 0 and m["counter_tagged"] >= .8 * m["counter_turns"]
                and m["counter_right"] >= .8 * m["counter_tagged"])
    if lever == "b":
        return (m["short_first_edit_p50"] is not None and m["short_first_edit_p50"] <= 8
                and m["short_looked"] >= .5 * m["short_traces"])
    if lever == "c":
        return m["gap_p50"] is not None and base["gap_p50"] is not None and m["gap_p50"] <= base["gap_p50"] - 2
    return m["warned_traces"] > 0 and m["warned_at_12"] == m["warned_traces"]


def readings(summary: dict[str, dict], metrics: dict[str, dict], common: int) -> list[str]:
    out = []
    valid = common >= MIN_COMMON and all(s["infra_errors"] <= MAX_INFRA for s in summary.values())
    out.append(f"1 (validity): {'OK' if valid else 'FAIL'} | {common} common tasks, infra "
               + ", ".join(f"{a} {s['infra_errors']}" for a, s in summary.items()))
    floor = summary[BASE]["late_or_never"] <= 2
    out.append(f"2 (floor): {'APPLIES' if floor else 'no'} | base late_or_never {summary[BASE]['late_or_never']}")
    ok = {}
    for arm, lever in LEVERS.items():
        if arm in metrics:
            ok[arm] = complies(lever, metrics[arm], metrics[BASE])
            out.append(f"3{lever} ({arm}): {'COMPLIES' if ok[arm] else 'no'} | "
                       + json.dumps({k: v for k, v in metrics[arm].items()}))
    passing = [a for a in ORDER if ok.get(a)]
    if not valid or not passing:
        first = "bf-gov-22"
    elif floor:
        first = passing[0]
    else:
        best = min(summary[a]["late_or_never"] for a in passing)
        first = next(a for a in passing if summary[a]["late_or_never"] == best)
    out.append(f"Submission order: {first} first, then " + ", ".join(a for a in ORDER if a != first))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True, help="name=results_dir,...")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    arms = {k: Path(v) for k, v in (item.split("=", 1) for item in a.arms.split(","))}
    if BASE not in arms:
        raise SystemExit(f"lever_report: the base arm {BASE} is required")
    verifiable = load_verifiable() or {}
    tasks = {json.loads(line)["instance_id"]: json.loads(line)
             for line in (ROOT / "data/competition/tasks.jsonl").read_text().splitlines() if line.strip()}
    harness = {n: _jsonl(p / "task_results.jsonl") for n, p in arms.items()}
    common = sorted(set.intersection(*(set(h) for h in harness.values())))
    short = {t for t in common if len(tasks[t]["problem_statement"]) <= SHORT}
    summary, metrics = {}, {}
    for n, p in arms.items():
        max_calls = json.loads((p / "run_meta.json").read_text())["budgets"]["max_tool_calls"]
        docker = _jsonl(p / "verified_task_results.jsonl")
        rows = [task_row(t, harness[n][t], docker, verifiable, p) for t in common]
        summary[n] = {"max_tool_calls": max_calls, **arm_summary(rows, max_calls)}
        traces = [(t, json.loads((p / "traces" / f"trace_{t}.json").read_text())) for t in common
                  if (p / "traces" / f"trace_{t}.json").exists()]
        metrics[n] = lever_metrics(traces, short)
    cols = ["max_tool_calls", "n", "R", "edited", "edit_by_12", "late_or_never", "mean_min", "aux", "writes_outside",
            "infra_errors"]
    lines = [f"# Lever report: {', '.join(arms)}", "", f"Common tasks: {len(common)} ({len(short)} short)", "",
             "| arm | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    lines += [f"| {n} | " + " | ".join(str(summary[n][c]) for c in cols) + " |" for n in arms]
    lines += ["", "## Readings (doc 03 §14.40)", ""] + [f"- {r}" for r in readings(summary, metrics, len(common))]
    text = "\n".join(lines) + "\n"
    print(text)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text)
        print("->", a.out)


if __name__ == "__main__":
    main()
