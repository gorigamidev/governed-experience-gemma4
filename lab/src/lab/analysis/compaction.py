"""Context compaction in base-family GPU traces (forum 753225: the summary call is sampled without the submission's
temperature, top_p and top_k). Our GPU runs use the Getting Started values (`run_eval.COMPACTION`: token threshold
14,336, 5 events kept), and ADK compacts before a request whose previous prompt reached the threshold.

The trace does not record the summary call, so a compaction is read as a turn whose prompt is at least 20% smaller
than the previous one. For each trace: how many compactions, at which tool call, whether the first one lands before
the first edit, and how many `read_file` calls in the next WINDOW calls reopen a file already read before it (a
re-read), against the re-read rate of every other `read_file` and of those at call LATE or later outside a
post-compaction window (re-reads grow along a trace). Rows come from `pool_report.task_row` as in
`edit_timing`. The association with resolving is confounded by length (long traces both compact and fail), so it is
reported, not interpreted as cause.

Two context-growth readings for the levers a submission has (it cannot set compaction itself, doc 03 §14.39):
`growth_fit` regresses each turn's prompt growth on the previous turn's completion tokens and observation size
(chars / 3.5); a completion coefficient near 1 means thinking stays in the context. `cap_counterfactual` removes,
turn by turn, the completion tokens above CAP (the `bf-gov-tb` thinking budget plus visible text) and counts the
traces whose first compaction would not trigger. It assumes the rest of the trajectory unchanged, so it is a bound,
not a prediction.

Usage: PYTHONPATH=lab/src python3 -m lab.analysis.compaction [--eval data/eval] [--out data/analysis/compaction.json]
"""

from __future__ import annotations

import argparse
import ast
import json
from collections import defaultdict
from pathlib import Path

from lab.analysis.edit_timing import BUCKETS, DEFAULT_CALLS, bucket, run_dirs
from lab.analysis.pool_report import AGENT_TOOLS, _jsonl, task_row
from lab.eval.verifiable import load_verifiable


def _run_eval_compaction() -> dict:
    """`run_eval.COMPACTION`, read from the source: importing run_eval needs yaml, which CI does not install."""
    tree = ast.parse((Path(__file__).resolve().parents[1] / "eval/run_eval.py").read_text())
    return next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "COMPACTION" for t in node.targets))


COMPACTION = _run_eval_compaction()
DROP = 0.8
WINDOW = 6
LATE = 12  # the base prompt's edit deadline
CAP = 2100  # thinking_budget 2048 of bf-gov-tb plus a short visible text


def _prompt_tokens(step: dict) -> int | None:
    m = step.get("metrics")
    if isinstance(m, str):
        m = ast.literal_eval(m)
    return (m or {}).get("prompt_tokens")


def compactions(trace: dict) -> dict:
    """Tool-call index of each compaction, max prompt, and re-reads right after a compaction versus elsewhere."""
    calls, prev, at, max_prompt = 0, None, [], 0
    reads: list[tuple[int, str]] = []
    for step in trace.get("steps", []):
        if step.get("source") != "agent":
            continue
        p = _prompt_tokens(step)
        if p:
            if prev and p < DROP * prev:
                at.append(calls)
            prev, max_prompt = p, max(max_prompt, p)
        for tc in step.get("tool_calls") or []:
            name = tc.get("function_name")
            if name == "submit_patch" or name in AGENT_TOOLS:
                continue
            calls += 1
            if name == "read_file":
                args = tc.get("arguments") or {}
                reads.append((calls, str(args.get("filepath") or args.get("file_path") or "")))
    after = {"reads": 0, "rereads": 0}
    other = {"reads": 0, "rereads": 0}
    late = {"reads": 0, "rereads": 0}
    for i, (call, path) in enumerate(reads):
        seen = path in {p for _, p in reads[:i]}
        near = any(c < call <= c + WINDOW for c in at)
        for b in [after if near else other] + ([late] if call >= LATE and not near else []):
            b["reads"] += 1
            b["rereads"] += seen
    return {"compaction_calls": at, "max_prompt": max_prompt, "after": after, "other": other, "late": late}


def growth_pairs(trace: dict) -> list[tuple[float, float, float]]:
    """(completion tokens, observation tokens estimate, prompt growth) for consecutive turns without compaction."""
    turns = [s for s in trace.get("steps", []) if s.get("source") == "agent" and _prompt_tokens(s)]
    out = []
    for a, b in zip(turns, turns[1:]):
        pa, pb = _prompt_tokens(a), _prompt_tokens(b)
        if pb < DROP * pa:
            continue
        m = a.get("metrics")
        m = ast.literal_eval(m) if isinstance(m, str) else (m or {})
        obs = len(str(a["observation"])) / 3.5 if a.get("observation") else 0.0
        out.append((float(m.get("completion_tokens") or 0), obs, float(pb - pa)))
    return out


def growth_fit(pairs: list[tuple[float, float, float]]) -> dict:
    """Least squares growth = a * completion + b * observation + c."""
    n = len(pairs)
    m1, m2, my = (sum(p[i] for p in pairs) / n for i in range(3))
    c11 = sum((p[0] - m1) ** 2 for p in pairs)
    c22 = sum((p[1] - m2) ** 2 for p in pairs)
    c12 = sum((p[0] - m1) * (p[1] - m2) for p in pairs)
    c1y = sum((p[0] - m1) * (p[2] - my) for p in pairs)
    c2y = sum((p[1] - m2) * (p[2] - my) for p in pairs)
    det = c11 * c22 - c12 ** 2
    a = (c1y * c22 - c2y * c12) / det
    b = (c2y * c11 - c1y * c12) / det
    return {"pairs": n, "completion_coef": round(a, 3), "observation_coef": round(b, 3),
            "intercept": round(my - a * m1 - b * m2, 1), "mean_completion": round(m1), "mean_observation": round(m2),
            "mean_growth": round(my)}


def cap_counterfactual(trace: dict, threshold: int = COMPACTION["token_threshold"]) -> dict:
    """Completion tokens above CAP, and whether the first compaction still triggers once they are removed."""
    total = excess = 0
    prev, removed, triggers, compacted = None, 0, False, False
    for step in trace.get("steps", []):
        p = _prompt_tokens(step)
        if step.get("source") != "agent" or not p:
            continue
        m = step.get("metrics")
        m = ast.literal_eval(m) if isinstance(m, str) else (m or {})
        ct = m.get("completion_tokens") or 0
        total += ct
        excess += max(0, ct - CAP)
        if compacted or triggers:
            continue
        if prev and p < DROP * prev:
            compacted = True
            continue
        triggers = p - removed >= threshold
        removed += max(0, ct - CAP)
        prev = p
    return {"completion": total, "excess": excess, "still_triggers": triggers}


def timing(row: dict) -> str:
    if not row["compaction_calls"]:
        return "none"
    first_edit = row["first_edit_call"]
    return "before_edit" if first_edit is None or row["compaction_calls"][0] < first_edit else "after_edit"


def _group(rows: list[dict]) -> dict:
    return {"traces": len(rows), "verifiable": sum(r["verifiable"] for r in rows),
            "resolved_verifiable": sum(r["resolved"] for r in rows if r["verifiable"]),
            "never_edit": sum(r["first_edit_call"] is None for r in rows)}


def _rate(rows: list[dict], key: str) -> dict:
    reads = sum(r[key]["reads"] for r in rows)
    rereads = sum(r[key]["rereads"] for r in rows)
    return {"reads": reads, "rereads": rereads, "rate": round(rereads / reads, 3) if reads else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", type=Path, default=Path("data/eval"))
    ap.add_argument("--out", type=Path, default=Path("data/analysis/compaction.json"))
    a = ap.parse_args()
    verifiable = load_verifiable() or {}
    rows = []
    for d in run_dirs(a.eval):
        harness, docker = _jsonl(d / "task_results.jsonl"), _jsonl(d / "verified_task_results.jsonl")
        for iid, h in harness.items():
            path = d / "traces" / f"trace_{iid.replace('/', '__')}.json"
            if not path.exists():
                continue
            r = task_row(iid, h, docker, verifiable, d)
            trace = json.loads(path.read_text())
            r.update(run=str(d.relative_to(a.eval)), **compactions(trace), pairs=growth_pairs(trace),
                     cap=cap_counterfactual(trace))
            r["timing"] = timing(r)
            r["bucket"] = bucket(r, DEFAULT_CALLS)
            rows.append(r)
    by_task = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)
    mixed = [v for v in by_task.values() if {"none", "after_edit"} & {r["timing"] for r in v}
             and any(r["timing"] == "before_edit" for r in v)]
    firsts = sorted(r["compaction_calls"][0] for r in rows if r["compaction_calls"])
    report = {
        "config": COMPACTION, "traces": len(rows), "tasks": len(by_task),
        "with_compaction": sum(bool(r["compaction_calls"]) for r in rows),
        "compactions_per_trace": {n: sum(len(r["compaction_calls"]) == n for r in rows) for n in range(4)},
        "first_compaction_call_p50": firsts[len(firsts) // 2] if firsts else None,
        "max_prompt_p50": sorted(r["max_prompt"] for r in rows)[len(rows) // 2],
        "timing": {t: _group([r for r in rows if r["timing"] == t]) for t in ("none", "before_edit", "after_edit")},
        "within_task": {"tasks": len(mixed),
                        "before_edit": _group([r for v in mixed for r in v if r["timing"] == "before_edit"]),
                        "rest": _group([r for v in mixed for r in v if r["timing"] != "before_edit"])},
        "by_edit_bucket": {b: {t: _group([r for r in rows if r["bucket"] == b and r["timing"] == t])
                               for t in ("none", "before_edit", "after_edit")} for b in BUCKETS},
        "rereads": {f"within_{WINDOW}_calls_after_compaction": _rate(rows, "after"), "elsewhere": _rate(rows, "other"),
                    f"from_call_{LATE}_outside_window": _rate(rows, "late")},
        "growth_fit": growth_fit([p for r in rows for p in r["pairs"]]),
        "cap_counterfactual": {
            "cap": CAP, "completion": sum(r["cap"]["completion"] for r in rows),
            "excess": sum(r["cap"]["excess"] for r in rows),
            "compacting_traces": sum(bool(r["compaction_calls"]) for r in rows),
            "first_compaction_avoided": sum(bool(r["compaction_calls"]) and not r["cap"]["still_triggers"]
                                            for r in rows)},
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
