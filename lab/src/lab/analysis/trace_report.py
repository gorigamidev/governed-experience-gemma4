"""Report on an agent run: does the agent use the skill, does locate surface the gold file, does the patch touch
it, and how outcomes relate to skill use and active habits. Outcomes are split by verifiability (a task whose
reference patch fails in our environment cannot be judged by our verification).

Usage: PYTHONPATH=lab/src python3 -m lab.analysis.trace_report --results data/eval/<run>
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

from lab.analysis.localization_bench import gold_files
from lab.eval.verifiable import load_verifiable
from lab.ingest.load_graphs import module_path
from lab.ingest.load_tasks import load_tasks
from lab.ingest.load_traces import TaskRun, load_results

ROOT = Path(__file__).resolve().parents[4]
# ADK raises when an agent calls a tool it does not declare; the harness then drops the task and its fallback diff.
UNDECLARED_TOOL = re.compile(r"Tool '[^']+' not found")


def task_row(run: TaskRun, gold: list[str], verifiable: dict[str, bool] | None,
             final_broken: dict[str, bool] | None = None) -> dict:
    gold_mod = {module_path(g) for g in gold}
    located = [f for c in run.skill_calls if c.action == "locate" for f in c.files]
    rank = next((i + 1 for i, f in enumerate(located) if module_path(f) in gold_mod), None)
    return {
        "task": run.instance_id,
        "resolved": run.resolved,
        "verifiable": None if verifiable is None else verifiable.get(run.instance_id, False),
        "skill_used": bool(run.skill_calls),
        "skill_first": run.skill_first,
        "first_tool": run.first_tool or "-",
        "locate_gold_rank": rank,
        "patch_touches_gold": bool({module_path(p) for p in run.patch_files} & gold_mod),
        "has_patch": run.patch_size > 0,
        "tool_calls": run.tool_calls,
        "minutes": round(run.duration_s / 60, 1),
        "habits": len(run.habits),
        "llm_calls": run.cost.llm_calls if run.cost else None,
        "gen_tokens": run.cost.completion_tokens if run.cost else None,
        "thinking_kchars": round(run.cost.thinking_chars / 1000, 1) if run.cost else None,
        "first_edit_s": round(run.cost.first_edit_s) if run.cost and run.cost.first_edit_s is not None else None,
        "failed_edits": run.cost.failed_edits if run.cost else None,
        "undeclared_tool": bool(UNDECLARED_TOOL.search(run.error or "")),
        "check_calls": sum(c.action == "check" for c in run.skill_calls),
        "check_broken": any(c.verdict == "broken" for c in run.skill_calls),
        "check_no_changes": any(c.verdict == "no_changes" for c in run.skill_calls),
        "submitted": run.tools_used.get("submit_patch", 0) > 0,
        # Habit G1's instrument replayed offline on the final patch (lab.eval.replay_check --results).
        "final_broken": None if final_broken is None else final_broken.get(run.instance_id),
        "error": (run.error or "")[:60],
    }


def rate(rows: list[dict], key: str) -> str:
    return f"{sum(bool(r[key]) for r in rows)}/{len(rows)}" if rows else "0/0"


def median(rows: list[dict], key: str) -> str:
    values = [r[key] for r in rows if r[key] is not None]
    return f"{statistics.median(values):g} (n={len(values)})" if values else "-"


def render(rows: list[dict], meta: dict) -> str:
    judged = [r for r in rows if r["verifiable"] is not False]
    out = [f"# Trace report: {meta.get('label', '?')}", "",
           f"Model `{meta.get('served_model', '?')}` on {meta.get('hardware', '?')}, sandbox {meta.get('sandbox', '?')}, "
           f"submission `{meta.get('submission_digest', '?')}`, {len(rows)} tasks.", "",
           "## Behavior (all tasks)",
           f"- Skill used: {rate(rows, 'skill_used')}; `locate` as first action: {rate(rows, 'skill_first')}",
           f"- `locate` listed a gold file: {sum(r['locate_gold_rank'] is not None for r in rows)}/{len(rows)}; "
           f"at rank 1: {sum(r['locate_gold_rank'] == 1 for r in rows)}",
           f"- Produced a patch: {rate(rows, 'has_patch')}; patch touches a gold file: {rate(rows, 'patch_touches_gold')}",
           f"- Patch touches gold when `locate` listed gold: "
           f"{rate([r for r in rows if r['locate_gold_rank'] is not None], 'patch_touches_gold')}",
           f"- Errors or timeouts: {sum(bool(r['error']) for r in rows)}; tasks lost to an undeclared tool: "
           f"{sum(r['undeclared_tool'] for r in rows)}", "",
           "## Deliberation cost (medians per task)",
           f"- Seconds to first successful edit: {median(rows, 'first_edit_s')}; tasks with an edit: "
           f"{sum(r['first_edit_s'] is not None for r in rows)}/{len(rows)}",
           f"- Generated tokens: {median(rows, 'gen_tokens')}; thinking kchars: {median(rows, 'thinking_kchars')}; "
           f"model calls: {median(rows, 'llm_calls')}; failed edits (total): "
           f"{sum(r['failed_edits'] or 0 for r in rows)}", "",
           "## Outcomes",
           f"- Verifiable tasks (reference patch passes locally): {len(judged)} of {len(rows)}",
           f"- Resolved among verifiable: {rate(judged, 'resolved')}"]
    for name, sub in (("skill first", [r for r in judged if r["skill_first"]]),
                      ("skill not first", [r for r in judged if not r["skill_first"]]),
                      ("locate listed gold", [r for r in judged if r["locate_gold_rank"]]),
                      ("locate missed gold", [r for r in judged if r["skill_used"] and not r["locate_gold_rank"]])):
        out.append(f"  - {name}: resolved {rate(sub, 'resolved')}")
    patched = [r for r in rows if r["has_patch"]]
    if any(r["final_broken"] is not None for r in rows) or any(r["check_calls"] for r in rows):
        seen = [r for r in rows if r["check_broken"]]
        out += ["", "## Habit G1 (check)",
                f"- B, final patch `broken` when check is replayed offline: "
                f"{sum(bool(r['final_broken']) for r in patched)}/{len(patched)} tasks with a patch",
                f"- K, tasks with a patch that called check: "
                f"{sum(r['check_calls'] > 0 for r in patched)}/{len(patched)}",
                f"- Rescue, check said `broken` during the run and the final patch is not broken: "
                f"{sum(r['final_broken'] is False for r in seen)}/{len(seen)}",
                # Submission 5: a coder read an empty check as a green light and submitted an empty patch.
                f"- E, check said `no_changes` and the model then submitted an empty patch: "
                f"{sum(r['submitted'] and not r['has_patch'] for r in rows if r['check_no_changes'])}/"
                f"{sum(r['check_no_changes'] for r in rows)}; tasks without a patch: {len(rows) - len(patched)}"]
    cols = list(rows[0]) if rows else []
    out += ["", "## Per task", "| " + " | ".join(cols) + " |", "|---" * len(cols) + "|"]
    out += ["| " + " | ".join("" if r[c] is None else str(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, type=Path)
    ap.add_argument("--tasks", default=ROOT / "data/competition/tasks.jsonl", type=Path)
    args = ap.parse_args()
    meta_path = args.results / "run_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    verifiable = load_verifiable()
    tasks = {t.instance_id: t for t in load_tasks(args.tasks)}
    replay = args.results / "check_replay.jsonl"
    final_broken = ({(r := json.loads(ln))["task"]: r.get("verdict") == "broken" for ln in replay.read_text().splitlines()}
                    if replay.exists() else None)
    rows = [task_row(r, gold_files(tasks[r.instance_id].patch), verifiable, final_broken)
            for r in load_results(args.results)]
    report = render(rows, meta)
    # One report per processed run: several runs share the notebook label.
    out = ROOT / "data/analysis" / f"trace_report_{args.results.name}.md"
    out.write_text(report)
    print(report)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
