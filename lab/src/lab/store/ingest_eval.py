"""Ingest a `swegemma eval` results directory into LINALDB as verified agent experience.

Each task becomes a `task_result` row; files and symbols the agent's patch touched become `experience` rows
with outcome resolved/unresolved (verified by the harness's Container B); each habit `locate` reported
active becomes a `habit_activation` row, which lets consolidation weigh habits by real outcomes.

Usage: PYTHONPATH=lab/src lab/.venv/bin/python -m lab.store.ingest_eval --results <dir> [--label name] [--submission-sha sha]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from lab.eval.verifiable import load_verifiable
from lab.ingest.load_graphs import module_path
from lab.ingest.load_symbols import gold_symbols_by_lines, load_symbol_table
from lab.ingest.load_tasks import parse_patch
from lab.ingest.load_traces import TaskRun, load_results
from lab.store.experience_store import ExperienceStore

SUBMISSION_REFS = Path(__file__).resolve().parents[4] / "submission/skills/mini-palace/references"


def run_meta(results_dir: Path) -> dict:
    """run_meta.json written by lab.eval.run_eval; missing means an external results dir (assumed faithful)."""
    path = results_dir / "run_meta.json"
    return json.loads(path.read_text()) if path.exists() else {"scorer_faithful": True}


def eval_run_id(results_dir: Path, label: str) -> str:
    digest = hashlib.sha1((results_dir / "task_results.jsonl").read_bytes()).hexdigest()[:10]
    return f"eval-{label}-{digest}"


def agent_experience(run: TaskRun) -> list[tuple[str, str | None]]:
    """(module file, symbol or None) touched by the agent's patch; symbols only when an exact table exists."""
    table = load_symbol_table(run.instance_id)
    out = []
    for change in parse_patch(run.patch):
        if not change.path.endswith(".py"):
            continue
        out.append((module_path(change.path), None))
        if table is not None and table.status == "exact" and not change.is_new:
            out.extend((module_path(change.path), q) for q in sorted(gold_symbols_by_lines(table, change)))
    return out


RUN_TABLES = ("run", "task_result", "experience", "habit_activation")


def ingest(store: ExperienceStore, results_dir: Path, label: str, submission_sha: str = "",
           verifiable: dict[str, bool] | None = None, replace: bool = False) -> dict:
    """verifiable: when given, outcomes of tasks outside it (or failing the reference check) are stored as
    'unverified': our verification is not faithful to the scorer there, so they are not evidence.
    replace: re-ingest a run whose verification changed (the run id only hashes the agent's own results)."""
    run_id = eval_run_id(results_dir, label)
    if store.has_run(run_id):
        if not replace:
            return {"run_id": run_id, "skipped": True}
        for table in RUN_TABLES:
            store.delete_run(table, run_id)
    runs = load_results(results_dir)
    n_unverified = 0
    for r in runs:
        outcome = "resolved" if r.resolved else "unresolved"
        if verifiable is not None and not verifiable.get(r.instance_id, False):
            outcome = "unverified"
            n_unverified += 1
        store.insert("task_result", {"run_id": run_id, "task_id": r.instance_id, "resolved": r.resolved,
                                     "error": (r.error or None) and r.error[:200], "duration_s": r.duration_s,
                                     "tool_calls": r.tool_calls, "patch_size": r.patch_size,
                                     "skill_calls": len(r.skill_calls), "skill_first": r.skill_first,
                                     "n_habits": len(r.habits)})
        for file, symbol in agent_experience(r):
            store.insert("experience", {"run_id": run_id, "task_id": r.instance_id, "file": file,
                                        "symbol": symbol, "outcome": outcome})
        for pid in r.habits:
            store.insert("habit_activation", {"run_id": run_id, "task_id": r.instance_id, "pattern_id": pid,
                                              "resolved": r.resolved})
    habits_version = ""
    if (SUBMISSION_REFS / "consolidation.json").exists():
        habits_version = f"habits v{json.loads((SUBMISSION_REFS / 'consolidation.json').read_text())['version']}"
    meta = run_meta(results_dir)
    # Only runs with the scorer's model count as verified agent experience; anything else (a small local
    # model testing the pipeline) is stored but kept out of consolidation.
    kind = "agent_eval" if meta.get("scorer_faithful") else "pipeline_test"
    note = (f"label={label} model={meta.get('served_model', '?')} hardware={meta.get('hardware', '?')} "
            f"sandbox={meta.get('sandbox', '?')} submission={meta.get('submission_digest') or submission_sha or '?'} "
            f"{habits_version}").strip()
    store.insert("run", {"run_id": run_id, "kind": kind,
                         "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                         "source": str(results_dir), "note": note})
    store.save()
    return {"run_id": run_id, "runs": runs, "kind": kind, "unverified": n_unverified}


def report(runs: list[TaskRun]) -> str:
    n = len(runs)
    lines = [f"tasks {n} | resolved {sum(r.resolved for r in runs)} ({100 * sum(r.resolved for r in runs) / n:.0f}%)"
             f" | with trace {sum(r.has_trace for r in runs)} | errors {sum(bool(r.error) for r in runs)}"]
    groups = {"skill used": [r for r in runs if r.skill_calls], "skill not used": [r for r in runs if not r.skill_calls],
              "skill first": [r for r in runs if r.skill_first]}
    for name, g in groups.items():
        if g:
            lines.append(f"  {name:15} {len(g):4} tasks | resolved {sum(r.resolved for r in g):3}"
                         f" ({100 * sum(r.resolved for r in g) / len(g):.0f}%)")
    actions = defaultdict(int)
    for r in runs:
        for c in r.skill_calls:
            actions[c.action] += 1
    if actions:
        lines.append("  skill actions: " + ", ".join(f"{a}={k}" for a, k in sorted(actions.items())))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, type=Path)
    ap.add_argument("--label", default="local")
    ap.add_argument("--submission-sha", default="")
    ap.add_argument("--replace", action="store_true", help="replace an already ingested run (re-verified)")
    args = ap.parse_args()
    out = ingest(ExperienceStore(), args.results, args.label, args.submission_sha, load_verifiable(), args.replace)
    if out.get("skipped"):
        print(f"{out['run_id']} already ingested")
        return
    print(f"{out['run_id']} ingested as {out['kind']} ({out['unverified']} task outcomes stored as unverified)\n"
          f"{report(out['runs'])}")


if __name__ == "__main__":
    main()
