"""Bootstrap the experience store with the reference patches as expert demonstrations.

These are not verified agent experience (resolved=True from Container B); they are tagged
kind=reference_patch / outcome=demonstration so consolidation and the paper can tell them apart.

Usage: PYTHONPATH=lab/src lab/.venv/bin/python -m lab.store.ingest_reference
"""

from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path

from lab.analysis.issue_types import profile_statement
from lab.analysis.localization_bench import gold_files
from lab.ingest.load_graphs import gold_symbols, load_graph, module_path
from lab.ingest.load_symbols import gold_symbols_by_lines, load_symbol_table
from lab.ingest.load_tasks import load_tasks, parse_patch
from lab.store.experience_store import ROOT, ExperienceStore


# Bump when the way gold symbols are derived changes; older runs are marked superseded, never deleted,
# so consolidations that used them keep their lineage.
DERIVATION = "ast-lines"


def reference_run_id(tasks_path: Path) -> str:
    return f"ref-{hashlib.sha1(tasks_path.read_bytes()).hexdigest()[:10]}-{DERIVATION}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=ROOT / "data/competition/tasks.jsonl", type=Path)
    ap.add_argument("--graphs", default=ROOT / "data/competition/graphs", type=Path)
    args = ap.parse_args()

    store = ExperienceStore()
    run_id = reference_run_id(args.tasks)
    if store.has_run(run_id):
        print(f"{run_id} already ingested")
        return

    for old in store.query('SELECT run_id FROM run WHERE kind = "reference_patch"'):
        store.execute(f'UPDATE run SET kind = "superseded" WHERE run_id = "{old["run_id"]}"')
        print(f"{old['run_id']} marked superseded")

    tasks = load_tasks(args.tasks)
    known = {r["task_id"] for r in store.query("SELECT task_id FROM task")}
    n_file = n_sym = 0
    for t in tasks:
        if t.instance_id not in known:
            p = profile_statement(t.problem_statement)
            store.insert("task", {"task_id": t.instance_id, "repo": t.repo_short, "base_commit": t.base_commit,
                                  "category": p.category, "change_kind": p.change_kind})
        gold = set(gold_files(t.patch))
        table = load_symbol_table(t.instance_id)
        exact = table is not None and table.status == "exact"
        if not exact:
            nodes, _ = load_graph(args.graphs, t)
            local = {n.name: n.local_name for n in nodes}
        for change in parse_patch(t.patch):
            if change.path not in gold:
                continue
            module = module_path(change.path)
            store.insert("experience", {"run_id": run_id, "task_id": t.instance_id, "file": module,
                                        "symbol": None, "outcome": "demonstration"})
            n_file += 1
            # Exact tables see async code; the graph fallback does not.
            names = gold_symbols_by_lines(table, change) if exact else {local[n] for n in gold_symbols(nodes, change)}
            for name in sorted(names):
                store.insert("experience", {"run_id": run_id, "task_id": t.instance_id, "file": module,
                                            "symbol": name, "outcome": "demonstration"})
                n_sym += 1

    store.insert("run", {"run_id": run_id, "kind": "reference_patch",
                         "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                         "source": str(args.tasks.relative_to(ROOT)),
                         "note": f"expert demonstrations, not agent-verified; gold symbols: {DERIVATION}"})
    store.save()
    print(f"{run_id}: {len(tasks)} tasks, {n_file} file experiences, {n_sym} symbol experiences")


if __name__ == "__main__":
    main()
