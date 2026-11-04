"""Audit of the provided code graphs against exact per-task symbol tables (ast at each base_commit).

Counts, per repository, what the official graphs miss: async definitions, modules without definitions,
and tasks whose reference fix changes a function the graph does not contain. Graph nodes are compared by
(module, short name), the only key both sides share (the graphs flatten nested definitions); an async
definition counts as present only if no sync definition with the same key exists (otherwise the match is a
name collision). Gold functions are counted per (file, qualname).

Usage: PYTHONPATH=lab/src python3 -m lab.analysis.graph_coverage
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from lab.analysis.localization_bench import GRAPHS, gold_files
from lab.ingest.load_graphs import load_graph, module_path
from lab.ingest.load_symbols import gold_symbols_by_lines, load_symbol_table
from lab.ingest.load_tasks import load_tasks, parse_patch

ROOT = Path(__file__).resolve().parents[4]
COLUMNS = ["tasks", "defs_exact", "defs_graph", "async_exact", "async_in_graph", "defless_modules",
           "defless_in_graph", "async_in_edited_files", "tasks_gold_fn", "tasks_gold_fn_async",
           "gold_fns", "gold_fns_async"]
PER_TASK = ("defs_exact", "defs_graph", "async_exact", "defless_modules")


def audit_task(task, graphs: Path) -> dict | None:
    table = load_symbol_table(task.instance_id)
    if table is None or table.status != "exact":
        return None
    nodes, _ = load_graph(graphs, task)
    graph_keys = {(n.module, n.short) for n in nodes}
    graph_modules = {n.module for n in nodes}

    def key(r):
        return module_path(r.path), r.qualname.rsplit(".", 1)[-1]

    by_file = defaultdict(list)
    for r in table.rows:
        by_file[r.path].append(r)
    sync_keys = {key(r) for r in table.rows if not r.is_async}
    async_only = {key(r) for r in table.rows if r.is_async} - sync_keys
    defless = {module_path(f) for f in table.files if not by_file.get(f)}

    edited = set(gold_files(task.patch))
    gold, gold_async = set(), set()
    for change in parse_patch(task.patch):
        if change.path not in edited:
            continue
        for q in gold_symbols_by_lines(table, change):
            row = next(r for r in by_file[change.path] if r.qualname == q)
            gold.add((change.path, q))
            if row.is_async:
                gold_async.add((change.path, q))

    return {
        "tasks": 1,
        "defs_exact": len(table.rows),
        "defs_graph": len(nodes),
        "async_exact": sum(r.is_async for r in table.rows),
        "async_in_graph": len(async_only & graph_keys),
        "defless_modules": len(defless),
        "defless_in_graph": len(defless & graph_modules),
        "async_in_edited_files": sum(r.is_async for f in edited for r in by_file.get(f, [])),
        "tasks_gold_fn": int(bool(gold)),
        "tasks_gold_fn_async": int(bool(gold_async)),
        "gold_fns": len(gold),
        "gold_fns_async": len(gold_async),
    }


def render(per_repo: dict[str, dict]) -> str:
    total = {c: sum(r[c] for r in per_repo.values()) for c in COLUMNS}
    rows = {**per_repo, "total": total}
    head = ["repo", "tasks", "defs/task (ast)", "defs/task (graph)", "async/task (ast)", "async in graph",
            "def-less modules/task", "def-less in graph", "async defs in edited files", "tasks with gold fn",
            "tasks with async gold fn", "gold fns", "async gold fns"]
    out = ["# Official code graphs vs exact symbol tables", "",
           "Tasks with an exact table (128). Per-task columns are averages; the rest are sums over tasks.", "",
           "| " + " | ".join(head) + " |", "|---" * len(head) + "|"]
    for repo, r in rows.items():
        n = r["tasks"]
        cells = [repo, str(n)] + [f"{r[c] / n:.0f}" if c in PER_TASK else str(r[c]) for c in COLUMNS[1:]]
        out.append("| " + " | ".join(cells) + " |")
    t = total
    out += ["", "## Summary",
            f"- Async definitions present in the graphs: {t['async_in_graph']} "
            f"(of {t['async_exact'] / t['tasks']:.0f} async definitions per task on average).",
            f"- Modules without definitions present in the graphs: {t['defless_in_graph']} of {t['defless_modules']}.",
            f"- Async definitions inside the files the reference fixes edit: {t['async_in_edited_files']}.",
            f"- Gold functions: {t['gold_fns']} in {t['tasks_gold_fn']} tasks; {t['gold_fns_async']} are async "
            f"(absent from the graphs), in {t['tasks_gold_fn_async']} tasks."]
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=ROOT / "data/competition/tasks.jsonl", type=Path)
    ap.add_argument("--out", default=ROOT / "data/analysis", type=Path)
    args = ap.parse_args()

    per_repo: dict[str, dict] = {}
    for task in load_tasks(args.tasks):
        res = audit_task(task, GRAPHS)
        if res is None:
            continue
        acc = per_repo.setdefault(task.repo_short, {c: 0 for c in COLUMNS})
        for c in COLUMNS:
            acc[c] += res[c]
    report = render(dict(sorted(per_repo.items())))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "graph_coverage.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
