"""Localization benchmark (no LLM) on the exact symbols of each task's base_commit.

Evaluates the runtime `locate_core` itself: file rank of the reference patch's files, and whether a
gold symbol is among the symbols `locate` shows. Habits are mined per fold (lab.consolidate), so the
"habits" rows are out-of-sample.

Usage: PYTHONPATH=lab/src python3 -m lab.analysis.localization_bench
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from lab.analysis.issue_types import profile_statement
from lab.ingest.load_graphs import gold_symbols, load_graph, module_path
from lab.ingest.load_symbols import gold_symbols_by_lines, load_symbol_table
from lab.ingest.load_tasks import Task, load_tasks, parse_patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "submission/skills/mini-palace/scripts"))
from indexer import Symbol, is_test_path  # noqa: E402
from locator import locate_core  # noqa: E402
from resolver import LexicalIndex  # noqa: E402
from router import DEFAULTS  # noqa: E402

GRAPHS = ROOT / "data/competition/graphs"
CFG_PATH = ROOT / "submission/skills/mini-palace/references/router_config.json"
KS = (1, 3, 5, 10, 20)


def router_cfg() -> dict:
    return {**DEFAULTS, **(json.loads(CFG_PATH.read_text()) if CFG_PATH.exists() else {})}


def gold_files(patch: str) -> list[str]:
    return [c.path for c in parse_patch(patch)
            if c.path.endswith(".py") and not c.is_new and not is_test_path(c.path)]


@dataclass
class Case:
    task: Task
    idx: LexicalIndex
    gold: list[str]  # module keys
    gold_syms: set[str] = field(default_factory=set)  # "module::local_qualname"
    source: str = "graph"  # symbols (exact, rebuilt) | graph (no async, no def-less modules)


def build_case(task: Task, graphs: Path = GRAPHS) -> Case | None:
    """Index of the task's base_commit: exact symbol table when available, precomputed graph otherwise.

    The graphs omit async functions and def-less modules; the rebuilt tables (lab.ingest.reconstruct_symbols)
    do not, and use the same qualnames as the runtime indexer.
    """
    gold_paths = gold_files(task.patch)
    if not gold_paths:
        return None
    table = load_symbol_table(task.instance_id)
    if table is not None and table.status == "exact":
        code_files = [f for f in table.files if not is_test_path(f)]
        idx = LexicalIndex(sorted({module_path(f) for f in code_files}),
                           [Symbol(r.qualname, module_path(r.path), r.start, r.kind)
                            for r in table.rows if not is_test_path(r.path)])
        syms = set()
        for c in parse_patch(task.patch):
            if c.path in gold_paths:
                syms |= {f"{module_path(c.path)}::{q}" for q in gold_symbols_by_lines(table, c)}
        return Case(task, idx, [module_path(p) for p in gold_paths], syms, "symbols")
    nodes, _ = load_graph(graphs, task)
    code = [n for n in nodes if not is_test_path(n.module)]
    idx = LexicalIndex(sorted({n.module for n in code}), [Symbol(n.local_name, n.module, 0, "node") for n in code])
    local = {n.name: n.local_name for n in nodes}
    syms = set()
    for c in parse_patch(task.patch):
        if c.path in gold_paths:
            syms |= {f"{module_path(c.path)}::{local[s]}" for s in gold_symbols(nodes, c)}
    return Case(task, idx, [module_path(p) for p in gold_paths], syms, "graph")


def evaluate_case(case: Case, pattern_list: list[dict], cfg: dict) -> dict:
    decision, shown, _, ranked, active = locate_core(case.idx, case.task.problem_statement, case.task.repo_short,
                                                      pattern_list, cfg)
    order = [r.file for r in ranked]
    ranks = [order.index(g) + 1 for g in case.gold if g in order]
    shown_syms = {f"{r.file}::{q}" for r in ranked[:5] for q in r.top_symbols}
    return {"rank": min(ranks) if ranks else None, "decision": decision,
            "shown_files": [r.file for r in shown], "symbol_shown": bool(case.gold_syms & shown_syms),
            "has_gold_syms": bool(case.gold_syms), "n_active": len(active)}


def summarize(results: list[dict]) -> dict:
    n = len(results)
    out = {"n": n, "mrr": sum(1 / r["rank"] for r in results if r["rank"]) / n}
    for k in KS:
        out[f"r@{k}"] = sum(1 for r in results if r["rank"] and r["rank"] <= k) / n
    with_syms = [r for r in results if r["has_gold_syms"]]
    out["sym_shown"] = sum(r["symbol_shown"] for r in with_syms) / len(with_syms) if with_syms else 0.0
    return out


def fmt_row(label: str, m: dict) -> str:
    return (f"| {label} | {m['n']} | " + " | ".join(f"{100 * m[f'r@{k}']:.0f}%" for k in KS)
            + f" | {m['mrr']:.3f} | {100 * m['sym_shown']:.0f}% |")


HEADER = ("| ranker | n | " + " | ".join(f"R@{k}" for k in KS) + " | MRR | symbol shown |\n|---|---|"
          + "---|" * (len(KS) + 2))


def main() -> None:
    from lab.consolidate.patterns import cross_validated_results, load_deployed_params  # avoid import cycle

    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=ROOT / "data/competition/tasks.jsonl", type=Path)
    ap.add_argument("--out", default=ROOT / "data/analysis", type=Path)
    args = ap.parse_args()

    cases = [c for c in (build_case(t) for t in load_tasks(args.tasks)) if c]
    cfg = router_cfg()
    lexical = {c.task.instance_id: evaluate_case(c, [], cfg) for c in cases}
    habits = cross_validated_results(cases, load_deployed_params(), cfg)

    rows = []
    for c in cases:
        lx, hb = lexical[c.task.instance_id], habits[c.task.instance_id]
        rows.append({"instance_id": c.task.instance_id, "repo": c.task.repo_short,
                     "category": profile_statement(c.task.problem_statement).category,
                     "gold": ";".join(c.gold), "gold_symbols": ";".join(sorted(c.gold_syms)),
                     "rank_lexical": lx["rank"], "rank_habits": hb["rank"],
                     "symbol_shown_lexical": lx["symbol_shown"], "symbol_shown_habits": hb["symbol_shown"],
                     "decision": hb["decision"], "n_active_habits": hb["n_active"]})

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "localization_bench.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    by_source = {s: sum(c.source == s for c in cases) for s in ("symbols", "graph")}
    lines = [f"# Localization benchmark ({len(cases)} tasks at base_commit; index source: {by_source})", "",
             "habits = consolidated patterns mined on the other folds (5-fold CV, out-of-sample).", "",
             "## All tasks", HEADER,
             fmt_row("lexical only", summarize(list(lexical.values()))),
             fmt_row("lexical + habits", summarize(list(habits.values())))]
    for group in ("repo", "category"):
        lines += ["", f"## lexical + habits by {group}", HEADER.replace("ranker", group)]
        groups = defaultdict(list)
        for r, c in zip(rows, cases):
            groups[r[group]].append(habits[c.task.instance_id])
        for v in sorted(groups):
            lines.append(fmt_row(v, summarize(groups[v])))
    report = "\n".join(lines) + "\n"
    (args.out / "localization_bench.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
