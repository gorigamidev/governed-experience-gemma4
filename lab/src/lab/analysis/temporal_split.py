"""Localization with habits mined only from earlier tasks (expanding window), against the shuffled 5-fold CV.

The shuffled folds let a later fix inform an earlier task. Here each task is ranked with habits mined from tasks
created strictly before it, with the deployed parameters held fixed (no tuning on the evaluated data). Tasks with
no earlier task in their repository get no habit, so they fall back to the lexical ranking; results are reported
both over all tasks and over tasks that have history.

Usage: PYTHONPATH=lab/src lab/.venv/bin/python -m lab.analysis.temporal_split
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from lab.analysis.localization_bench import HEADER, evaluate_case, fmt_row, router_cfg, summarize
from lab.consolidate.patterns import _context, cv_results, load_deployed_params, mine

ROOT = Path(__file__).resolve().parents[4]


def earlier(created: dict[str, str], task_id: str) -> list[str]:
    return [t for t, when in created.items() if when < created[task_id]]


def temporal_results(cases, exp, terms, params, cfg) -> dict[str, dict]:
    created = {c.task.instance_id: c.task.created_at for c in cases}
    out, cache = {}, {}
    for c in cases:
        train = tuple(sorted(earlier(created, c.task.instance_id)))
        if train not in cache:
            cache[train] = mine(list(train), exp, terms, params["min_support"], params["min_lift"], params["kinds"],
                                params.get("min_support_symbol"))
        out[c.task.instance_id] = evaluate_case(c, cache[train], {**cfg, "pattern_weight": params["pattern_weight"]})
    return out


def paired(a: dict, b: dict, ids) -> tuple[int, int]:
    """Tasks where b ranks the gold file better / worse than a (missing rank counts as worst)."""
    def rank(r):
        return r["rank"] or 10**6
    return (sum(rank(b[i]) < rank(a[i]) for i in ids), sum(rank(b[i]) > rank(a[i]) for i in ids))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=ROOT / "data/competition/tasks.jsonl", type=Path)
    ap.add_argument("--out", default=ROOT / "data/analysis/temporal_split.md", type=Path)
    args = ap.parse_args()

    cases, exp, terms = _context(args.tasks)
    params, cfg = load_deployed_params(), router_cfg()
    lexical = {c.task.instance_id: evaluate_case(c, [], cfg) for c in cases}
    shuffled = cv_results(cases, exp, terms, params, cfg)
    temporal = temporal_results(cases, exp, terms, params, cfg)

    created = {c.task.instance_id: c.task.created_at for c in cases}
    repo = {c.task.instance_id: c.task.repo_short for c in cases}
    history = [i for i in created if any(repo[j] == repo[i] for j in earlier(created, i))]
    subsets = {"all tasks": list(created), "tasks with earlier tasks in their repo": history}

    lines = [f"# Temporal split ({len(cases)} tasks)", "",
             f"Deployed parameters, fixed: {params}. Habits for each task are mined only from tasks created strictly "
             "before it; the shuffled row is the 5-fold CV used in Table 3 of the paper.", ""]
    for name, ids in subsets.items():
        lines += [f"## {name}", HEADER]
        for label, res in (("lexical only", lexical), ("habits, shuffled 5-fold CV", shuffled),
                           ("habits, earlier tasks only", temporal)):
            lines.append(fmt_row(label, summarize([res[i] for i in ids])))
        b, w = paired(shuffled, temporal, ids)
        bl, wl = paired(lexical, temporal, ids)
        lines += ["", f"Temporal vs shuffled: better {b}, worse {w}. Temporal vs lexical: better {bl}, worse {wl}.", ""]
    lines += ["## Tasks with history, by repository", "",
              "| repo | tasks with history | of all |", "|---|---|---|"]
    all_by_repo, hist_by_repo = Counter(repo.values()), Counter(repo[i] for i in history)
    lines += [f"| {r} | {hist_by_repo[r]} | {all_by_repo[r]} |" for r in sorted(all_by_repo)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
