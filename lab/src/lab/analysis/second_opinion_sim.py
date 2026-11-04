"""Rule 1 of doc 03 §14.45: offline simulation of `locate` as a second opinion for E4B (no LLM).

E4B keeps the file of its first edit when `locate` shows it among its first 3 files; otherwise it moves to `locate`'s
first file. `locate` runs on the full statement with out-of-sample habits (the folds of localization_bench), never the
deployed habits, which were mined on these tasks. Assumes perfect compliance: an upper bound.

Usage: PYTHONPATH=lab/src lab/.venv/bin/python -m lab.analysis.second_opinion_sim [--runs <run> ...] [--out ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lab.ingest.load_graphs import module_path

ROOT = Path(__file__).resolve().parents[4]
RUNS = ("data/eval/preflight-20261103-050924-bf-gov-tb-e4b", "data/eval/preflight-20261103-110441-bf-gov-tb-e4b")
EDITS = ("edit_file", "write_file")
TOP = 3


def first_edit(trace: dict) -> str | None:
    for step in trace.get("steps", []):
        for c in step.get("tool_calls") or []:
            if c.get("function_name") in EDITS:
                return (c.get("arguments") or {}).get("filepath")
    return None


def apply_policy(own: str | None, shown: list[str], gold: set[str]) -> str:
    """'none' (no edit), 'keep_right', 'keep_wrong', 'gain', 'loss', 'switch_right' (both right), 'switch_wrong'."""
    if own is None:
        return "none"
    if not shown or own in shown[:TOP]:
        return "keep_right" if own in gold else "keep_wrong"
    own_ok, new_ok = own in gold, shown[0] in gold
    if new_ok and not own_ok:
        return "gain"
    if own_ok and not new_ok:
        return "loss"
    return "switch_right" if own_ok else "switch_wrong"


def rule_1(outcomes: list[str]) -> str:
    gains, losses = outcomes.count("gain"), outcomes.count("loss")
    ok = gains - losses >= 4 and losses <= 2
    return (f"Rule 1 (simulation, decides the build): {'MET' if ok else 'NOT MET'} | gains {gains} - losses {losses} "
            f"= {gains - losses} (needs >= 4), losses {losses} (needs <= 2), n {len(outcomes)}")


def rankings(task_ids: set[str]) -> tuple[dict, dict, dict]:
    from lab.analysis.localization_bench import build_case, evaluate_case, router_cfg
    from lab.consolidate.patterns import cross_validated_results, load_deployed_params
    from lab.ingest.load_tasks import load_tasks

    cases = [c for c in (build_case(t) for t in load_tasks(ROOT / "data/competition/tasks.jsonl")) if c]
    cfg = router_cfg()
    habits = cross_validated_results(cases, load_deployed_params(), cfg)
    mine = [c for c in cases if c.task.instance_id in task_ids]
    lexical = {c.task.instance_id: evaluate_case(c, [], cfg)["shown_files"] for c in mine}
    return ({t: habits[t]["shown_files"] for t in lexical}, lexical, {c.task.instance_id: set(c.gold) for c in mine})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=list(RUNS))
    ap.add_argument("--out", type=Path, default=ROOT / "data/analysis/second_opinion_sim.md")
    a = ap.parse_args()
    edits = {}
    for run in a.runs:
        for f in sorted((ROOT / run / "traces").glob("trace_*.json")):
            fp = first_edit(json.loads(f.read_text()))
            edits[(run, f.stem.removeprefix("trace_"))] = module_path(fp) if fp else None
    habits, lexical, gold = rankings({t for _, t in edits})
    keys = [k for k in edits if k[1] in gold]
    out_h = {k: apply_policy(edits[k], habits[k[1]], gold[k[1]]) for k in keys}
    out_l = {k: apply_policy(edits[k], lexical[k[1]], gold[k[1]]) for k in keys}
    base_right = sum(edits[k] in gold[k[1]] for k in keys if edits[k])
    lines = ["# locate as a second opinion for E4B (doc 03 §14.45)", "", f"runs: {', '.join(a.runs)}",
             f"observations {len(keys)} (tasks without a gold .py file are skipped); E4B first edit in a gold file "
             f"{base_right}", "", "| ranking | gain | loss | keep_right | keep_wrong | switch_right | switch_wrong | none |",
             "|---|---|---|---|---|---|---|---|"]
    for name, o in (("habits, out of sample (decides)", out_h), ("lexical (reported)", out_l)):
        v = list(o.values())
        lines.append(f"| {name} | " + " | ".join(str(v.count(x)) for x in
                     ("gain", "loss", "keep_right", "keep_wrong", "switch_right", "switch_wrong", "none")) + " |")
    lines += ["", "## Rule", "", f"- {rule_1(list(out_h.values()))}", "", "## Switches (habits)", "",
              "| run | task | E4B file | locate top-3 | outcome |", "|---|---|---|---|---|"]
    for k in keys:
        if out_h[k] not in ("keep_right", "keep_wrong", "none"):
            lines.append(f"| {k[0].split("preflight-")[-1][:15]} | {k[1]} | {edits[k]} | {', '.join(habits[k[1]][:TOP])} | {out_h[k]} |")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
