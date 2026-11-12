"""Where the bf family loses tasks: every GPU pool run on the same 24 tasks (doc 03 §14 item 48).

Each losing run falls in one bucket, in this order: no patch, patch outside the gold files, patch that touches the
gold files but misses one, patch on all gold files. Tasks are split by how often they resolve across runs.
These pools have no Docker pass, so `resolved` is the harness verdict in the Kaggle subprocess sandbox, which fails
reference patches on many fastapi and requests tasks. Only the tasks whose reference patch passes (`verifiable`) can
be read as agent outcomes; the rest are listed apart and must not be read as lost by the agent."""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

from lab.analysis.pool_report import ROOT, _jsonl, gold_files, task_row
from lab.analysis import pool_report

SESSIONS = {  # session -> arms; same 24 tasks and stack, the base arm of each pool repeats across sessions
    "pool-gov-gpu": ("bf", "bf-gov"),
    "pool-to-gpu": ("bf-gov", "bf-gov-to"),
    "pool-tb-gpu": ("bf-gov-to", "bf-gov-tb"),
    "pool-tb22-gpu": ("bf-gov-tb", "bf-gov-tb-22"),
    "pool-tbslec-gpu": ("bf-gov-tb", "bf-gov-tb-sl", "bf-gov-tb-ec"),
}


def load_runs(root: Path) -> list[dict]:
    verifiable = pool_report.load_verifiable() or {}
    gold = gold_files()
    runs = []
    for session, arms in SESSIONS.items():
        for arm in arms:
            p = root / session / "results" / arm
            harness = _jsonl(p / "task_results.jsonl")
            docker = _jsonl(p / "verified_task_results.jsonl")
            for task, h in harness.items():
                row = task_row(task, h, docker, verifiable, p)
                patch = p / "patches" / f"{task}.patch"
                files = set(re.findall(r"^diff --git a/(\S+)", patch.read_text(), re.M)) if patch.exists() else set()
                runs.append({**row, "session": session, "arm": arm, "files": files, "gold": gold.get(task, set())})
    return runs


def bucket(r: dict) -> str:
    if r["resolved"]:
        return "resolved"
    src = {f for f in r["files"] if not re.search(r"(^|/)tests?/|test_", f)}
    if not r["files"]:
        return "no_patch"
    if not r["gold"] & r["files"]:
        return "outside_gold" if src else "tests_only"
    return "all_gold" if r["gold"] <= r["files"] else "part_gold"


def median(xs: list) -> float | None:
    return st.median(xs) if xs else None


def refused_summary(runs: list[dict]) -> list[str]:
    """Runs with no patch on verifiable tasks: did they run out of budget right after finding the file?"""
    np = [r for r in runs if r["verifiable"] and bucket(r) == "no_patch"]
    refused = [r for r in np if r["edits_after_budget"]]
    timed_out = [r for r in np if "session timeout" in r["error"]]
    left = median([round(8 - r["minutes"], 1) for r in refused])
    return ["## No patch on verifiable tasks", "",
            f"{len(np)} runs: {len(refused)} tried an edit refused by the call budget (median {left} min of the 8 "
            f"left, tasks {dict(Counter(r['task'] for r in refused))}); {len(timed_out)} hit the session timeout "
            f"(arms {dict(Counter(r['arm'] for r in timed_out))}).", ""]


def arm_noise(runs: list[dict]) -> list[str]:
    """Is the spread of R across arm-sessions larger than fixed per-task resolve rates alone would give?"""
    cells, by_task = defaultdict(int), defaultdict(list)
    for r in runs:
        if r["verifiable"]:
            cells[(r["session"], r["arm"])] += r["resolved"]
            by_task[r["task"]].append(r["resolved"])
    if len(cells) < 2:
        return []
    ps = [sum(v) / len(v) for v in by_task.values()]
    per_arm = defaultdict(list)
    for (s, a), w in cells.items():
        per_arm[a].append(w)
    out = ["## Arm effect against session noise", "",
           f"R on the {len(by_task)} verifiable tasks over {len(cells)} arm-sessions: mean {st.mean(cells.values()):.2f}, "
           f"sd {st.stdev(cells.values()):.2f}, range {min(cells.values())}-{max(cells.values())}. "
           f"With every arm sharing the same per-task resolve rates the expected sd is "
           f"{sum(p * (1 - p) for p in ps) ** 0.5:.2f}.", "", "| arm | R per session |", "|---|---|"]
    out += [f"| {a} | {', '.join(map(str, v))} |" for a, v in sorted(per_arm.items())]
    return out + [""]


def report(runs: list[dict]) -> str:
    by_task = defaultdict(list)
    for r in runs:
        by_task[r["task"]].append(r)
    n = len(runs)
    kinds = Counter(bucket(r) for r in runs if r["verifiable"])
    n_ver = sum(r["verifiable"] for r in runs)
    out = ["# Near-miss review of the bf family (doc 03 §14 item 48)", "",
           f"{n} runs, {len(by_task)} tasks, {len(SESSIONS)} GPU sessions, arms " + ", ".join(
               sorted({r['arm'] for r in runs})), "", "## Outcome of the runs on verifiable tasks", ""]
    out += [f"- {k}: {v} ({100 * v / n_ver:.0f}%)" for k, v in kinds.most_common()]
    out += [""] + arm_noise(runs) + refused_summary(runs)
    out += ["", "## Per task", "", "| task | verifiable | resolved | outcomes | gold files | median first edit | median calls | median min |",
            "|---|---|---|---|---|---|---|---|"]
    tiers = {"always": 0, "flips": 0, "never": 0}
    for task, rs in sorted(by_task.items(), key=lambda kv: (not kv[1][0]["verifiable"],
                                                            -sum(r["resolved"] for r in kv[1]))):
        w = sum(r["resolved"] for r in rs)
        if rs[0]["verifiable"]:
            tiers["always" if w == len(rs) else "never" if w == 0 else "flips"] += 1
        c = Counter(bucket(r) for r in rs if not r["resolved"])
        out.append(f"| {task} | {'yes' if rs[0]['verifiable'] else 'no'} | {w}/{len(rs)} | {dict(c)} | {len(rs[0]['gold'])} | "
                   f"{median([r['first_edit_call'] for r in rs if r['first_edit_call']])} | "
                   f"{median([r['tool_calls'] for r in rs])} | {median([round(r['minutes'], 1) for r in rs])} |")
    ver_tasks = sum(rs[0]["verifiable"] for rs in by_task.values())
    expected = sum(r["resolved"] for r in runs if r["verifiable"]) / (n_ver / ver_tasks)
    out += ["", f"Tiers on verifiable tasks: {tiers}. Expected resolved per run: {expected:.1f} of {ver_tasks}.", ""]
    out += ["## Flipping tasks: resolved against lost runs of the same task", "",
            "| metric | resolved (median) | lost (median) |", "|---|---|---|"]
    flips = [rs for rs in by_task.values() if rs[0]["verifiable"] and 0 < sum(r["resolved"] for r in rs) < len(rs)]
    for label, f in (("first edit call", lambda r: r["first_edit_call"]), ("tool calls", lambda r: r["tool_calls"]),
                     ("minutes", lambda r: round(r["minutes"], 2)), ("failed edits", lambda r: r["failed_edits"]),
                     ("python runs", lambda r: r["py_runs"]), ("max turn tokens", lambda r: r["max_turn_tokens"])):
        a = [median([f(r) for r in rs if r["resolved"] and f(r) is not None]) for rs in flips]
        b = [median([f(r) for r in rs if not r["resolved"] and f(r) is not None]) for rs in flips]
        a, b = [x for x in a if x is not None], [x for x in b if x is not None]
        out.append(f"| {label} | {median(a)} | {median(b)} |")
    lost = [r for rs in flips for r in rs if not r["resolved"]]
    never_edited = sum(r["first_edit_call"] is None for r in lost)
    timeouts = sum("timeout" in r["error"].lower() for r in lost)
    out += ["", f"Lost runs of flipping tasks: {len(lost)}; no successful edit {never_edited}; "
            f"session timeout {timeouts}; outcomes {dict(Counter(bucket(r) for r in lost))}.", ""]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", type=Path, default=ROOT / "data/eval")
    ap.add_argument("--out", type=Path, default=ROOT / "data/analysis/near_miss.md")
    args = ap.parse_args()
    text = report(load_runs(args.eval))
    args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
