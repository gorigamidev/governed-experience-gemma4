"""Conditional authority (doc 03 §14 item 58, phase A): does the memory's top-1 file beat the model's own first edit
where the model has the least information (title-only statements)?

Per run: model = the first successful edit lands in a gold file; memory = `locate` ranks a gold file first
(`localization_bench.csv`: `rank_habits` is out-of-sample habits, `rank_lexical` lexical only). Statements of at most
SHORT characters are short (threshold fixed in `edit_deadline`, October).

Usage: PYTHONPATH=lab/src python3 -m lab.analysis.authority_split [--out data/analysis/authority_split.md]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from lab.analysis.pool_report import gold_files
from lab.analysis.self_imitation import CORPUS, _calls
from lab.ingest.load_traces import EDIT_TOOLS, _is_error

SHORT = 243
EVAL = Path("data/eval")
E4B = ("preflight-20261103-050924-bf-gov-tb-e4b", "preflight-20261103-110441-bf-gov-tb-e4b")


def first_edit_file(trace: dict) -> str | None:
    for name, args, content in _calls(trace):
        if name in EDIT_TOOLS and not _is_error(content):
            f = args.get("filepath") or args.get("file_path") or args.get("path")
            return str(f).removeprefix("/workspace/").lstrip("./") if f else None
    return None


def mcnemar(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value from the discordant counts."""
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)


def compare(rows: list[dict], memory: str) -> dict:
    """Runs and per-task majority: model accuracy, memory accuracy, discordant tasks and McNemar."""
    n = len(rows)
    model = sum(r["model"] for r in rows)
    mem = sum(r[memory] for r in rows)
    by_task = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)
    maj = {t: sum(x["model"] for x in rs) / len(rs) >= 0.5 for t, rs in by_task.items()}
    memt = {t: rs[0][memory] for t, rs in by_task.items()}
    b = sum(maj[t] and not memt[t] for t in maj)
    c = sum(memt[t] and not maj[t] for t in maj)
    return {"runs": n, "model": model, "memory": mem, "diff_pts": round(100 * (mem - model) / n, 1) if n else 0.0,
            "tasks": len(maj), "model_tasks": sum(maj.values()), "memory_tasks": sum(memt.values()),
            "model_only": b, "memory_only": c, "p": round(mcnemar(b, c), 3)}


def load(sources, tasks: dict, bench: dict, gold: dict) -> list[dict]:
    rows = []
    for src in sources:
        for tf in sorted((EVAL / src / "traces").glob("trace_*.json")):
            task = tf.stem.removeprefix("trace_")
            f = first_edit_file(json.loads(tf.read_text()))
            if f is None or task not in bench:
                continue
            rows.append({"task": task, "short": len(tasks[task]["problem_statement"]) <= SHORT,
                         "model": f in gold.get(task, set()),
                         "habits": bench[task]["rank_habits"] == "1", "lexical": bench[task]["rank_lexical"] == "1"})
    return rows


def verdict(c31: dict) -> list[str]:
    a1 = c31["short"]["habits"]["diff_pts"] >= 10
    a2 = -c31["long"]["habits"]["diff_pts"] >= 10
    return [f"- A1 (short: memory - model >= +10 points, 31B runs): {'MET' if a1 else 'NOT MET'} "
            f"({c31['short']['habits']['diff_pts']:+} points)",
            f"- A2 (long: model - memory >= +10 points, 31B runs): {'MET' if a2 else 'NOT MET'} "
            f"({-c31['long']['habits']['diff_pts']:+} points)"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/analysis/authority_split.md"))
    a = ap.parse_args()
    tasks = {json.loads(x)["instance_id"]: json.loads(x) for x in Path("data/competition/tasks.jsonl").read_text().splitlines() if x.strip()}
    bench = {r["instance_id"]: r for r in csv.DictReader(open("data/analysis/localization_bench.csv"))}
    gold = gold_files()
    out = ["# Conditional authority, phase A (doc 03 §14 item 58)", "",
           f"Short statement: <= {SHORT} characters. Model: first successful edit in a gold file. Memory: `locate` top-1 is gold.", ""]
    res = {}
    for name, src in (("31B", CORPUS), ("E4B", E4B)):
        rows = load(src, tasks, bench, gold)
        res[name] = {split: {m: compare([r for r in rows if r["short"] == (split == "short")], m) for m in ("habits", "lexical")}
                     for split in ("short", "long")}
        res[name]["all"] = {m: compare(rows, m) for m in ("habits", "lexical")}
        out += [f"## {name}", "", "| statements | memory | runs | model | memory | diff (pts) | tasks | model tasks | memory tasks | model only | memory only | McNemar p |",
                "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for split in ("short", "long", "all"):
            for m in ("habits", "lexical"):
                c = res[name][split][m]
                out.append(f"| {split} | {m} | {c['runs']} | {c['model']} | {c['memory']} | {c['diff_pts']:+} | {c['tasks']} | "
                           f"{c['model_tasks']} | {c['memory_tasks']} | {c['model_only']} | {c['memory_only']} | {c['p']} |")
        out.append("")
    out += ["## Rules (31B)", ""] + verdict(res["31B"])
    a.out.write_text("\n".join(out) + "\n")
    print(a.out.read_text())


if __name__ == "__main__":
    main()
