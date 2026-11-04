"""Rules of doc 03 §14.44 (memory for E4B): does `locate` cut the edits E4B makes outside the gold region?

"Misplaced" = a non-empty patch whose non-test hunks touch no non-test hunk of the reference patch (5-line slack),
the measure that produced the hypothesis. Local runs only (`pipeline_test`); times are not compared.

Usage: PYTHONPATH=lab/src lab/.venv/bin/python -m lab.analysis.memory_e4b --base <run> --mp <run> \
           [--prev <run>] [--out data/analysis/memory_e4b.md]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from lab.analysis.pool_report import ROOT, trace_metrics

SLACK = 5
SKILL = "run_skill_script"
EDITS = ("edit_file", "write_file")


def _is_test(path: str) -> bool:
    return path.startswith(("tests/", "docs")) or "/tests/" in path


def hunks(patch: str) -> list[tuple[str, int, int]]:
    out, f = [], None
    for line in patch.splitlines():
        m = re.match(r"^\+\+\+ b/(.+)$", line)
        if m:
            f = m.group(1)
            continue
        m = re.match(r"^@@ -(\d+)(?:,(\d+))? ", line)
        if m and f and not _is_test(f):
            start, n = int(m.group(1)), int(m.group(2) or 1)
            out.append((f, start, start + max(n, 1)))
    return out


def placement(patch: str, gold: str) -> str:
    """'empty', 'region' (touches a gold hunk), 'file' (gold file, other region) or 'miss'."""
    if not patch.strip():
        return "empty"
    a, g = hunks(patch), hunks(gold)
    if any(fa == fg and sa - SLACK <= eg and sg - SLACK <= ea for fa, sa, ea in a for fg, sg, eg in g):
        return "region"
    return "file" if {f for f, _, _ in a} & {f for f, _, _ in g} else "miss"


def locate_stats(trace: dict) -> tuple[bool, int, int]:
    """(locate before the first edit, locate calls, locate calls that returned at least one candidate)."""
    calls = hits = 0
    before = False
    for step in trace.get("steps", []):
        for c in step.get("tool_calls") or []:
            name, args = c.get("function_name"), c.get("arguments") or {}
            if name in EDITS:
                return before, calls, hits
            inner = args.get("args") or {}
            if name == SKILL and inner.get("action", "locate") == "locate" and "file" not in inner:
                calls += 1
                before = True
                flat = json.dumps(step.get("observation", "")).replace("\\", "").replace('"', "")
                if re.search(r"items:\s*\[\s*\{", flat):
                    hits += 1
    return before, calls, hits


def load_run(run: Path, gold: dict[str, str]) -> dict[str, dict]:
    res = {}
    for line in (run / "task_results.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            res[r.get("instance_id") or r.get("task_id")] = r
    cap = json.loads((run / "run_meta.json").read_text())["budgets"]["max_tool_calls"]
    rows = {}
    for t, r in res.items():
        if t not in gold:
            continue
        p = run / "patches" / f"{t}.patch"
        tr = run / "traces" / f"trace_{t}.json"
        trace = json.loads(tr.read_text()) if tr.exists() else {}
        before, calls, hits = locate_stats(trace)
        rows[t] = {"placement": placement(p.read_text() if p.exists() else "", gold[t]),
                   "resolved": bool(r.get("resolved")),
                   # Same split as pool_report: the harness's own "Agent ..." endings are model behavior, not infra.
                   "infra": bool(r.get("error")) and not r["error"].startswith("Agent"),
                   "call_cap": trace_metrics(trace)["charged_calls"] >= cap if trace else False,
                   "locate_first": before, "locate_calls": calls, "locate_hits": hits,
                   "max_prompt": max([(s.get("metrics") or {}).get("prompt_tokens") or 0
                                      for s in trace.get("steps", [])] or [0])}
    return rows


def summary(rows: dict[str, dict]) -> dict:
    v = rows.values()
    return {"n": len(rows), "misplaced": sum(r["placement"] in ("file", "miss") for r in v),
            "region": sum(r["placement"] == "region" for r in v), "empty": sum(r["placement"] == "empty" for r in v),
            "R": sum(r["resolved"] for r in v), "infra": sum(r["infra"] for r in v),
            "call_cap": sum(r["call_cap"] for r in v), "locate_first": sum(r["locate_first"] for r in v),
            "locate_calls": sum(r["locate_calls"] for r in v), "locate_hits": sum(r["locate_hits"] for r in v),
            "max_prompt": max((r["max_prompt"] for r in v), default=0)}


def rules(base: dict, mp: dict, common: int, prev: dict | None = None) -> list[str]:
    out = []
    valid = common >= 20 and base["infra"] == 0 and mp["infra"] == 0 and max(base["max_prompt"], mp["max_prompt"]) < 32768
    out.append(f"Rule 1 (validity): {'OK' if valid else 'NOT MET'} | {common} common tasks, infra {base['infra']}/"
               f"{mp['infra']}, max prompt {max(base['max_prompt'], mp['max_prompt'])}")
    hit_rate = mp["locate_hits"] / mp["locate_calls"] if mp["locate_calls"] else 0.0
    complies = mp["locate_first"] >= 18 and hit_rate >= 0.9
    out.append(f"Rule 2 (compliance): {'MET' if complies else 'NOT MET'} | locate before first edit "
               f"{mp['locate_first']}/{mp['n']}, candidates in {mp['locate_hits']}/{mp['locate_calls']} calls")
    if not valid or not complies:
        out.append("Rule 3: not read")
    else:
        ok = (mp["misplaced"] <= base["misplaced"] - 3 and mp["R"] >= base["R"] - 1
              and mp["call_cap"] <= base["call_cap"] + 2)
        out.append(f"Rule 3 (mechanism, goes to the scorer): {'YES' if ok else 'NO'} | misplaced {mp['misplaced']} vs "
                   f"{base['misplaced']} - 3, R {mp['R']} vs {base['R']} - 1, call cap {mp['call_cap']} vs "
                   f"{base['call_cap']} + 2")
    if prev is not None:
        moved = abs(base["misplaced"] - prev["misplaced"])
        out.append(f"Rule 5 (noise, no decision): base misplaced {base['misplaced']} now vs {prev['misplaced']} before"
                   + (" -> the rule-3 threshold is within the noise" if moved >= 3 else ""))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--mp", type=Path, required=True)
    ap.add_argument("--prev", type=Path, help="earlier run of the base arm (rule 5)")
    ap.add_argument("--out", type=Path, default=ROOT / "data/analysis/memory_e4b.md")
    a = ap.parse_args()
    tasks = [json.loads(ln) for ln in (ROOT / "data/competition/tasks.jsonl").read_text().splitlines() if ln.strip()]
    gold = {t["instance_id"]: t["patch"] for t in tasks}
    b, m = load_run(a.base, gold), load_run(a.mp, gold)
    common = sorted(set(b) & set(m))
    b, m = {t: b[t] for t in common}, {t: m[t] for t in common}
    sb, sm = summary(b), summary(m)
    sp = None
    if a.prev:
        p = load_run(a.prev, gold)
        sp = summary({t: p[t] for t in common if t in p})
    lines = [f"# Memory for E4B (doc 03 §14.44)", "", f"base `{a.base}` | mp `{a.mp}`", "",
             "| arm | n | misplaced | region | empty | R | call_cap | locate_first | locate hits/calls | max_prompt |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name, s in (("base", sb), ("mp", sm)) + ((("prev base", sp),) if sp else ()):
        lines.append(f"| {name} | {s['n']} | {s['misplaced']} | {s['region']} | {s['empty']} | {s['R']} | "
                     f"{s['call_cap']} | {s['locate_first']} | {s['locate_hits']}/{s['locate_calls']} | {s['max_prompt']} |")
    lines += ["", "## Rules", ""] + [f"- {r}" for r in rules(sb, sm, len(common), sp)]
    lines += ["", "## Per task", "", "| task | base | mp |", "|---|---|---|"]
    for t in common:
        lines.append(f"| {t} | {b[t]['placement']}{' R' if b[t]['resolved'] else ''} | "
                     f"{m[t]['placement']}{' R' if m[t]['resolved'] else ''} |")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
