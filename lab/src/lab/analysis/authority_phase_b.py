"""Conditional authority, phase B (doc 03 §14 item 58): out-of-sample short statements, E4B local, base against
`mp-nohab` in one session. Reads rules B1-B3 as written on Nov 11 before the run.

Per task and arm: first successful edit in a gold file, empty patch, `locate` called before the first edit, the
largest prompt, and (descriptive) the rank of the gold file in the first valid `locate` result.

Usage: PYTHONPATH=lab/src python3 -m lab.analysis.authority_phase_b --stamp 20261111-102124 [--suffix -gemma]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lab.analysis.authority_split import first_edit_file, mcnemar
from lab.analysis.pool_report import gold_files
from lab.analysis.self_imitation import _calls
from lab.ingest.load_traces import EDIT_TOOLS

EVAL = Path("data/eval")
TASKS = ("fastapi_13920 fastapi_14361 fastapi_14487 fastapi_14609 fastapi_14962 fastapi_14978 fastapi_15023 "
         "fastapi_15280 httpx_3672 rich_3063 rich_3506 rich_3676 rich_3934 rich_3944 rich_4077").split()
ARMS = {"base": "bf-gov-tb-e4b", "mp": "bf-gov-tb-mp-nohab"}


def is_locate(name: str, args: dict) -> bool:
    return name == "run_skill_script" and "text" in (args.get("args") or {})


def locate_files(content) -> list[str] | None:
    """Files of a valid `locate` result, or None when the call failed."""
    try:
        outer = json.loads(content) if isinstance(content, str) else content
        return [i["file"] for i in json.loads(outer["stdout"]).get("items", [])]
    except (TypeError, KeyError, ValueError):
        return None


def task_row(trace: dict, gold: set[str], patch_empty: bool) -> dict:
    calls = _calls(trace)
    first = next((i for i, (n, _, _) in enumerate(calls) if n in EDIT_TOOLS), len(calls))
    ranks = [f for n, a, c in calls if is_locate(n, a) and (f := locate_files(c)) is not None]
    fe = first_edit_file(trace)
    return {"first_edit": fe, "gold": fe in gold if fe else False, "empty": patch_empty,
            "locate_first": any(is_locate(n, a) for n, a, _ in calls[:first]),
            "max_prompt": max((s.get("metrics", {}).get("prompt_tokens", 0) for s in trace["steps"]), default=0),
            "gold_rank": next((i + 1 for i, f in enumerate(ranks[0]) if f in gold), None) if ranks else None,
            "located": bool(ranks)}


def verdict(base: dict, mp: dict, infra: int) -> dict:
    both = [t for t in base if t in mp]
    gb, gm = sum(base[t]["gold"] for t in both), sum(mp[t]["gold"] for t in both)
    eb, em = sum(base[t]["empty"] for t in both), sum(mp[t]["empty"] for t in both)
    loc = sum(mp[t]["locate_first"] for t in mp)
    maxp = max(r["max_prompt"] for arm in (base, mp) for r in arm.values())
    b1 = len(both) >= 13 and infra == 0 and maxp < 32768 and loc >= 0.8 * len(mp)
    only_b = sum(base[t]["gold"] and not mp[t]["gold"] for t in both)
    only_m = sum(mp[t]["gold"] and not base[t]["gold"] for t in both)
    return {"tasks": len(both), "infra": infra, "max_prompt": maxp, "locate_first": loc, "b1": b1,
            "gold_base": gb, "gold_mp": gm, "empty_base": eb, "empty_mp": em,
            "b2": b1 and gm >= gb + 2 and em <= eb + 1, "b3": b1 and gm <= gb - 2,
            "only_base": only_b, "only_mp": only_m, "p": round(mcnemar(only_b, only_m), 4),
            "gains_top1_gold": sum(mp[t]["gold_rank"] == 1 for t in both if mp[t]["gold"] and not base[t]["gold"])}


def load(stamp: str, suffix: str) -> tuple[dict, dict, int]:
    from lab.tools.preflight import trace_problems
    gold, out, infra = gold_files(), {}, 0
    for arm, name in ARMS.items():
        d = EVAL / f"preflight-{stamp}-{name}{suffix}"
        infra += len(trace_problems(d, arm == "mp", require_skill=False)[0])
        rows = {}
        for t in TASKS:
            tf, pf = d / "traces" / f"trace_{t}.json", d / "patches" / f"{t}.patch"
            if tf.exists():
                rows[t] = task_row(json.loads(tf.read_text()), gold.get(t, set()), not pf.exists() or pf.stat().st_size == 0)
        out[arm] = rows
    return out["base"], out["mp"], infra


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stamp", default="20261111-102124")
    ap.add_argument("--suffix", default="", help="'-gemma' reads the preflight's unplanned replicate (descriptive)")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    base, mp, infra = load(a.stamp, a.suffix)
    v = verdict(base, mp, infra)
    out = [f"# Conditional authority, phase B (doc 03 §14 item 58), preflight-{a.stamp}{a.suffix}", "",
           "| task | base first edit gold | mp first edit gold | base empty | mp empty | mp locate first | gold rank in mp locate |",
           "|---|---|---|---|---|---|---|"]
    for t in TASKS:
        if t in base and t in mp:
            b, m = base[t], mp[t]
            rank = m["gold_rank"] if m["gold_rank"] else ("not listed" if m["located"] else "no valid locate")
            out.append(f"| {t} | {b['gold']} | {m['gold']} | {b['empty']} | {m['empty']} | {m['locate_first']} | {rank} |")
    out += ["", f"- B1 validity: {'MET' if v['b1'] else 'NOT MET'} ({v['tasks']} tasks, {v['infra']} infra errors, "
            f"max prompt {v['max_prompt']}, locate before first edit {v['locate_first']}/{len(mp)})",
            f"- First edit in gold: base {v['gold_base']}, mp {v['gold_mp']}; empty patches: base {v['empty_base']}, mp {v['empty_mp']}",
            f"- B2 (mp >= base + 2 and empty mp <= base + 1): {'MET' if v['b2'] else 'NOT MET'}",
            f"- B3 (mp <= base - 2): {'MET' if v['b3'] else 'NOT MET'}",
            f"- Discordant tasks: base only {v['only_base']}, mp only {v['only_mp']}, McNemar p = {v['p']}",
            f"- Descriptive: of the {v['only_mp']} mp-only gains, `locate` ranked the gold file first in {v['gains_top1_gold']}"]
    text = "\n".join(out) + "\n"
    out_path = a.out or Path(f"data/analysis/authority_phase_b{a.suffix}.md")
    out_path.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
