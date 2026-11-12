"""Doc 03 §14 item 54, phase A: would a read lock after N pre-edit calls have found the edit already in hand?

Over every GPU pool trace of the bf family (`near_miss.SESSIONS`):
- lock call: a charged call before the first successful edit, other than `get_status`;
- final edit: the first `edit_file` with an `old_string` that succeeded or was refused by the call budget;
- affected: the final edit comes after lock call N, or there is none;
- in hand: the final edit's `old_string` already appears verbatim in an observation of the first N lock calls.

Usage: PYTHONPATH=lab/src lab/.venv/bin/python -m lab.analysis.lock_sim [--out data/analysis/lock_sim.md]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from lab.analysis import near_miss
from lab.analysis.pool_report import AGENT_TOOLS, ROOT
from lab.ingest.load_traces import _is_error

N, N_ALT = 10, (8, 12)
FREE = {"get_status", "submit_patch"}


def _text(obs) -> str:
    """Every string inside a tool observation, decoded from its JSON envelope."""
    raw = (obs or {}).get("content") if isinstance(obs, dict) else obs
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
    except ValueError:
        return str(raw or "")
    out = []

    def walk(x):
        if isinstance(x, str):
            out.append(x)
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(data)
    return "\n".join(out)


def scan(trace: dict) -> dict:
    """Lock calls (with their observation text) before the final edit, and the final edit."""
    lock_obs, final, first_ok = [], None, None
    for step in trace.get("steps", []):
        obs = step.get("observation")
        for tc in step.get("tool_calls") or []:
            name = tc.get("function_name")
            if name in AGENT_TOOLS or name in FREE:
                continue
            args = tc.get("arguments") or {}
            if name in ("edit_file", "write_file"):
                content = str((obs or {}).get("content") if isinstance(obs, dict) else obs)
                ok = not _is_error(content)
                refused = "BudgetExceeded" in content
                if name == "edit_file" and args.get("old_string") and (ok or refused) and final is None:
                    final = {"old_string": str(args["old_string"]), "refused": refused, "at": len(lock_obs) + 1}
                if ok:
                    first_ok = first_ok or len(lock_obs) + 1
                    return {"lock": lock_obs, "final": final, "first_ok": first_ok}
                if final is not None:
                    return {"lock": lock_obs, "final": final, "first_ok": first_ok}
                continue  # a failed edit before any final edit is charged, but it is not a read
            lock_obs.append(_text(obs))
    return {"lock": lock_obs, "final": final, "first_ok": first_ok}


def in_hand(old: str, seen: list[str]) -> bool:
    old = "\n".join(line.rstrip() for line in old.strip("\n").split("\n")).strip()
    text = "\n".join("\n".join(line.rstrip() for line in s.split("\n")) for s in seen)
    return bool(old) and old in text


def rows(eval_root: Path) -> list[dict]:
    out = []
    for r in near_miss.load_runs(eval_root):
        path = eval_root / r["session"] / "results" / r["arm"] / "traces" / f"trace_{r['task']}.json"
        if not path.exists():
            continue
        s = scan(json.loads(path.read_text()))
        row = {"session": r["session"], "arm": r["arm"], "task": r["task"], "resolved": bool(r["resolved"]),
               "verifiable": bool(r["verifiable"]), "has_final": s["final"] is not None,
               "final_at": s["final"]["at"] if s["final"] else None,
               "refused": bool(s["final"] and s["final"]["refused"]), "lock_calls": len(s["lock"])}
        for n in (N, *N_ALT):
            row[f"affected_{n}"] = s["final"] is None or s["final"]["at"] > n
            row[f"in_hand_{n}"] = bool(s["final"]) and in_hand(s["final"]["old_string"], s["lock"][:n])
        out.append(row)
    return out


def verdict(rs: list[dict], n: int = N) -> dict:
    aff = [r for r in rs if r[f"affected_{n}"] and r["has_final"]]
    hand = sum(r[f"in_hand_{n}"] for r in aff)
    by_task = defaultdict(list)
    for r in aff:
        by_task[r["task"]].append(r[f"in_hand_{n}"])
    task_ok = {t: sum(v) > len(v) / 2 for t, v in by_task.items()}
    resolved = [r for r in rs if r["resolved"]]
    harm = [r for r in resolved if r[f"affected_{n}"] and not r[f"in_hand_{n}"]]
    a1 = bool(aff) and hand / len(aff) >= 0.5 and sum(task_ok.values()) / len(task_ok) >= 0.5
    a2 = bool(resolved) and len(harm) / len(resolved) <= 0.25
    return {"n": n, "affected_with_final": len(aff), "affected_no_final": sum(r[f"affected_{n}"] and not r["has_final"] for r in rs),
            "in_hand": hand, "tasks": len(task_ok), "tasks_in_hand": sum(task_ok.values()), "task_ok": task_ok,
            "resolved": len(resolved), "harm": len(harm), "a1": a1, "a2": a2, "go": a1 and a2}


def report(rs: list[dict]) -> list[str]:
    v = verdict(rs)
    lines = ["# Read lock simulation (doc 03 §14 item 54, phase A)", "",
             f"- Traces: {len(rs)}; with a final edit {sum(r['has_final'] for r in rs)} "
             f"(refused by the budget {sum(r['refused'] for r in rs)})",
             f"- Lock at {N}: affected with a final edit {v['affected_with_final']}, in hand {v['in_hand']} "
             f"({v['in_hand'] / max(1, v['affected_with_final']):.0%}, rule >= 50%); tasks in hand by majority "
             f"{v['tasks_in_hand']}/{v['tasks']} (rule >= 50%); affected without a final edit {v['affected_no_final']}",
             f"- Harm: resolved runs affected and not in hand {v['harm']}/{v['resolved']} "
             f"({v['harm'] / max(1, v['resolved']):.0%}, rule <= 25%)",
             f"- A1 {'MET' if v['a1'] else 'NOT MET'}; A2 {'MET' if v['a2'] else 'NOT MET'}; "
             f"**Phase A: {'GO' if v['go'] else 'NO GO'}**", "", "## Descriptive: other lock points", ""]
    for n in N_ALT:
        w = verdict(rs, n)
        lines.append(f"- {n}: affected {w['affected_with_final']}, in hand {w['in_hand']}, tasks {w['tasks_in_hand']}/"
                     f"{w['tasks']}, harm {w['harm']}/{w['resolved']}")
    lines += ["", f"## Per task at {N}", "", "| task | verifiable | resolved | affected (final) | in hand | no final edit |",
              "|---|---|---|---|---|---|"]
    by_task = defaultdict(list)
    for r in rs:
        by_task[r["task"]].append(r)
    for t, g in sorted(by_task.items()):
        aff = [r for r in g if r[f"affected_{N}"] and r["has_final"]]
        lines.append(f"| {t} | {'yes' if g[0]['verifiable'] else 'no'} | {sum(r['resolved'] for r in g)}/{len(g)} | "
                     f"{len(aff)} | {sum(r[f'in_hand_{N}'] for r in aff)} | {sum(not r['has_final'] for r in g)} |")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", type=Path, default=ROOT / "data/eval")
    ap.add_argument("--out", type=Path, default=ROOT / "data/analysis/lock_sim.md")
    args = ap.parse_args()
    rs = rows(args.eval)
    lines = report(rs)
    args.out.write_text("\n".join(lines) + "\n")
    args.out.with_suffix(".jsonl").write_text("".join(json.dumps(r) + "\n" for r in rs))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
