"""Doc 03 §14 item 54: read-lock metrics from the traces of a `bf-gov-tb-lk` run, and the phase C rules.

Per task: skill calls and their failures, calls to tools the agent does not have, lock activations and what the
agent did next, first successful edit (in charged calls), resolved.

C1  >= 95% of skill calls without REGISTRY_ERROR or a script error (no JSON / stderr traceback), as written in doc 03;
    malformed calls the model sends (INVALID_ARGUMENTS, SKILL_NOT_FOUND) are reported apart and do not count
C2  no task ends because of an undeclared tool
C3  >= 2 of the 3 always-resolved tasks resolved (rich_3006, rich_3882, rich_3905)
C4  the lock fires in >= 2 of the 4 flipping tasks (rich_3480, rich_3521, rich_3470, rich_4075), and in >= 50% of
    activations the next call that is not a skill call is a successful edit within 3 calls

Usage: PYTHONPATH=lab/src lab/.venv/bin/python -m lab.analysis.lock_report --results <dir with traces/> [--out f.md]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lab.ingest.load_traces import _is_error

ALLOWED = {"edit_file", "write_file", "get_status", "submit_patch", "run_skill_script", "list_skills", "load_skill",
           "load_skill_resource"}
ALWAYS = ("rich_3006", "rich_3882", "rich_3905")
FLIPS = ("rich_3480", "rich_3521", "rich_3470", "rich_4075")
SKILL_ERRORS = ("REGISTRY_ERROR", "INVALID_ARGUMENTS")


def _content(obs) -> str:
    return str((obs or {}).get("content") if isinstance(obs, dict) else obs or "")


def skill_payload(content: str) -> tuple[dict | None, str]:
    """The script's own JSON from a run_skill_script observation, and an error label ('' if fine)."""
    try:
        outer = json.loads(content)
    except ValueError:
        outer = None
    if isinstance(outer, dict) and outer.get("error_code"):
        return None, str(outer["error_code"])
    if "BudgetExceeded" in content:  # the harness refused the call (budget spent): not a skill failure
        return None, "budget"
    for code in SKILL_ERRORS:
        if code in content:
            return None, code
    if not isinstance(outer, dict):
        return None, "unparseable observation"
    stdout = outer.get("stdout")
    if stdout is None and isinstance(outer.get("result"), dict):
        stdout = outer["result"].get("stdout")
    if stdout is None:
        return None, f"no stdout ({str(outer.get('error') or outer.get('status'))[:60]})"
    lines = [ln for ln in str(stdout).strip().split("\n") if ln.strip()]
    try:
        payload = json.loads(lines[-1]) if lines else None
    except ValueError:
        payload = None
    if not isinstance(payload, dict):
        return None, "script printed no JSON"
    if "Traceback" in str(outer.get("stderr") or ""):
        return payload, "script traceback"
    return payload, ""


def corrupt_call(args) -> bool:
    """Doc 03 §14 item 54, C5: a skill call is corrupt if file_path holds a backtick or skill_name is missing
    (the 31B's "scripts/entrypoint.py`,skill_name:" shape)."""
    if not isinstance(args, dict):
        return True
    return "skill_name" not in args or "`" in str(args.get("file_path", ""))


def task_metrics(trace: dict) -> dict:
    calls = []
    for step in trace.get("steps", []):
        obs = step.get("observation")
        for tc in step.get("tool_calls") or []:
            calls.append((tc.get("function_name"), tc.get("arguments") or {}, _content(obs)))
    m = {"calls": len(calls), "skill_calls": 0, "skill_errors": [], "model_errors": [], "undeclared": [], "activations": 0,
         "edit_after_lock": 0, "rereads": 0, "first_edit_call": None, "first_lock_call": None,
         "corrupt": 0, "first_corrupt": None}
    for i, (name, args, content) in enumerate(calls, 1):
        if name not in ALLOWED:
            m["undeclared"].append(name)
        if name in ("edit_file", "write_file") and m["first_edit_call"] is None and not _is_error(content):
            m["first_edit_call"] = i
        if name != "run_skill_script":
            continue
        m["skill_calls"] += 1
        bad_shape = corrupt_call(args)
        m["corrupt"] += bad_shape
        if m["first_corrupt"] is None:
            m["first_corrupt"] = bad_shape
        payload, err = skill_payload(content)
        well_formed = ((args.get("skill_name") == "code" and args.get("file_path") in ("scripts/look.py", "scripts/run.py"))
                       or (args.get("skill_name") == "repo-tools" and args.get("file_path") == "scripts/entrypoint.py"))
        if err == "budget":
            m["budget_refused"] = m.get("budget_refused", 0) + 1
        elif err and well_formed:
            m["skill_errors"].append(err)  # a correct call that failed: the skill or the harness is broken
        elif err:
            m["model_errors"].append(err)  # the model sent a wrong call (missing name, unknown skill or script)
        if payload and payload.get("reread"):
            m["rereads"] += 1
        if payload and payload.get("locked") and not payload.get("reread"):
            if m["first_lock_call"] is None:
                m["first_lock_call"] = i
            m["activations"] += 1
            nxt = [(j, c) for j, c in enumerate(calls[i:], i + 1) if c[0] != "run_skill_script"][:1]
            if nxt and nxt[0][0] - i <= 3 and nxt[0][1][0] in ("edit_file", "write_file") and not _is_error(nxt[0][1][2]):
                m["edit_after_lock"] += 1
    return m


def load(results: Path) -> dict[str, dict]:
    out = {}
    tr = {}
    p = results / "task_results.jsonl"
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                tr[r.get("task_id") or r.get("instance_id")] = r  # the scorer's runner writes instance_id
    for path in sorted((results / "traces").glob("trace_*.json")):
        task = path.stem.removeprefix("trace_")
        m = task_metrics(json.loads(path.read_text()))
        r = tr.get(task, {})
        m["resolved"] = bool(r.get("resolved"))
        m["error"] = str(r.get("error") or "")
        m["ended_by_undeclared"] = bool(m["undeclared"]) and any(
            u and u in m["error"] for u in m["undeclared"])
        out[task] = m
    return out


def rules(rows: dict[str, dict]) -> tuple[list[str], dict]:
    skill = sum(r["skill_calls"] for r in rows.values())
    bad = sum(len(r["skill_errors"]) + sum(e == "REGISTRY_ERROR" for e in r.get("model_errors", []))
              for r in rows.values())
    malformed = sum(len(r.get("model_errors", [])) for r in rows.values())
    c1 = skill > 0 and (skill - bad) / skill >= 0.95
    c2 = not any(r["ended_by_undeclared"] for r in rows.values())
    always = [t for t in ALWAYS if t in rows]
    c3 = len(always) == 3 and sum(rows[t]["resolved"] for t in always) >= 2
    fired = [t for t in FLIPS if t in rows and rows[t]["activations"]]
    acts = sum(r["activations"] for r in rows.values())
    good = sum(r["edit_after_lock"] for r in rows.values())
    c4 = len(fired) >= 2 and acts > 0 and good / acts >= 0.5
    corrupt = sum(r.get("corrupt", 0) for r in rows.values())
    first_bad = sum(bool(r.get("first_corrupt")) for r in rows.values())
    c5 = skill > 0 and corrupt / skill <= 0.05 and first_bad <= 1
    verdict = {"C1": c1, "C2": c2, "C3": c3, "C4": c4, "C5": c5}
    infra = sum(len(r["skill_errors"]) for r in rows.values())
    lines = [f"- C1 skill calls without REGISTRY_ERROR or script error {skill - bad}/{skill} (well-formed calls that "
             f"failed: {infra}; malformed calls, descriptive: {malformed}): {'MET' if c1 else 'NOT MET'}",
             f"- C2 tasks ended by an undeclared tool {sum(r['ended_by_undeclared'] for r in rows.values())} "
             f"(undeclared calls {sum(len(r['undeclared']) for r in rows.values())}): {'MET' if c2 else 'NOT MET'}",
             f"- C3 always-resolved tasks resolved {sum(rows[t]['resolved'] for t in always)}/{len(always)}: "
             f"{'MET' if c3 else 'NOT MET'}",
             f"- C4 lock fired in {len(fired)}/{sum(t in rows for t in FLIPS)} flipping tasks; edit right after "
             f"{good}/{acts}: {'MET' if c4 else 'NOT MET'}",
             f"- Flipping tasks resolved (descriptive): {sum(rows[t]['resolved'] for t in FLIPS if t in rows)}"
             f"/{sum(t in rows for t in FLIPS)}",
             f"- C5 (probe v4) corrupt skill calls {corrupt}/{skill} ({corrupt / max(skill, 1):.0%}, max 5%); first skill "
             f"call corrupt in {first_bad}/{sum(r['skill_calls'] > 0 for r in rows.values())} tasks (max 1): "
             f"{'MET' if c5 else 'NOT MET'}"]
    return lines, verdict


def report(rows: dict[str, dict]) -> str:
    lines = ["# Read lock report (doc 03 §14 item 54)", "",
             "| task | resolved | calls | skill calls | skill errors | malformed | undeclared | lock at | activations | "
             "edit after lock | rereads | first edit |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for t, r in rows.items():
        lines.append(f"| {t} | {r['resolved']} | {r['calls']} | {r['skill_calls']} | {', '.join(r['skill_errors']) or 0} "
                     f"| {', '.join(r['model_errors']) or 0} | {', '.join(map(str, r['undeclared'])) or 0} | {r['first_lock_call'] or '-'} | {r['activations']} "
                     f"| {r['edit_after_lock']} | {r['rereads']} | {r['first_edit_call'] or '-'} |")
    rule_lines, _ = rules(rows)
    return "\n".join(lines + ["", "## Phase C rules", ""] + rule_lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    text = report(load(args.results))
    if args.out:
        args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
