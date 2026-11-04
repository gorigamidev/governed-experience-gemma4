"""Paired pool report: the metrics and rules of doc 03 §14.33 for several arms run on the same tasks.

R counts the local Docker verdict on the verifiable tasks and the harness verdict elsewhere (fastapi is not
verifiable locally). Only tasks present in every arm are compared.

Usage: PYTHONPATH=lab/src lab/.venv/bin/python -m lab.analysis.pool_report \
         --arms bf=data/eval/kaggle-pool-bf,h24=data/eval/kaggle-pool-h24,h24-long=data/eval/kaggle-pool-h24-long
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from lab.analysis.command_timeouts import quantile, trace_commands
from lab.analysis.long_turns import LONG, trace_turns
from lab.eval.verifiable import load_verifiable
from lab.ingest.load_traces import EDIT_TOOLS, _is_error

ROOT = Path(__file__).resolve().parents[4]
AUX_RE = re.compile(r"(repro|_issue|test_graphemes)")  # experiments/autopsy/metrics.py
EDIT_BY_CALL = 12  # the call-plan deadline of bf and h24
MIN_COMMON = 18
AGENT_TOOLS = {"locator"}  # agent_tool calls: the harness charges the sub-agent's tool calls, not the call itself
UNDECLARED_TOOL = re.compile(r"Tool '[^']+' not found")
PLAN_FILE = re.compile(r"^\s*FILES?:\s*`?([^\s:`]+)", re.M)
REJECTED = "mandatory input parameters are not present"  # ADK refuses the call; swegemma does not charge it


def _jsonl(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {r["instance_id"]: r for r in rows}


def _error_type(content) -> str:
    try:
        data = json.loads(content) if isinstance(content, str) else content
    except ValueError:
        return ""
    return str(data.get("error_type", "")) if isinstance(data, dict) else ""


def harness_warning_call(trace: dict) -> int | None:
    """Like `warning_call`, but counting only the calls the harness charges: not the ones ADK rejected for a
    missing mandatory parameter."""
    calls = 0
    for step in trace.get("steps", []):
        obs = str(step.get("observation") or "")
        for tc in step.get("tool_calls") or []:
            if tc.get("function_name") != "submit_patch" and tc.get("function_name") not in AGENT_TOOLS:
                calls += REJECTED not in obs
        if "budget_warning" in obs:
            return calls
    return None


def trace_metrics(trace: dict) -> dict:
    """Edit behavior from one ATIF trace: first successful edit by call number, failed edits, identical retries.

    An edit refused because the call budget is spent (BudgetExceeded) or because the path is outside /workspace
    (FileWriteError) is counted apart: neither is a failed match (correction of 19 Oct, doc 03 §14.33)."""
    calls = 0
    first_edit_call = None
    failed = repeats = after_budget = outside = 0
    prev_failed_args = None
    for step in trace.get("steps", []):
        for tc in step.get("tool_calls") or []:
            name = tc.get("function_name")
            if name == "submit_patch" or name in AGENT_TOOLS:
                continue
            calls += 1
            args = json.dumps(tc.get("arguments"), sort_keys=True)
            if prev_failed_args is not None and name in EDIT_TOOLS and args == prev_failed_args:
                repeats += 1
            prev_failed_args = None
            if name in EDIT_TOOLS:
                content = (step.get("observation") or {}).get("content")
                if _is_error(content):
                    kind = _error_type(content)
                    if kind == "BudgetExceeded":
                        after_budget += 1
                    elif kind == "FileWriteError":
                        outside += 1
                    else:
                        failed += 1
                        prev_failed_args = args
                elif first_edit_call is None:
                    first_edit_call = calls
    runs = [c for c in trace_commands(trace) if c["kind"] in ("pytest", "inline", "script")]
    turns = trace_turns(trace)
    return {**locator_metrics(trace), "max_turn_tokens": max((x["tokens"] for x in turns), default=0),
            "long_turn_seconds": [x["seconds"] for x in turns if x["tokens"] >= LONG and x["seconds"] is not None],
            "first_edit_call": first_edit_call, "failed_edits": failed, "repeats": repeats,
            "edits_after_budget": after_budget, "writes_outside": outside,
            "py_runs": len(runs), "wrapped": sum(c["wrapped"] for c in runs),
            "hangs": sum(c["hung"] for c in trace_commands(trace)), "cut_by_timeout": sum(c["cut"] for c in runs),
            "charged_calls": calls - sum(REJECTED in str(st.get("observation") or "") for st in trace.get("steps", [])
                                         for tc in st.get("tool_calls") or []
                                         if tc.get("function_name") != "submit_patch"
                                         and tc.get("function_name") not in AGENT_TOOLS),
            "warning_call": harness_warning_call(trace),
            "rejected_calls": sum(REJECTED in str(st.get("observation") or "") for st in trace.get("steps", []))}


def locator_metrics(trace: dict) -> dict:
    """The locator sub-agent of `bf-gov-ed` (doc 03 §14.38): whether it ran, its tool calls, and the first file of its
    plan (the `FILE:` line of its last reply). Zero and None for packages without a locator."""
    calls, plan = 0, None
    for step in trace.get("steps", []):
        if (step.get("extra") or {}).get("author") not in AGENT_TOOLS:
            continue
        calls += len(step.get("tool_calls") or [])
        m = PLAN_FILE.search(str(step.get("message") or ""))
        if m:
            plan = m.group(1).removeprefix("/workspace/")
    used = any(tc.get("function_name") in AGENT_TOOLS for st in trace.get("steps", []) for tc in st.get("tool_calls") or [])
    return {"locator_used": used, "locator_calls": calls, "plan_file": plan}


def task_row(iid: str, harness: dict, docker: dict, verifiable: dict[str, bool], results: Path) -> dict:
    safe = iid.replace("/", "__")
    patch = results / "patches" / f"{safe}.patch"
    files = re.findall(r"^diff --git a/(\S+)", patch.read_text(), re.M) if patch.exists() else []
    trace = results / "traces" / f"trace_{safe}.json"
    tm = trace_metrics(json.loads(trace.read_text())) if trace.exists() else {
        "first_edit_call": None, "failed_edits": 0, "repeats": 0, "edits_after_budget": 0, "writes_outside": 0,
        "py_runs": 0, "wrapped": 0, "hangs": 0, "cut_by_timeout": 0, "max_turn_tokens": 0, "long_turn_seconds": [],
        "locator_used": False, "locator_calls": 0, "plan_file": None, "charged_calls": 0, "warning_call": None,
        "rejected_calls": 0}
    is_verifiable = verifiable.get(iid, False)
    resolved = bool(docker[iid]["resolved"]) if is_verifiable and iid in docker else bool(harness.get("resolved"))
    return {"task": iid, "resolved": resolved, "harness_resolved": bool(harness.get("resolved")),
            "verifiable": is_verifiable, "minutes": float(harness.get("duration_seconds") or 0) / 60,
            "tool_calls": int(harness.get("tool_calls") or 0), "error": harness.get("error") or "",
            "aux": any(AUX_RE.search(f.split("/")[-1]) for f in files),
            "undeclared_tool": bool(UNDECLARED_TOOL.search(str(harness.get("error") or ""))), **tm}


def bucket(row: dict, max_calls: int) -> str:
    """Where the first successful edit falls in the call budget (doc 03 §14.38); `late` is the last 4 calls."""
    c = row["first_edit_call"]
    if c is None:
        return "refused_only" if row["edits_after_budget"] else "never"
    if c <= EDIT_BY_CALL:
        return "early"
    return "late" if c > max_calls - 4 else "mid"


LATE_OR_NEVER = ("late", "refused_only", "never")


def p90(xs: list[float]) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, math.ceil(0.9 * len(s)) - 1)] if s else 0.0


def arm_summary(rows: list[dict], max_calls: int) -> dict:
    mins = [r["minutes"] for r in rows]
    long_s = sorted(x for r in rows for x in r["long_turn_seconds"])
    return {"n": len(rows), "R": sum(r["resolved"] for r in rows),
            "R_verifiable": sum(r["resolved"] for r in rows if r["verifiable"]),
            "verifiable": sum(r["verifiable"] for r in rows),
            "edited": sum(r["first_edit_call"] is not None for r in rows),
            "edit_by_12": sum(r["first_edit_call"] is not None and r["first_edit_call"] <= EDIT_BY_CALL for r in rows),
            "mean_min": round(sum(mins) / len(mins), 2) if mins else 0.0, "p90_min": round(p90(mins), 2),
            "time_cuts": sum(r["error"].startswith("Agent exceeded") for r in rows),
            "session_timeouts": sum("session timeout" in r["error"] for r in rows),
            "call_cap": sum(r["tool_calls"] >= max_calls for r in rows),
            "aux": sum(r["aux"] for r in rows), "failed_edits": sum(r["failed_edits"] for r in rows),
            "repeats": sum(r["repeats"] for r in rows),
            "edits_after_budget": sum(r["edits_after_budget"] for r in rows),
            "writes_outside": sum(r["writes_outside"] for r in rows),
            "hangs": sum(r["hangs"] for r in rows),
            "wrapped_pct": round(100 * sum(r["wrapped"] for r in rows) / max(1, sum(r["py_runs"] for r in rows))),
            "cut_by_timeout": sum(r["cut_by_timeout"] for r in rows),
            "max_turn_tokens": max((r["max_turn_tokens"] for r in rows), default=0),
            "long_turns": len(long_s),
            "long_turn_s_p50": round(quantile(long_s, .5)) if long_s else None,
            "late_or_never": sum(bucket(r, max_calls) in LATE_OR_NEVER for r in rows),
            "undeclared": sum(r["undeclared_tool"] for r in rows),
            "plan_ok": sum(r["plan_file"] is not None for r in rows),
            "infra_errors": sum(bool(r["error"]) and not r["error"].startswith("Agent exceeded") for r in rows)}


def apply_rules(s: dict[str, dict]) -> list[str]:
    """Doc 03 §14.33 rules 1-4, as written on 17 Oct before any pool data."""
    out = []
    bad = [a for a, v in s.items() if v["infra_errors"] > 2]
    n = min(v["n"] for v in s.values())
    if bad or n < MIN_COMMON:
        return [f"Rule 1 (validity): FAIL ({'infra > 2: ' + ', '.join(bad) if bad else ''}"
                f"{'; ' if bad and n < MIN_COMMON else ''}{f'{n} common tasks < {MIN_COMMON}' if n < MIN_COMMON else ''})"
                " -> no rule applies"]
    out.append(f"Rule 1 (validity): OK, {n} common tasks")
    bf, h24, long = s["bf"], s["h24"], s["h24-long"]
    base = (h24["R"] >= bf["R"] - 1 and h24["aux"] <= bf["aux"] and h24["mean_min"] <= 1.10 * bf["mean_min"])
    out.append(f"Rule 2 (h24 base): {'YES' if base else 'NO'} | R {h24['R']} vs bf {bf['R']} - 1, aux {h24['aux']} vs "
               f"{bf['aux']}, min {h24['mean_min']} vs 1.10 x {bf['mean_min']} = {round(1.10 * bf['mean_min'], 2)}")
    margin = long["mean_min"] <= 1.12 * bf["mean_min"] and long["p90_min"] <= 9.2
    candidate = margin and long["R"] >= h24["R"]
    verdict = "CANDIDATE" if candidate else ("PROBE ONLY (user decides)" if margin else "NOT SENT (time margin)")
    out.append(f"Rule 3 (h24-long): {verdict} | min {long['mean_min']} vs 1.12 x {bf['mean_min']} = "
               f"{round(1.12 * bf['mean_min'], 2)}, p90 {long['p90_min']} <= 9.2, R {long['R']} vs h24 {h24['R']}")
    nxt = "h24-long" if candidate else ("h24" if base else "none")
    out.append(f"Rule 4 (next submission): {nxt}")
    return out


def apply_gov_rules(s: dict[str, dict]) -> list[str]:
    """Doc 03 §14.34 rules 1-3, written on 19 Oct before building `bf-gov`; thresholds as in §14.33 rule 2."""
    out = []
    bad = [a for a in ("bf", "bf-gov") if s[a]["infra_errors"] > 2]
    n = min(s[a]["n"] for a in ("bf", "bf-gov"))
    if bad or n < MIN_COMMON:
        return [f"Rule 1 (validity): FAIL ({', '.join(bad) or f'{n} common tasks < {MIN_COMMON}'}) -> no rule applies"]
    out.append(f"Rule 1 (validity): OK, {n} common tasks")
    bf, gov = s["bf"], s["bf-gov"]
    base = gov["R"] >= bf["R"] - 1 and gov["aux"] <= bf["aux"] and gov["mean_min"] <= 1.10 * bf["mean_min"]
    out.append(f"Rule 2 (bf-gov base): {'YES' if base else 'NO'} | R {gov['R']} vs bf {bf['R']} - 1, aux {gov['aux']} "
               f"vs {bf['aux']}, min {gov['mean_min']} vs 1.10 x {bf['mean_min']} = {round(1.10 * bf['mean_min'], 2)}")
    out.append(f"Rule 3 (next submission): {'bf-gov' if base else 'none'}")
    return out


def apply_to_rules(s: dict[str, dict], rows: dict[str, list[dict]]) -> list[str]:
    """Doc 03 §14.35 rules 1-4, written on 25 Oct before building `bf-gov-to`."""
    out = []
    bad = [a for a in ("bf-gov", "bf-gov-to") if s[a]["infra_errors"] > 2]
    n = min(s[a]["n"] for a in ("bf-gov", "bf-gov-to"))
    if bad or n < MIN_COMMON:
        return [f"Rule 1 (validity): FAIL ({', '.join(bad) or f'{n} common tasks < {MIN_COMMON}'}) -> no rule applies"]
    out.append(f"Rule 1 (validity): OK, {n} common tasks")
    gov, to = s["bf-gov"], s["bf-gov-to"]
    complies = to["wrapped_pct"] >= 80 and to["hangs"] == 0
    cut = sorted(r["task"] for r in rows["bf-gov-to"] if r["cut_by_timeout"])
    out.append(f"Rule 2 (compliance, no decision): {'MET' if complies else 'NOT MET'} | wrapped {to['wrapped_pct']}% "
               f">= 80, hangs {to['hangs']} == 0; cut by timeout in: {', '.join(cut) or 'none'}")
    base = (to["R"] >= gov["R"] - 1 and to["aux"] <= gov["aux"] + 1 and to["mean_min"] <= 1.10 * gov["mean_min"]
            and to["hangs"] <= gov["hangs"])
    out.append(f"Rule 3 (bf-gov-to base, GPU only): {'YES' if base else 'NO'} | R {to['R']} vs {gov['R']} - 1, aux "
               f"{to['aux']} vs {gov['aux']} + 1, min {to['mean_min']} vs 1.10 x {gov['mean_min']} = "
               f"{round(1.10 * gov['mean_min'], 2)}, hangs {to['hangs']} vs {gov['hangs']}")
    out.append(f"Rule 4 (next submission, once the scorer is fixed): {'bf-gov-to' if base else 'bf-gov'}")
    return out


TB_TASKS = ("rich_3521", "rich_3471", "rich_3938")


def apply_tb_rules(s: dict[str, dict], rows: dict[str, list[dict]]) -> list[str]:
    """Doc 03 §14.37 rules 1-4, written on 26 Oct before building `bf-gov-tb`. Without compliance (the 2048 budget
    not applied on GPU) the variant is not read."""
    out = []
    bad = [a for a in ("bf-gov-to", "bf-gov-tb") if s[a]["infra_errors"] > 2]
    n = min(s[a]["n"] for a in ("bf-gov-to", "bf-gov-tb"))
    if bad or n < MIN_COMMON:
        return [f"Rule 1 (validity): FAIL ({', '.join(bad) or f'{n} common tasks < {MIN_COMMON}'}) -> no rule applies"]
    out.append(f"Rule 1 (validity): OK, {n} common tasks")
    to, tb = s["bf-gov-to"], s["bf-gov-tb"]
    p50 = tb["long_turn_s_p50"]
    # Corrected on 26 Oct before any bf-gov-tb data: the "p50 < 70 s" half dropped (turns of 2000-2999 tokens
    # already take 69 s in the base traces), so compliance is read on tokens and the time is reported.
    complies = tb["max_turn_tokens"] <= 3000
    out.append(f"Rule 2 (compliance): {'MET' if complies else 'NOT MET'} | max turn {tb['max_turn_tokens']} <= 3000 "
               f"tokens; p50 of turns >= {LONG} tokens {p50} s over {tb['long_turns']} (bf-gov-to "
               f"{to['long_turn_s_p50']} s over {to['long_turns']}), reported")
    if not complies:
        return out + ["Rule 3 (bf-gov-tb base): NOT READ (rule 2 not met)"]
    base = (tb["R"] >= to["R"] - 1 and tb["aux"] <= to["aux"] + 1 and tb["mean_min"] <= 1.10 * to["mean_min"]
            and tb["session_timeouts"] <= to["session_timeouts"])
    out.append(f"Rule 3 (bf-gov-tb base): {'YES' if base else 'NO'} | R {tb['R']} vs {to['R']} - 1, aux {tb['aux']} "
               f"vs {to['aux']} + 1, min {tb['mean_min']} vs 1.10 x {to['mean_min']} = "
               f"{round(1.10 * to['mean_min'], 2)}, session timeouts {tb['session_timeouts']} vs {to['session_timeouts']}")
    mech = []
    for a in ("bf-gov-to", "bf-gov-tb"):
        sel = [r for r in rows[a] if r["task"] in TB_TASKS]
        mech.append(f"{a}: timeouts {sum('session timeout' in r['error'] for r in sel)}/{len(sel)}, first edit call "
                    + ", ".join(f"{r['task']}={r['first_edit_call']}" for r in sel))
    out.append("Rule 4 (mechanism, no decision): " + "; ".join(mech))
    return out


def apply_tb22_rules(s: dict[str, dict], rows: dict[str, list[dict]], noise: list[dict] | None) -> list[str]:
    """Doc 03 §14.42 rules 1-5, written on 2 Nov before building the `bf-gov-tb` / `bf-gov-tb-22` pool. The warning
    call is counted as the harness counts (calls ADK rejects for a missing parameter are not charged, §14.40)."""
    pair = ("bf-gov-tb", "bf-gov-tb-22")
    bad = [a for a in pair if s[a]["infra_errors"] > 2]
    n = min(s[a]["n"] for a in pair)
    if bad or n < MIN_COMMON:
        return [f"Rule 1 (validity): FAIL ({', '.join(bad) or f'{n} common tasks < {MIN_COMMON}'}) -> no rule applies"]
    out = [f"Rule 1 (validity): OK, {n} common tasks"]
    tb, c22 = s["bf-gov-tb"], s["bf-gov-tb-22"]
    warned = [r["warning_call"] for r in rows["bf-gov-tb-22"] if r["charged_calls"] >= 12]
    complies = c22["max_tool_calls"] == 22 and bool(warned) and all(w == 12 for w in warned)
    out.append(f"Rule 2 (compliance): {'MET' if complies else 'NOT MET'} | max_tool_calls {c22['max_tool_calls']}, "
               f"first budget_warning at harness call 12 in {sum(w == 12 for w in warned)}/{len(warned)} traces "
               f"with >= 12 charged calls")
    if not complies:
        out.append("Rule 3 (bf-gov-tb-22 base): NOT READ (rule 2 not met)")
    else:
        base = (c22["R"] >= tb["R"] - 1 and c22["aux"] <= tb["aux"] + 1 and c22["mean_min"] <= 1.10 * tb["mean_min"]
                and c22["session_timeouts"] <= tb["session_timeouts"])
        out.append(f"Rule 3 (bf-gov-tb-22 base): {'YES' if base else 'NO'} | R {c22['R']} vs {tb['R']} - 1, aux "
                   f"{c22['aux']} vs {tb['aux']} + 1, min {c22['mean_min']} vs 1.10 x {tb['mean_min']} = "
                   f"{round(1.10 * tb['mean_min'], 2)}, session timeouts {c22['session_timeouts']} vs "
                   f"{tb['session_timeouts']}")
    mech = []
    for a in pair:
        firsts = sorted(r["first_edit_call"] for r in rows[a] if r["first_edit_call"] is not None)
        mech.append(f"{a}: edit_by_12 {s[a]['edit_by_12']}, late_or_never {s[a]['late_or_never']}, first edit p50 "
                    f"{quantile(firsts, .5) if firsts else None}, rejected calls {sum(r['rejected_calls'] for r in rows[a])}, "
                    f"call cap {s[a]['call_cap']}")
    out.append("Rule 4 (mechanism, no decision): " + "; ".join(mech))
    if noise is not None:
        now = rows["bf-gov-tb"]
        flips = sorted(x["task"] for x, y in zip(now, noise) if x["resolved"] != y["resolved"])
        out.append(f"Rule 5 (noise, no decision): bf-gov-tb R {sum(r['resolved'] for r in now)} vs previous session "
                   f"{sum(r['resolved'] for r in noise)}, {len(flips)} tasks change ({', '.join(flips) or 'none'})")
    return out


def apply_ed_rules(s: dict[str, dict], rows: dict[str, list[dict]], base_arm: str,
                   gold: dict[str, set[str]]) -> list[str]:
    """Doc 03 §14.38 rules 1-4, written on 27 Oct before building `bf-gov-ed`. Without compliance of the structure
    (rule 2b) rule 3 is not read."""
    out = []
    pair = (base_arm, "bf-gov-ed")
    bad = [a for a in pair if s[a]["infra_errors"] > 2]
    n = min(s[a]["n"] for a in pair)
    ed, base = s["bf-gov-ed"], s[base_arm]
    if bad or n < MIN_COMMON or ed["undeclared"] > 1:
        why = ", ".join(bad) or (f"{n} common tasks < {MIN_COMMON}" if n < MIN_COMMON else f"undeclared {ed['undeclared']} > 1")
        return [f"Rule 1 (validity): FAIL ({why}) -> no rule applies"]
    out.append(f"Rule 1 (validity): OK, {n} common tasks, undeclared {ed['undeclared']}")
    a_met = ed["late_or_never"] <= base["late_or_never"] / 2 and ed["edited"] >= base["edited"]
    b_met = ed["plan_ok"] >= 0.8 * ed["n"]
    out.append(f"Rule 2a (compliance): {'MET' if a_met else 'NOT MET'} | late or never {ed['late_or_never']} <= "
               f"{base['late_or_never']} / 2, edited {ed['edited']} >= {base['edited']}")
    out.append(f"Rule 2b (structure): {'MET' if b_met else 'NOT MET'} | plan with FILE in {ed['plan_ok']}/{ed['n']} "
               f">= 80%")
    if not (a_met and b_met):
        out.append("Rule 3 (bf-gov-ed base): NOT READ (rule 2 not met)")
    else:
        ok = (ed["R"] >= base["R"] - 1 and ed["aux"] <= base["aux"] + 1 and ed["mean_min"] <= 1.10 * base["mean_min"]
              and ed["session_timeouts"] <= base["session_timeouts"])
        out.append(f"Rule 3 (bf-gov-ed base): {'YES' if ok else 'NO'} | R {ed['R']} vs {base['R']} - 1, aux {ed['aux']} "
                   f"vs {base['aux']} + 1, min {ed['mean_min']} vs 1.10 x {base['mean_min']} = "
                   f"{round(1.10 * base['mean_min'], 2)}, session timeouts {ed['session_timeouts']} vs "
                   f"{base['session_timeouts']}")
    sel = rows["bf-gov-ed"]
    calls = sorted(r["locator_calls"] for r in sel if r["locator_used"])
    hits = sum(r["plan_file"] in gold.get(r["task"], set()) for r in sel if r["plan_file"])
    firsts = {a: sorted(r["first_edit_call"] for r in rows[a] if r["first_edit_call"] is not None) for a in pair}
    out.append(f"Rule 4 (mechanism, no decision): locator used {len(calls)}/{len(sel)}, its calls p50 "
               f"{quantile(calls, .5) if calls else None} max {calls[-1] if calls else None}; plan file in gold "
               f"{hits}/{ed['plan_ok']}; first edit call p50 {base_arm} {quantile(firsts[base_arm], .5) if firsts[base_arm] else None}"
               f" vs bf-gov-ed {quantile(firsts['bf-gov-ed'], .5) if firsts['bf-gov-ed'] else None}; edits after budget "
               f"{base['edits_after_budget']} vs {ed['edits_after_budget']}; call cap {base['call_cap']} vs {ed['call_cap']}")
    return out


def gold_files() -> dict[str, set[str]]:
    f = ROOT / "data/competition/tasks.jsonl"
    if not f.exists():
        return {}
    rows = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
    return {r["instance_id"]: set(re.findall(r"^\+\+\+ b/(.+)$", r["patch"], re.M)) for r in rows}


def replica_noise(a: list[dict], b: list[dict]) -> str:
    """Same package, two sessions, same tasks: how often `resolved` agrees and how far the mean minutes move."""
    same = sum(x["resolved"] == y["resolved"] for x, y in zip(a, b))
    diff = sum(x["minutes"] for x in a) / len(a) - sum(y["minutes"] for y in b) / len(b)
    return f"bf vs bf-prev: same resolved on {same}/{len(a)} tasks, mean minutes differ by {diff:+.2f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True, help="name=results_dir,...")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    arms = {k: Path(v) for k, v in (item.split("=", 1) for item in args.arms.split(","))}
    verifiable = load_verifiable() or {}
    harness = {a: _jsonl(p / "task_results.jsonl") for a, p in arms.items()}
    docker = {a: _jsonl(p / "verified_task_results.jsonl") for a, p in arms.items()}
    common = sorted(set.intersection(*(set(h) for h in harness.values())))
    rows, summary = {}, {}
    for a, p in arms.items():
        max_calls = json.loads((p / "run_meta.json").read_text())["budgets"]["max_tool_calls"]
        rows[a] = [task_row(t, harness[a][t], docker[a], verifiable, p) for t in common]
        summary[a] = {"max_tool_calls": max_calls, **arm_summary(rows[a], max_calls)}
    cols = ["n", "R", "R_verifiable", "verifiable", "edited", "edit_by_12", "mean_min", "p90_min", "time_cuts",
            "call_cap", "aux", "failed_edits", "repeats", "edits_after_budget", "writes_outside", "hangs",
            "wrapped_pct", "cut_by_timeout", "session_timeouts", "max_turn_tokens", "long_turns", "long_turn_s_p50",
            "late_or_never", "plan_ok", "undeclared", "infra_errors"]
    lines = [f"# Pool report: {', '.join(arms)}", "", f"Common tasks: {len(common)}", "",
             "| arm | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    lines += [f"| {a} | " + " | ".join(str(summary[a][c]) for c in cols) + " |" for a in arms]
    if {"bf", "h24", "h24-long"} <= set(arms):
        lines += ["", "## Rules (doc 03 §14.33)", ""] + [f"- {r}" for r in apply_rules(summary)]
    if {"bf", "bf-gov"} <= set(arms):
        lines += ["", "## Rules (doc 03 §14.34)", ""] + [f"- {r}" for r in apply_gov_rules(summary)]
    if {"bf-gov", "bf-gov-to"} <= set(arms):
        lines += ["", "## Rules (doc 03 §14.35)", ""] + [f"- {r}" for r in apply_to_rules(summary, rows)]
    if {"bf-gov-to", "bf-gov-tb"} <= set(arms):
        lines += ["", "## Rules (doc 03 §14.37)", ""] + [f"- {r}" for r in apply_tb_rules(summary, rows)]
    if {"bf-gov-tb", "bf-gov-tb-22"} <= set(arms):
        lines += ["", "## Rules (doc 03 §14.42)", ""] + [
            f"- {r}" for r in apply_tb22_rules(summary, rows, rows.get("bf-gov-tb-prev"))]
    if "bf-gov-ed" in arms:
        base_arm = next((a for a in ("bf-gov-tb", "bf-gov-to") if a in arms), None)
        if base_arm:
            lines += ["", f"## Rules (doc 03 §14.38, base {base_arm})", ""] + [
                f"- {r}" for r in apply_ed_rules(summary, rows, base_arm, gold_files())]
    if {"bf", "bf-prev"} <= set(arms):
        lines += ["", f"- {replica_noise(rows['bf'], rows['bf-prev'])}"]
    lines += ["", "## Per task", "", "| task | " + " | ".join(arms) + " |", "|---" * (len(arms) + 1) + "|"]
    for i, t in enumerate(common):
        cells = [f"{'R' if rows[a][i]['resolved'] else '-'} {rows[a][i]['minutes']:.1f}m {rows[a][i]['tool_calls']}c"
                 for a in arms]
        lines.append(f"| {t} | " + " | ".join(cells) + " |")
    text = "\n".join(lines) + "\n"
    out = args.out or ROOT / "data/analysis" / f"pool_report_{'_'.join(arms)}.md"
    out.write_text(text)
    print(text)
    print("->", out)


if __name__ == "__main__":
    main()
