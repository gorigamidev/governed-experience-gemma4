"""Self-imitation inside an episode (doc 03 §14 item 56): does a behavior the 31B shows early come back later in the
same run? Q1: a test run before the first edit vs a test run after the last edit. Q2: a test run after the last edit
vs resolved, on verifiable tasks. Q3 (descriptive): the first command's form and identical consecutive calls.

The rules (S1, S2 and the decision) were written in doc 03 §56 before any number was computed.

Usage: PYTHONPATH=lab/src python3 -m lab.analysis.self_imitation [--out data/analysis/self_imitation.md]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lab.analysis.pool_report import REJECTED, TEST_RE, after_edit_checks
from lab.ingest.load_traces import EDIT_TOOLS, _is_error

EVAL = Path("data/eval")
CORPUS = (
    "kaggle-bf-traces", "kaggle-pool-bf", "kaggle-pool-h24", "kaggle-pool-h24-long",
    "pool-gov-gpu/results/bf", "pool-gov-gpu/results/bf-gov",
    "pool-to-gpu/results/bf-gov", "pool-to-gpu/results/bf-gov-to",
    "pool-tb-gpu/results/bf-gov-tb", "pool-tb-gpu/results/bf-gov-to",
    "pool-tb22-gpu/results/bf-gov-tb", "pool-tb22-gpu/results/bf-gov-tb-22",
    "pool-tbslec-gpu/results/bf-gov-tb", "pool-tbslec-gpu/results/bf-gov-tb-ec", "pool-tbslec-gpu/results/bf-gov-tb-sl",
    "pool-tont-gpu/results/bf-gov-to",
)
S1_DIFF, S1_MIN_N = 0.25, 15
S2_DIFF = 0.15
TASK_SHARE, TASK_MIN = 2 / 3, 6


def _calls(trace: dict) -> list[tuple[str, dict, object]]:
    return [(tc.get("function_name"), tc.get("arguments") or {}, (st.get("observation") or {}).get("content"))
            for st in trace.get("steps", []) for tc in st.get("tool_calls") or []]


def _cmd(name: str, args) -> str | None:
    return str(args.get("command", "")) if name == "run_command" and isinstance(args, dict) else None


def _repeats(keys: list) -> tuple[list[bool], list[bool]]:
    same = [keys[i] == keys[i - 1] for i in range(1, len(keys))]
    return ([same[i] for i in range(1, len(same)) if same[i - 1]],
            [same[i] for i in range(1, len(same)) if not same[i - 1]])


def run_features(trace: dict) -> dict:
    """Per run: whether it edited, the call number of the first edit, a test before the first edit (early) and after
    the last one (final), whether the first command is wrapped in `timeout`, and identical consecutive calls."""
    calls = _calls(trace)
    edits = [i for i, (n, _, c) in enumerate(calls) if n in EDIT_TOOLS and not _is_error(c)]
    first = edits[0] if edits else None
    cmds = [(i, c) for i, (n, a, _) in enumerate(calls) if (c := _cmd(n, a)) is not None]
    early = any(TEST_RE.search(c) for i, c in cmds if first is not None and i < first) if first is not None else None
    keys = [(n, json.dumps(a, sort_keys=True)) for n, a, _ in calls]
    after_same, after_diff = _repeats(keys)
    # Review (Nov 11): without the calls ADK rejects uncharged, which inflate the repetition
    v_same, v_diff = _repeats([k for k, (_, _, c) in zip(keys, calls) if REJECTED not in str(c)])
    return {"edited": first is not None, "first_edit": first + 1 if first is not None else None,
            "early_test": early, "final_test": after_edit_checks(trace)["test_after_edit"],
            "first_cmd_timeout": cmds[0][1].lstrip().startswith("timeout") if cmds else None,
            "later_cmd_timeout": [c.lstrip().startswith("timeout") for _, c in cmds[1:]],
            "same_after_same": after_same, "same_after_diff": after_diff,
            "valid_same_after_same": v_same, "valid_same_after_diff": v_diff}


def load_rows(eval_dir: Path, verifiable: dict[str, bool], corpus=CORPUS) -> list[dict]:
    rows = []
    for src in corpus:
        run = eval_dir / src
        res = run / "task_results.jsonl"
        if not res.exists():
            raise FileNotFoundError(res)
        for line in res.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            iid = r["instance_id"]
            tf = run / "traces" / f"trace_{iid}.json"
            if not tf.exists():
                continue
            rows.append({"source": src, "task": iid, "resolved": bool(r.get("resolved")),
                         "verifiable": verifiable.get(iid, False), **run_features(json.loads(tf.read_text()))})
    return rows


def compare(rows: list[dict], group: str, outcome: str) -> dict:
    """Pooled outcome rate by group (True vs False) and the per-task sign count over tasks with runs in both groups."""
    g1 = [r for r in rows if r[group] is True]
    g0 = [r for r in rows if r[group] is False]
    rate = lambda rs: sum(bool(r[outcome]) for r in rs) / len(rs) if rs else 0.0
    pos = neg = ties = 0
    for t in sorted({r["task"] for r in rows}):
        a = [r for r in g1 if r["task"] == t]
        b = [r for r in g0 if r["task"] == t]
        if not a or not b:
            continue
        d = rate(a) - rate(b)
        pos, neg, ties = pos + (d > 0), neg + (d < 0), ties + (d == 0)
    return {"n1": len(g1), "n0": len(g0), "rate1": rate(g1), "rate0": rate(g0), "diff": rate(g1) - rate(g0),
            "tasks_pos": pos, "tasks_neg": neg, "tasks_tie": ties}


def _within_task_ok(c: dict) -> bool:
    nz = c["tasks_pos"] + c["tasks_neg"]
    return nz >= TASK_MIN and c["tasks_pos"] >= TASK_SHARE * nz


def signals(rows: list[dict]) -> dict:
    edited = [r for r in rows if r["edited"]]
    q1 = compare(edited, "early_test", "final_test")
    s1 = q1["diff"] >= S1_DIFF and min(q1["n1"], q1["n0"]) >= S1_MIN_N and _within_task_ok(q1)
    ver = [r for r in edited if r["verifiable"]]
    q2 = compare(ver, "final_test", "resolved")
    s2 = q2["diff"] >= S2_DIFF and _within_task_ok(q2)
    if s1 and s2:
        decision = "S1 and S2: design a variant that seeds the behavior early on bf-gov-to (own rules, preflight)"
    elif s1:
        decision = "S1 only: imitation without association with resolving; paper/notes.md, no variant"
    elif s2:
        decision = "S2 only: value without imitation; asking for it was tested (item 55); paper/notes.md"
    else:
        decision = "neither: archived, negative result to paper/notes.md"
    return {"q1": q1, "s1": s1, "q2": q2, "s2": s2, "decision": decision}


def _fmt(c: dict) -> str:
    return (f"{c['rate1']:.0%} (n={c['n1']}) vs {c['rate0']:.0%} (n={c['n0']}), diff {c['diff'] * 100:+.0f} points; "
            f"per task +{c['tasks_pos']} / -{c['tasks_neg']} / ={c['tasks_tie']}")


def report(rows: list[dict]) -> str:
    sig = signals(rows)
    edited = [r for r in rows if r["edited"]]
    ver = [r for r in edited if r["verifiable"]]
    out = ["# Self-imitation inside the episode (doc 03 §14 item 56)", "",
           f"Runs: {len(rows)} from {len(CORPUS)} sources; with an edit: {len(edited)}; with an edit on verifiable tasks: "
           f"{len(ver)}.", "", "## Rules", "",
           f"- Q1 (test before the first edit -> test after the last edit): {_fmt(sig['q1'])}",
           f"- S1 (>= {S1_DIFF * 100:.0f} points, >= {S1_MIN_N} per group, per task >= 2/3 of >= {TASK_MIN} non-tied): "
           f"{'MET' if sig['s1'] else 'NOT MET'}",
           f"- Q2 (test after the last edit -> resolved, verifiable): {_fmt(sig['q2'])}",
           f"- S2 (>= {S2_DIFF * 100:.0f} points, per task >= 2/3 of >= {TASK_MIN} non-tied): "
           f"{'MET' if sig['s2'] else 'NOT MET'}",
           f"- Decision: {sig['decision']}", "", "## Descriptive (no decision)", ""]
    fe = sorted(r["first_edit"] for r in ver)
    if fe:
        med = fe[len(fe) // 2]
        for name, part in (("first edit <= " + str(med), [r for r in ver if r["first_edit"] <= med]),
                           ("first edit > " + str(med), [r for r in ver if r["first_edit"] > med])):
            out.append(f"- Q2, {name}: {_fmt(compare(part, 'final_test', 'resolved'))}")
    t1 = [x for r in rows if r["first_cmd_timeout"] is True for x in r["later_cmd_timeout"]]
    t0 = [x for r in rows if r["first_cmd_timeout"] is False for x in r["later_cmd_timeout"]]
    share = lambda xs: f"{sum(xs) / len(xs):.0%} of {len(xs)}" if xs else "n/a"
    out.append(f"- Q3, later commands with `timeout`: first command wrapped {share(t1)}; first command not wrapped "
               f"{share(t0)}")
    ss = [x for r in rows for x in r["same_after_same"]]
    sd = [x for r in rows for x in r["same_after_diff"]]
    out.append(f"- Q3, identical consecutive call: after an identical pair {share(ss)}; after a different pair {share(sd)}")
    vs = [x for r in rows for x in r["valid_same_after_same"]]
    vd = [x for r in rows for x in r["valid_same_after_diff"]]
    out.append(f"- Q3, same without calls rejected uncharged by ADK: after an identical pair {share(vs)}; after a different pair {share(vd)}")
    by_src = {}
    for r in edited:
        by_src.setdefault(r["source"], []).append(r)
    out += ["", "| source | runs with edit | early test | final test | final test given early | given no early |",
            "|---|---|---|---|---|---|"]
    for src, rs in by_src.items():
        e1 = [r for r in rs if r["early_test"]]
        e0 = [r for r in rs if not r["early_test"]]
        f = lambda xs: f"{sum(r['final_test'] for r in xs)}/{len(xs)}"
        out.append(f"| {src} | {len(rs)} | {sum(r['early_test'] for r in rs)} | {f(rs)} | {f(e1)} | {f(e0)} |")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", type=Path, default=EVAL)
    ap.add_argument("--verifiable", type=Path, default=Path("data/analysis/verifiable_tasks.json"))
    ap.add_argument("--out", type=Path, default=Path("data/analysis/self_imitation.md"))
    a = ap.parse_args()
    text = report(load_rows(a.eval, json.loads(a.verifiable.read_text())))
    a.out.write_text(text)
    print(text)
    print("->", a.out)


if __name__ == "__main__":
    main()
