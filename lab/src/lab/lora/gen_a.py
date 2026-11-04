"""Data A (doc 03 §14.25): good turns taken from our own scorer runs.

The script counts, per run and task, the turns that §14.25 keeps (first `submit_patch`, a `DONE` added when
cutting, and the calls that do not immediately repeat the previous one). Reserve tasks (§14.24) are excluded
here and never enter the manifest (METHOD.md rule 2).

This module does not reconstruct chat requests: it records the selection (run, task, number of good turns) so
the turn-level filtering can be audited independently of how the prompts are later rendered. Usage:

    PYTHONPATH=lab/src python3 -m lab.lora.gen_a [--out build/scratch/a-selection.json]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from lab.lora import scripted as S

ROOT = Path(__file__).resolve().parents[4]

DEFAULT_RUNS = list(S.NEW_SCORER_RUNS)


def is_reserve(iid: str) -> bool:
    return iid in S.RESERVE


def repo_of(iid: str) -> str:
    return iid.split("_", 1)[0]


def _sig(step: dict) -> tuple:
    calls = step.get("tool_calls") or []
    return tuple(sorted((c.get("function_name"), json.dumps(c.get("arguments", {}), sort_keys=True))
                 for c in calls))


def good_turn_count(trace: dict) -> tuple[int, bool]:
    """Return (number of kept turns, whether a first submit_patch was seen)."""
    kept = 0
    prev = None
    submitted = False
    for step in trace.get("steps", []):
        if step.get("source") != "agent":
            continue
        calls = step.get("tool_calls") or []
        if not calls:
            continue
        sig = _sig(step)
        if sig == prev:
            continue
        kept += 1
        prev = sig
        if any(c.get("function_name") == "submit_patch" for c in calls):
            submitted = True
            break
    if submitted:
        kept += 1  # DONE added when cutting the rest of the trajectory
    return kept, submitted


def _run_resolved_maps(base: Path, run: str) -> tuple[dict[str, bool], dict[str, bool]]:
    rd = base / run
    verified = {}
    vf = rd / "verified_task_results.jsonl"
    if vf.is_file():
        for line in vf.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            verified[row["instance_id"]] = bool(row.get("resolved", False))
    harness = {}
    tf = rd / "task_results.jsonl"
    if tf.is_file():
        for line in tf.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            harness[row["instance_id"]] = bool(row.get("resolved", False))
    return verified, harness


def _thinking_of(run: str, base: Path) -> tuple[bool, str]:
    rd = base / run
    rm = rd / "run_meta.json"
    if rm.is_file():
        try:
            meta = json.loads(rm.read_text())
        except (json.JSONDecodeError, OSError):
            meta = {}
        for key in ("coder_thinking", "coder_enable_thinking", "enable_thinking_coder",
                    "thinking_enabled", "thinking"):
            if key in meta:
                return bool(meta[key]), f"run_meta.json:{key}"
        label = meta.get("label")
        if isinstance(label, str) and (label.endswith("-think") or label.endswith("_think")):
            return True, "run_meta.json label (ends with -think)"
        if isinstance(label, str) and (label.endswith("-think-off") or label.endswith("_think_off")):
            return False, "run_meta.json label (ends with -think-off)"
        if meta.get("ollama_thinking_off") is True:
            return False, "run_meta.json ollama_thinking_off=true"
    if run.endswith("-think") or run.endswith("_think"):
        return True, "run name (ends with -think)"
    if run.endswith("-think-off") or run.endswith("_think_off"):
        return False, "run name (ends with -think-off)"
    # Unlabeled runs (e.g. kaggle-bf-traces): thinking if most traces carry thinking events.
    traces = sorted((rd / "traces").glob("trace_*.json")) if (rd / "traces").is_dir() else []
    with_events = 0
    for tf in traces:
        try:
            steps = json.loads(tf.read_text()).get("steps") or []
        except (json.JSONDecodeError, OSError):
            continue
        with_events += any((st.get("extra") or {}).get("event_type") == "thinking" for st in steps)
    if traces and with_events * 2 > len(traces):
        return True, f"trace thinking events ({with_events}/{len(traces)})"
    return False, "default (no thinking marker)"


def collect(run_names: list[str], base: Path = ROOT / "data/eval") -> list[dict]:
    out = []
    for run in run_names:
        rd = base / run
        verified, harness = _run_resolved_maps(base, run)
        thinking, thinking_source = _thinking_of(run, base)
        traces_dir = rd / "traces"
        if not traces_dir.is_dir():
            continue
        for tf in sorted(traces_dir.glob("trace_*.json")):
            iid = tf.stem.replace("trace_", "")
            if is_reserve(iid):
                continue
            try:
                trace = json.loads(tf.read_text())
            except Exception:
                continue
            turns, submitted = good_turn_count(trace)
            if turns == 0:
                continue
            if iid in verified:
                resolved = verified[iid]
                resolved_source = "verified"
            elif iid in harness:
                resolved = harness[iid]
                resolved_source = "harness"
            else:
                resolved = False
                resolved_source = "none"
            out.append({"run": run, "task": iid, "repo": repo_of(iid), "good_turns": turns,
                        "submitted": submitted, "resolved": resolved,
                        "resolved_source": resolved_source, "resolved_from": resolved_source,
                        "thinking": thinking, "thinking_source": thinking_source,
                        "thinking_from": thinking_source})
    return out


def summarize(selection: list[dict]) -> dict:
    tasks = {s["task"] for s in selection}
    resolved_trajectories = sum(1 for s in selection if s.get("resolved"))
    thinking_trajectories = sum(1 for s in selection if s.get("thinking"))

    def split_count(key: str, want: bool) -> int:
        return sum(1 for s in selection if bool(s.get(key)) == want)

    def split_tasks(key: str, want: bool) -> int:
        return len({s["task"] for s in selection if bool(s.get(key)) == want})

    def split_turns(key: str, want: bool) -> int:
        return sum(s["good_turns"] for s in selection if bool(s.get(key)) == want)

    summary = {
        "runs": len({s["run"] for s in selection}),
        "trajectories": len(selection),
        "unique_tasks": len(tasks),
        "good_turns": sum(s["good_turns"] for s in selection),
        "submitted_trajectories": sum(1 for s in selection if s["submitted"]),
        "unique_tasks_by_repo": dict(Counter(repo_of(t) for t in tasks)),
        "good_turns_by_repo": dict(Counter(s["repo"] for s in selection for _ in range(s["good_turns"]))),
        "resolved_trajectories": resolved_trajectories,
        "unresolved_trajectories": len(selection) - resolved_trajectories,
        "thinking_trajectories": thinking_trajectories,
        "no_thinking_trajectories": len(selection) - thinking_trajectories,
        "unique_tasks_by_resolved": {
            "true": split_tasks("resolved", True),
            "false": split_tasks("resolved", False),
        },
        "good_turns_by_resolved": {
            "true": split_turns("resolved", True),
            "false": split_turns("resolved", False),
        },
        "trajectories_by_resolved": {
            "true": split_count("resolved", True),
            "false": split_count("resolved", False),
        },
        "unique_tasks_by_thinking": {
            "true": split_tasks("thinking", True),
            "false": split_tasks("thinking", False),
        },
        "good_turns_by_thinking": {
            "true": split_turns("thinking", True),
            "false": split_turns("thinking", False),
        },
        "trajectories_by_thinking": {
            "true": split_count("thinking", True),
            "false": split_count("thinking", False),
        },
    }
    summary["tasks_by_resolved"] = summary["unique_tasks_by_resolved"]
    summary["turns_by_resolved"] = summary["good_turns_by_resolved"]
    summary["tasks_by_thinking"] = summary["unique_tasks_by_thinking"]
    summary["turns_by_thinking"] = summary["good_turns_by_thinking"]
    summary["resolvedTrajectories"] = resolved_trajectories
    summary["thinkingTrajectories"] = thinking_trajectories
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=",".join(DEFAULT_RUNS))
    ap.add_argument("--base", type=Path, default=ROOT / "data/eval")
    ap.add_argument("--out", type=Path, default=ROOT / "build/scratch" / "a-selection.json")
    a = ap.parse_args()
    selection = collect([r for r in a.runs.split(",") if r], base=a.base)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"selection": selection, "summary": summarize(selection)}, indent=1))
    s = summarize(selection)
    print(f"runs={s['runs']} trajectories={s['trajectories']} unique_tasks={s['unique_tasks']} "
          f"good_turns={s['good_turns']} submitted={s['submitted_trajectories']}")
    print(f"unique_tasks_by_repo={s['unique_tasks_by_repo']}")
    print(f"good_turns_by_repo={s['good_turns_by_repo']}")
    print(f"resolved_trajectories={s['resolved_trajectories']} thinking_trajectories={s['thinking_trajectories']}")
    print(f"trajectories_by_resolved={s['trajectories_by_resolved']}")
    print(f"trajectories_by_thinking={s['trajectories_by_thinking']}")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
