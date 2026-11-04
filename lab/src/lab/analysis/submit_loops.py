"""Repeated `submit_patch` calls in agent traces (lineage for doc 03 §14.36): how often the model keeps submitting,
whether the patch is empty, whether a later submit recovers, and what the calls after the first submit cost.

`submit_patch` does not end the episode and does not count against the call budget, so a model that keeps
calling it only spends wall time and prompt tokens. Copies of the same run and local or rehearsal runs are skipped,
as in `command_timeouts`.

Usage: PYTHONPATH=lab/src python3 -m lab.analysis.submit_loops [--eval data/eval] [--out data/analysis/submit_loops.json]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from lab.analysis.command_timeouts import SKIP

LOOP = 3  # consecutive submits that count as a loop


def _patch_size(content) -> int | None:
    try:
        data = json.loads(content) if isinstance(content, str) else content
    except ValueError:
        return None
    return data.get("patch_size") if isinstance(data, dict) else None


def trace_submits(trace: dict) -> dict:
    """Submit behavior of one ATIF trace."""
    sizes: list[int | None] = []
    run = max_run = calls_after = 0
    after_first = budget_before = False
    t_first = t_last = None
    tokens_after = 0
    for step in trace.get("steps", []):
        obs = step.get("observation") or {}
        content = obs.get("content") if isinstance(obs, dict) else None
        names = [tc.get("function_name") for tc in step.get("tool_calls") or []]
        if after_first:
            tokens_after += (step.get("metrics") or {}).get("prompt_tokens", 0)
            calls_after += len(names)
        for name in names:
            if name == "submit_patch":
                sizes.append(_patch_size(content))
                run += 1
                max_run = max(max_run, run)
                if not after_first:
                    after_first = True
                    t_first = (obs.get("extra") or {}).get("elapsed_s")
            else:
                run = 0
                if not after_first and "BudgetExceeded" in str(content):
                    budget_before = True
        if isinstance(obs, dict) and (obs.get("extra") or {}).get("elapsed_s") is not None:
            t_last = obs["extra"]["elapsed_s"]
    first_empty = next((i for i, s in enumerate(sizes) if s == 0), None)
    recovered = first_empty is not None and any(s for s in sizes[first_empty + 1:])
    return {"submits": len(sizes), "empty_submits": sum(s == 0 for s in sizes), "max_run": max_run,
            "last_empty": bool(sizes) and sizes[-1] == 0, "recovered": recovered,
            "resubmit_after_patch": any(s for s in sizes[:-1]), "budget_before_first": budget_before,
            "calls_after_first": calls_after,
            "seconds_after_first": t_last - t_first if t_first is not None and t_last is not None else None,
            "prompt_tokens_after_first": tokens_after}


def _resolved(path: Path) -> dict[str, bool]:
    f = path.parent.parent / "task_results.jsonl"
    if not f.exists():
        return {}
    rows = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
    return {r["instance_id"]: bool(r.get("resolved")) for r in rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", type=Path, default=Path("data/eval"))
    ap.add_argument("--out", type=Path, default=Path("data/analysis/submit_loops.json"))
    a = ap.parse_args()
    rows = []
    resolved_cache: dict[Path, dict[str, bool]] = {}
    for path in sorted(a.eval.glob("**/traces/trace_*.json")):
        if any(s in str(path) for s in SKIP):
            continue
        task = path.stem.removeprefix("trace_")
        res = resolved_cache.setdefault(path.parent, _resolved(path))
        row = trace_submits(json.loads(path.read_text()))
        row.update(task=task, run=str(path.parent.parent.relative_to(a.eval)), resolved=res.get(task))
        rows.append(row)
    loops = [r for r in rows if r["max_run"] >= LOOP]
    empty = [r for r in rows if r["empty_submits"]]
    runs = Counter(r["max_run"] for r in rows)
    report = {
        "traces": len(rows),
        "with_submit": sum(r["submits"] > 0 for r in rows),
        "without_submit": sum(r["submits"] == 0 for r in rows),
        "with_empty_submit": len(empty),
        "empty_then_recovered": sum(r["recovered"] for r in empty),
        "empty_then_recovered_resolved": sum(bool(r["recovered"] and r["resolved"]) for r in empty),
        "resubmit_after_patch": sum(r["resubmit_after_patch"] for r in rows),
        "max_run_hist": {str(k): runs[k] for k in sorted(runs)},
        f"loops_ge_{LOOP}": len(loops),
        "loops_ge_5": sum(r["max_run"] >= 5 for r in rows),
        "loops_ge_10": sum(r["max_run"] >= 10 for r in rows),
        "loops_last_empty": sum(r["last_empty"] for r in loops),
        "loops_budget_before_first": sum(r["budget_before_first"] for r in loops),
        "loops_resolved": sum(bool(r["resolved"]) for r in loops),
        "loop_cases": [{k: r[k] for k in ("run", "task", "submits", "empty_submits", "max_run", "recovered",
                                           "budget_before_first", "resolved", "seconds_after_first",
                                           "prompt_tokens_after_first")}
                       for r in sorted(loops, key=lambda r: -r["max_run"])],
        "seconds_after_first_all": round(sum(r["seconds_after_first"] or 0 for r in rows), 1),
        "seconds_after_first_loops": round(sum(r["seconds_after_first"] or 0 for r in loops), 1),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
