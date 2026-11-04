"""Long model turns in agent traces (lineage for doc 03 §14.37): how many completion tokens each turn generates, how
long it takes, and whether long turns go with tasks cut by the session timeout.

A turn's time is the gap between the previous observation and the turn's own timestamp, so it covers generation
and queueing in vLLM. Only GPU runs of the base family are read: TPU and older agents have other throughput and
prompts.

Usage: PYTHONPATH=lab/src python3 -m lab.analysis.long_turns [--eval data/eval] [--out data/analysis/long_turns.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lab.analysis.command_timeouts import SKIP, quantile

BASE_RUNS = ("kaggle-bf-traces", "kaggle-pool-bf", "kaggle-pool-h24", "pool-gov-gpu", "pool-to-gpu")
LONG = 2000  # completion tokens
CAPPED = 4000  # at the thinking budget of bf (4096) plus the visible text


def trace_turns(trace: dict) -> list[dict]:
    """Each model turn: completion tokens, seconds, and how many tool calls the trace had made before it."""
    out = []
    last = None
    calls = 0
    for step in trace.get("steps", []):
        t = (step.get("extra") or {}).get("elapsed_s")
        if step.get("source") == "agent" and step.get("metrics"):
            out.append({"tokens": step["metrics"].get("completion_tokens", 0), "calls_before": calls,
                        "seconds": t - last if t is not None and last is not None else None})
        calls += len(step.get("tool_calls") or [])
        obs = step.get("observation")
        t_obs = ((obs or {}).get("extra") or {}).get("elapsed_s") if isinstance(obs, dict) else None
        last = t_obs if t_obs is not None else t if t is not None else last
    return out


def trace_row(turns: list[dict], result: dict) -> dict:
    long = [x for x in turns if x["tokens"] >= LONG]
    secs = sum(x["seconds"] or 0 for x in turns)
    long_secs = sum(x["seconds"] or 0 for x in long)
    return {"turns": len(turns), "long": len(long), "capped": sum(x["tokens"] >= CAPPED for x in turns), "tokens": sum(x["tokens"] for x in turns),
            "gen_seconds": round(secs, 1), "long_seconds": round(long_secs, 1),
            "long_before_first_call_5": sum(x["calls_before"] < 5 for x in long),
            "duration": result.get("duration_seconds"), "resolved": bool(result.get("resolved")),
            "timed_out": "session timeout" in str(result.get("error"))}


def _results(run_dir: Path) -> dict[str, dict]:
    f = run_dir / "task_results.jsonl"
    rows = [json.loads(line) for line in f.read_text().splitlines() if line.strip()] if f.exists() else []
    return {r["instance_id"]: r for r in rows}


def _group(rows: list[dict]) -> dict:
    n = len(rows) or 1
    dur = sum(r["duration"] or 0 for r in rows) or 1
    return {"traces": len(rows), "with_long": sum(r["long"] > 0 for r in rows),
            "long_per_trace": round(sum(r["long"] for r in rows) / n, 2),
            "long_share_of_time": round(sum(r["long_seconds"] for r in rows) / dur, 3),
            "gen_share_of_time": round(sum(r["gen_seconds"] for r in rows) / dur, 3),
            "timed_out": sum(r["timed_out"] for r in rows), "resolved": sum(r["resolved"] for r in rows)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", type=Path, default=Path("data/eval"))
    ap.add_argument("--out", type=Path, default=Path("data/analysis/long_turns.json"))
    a = ap.parse_args()
    rows, turns_all = [], []
    for path in sorted(a.eval.glob("**/traces/trace_*.json")):
        run = str(path.parent.parent.relative_to(a.eval))
        if any(s in str(path) for s in SKIP) or not run.startswith(BASE_RUNS):
            continue
        task = path.stem.removeprefix("trace_")
        turns = trace_turns(json.loads(path.read_text()))
        turns_all += turns
        row = trace_row(turns, _results(path.parent.parent).get(task, {}))
        row.update(run=run, task=task)
        rows.append(row)
    tokens = sorted(x["tokens"] for x in turns_all)
    timed = [x for x in turns_all if x["seconds"] is not None and x["tokens"] >= LONG]
    tps = sum(x["tokens"] for x in timed) / (sum(x["seconds"] for x in timed) or 1)
    report = {
        "traces": len(rows), "turns": len(turns_all),
        "turn_tokens": {"p50": quantile(tokens, .5), "p90": quantile(tokens, .9), "p99": quantile(tokens, .99),
                        "max": tokens[-1], f"ge_{LONG}": sum(x >= LONG for x in tokens),
                        "ge_3000": sum(x >= 3000 for x in tokens), "ge_4000": sum(x >= 4000 for x in tokens)},
        "long_tokens_per_second": round(tps, 1),
        "timed_out": _group([r for r in rows if r["timed_out"]]),
        "finished": _group([r for r in rows if not r["timed_out"]]),
        "capped_seconds_p50": quantile(sorted(x["seconds"] for x in turns_all
                                              if x["tokens"] >= CAPPED and x["seconds"] is not None), .5),
        "with_capped": _group([r for r in rows if r["capped"]]),
        "without_capped": _group([r for r in rows if not r["capped"]]),
        "long_before_call_5": sum(r["long_before_first_call_5"] for r in rows),
        "top_traces": sorted(({k: r[k] for k in ("run", "task", "long", "long_seconds", "duration", "timed_out",
                                                  "resolved")} for r in rows),
                             key=lambda r: -r["long_seconds"])[:15],
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
