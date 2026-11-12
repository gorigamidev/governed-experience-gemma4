"""Hidden thinking with `enable_thinking: False` (lineage for doc 03 §14 item 55 and the paper).

A model turn is hidden when it generates more than LONG completion tokens and less than half of them can be seen in
the trace: the visible characters (message plus tool-call arguments) are fewer than VISIBLE_RATIO x tokens x ~4
characters per token, i.e. fewer than 2 x tokens. The report gives the sensitivity to both thresholds.

Usage: PYTHONPATH=lab/src python3 -m lab.analysis.hidden_thinking [--run data/eval/pool-tont-gpu/results]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

LONG = 800
CHARS_PER_TOKEN = 4
VISIBLE_RATIO = 0.5
ARMS = ("bf-gov-to", "bf-gov-to-nt", "bf-gov-to-nt-vr")


def turns(trace: dict) -> list[tuple[int, int]]:
    """(completion tokens, visible characters) for each model turn with metrics."""
    out = []
    for s in trace.get("steps", []):
        if s.get("source") != "agent" or not s.get("metrics"):
            continue
        vis = len(str(s.get("message") or "")) + sum(len(json.dumps(tc.get("arguments", {}))) for tc in s.get("tool_calls") or [])
        out.append((s["metrics"].get("completion_tokens", 0), vis))
    return out


def is_hidden(tokens: int, visible: int, long: int = LONG, ratio: float = VISIBLE_RATIO) -> bool:
    return tokens > long and visible < ratio * CHARS_PER_TOKEN * tokens


def summarize(all_turns: list[tuple[int, int]], long: int = LONG, ratio: float = VISIBLE_RATIO) -> dict:
    total = sum(t for t, _ in all_turns)
    hidden = [t for t, v in all_turns if is_hidden(t, v, long, ratio)]
    return {"turns": len(all_turns), "tokens": total, "hidden_turns": len(hidden), "hidden_tokens": sum(hidden),
            "hidden_share": round(sum(hidden) / total, 3) if total else 0.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=Path("data/eval/pool-tont-gpu/results"))
    ap.add_argument("--out", type=Path, default=Path("data/analysis/hidden_thinking.md"))
    a = ap.parse_args()
    per_arm = {arm: [x for f in sorted((a.run / arm / "traces").glob("trace_*.json")) for x in turns(json.loads(f.read_text()))]
               for arm in ARMS}
    lines = ["# Hidden thinking with `enable_thinking: False` (doc 03 §14 item 55)", "",
             f"Hidden turn: > {LONG} completion tokens and visible characters (message + tool-call arguments) "
             f"< {VISIBLE_RATIO} x {CHARS_PER_TOKEN} x tokens.", "",
             "| arm | turns | tokens | hidden turns | hidden tokens | hidden share |", "|---|---|---|---|---|---|"]
    for arm, ts in per_arm.items():
        s = summarize(ts)
        lines.append(f"| {arm} | {s['turns']} | {s['tokens']} | {s['hidden_turns']} | {s['hidden_tokens']} | {s['hidden_share']:.0%} |")
    lines += ["", "## Sensitivity (hidden turns / share of tokens)", "", "| long | visible ratio | " + " | ".join(ARMS) + " |",
              "|---|---|" + "---|" * len(ARMS)]
    for long in (800, 2000):
        for ratio in (0.25, 0.5, 1.0):
            cells = [summarize(per_arm[arm], long, ratio) for arm in ARMS]
            lines.append(f"| {long} | {ratio} | " + " | ".join(f"{c['hidden_turns']} / {c['hidden_share']:.0%}" for c in cells) + " |")
    a.out.write_text("\n".join(lines) + "\n")
    print(a.out.read_text())


if __name__ == "__main__":
    main()
