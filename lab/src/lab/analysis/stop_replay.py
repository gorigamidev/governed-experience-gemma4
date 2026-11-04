"""Where a task's time goes, how behavior changes as the context grows, and what stop rules would have saved.

Reads processed agent runs (data/eval/<label>/: traces plus Docker re-verified results) and replays, offline and
without a model, candidate per-task stop rules on the recorded traces:

- max_turns X (eval_config): the harness ends the task after X model calls and keeps the last submitted patch,
  or the working-tree diff when nothing was submitted (swegemma/harness/agent_runner.py).
- stop at the first submit_patch: submit_patch does not count against the tool-call budget and only the last
  submission is scored, so after a successful submit the session runs until the model answers with text only.
- stop after N failed self-verifications without a pass (the agent's own pytest / repro runs).

Resolved counts only verifiable tasks (reference patch passes and empty patch fails in our Docker environment).
A rule "loses" a resolved task whenever it would have fired before the end of that task: conservative, because the
harness keeps the patch that existed at the cut.

Usage: PYTHONPATH=lab/src lab/.venv/bin/python -m lab.analysis.stop_replay
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
from dataclasses import dataclass, field
from pathlib import Path

from lab.eval.verifiable import load_verifiable

ROOT = Path(__file__).resolve().parents[4]
EVAL = ROOT / "data/eval"
LABELS = ["kaggle-v7-top-a", "kaggle-v7-top-b", "kaggle-v7-top-check", "kaggle-think-probe-top",
          "kaggle-think-probe-top-think", "kaggle-thinking-top", "kaggle-thinking-top-think"]
CANARY = EVAL / "canary2/canary2.json"
TASKS_HIDDEN = 120  # ~120 hidden tasks run serially in one 12 h window (doc 01, section 11.1, row 22)

DESTRUCTIVE_GIT = re.compile(r"\bgit\s+(restore|checkout|reset|stash|clean)\b")
SELF_TEST = re.compile(r"\b(pytest|python3?\s+(-m\s+pytest|\S*(repro|test)\S*\.py))")
EXIT_CODE = re.compile(r'"exit_code":\s*(-?\d+)')
PATCH_SIZE = re.compile(r'"patch_size":\s*(\d+)')
BUCKETS = ((0, 5), (5, 10), (10, 15), (15, 33))


@dataclass
class Step:
    author: str | None
    prompt: int
    completion: int
    emit: float | None
    end: float | None
    model_s: float | None
    tool: str | None
    args: dict
    obs: str
    repeat: bool


@dataclass
class Task:
    task: str
    resolved: bool | None  # None: not verifiable
    duration: float
    llm_calls: int
    steps: list[Step] = field(default_factory=list)

    def first_submit(self) -> int | None:
        return next((i for i, s in enumerate(self.steps) if s.tool == "submit_patch"), None)


def parse_trace(path: Path) -> list[Step]:
    out, prev_end, last_call = [], None, {}
    for s in json.loads(path.read_text())["steps"]:
        e = s.get("extra") or {}
        if s.get("source") != "agent" or not s.get("metrics"):
            prev_end = e.get("elapsed_s", prev_end)
            continue
        obs = s.get("observation") if isinstance(s.get("observation"), dict) else {}
        calls = s.get("tool_calls") or []
        call = calls[0] if calls else None
        emit, end = e.get("elapsed_s"), (obs.get("extra") or {}).get("elapsed_s")
        author = e.get("author")
        sig = (call["function_name"], json.dumps(call["arguments"], sort_keys=True)) if call else None
        repeat = sig is not None and last_call.get(author) == sig
        if sig:
            last_call[author] = sig
        out.append(Step(author, s["metrics"].get("prompt_tokens", 0), s["metrics"].get("completion_tokens", 0),
                        emit, end, emit - prev_end if emit is not None and prev_end is not None else None,
                        call["function_name"] if call else None, call["arguments"] if call else {},
                        str(obs.get("content", "")), repeat))
        prev_end = end if end is not None else (emit if emit is not None else prev_end)
    return out


def load_run(label: str, verifiable: dict[str, bool] | None) -> list[Task]:
    run = EVAL / label
    raw = {json.loads(l)["instance_id"]: json.loads(l) for l in (run / "task_results.jsonl").read_text().splitlines()}
    verified_path = run / "verified_task_results.jsonl"
    verified = ({json.loads(l)["instance_id"]: json.loads(l)["resolved"] for l in verified_path.read_text().splitlines()}
                if verified_path.exists() else {})
    tasks = []
    for iid, r in sorted(raw.items()):
        trace = run / "traces" / f"trace_{iid}.json"
        ok = verifiable.get(iid, False) if verifiable is not None else True
        resolved = bool(verified.get(iid, r["resolved"])) if ok else None
        tasks.append(Task(iid, resolved, r["duration_seconds"], r.get("total_llm_calls") or 0,
                          parse_trace(trace) if trace.exists() else []))
    return tasks


def ols(rows: list[list[float]], y: list[float]) -> list[float]:
    n = len(rows[0])
    a = [[sum(r[i] * r[j] for r in rows) for j in range(n)] for i in range(n)]
    b = [sum(r[i] * v for r, v in zip(rows, y)) for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            f = a[j][i] / a[i][i]
            a[j] = [x - f * z for x, z in zip(a[j], a[i])]
            b[j] -= f * b[i]
    x = [0.0] * n
    for i in reversed(range(n)):
        x[i] = (b[i] - sum(a[i][k] * x[k] for k in range(i + 1, n))) / a[i][i]
    return x


def cost(tasks: list[Task]) -> dict:
    steps = [s for t in tasks for s in t.steps if s.model_s and s.model_s > 0 and s.end is not None and s.emit is not None]
    model = sum(s.model_s for s in steps)
    tool = sum(max(0.0, s.end - s.emit) for s in steps)
    a, per_prompt, per_completion = ols([[1, s.prompt / 1000, s.completion / 1000] for s in steps], [s.model_s for s in steps])
    long_calls = [s for s in steps if s.completion > 500]
    return {"calls": len(steps), "model_share": model / (model + tool), "intercept_s": a,
            "s_per_kprompt": per_prompt, "s_per_kcompletion": per_completion,
            "long_share": sum(s.model_s for s in long_calls) / model, "long_calls": len(long_calls)}


def repetition_by_context(tasks: list[Task]) -> list[tuple[str, int, float]]:
    pre = [s for t in tasks for s in t.steps[:t.first_submit() if t.first_submit() is not None else len(t.steps)]]
    out = []
    for lo, hi in BUCKETS:
        b = [s for s in pre if lo * 1000 <= s.prompt < hi * 1000]
        if b:
            out.append((f"{lo}-{hi}k", len(b), sum(s.repeat for s in b) / len(b)))
    return out


def post_submit(tasks: list[Task]) -> dict:
    out = {"submitted": 0, "continued": 0, "shrank": 0, "destructive": 0, "seconds_after": 0.0, "cases": []}
    for t in tasks:
        i = t.first_submit()
        if i is None:
            continue
        out["submitted"] += 1
        sizes = submit_sizes(t)
        after = [s for s in t.steps[i + 1:] if s.tool]
        destructive = any(s.tool == "run_command" and DESTRUCTIVE_GIT.search(s.args.get("command", "")) for s in after)
        out["seconds_after"] += max(0.0, t.duration - (t.steps[i].emit or 0))
        out["continued"] += bool(after)
        out["destructive"] += destructive
        if sizes and sizes[-1] < sizes[0]:
            out["shrank"] += 1
            out["cases"].append((t.task, sizes[0], sizes[-1], destructive, t.resolved))
    return out


def self_checks(t: Task) -> str:
    """P/F per self-verification run of the coder before its first submit."""
    end = t.first_submit()
    seq = ""
    for s in t.steps[:end if end is not None else len(t.steps)]:
        if s.author == "swe_coder" and s.tool == "run_command" and SELF_TEST.search(s.args.get("command", "")):
            m = EXIT_CODE.search(s.obs)
            seq += "P" if m and m.group(1) == "0" else "F"
    return seq


def first_pass(tasks: list[Task]) -> dict[str, list[int | None]]:
    out = {"resolved": [], "unresolved": []}
    for t in tasks:
        if t.resolved is None:
            continue
        seq = self_checks(t)
        out["resolved" if t.resolved else "unresolved"].append(seq.index("P") + 1 if "P" in seq else None)
    return out


def cut_at(t: Task, index: int | None) -> float:
    """Seconds saved if the task had ended right after step `index`."""
    if index is None or index >= len(t.steps) - 1:
        return 0.0
    when = t.steps[index].end or t.steps[index].emit
    return max(0.0, t.duration - when) if when is not None else 0.0


def rule_max_turns(t: Task, x: int) -> int | None:
    return x - 1 if t.llm_calls > x and len(t.steps) > x else None


def submit_sizes(t: Task) -> list[int]:
    return [int(m.group(1)) for s in t.steps if s.tool == "submit_patch" and (m := PATCH_SIZE.search(s.obs))]


def rule_first_submit(t: Task) -> int | None:
    """Fires only when the agent kept calling tools after its first submit."""
    i = t.first_submit()
    return i if i is not None and any(s.tool for s in t.steps[i + 1:]) else None


def rule_failed_checks(t: Task, n: int) -> int | None:
    fails, end = 0, t.first_submit()
    for i, s in enumerate(t.steps[:end if end is not None else len(t.steps)]):
        if s.author == "swe_coder" and s.tool == "run_command" and SELF_TEST.search(s.args.get("command", "")):
            m = EXIT_CODE.search(s.obs)
            if m and m.group(1) == "0":
                return None
            fails += 1
            if fails >= n:
                return i
    return None


def patch_changed(t: Task) -> bool:
    """Stopping at the first submit loses the task only if a later submit scored a different patch."""
    sizes = submit_sizes(t)
    return len(set(sizes)) > 1


def replay(tasks: list[Task], rule, lost_if=None) -> dict:
    """`lost_if(task)` decides whether a cut resolved task is lost; by default every cut resolved task is."""
    fired = [(t, rule(t)) for t in tasks]
    fired = [(t, i) for t, i in fired if i is not None and cut_at(t, i) > 0]
    saved = sum(cut_at(t, i) for t, i in fired)
    lost = [t.task for t, _ in fired if t.resolved and (lost_if is None or lost_if(t))]
    total = sum(t.duration for t in tasks)
    return {"fired": len(fired), "saved_min": saved / 60, "saved_share": saved / total if total else 0.0,
            "lost": lost, "mean_min_after": (total - saved) / 60 / len(tasks)}


def render(runs: dict[str, list[Task]]) -> str:
    every = [t for ts in runs.values() for t in ts]
    judged = [t for t in every if t.resolved is not None]
    lines = ["# Stop rules replayed on recorded traces", "",
             "Generated by `make stop-replay` (`lab/analysis/stop_replay.py`). Runs: " + ", ".join(f"`{k}`" for k in runs) + ".",
             f"Tasks: {len(every)}; verifiable: {len(judged)}; resolved among verifiable: {sum(t.resolved for t in judged)}.", ""]

    lines += ["## 1. Where the time goes (per model call)", "",
              "| run | calls | model share | s per 1k prompt tokens | s per 1k generated tokens | calls > 500 tokens | their share of model time |",
              "|---|---|---|---|---|---|---|"]
    for k, ts in runs.items():
        c = cost(ts)
        lines.append(f"| {k} | {c['calls']} | {c['model_share']:.0%} | {c['s_per_kprompt']:.3f} | {c['s_per_kcompletion']:.1f} | "
                     f"{c['long_calls']} | {c['long_share']:.0%} |")
    if CANARY.exists():
        base = json.loads(CANARY.read_text())["long"]["base"]
        lines += ["", f"Cold prompt (no prefix cache), LoRA canary v2: {base['prompt_tokens']:,} tokens in {base['seconds']} s "
                      f"(~{base['seconds'] / base['prompt_tokens'] * 1000:.2f} s per 1k tokens, including a short answer)."]

    lines += ["", "## 2. Identical consecutive calls by context size (before the first submit)", "",
              "| run | " + " | ".join(f"{lo}-{hi}k" for lo, hi in BUCKETS) + " |", "|---|" + "---|" * len(BUCKETS)]
    for k, ts in runs.items():
        cells = {b: f"{r:.0%} ({n})" for b, n, r in repetition_by_context(ts)}
        lines.append(f"| {k} | " + " | ".join(cells.get(f"{lo}-{hi}k", "") for lo, hi in BUCKETS) + " |")

    ps = post_submit(every)
    total = sum(t.duration for t in every)
    lines += ["", "## 3. After the first successful submit", "",
              f"- Tasks with a submit: {ps['submitted']}; kept working after it: {ps['continued']}; "
              f"final patch smaller than the first: {ps['shrank']}; destructive git after it: {ps['destructive']}.",
              f"- Time after the first submit: {ps['seconds_after'] / 60:.0f} of {total / 60:.0f} min ({ps['seconds_after'] / total:.0%}).",
              "", "| task | first size | final size | destructive git | resolved |", "|---|---|---|---|---|"]
    lines += [f"| {c[0]} | {c[1]} | {c[2]} | {c[3]} | {c[4]} |" for c in ps["cases"]]

    fp = first_pass(every)
    lines += ["", "## 4. Self-verification before the first submit (attempt of the first passing run)", ""]
    for k, v in fp.items():
        hit = sorted(x for x in v if x is not None)
        lines.append(f"- {k} (n={len(v)}): passed at some point {len(hit)}; first pass at attempt {hit}")

    lines += ["", "## 5. Rules replayed (all runs pooled)", "",
              "| rule | tasks cut | minutes saved | share | resolved lost | mean min per task after | projected h for 120 |",
              "|---|---|---|---|---|---|---|"]
    rules: list = [(f"max_turns {x}", lambda t, x=x: rule_max_turns(t, x)) for x in (100, 120, 150, 200, 300)]
    rules += [("stop at first submit", rule_first_submit, patch_changed)]
    rules += [(f"{n} failed self-checks, no pass", lambda t, n=n: rule_failed_checks(t, n)) for n in (2, 3, 4, 5)]
    for name, rule, *lost_if in rules:
        r = replay(every, rule, *lost_if)
        lines.append(f"| {name} | {r['fired']} | {r['saved_min']:.0f} | {r['saved_share']:.0%} | "
                     f"{len(r['lost'])} {', '.join(r['lost'])} | {r['mean_min_after']:.1f} | {r['mean_min_after'] * TASKS_HIDDEN / 60:.1f} |")

    lines += ["", "## 6. Per run: max_turns 150 and stop at first submit", "",
              "| run | mean min per task | h for 120 | max_turns 150: saved, lost | first submit: saved, lost |", "|---|---|---|---|---|"]
    for k, ts in runs.items():
        mean = sum(t.duration for t in ts) / 60 / len(ts)
        a, b = replay(ts, lambda t: rule_max_turns(t, 150)), replay(ts, rule_first_submit, patch_changed)
        lines.append(f"| {k} | {mean:.1f} | {mean * TASKS_HIDDEN / 60:.1f} | {a['saved_min']:.0f} min, {len(a['lost'])} | "
                     f"{b['saved_min']:.0f} min, {len(b['lost'])} |")
    lines += ["", "Notes: the tool budget does not count submit_patch, so only max_turns, the time cap or a text-only answer "
                  "end a submit loop. Whether max_turns counts the analyzer sub-agent's calls is not verified; the replay "
                  "uses the harness's total_llm_calls. Stopping at the first submit counts a resolved task as lost only when a "
                  "later submit changed the patch size. Self-checks are the coder's pytest / repro runs, a weak proxy."]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs="*", default=LABELS)
    ap.add_argument("--out", type=Path, default=ROOT / "data/analysis/stop_replay.md")
    args = ap.parse_args()
    verifiable = load_verifiable()
    runs = {label: load_run(label, verifiable) for label in args.labels if (EVAL / label / "task_results.jsonl").exists()}
    args.out.write_text(render(runs))
    print(args.out)


if __name__ == "__main__":
    main()
