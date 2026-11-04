"""Generate core B (doc 03 §14.25, amendment 1): one harness run per (task, real analyzer answer).

Each run serves a scripted trajectory to the real harness (`subprocess` sandbox, the submission's own config),
so every tool output is real. Output, under data/lora/b/:
  requests.jsonl        every request the harness sent and the scripted response (one line per model call)
  runs/<task>-<k>/      the harness results of each variant (trace, patch, task result)
  plans.json            the plan of each variant (steps, analyzer answer and its source run)
  summary.json          per variant: test step kept or dropped, harness errors, patch equals reference

If the module test step fails at the base commit (environment or pre-existing failures), the variant is rerun
without it: the trajectory never shows a failing check followed by a submission.

Usage:  PYTHONPATH=lab/src lab/.venv/bin/python -m lab.lora.gen_b [--tasks rich_3882,...] [--max-variants N]
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from lab.lora import scripted as S
from lab.lora.scripted_server import serve

OUT = S.ROOT / "data/lora/b"
SUBMISSION = S.ROOT / "build/top-mix-1k/submission"


def candidates(task_ids: list[str] | None) -> list[dict]:
    snaps = {p.stem for p in (S.DATA / "snapshots").glob("*.tgz")}
    tasks = [json.loads(ln) for ln in (S.DATA / "tasks.jsonl").read_text().splitlines()]
    return [t for t in tasks if t["instance_id"] in snaps and not S.eligible(t)
            and (not task_ids or t["instance_id"] in task_ids)]


def build_plan(task: dict, with_test: bool) -> list[S.Step]:
    with tempfile.TemporaryDirectory() as d:
        snap = S.extract(task["instance_id"], Path(d))
        files = list(dict.fromkeys(h.path for h in S.parse_patch(task["patch"])))
        return S.plan(task, snap, S.module_test_file(snap, files) if with_test else None)


def run_variant(task: dict, k: int, answer: dict, with_test: bool, plan_file: Path, port: int) -> dict:
    iid = task["instance_id"]
    steps = build_plan(task, with_test)
    variant = f"{iid}-{k}{'' if with_test else '-notest'}"
    entry = {"variant": variant, "problem_head": task["problem_statement"][:200], "analyzer_text": answer["text"],
             "analyzer_source": answer["run"], "steps": [s.to_json() for s in steps]}
    plan_file.write_text(json.dumps({iid: entry}))
    results = OUT / "runs" / variant
    shutil.rmtree(results, ignore_errors=True)
    proc = subprocess.run(
        [sys.executable, "-W", "ignore", "-m", "lab.eval.run_eval", "--submission", str(SUBMISSION),
         "--task-ids", iid, "--sandbox", "subprocess", "--api-base", f"http://127.0.0.1:{port}/v1",
         "--served-model", "scripted", "--label", "lora-b", "--results", str(results)],
        capture_output=True, text=True, cwd=S.ROOT, env={**_env()})
    return {"variant": variant, "entry": entry, "returncode": proc.returncode, "stderr_tail": proc.stderr[-800:],
            **inspect(results, iid, steps, task)}


def _env() -> dict:
    import os
    return {**os.environ, "PYTHONPATH": str(S.ROOT / "lab/src")}


def inspect(results: Path, iid: str, steps: list[S.Step], task: dict) -> dict:
    """What the harness actually did with the script."""
    trace = results / "traces" / f"trace_{iid}.json"
    if not trace.exists():
        return {"trace": False}
    obs = []
    for s in json.loads(trace.read_text())["steps"]:
        o = s.get("observation") or {}
        for tc in s.get("tool_calls") or []:
            if (tc.get("extra") or {}).get("author") == "swe_coder":
                obs.append((tc["function_name"], o.get("content") or ""))
    errors = [name for name, c in obs if '"status": "error"' in c or re.search(r'"exit_code": [1-9]', c)
              or "Traceback (most recent call last)" in c]
    test_ok = None
    for name, c in obs:
        if name == "run_command" and "pytest" in c or (name == "run_command" and re.search(r"\d+ (passed|failed)", c)):
            test_ok = bool(re.search(r"\d+ passed", c)) and not re.search(r"\d+ (failed|error)", c)
    patch_file = next(iter((results / "patches").glob("*")), None) if (results / "patches").exists() else None
    patch = patch_file.read_text() if patch_file else ""
    return {"trace": True, "coder_calls": [n for n, _ in obs], "tool_errors": errors, "test_ok": test_ok,
            "patch_equals_reference": _norm(patch) == _norm(task["patch"]) if patch else False}


def _norm(patch: str) -> list[str]:
    """Changed lines per file, ignoring hunk offsets, index lines and whitespace at line ends."""
    out = []
    for ln in patch.splitlines():
        if ln.startswith(("+++ ", "--- ")) or (ln[:1] in "+-" and not ln.startswith(("+++", "---"))):
            out.append(ln.rstrip())
    return out


def reserve_prompts(port: int) -> None:
    """Turn-0 requests of the 16 reserve tasks (evaluation prompts, never training data): the harness builds the
    first request and the script answers DONE at once. Written to data/lora/reserve/requests.jsonl."""
    out = S.ROOT / "data/lora/reserve"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    plan_file, log = out / "current_plan.json", out / "all_requests.jsonl"
    tasks = [json.loads(ln) for ln in (S.DATA / "tasks.jsonl").read_text().splitlines()]
    srv = serve(plan_file, log, port)
    try:
        for t in (t for t in tasks if t["instance_id"] in S.RESERVE):
            iid = t["instance_id"]
            plan_file.write_text(json.dumps({iid: {"variant": iid, "problem_head": t["problem_statement"][:200],
                                                   "analyzer_text": "", "steps": [S.Step("", content="DONE").to_json()]}}))
            subprocess.run([sys.executable, "-W", "ignore", "-m", "lab.eval.run_eval", "--submission", str(SUBMISSION),
                            "--task-ids", iid, "--sandbox", "subprocess", "--api-base", f"http://127.0.0.1:{port}/v1",
                            "--served-model", "scripted", "--label", "lora-reserve", "--results", str(out / "runs" / iid)],
                           capture_output=True, text=True, cwd=S.ROOT, env=_env())
    finally:
        srv.shutdown()
    rows = [json.loads(ln) for ln in log.read_text().splitlines()]
    first = {r["task"]: r for r in rows if r["role"] == "coder" and r.get("turn") == 0}
    (out / "requests.jsonl").write_text("".join(json.dumps(first[i]) + "\n" for i in sorted(first)))
    print(f"{len(first)} of {len(S.RESERVE)} reserve tasks with a turn-0 request -> {out / 'requests.jsonl'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reserve-prompts", action="store_true", help="only write the reserve tasks' turn-0 requests")
    ap.add_argument("--tasks", default="")
    ap.add_argument("--max-variants", type=int, default=0, help="per task; 0 = every analyzer answer")
    ap.add_argument("--port", type=int, default=18765)
    a = ap.parse_args()
    if a.reserve_prompts:
        return reserve_prompts(a.port)
    OUT.mkdir(parents=True, exist_ok=True)
    plan_file, log = OUT / "current_plan.json", OUT / "requests.jsonl"
    plan_file.write_text("{}")
    srv = serve(plan_file, log, a.port)
    summary, plans = [], {}
    try:
        for task in candidates([t for t in a.tasks.split(",") if t]):
            answers = S.analyzer_answers(task)
            if a.max_variants:
                answers = answers[:a.max_variants]
            for k, ans in enumerate(answers):
                r = run_variant(task, k, ans, True, plan_file, a.port)
                if r.get("test_ok") is False:
                    r = run_variant(task, k, ans, False, plan_file, a.port)
                plans[r["variant"]] = r.pop("entry")
                summary.append(r)
                print(json.dumps({k2: v for k2, v in r.items() if k2 != "stderr_tail"}), flush=True)
    finally:
        srv.shutdown()
        (OUT / "plans.json").write_text(json.dumps(plans, indent=1))
        (OUT / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
