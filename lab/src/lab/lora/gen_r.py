"""Generate recovery (R1-R3) LoRA trajectories (doc 03 §14.31) from a task's reference patch.

Each trajectory provokes a real error and shows the correct recovery; the turn that provokes the error carries
``loss: False`` in the plan and is dropped from training (`train.coder_turns`). Output layout mirrors ``gen_b``:

    data/lora/<r1|r2|r3>/requests.jsonl   every request/response (coder + analyzer)
    data/lora/<r1|r2|r3>/runs/<variant>/   harness results
    data/lora/<r1|r2|r3>/plans.json       plan of each variant
    data/lora/<r1|r2|r3>/summary.json     per variant: calls, errors, test result, patch equality

Usage:  PYTHONPATH=lab/src lab/.venv/bin/python -m lab.lora.gen_r --type R1 [--tasks rich_3006,...] [--port 18765]
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

OUT_BASE = S.ROOT / "data/lora"
SUBMISSION = S.ROOT / "build/top-mix-1k/submission"


def recovery_tasks(task_ids: list[str] | None) -> list[tuple[str, dict]]:
    """Tasks with a usable analyzer answer and snapshot, tagged by recovery type availability."""
    snaps = {p.stem for p in (S.DATA / "snapshots").glob("*.tgz")}
    tasks = [t for t in (json.loads(ln) for ln in (S.DATA / "tasks.jsonl").read_text().splitlines())
             if t["instance_id"] in snaps and not S.eligible(t)
             and (not task_ids or t["instance_id"] in task_ids)]
    out = []
    for t in tasks:
        if not S.analyzer_answers(t):
            continue
        out.append(t)
    return out


def _test_file(task: dict, snapshot_dir: Path) -> str | None:
    files = list(dict.fromkeys(h.path for h in S.parse_patch(task["patch"])))
    return S.module_test_file(snapshot_dir, files)


def build_plan(kind: str, task: dict) -> list[S.Step] | None:
    with tempfile.TemporaryDirectory() as d:
        snap = S.extract(task["instance_id"], Path(d))
        tf = _test_file(task, snap)
        if kind == "R1":
            return S.plan_r1(task, snap, tf)
        if kind == "R2":
            return S.plan_r2(task, snap, tf)
        if kind == "R3":
            return S.plan_r3(task, snap, tf)
        raise ValueError(kind)


def _env() -> dict:
    import os
    return {**os.environ, "PYTHONPATH": str(S.ROOT / "lab/src")}


def inspect(results: Path, iid: str, steps: list[S.Step], task: dict) -> dict:
    trace = results / "traces" / f"trace_{iid}.json"
    if not trace.exists():
        return {"trace": False}
    obs = []
    for s in json.loads(trace.read_text())["steps"]:
        o = s.get("observation") or {}
        for tc in s.get("tool_calls") or []:
            if (tc.get("extra") or {}).get("author") == "swe_coder":
                obs.append((tc["function_name"], o.get("content") or ""))
    errors = [c for _, c in obs if '"status": "error"' in c or re.search(r'"exit_code": [1-9]', c)
              or "Traceback (most recent call last)" in c or "mandatory input parameters" in c
              or "Failed to replace" in c or "old_string not found" in c]
    test_ok = None
    for name, c in obs:
        if name == "run_command" and ("pytest" in c or re.search(r"\d+ (passed|failed)", c)):
            test_ok = bool(re.search(r"\d+ passed", c)) and not re.search(r"\d+ (failed|error)", c)
    patch_file = next(iter((results / "patches").glob("*")), None) if (results / "patches").exists() else None
    patch = patch_file.read_text() if patch_file else ""
    return {"trace": True, "coder_calls": [n for n, _ in obs], "tool_errors": [c[:300] for c in errors],
            "tool_error_count": len(errors), "test_ok": test_ok,
            "patch_equals_reference": bool(patch) and _norm(patch) == _norm(task["patch"])}

def _norm(patch: str) -> list[str]:
    out = []
    for ln in patch.splitlines():
        if ln.startswith(("+++ ", "--- ")) or (ln[:1] in "+-" and not ln.startswith(("+++", "---"))):
            out.append(ln.rstrip())
    return out


def run_one(kind: str, task: dict, out: Path, port: int) -> dict | None:
    iid = task["instance_id"]
    steps = build_plan(kind, task)
    if steps is None:
        return None
    variant = f"{iid}-{kind.lower()}"
    plan_file = out / "current_plan.json"
    out.mkdir(parents=True, exist_ok=True)
    (out / "runs").mkdir(exist_ok=True)
    results = out / "runs" / variant
    shutil.rmtree(results, ignore_errors=True)
    entry = {"variant": variant, "problem_head": task["problem_statement"][:200],
             "analyzer_text": next((a["text"] for a in S.analyzer_answers(task)), ""),
             "steps": [s.to_json() for s in steps]}
    plan_file.write_text(json.dumps({iid: entry}))
    proc = subprocess.run(
        [sys.executable, "-W", "ignore", "-m", "lab.eval.run_eval", "--submission", str(SUBMISSION),
         "--task-ids", iid, "--sandbox", "subprocess", "--api-base", f"http://127.0.0.1:{port}/v1",
         "--served-model", "scripted", "--label", f"lora-{kind.lower()}", "--results", str(results)],
        capture_output=True, text=True, cwd=S.ROOT, env=_env())
    info = inspect(results, iid, steps, task)
    info.update({"variant": variant, "returncode": proc.returncode, "stderr_tail": proc.stderr[-500:]})
    if proc.returncode:
        print("STDERR", variant, proc.stderr[-300:], file=sys.stderr)
    return {"variant": variant, "entry": entry, "info": info}


def _reference_test_passes(task: dict, snapshot_dir: Path, test_file: str | None) -> bool:
    if not test_file:
        return False
    import subprocess
    p = snapshot_dir / "patch.diff"
    p.write_text(task["patch"].rstrip("\n") + "\n")
    subprocess.run(["git", "init", "-q"], cwd=str(snapshot_dir), capture_output=True)
    applied = subprocess.run(["git", "apply", "--whitespace=nowarn", "patch.diff"], cwd=str(snapshot_dir),
                             capture_output=True, text=True)
    if applied.returncode:
        return False
    tr = subprocess.run([sys.executable, "-m", "pytest", test_file, "-x", "-q"],
                        cwd=str(snapshot_dir), capture_output=True, text=True)
    return " passed" in tr.stdout and "failed" not in tr.stdout and "error" not in tr.stdout.lower()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", required=True, choices=["R1", "R2", "R3", "all"])
    ap.add_argument("--tasks", default="")
    ap.add_argument("--port", type=int, default=18765)
    a = ap.parse_args()

    kinds = ["R1", "R2", "R3"] if a.type == "all" else [a.type]
    task_list = recovery_tasks([t for t in a.tasks.split(",") if t])
    for kind in kinds:
        out = OUT_BASE / kind.lower()
        out.mkdir(parents=True, exist_ok=True)
        log = out / "requests.jsonl"
        plan = out / "current_plan.json"
        plan.write_text("{}")
        log.write_text("")
        srv = serve(plan, log, a.port)
        plans, summary = {}, []
        try:
            for task in task_list:
                iid = task["instance_id"]
                # R3 only applicable to multi-hunk + test tasks.
                if kind == "R3":
                    with tempfile.TemporaryDirectory() as d:
                        snap = S.extract(iid, Path(d))
                        tf = _test_file(task, snap)
                        if S.plan_r3(task, snap, tf) is None or not _reference_test_passes(task, snap, tf):
                            print(json.dumps({"variant": f"{iid}-r3", "skipped": "not applicable (single hunk, no test, or reference patch fails tests)"}),
                                  flush=True)
                            continue
                # R1/R2: exclude tasks whose reference patch does not pass in our env (red tests).
                if kind in ("R1", "R2"):
                    with tempfile.TemporaryDirectory() as d:
                        snap = S.extract(iid, Path(d))
                        tf = _test_file(task, snap)
                        if tf is not None and not _reference_test_passes(task, snap, tf):
                            print(json.dumps({"variant": f"{iid}-{kind.lower()}", "skipped": "excluded: reference patch fails tests in our env"}),
                                  flush=True)
                            continue
                r = run_one(kind, task, out, a.port)
                if r is None:
                    continue
                plans[r["variant"]] = r.pop("entry")
                summary.append(r["info"])
                print(json.dumps({k: v for k, v in r["info"].items() if k not in ("stderr_tail",)}, ), flush=True)
        finally:
            srv.shutdown()
            (out / "plans.json").write_text(json.dumps(plans, indent=1))
            (out / "summary.json").write_text(json.dumps(summary, indent=1))
        # Aggregate report of error texts.
        print(f"{kind}: {len(plans)} variants -> {out}", flush=True)


if __name__ == "__main__":
    main()
