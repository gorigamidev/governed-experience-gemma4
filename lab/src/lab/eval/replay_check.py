"""Offline admission test for habit G1 (doc 03, 14.16): replay `check` on patches the agent already produced.

Each patch is applied in the official sandbox image exactly as Phase 2 prepares the container (wheels, snapshot,
editable install, baseline commit, agent patch), but instead of the hidden tests the b2-check skill's `check`
action runs, through the same run_skill_script wrapper the harness uses. The sets are fixed by the rule:
F = the 9 autopsy attempts of the habit's lineage, OK = the resolved attempts on verifiable tasks.

Usage: PYTHONPATH=lab/src lab/.venv/bin/python -m lab.eval.replay_check [--only F|OK] [--limit N]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import tempfile
import time
import uuid
from pathlib import Path

from lab.eval.verify_patches import DATA, ROOT, use_wheel_set, wheels_for
from lab.tools.run_skill import build_wrapper, load_skill_files

AUTOPSY = ROOT / "data/analysis/autopsy-2026-10-01.md"
LINEAGE_CODES = {"A1", "A2", "A3", "E"}
OUT = ROOT / "data/analysis/replay-check-g1.jsonl"


def report_rows(label: str) -> list[dict]:
    text = (ROOT / f"data/analysis/trace_report_{label}.md").read_text().split("## Per task")[1].strip().splitlines()
    hdr = [h.strip() for h in text[0].strip("|").split("|")]
    return [dict(zip(hdr, [x.strip() for x in ln.strip("|").split("|")])) for ln in text[2:] if ln.startswith("|")]


def fixed_sets() -> list[tuple[str, str, str, str]]:
    """(set, code, task, run label) for every attempt the rule names."""
    rows = []
    for line in AUTOPSY.read_text().split("## Por intento")[1].splitlines():
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) == 3 and cells[2] in LINEAGE_CODES:
            rows.append(("F", cells[2], cells[0], cells[1]))
    for report in sorted((ROOT / "data/analysis").glob("trace_report_kaggle-*.md")):
        label = report.stem.removeprefix("trace_report_")
        for r in report_rows(label):
            if r.get("verifiable") == "True" and r.get("resolved") == "True":
                rows.append(("OK", "R", r["task"], label))
    return rows


async def replay_one(evaluator, cfg, task, label: str, skill_files: dict) -> dict:
    from swegemma.deduplication import resolve_task_snapshot_paths
    from swegemma.harness.container_setup import (apply_patch_in_container, extract_snapshot,
                                                  install_editable_package, install_test_dependencies,
                                                  setup_baseline_commit, setup_container_wheels, setup_git_exclude,
                                                  setup_synthetic_hardware, setup_workspace_test_config)
    from swegemma.sandbox import sandbox_exec, sandbox_start, sandbox_stop

    docker = evaluator.docker
    patch = (ROOT / "data/eval" / label / "patches" / f"{task.instance_id}.patch").read_text()
    snapshot, base, patch_path = resolve_task_snapshot_paths(cfg.snapshots_dir, task.instance_id, task.repo)
    cid = await sandbox_start(docker)
    try:
        await asyncio.to_thread(setup_container_wheels, docker, cid, cfg)
        await asyncio.to_thread(extract_snapshot, docker, cid, snapshot, patch_path=patch_path,
                                base_snapshot_path=base)
        await asyncio.to_thread(setup_synthetic_hardware, docker, cid)
        await asyncio.to_thread(setup_git_exclude, docker, cid)
        await asyncio.to_thread(install_editable_package, docker, cid)
        await asyncio.to_thread(install_test_dependencies, docker, cid, task.repo, fast_path=True, config=cfg)
        await asyncio.to_thread(setup_workspace_test_config, docker, cid, repo=task.repo)
        await asyncio.to_thread(setup_baseline_commit, docker, cid, "eval_baseline")
        with tempfile.TemporaryDirectory() as td:
            pf = Path(td, f"agent-{uuid.uuid4().hex[:6]}.patch")
            pf.write_text(patch if patch.endswith("\n") else patch + "\n")
            await asyncio.to_thread(docker.copy_to, cid, pf, "/tmp/")
            code, out, err = await asyncio.to_thread(apply_patch_in_container, docker, cid, f"/tmp/{pf.name}")
            if code != 0:
                return {"applied": False, "error": (err or out)[-300:]}
            wrapper = Path(td, f".adk_exec_{uuid.uuid4().hex[:8]}.py")
            wrapper.write_text(build_wrapper(skill_files, "entrypoint.py", {"action": "check"}))
            await asyncio.to_thread(docker.copy_to, cid, wrapper, "/workspace/")
        t0 = time.perf_counter()
        res = await sandbox_exec(docker, cid, f"cd /workspace && python3 {wrapper.name}")
        seconds = round(time.perf_counter() - t0, 1)
        try:
            payload = json.loads(res.stdout)
        except ValueError:
            return {"applied": True, "seconds": seconds, "error": (res.stderr or res.stdout)[-300:]}
        return {"applied": True, "seconds": seconds, "verdict": payload.get("verdict"), "check": payload}
    finally:
        await sandbox_stop(docker, cid)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skill", default=ROOT / "build/variants/b2-check/skills/mini-palace", type=Path)
    ap.add_argument("--only", choices=["F", "OK"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sandbox", choices=["docker", "subprocess"], default="docker",
                    help="subprocess = the Kaggle notebooks' sandbox (pipeline check; the rule's numbers use docker)")
    ap.add_argument("--out", type=Path, help="write rows here instead of the default file")
    ap.add_argument("--results", type=Path, help="replay every patch of this run instead (writes check_replay.jsonl)")
    args = ap.parse_args()

    from adk_submission.registry import ModelRegistry
    from swegemma.config import EvalConfig, build_submission_limits
    from swegemma.evaluate import Evaluator
    from swegemma.models import load_tasks

    tasks = {t.instance_id: t for t in load_tasks(DATA / "tasks.jsonl")}
    skill_files = load_skill_files(args.skill)
    if args.results:
        label = args.results.name
        todo = [("RUN", "-", p.stem, label) for p in sorted((args.results / "patches").glob("*.patch"))
                if p.read_text().strip()]
    else:
        todo = [r for r in fixed_sets() if not args.only or r[0] == args.only]
    todo = todo[: args.limit] if args.limit else todo
    limits, gen = build_submission_limits()
    rows = []
    for set_name, code, iid, label in todo:
        wheels = wheels_for(tasks[iid].repo, None)
        use_wheel_set(wheels)
        cfg = EvalConfig(tasks_path=DATA / "tasks.jsonl", snapshots_dir=DATA / "snapshots", results_dir=ROOT / "data/eval",
                         submission_dir=ROOT / "submission", models=ModelRegistry(), sandbox=args.sandbox, wheels_dir=wheels,
                         timeout_seconds=300, limits=limits, generation_constraints=gen, display_mode="quiet")
        res = asyncio.run(replay_one(Evaluator(cfg), cfg, tasks[iid], label, skill_files))
        row = {"set": set_name, "code": code, "task": iid, "run": label, **res}
        rows.append(row)
        print(f"{set_name} {code:2} {iid:15} {label:32} -> {res.get('verdict') or res.get('error', '')[:80]} "
              f"({res.get('seconds', '-')}s)", flush=True)
    out = args.out or (args.results / "check_replay.jsonl" if args.results else OUT)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    if args.results:
        print(f"broken {sum(r.get('verdict') == 'broken' for r in rows)}/{len(rows)} -> {out}")
        return

    f = [r for r in rows if r["set"] == "F"]
    ok = [r for r in rows if r["set"] == "OK"]
    flagged = sum(r.get("verdict") in ("broken", "regressions", "hygiene") for r in f)
    a12_broken = sum(r.get("verdict") == "broken" for r in f if r["code"] in ("A1", "A2"))
    false_broken = sum(r.get("verdict") == "broken" for r in ok)
    soft_ok = sum(r.get("verdict") == "regressions" for r in ok)
    secs = [r["seconds"] for r in rows if "seconds" in r]
    print(f"\nF flagged {flagged}/{len(f)} | A1+A2 broken {a12_broken}/{sum(r['code'] in ('A1', 'A2') for r in f)} | "
          f"OK broken {false_broken}/{len(ok)} | OK regressions {soft_ok}/{len(ok)} | "
          f"median check {statistics.median(secs) if secs else '-'}s -> {out}")


if __name__ == "__main__":
    main()
