"""Re-run Phase 2 (official verification) locally with the Docker sandbox on patches produced elsewhere.

Kaggle notebooks have no Docker, so a GPU run there verifies with the subprocess sandbox, which the forum reports
fails gold patches on many fastapi and requests tasks (missing test dependencies). This re-verifies each patch in
`patches/` with swegemma's own verify_task and the Docker sandbox, writing verified_task_results.jsonl, which
lab.ingest.load_traces prefers over the original outcome.

Usage: PYTHONPATH=lab/src lab/.venv/bin/python -m lab.eval.verify_patches --results data/eval/<run> [--download-missing]
       (--gold writes each task's reference patch first: a check that verification itself is sound;
        --empty verifies no patch at all: a task that passes without a fix is not verifiable, lab.eval.verifiable)
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import subprocess
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / "data/competition"
COMPETITION = "gemma-4-developer-agent"
# Only fastapi's tests need the extra wheels (make wheels-local). Elsewhere inline-snapshot's pytest plugin does
# `import attrs`, which in rich resolves to the repo's examples/attrs.py and aborts pytest (INTERNALERROR).
LOCAL_WHEELS_REPOS = {"fastapi/fastapi"}


def wheels_for(repo: str, override: Path | None) -> Path:
    if override:
        return override
    local = DATA / "wheels_local"
    return local if repo in LOCAL_WHEELS_REPOS and local.is_dir() else DATA / "wheels"


def use_wheel_set(wheels: Path) -> str:
    """swegemma caches unpacked wheels under tempfile.gettempdir()/swegemma_sp_cache_v8 by a fixed name, whatever
    the wheel set; a different wheels dir would silently reuse a stale cache. One temp dir per wheel set."""
    wheel_set = hashlib.sha1("\n".join(sorted(p.name for p in wheels.glob("*.whl"))).encode()).hexdigest()[:10]
    tempfile.tempdir = str(ROOT / "data/.cache" / f"wheels-{wheel_set}")
    Path(tempfile.tempdir).mkdir(parents=True, exist_ok=True)
    return wheel_set


def ensure_snapshot(snapshots: Path, instance_id: str) -> bool:
    if (snapshots / f"{instance_id}.tgz").exists():
        return True
    res = subprocess.run(["kaggle", "competitions", "download", COMPETITION, "-f", f"snapshots/{instance_id}.tgz",
                          "-p", str(snapshots), "-q"], capture_output=True, text=True)
    return res.returncode == 0 and (snapshots / f"{instance_id}.tgz").exists()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, type=Path)
    ap.add_argument("--tasks", default=DATA / "tasks.jsonl", type=Path)
    ap.add_argument("--snapshots", default=DATA / "snapshots", type=Path)
    ap.add_argument("--wheels", type=Path, help="one wheel set for every task; default: per repo (wheels_for)")
    ap.add_argument("--task-ids", default="", help="restrict to these tasks (comma-separated)")
    ap.add_argument("--download-missing", action="store_true")
    ap.add_argument("--gold", action="store_true", help="verify the reference patch instead (sanity check)")
    ap.add_argument("--empty", action="store_true", help="verify an empty patch (control: the tests must fail)")
    args = ap.parse_args()

    from adk_submission.registry import ModelRegistry
    from swegemma.config import EvalConfig, build_submission_limits
    from swegemma.evaluate import Evaluator
    from swegemma.models import load_tasks

    meta_path = args.results / "run_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    timeout = int(meta.get("budgets", {}).get("timeout_seconds", 300))
    tasks = {t.instance_id: t for t in load_tasks(args.tasks)}
    patches_dir = args.results / "patches"
    patches_dir.mkdir(parents=True, exist_ok=True)
    wanted = [t for t in args.task_ids.split(",") if t]
    if args.gold and args.empty:
        raise SystemExit("--gold and --empty are exclusive")
    if args.gold or args.empty:
        for iid in wanted:
            (patches_dir / f"{iid}.patch").write_text("" if args.empty else tasks[iid].patch)
    ids = wanted or sorted(p.stem for p in patches_dir.glob("*.patch"))

    limits, gen = build_submission_limits()
    submission_dir = args.results / "submission" if (args.results / "submission").is_dir() else ROOT / "submission"
    ids = [i for i in ids if i in tasks and (patches_dir / f"{i}.patch").exists()]
    if args.empty:
        ids = [i for i in ids if i in wanted]
    by_wheels: dict[Path, list[str]] = {}
    for iid in ids:
        by_wheels.setdefault(wheels_for(tasks[iid].repo, args.wheels), []).append(iid)

    rows, wheel_sets = [], {}
    for wheels, group in sorted(by_wheels.items()):
        wheel_sets[wheels.name] = use_wheel_set(wheels)
        cfg = EvalConfig(tasks_path=args.tasks, snapshots_dir=args.snapshots, results_dir=args.results,
                         submission_dir=submission_dir, models=ModelRegistry(), sandbox="docker",
                         wheels_dir=wheels, timeout_seconds=timeout, limits=limits, generation_constraints=gen,
                         display_mode="quiet")
        rows += verify_group(Evaluator(cfg), cfg, tasks, group, patches_dir, args)

    out = args.results / ("gold_check.jsonl" if args.gold else "empty_check.jsonl" if args.empty
                          else "verified_task_results.jsonl")
    if args.gold or args.empty:
        # Accumulate the controls: reference patch passes (verifiable_tasks.json), empty patch passes (should not).
        registry = ROOT / "data/analysis" / ("verifiable_tasks.json" if args.gold else "empty_patch_passes.json")
        known = json.loads(registry.read_text()) if registry.exists() else {}
        known.update({r["instance_id"]: r["resolved"] for r in rows})
        registry.parent.mkdir(parents=True, exist_ok=True)
        registry.write_text(json.dumps(dict(sorted(known.items())), indent=1) + "\n")
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    if not (args.gold or args.empty) and meta:
        meta["phase2_local"] = {"sandbox": "docker", "wheel_sets": wheel_sets,
                                "verified": len(rows), "resolved": sum(r["resolved"] for r in rows),
                                "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        meta_path.write_text(json.dumps(meta, indent=1))
    print(f"{sum(r['resolved'] for r in rows)}/{len(rows)} resolved -> {out}")


def verify_group(evaluator, cfg, tasks, ids, patches_dir: Path, args) -> list[dict]:
    from swegemma.deduplication import resolve_task_snapshot_paths
    from swegemma.harness.verification import verify_task

    rows = []
    for iid in ids:
        if args.download_missing and not ensure_snapshot(args.snapshots, iid):
            print(f"{iid}: snapshot unavailable, skipped")
            continue
        snapshot, base, patch_path = resolve_task_snapshot_paths(args.snapshots, iid, tasks[iid].repo)
        t0 = time.perf_counter()
        res = asyncio.run(verify_task(evaluator.docker, cfg, tasks[iid], snapshot, base_snapshot_path=base,
                                      patch_path=patch_path, agent_patch=(patches_dir / f"{iid}.patch").read_text()))
        rows.append({"instance_id": iid, "resolved": bool(res.resolved), "test_exit_code": res.test_exit_code,
                     "error": res.error, "verify_seconds": round(time.perf_counter() - t0, 1),
                     "wheels": cfg.wheels_dir.name, "test_output_tail": (res.test_output or "")[-1500:]})
        print(f"{iid}: resolved={res.resolved} exit={res.test_exit_code} ({rows[-1]['verify_seconds']}s)"
              + (f" error={res.error[:120]}" if res.error else ""))
    return rows


if __name__ == "__main__":
    main()
