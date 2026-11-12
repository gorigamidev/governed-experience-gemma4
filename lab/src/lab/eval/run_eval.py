"""Run swegemma's Evaluator on a task subset against any OpenAI-compatible model server.

The same entry point serves the Kaggle GPU notebook (vLLM + Gemma 4, the scorer's stack) and local pipeline
tests (e.g. Ollama + a small open model). The submission always declares gemma-4-31b-it-qat-w4a16-ct; the
model registry maps that alias to whatever server is given, and run_meta.json records what actually ran so
the experience store can keep scorer-faithful evidence apart from pipeline tests.

Usage (local):  PYTHONPATH=lab/src lab/.venv/bin/python -m lab.eval.run_eval --task-ids rich_4079 \
                    --api-base http://localhost:11434/v1 --served-model qwen3:8b --label local-qwen3-8b
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import platform
import shutil
import subprocess
import time
from pathlib import Path

import yaml

# Inside the repo: lab/src/lab/eval/run_eval.py -> repo root. Standalone (e.g. copied into a Kaggle notebook):
# the current directory, with every data path passed explicitly.
_here = Path(__file__).resolve()
ROOT = _here.parents[4] if len(_here.parents) > 4 and (_here.parents[4] / "lab").is_dir() else Path.cwd()
DECLARED_MODEL = "gemma-4-31b-it-qat-w4a16-ct"
# Every checkpoint the scorer may serve. The 31B documents declare the 31B alias, a 12B or E4B package declares its
# own name (make variant MODEL=12b; doc 03 §14.30), and all of them run against the single server given here, so
# every alias points at it. Kept in step with lab/export/models.py NAMES; run_eval is embedded standalone in the
# Kaggle notebook and cannot import lab.*.
CHECKPOINTS = ("gemma-4-31b-it-qat-w4a16-ct", "gemma-4-12b-it-qat-w4a16-ct", "gemma-4-e4b-it-qat-w4a16-ct")
DATA = ROOT / "data/competition"
# Scorer compaction values are unpublished; these are the official Getting Started notebook's (doc 01, §6.8).
COMPACTION = {"compaction_interval": 15, "overlap_size": 2, "token_threshold": 14336, "event_retention_size": 5}
CONTEXT_CACHE = {"min_tokens": 2048, "ttl_seconds": 1800, "cache_intervals": 10}


def submission_digest(submission: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(submission.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            h.update(p.relative_to(submission).as_posix().encode())
            h.update(p.read_bytes())
    return h.hexdigest()[:12]


def git_commit() -> str:
    res = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    return res.stdout.strip() or "unknown"


def budgets(submission: Path, max_turns_override: int | None, max_minutes_override: float | None = None) -> dict:
    raw = yaml.safe_load((submission / "eval_config.yaml").read_text()) if (submission / "eval_config.yaml").exists() else {}
    ev = raw.get("evaluation", raw) or {}
    return {"timeout_seconds": int(ev.get("timeout_seconds", 300)), "max_tool_calls": int(ev.get("max_tool_calls", 100)),
            "max_time_minutes": max_minutes_override or float(ev.get("max_time_minutes", 60.0)),
            "max_turns": max_turns_override or (int(ev["max_turns"]) if "max_turns" in ev else None)}


def patch_thinking_off_for_ollama() -> None:
    """Local pipeline tests only. The scorer's vLLM switches thinking off through chat_template_kwargs, which Ollama
    ignores (qwen3 still reasons); Ollama honors reasoning_effort="none" instead. The submission is unchanged."""
    import adk_submission.builders.llm as llm

    original = llm.apply_thinking_config_to_model

    def apply(model, gen_config):
        model = original(model, gen_config)
        args = getattr(model, "_additional_args", None) or {}
        if (args.get("extra_body") or {}).get("chat_template_kwargs", {}).get("enable_thinking") is False:
            args["reasoning_effort"] = "none"
        return model

    llm.apply_thinking_config_to_model = apply


def model_registry(api_base: str, served_model: str, prefix: str, api_key: str):
    from adk_submission.registry import ModelRegistry
    from google.adk.models.lite_llm import LiteLlm

    registry = ModelRegistry()
    for alias in CHECKPOINTS:
        registry.register(alias, LiteLlm(model=f"{prefix}{served_model}", api_base=api_base, api_key=api_key))
    return registry



def harness_versions() -> dict:
    """Installed versions of the harness packages (forum 750350: skill packages changed behavior across swegemma)."""
    from importlib import metadata
    out = {}
    for pkg in ("swegemma", "adk-submission", "adk-eval-core", "google-adk", "vllm"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", default=ROOT / "submission", type=Path)
    ap.add_argument("--tasks", default=DATA / "tasks.jsonl", type=Path)
    ap.add_argument("--snapshots", default=DATA / "snapshots", type=Path)
    ap.add_argument("--graphs", default=DATA / "graphs", type=Path)
    ap.add_argument("--embeddings", default=DATA / "embeddings", type=Path)
    ap.add_argument("--wheels", default=DATA / "wheels", type=Path)
    ap.add_argument("--results", type=Path, help="default: data/eval/<label>-<timestamp>")
    ap.add_argument("--task-ids", default="", help="comma-separated; default: every task with a local snapshot")
    ap.add_argument("--sandbox", choices=("docker", "subprocess"), default="docker")
    ap.add_argument("--api-base", required=True)
    ap.add_argument("--served-model", required=True, help="model name as the server knows it")
    ap.add_argument("--model-prefix", default="openai/")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--label", required=True, help="e.g. kaggle-l4x4-gemma4 or local-qwen3-8b")
    ap.add_argument("--hardware", default=platform.machine())
    ap.add_argument("--ollama-thinking-off", action="store_true",
                    help="local tests on Ollama: send reasoning_effort=none when the submission turns thinking off")
    ap.add_argument("--max-minutes", type=float, default=None,
                    help="local cap for packages without eval_config (the scorer default is 60 min per task)")
    ap.add_argument("--max-turns", type=int, default=None,
                    help="cap turns instead of trusting wall-clock minutes on faster-than-scorer hardware")
    args = ap.parse_args()

    import litellm
    from google.adk.agents.context_cache_config import ContextCacheConfig
    from google.adk.apps._configs import EventsCompactionConfig
    from swegemma.config import EvalConfig, build_submission_limits
    from swegemma.evaluate import Evaluator

    litellm.drop_params = True  # as the official notebook: servers other than vLLM reject some params
    if args.ollama_thinking_off:
        patch_thinking_off_for_ollama()
    task_ids = [t for t in args.task_ids.split(",") if t] or sorted(p.name.removesuffix(".tgz")
                                                                   for p in args.snapshots.glob("*.tgz"))
    stamp = time.strftime("%Y%m%d-%H%M%S")
    results = args.results or ROOT / "data/eval" / f"{args.label}-{stamp}"
    results.mkdir(parents=True, exist_ok=True)
    # Evaluate a clean copy stored with the results: no __pycache__ (rejected by the validator), and the run
    # keeps the exact submission it used.
    staged = results / "submission"
    shutil.rmtree(staged, ignore_errors=True)
    shutil.copytree(args.submission, staged, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"))
    args.submission = staged
    b = budgets(args.submission, args.max_turns, args.max_minutes)
    meta = {"label": args.label, "served_model": args.served_model, "declared_model": DECLARED_MODEL,
            "api_base": args.api_base, "hardware": args.hardware, "sandbox": args.sandbox,
            "scorer_faithful": args.served_model.startswith("/") or "gemma-4-31b" in args.served_model,
            "submission_digest": submission_digest(args.submission), "git_commit": git_commit(),
            "ollama_thinking_off": args.ollama_thinking_off, "budgets": b, "compaction": COMPACTION, "task_ids": task_ids,
            "harness_versions": harness_versions(),
            "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (results / "run_meta.json").write_text(json.dumps(meta, indent=1))

    limits, gen_constraints = build_submission_limits()
    cfg = EvalConfig(
        tasks_path=args.tasks, snapshots_dir=args.snapshots, results_dir=results, submission_dir=args.submission,
        models=model_registry(args.api_base, args.served_model, args.model_prefix, args.api_key),
        sandbox=args.sandbox, wheels_dir=args.wheels, graph_dir=str(args.graphs), embeddings_dir=str(args.embeddings),
        timeout_seconds=b["timeout_seconds"], max_time_minutes=b["max_time_minutes"],
        max_tool_calls=b["max_tool_calls"], max_turns=b["max_turns"], limits=limits,
        generation_constraints=gen_constraints, task_ids=task_ids,
        context_cache_config=ContextCacheConfig(**CONTEXT_CACHE),
        events_compaction_config=EventsCompactionConfig(**COMPACTION), display_mode="quiet")
    t0 = time.perf_counter()
    res = asyncio.run(Evaluator(cfg).run())
    meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    meta["wall_seconds"] = round(time.perf_counter() - t0, 1)
    (results / "run_meta.json").write_text(json.dumps(meta, indent=1))
    solved = sum(r.resolved for r in res.task_results)
    print(f"{results}: {solved}/{len(res.task_results)} resolved in {meta['wall_seconds']}s")


if __name__ == "__main__":
    main()
