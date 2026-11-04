"""Build a Kaggle GPU notebook (4x L4) that runs the agent on a task subset and saves traces as output.

Same stack as the scorer: the official wheelhouse, vLLM serving the requested Gemma 4 checkpoint (default
gemma-4-31b-it-qat-w4a16-ct) with the gemma4 tool-call and reasoning parsers (setup copied from the organizers'
Getting Started notebook). The validated submission.zip and lab/eval/run_eval.py are embedded, so the kernel needs
no other dataset. Phase 2 runs with the subprocess sandbox (no Docker on Kaggle) and is re-verified locally
(lab.eval.verify_patches).

Several submissions ("arms") can run in one GPU session on the same tasks, for paired comparisons on the same
hardware: --arms b2=build/submission.zip,b1=build/b1-no-skill/submission.zip. Results land in results/<arm>/.
With --interleave the arms alternate per task (task 1 on every arm, then task 2, ...) under a wall-clock budget
(--budget-min), so a cut session still leaves every arm on the same tasks; each run goes to results/_parts/ and is
merged into results/<arm>/ in the same layout as a sequential run.

Usage: PYTHONPATH=lab/src python3 -m lab.export.kaggle_eval_notebook --tasks-per-repo fastapi=10,rich=7,requests=2,httpx=1
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import random
from pathlib import Path

from lab.export import models
from lab.export.kaggle_notebook import with_cell_ids
from lab.ingest.load_tasks import load_tasks

ROOT = Path(__file__).resolve().parents[4]
OWNER = "gorigamidev"
COMPETITION = "gemma-4-developer-agent"
WHEELHOUSE = "metric/gemma-4-developer-agent-wheelhouse"
# Kaggle image of every GPU run that worked (v4, v5, think probe). On 3 Oct the default image moved to Python 3.13 and
# the wheelhouse (cp312) no longer installs; pin the image instead of taking the latest.
DOCKER_IMAGE = "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"

SETUP = r'''import glob, importlib, os, subprocess, sys
from pathlib import Path

# Environment and wheelhouse install: same as the organizers' Getting Started notebook.
os.environ['LITELLM_LOCAL_MODEL_COST_MAP'] = 'True'
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS'] = '1'
# 40 min, not the Getting Started 20: since 21 Oct 4xL4 vLLM starts take up to ~1,700 s (forum 747519).
os.environ['VLLM_ENGINE_READY_TIMEOUT_S'] = '2400'
os.environ['VLLM_NO_USAGE_STATS'] = '1'
os.environ['OTEL_SDK_DISABLED'] = 'true'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
WHEELHOUSE_DIR = Path('/kaggle/input/datasets/metric/gemma-4-developer-agent-wheelhouse')
for pattern in ('/usr/local/lib/python*/dist-packages/*cutlass*.pth', '/usr/local/lib/python*/site-packages/*cutlass*.pth'):
    for pth in glob.glob(pattern):
        try:
            os.unlink(pth)
        except OSError:
            pass
tmp_whl = Path('/tmp/wheelhouse'); tmp_whl.mkdir(parents=True, exist_ok=True)
for w in WHEELHOUSE_DIR.glob('*.whl'):
    if 'cutlass' in w.name.lower():
        continue
    name = w.name.replace('cu128', '+cu128') if ('cu128' in w.name and '+' not in w.name) else w.name
    if not (tmp_whl / name).exists():
        os.symlink(w, tmp_whl / name)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', '--force-reinstall',
                *sorted(str(w) for w in tmp_whl.glob('*.whl'))], check=True)
importlib.invalidate_caches()
print('wheelhouse installed')
'''

MATERIALIZE = r'''import base64, hashlib, io, zipfile
from pathlib import Path

WORK = Path('/kaggle/working')
ARMS = {arms!r}  # name -> (sha256, base64 zip), run in this order
RUN_EVAL_SHA256 = "{runner_sha}"
RUN_EVAL_PY = "{runner_b64}"
TASK_IDS = {task_ids!r}
# Per-task cap for every arm (None: each package's eval_config, or the scorer default when it has none).
MAX_MINUTES = {max_minutes!r}
# Interleaved runs only: wall-clock budget from the run cell, and the time one task needs on every arm.
BUDGET_MIN = {budget_min!r}
ROUND_MIN = {round_min!r}

subs = {{}}
for name, (sha, b64) in ARMS.items():
    data = base64.b64decode(b64)
    assert hashlib.sha256(data).hexdigest() == sha, name
    subs[name] = WORK / 'submissions' / name
    zipfile.ZipFile(io.BytesIO(data)).extractall(subs[name])
    print('arm', name, sha[:12])
sub = next(iter(subs.values()))  # vLLM adapters are discovered from the first arm
code = base64.b64decode(RUN_EVAL_PY)
assert hashlib.sha256(code).hexdigest() == RUN_EVAL_SHA256
(WORK / 'run_eval.py').write_bytes(code)
print('runner', RUN_EVAL_SHA256[:12], '|', len(TASK_IDS), 'tasks x', len(ARMS), 'arms')
'''

SERVE = r'''import litellm, torch
from adk_submission import VllmConfig, VllmServer, discover_adapters
from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS

litellm.drop_params = True
MODEL_PATH = Path('__MODEL_PATH__')
adapters = discover_adapters(str(sub), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
server = VllmServer(VllmConfig(
    model=str(MODEL_PATH), port=8000, host='127.0.0.1', tool_call_parser='gemma4', reasoning_parser='gemma4',
    max_model_len=32768, dtype='bfloat16' if torch.cuda.is_bf16_supported() else 'auto',
    gpu_memory_utilization=0.90, enable_auto_tool_choice=True, enable_lora=True, max_loras=8, max_lora_rank=128,
__EXTRA_ARGS__    tensor_parallel_size=4 if gpus >= 4 else (2 if gpus >= 2 else 1), startup_timeout=60 * 40,
), adapter_manifest=adapters)
server.start()
SERVED_MODEL = getattr(server, 'effective_model_path', str(MODEL_PATH))
print('vLLM up at', server.base_url, '| gpus', gpus, '| served', SERVED_MODEL)
'''

MERGE = r'''import json, shutil
from pathlib import Path

def merge_part(part: Path, dest: Path) -> None:
    """Fold one single-task run directory into the arm's results, as if one run had produced both."""
    dest.mkdir(parents=True, exist_ok=True)
    rows = (part / 'task_results.jsonl').read_text() if (part / 'task_results.jsonl').exists() else ''
    with open(dest / 'task_results.jsonl', 'a') as f:
        f.write(rows)
    for sub in part.iterdir():
        if sub.is_dir() and sub.name != 'submission':
            shutil.copytree(sub, dest / sub.name, dirs_exist_ok=True)
    meta = json.loads((part / 'run_meta.json').read_text())
    if (dest / 'run_meta.json').exists():
        first = json.loads((dest / 'run_meta.json').read_text())
        first['task_ids'] += [t for t in meta['task_ids'] if t not in first['task_ids']]
        first['finished'] = meta.get('finished')
        first['wall_seconds'] = round(first.get('wall_seconds', 0) + meta.get('wall_seconds', 0), 1)
        meta = first
    (dest / 'run_meta.json').write_text(json.dumps(meta, indent=1))
    lines = [json.loads(l) for l in (dest / 'task_results.jsonl').read_text().splitlines() if l.strip()]
    summary = {'total_tasks': len(lines), 'resolved': sum(bool(r.get('resolved')) for r in lines),
               'errors': sum(1 for r in lines if r.get('error'))}
    (dest / 'summary.json').write_text(json.dumps(summary, indent=2))
'''

RUN_INTERLEAVED = r'''import subprocess, sys, tarfile, time
DATA = Path('/kaggle/input/competitions/gemma-4-developer-agent')
RESULTS = WORK / 'results'
PARTS = RESULTS / '_parts'
DEADLINE = time.time() + BUDGET_MIN * 60
names = list(subs)
done = []

def save():
    (WORK / 'progress.json').write_text(json.dumps({'done': done}, indent=1))
    if not RESULTS.exists():
        return
    with tarfile.open(WORK / 'results.tar.gz', 'w:gz') as tar:
        tar.add(RESULTS, arcname='results',
                filter=lambda ti: None if ('/submission/' in ti.name + '/' or '/_parts' in ti.name) else ti)
    print('saved', WORK / 'results.tar.gz')

try:
    for i, task in enumerate(TASK_IDS):
        left = (DEADLINE - time.time()) / 60
        if left < ROUND_MIN:
            print('budget: stopping before', task, '| left', round(left), 'min')
            break
        for name in names[i % len(names):] + names[:i % len(names)]:  # rotate the arm order per task
            part = PARTS / name / task
            cmd = [sys.executable, str(WORK / 'run_eval.py'), '--submission', str(subs[name]),
                   '--tasks', str(DATA / 'tasks.jsonl'), '--snapshots', str(DATA / 'snapshots'),
                   '--graphs', str(DATA / 'graphs'), '--embeddings', str(DATA / 'embeddings'),
                   '--wheels', str(DATA / 'wheels'), '--results', str(part), '--task-ids', task,
                   '--sandbox', 'subprocess', '--api-base', server.base_url, '--served-model', SERVED_MODEL,
                   '--model-prefix', 'openai/', '--label', f'kaggle-l4x4-gemma4-{name}', '--hardware', f'{gpus}xL4']
            t0 = time.time()
            rc = subprocess.run(cmd, check=False, cwd=WORK).returncode
            if (part / 'run_meta.json').exists():
                merge_part(part, RESULTS / name)
            done.append({'task': task, 'arm': name, 'rc': rc, 'minutes': round((time.time() - t0) / 60, 2)})
            print(done[-1], flush=True)
        save()
finally:
    server.stop() if hasattr(server, 'stop') else None
    save()
full = [t for t in TASK_IDS if sum(d['task'] == t for d in done) == len(names)]
print(len(done), 'runs |', len(full), 'tasks with every arm | left', round((DEADLINE - time.time()) / 60), 'min')
'''

RUN = r'''import subprocess, sys, tarfile
DATA = Path('/kaggle/input/competitions/gemma-4-developer-agent')
RESULTS = WORK / 'results'
single = len(subs) == 1

def save():
    # Traces, patches and run_meta.json travel as one archive; the staged submissions are already known.
    # Rewritten after every arm, so a session cut short keeps the arms that finished.
    if not RESULTS.exists():
        return
    with tarfile.open(WORK / 'results.tar.gz', 'w:gz') as tar:
        tar.add(RESULTS, arcname='results', filter=lambda ti: None if '/submission/' in ti.name + '/' else ti)
    print('saved', WORK / 'results.tar.gz')

try:
    for name, arm_sub in subs.items():
        out = RESULTS if single else RESULTS / name
        cmd = [sys.executable, str(WORK / 'run_eval.py'), '--submission', str(arm_sub), '--tasks', str(DATA / 'tasks.jsonl'),
               '--snapshots', str(DATA / 'snapshots'), '--graphs', str(DATA / 'graphs'), '--embeddings', str(DATA / 'embeddings'),
               '--wheels', str(DATA / 'wheels'), '--results', str(out), '--task-ids', ','.join(TASK_IDS),
               '--sandbox', 'subprocess', '--api-base', server.base_url, '--served-model', SERVED_MODEL,
               '--model-prefix', 'openai/', '--label', 'kaggle-l4x4-gemma4' + ('' if single else f'-{name}'),
               '--hardware', f'{gpus}xL4'] + (['--max-minutes', str(MAX_MINUTES)] if MAX_MINUTES else [])
        subprocess.run(cmd, check=False, cwd=WORK)
        save()
finally:
    server.stop() if hasattr(server, 'stop') else None
    save()
'''


def pick_tasks(spec: str, seed: int) -> list[str]:
    """Deterministic stratified sample: repo=count pairs, e.g. fastapi=10,rich=7."""
    want = dict((k, int(v)) for k, v in (item.split("=") for item in spec.split(",") if item))
    by_repo: dict[str, list[str]] = {}
    for t in load_tasks(ROOT / "data/competition/tasks.jsonl"):
        by_repo.setdefault(t.repo_short, []).append(t.instance_id)
    rng = random.Random(seed)
    picked = []
    for repo, n in want.items():
        ids = sorted(by_repo.get(repo, []))
        rng.shuffle(ids)
        picked += sorted(ids[:n])
    return picked


def cell(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": source.splitlines(keepends=True)}


def serve_cell(model: str) -> str:
    """The SERVE cell for `model`: its mount path, plus any per-checkpoint vLLM flags."""
    extra = models.extra_args(model)
    return SERVE.replace("__MODEL_PATH__", models.path(model)).replace(
        "__EXTRA_ARGS__", "" if not extra else f"    extra_args={list(extra)!r},\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=ROOT / "build/submission.zip", type=Path)
    ap.add_argument("--arms", default="", help="name=zip,... run in order in one session; overrides --zip")
    ap.add_argument("--out", default=ROOT / "build/kaggle_eval", type=Path)
    ap.add_argument("--slug", default="gemma4-agent-traces")
    ap.add_argument("--tasks-per-repo", default="fastapi=10,rich=7,requests=2,httpx=1")
    ap.add_argument("--task-ids", default="", help="explicit list; overrides --tasks-per-repo")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--model", default=models.DEFAULT, help="checkpoint short name (31b, 12b, e4b) or full name")
    ap.add_argument("--max-minutes", type=float, default=None,
                    help="per-task cap for every arm; needed for packages without eval_config (scorer default: none)")
    ap.add_argument("--interleave", action="store_true", help="alternate the arms per task under --budget-min")
    ap.add_argument("--budget-min", type=float, default=630, help="interleaved: minutes for the run cell")
    ap.add_argument("--round-min", type=float, default=30, help="interleaved: stop when less than this is left")
    args = ap.parse_args()
    if args.interleave and args.max_minutes:
        ap.error("--interleave runs each package with its own eval_config; drop --max-minutes")

    serve = serve_cell(args.model)
    arm_zips = dict(item.split("=", 1) for item in args.arms.split(",") if item) or {"main": str(args.zip)}
    arms = {}
    for name, path in arm_zips.items():
        data = Path(path).read_bytes()
        arms[name] = (hashlib.sha256(data).hexdigest(), base64.b64encode(data).decode())
    runner = (ROOT / "lab/src/lab/eval/run_eval.py").read_bytes()
    task_ids = [t for t in args.task_ids.split(",") if t] or pick_tasks(args.tasks_per_repo, args.seed)
    materialize = MATERIALIZE.format(
        arms=arms, runner_sha=hashlib.sha256(runner).hexdigest(), runner_b64=base64.b64encode(runner).decode(),
        task_ids=task_ids, max_minutes=args.max_minutes, budget_min=args.budget_min, round_min=args.round_min)
    run_cells = [cell(MERGE), cell(RUN_INTERLEAVED)] if args.interleave else [cell(RUN)]
    notebook = {
        "cells": with_cell_ids([{"cell_type": "markdown", "metadata": {}, "source": [
                      f"# {args.slug}\n\nRuns the mini-palace agent on {len(task_ids)} public tasks with the scorer's "
                      f"stack (vLLM + {models.name(args.model)} on 4x L4) and saves `results.tar.gz` "
                      "(task results, patches, ATIF traces, run metadata)."]},
                  cell(SETUP), cell(materialize), cell(serve), *run_cells]),
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "eval.ipynb").write_text(json.dumps(notebook, indent=1))
    meta = {"id": f"{OWNER}/{args.slug}", "title": args.slug.replace("-", " "), "code_file": "eval.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
            "enable_tpu": False, "enable_internet": False, "dataset_sources": [WHEELHOUSE], "kernel_sources": [],
            "competition_sources": [COMPETITION], "model_sources": [models.source(args.model)], "docker_image": DOCKER_IMAGE,
            "machine_shape": "NvidiaL4"}
    (args.out / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    print(f"kernel {meta['id']} -> {args.out} | arms {', '.join(f'{n}={v[0][:12]}' for n, v in arms.items())} "
          f"| {'interleaved, budget ' + str(args.budget_min) + ' min | ' if args.interleave else ''}"
          f"{len(task_ids)} tasks: {','.join(task_ids)}")


if __name__ == "__main__":
    main()
