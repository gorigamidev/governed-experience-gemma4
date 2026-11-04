"""31B diagnostic (doc 03 §14.26): render parity against vLLM's /tokenize, and the analyzer without thinking.

Serves the 31B W4A16 checkpoint (TP 4, no adapter) once and answers two questions:
  1. Do our training renders (`lab/lora/train.py`) match, token for token, what the server builds from the same
     requests? A fixed sample of logged coder turns is sent to /tokenize, prompt alone and prompt + response.
  2. Does `top-mix-1k-an-off` (the analyzer on the top bundle's sampling, thinking off) free task time? The arm runs
     the 16 reserve tasks one at a time, saved after each, until the session budget.
Writes `diag.json`, `parity.json`, `effect.tar.gz` and `effect_progress.json`; the rules are applied locally.

Usage: PYTHONPATH=lab/src python3 -m lab.export.kaggle_diag_notebook
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
from pathlib import Path

from lab.export import kaggle_lora_smoke_notebook as smoke
from lab.export import models
from lab.export.kaggle_eval_notebook import COMPETITION, DOCKER_IMAGE, OWNER, WHEELHOUSE, cell
from lab.export.kaggle_notebook import with_cell_ids

ROOT = Path(__file__).resolve().parents[4]
ARM = "top-mix-1k-an-off"
LORA_DATA = "gorigamidev/gemma4-lora-data"

SETUP_RUN = r'''import base64, hashlib, io, json, subprocess, sys, tarfile, time, zipfile
from pathlib import Path
T0 = time.time()
BUDGET_MIN = __BUDGET_MIN__
TASK_MIN = 6.5  # a task starts only with this much left (4.5 min cap, setup and teardown)
def minutes_left():
    return BUDGET_MIN - (time.time() - T0) / 60
WORK = Path('/kaggle/working')

def run_logged(cmd, log_name):
    lines = []
    with open(WORK / log_name, 'a') as log, subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                                             text=True, bufsize=1) as p:
        for line in p.stdout:
            print(line, end='', flush=True)
            log.write(line)
            log.flush()
            lines.append(line)
    return p.returncode, ''.join(lines)

DATA_DIR = next(p.parent for p in Path('/kaggle/input').rglob('plans.json') if (p.parent / 'reserve_requests.jsonl').exists())
SERVED = Path('__SERVED_PATH__')
if not SERVED.exists():
    SERVED = next(p.parent for p in Path('/kaggle/input').rglob('config.json') if '__MODEL_TOKEN__' in str(p))
(WORK / 'train.py').write_text(TRAIN_PY)
ARM_DIR = WORK / 'arms' / ARM
pkg = base64.b64decode(PACKAGE_B64)
assert hashlib.sha256(pkg).hexdigest() == PACKAGE_SHA256
zipfile.ZipFile(io.BytesIO(pkg)).extractall(ARM_DIR)
code = base64.b64decode(RUN_EVAL_B64)
assert hashlib.sha256(code).hexdigest() == RUN_EVAL_SHA256
(WORK / 'run_eval.py').write_bytes(code)
print('data', DATA_DIR, '| served', SERVED, '| arm', ARM, PACKAGE_SHA256[:12], '| budget', BUDGET_MIN, 'min')
'''

SERVE = r'''import litellm, torch
from adk_submission import VllmConfig, VllmServer
litellm.drop_params = True
server = VllmServer(VllmConfig(
    model=str(SERVED), port=8000, host='127.0.0.1', tool_call_parser='gemma4', reasoning_parser='gemma4',
    max_model_len=32768, dtype='bfloat16', gpu_memory_utilization=0.90, enable_auto_tool_choice=True,
__EXTRA_ARGS__    tensor_parallel_size=4, startup_timeout=60 * 20,
))
server.start()
SERVED_MODEL = getattr(server, 'effective_model_path', str(SERVED))
print('vLLM up at', server.base_url, '| served', SERVED_MODEL, '| minutes left', round(minutes_left()))
'''

PARITY = r'''import urllib.request
sys.path.insert(0, str(WORK))
from transformers import AutoTokenizer
from train import as_vllm, coder_turns
tok = AutoTokenizer.from_pretrained(str(SERVED))

def tokenize(body):
    req = urllib.request.Request(server.base_url.rstrip('/').removesuffix('/v1') + '/tokenize', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)['tokens']

def ids(x):
    # apply_chat_template(tokenize=True) returns a BatchEncoding, which is not a dict: list() of it gives its keys.
    return list(x['input_ids'] if hasattr(x, 'keys') else x)

turns = coder_turns(DATA_DIR)
by_task = {}
for t in turns:
    task = t['variant'].split('-')[0]
    by_task.setdefault(task, {}).setdefault(t['variant'], []).append(t)
sample = []
for task in sorted(by_task):
    first = sorted(by_task[task])[0]
    ts = sorted(by_task[task][first], key=lambda t: t['turn'])
    calls = [t for t in ts if t['response'].get('tool_calls')]
    picks = {ts[0]['turn'], ts[1]['turn'] if len(ts) > 1 else ts[0]['turn'], calls[-1]['turn']}
    sample += [t for t in ts if t['turn'] in picks]
results = []
for t in sample:
    req = t['request']
    kw = req.get('chat_template_kwargs') or {}
    row = {'variant': t['variant'], 'turn': t['turn']}
    try:
        for kind, msgs, gen in (('prompt', req['messages'], True), ('full', req['messages'] + [t['response']], False)):
            ours = ids(tok.apply_chat_template(as_vllm(msgs), tools=req.get('tools'), add_generation_prompt=gen, tokenize=True, **kw))
            theirs = tokenize({'model': SERVED_MODEL, 'messages': msgs, 'tools': req.get('tools'),
                               'add_generation_prompt': gen, 'chat_template_kwargs': kw})
            diff = next((i for i, (a, b) in enumerate(zip(ours, theirs)) if a != b), None)
            if diff is None and len(ours) != len(theirs):
                diff = min(len(ours), len(theirs))
            row[kind] = {'ours': len(ours), 'server': len(theirs), 'equal': ours == theirs, 'first_diff': diff,
                         'ours_text': tok.decode(ours[max(0, (diff or 0) - 20):(diff or 0) + 30]) if diff is not None else None,
                         'server_text': tok.decode(theirs[max(0, (diff or 0) - 20):(diff or 0) + 30]) if diff is not None else None}
    except Exception as e:
        row['error'] = f'{type(e).__name__}: {e}'[:400]
    results.append(row)
ok = [r for r in results if 'error' not in r]
equal_prompt = sum(r['prompt']['equal'] for r in ok)
equal_full = sum(r['full']['equal'] for r in ok)
if not ok:
    PARITY_VERDICT = 'not measured'
elif equal_prompt == len(results) and equal_full == len(results):
    PARITY_VERDICT = 'parity'
else:
    PARITY_VERDICT = 'difference'
(WORK / 'parity.json').write_text(json.dumps({'verdict': PARITY_VERDICT, 'sample': len(results), 'errors': len(results) - len(ok),
                                              'prompt_equal': equal_prompt, 'full_equal': equal_full, 'rows': results}, indent=1))
print('PARITY:', PARITY_VERDICT, '| prompt', equal_prompt, '/', len(results), '| full', equal_full, '/', len(results),
      '| errors', len(results) - len(ok), '| minutes left', round(minutes_left()))
for r in results:
    if 'error' in r or not (r['prompt']['equal'] and r['full']['equal']):
        print(json.dumps(r)[:700])
        break
'''

RUN = r'''COMP = Path('/kaggle/input/competitions/gemma-4-developer-agent')
EFFECT = WORK / 'effect'
done = []

def save():
    if EFFECT.exists():
        with tarfile.open(WORK / 'effect.tar.gz', 'w:gz') as tar:
            tar.add(EFFECT, arcname='effect', filter=lambda ti: None if '/submission/' in ti.name + '/' else ti)
    (WORK / 'effect_progress.json').write_text(json.dumps({'arm': ARM, 'done': done}, indent=1))

try:
    for task in RESERVE_IDS:
        if minutes_left() < TASK_MIN:
            print('time budget: stopping before', task, '| minutes left', round(minutes_left()))
            break
        cmd = [sys.executable, str(WORK / 'run_eval.py'), '--submission', str(ARM_DIR), '--tasks', str(COMP / 'tasks.jsonl'),
               '--snapshots', str(COMP / 'snapshots'), '--graphs', str(COMP / 'graphs'), '--embeddings', str(COMP / 'embeddings'),
               '--wheels', str(COMP / 'wheels'), '--results', str(EFFECT / ARM / task), '--task-ids', task,
               '--sandbox', 'subprocess', '--api-base', server.base_url, '--served-model', SERVED_MODEL,
               '--model-prefix', 'openai/', '--label', f'kaggle-diag-{ARM}', '--hardware', '4xL4']
        t0 = time.time()
        rc, _ = run_logged(cmd, 'effect.log')
        done.append({'task': task, 'rc': rc, 'minutes': round((time.time() - t0) / 60, 2)})
        print(task, done[-1]['minutes'], 'min | minutes left', round(minutes_left()), flush=True)
        save()
finally:
    server.stop() if hasattr(server, 'stop') else None
    save()
(WORK / 'diag.json').write_text(json.dumps({'parity': PARITY_VERDICT, 'arm': ARM, 'package_sha256': PACKAGE_SHA256,
                                            'tasks_run': len(done), 'minutes_left_at_end': round(minutes_left())}, indent=1))
print('tasks run:', len(done), 'of', len(RESERVE_IDS), '| parity:', PARITY_VERDICT)
'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=ROOT / "build/kaggle_diag", type=Path)
    ap.add_argument("--slug", default="gemma4-diag-analyzer")
    ap.add_argument("--budget-min", type=int, default=80)
    ap.add_argument("--tasks", default="", help="comma-separated reserve tasks to run (default: all 16)")
    ap.add_argument("--model", default=models.DEFAULT, help="checkpoint short name (31b, 12b, e4b) or full name")
    args = ap.parse_args()
    sys.path.insert(0, str(ROOT / "lab/src"))
    from lab.lora.scripted import RESERVE
    package_path = ROOT / f"build/{ARM}{models.variant_suffix(args.model)}/submission.zip"
    package = package_path.read_bytes()
    model_extra = models.extra_args(args.model)
    setup_run = (SETUP_RUN.replace("__BUDGET_MIN__", str(args.budget_min))
                 .replace("__SERVED_PATH__", models.path(args.model))
                 .replace("__MODEL_TOKEN__", models.token(args.model)))
    serve = SERVE.replace("__EXTRA_ARGS__", "" if not model_extra else f"    extra_args={list(model_extra)!r},\n")
    runner = (ROOT / "lab/src/lab/eval/run_eval.py").read_bytes()
    consts = (f"ARM = {ARM!r}\n"
              f"TRAIN_PY = {smoke.TRAIN_SRC.read_text()!r}\n"
              f"PACKAGE_SHA256 = {hashlib.sha256(package).hexdigest()!r}  # {package_path.relative_to(ROOT)}\n"
              f"PACKAGE_B64 = {base64.b64encode(package).decode()!r}\n"
              f"RUN_EVAL_SHA256 = {hashlib.sha256(runner).hexdigest()!r}\n"
              f"RUN_EVAL_B64 = {base64.b64encode(runner).decode()!r}\n"
              f"RESERVE_IDS = {sorted(t for t in args.tasks.split(',') if t) or sorted(RESERVE)!r}\n")
    notebook = {
        "cells": with_cell_ids([{"cell_type": "markdown", "metadata": {}, "source": [
                      f"# {args.slug}\n\nDiagnostic of doc 03 §14.26 on Gemma 4 {models.short(args.model).upper()} "
                      f"(W4A16, TP 4): render parity of the LoRA training data against vLLM's `/tokenize`, then "
                      f"`{ARM}` (analyzer without thinking) on the 16 reserve tasks within a fixed time budget. "
                      "Writes `parity.json`, `effect.tar.gz` and `diag.json`."]},
                  cell(smoke.VERSIONS), cell(smoke.SETUP), cell(consts),
                  cell(setup_run), cell(serve), cell(PARITY), cell(RUN)]),
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "diag.ipynb").write_text(json.dumps(notebook, indent=1))
    meta = {"id": f"{OWNER}/{args.slug}", "title": args.slug.replace("-", " "), "code_file": "diag.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
            "enable_tpu": False, "enable_internet": False, "dataset_sources": [WHEELHOUSE, LORA_DATA],
            "kernel_sources": [], "competition_sources": [COMPETITION], "model_sources": [models.source(args.model)],
            "docker_image": DOCKER_IMAGE, "machine_shape": "NvidiaL4"}
    (args.out / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    print(f"kernel {meta['id']} -> {args.out} | arm {ARM} sha256 {hashlib.sha256(package).hexdigest()[:12]}")


if __name__ == "__main__":
    main()
