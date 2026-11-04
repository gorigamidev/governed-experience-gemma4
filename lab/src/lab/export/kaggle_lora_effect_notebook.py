"""Effect of a trained adapter (doc 03 §14.25, rule 4), without retraining.

Reads the adapter from the output of `gorigamidev/gemma4-lora-<model>` (kernel source), checks its tensors and the
official validator, serves the matching W4A16 checkpoint (TP 4) with it, and runs the 16 reserve tasks with the
package and the package + `main_lora`, interleaved per task (adapter arm first) and saved after every run: a
session cut short by the GPU quota still leaves complete pairs. Rule 4 is applied locally on the downloaded traces.

Usage: PYTHONPATH=lab/src python3 -m lab.export.kaggle_lora_effect_notebook
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
PACKAGE = ROOT / "build/top-mix-1k/submission.zip"


def arm_base(package: Path) -> str:
    """Arm name stem from the package path: build/<name>/submission.zip -> <name>."""
    return package.parent.name


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


ARMS = r'''import base64, hashlib, io, json, re, shutil, zipfile
from pathlib import Path
from safetensors import safe_open
WORK = Path('/kaggle/working')
INPUT = Path('/kaggle/input')
SERVED = Path('__SERVED_PATH__')
if not SERVED.exists():
    SERVED = next(p.parent for p in INPUT.rglob('config.json') if '__MODEL_TOKEN__' in str(p))
# The adapter written by the matching LoRA notebook: prefer its packaged copy, fall back to the raw training output.
found = sorted(INPUT.rglob('adapter_model.safetensors'), key=lambda p: ('lora_submission' not in str(p), str(p)))
assert found, 'no adapter_model.safetensors in the inputs: the LoRA notebook output was not kept'
ADAPTER_SRC = found[0].parent
print('adapter from', ADAPTER_SRC, '| candidates', [str(p) for p in found])
with safe_open(str(ADAPTER_SRC / 'adapter_model.safetensors'), framework='pt') as f:
    keys = list(f.keys())
modules = {re.sub(r'^base_model\.model\.', '', k.rsplit('.lora_', 1)[0]) for k in keys}
served_modules = set()
for shard in sorted(SERVED.glob('*.safetensors')):
    with safe_open(str(shard), framework='pt') as f:
        served_modules |= {m.group(1) for k in f.keys() if (m := re.match(r'(.*language_model\.layers\.\d+\.\w+\.\w+_proj)\.', k))}
MISSING = sorted(modules - served_modules)
print('adapter tensors', len(keys), '| modules', len(modules), '| not in served checkpoint', MISSING[:5])
assert keys and not MISSING, 'adapter empty or not matching the served checkpoint'

ARMS = {'__ARM__-lora': WORK / 'arms' / '__ARM__-lora', '__ARM__': WORK / 'arms' / '__ARM__'}
for arm in ARMS.values():
    shutil.rmtree(arm, ignore_errors=True)
    zipfile.ZipFile(io.BytesIO(base64.b64decode(PACKAGE_B64))).extractall(arm)
lora = ARMS['__ARM__-lora']
dest = lora / 'adapters' / 'main_lora'
dest.mkdir(parents=True)
shutil.copy(ADAPTER_SRC / 'adapter_model.safetensors', dest / 'adapter_model.safetensors')
cfg = json.loads((ADAPTER_SRC / 'adapter_config.json').read_text())
cfg['target_modules'] = sorted({m.rsplit('.', 1)[1] for m in modules})
cfg['base_model_name_or_path'] = 'google/__MODEL_NAME__'
(dest / 'adapter_config.json').write_text(json.dumps(cfg, indent=1))
agent = (lora / 'agent.yaml').read_text()
(lora / 'agent.yaml').write_text(re.sub(r'^(model: [^\n]*\n)', r'\1adapter: main_lora\n', agent, count=1, flags=re.M))
code = base64.b64decode(RUN_EVAL_B64)
assert hashlib.sha256(code).hexdigest() == RUN_EVAL_SHA256
(WORK / 'run_eval.py').write_bytes(code)
print('arms', list(ARMS), '| adapter_config', {k: cfg[k] for k in ('r', 'lora_alpha', 'target_modules')})
'''

VALIDATE = r'''from adk_submission import compile_submission, discover_adapters, validate_directory
from adk_submission.registry import ModelRegistry
from swegemma.config import build_submission_limits
# The steps of `make package`; they check the package, not the tensors (doc 01, row 52): those were loaded above.
HARNESS_TOOLS = ['run_command', 'read_file', 'write_file', 'edit_file', 'submit_patch', 'get_status',
                 'search_similar_code', 'get_code_neighbors', 'get_code_subgraph']
MODEL = '__MODEL_NAME__'
limits, gen = build_submission_limits()
validate_directory(lora, limits)
manifest = discover_adapters(lora, limits.adapter_extensions)
models = ModelRegistry()
models.register(MODEL, MODEL)
for name in manifest.adapters:
    models.register(name, MODEL)
compile_submission(submission_dir=lora, tool_registry={t: (lambda **_: '') for t in HARNESS_TOOLS}, model_registry=models,
                   limits=limits, generation_constraints=gen, adapter_manifest=manifest)
print('validator: ok, adapters', sorted(manifest.adapters))
'''

RUN = r'''import subprocess, sys, tarfile, time
import litellm, torch
from adk_submission import VllmConfig, VllmServer
litellm.drop_params = True
server = VllmServer(VllmConfig(
    model=str(SERVED), port=8000, host='127.0.0.1', tool_call_parser='gemma4', reasoning_parser='gemma4',
    max_model_len=32768, dtype='bfloat16', gpu_memory_utilization=0.90, enable_auto_tool_choice=True,
    enable_lora=True, max_loras=8, max_lora_rank=128, __EXTRA_ARGS__tensor_parallel_size=4, startup_timeout=60 * 20,
), adapter_manifest=manifest)
server.start()
SERVED_MODEL = getattr(server, 'effective_model_path', str(SERVED))
print('vLLM up at', server.base_url, '| served', SERVED_MODEL)
COMP = INPUT / 'competitions' / 'gemma-4-developer-agent'
EFFECT = WORK / 'effect'
done = []

def save():
    if EFFECT.exists():
        with tarfile.open(WORK / 'effect.tar.gz', 'w:gz') as tar:
            tar.add(EFFECT, arcname='effect', filter=lambda ti: None if '/submission/' in ti.name + '/' else ti)
    (WORK / 'effect_progress.json').write_text(json.dumps({'done': done}, indent=1))

try:
    for task in RESERVE_IDS:  # interleaved: a cut session still leaves complete pairs
        for name, arm in ARMS.items():
            cmd = [sys.executable, str(WORK / 'run_eval.py'), '--submission', str(arm), '--tasks', str(COMP / 'tasks.jsonl'),
                   '--snapshots', str(COMP / 'snapshots'), '--graphs', str(COMP / 'graphs'), '--embeddings', str(COMP / 'embeddings'),
                   '--wheels', str(COMP / 'wheels'), '--results', str(EFFECT / name / task), '--task-ids', task,
                   '--sandbox', 'subprocess', '--api-base', server.base_url, '--served-model', SERVED_MODEL,
                   '--model-prefix', 'openai/', '--label', f'kaggle-lora-31b-{name}', '--hardware', '4xL4']
            t0 = time.time()
            subprocess.run(cmd, check=False, cwd=WORK)
            done.append({'task': task, 'arm': name, 'minutes': round((time.time() - t0) / 60, 2)})
            print(task, name, done[-1]['minutes'], 'min', flush=True)
            save()
finally:
    server.stop() if hasattr(server, 'stop') else None
    save()
print('pairs complete:', sum(1 for t in RESERVE_IDS if sum(d['task'] == t for d in done) == 2), 'of', len(RESERVE_IDS))
'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=ROOT / "build/kaggle_lora_effect", type=Path)
    ap.add_argument("--slug", default="gemma4-lora-effect")
    ap.add_argument("--package", type=Path, default=PACKAGE,
                    help="submission.zip both arms are built from; the arm names come from its directory name")
    ap.add_argument("--model", default=models.DEFAULT, help="checkpoint short name (31b, 12b, e4b) or full name")
    args = ap.parse_args()
    sys.path.insert(0, str(ROOT / "lab/src"))
    from lab.lora.scripted import RESERVE
    base = arm_base(args.package)
    source_kernel = f"gorigamidev/gemma4-lora-{models.short(args.model)}"
    package = args.package.read_bytes()
    model_extra = models.extra_args(args.model)
    arms = (ARMS.replace("__SERVED_PATH__", models.path(args.model))
            .replace("__MODEL_TOKEN__", models.token(args.model))
            .replace("__MODEL_NAME__", models.name(args.model))
            .replace("__ARM__", base))
    validate = VALIDATE.replace("__MODEL_NAME__", models.name(args.model))
    run = RUN.replace("__EXTRA_ARGS__", "" if not model_extra else f"extra_args={list(model_extra)!r}, ")
    runner = (ROOT / "lab/src/lab/eval/run_eval.py").read_bytes()
    consts = (f"PACKAGE_SHA256 = {hashlib.sha256(package).hexdigest()!r}  # {_rel(args.package)}\n"
              f"PACKAGE_B64 = {base64.b64encode(package).decode()!r}\n"
              f"RUN_EVAL_SHA256 = {hashlib.sha256(runner).hexdigest()!r}\n"
              f"RUN_EVAL_B64 = {base64.b64encode(runner).decode()!r}\n"
              f"RESERVE_IDS = {sorted(RESERVE)!r}\n")
    notebook = {
        "cells": with_cell_ids([{"cell_type": "markdown", "metadata": {}, "source": [
                      f"# {args.slug}\n\nRule 4 of doc 03 §14.25 with the adapter trained by `{source_kernel}`: "
                      f"`{base}` + `main_lora` against `{base}` on the 16 reserve tasks, interleaved per task "
                      "and saved after every run. Writes `effect.tar.gz` and `effect_progress.json`."]},
                  cell(smoke.VERSIONS), cell(smoke.SETUP), cell(consts),
                  cell(arms), cell(validate), cell(run)]),
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "effect.ipynb").write_text(json.dumps(notebook, indent=1))
    meta = {"id": f"{OWNER}/{args.slug}", "title": args.slug.replace("-", " "), "code_file": "effect.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
            "enable_tpu": False, "enable_internet": False, "dataset_sources": [WHEELHOUSE],
            "kernel_sources": [source_kernel], "competition_sources": [COMPETITION],
            "model_sources": [models.source(args.model)], "docker_image": DOCKER_IMAGE, "machine_shape": "NvidiaL4"}
    (args.out / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    print(f"kernel {meta['id']} -> {args.out} | package sha256 {hashlib.sha256(package).hexdigest()[:12]}")


if __name__ == "__main__":
    main()
