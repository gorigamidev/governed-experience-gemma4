"""LoRA canary v2 (doc 03, 14.22): does the scorer's wheelhouse of 30 Sep apply full adapters and serve long prompts?

Three rank-16 adapters are generated inside the notebook from the real weight shapes of the served checkpoint
(compressed-tensors keeps each quantized module's logical shape in `weight_shape`): attention in every layer,
MLP in every layer, and q/o of the last layer only. Each must change greedy outputs; a ~20k-token prompt must come
back for the base and every adapter; the KV cache vLLM reports must hold at least one full 32,768-token request.

Usage: PYTHONPATH=lab/src python3 -m lab.export.kaggle_canary2_notebook
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lab.export import models
from lab.export.kaggle_eval_notebook import (COMPETITION, DOCKER_IMAGE, OWNER, SETUP, WHEELHOUSE, cell, serve_cell)
from lab.export.kaggle_notebook import with_cell_ids

ROOT = Path(__file__).resolve().parents[4]

GENERATE = r'''import json, re, shutil
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file

WORK = Path('/kaggle/working')
CKPT = Path('__CKPT_PATH__')
RANK, SEED = 16, 2026_10_03
ATTN, MLP = ('q_proj', 'k_proj', 'v_proj', 'o_proj'), ('gate_proj', 'up_proj', 'down_proj')

# Logical shape of every linear module of the text model, read from the checkpoint itself. The checkpoint has no
# model.safetensors.index.json (first run of this canary, 3 Oct): read every shard there is.
shards = sorted(CKPT.glob('*.safetensors'))
print('checkpoint files:', sorted(p.name for p in CKPT.iterdir())[:20])
assert shards, f'no safetensors in {CKPT}'
shapes = {}
for shard in shards:
    with safe_open(str(shard), framework='pt') as f:
        for key in f.keys():
            m = re.match(r'(.*language_model\.layers\.(\d+)\.\w+\.(\w+_proj))\.(weight_shape|weight)$', key)
            if not m:
                continue
            module, kind = m.group(1), m.group(4)
            if kind == 'weight_shape':
                shapes[module] = tuple(int(v) for v in f.get_tensor(key).tolist())
            elif module not in shapes and len(f.get_slice(key).get_shape()) == 2:
                shapes[module] = tuple(f.get_slice(key).get_shape())
if not shapes:
    with safe_open(str(shards[0]), framework='pt') as f:
        print('sample keys:', sorted(f.keys())[:40])
assert shapes, 'no language_model linear modules found in the checkpoint'
layers = sorted({int(re.search(r'layers\.(\d+)\.', k).group(1)) for k in shapes})
print('modules with a shape:', len(shapes), '| layers', layers[0], '-', layers[-1])

def build(name, keep):
    gen = torch.Generator().manual_seed(SEED + len(name))
    tensors, targets = {}, set()
    for module, (out_dim, in_dim) in sorted(shapes.items()):
        layer = int(re.search(r'layers\.(\d+)\.', module).group(1))
        proj = module.rsplit('.', 1)[1]
        if not keep(layer, proj):
            continue
        targets.add(proj)
        prefix = 'base_model.model.' + module
        tensors[prefix + '.lora_A.weight'] = (torch.randn(RANK, in_dim, generator=gen) * 0.1).to(torch.bfloat16)
        tensors[prefix + '.lora_B.weight'] = (torch.randn(out_dim, RANK, generator=gen) * 1.0).to(torch.bfloat16)
    out = WORK / 'canary2_submission' / 'adapters' / name
    out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out / 'adapter_model.safetensors'))
    config = {'peft_type': 'LORA', 'task_type': 'CAUSAL_LM', 'r': RANK, 'lora_alpha': 2 * RANK, 'lora_dropout': 0.0,
              'bias': 'none', 'target_modules': sorted(targets), 'base_model_name_or_path': 'google/__MODEL_NAME__',
              'inference_mode': True, 'fan_in_fan_out': False, 'use_rslora': False, 'use_dora': False}
    (out / 'adapter_config.json').write_text(json.dumps(config, indent=1))
    size = (out / 'adapter_model.safetensors').stat().st_size
    print(f'{name}: {len(tensors) // 2} modules, targets {sorted(targets)}, {size / 2**20:.1f} MiB')

shutil.rmtree(WORK / 'canary2_submission', ignore_errors=True)
build('attn_all', lambda layer, proj: proj in ATTN)
build('mlp_all', lambda layer, proj: proj in MLP)
build('last_layer', lambda layer, proj: layer == layers[-1] and proj in ('q_proj', 'o_proj'))
sub = WORK / 'canary2_submission'
'''

PROBE = r'''import json, re, time, urllib.request

SHORT = [
    'Write a Python function that returns the sum of a list of integers.',
    'In one sentence, what does git rebase do?',
    'List three HTTP status codes and their meaning.',
]
filler = '\n'.join(f'record {i:05d}: sensor reading {(i * 7919) % 1000:03d} within normal range' for i in range(1100))  # ~19.8k Gemma tokens (18 per line, measured)
NEEDLE = 'record 00777: the maintenance code is PAPAYA-QUOKKA'
LONG = filler.replace('record 00777: sensor reading', NEEDLE + ' ; sensor reading', 1) + \
    '\n\nWhat is the maintenance code in record 00777? Answer with the code only.'

def complete(model, prompt, max_tokens):
    body = json.dumps({'model': model, 'messages': [{'role': 'user', 'content': prompt}], 'temperature': 0,
                       'max_tokens': max_tokens, 'seed': 0,
                       'chat_template_kwargs': {'enable_thinking': False}}).encode()
    req = urllib.request.Request(server.base_url.rstrip('/') + '/chat/completions', data=body,
                                 headers={'Content-Type': 'application/json'})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            res = json.load(r)
        return res['choices'][0]['message'].get('content') or '', res.get('usage', {}).get('prompt_tokens'), round(time.time() - t0, 1)
    except Exception as e:  # a hang or an unknown adapter is a finding, not a crash
        return f'ERROR {type(e).__name__}: {e}', None, round(time.time() - t0, 1)

names = ('base', 'attn_all', 'mlp_all', 'last_layer')
model_of = {'base': SERVED_MODEL, 'attn_all': 'attn_all', 'mlp_all': 'mlp_all', 'last_layer': 'last_layer'}
try:
    print('served models:', [m['id'] for m in json.load(urllib.request.urlopen(server.base_url.rstrip('/') + '/models')).get('data', [])])
    short = {n: [complete(model_of[n], p, 64)[0] for p in SHORT] for n in names}
    long = {n: complete(model_of[n], LONG, 32) for n in names}
    log = open(server.log_path, encoding='utf-8', errors='replace').read() if getattr(server, 'log_path', None) else ''
finally:
    server.stop() if hasattr(server, 'stop') else None

kv_lines = [ln.strip() for ln in log.splitlines() if re.search(r'KV cache|max-loras|max_loras|max-lora-rank|Maximum concurrency', ln)]
kv_tokens = [int(m.replace(',', '')) for ln in kv_lines for m in re.findall(r'KV cache size: ([\d,]+) tokens', ln)]
applied = {n: sum(short[n][i] != short['base'][i] for i in range(len(SHORT))) for n in names[1:]}
long_ok = {n: not long[n][0].startswith('ERROR') for n in names}
assert NEEDLE in LONG
summary = {
    'applied': {n: applied[n] >= 2 for n in applied}, 'differs': applied, 'long_ok': long_ok,
    'long': {n: {'answer': long[n][0][:120], 'prompt_tokens': long[n][1], 'seconds': long[n][2]} for n in names},
    'kv_cache_tokens': kv_tokens, 'kv_ok': bool(kv_tokens) and min(kv_tokens) >= 32768, 'log_lines': kv_lines[:12],
    'short': {n: [t[:160] for t in short[n]] for n in names},
}
if not kv_lines and not any(long_ok.values()):
    verdict = 'inconclusive: server did not serve'
elif all(summary['applied'].values()) and all(long_ok.values()) and summary['kv_ok']:
    verdict = 'adapters usable'
else:
    failing = [f'{n} not applied' for n, ok in summary['applied'].items() if not ok]
    failing += [f'long prompt failed for {n}' for n, ok in long_ok.items() if not ok]
    failing += [] if summary['kv_ok'] else [f'KV cache {kv_tokens} < 32768 or not reported']
    verdict = 'not usable: ' + '; '.join(failing)
summary['verdict'] = verdict
(WORK / 'canary2.json').write_text(json.dumps(summary, indent=1))
for ln in kv_lines[:12]:
    print('LOG', ln[:200])
for n in names:
    print(n, '| short differs', applied.get(n, '-'), '| long', long[n][1], 'tokens', long[n][2], 's ->', repr(long[n][0][:80]))
print('VERDICT:', verdict)
'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=ROOT / "build/kaggle_canary2", type=Path)
    ap.add_argument("--slug", default="gemma4-lora-canary-v2")
    ap.add_argument("--model", default=models.DEFAULT, help="checkpoint short name (31b, 12b, e4b) or full name")
    args = ap.parse_args()
    generate = (GENERATE.replace("__CKPT_PATH__", models.path(args.model))
                .replace("__MODEL_NAME__", models.name(args.model)))
    notebook = {
        "cells": with_cell_ids([{"cell_type": "markdown", "metadata": {}, "source": [
                      f"# {args.slug}\n\nDoes the wheelhouse of 30 Sep apply full LoRA adapters and serve long "
                      "prompts? Three rank-16 adapters built from the checkpoint's real shapes (attention, MLP, last "
                      "layer), greedy comparisons against the base, a ~20k-token prompt, and the KV cache size from "
                      "the vLLM log. Writes `canary2.json`."]},
                  cell(SETUP), cell(generate), cell(serve_cell(args.model)), cell(PROBE)]),
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "canary2.ipynb").write_text(json.dumps(notebook, indent=1))
    meta = {"id": f"{OWNER}/{args.slug}", "title": args.slug.replace("-", " "), "code_file": "canary2.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
            "enable_tpu": False, "enable_internet": False, "dataset_sources": [WHEELHOUSE], "kernel_sources": [],
            "competition_sources": [COMPETITION], "model_sources": [models.source(args.model)], "docker_image": DOCKER_IMAGE,
            "machine_shape": "NvidiaL4"}
    (args.out / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    print(f"kernel {meta['id']} -> {args.out}")


if __name__ == "__main__":
    main()
