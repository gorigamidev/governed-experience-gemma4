"""Build a Kaggle GPU notebook that checks whether the scorer's vLLM applies LoRA weights (forum 743508).

Same install and vLLM setup as the evaluation notebook (SETUP and SERVE are shared). Two adapters are served
next to the base model, both with the module names, shapes and config of the organizers' sample main_lora:
  zero_lora   : the sample adapter itself (lora_B is all zeros), a negative control that must match the base;
  canary_lora : same layout with large random lora_A and lora_B, which must change greedy outputs if applied.
Decision rule (fixed before running): canary differs from base on at least 2 of 3 prompts and zero matches base on
all 3 -> adapters are applied. Canary matches base on all 3 -> weights are dropped. Anything else -> inconclusive.

Usage: PYTHONPATH=lab/src python3 -m lab.export.kaggle_canary_notebook
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import random
import struct
from pathlib import Path

from lab.export import models
from lab.export.kaggle_eval_notebook import COMPETITION, OWNER, SETUP, WHEELHOUSE, cell, serve_cell
from lab.export.kaggle_notebook import with_cell_ids

ROOT = Path(__file__).resolve().parents[4]
SAMPLE = ROOT / "data/competition/sample_submission/adapters/main_lora"
SEED = 743508
SCALE_A, SCALE_B = 0.1, 1.0


def read_safetensors_header(blob: bytes) -> dict:
    size = struct.unpack("<Q", blob[:8])[0]
    header = json.loads(blob[8:8 + size])
    header.pop("__metadata__", None)
    return header


def bf16(values: list[float]) -> bytes:
    """Truncate float32 to bfloat16 (upper 16 bits), little endian."""
    return b"".join(struct.pack("<f", v)[2:] for v in values)


def canary_weights(sample: bytes) -> bytes:
    """Same tensor names, shapes and dtype as the sample; nonzero lora_A and lora_B from a fixed seed."""
    header = read_safetensors_header(sample)
    rng = random.Random(SEED)
    names = sorted(header)
    out_header, chunks, offset = {}, [], 0
    for name in names:
        spec = header[name]
        if spec["dtype"] != "BF16":
            raise SystemExit(f"canary: expected BF16 in the sample, got {spec['dtype']} for {name}")
        rows, cols = spec["shape"]
        scale = SCALE_B if ".lora_B." in name else SCALE_A
        data = bf16([rng.gauss(0.0, scale) for _ in range(rows * cols)])
        out_header[name] = {"dtype": "BF16", "shape": [rows, cols], "data_offsets": [offset, offset + len(data)]}
        chunks.append(data)
        offset += len(data)
    blob = json.dumps(out_header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    return struct.pack("<Q", len(blob)) + blob + b"".join(chunks)


MATERIALIZE = r'''import base64, hashlib, shutil
from pathlib import Path

WORK = Path('/kaggle/working')
SAMPLE = Path('/kaggle/input/competitions/gemma-4-developer-agent/sample_submission/adapters/main_lora')
CANARY_SHA256 = "{canary_sha}"
CANARY_B64 = "{canary_b64}"
SAMPLE_SHA256 = "{sample_sha}"

sub = WORK / 'canary_submission'
shutil.rmtree(sub, ignore_errors=True)
zero, canary = sub / 'adapters' / 'zero_lora', sub / 'adapters' / 'canary_lora'
shutil.copytree(SAMPLE, zero)
assert hashlib.sha256((zero / 'adapter_model.safetensors').read_bytes()).hexdigest() == SAMPLE_SHA256, 'sample changed'
canary.mkdir(parents=True)
shutil.copy(SAMPLE / 'adapter_config.json', canary / 'adapter_config.json')
data = base64.b64decode(CANARY_B64)
assert hashlib.sha256(data).hexdigest() == CANARY_SHA256
(canary / 'adapter_model.safetensors').write_bytes(data)
print('adapters', sorted(p.name for p in (sub / 'adapters').iterdir()), '| canary', CANARY_SHA256[:12])
'''

PROBE = r'''import json, urllib.request

PROMPTS = [
    'Write a Python function that returns the sum of a list of integers.',
    'In one sentence, what does git rebase do?',
    'List three HTTP status codes and their meaning.',
]

def complete(model, prompt):
    body = json.dumps({'model': model, 'messages': [{'role': 'user', 'content': prompt}], 'temperature': 0,
                       'max_tokens': 64, 'seed': 0,
                       'chat_template_kwargs': {'enable_thinking': False}}).encode()
    req = urllib.request.Request(server.base_url.rstrip('/') + '/chat/completions', data=body,
                                 headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.load(r)['choices'][0]['message'].get('content') or ''
    except Exception as e:  # an adapter the server does not know is a finding, not a crash
        return f'ERROR {type(e).__name__}: {e}'

try:
    models = json.load(urllib.request.urlopen(server.base_url.rstrip('/') + '/models'))
    print('served models:', [m['id'] for m in models.get('data', [])])
    out = {name: [complete(model, p) for p in PROMPTS]
           for name, model in (('base', SERVED_MODEL), ('zero_lora', 'zero_lora'), ('canary_lora', 'canary_lora'))}
finally:
    server.stop() if hasattr(server, 'stop') else None

for i, p in enumerate(PROMPTS):
    print(f'--- prompt {i}: {p}')
    for name in out:
        print(f'[{name}]', repr(out[name][i][:200]))
canary_diff = sum(out['canary_lora'][i] != out['base'][i] for i in range(len(PROMPTS)))
zero_same = sum(out['zero_lora'][i] == out['base'][i] for i in range(len(PROMPTS)))
errors = [n for n in out if any(t.startswith('ERROR') for t in out[n])]
if errors:
    verdict = f'inconclusive: request errors for {errors}'
elif canary_diff >= 2 and zero_same == len(PROMPTS):
    verdict = 'adapters applied'
elif canary_diff == 0:
    verdict = 'adapter weights dropped (canary == base)'
else:
    verdict = 'inconclusive'
summary = {'verdict': verdict, 'canary_differs': canary_diff, 'zero_matches': zero_same, 'outputs': out}
(WORK / 'canary.json').write_text(json.dumps(summary, indent=1))
print('VERDICT:', verdict, '| canary differs', canary_diff, '/ 3 | zero matches', zero_same, '/ 3')
'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=ROOT / "build/kaggle_canary", type=Path)
    ap.add_argument("--slug", default="gemma4-lora-canary")
    ap.add_argument("--model", default=models.DEFAULT, help="checkpoint short name (31b, 12b, e4b) or full name")
    args = ap.parse_args()

    sample = (SAMPLE / "adapter_model.safetensors").read_bytes()
    canary = canary_weights(sample)
    materialize = MATERIALIZE.format(canary_sha=hashlib.sha256(canary).hexdigest(),
                                     canary_b64=base64.b64encode(canary).decode(),
                                     sample_sha=hashlib.sha256(sample).hexdigest())
    notebook = {
        "cells": with_cell_ids([{"cell_type": "markdown", "metadata": {}, "source": [
                      f"# {args.slug}\n\nDoes the scorer's vLLM apply LoRA weights? Serves the base model, the sample "
                      "adapter (zero lora_B, negative control) and a canary adapter with the same layout and large "
                      "weights, then compares greedy outputs. Writes `canary.json`."]},
                  cell(SETUP), cell(materialize), cell(serve_cell(args.model)), cell(PROBE)]),
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "canary.ipynb").write_text(json.dumps(notebook, indent=1))
    meta = {"id": f"{OWNER}/{args.slug}", "title": args.slug.replace("-", " "), "code_file": "canary.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
            "enable_tpu": False, "enable_internet": False, "dataset_sources": [WHEELHOUSE], "kernel_sources": [],
            "competition_sources": [COMPETITION], "model_sources": [models.source(args.model)], "machine_shape": "NvidiaL4"}
    (args.out / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    print(f"kernel {meta['id']} -> {args.out} | canary sha {hashlib.sha256(canary).hexdigest()[:12]}")


if __name__ == "__main__":
    main()
