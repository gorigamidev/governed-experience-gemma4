"""LoRA smoke test with Gemma 4 E4B (doc 03 §14.25, amendment 1): does the training and serving pipeline work?

Trains `lab/lora/train.py` for a few steps on toy turns (a fake repository, fabricated tool outputs: pipeline
data, never training data), on the bf16 QAT checkpoint, then serves the W4A16 checkpoint with the adapter in the
scorer's vLLM. It answers only pipeline questions: library versions in the fixed image, that the adapter loads
and changes outputs, that a long prompt still works, that the agent request still yields a parsed tool call, and
that our rendering of a turn is token-for-token what vLLM renders. It evaluates nothing about behavior.

Usage: PYTHONPATH=lab/src python3 -m lab.export.kaggle_lora_smoke_notebook
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lab.export.kaggle_eval_notebook import COMPETITION, DOCKER_IMAGE, OWNER, WHEELHOUSE, cell
from lab.export.kaggle_eval_notebook import SETUP as _EVAL_SETUP
from lab.export.kaggle_notebook import with_cell_ids

ROOT = Path(__file__).resolve().parents[4]
TRAIN_SRC = ROOT / "lab/src/lab/lora/train.py"
BF16_SOURCE = "google/gemma-4/Transformers/gemma-4-e4b-it-qat-q4_0-unquantized/2"
SERVED_SOURCE = "google/gemma-4/Other/gemma-4-e4b-it-qat-w4a16-ct/2"
LORA_WHEELS = "gorigamidev/gemma4-lora-wheels"  # peft 0.21.2 and accelerate 1.15.0, pure-Python wheels
BF16_PATH = "/kaggle/input/models/google/gemma-4/transformers/gemma-4-e4b-it-qat-q4_0-unquantized/2"
SERVED_PATH = "/kaggle/input/models/google/gemma-4/other/gemma-4-e4b-it-qat-w4a16-ct/2"

_WHEELHOUSE_LINE = "WHEELHOUSE_DIR = Path('/kaggle/input/datasets/metric/gemma-4-developer-agent-wheelhouse')\n"
assert _EVAL_SETUP.count(_WHEELHOUSE_LINE) == 1
# Smoke v1 (5 Oct) found no wheels at the documented mount path and ran `pip install` with no files: look for the
# wheelhouse by one of its wheels, list /kaggle/input if it is missing, and fail before pip.
SETUP = _EVAL_SETUP.replace(_WHEELHOUSE_LINE, _WHEELHOUSE_LINE + (
    "if not any(WHEELHOUSE_DIR.glob('*.whl')):\n"
    "    print('wheelhouse not at', WHEELHOUSE_DIR, '| /kaggle/input:', sorted(str(p) for p in Path('/kaggle/input').glob('*/*'))[:40])\n"
    "    found = sorted(Path('/kaggle/input').rglob('adk_submission-*.whl'))\n"
    "    assert found, 'wheelhouse not mounted'\n"
    "    WHEELHOUSE_DIR = found[0].parent\n"
    "print('wheelhouse:', WHEELHOUSE_DIR, len(list(WHEELHOUSE_DIR.glob('*.whl'))), 'wheels')\n"))

VERSIONS = r'''import json, subprocess
from importlib import metadata
# Read installed versions without importing: importing transformers here, before the wheelhouse replaces it, leaves
# the old modules cached and mixes two versions in one process (rehearsal v2, 5 Oct: ImportError 'hf_api').
def versions():
    out = {}
    for m in ('torch', 'transformers', 'peft', 'accelerate', 'bitsandbytes', 'safetensors', 'vllm'):
        try:
            out[m] = metadata.version(m)
        except metadata.PackageNotFoundError:
            out[m] = 'MISSING'
    return out
BEFORE = versions()
print('image, before the wheelhouse:', json.dumps(BEFORE))
print(subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv'], capture_output=True, text=True).stdout)
'''

PEFT = r'''import glob, importlib, subprocess, sys
AFTER = versions()
print('after the wheelhouse:', json.dumps(AFTER))
# The fixed image refuses internet (push with both returns 400) and ships peft 0.19.1 / accelerate 1.13.0 (smoke v1,
# 5 Oct). Install the versions verified locally with transformers 5.13.1, from a private dataset of pure-Python
# wheels, with --no-deps so the wheelhouse versions stay.
wheels = [w for m in ('peft-0.21.2', 'accelerate-1.15.0') for w in glob.glob(f'/kaggle/input/**/{m}-*.whl', recursive=True)[:1]]
assert len(wheels) == 2, f'private wheels not found: {wheels}'
print('installing', wheels)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', '--force-reinstall', *wheels], check=True)
importlib.invalidate_caches()
AFTER = versions()
assert AFTER['peft'] == '0.21.2' and AFTER['transformers'] == '5.13.1', AFTER
print('after the private wheels:', json.dumps(AFTER))
'''

TRAIN = r'''import json, subprocess, sys, time
from pathlib import Path
WORK = Path('/kaggle/working')
DATA = WORK / 'toy_data'
DATA.mkdir(parents=True, exist_ok=True)
(DATA / 'requests.jsonl').write_text(TOY_REQUESTS)
(DATA / 'plans.json').write_text(TOY_PLANS)
(WORK / 'train.py').write_text(TRAIN_PY)
BF16 = Path('__BF16_PATH__')
if not BF16.exists():  # mount path of the Transformers framework not verified yet
    BF16 = next(p.parent for p in Path('/kaggle/input').rglob('config.json') if 'e4b-it-qat-q4_0-unquantized' in str(p))
print('bf16 checkpoint:', sorted(p.name for p in BF16.iterdir()))
t0 = time.time()
proc = subprocess.run([sys.executable, str(WORK / 'train.py'), '--data', str(DATA), '--tokenizer', str(BF16),
                       '--model', str(BF16), '--out', str(WORK / 'adapter_raw'), '--max-steps', '12',
                       '--val-frac', '0.34', '--max-len', '16384'], capture_output=True, text=True)
TRAIN_SECONDS = round(time.time() - t0, 1)
print(proc.stdout[-4000:]); print(proc.stderr[-4000:])
TRAIN_OK = proc.returncode == 0 and (WORK / 'adapter_raw' / 'adapter_model.safetensors').exists()
print('TRAIN_OK', TRAIN_OK, TRAIN_SECONDS, 's')
'''

ADAPTER = r'''import json, re, shutil
from safetensors import safe_open
SERVED = Path('__SERVED_PATH__')
if not SERVED.exists():
    SERVED = next(p.parent for p in Path('/kaggle/input').rglob('config.json') if 'e4b-it-qat-w4a16-ct' in str(p))
print('served checkpoint:', SERVED)
KEYS = {}
if TRAIN_OK:
    with safe_open(str(WORK / 'adapter_raw' / 'adapter_model.safetensors'), framework='pt') as f:
        KEYS = {k: tuple(f.get_slice(k).get_shape()) for k in f.keys()}
served_modules = set()
for shard in sorted(SERVED.glob('*.safetensors')):
    with safe_open(str(shard), framework='pt') as f:
        for k in f.keys():
            m = re.match(r'(.*language_model\.layers\.\d+\.\w+\.\w+_proj)\.', k)
            if m:
                served_modules.add(m.group(1))
adapter_modules = {re.sub(r'^base_model\.model\.', '', k.rsplit('.lora_', 1)[0]) for k in KEYS}
MISSING_IN_SERVED = sorted(adapter_modules - served_modules)
print('adapter tensors', len(KEYS), '| modules', len(adapter_modules), '| not in served checkpoint', MISSING_IN_SERVED[:10])
print('sample keys', sorted(KEYS)[:4])
SUB = WORK / 'smoke_submission'
shutil.rmtree(SUB, ignore_errors=True)
if TRAIN_OK:
    dest = SUB / 'adapters' / 'main_lora'
    dest.mkdir(parents=True)
    shutil.copy(WORK / 'adapter_raw' / 'adapter_model.safetensors', dest / 'adapter_model.safetensors')
    cfg = json.loads((WORK / 'adapter_raw' / 'adapter_config.json').read_text())
    # The canary v2 format, which the scorer's vLLM served: target modules as a plain list of projection names.
    cfg['target_modules'] = sorted({m.rsplit('.', 1)[1] for m in adapter_modules})
    cfg['base_model_name_or_path'] = 'google/gemma-4-e4b-it-qat-w4a16-ct'
    (dest / 'adapter_config.json').write_text(json.dumps(cfg, indent=1))
    print('adapter_config', json.dumps({k: cfg[k] for k in ('r', 'lora_alpha', 'target_modules')}))
'''

SERVE = r'''import litellm, torch
from adk_submission import VllmConfig, VllmServer, discover_adapters
from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS

litellm.drop_params = True
adapters = discover_adapters(str(SUB), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS) if TRAIN_OK else None
gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
server = VllmServer(VllmConfig(
    model=str(SERVED), port=8000, host='127.0.0.1', tool_call_parser='gemma4', reasoning_parser='gemma4',
    max_model_len=32768, dtype='bfloat16' if torch.cuda.is_bf16_supported() else 'auto',
    gpu_memory_utilization=0.90, enable_auto_tool_choice=True, enable_lora=bool(adapters), max_loras=8, max_lora_rank=128,
    tensor_parallel_size=1, startup_timeout=60 * 20,
), adapter_manifest=adapters)
server.start()
SERVED_MODEL = getattr(server, 'effective_model_path', str(SERVED))
print('vLLM up at', server.base_url, '| gpus', gpus, '| served', SERVED_MODEL)
'''

PROBE = r'''import json, re, sys, time, urllib.request
sys.path.insert(0, str(WORK))
from transformers import AutoTokenizer
from train import as_vllm
tok = AutoTokenizer.from_pretrained(str(BF16))

def post(path, body, timeout=600):
    req = urllib.request.Request(server.base_url.rstrip('/').removesuffix('/v1') + path, data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)

def chat(model, messages, max_tokens, tools=None, thinking=False):
    body = {'model': model, 'messages': messages, 'temperature': 0, 'max_tokens': max_tokens, 'seed': 0,
            'chat_template_kwargs': {'enable_thinking': thinking}}
    if tools:
        body['tools'] = tools
    t0 = time.time()
    try:
        msg = post('/v1/chat/completions', body)['choices'][0]['message']
        return msg, round(time.time() - t0, 1)
    except Exception as e:
        return {'content': f'ERROR {type(e).__name__}: {e}'}, round(time.time() - t0, 1)

SHORT = ['Write a Python function that returns the sum of a list of integers.', 'In one sentence, what does git rebase do?',
         'List three HTTP status codes and their meaning.']
filler = '\n'.join(f'record {i:05d}: sensor reading {(i * 7919) % 1000:03d} within normal range' for i in range(1100))
NEEDLE = 'record 00777: the maintenance code is PAPAYA-QUOKKA'
LONG = filler.replace('record 00777: sensor reading', NEEDLE + ' ; sensor reading', 1) + \
    '\n\nWhat is the maintenance code in record 00777? Answer with the code only.'
toy = [json.loads(l) for l in TOY_REQUESTS.splitlines() if json.loads(l)['role'] == 'coder']
agent_turns = [t for t in toy if t['turn'] in (0, 1)]
names = ['base'] + (['main_lora'] if TRAIN_OK else [])
model_of = {'base': SERVED_MODEL, 'main_lora': 'main_lora'}
res = {}
try:
    for n in names:
        short = [chat(model_of[n], [{'role': 'user', 'content': p}], 64)[0].get('content') or '' for p in SHORT]
        long_msg, long_s = chat(model_of[n], [{'role': 'user', 'content': LONG}], 32)
        calls = []
        for t in agent_turns:
            req = t['request']
            m, s = chat(model_of[n], req['messages'], 1024, tools=req['tools'], thinking=True)
            calls.append({'turn': t['turn'], 'tool_calls': [c['function']['name'] for c in m.get('tool_calls') or []],
                          'content': (m.get('content') or '')[:200], 'seconds': s})
        res[n] = {'short': short, 'long': (long_msg.get('content') or '')[:120], 'long_seconds': long_s, 'agent': calls}
    # Our rendering of a turn against the server's, token for token.
    parity = []
    for t in agent_turns:
        req = t['request']
        ours = tok.apply_chat_template(as_vllm(req['messages']), tools=req['tools'], add_generation_prompt=True,
                                       enable_thinking=True, tokenize=True)
        ours = list(ours['input_ids']) if hasattr(ours, 'keys') else list(ours)  # BatchEncoding is not a dict
        try:
            theirs = post('/tokenize', {'model': SERVED_MODEL, 'messages': req['messages'], 'tools': req['tools'],
                                        'add_generation_prompt': True, 'chat_template_kwargs': {'enable_thinking': True}})['tokens']
            first_diff = next((i for i, (a, b) in enumerate(zip(ours, theirs)) if a != b), None)
            parity.append({'turn': t['turn'], 'ours': len(ours), 'server': len(theirs), 'equal': ours == theirs,
                           'first_diff': first_diff})
        except Exception as e:
            parity.append({'turn': t['turn'], 'error': f'{type(e).__name__}: {e}'})
    log = open(server.log_path, encoding='utf-8', errors='replace').read() if getattr(server, 'log_path', None) else ''
finally:
    server.stop() if hasattr(server, 'stop') else None

kv = [int(m.replace(',', '')) for m in re.findall(r'KV cache size: ([\d,]+) tokens', log)]
lora_lines = [ln.strip()[:200] for ln in log.splitlines() if re.search(r'(?i)lora', ln)][:12]
checks = {
    'versions_before': BEFORE, 'versions_after': AFTER, 'train_ok': TRAIN_OK, 'train_seconds': TRAIN_SECONDS,
    'adapter_modules_missing_in_served': MISSING_IN_SERVED[:20],
    'applies': sum(a != b for a, b in zip(res.get('main_lora', {}).get('short', []), res['base']['short'])),
    'long_needle': {n: 'PAPAYA-QUOKKA' in res[n]['long'] for n in res},
    'agent_tool_call': {n: sum(bool(c['tool_calls']) for c in res[n]['agent']) for n in res},
    'render_parity': parity, 'kv_cache_tokens': kv, 'lora_log': lora_lines, 'results': res,
}
ok = (TRAIN_OK and not MISSING_IN_SERVED and checks['applies'] >= 2 and all(checks['long_needle'].values())
      and checks['agent_tool_call'].get('main_lora', 0) == len(agent_turns) and bool(kv) and min(kv) >= 32768
      and all(p.get('equal') for p in parity))
checks['verdict'] = 'pipeline OK' if ok else 'pipeline NOT OK (see checks)'
(WORK / 'smoke.json').write_text(json.dumps(checks, indent=1))
print(json.dumps({k: v for k, v in checks.items() if k not in ('results',)}, indent=1)[:6000])
print('VERDICT:', checks['verdict'])
'''


def toy_turns(source_log: Path) -> tuple[str, str]:
    """Toy coder turns that reuse only the harness's own parts of a logged request (system prompt, tool schemas,
    the continuation nudge). The task, repository and every tool output are invented: no competition data."""
    real = next(json.loads(ln) for ln in source_log.read_text().splitlines() if json.loads(ln)["role"] == "coder")
    req = real["request"]
    system, tools = req["messages"][0], req["tools"]
    nudge = next(m for m in json.loads(next(ln for ln in source_log.read_text().splitlines()
                                            if json.loads(ln).get("turn") == 1))["request"]["messages"]
                 if m["role"] == "user" and "continue" in m["content"])
    kw = {k: req[k] for k in ("chat_template_kwargs", "thinking_token_budget") if k in req}
    issue = ("You are evaluating a software engineering task for repository toy/shapes.\n\nProblem Statement:\n"
             "Circle.area returns the circumference\n\n`Circle(2).area()` returns 12.57 instead of 12.57 squared units; "
             "it uses 2*pi*r instead of pi*r**2.")
    def call(i, name, args):
        return {"role": "assistant", "content": None, "tool_calls": [
            {"id": f"call_toy_{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}
    def tool(i, content):
        return {"role": "tool", "tool_call_id": f"call_toy_{i}", "content": json.dumps(content)}
    steps = [
        ("I'll ask code_analyzer for the location first.", "code_analyzer", {"request": issue.split("Problem Statement:\n")[1]},
         {"result": "LOCATION: shapes/circle.py:9 (Circle.area)\nROOT CAUSE: uses 2*pi*r.\nFIX PLAN: return math.pi * self.r ** 2"}),
        ("The analyzer's answer points here. Read shapes/circle.py lines 6-12 to confirm before editing.", "read_file",
         {"filepath": "shapes/circle.py", "start_line": 6, "end_line": 12},
         {"status": "ok", "filepath": "shapes/circle.py", "content": "class Circle:\n    def __init__(self, r):\n        self.r = r\n\n    def area(self):\n        return 2 * math.pi * self.r\n"}),
        ("Edit shapes/circle.py near line 11, old_string copied exactly from what I read.", "edit_file",
         {"filepath": "shapes/circle.py", "old_string": "        return 2 * math.pi * self.r", "new_string": "        return math.pi * self.r ** 2"},
         {"status": "ok", "filepath": "shapes/circle.py", "occurrences": 1, "strategy": "exact"}),
        ("Check that the edited files compile.", "run_command", {"command": "python -m py_compile shapes/circle.py"},
         {"status": "ok", "stdout": "", "stderr": "", "exit_code": 0}),
        ("Submit the patch.", "submit_patch", {}, {"status": "ok", "patch_size": 210, "files_changed": 1}),
    ]
    rows, history = [], [system, {"role": "user", "content": issue}]
    for variant in ("toy_1-0", "toy_2-0", "toy_3-0"):
        history = [system, {"role": "user", "content": issue.replace("toy/shapes", f"toy/{variant[:5]}")}]
        for i, (thought, name, args, out) in enumerate(steps):
            msg = call(i, name, args)
            msg["reasoning_content"] = thought
            rows.append({"role": "coder", "task": variant[:5], "turn": i, "variant": variant,
                         "request": {"messages": json.loads(json.dumps(history)), "tools": tools, **kw}, "response": msg})
            history += [msg, tool(i, out)]
            if i == 0:
                history.append(nudge)
        rows.append({"role": "coder", "task": variant[:5], "turn": len(steps), "variant": variant,
                     "request": {"messages": json.loads(json.dumps(history)), "tools": tools, **kw},
                     "response": {"role": "assistant", "content": "DONE"}})
    plans = {v: {"toy": True} for v in ("toy_1-0", "toy_2-0", "toy_3-0")}
    return "\n".join(json.dumps(r) for r in rows) + "\n", json.dumps(plans)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=ROOT / "build/kaggle_lora_smoke", type=Path)
    ap.add_argument("--slug", default="gemma4-lora-smoke-e4b")
    ap.add_argument("--source-log", default=ROOT / "data/lora/b/requests.jsonl", type=Path)
    args = ap.parse_args()
    requests, plans = toy_turns(args.source_log)
    consts = (f"TOY_REQUESTS = {requests!r}\nTOY_PLANS = {plans!r}\nTRAIN_PY = {TRAIN_SRC.read_text()!r}\n")
    notebook = {
        "cells": with_cell_ids([{"cell_type": "markdown", "metadata": {}, "source": [
                      f"# {args.slug}\n\nPipeline smoke test for the behavior LoRA (doc 03 §14.25, amendment 1): "
                      "train a few steps on toy turns with Gemma 4 E4B (bf16 QAT), serve the W4A16 checkpoint with "
                      "the adapter in the scorer's vLLM, and check versions, adapter loading, long prompts, tool "
                      "calls and render parity. Writes `smoke.json`. Evaluates no behavior."]},
                  cell(VERSIONS), cell(SETUP), cell(PEFT), cell(consts),
                  cell(TRAIN.replace("__BF16_PATH__", BF16_PATH)),
                  cell(ADAPTER.replace("__SERVED_PATH__", SERVED_PATH)), cell(SERVE), cell(PROBE)]),
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "smoke.ipynb").write_text(json.dumps(notebook, indent=1))
    meta = {"id": f"{OWNER}/{args.slug}", "title": args.slug.replace("-", " "), "code_file": "smoke.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
            "enable_tpu": False, "enable_internet": False, "dataset_sources": [WHEELHOUSE, LORA_WHEELS], "kernel_sources": [],
            "competition_sources": [COMPETITION], "model_sources": [BF16_SOURCE, SERVED_SOURCE],
            "docker_image": DOCKER_IMAGE, "machine_shape": "NvidiaL4"}
    (args.out / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    print(f"kernel {meta['id']} -> {args.out}")


if __name__ == "__main__":
    main()
