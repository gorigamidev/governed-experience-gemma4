"""LoRA rehearsal with Gemma 4 E4B (doc 03 §14.25, rule 2) on the audited core-B data.

Trains `lab/lora/train.py` (2 epochs) on the bf16 QAT checkpoint and checks, in order: the vLLM `gemma4` tool and
reasoning parsers read every training target back as its scripted call; validation loss falls from its initial
value; the official validator accepts the package with `adapter: main_lora`; served on the W4A16
checkpoint, the adapter changes outputs (>= 2 of 3 short prompts), a ~20k-token prompt still returns the base's
answer, the KV cache holds >= 32,768 tokens, and on the turn-0 prompts of the 16 reserve tasks >= 90% of the
adapter's answers are a valid tool call. Writes `rehearsal.json` with the verdict.

Inputs: the wheelhouse, `gorigamidev/gemma4-lora-wheels` (peft, accelerate) and `gorigamidev/gemma4-lora-data`
(requests.jsonl, plans.json, reserve_requests.jsonl). No internet (the fixed image refuses it).

Usage: PYTHONPATH=lab/src python3 -m lab.export.kaggle_lora_rehearsal_notebook
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
LORA_DATA = "gorigamidev/gemma4-lora-data"
PACKAGE = ROOT / "build/top-mix-1k/submission.zip"


def arm_base(package: Path) -> str:
    """Arm name stem from the package path: build/<name>/submission.zip -> <name>."""
    return package.parent.name


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


DATA = r'''import base64, io, json, subprocess, sys, time, zipfile
from pathlib import Path
# Time budget: every stage checks it, so the notebook always ends on its own and Kaggle keeps /kaggle/working.
T0 = time.time()
BUDGET_MIN = __BUDGET_MIN__
def minutes_left():
    return BUDGET_MIN - (time.time() - T0) / 60
WORK = Path('/kaggle/working')

def run_logged(cmd, log_name):
    # Streams every line to the cell (visible while it runs) and to a file in /kaggle/working.
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
print('data:', DATA_DIR, sorted(p.name for p in DATA_DIR.iterdir()))
BF16 = Path('__BF16_PATH__')
if not BF16.exists():
    BF16 = next(p.parent for p in Path('/kaggle/input').rglob('config.json') if 'e4b-it-qat-q4_0-unquantized' in str(p))
print('bf16 checkpoint:', BF16, '| budget', BUDGET_MIN, 'min')
(WORK / 'train.py').write_text(TRAIN_PY)
RESERVE = [json.loads(l) for l in (DATA_DIR / 'reserve_requests.jsonl').read_text().splitlines()]
print('reserve turn-0 prompts:', len(RESERVE))
'''

PARSER = r'''import sys
sys.path.insert(0, str(WORK))
from transformers import AutoTokenizer
from train import coder_turns, render
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.reasoning.gemma4_reasoning_parser import Gemma4ReasoningParser
from vllm.tool_parsers.gemma4_tool_parser import Gemma4ToolParser
tok = AutoTokenizer.from_pretrained(str(BF16))
TURNS = coder_turns(DATA_DIR)
tool_parser, reasoning_parser = Gemma4ToolParser(tok), Gemma4ReasoningParser(tok)
PARSER_BAD = []
for t in TURNS:
    _, target = render(tok, t)
    text = target.removesuffix('<|tool_response>').removesuffix('<turn|>')
    req = ChatCompletionRequest(model='m', messages=[{'role': 'user', 'content': 'x'}], tools=t['request']['tools'])
    reasoning, content = reasoning_parser.extract_reasoning(text, req)
    info = tool_parser.extract_tool_calls(content or '', req)
    found = [(c.function.name, json.loads(c.function.arguments)) for c in info.tool_calls]
    want = [(c['function']['name'], json.loads(c['function']['arguments'])) for c in t['response'].get('tool_calls') or []]
    if found != want or (want and reasoning is None and t['response'].get('reasoning_content')):
        PARSER_BAD.append(f"{t['variant']} turn {t['turn']}: {str(found)[:100]} != {str(want)[:100]}")
print('parser round trip:', len(TURNS) - len(PARSER_BAD), 'of', len(TURNS), PARSER_BAD[:3])
'''

TRAIN = r'''# Training stops early (and still saves the adapter) if it would eat the time the later checks need.
TRAIN_ARGS = __TRAIN_ARGS__
t0 = time.time()
rc, out = run_logged([sys.executable, str(WORK / 'train.py'), '--data', str(DATA_DIR), '--tokenizer', str(BF16),
                      '--model', str(BF16), '--out', str(WORK / 'adapter_raw'), '--epochs', '2', '--val-frac', '0.1',
                      '--max-len', '16384', '--max-minutes', str(max(5, round(minutes_left() - AFTER_TRAIN_MIN, 1))),
                      *TRAIN_ARGS], 'train.log')
TRAIN_SECONDS = round(time.time() - t0, 1)
TRAIN_OK = rc == 0 and (WORK / 'adapter_raw' / 'adapter_model.safetensors').exists()
REPORT = json.loads((WORK / 'adapter_raw' / 'train_report.json').read_text()) if TRAIN_OK else {}
REPORT['recipe'] = 'qlora-4bit' if '--load-4bit' in TRAIN_ARGS else 'bf16'
print('TRAIN_OK', TRAIN_OK, REPORT['recipe'], TRAIN_SECONDS, 's | val', REPORT.get('val_initial'), '->',
      REPORT.get('val_final'), '| stopped at step', REPORT.get('stopped_at_step'), '| minutes left', round(minutes_left()))
'''

VALIDATE = r'''import base64, io, re, zipfile
from adk_submission import compile_submission, discover_adapters, validate_directory
from adk_submission.registry import ModelRegistry
from swegemma.config import build_submission_limits
# Same steps as `make package` (lab/export/package.py): validate, discover adapters, compile with the adapter's
# model registered. The validator checks the package format, not the adapter's tensors (a junk .safetensors
# passes: local control, 5 Oct); the tensors are checked by loading them in the adapter cell and in vLLM.
HARNESS_TOOLS = ['run_command', 'read_file', 'write_file', 'edit_file', 'submit_patch', 'get_status',
                 'search_similar_code', 'get_code_neighbors', 'get_code_subgraph']
VALIDATOR = 'not run'
if TRAIN_OK:
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(PACKAGE_B64))) as z:
        z.extractall(SUB)
    agent = (SUB / 'agent.yaml').read_text()
    MODEL = re.search(r'^model: (\S+)$', agent, re.M).group(1)  # the package's checkpoint, whatever it is
    (SUB / 'agent.yaml').write_text(re.sub(r'^(model: [^\n]*\n)', r'\1adapter: main_lora\n', agent, count=1, flags=re.M))
    limits, gen = build_submission_limits()
    try:
        validate_directory(SUB, limits)
        manifest = discover_adapters(SUB, limits.adapter_extensions)
        models = ModelRegistry()
        models.register(MODEL, MODEL)
        for name in manifest.adapters:
            models.register(name, MODEL)
        compile_submission(submission_dir=SUB, tool_registry={t: (lambda **_: '') for t in HARNESS_TOOLS},
                           model_registry=models, limits=limits, generation_constraints=gen, adapter_manifest=manifest)
        VALIDATOR = 'ok: adapters ' + ', '.join(sorted(manifest.adapters))
    except Exception as e:
        VALIDATOR = f'FAILED {type(e).__name__}: {e}'
print('validator:', VALIDATOR)
'''

PROBE = r'''import re, time, urllib.request
def post(path, body, timeout=900):
    req = urllib.request.Request(server.base_url.rstrip('/').removesuffix('/v1') + path, data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)

def chat(model, messages, max_tokens, extra=None):
    body = {'model': model, 'messages': messages, 'temperature': 0, 'max_tokens': max_tokens, 'seed': 0,
            'chat_template_kwargs': {'enable_thinking': False}, **(extra or {})}
    t0 = time.time()
    try:
        return post('/v1/chat/completions', body)['choices'][0]['message'], round(time.time() - t0, 1)
    except Exception as e:
        return {'content': f'ERROR {type(e).__name__}: {e}'}, round(time.time() - t0, 1)

def valid_call(msg, tools):
    names = {t['function']['name'] for t in tools}
    calls = msg.get('tool_calls') or []
    try:
        return bool(calls) and all(c['function']['name'] in names and isinstance(json.loads(c['function']['arguments'] or '{}'), dict)
                                   for c in calls)
    except Exception:
        return False

SHORT = ['Write a Python function that returns the sum of a list of integers.', 'In one sentence, what does git rebase do?',
         'List three HTTP status codes and their meaning.']
filler = '\n'.join(f'record {i:05d}: sensor reading {(i * 7919) % 1000:03d} within normal range' for i in range(1100))
NEEDLE = 'record 00777: the maintenance code is PAPAYA-QUOKKA'
LONG = filler.replace('record 00777: sensor reading', NEEDLE + ' ; sensor reading', 1) + \
    '\n\nWhat is the maintenance code in record 00777? Answer with the code only.'
names = ['base'] + (['main_lora'] if TRAIN_OK else [])
model_of = {'base': SERVED_MODEL, 'main_lora': 'main_lora'}
res = {}
try:
    for n in names:
        short = [chat(model_of[n], [{'role': 'user', 'content': p}], 64)[0].get('content') or '' for p in SHORT]
        long_msg, long_s = chat(model_of[n], [{'role': 'user', 'content': LONG}], 32)
        reserve = []
        for r in RESERVE:
            req = r['request']
            extra = {'tools': req['tools'], 'chat_template_kwargs': req.get('chat_template_kwargs', {}),
                     'thinking_token_budget': req.get('thinking_token_budget')}
            m, s = chat(model_of[n], req['messages'], 2048, {k: v for k, v in extra.items() if v is not None})
            reserve.append({'task': r['task'], 'valid': valid_call(m, req['tools']), 'seconds': s,
                            'calls': [c['function']['name'] for c in m.get('tool_calls') or []],
                            'content': (m.get('content') or '')[:160]})
        res[n] = {'short': short, 'long': (long_msg.get('content') or '')[:120], 'long_seconds': long_s, 'reserve': reserve}
    log = open(server.log_path, encoding='utf-8', errors='replace').read() if getattr(server, 'log_path', None) else ''
finally:
    server.stop() if hasattr(server, 'stop') else None

kv = [int(m.replace(',', '')) for m in re.findall(r'KV cache size: ([\d,]+) tokens', log)]
valid = {n: sum(x['valid'] for x in res[n]['reserve']) for n in res}
lora = res.get('main_lora', {})
checks = {
    'versions_after': AFTER, 'parser_round_trip': f'{len(TURNS) - len(PARSER_BAD)} of {len(TURNS)}', 'parser_bad': PARSER_BAD[:10],
    'train_ok': TRAIN_OK, 'train_seconds': TRAIN_SECONDS, 'train_report': {k: REPORT.get(k) for k in
        ('train_examples', 'val_examples', 'train_tokens', 'train_loss_tokens', 'val_initial', 'val_final', 'stopped_at_step', 'recipe')},
    'adapter_modules_missing_in_served': MISSING_IN_SERVED[:20], 'validator': VALIDATOR,
    'applies': sum(a != b for a, b in zip(lora.get('short', []), res['base']['short'])),
    'long': {n: res[n]['long'] for n in res}, 'kv_cache_tokens': kv,
    'reserve_valid_calls': {n: f'{valid[n]} of {len(RESERVE)}' for n in valid},
    'reserve_first_call': {n: [x['calls'][:1] for x in res[n]['reserve']] for n in res}, 'results': res,
}
rule = {
    'parser': not PARSER_BAD,
    'val_loss_falls': REPORT.get('val_final') is not None and REPORT['val_final'] < REPORT['val_initial'],
    'validator': VALIDATOR.startswith('ok'),
    'served_modules': not MISSING_IN_SERVED,
    'applies': checks['applies'] >= 2,
    'long_ok': 'main_lora' in res and not res['main_lora']['long'].startswith('ERROR'),
    'kv_ok': bool(kv) and min(kv) >= 32768,
    'capability_long': 'main_lora' in res and ('PAPAYA-QUOKKA' in res['main_lora']['long']) == ('PAPAYA-QUOKKA' in res['base']['long']),
    'capability_reserve': valid.get('main_lora', 0) >= 0.9 * len(RESERVE),
}
checks['rule2'] = rule
checks['verdict'] = 'rehearsal passes' if all(rule.values()) else 'rehearsal fails: ' + ', '.join(k for k, v in rule.items() if not v)
(WORK / 'rehearsal.json').write_text(json.dumps(checks, indent=1))
print(json.dumps({k: v for k, v in checks.items() if k not in ('results', 'reserve_first_call')}, indent=1)[:8000])
print('VERDICT:', checks['verdict'])
'''




EFFECT = r'''import base64, hashlib, io, subprocess, sys, tarfile, zipfile
from adk_submission import VllmConfig, VllmServer, discover_adapters
from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS
# Rule 4 (doc 03 §14.25), run in this session only if every rule-3 check passed (L6 and L7 joined, decision of 6 Oct).
# Interleaved per task, adapter arm first, saved after every run; no task starts without time for both arms.
RULE3_OK = all(rule.values())
print('rule 3:', checks['verdict'], '| minutes left', round(minutes_left()))
done = []
if RULE3_OK:
    arms = {'__ARM__-lora': SUB, '__ARM__': WORK / 'arms' / '__ARM__'}
    zipfile.ZipFile(io.BytesIO(base64.b64decode(PACKAGE_B64))).extractall(arms['__ARM__'])
    code = base64.b64decode(RUN_EVAL_B64)
    assert hashlib.sha256(code).hexdigest() == RUN_EVAL_SHA256
    (WORK / 'run_eval.py').write_bytes(code)
    adapters = discover_adapters(str(SUB), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
    server = VllmServer(VllmConfig(
        model=str(SERVED), port=8000, host='127.0.0.1', tool_call_parser='gemma4', reasoning_parser='gemma4',
        max_model_len=32768, dtype='bfloat16', gpu_memory_utilization=0.90, enable_auto_tool_choice=True,
        enable_lora=True, max_loras=8, max_lora_rank=128, tensor_parallel_size=4, startup_timeout=60 * 20,
    ), adapter_manifest=adapters)
    server.start()
    SERVED_MODEL = getattr(server, 'effective_model_path', str(SERVED))
    COMP = Path('/kaggle/input/competitions/gemma-4-developer-agent')
    EFFECT = WORK / 'effect'

    def save():
        if EFFECT.exists():
            with tarfile.open(WORK / 'effect.tar.gz', 'w:gz') as tar:
                tar.add(EFFECT, arcname='effect', filter=lambda ti: None if '/submission/' in ti.name + '/' else ti)
        (WORK / 'effect_progress.json').write_text(json.dumps({'done': done}, indent=1))

    try:
        for task in RESERVE_IDS:
            if minutes_left() < PAIR_MIN:
                print('time budget: stopping the effect before', task, '| minutes left', round(minutes_left()))
                break
            for name, arm in arms.items():
                cmd = [sys.executable, str(WORK / 'run_eval.py'), '--submission', str(arm), '--tasks', str(COMP / 'tasks.jsonl'),
                       '--snapshots', str(COMP / 'snapshots'), '--graphs', str(COMP / 'graphs'), '--embeddings', str(COMP / 'embeddings'),
                       '--wheels', str(COMP / 'wheels'), '--results', str(EFFECT / name / task), '--task-ids', task,
                       '--sandbox', 'subprocess', '--api-base', server.base_url, '--served-model', SERVED_MODEL,
                       '--model-prefix', 'openai/', '--label', f'kaggle-lora-31b-{name}', '--hardware', '4xL4']
                t0 = time.time()
                rc, _ = run_logged(cmd, 'effect.log')
                done.append({'task': task, 'arm': name, 'rc': rc, 'minutes': round((time.time() - t0) / 60, 2)})
                print(task, name, done[-1]['minutes'], 'min | minutes left', round(minutes_left()), flush=True)
                save()
    finally:
        server.stop() if hasattr(server, 'stop') else None
        save()
else:
    print('rule 3 not passed: the effect arms are not run and the adapter is not used')
checks['effect_pairs_complete'] = sum(1 for t in RESERVE_IDS if sum(d['task'] == t for d in done) == 2)
checks['minutes_left_at_end'] = round(minutes_left())
(WORK / 'rehearsal.json').write_text(json.dumps(checks, indent=1))
print('effect pairs complete:', checks['effect_pairs_complete'], 'of', len(RESERVE_IDS))
'''


GPU_MEM = r'''print(subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total', '--format=csv'],
                     capture_output=True, text=True).stdout)
print('report:', json.dumps({k: REPORT.get(k) for k in ('train_examples', 'val_initial', 'val_final', 'stopped_at_step', 'recipe')}))
'''


def fix_model(src: str, model: str) -> str:
    """Retarget the E4B-based templates to `model`: checkpoint paths, adapter config and the tensor-parallel size."""
    short = models.short(model)
    src = src.replace("e4b-it-qat", f"{short}-it-qat")
    return src if short == "e4b" else src.replace("tensor_parallel_size=1", "tensor_parallel_size=4")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=tuple(models.NAMES), default="e4b")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--slug")
    ap.add_argument("--package", type=Path, default=PACKAGE,
                    help="submission.zip the arms are built from; the arm names come from its directory name")
    ap.add_argument("--budget-min", type=int, help="wall-clock budget of the session; the 4xL4 quota is charged ~3x")
    ap.add_argument("--smoke-steps", type=int, default=0,
                    help="training smoke only: stop after N steps, no serving or checks (31B QLoRA pre-flight, 7 Oct)")
    args = ap.parse_args()
    short = models.short(args.model)
    big = short != "e4b"
    args.out = args.out or ROOT / (f"build/kaggle_lora_{short}" if big else "build/kaggle_lora_rehearsal")
    args.slug = args.slug or (f"gemma4-lora-{short}" if big else "gemma4-lora-rehearsal-e4b")
    fix = lambda src: fix_model(src, args.model)  # noqa: E731
    # E4B: ~50 min measured (rehearsal v6). 31B: QLoRA directly (bf16 ran out of memory at ~23 min, 7 Oct); training
    # 2 h 20 min measured, then ~12 min for adapter, vLLM and the rule-3 probes, ~8 min per task pair in the effect.
    budget = args.budget_min or (420 if big else 75)
    after_train = 25 if big else 15
    train_args = (["--load-4bit"] if big else []) + (["--max-steps", str(args.smoke_steps), "--val-frac", "0"] if args.smoke_steps else [])
    timing = (f"AFTER_TRAIN_MIN = {after_train}  # minutes kept for adapter checks, vLLM and probes after training\n"
              f"PAIR_MIN = 14  # an effect task starts only with time for both arms (4.5 min cap each, plus setup)\n")
    if big:
        sys.path.insert(0, str(ROOT / "lab/src"))
        from lab.lora.scripted import RESERVE
        runner = (ROOT / "lab/src/lab/eval/run_eval.py").read_bytes()
        effect_consts = (f"RUN_EVAL_SHA256 = {hashlib.sha256(runner).hexdigest()!r}\n"
                         f"RUN_EVAL_B64 = {base64.b64encode(runner).decode()!r}\n"
                         f"RESERVE_IDS = {sorted(RESERVE)!r}\n")
    base = arm_base(args.package)
    package = args.package.read_bytes()
    consts = (f"TRAIN_PY = {smoke.TRAIN_SRC.read_text()!r}\n"
              f"PACKAGE_SHA256 = {hashlib.sha256(package).hexdigest()!r}  # {_rel(args.package)}\n"
              f"PACKAGE_B64 = {base64.b64encode(package).decode()!r}\n")
    adapter = smoke.ADAPTER.replace("__SERVED_PATH__", smoke.SERVED_PATH).replace("smoke_submission", "lora_submission")
    notebook = {
        "cells": with_cell_ids([{"cell_type": "markdown", "metadata": {}, "source": [
                      (f"# {args.slug}\n\nBehavior LoRA on Gemma 4 {short.upper()} (doc 03 §14.25, rule 3): the rehearsal's checks on the "
                      "trained adapter, served as the scorer serves it (W4A16, TP 4); if they all pass, rule 4 in the same "
                      f"session: `{base}` against `{base}` + `main_lora` on the 16 reserve tasks. Writes "
                      "`rehearsal.json` and `effect.tar.gz`.") if big else
                      (f"# {args.slug}\n\nBehavior LoRA rehearsal (doc 03 §14.25, rule 2) with Gemma 4 E4B on the "
                      "audited core-B data: parser round trip, 2-epoch training, official validator, canary and "
                      "turn-0 tool calls on the 16 reserve tasks. Writes `rehearsal.json`.")]},
                  cell(smoke.VERSIONS), cell(smoke.SETUP), cell(smoke.PEFT), cell(consts),
                  cell(fix(DATA.replace("__BF16_PATH__", smoke.BF16_PATH).replace("__BUDGET_MIN__", str(budget)))),
                  cell(timing), cell(PARSER), cell(TRAIN.replace("__TRAIN_ARGS__", repr(train_args))), cell(fix(adapter)), cell(VALIDATE), cell(fix(smoke.SERVE)),
                  cell(PROBE)] + ([cell(effect_consts), cell(EFFECT.replace("__ARM__", base))] if big else []))
                 if not args.smoke_steps else
                 with_cell_ids([{"cell_type": "markdown", "metadata": {}, "source": [
                      f"# {args.slug}\n\nTraining smoke of the behavior LoRA on Gemma 4 {args.model.upper()}: load, "
                      f"{args.smoke_steps} steps with the corrected loss, timings and memory. No serving, no checks."]},
                  cell(smoke.VERSIONS), cell(smoke.SETUP), cell(smoke.PEFT), cell(consts),
                  cell(fix(DATA.replace("__BF16_PATH__", smoke.BF16_PATH).replace("__BUDGET_MIN__", str(budget)))),
                  cell(timing), cell(TRAIN.replace("__TRAIN_ARGS__", repr(train_args))), cell(GPU_MEM)]),
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "rehearsal.ipynb").write_text(json.dumps(notebook, indent=1))
    meta = {"id": f"{OWNER}/{args.slug}", "title": args.slug.replace("-", " "), "code_file": "rehearsal.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
            "enable_tpu": False, "enable_internet": False,
            "dataset_sources": [WHEELHOUSE, smoke.LORA_WHEELS, LORA_DATA], "kernel_sources": [],
            "competition_sources": [COMPETITION],
            "model_sources": [fix(smoke.BF16_SOURCE), fix(smoke.SERVED_SOURCE)],
            "docker_image": DOCKER_IMAGE, "machine_shape": "NvidiaL4"}
    (args.out / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    print(f"kernel {meta['id']} -> {args.out} | package sha256 {hashlib.sha256(package).hexdigest()[:12]}")


if __name__ == "__main__":
    main()
