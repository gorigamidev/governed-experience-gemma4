"""G1 notebook (doc 03 §14.30): Gemma 4 12B against 31B as the base model.

One GPU session answers whether the 12B can replace the 31B before spending quota on a LoRA:
  * three arms on the 16 reserve tasks, 12B first with A12/B12 interleaved per task on one server (<= 2 h), then
    ANC on the 31B (<= 75 min), saved after every task so a cut session keeps the tasks that finished;
  * render parity of the 12B against vLLM's `/tokenize` (the §14.26 test, 42 prompts); only conditions G2;
  * the extra analyzer-only cell (added 12 oct, no effect on the rules): pool tasks without traces, never reserve,
    first on the 12B then on the 31B, each answer saved with the model that produced it (input for G2, §14.31);
  * a preflight cell that imports the whole wheelhouse stack and stops in the first minute on failure;
  * a 12B load cell that serves the 12B with `models.py` flags, makes one request, and only on
    `KeyError: 'vision_embedder...'` applies the forum 748287 patch (`lab/export/vllm_patch.py`) and prints it;
    any other error stops the notebook.
The rules of §14.30 are applied locally on the downloaded traces; this notebook only produces the data.

Usage: PYTHONPATH=lab/src python3 -m lab.export.kaggle_g1_notebook
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
from pathlib import Path

from lab.export import models
from lab.export import vllm_patch
from lab.export.kaggle_eval_notebook import COMPETITION, DOCKER_IMAGE, OWNER, WHEELHOUSE, cell
from lab.export.kaggle_lora_rehearsal_notebook import LORA_DATA
from lab.export.kaggle_lora_smoke_notebook import SETUP, VERSIONS
from lab.export.kaggle_notebook import with_cell_ids

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_A12 = ROOT / "build/top-mix-1k-an-off-tb0-12b/submission.zip"
DEFAULT_B12 = ROOT / "build/top-mix-1k-12b/submission.zip"
DEFAULT_ANC = ROOT / "build/top-mix-1k-an-off-tb0/submission.zip"
# Pool tasks with a local snapshot and not in reserve, minus the ones our new-scorer runs already traced
# (any `trace_<id>.json` in `scripted.NEW_SCORER_RUNS`): 54 - 25 = 29, counted 13 oct from data/eval. The §14.25
# amendment said 31 with an unrecorded criterion (doc 03 §14.25 correction). IDs only, no task text.
POOL_WITHOUT_TRACES = (
    "fastapi_11355 fastapi_13713 fastapi_14077 fastapi_14186 fastapi_14266 fastapi_14360 fastapi_14361 "
    "fastapi_14419 fastapi_14458 fastapi_14459 fastapi_14482 fastapi_14485 fastapi_14487 fastapi_14512 "
    "fastapi_14609 fastapi_14616 fastapi_14786 fastapi_14953 fastapi_14962 fastapi_15763 fastapi_15785 "
    "fastapi_5077 fastapi_9555 httpx_3672 requests_7315 requests_7328 requests_7433 requests_7505 rich_4079"
).split()


def first_difference(ours: list, theirs: list) -> int | None:
    """Index of the first token that differs, the length of the shorter list if one is a prefix of the other."""
    for i, (a, b) in enumerate(zip(ours, theirs)):
        if a != b:
            return i
    if len(ours) != len(theirs):
        return min(len(ours), len(theirs))
    return None


def parity_verdict(results: list[dict]) -> str:
    """"parity" only when every prompt and full render matches; "not measured" with no rows."""
    ok = [r for r in results if "error" not in r]
    if not ok:
        return "not measured"
    return "parity" if all(r["prompt"]["equal"] and r["full"]["equal"] for r in ok) else "difference"


def analyzer_config(package_dir):
    """The code_analyzer request the harness sends: instruction + the identity footer, and its thinking config."""
    from adk_submission import yaml_loader
    package_dir = Path(package_dir)
    y = yaml_loader.load_yaml(package_dir / 'sub_agents' / 'code_analyzer.yaml', root_dir=package_dir)
    system = y['instruction'] + '\n\n' + (
        f'You are an agent. Your internal name is "{y["name"]}". The description about you is "{y["description"]}".')
    cfg = y['generate_content_config']
    tc = cfg.get('thinking_config') or {}
    budget = int(tc.get('thinking_budget') or 0)
    enable = not (tc.get('include_thoughts') is False or budget <= 0)
    return {'system': system, 'enable_thinking': enable, 'thinking_token_budget': budget or None,
            'temperature': cfg.get('temperature', 0.2), 'top_p': cfg.get('top_p', 0.95),
            'max_tokens': cfg.get('max_output_tokens', 8192)}


PREFLIGHT = r'''import importlib
# Everything the notebook uses, imported from the current wheelhouse before spending minutes loading weights.
for mod in ('torch', 'transformers', 'vllm', 'litellm', 'safetensors', 'yaml'):
    m = importlib.import_module(mod)
    print('preflight', mod, getattr(m, '__version__', '?'))
import adk_submission, swegemma
from adk_submission import VllmConfig, VllmServer, compile_submission, discover_adapters, validate_directory
from adk_submission.registry import ModelRegistry
from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS, EvalConfig, build_submission_limits
from google.adk.models.lite_llm import LiteLlm
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.apps._configs import EventsCompactionConfig
print('preflight ok:', adk_submission.__name__, swegemma.__name__)
'''

MATERIALIZE = r'''import base64, hashlib, io, json, re, subprocess, sys, tarfile, time, zipfile
from pathlib import Path
WORK = Path('/kaggle/working')
T0 = time.time()
TASK_MIN = 6.5  # a task starts only with this much left (4.5 min cap, setup and teardown)
DATA_DIR = Path('/kaggle/input/competitions/gemma-4-developer-agent')
PACKAGES = {'A12': (A12_SHA256, A12_B64), 'B12': (B12_SHA256, B12_B64), 'ANC': (ANC_SHA256, ANC_B64)}

def minutes_left(deadline):
    return (deadline - time.time()) / 60

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

def find_checkpoint(token):
    p = Path('/kaggle/input')
    return next(q.parent for q in p.rglob('config.json') if token in str(q))

SERVED_12B = Path(MODEL_12B_PATH) if Path(MODEL_12B_PATH).exists() else find_checkpoint(MODEL_12B_TOKEN)
SERVED_31B = Path(MODEL_31B_PATH) if Path(MODEL_31B_PATH).exists() else find_checkpoint(MODEL_31B_TOKEN)
ARM_DIRS = {}
for name, (sha, b64) in PACKAGES.items():
    data = base64.b64decode(b64)
    assert hashlib.sha256(data).hexdigest() == sha, name
    ARM_DIRS[name] = WORK / 'arms' / name
    zipfile.ZipFile(io.BytesIO(data)).extractall(ARM_DIRS[name])
    print('arm', name, sha[:12])
code = base64.b64decode(RUN_EVAL_B64)
assert hashlib.sha256(code).hexdigest() == RUN_EVAL_SHA256
(WORK / 'run_eval.py').write_bytes(code)
assert hashlib.sha256(VLLM_PATCH_PY.encode()).hexdigest() == VLLM_PATCH_SHA256
(WORK / 'vllm_patch.py').write_text(VLLM_PATCH_PY)
(WORK / 'train.py').write_text(TRAIN_PY)
PROBLEMS = {}
for line in (DATA_DIR / 'tasks.jsonl').read_text().splitlines():
    t = json.loads(line)
    PROBLEMS[t['instance_id']] = t['problem_statement']
# Coder traces for the parity cell live in LORA_DATA, not in the competition data (v1 died on this after the 12B load).
TRACE_DIR = next((q.parent for top in Path('/kaggle/input').iterdir() if top.name not in ('competitions', 'models')
                  for q in top.rglob('plans.json') if (q.parent / 'requests.jsonl').exists()), None)
assert TRACE_DIR, 'no plans.json + requests.jsonl under /kaggle/input: is ' + LORA_DATA + ' attached?'
print('traces', TRACE_DIR)
print('12B', SERVED_12B, '| 31B', SERVED_31B, '| reserve', len(RESERVE_IDS), '| pool', len(POOL_IDS),
      '| budgets', BUDGET_12B_MIN, '/', BUDGET_31B_MIN, 'min')

def analyzer_config(package_dir):
    """The code_analyzer request the harness sends: instruction + the identity footer, and its thinking config."""
    from adk_submission import yaml_loader
    package_dir = Path(package_dir)
    y = yaml_loader.load_yaml(package_dir / 'sub_agents' / 'code_analyzer.yaml', root_dir=package_dir)
    system = y['instruction'] + '\n\n' + (
        f'You are an agent. Your internal name is "{y["name"]}". The description about you is "{y["description"]}".')
    cfg = y['generate_content_config']
    tc = cfg.get('thinking_config') or {}
    budget = int(tc.get('thinking_budget') or 0)
    enable = not (tc.get('include_thoughts') is False or budget <= 0)
    return {'system': system, 'enable_thinking': enable, 'thinking_token_budget': budget or None,
            'temperature': cfg.get('temperature', 0.2), 'top_p': cfg.get('top_p', 0.95),
            'max_tokens': cfg.get('max_output_tokens', 8192)}

def run_analyzer(tag, package_dir, server, served_model, deadline):
    """Only the analyzer on the pool tasks without traces, within the model's budget. Saves each answer."""
    import litellm
    cfg = analyzer_config(package_dir)
    out = WORK / f'analyzer_{tag}.jsonl'
    done = 0
    for iid in POOL_IDS:
        if minutes_left(deadline) < 0.2:
            print('analyzer budget: stopping before', iid)
            break
        extra_body = {'chat_template_kwargs': {'enable_thinking': cfg['enable_thinking']}}
        if cfg['enable_thinking'] and cfg['thinking_token_budget']:
            extra_body['thinking_token_budget'] = cfg['thinking_token_budget']
        t0 = time.time()
        try:
            r = litellm.completion(model='openai/' + served_model, api_base=server.base_url, api_key='EMPTY',
                                   messages=[{'role': 'system', 'content': cfg['system']},
                                             {'role': 'user', 'content': PROBLEMS[iid]}],
                                   temperature=cfg['temperature'], top_p=cfg['top_p'], max_tokens=cfg['max_tokens'],
                                   extra_body=extra_body, timeout=600)
            row = {'task': iid, 'tag': tag, 'served_model': served_model, 'enable_thinking': cfg['enable_thinking'],
                   'thinking_token_budget': cfg['thinking_token_budget'] if cfg['enable_thinking'] else None,
                   'response': r.choices[0].message.content or ''}
        except Exception as e:
            row = {'task': iid, 'tag': tag, 'served_model': served_model, 'enable_thinking': cfg['enable_thinking'],
                   'error': f'{type(e).__name__}: {e}'[:300]}
        row['minutes'] = round((time.time() - t0) / 60, 2)
        with open(out, 'a') as f:
            f.write(json.dumps(row) + '\n')
        done += 1
        print('analyzer', tag, iid, row['minutes'], 'min | left', round(minutes_left(deadline)), flush=True)
    print('analyzer', tag, done, 'of', len(POOL_IDS))
    return done
'''

# Always runs, in both arm sets. Two things live here because the 12B cells are skipped with `--arms ANC` and
# `SUMMARY`/`LOAD31B` need them anyway (found by the v3 end-to-end rehearsal):
#   * the defaults for the 12B-only state that SUMMARY reads (`done12b`, `PARITY_VERDICT`, `PATCH_APPLIED`,
#     `ANALYZER_12B`, `DEADLINE_12B`) and `server`, which `LOAD31B` stops before rebooting;
#   * `boot_server`, the `litellm`/`VllmServer` imports and `vllm_patch`, which `LOAD31B` uses but which used to be
#     defined inside the skipped LOAD12B cell.
SERVERS = r'''import litellm
from adk_submission import VllmConfig, VllmServer
litellm.drop_params = True
sys.path.insert(0, str(WORK))
import vllm_patch

done12b, done31b = [], []
ANALYZER_12B = ANALYZER_31B = 0
PARITY_VERDICT, PATCH_APPLIED = 'not measured', False
DEADLINE_12B = DEADLINE_31B = None
server = None

def boot_server(model_path, extra_args):
    srv = VllmServer(VllmConfig(
        model=str(model_path), port=8000, host='127.0.0.1', tool_call_parser='gemma4', reasoning_parser='gemma4',
        max_model_len=32768, dtype='bfloat16', gpu_memory_utilization=0.90, enable_auto_tool_choice=True,
        extra_args=list(extra_args), tensor_parallel_size=4, startup_timeout=60 * 20))
    srv.start()
    return srv
'''

LOAD12B = r'''PATCH_APPLIED = False
try:
    server = boot_server(SERVED_12B, EXTRA_12B)
except Exception as e:
    if 'vision_embedder' not in str(e):
        raise
    target = vllm_patch.module_path()
    changed = vllm_patch.apply_to_file(target)
    PATCH_APPLIED = True
    print('12B load hit vision_embedder; applied the forum 748287 patch to', target, '| changed', changed)
    server = boot_server(SERVED_12B, EXTRA_12B)
SERVED_MODEL = getattr(server, 'effective_model_path', str(SERVED_12B))
ping = litellm.completion(model='openai/' + SERVED_MODEL, api_base=server.base_url, api_key='EMPTY',
                          messages=[{'role': 'user', 'content': 'ping'}], max_tokens=4, timeout=300)
DEADLINE_12B = time.time() + BUDGET_12B_MIN * 60
print('12B up at', server.base_url, '| served', SERVED_MODEL, '| patch', PATCH_APPLIED,
      '| reply', (ping.choices[0].message.content or '')[:40])
'''

PARITY = r'''import urllib.request
sys.path.insert(0, str(WORK))
from transformers import AutoTokenizer
from train import as_vllm, coder_turns
tok = AutoTokenizer.from_pretrained(str(SERVED_12B))

def tokenize(body):
    url = server.base_url.rstrip('/').removesuffix('/v1') + '/tokenize'
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)['tokens']

def ids(x):
    return list(x['input_ids'] if hasattr(x, 'keys') else x)

turns = coder_turns(TRACE_DIR)
by_task = {}
for t in turns:
    by_task.setdefault(t['variant'].split('-')[0], {}).setdefault(t['variant'], []).append(t)
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
            ours = ids(tok.apply_chat_template(as_vllm(msgs), tools=req.get('tools'), add_generation_prompt=gen,
                                               tokenize=True, **kw))
            theirs = tokenize({'model': SERVED_MODEL, 'messages': msgs, 'tools': req.get('tools'),
                               'add_generation_prompt': gen, 'chat_template_kwargs': kw})
            diff = first_difference(ours, theirs)
            row[kind] = {'ours': len(ours), 'server': len(theirs), 'equal': ours == theirs, 'first_diff': diff,
                         'ours_text': tok.decode(ours[max(0, (diff or 0) - 20):(diff or 0) + 30]) if diff is not None else None,
                         'server_text': tok.decode(theirs[max(0, (diff or 0) - 20):(diff or 0) + 30]) if diff is not None else None}
    except Exception as e:
        row['error'] = f'{type(e).__name__}: {e}'[:400]
    results.append(row)
PARITY_VERDICT = parity_verdict(results)
(WORK / 'parity.json').write_text(json.dumps({'verdict': PARITY_VERDICT, 'sample': len(results),
                                              'errors': sum('error' in r for r in results), 'rows': results}, indent=1))
print('PARITY:', PARITY_VERDICT, '| rows', len(results), '| errors', sum('error' in r for r in results),
      '| minutes left', round(minutes_left(DEADLINE_12B)))
'''

RUN12B = r'''import subprocess, sys, tarfile
ARMS12B = WORK / 'g1_12b'
done12b = []

def save12b():
    if ARMS12B.exists():
        with tarfile.open(WORK / 'g1_12b.tar.gz', 'w:gz') as tar:
            tar.add(ARMS12B, arcname='g1_12b', filter=lambda ti: None if '/submission/' in ti.name + '/' else ti)
    (WORK / 'g1_12b_progress.json').write_text(json.dumps({'done': done12b}, indent=1))

def run_arm(name, task):
    cmd = [sys.executable, str(WORK / 'run_eval.py'), '--submission', str(ARM_DIRS[name]),
           '--tasks', str(DATA_DIR / 'tasks.jsonl'), '--snapshots', str(DATA_DIR / 'snapshots'),
           '--graphs', str(DATA_DIR / 'graphs'), '--embeddings', str(DATA_DIR / 'embeddings'),
           '--wheels', str(DATA_DIR / 'wheels'), '--results', str(ARMS12B / name / task), '--task-ids', task,
           '--sandbox', 'subprocess', '--api-base', server.base_url, '--served-model', SERVED_MODEL,
           '--model-prefix', 'openai/', '--label', f'kaggle-g1-{name}', '--hardware', '4xL4']
    t0 = time.time()
    rc, _ = run_logged(cmd, 'g1_12b.log')
    return {'task': task, 'arm': name, 'rc': rc, 'minutes': round((time.time() - t0) / 60, 2)}

try:
    for task in RESERVE_IDS:
        for name in ('A12', 'B12'):
            if minutes_left(DEADLINE_12B) < TASK_MIN:
                print('12B budget: stopping before', task, '| left', round(minutes_left(DEADLINE_12B)))
                raise StopIteration
            done12b.append(run_arm(name, task))
            print(done12b[-1], '| left', round(minutes_left(DEADLINE_12B)), flush=True)
            save12b()
except StopIteration:
    pass
finally:
    save12b()
print('12B arms:', len(done12b), 'runs |', sum(1 for t in RESERVE_IDS if sum(d['task'] == t for d in done12b) == 2),
      'tasks with both arms | left', round(minutes_left(DEADLINE_12B)))
'''

ANALYZER12B = r'''ANALYZER_12B = run_analyzer('12b', ARM_DIRS['ANC'], server, SERVED_MODEL, DEADLINE_12B)
'''

LOAD31B = r'''server.stop() if hasattr(server, 'stop') else None
server = boot_server(SERVED_31B, EXTRA_31B)
SERVED_MODEL = getattr(server, 'effective_model_path', str(SERVED_31B))
ping = litellm.completion(model='openai/' + SERVED_MODEL, api_base=server.base_url, api_key='EMPTY',
                          messages=[{'role': 'user', 'content': 'ping'}], max_tokens=4, timeout=300)
DEADLINE_31B = time.time() + BUDGET_31B_MIN * 60
print('31B up at', server.base_url, '| served', SERVED_MODEL)
'''

RUN31B = r'''import tarfile
ARMS31B = WORK / 'g1_31b'
done31b = []

def save31b():
    if ARMS31B.exists():
        with tarfile.open(WORK / 'g1_31b.tar.gz', 'w:gz') as tar:
            tar.add(ARMS31B, arcname='g1_31b', filter=lambda ti: None if '/submission/' in ti.name + '/' else ti)
    (WORK / 'g1_31b_progress.json').write_text(json.dumps({'done': done31b}, indent=1))

def run_anc(task):
    cmd = [sys.executable, str(WORK / 'run_eval.py'), '--submission', str(ARM_DIRS['ANC']),
           '--tasks', str(DATA_DIR / 'tasks.jsonl'), '--snapshots', str(DATA_DIR / 'snapshots'),
           '--graphs', str(DATA_DIR / 'graphs'), '--embeddings', str(DATA_DIR / 'embeddings'),
           '--wheels', str(DATA_DIR / 'wheels'), '--results', str(ARMS31B / 'ANC' / task), '--task-ids', task,
           '--sandbox', 'subprocess', '--api-base', server.base_url, '--served-model', SERVED_MODEL,
           '--model-prefix', 'openai/', '--label', 'kaggle-g1-ANC', '--hardware', '4xL4']
    t0 = time.time()
    rc, _ = run_logged(cmd, 'g1_31b.log')
    return {'task': task, 'arm': 'ANC', 'rc': rc, 'minutes': round((time.time() - t0) / 60, 2)}

try:
    for task in RESERVE_IDS:
        if minutes_left(DEADLINE_31B) < TASK_MIN:
            print('31B budget: stopping before', task, '| left', round(minutes_left(DEADLINE_31B)))
            break
        done31b.append(run_anc(task))
        print(done31b[-1], '| left', round(minutes_left(DEADLINE_31B)), flush=True)
        save31b()
finally:
    save31b()
print('ANC:', len(done31b), 'runs | left', round(minutes_left(DEADLINE_31B)))
'''

ANALYZER31B = r'''ANALYZER_31B = run_analyzer('31b', ARM_DIRS['ANC'], server, SERVED_MODEL, DEADLINE_31B)
'''

SUMMARY = r'''server.stop() if hasattr(server, 'stop') else None
(WORK / 'g1.json').write_text(json.dumps({
    'arms': {'A12': A12_SHA256, 'B12': B12_SHA256, 'ANC': ANC_SHA256},
    'run_eval_sha256': RUN_EVAL_SHA256, 'vllm_patch_sha256': VLLM_PATCH_SHA256,
    'patch_applied': PATCH_APPLIED, 'parity': PARITY_VERDICT,
    'reserve': RESERVE_IDS, 'pool_without_traces': POOL_IDS,
    'runs_12b': done12b, 'runs_31b': done31b,
    'analyzer_12b': ANALYZER_12B, 'analyzer_31b': ANALYZER_31B,
    'budgets_min': {'12b': BUDGET_12B_MIN, 'anc': BUDGET_31B_MIN},
    'minutes_left_12b': round(minutes_left(DEADLINE_12B)) if DEADLINE_12B else None,
    'minutes_left_31b': round(minutes_left(DEADLINE_31B)) if DEADLINE_31B else None,
}, indent=1))
print('g1.json written | arms', len(done12b), '+', len(done31b), '| analyzer', ANALYZER_12B, '+', ANALYZER_31B,
      '| parity', PARITY_VERDICT, '| patch', PATCH_APPLIED)
'''


def _consts(a12: Path, b12: Path, anc: Path, runner: Path, patch: Path,
            reserve: list[str], pool: list[str], budget_12b: int, budget_31b: int) -> str:
    from lab.export.kaggle_lora_smoke_notebook import TRAIN_SRC

    def embedded(name: str, package: Path) -> list[str]:
        return [f"{name}_SHA256 = {hashlib.sha256(package.read_bytes()).hexdigest()!r}  # {_rel(package)}",
                f"{name}_B64 = {base64.b64encode(package.read_bytes()).decode()!r}"]

    lines = ["import json", f"LORA_DATA = {LORA_DATA!r}"]
    for name, package in (("A12", a12), ("B12", b12), ("ANC", anc)):
        lines += embedded(name, package)
    lines += [f"RUN_EVAL_SHA256 = {hashlib.sha256(runner.read_bytes()).hexdigest()!r}",
              f"RUN_EVAL_B64 = {base64.b64encode(runner.read_bytes()).decode()!r}",
              f"VLLM_PATCH_SHA256 = {hashlib.sha256(patch.read_bytes()).hexdigest()!r}",
              f"VLLM_PATCH_PY = {patch.read_text()!r}",
              f"TRAIN_PY = {TRAIN_SRC.read_text()!r}",
              f"MODEL_12B_PATH = {models.path('12b')!r}", f"MODEL_12B_TOKEN = {models.token('12b')!r}",
              f"MODEL_31B_PATH = {models.path('31b')!r}", f"MODEL_31B_TOKEN = {models.token('31b')!r}",
              f"EXTRA_12B = {list(models.extra_args('12b'))!r}", f"EXTRA_31B = {list(models.extra_args('31b'))!r}",
              f"BUDGET_12B_MIN = {budget_12b}", f"BUDGET_31B_MIN = {budget_31b}",
              f"RESERVE_IDS = {sorted(reserve)!r}", f"POOL_IDS = {sorted(pool)!r}"]
    return "\n".join(lines) + "\n"


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def build(out: Path, slug: str, a12: Path, b12: Path, anc: Path, budget_12b: int, budget_31b: int,
          reserve: list[str], pool: list[str], arms: list[str] | None = None) -> dict:
    runner = ROOT / "lab/src/lab/eval/run_eval.py"
    patch = Path(vllm_patch.__file__)
    consts = _consts(a12, b12, anc, runner, patch, reserve, pool, budget_12b, budget_31b)
    # The parity cell runs in the kernel and cannot import lab.*, so the two helpers it uses travel inlined.
    parity = _helpers_src() + "\n" + PARITY
    header = ("# G1\n\n12B against 31B on the 16 reserve tasks (doc 03 §14.30): A12 and B12 interleaved per task on one "
              "12B server, then ANC on the 31B, plus 12B render parity against vLLM `/tokenize` and the extra "
              "analyzer-only cell over the pool tasks without traces. Applies the forum 748287 `vision_embedder` "
              "patch if the 12B fails to load. Writes `parity.json`, `g1_12b*`, `g1_31b*`, `analyzer_12b.jsonl`, "
              "`analyzer_31b.jsonl` and `g1.json`.")
    cells = [{"cell_type": "markdown", "metadata": {}, "source": [header]},
             cell(VERSIONS), cell(SETUP), cell(PREFLIGHT), cell(consts),
             cell(MATERIALIZE), cell(SERVERS)]
    arm_set = set((arms or ["A12", "B12", "ANC"]))
    run_12b = any(a in arm_set for a in ("A12", "B12"))
    if run_12b:
        cells.append(cell(LOAD12B))
        cells.append(cell(parity))
        cells.append(cell(RUN12B))
        cells.append(cell(ANALYZER12B))
    cells.append(cell(LOAD31B))
    cells.append(cell(RUN31B))
    cells.append(cell(ANALYZER31B))
    cells.append(cell(SUMMARY))
    notebook = {
        "cells": with_cell_ids(cells),
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    meta = {"id": f"{OWNER}/{slug}", "title": slug.replace("-", " "), "code_file": "g1.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
            "enable_tpu": False, "enable_internet": False, "dataset_sources": [WHEELHOUSE, LORA_DATA],
            "kernel_sources": [], "competition_sources": [COMPETITION],
            "model_sources": [models.source("12b"), models.source("31b")],
            "docker_image": DOCKER_IMAGE, "machine_shape": "NvidiaL4"}
    return {"g1.ipynb": notebook, "kernel-metadata.json": meta}


def _helpers_src() -> str:
    import inspect
    return inspect.getsource(first_difference).rstrip() + "\n\n" + inspect.getsource(parity_verdict).rstrip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=ROOT / "build/kaggle_g1", type=Path)
    ap.add_argument("--slug", default="gemma4-g1-12b-vs-31b")
    ap.add_argument("--a12", type=Path, default=DEFAULT_A12)
    ap.add_argument("--b12", type=Path, default=DEFAULT_B12)
    ap.add_argument("--anc", type=Path, default=DEFAULT_ANC)
    ap.add_argument("--budget-12b-min", type=int, default=120)
    ap.add_argument("--budget-31b-min", type=int, default=75)
    ap.add_argument("--reserve-tasks", default="", help="comma-separated; default: the 16 reserve tasks")
    ap.add_argument("--pool-tasks", default="", help="comma-separated; default: the embedded pool-without-traces list")
    ap.add_argument("--arms", default="A12,B12,ANC", help="comma-separated arm set to run (default A12,B12,ANC)")
    args = ap.parse_args()
    sys.path.insert(0, str(ROOT / "lab/src"))
    from lab.lora.scripted import RESERVE
    reserve = [t for t in args.reserve_tasks.split(",") if t] or sorted(RESERVE)
    pool = [t for t in args.pool_tasks.split(",") if t] or sorted(POOL_WITHOUT_TRACES)
    arms = [a.strip().upper() for a in args.arms.split(",") if a.strip()]
    files = build(args.out, args.slug, args.a12, args.b12, args.anc, args.budget_12b_min, args.budget_31b_min,
                  reserve, pool, arms)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, body in files.items():
        (args.out / name).write_text(json.dumps(body, indent=1) + "\n")
    nb = files["g1.ipynb"]
    print(f"kernel {files['kernel-metadata.json']['id']} -> {args.out} | cells {len(nb['cells'])} | "
          f"reserve {len(reserve)} | pool {len(pool)}")


if __name__ == "__main__":
    main()
