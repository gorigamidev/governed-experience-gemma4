"""TPU smoke test for the behavior LoRA: can `train.py`'s recipe train the 31B on a Kaggle TPU v5e-8 with torch_xla?

No adapter from this run is used. It answers, stage by stage, what would break a real G2 training on TPU:

  env     peft from the wheel dataset, torch_xla devices and SPMD mesh
  data    our coder turns rendered and merged with the checkpoint's tokenizer (train.py)
  load    the bf16 checkpoint into host RAM
  shard   LoRA attached, every parameter moved to the TPUs and sharded FSDP-style (one tensor at a time)
  step    3 optimizer steps per length bucket (batch = devices): compile time, step time, peak HBM, finite loss
  learn   repeated steps on one batch: the loss must fall
  save    the adapter in PEFT format: key names, finite values, lora_B no longer zero
  verdict throughput per bucket and the projected time of one epoch of our data

Static shapes: XLA recompiles for every new shape, so examples are padded to a bucket and the labeled positions to a
fixed count per bucket (`select_labeled`). The loss is the same as `train.labeled_loss` (test in test_lora_scripted.py).

Local pipeline test (CPU, tiny random Gemma 4):
  lab/.venv-train/bin/python lab/src/lab/lora/tpu_smoke.py --tiny data/models/e4b-bf16/config.json \
      --tokenizer data/models/gemma4-31b-tok --data data/lora/b --out build/tpu-smoke-local --buckets 512,1024 --batch 2
"""

from __future__ import annotations

import argparse
import json
import math
import os
import resource
import subprocess
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import LORA_TARGETS, coder_turns, merge, split_by_task, tiny_model  # noqa: E402

REPORT: dict = {"stages": {}}
REPORT_PATH: list[Path] = []


def stage(name: str, **fields) -> None:
    """Record a stage and rewrite the report: a process killed by the host (out of RAM) still leaves the stages run."""
    REPORT["stages"][name] = fields
    print(f"[{name}] {json.dumps(fields, default=str)}", flush=True)
    if REPORT_PATH:
        REPORT_PATH[0].write_text(json.dumps(REPORT, indent=1, default=str))


def select_labeled(labels: list[int], k: int) -> tuple[list[int], list[int], list[float], int]:
    """Up to k (logit position, target id) pairs, padded to exactly k with zero weight: a fixed shape for XLA.

    Returns (positions, targets, weights, dropped): logit i predicts token i + 1, so a labeled token at i > 0 comes
    from position i - 1. Labeled tokens beyond k are dropped and counted (the smoke test caps them; G2 would not)."""
    pairs = [(i - 1, t) for i, t in enumerate(labels) if t != -100 and i > 0]
    kept = pairs[:k]
    pad = k - len(kept)
    return ([p for p, _ in kept] + [0] * pad, [t for _, t in kept] + [0] * pad,
            [1.0] * len(kept) + [0.0] * pad, len(pairs) - len(kept))


def batch_loss(hf_model, ids, mask, pos, tgt, w):
    """Mean cross-entropy over the weighted (position, target) pairs of each row: `train.labeled_loss` for a padded
    batch. Gathers the hidden states first, so the logits are [batch, k, vocab] instead of [batch, length, vocab]."""
    import torch

    h = hf_model.model(input_ids=ids, attention_mask=mask, return_dict=True).last_hidden_state
    hk = torch.gather(h, 1, pos.unsqueeze(-1).expand(-1, -1, h.size(-1)))
    logits = hf_model.lm_head(hk)
    cap = hf_model.config.get_text_config().final_logit_softcapping
    if cap is not None:
        logits = torch.tanh(logits / cap) * cap
    ce = torch.nn.functional.cross_entropy(logits.float().reshape(-1, logits.size(-1)), tgt.reshape(-1),
                                           reduction="none")
    return (ce * w.reshape(-1)).sum() / w.sum()


def make_batch(examples: list[dict], length: int, k: int, pad_id: int):
    import torch

    ids, mask, pos, tgt, w, dropped = [], [], [], [], [], 0
    for e in examples:
        n = len(e["input_ids"])
        ids.append(e["input_ids"] + [pad_id] * (length - n))
        mask.append([1] * n + [0] * (length - n))
        p, t, ww, d = select_labeled(e["labels"], k)
        pos.append(p), tgt.append(t), w.append(ww)
        dropped += d
    return ([torch.tensor(x) for x in (ids, mask, pos, tgt)] + [torch.tensor(w, dtype=torch.float32)]), dropped


def pick(examples: list[dict], low: int, high: int, n: int) -> list[dict]:
    """n examples with low < length <= high, longest first, repeated if there are fewer than n."""
    fit = sorted((e for e in examples if low < len(e["input_ids"]) <= high), key=lambda e: -len(e["input_ids"]))
    return [fit[i % len(fit)] for i in range(n)] if fit else []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, help="defaults to --model")
    ap.add_argument("--model", type=Path, help="bf16 checkpoint (TPU run)")
    ap.add_argument("--tiny", type=Path, help="config.json for a tiny random model (local CPU run)")
    ap.add_argument("--device", choices=["auto", "xla", "cpu"], default="auto",
                    help="auto: xla when torch_xla is importable (cpu runs the --model path in a local rehearsal)")
    ap.add_argument("--peft-wheels", type=Path, help="dir with the peft wheel (installed with --no-deps)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--buckets", default="4096,8192,16384")
    ap.add_argument("--batch", type=int, default=0, help="rows per step (default: number of devices)")
    ap.add_argument("--labels-per-row", type=int, default=1024,
                    help="labeled positions per row (fixed for XLA); logits are rows x this x vocab")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--learn-steps", type=int, default=10)
    ap.add_argument("--budget-min", type=float, default=150, help="skip remaining stages after this many minutes")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.append(a.out / "tpu_smoke_report.json")
    t_start = time.time()
    try:
        run(a, t_start)
    except Exception as e:  # report the failing stage and keep the report
        REPORT["failed"] = {"stage": REPORT.get("current"), "error": repr(e)[:2000], "trace": traceback.format_exc()[-6000:]}
        print("FAILED", json.dumps(REPORT["failed"])[:3000], flush=True)
    REPORT["minutes"] = round((time.time() - t_start) / 60, 1)
    (a.out / "tpu_smoke_report.json").write_text(json.dumps(REPORT, indent=1, default=str))
    print("REPORT", json.dumps({k: v for k, v in REPORT.items() if k != "stages"}, default=str), flush=True)
    sys.exit(1 if "failed" in REPORT else 0)


def run(a, t_start: float) -> None:
    def over_budget() -> bool:
        return (time.time() - t_start) / 60 > a.budget_min

    REPORT["current"] = "env"
    if a.peft_wheels:
        whl = sorted(a.peft_wheels.rglob("peft-*.whl"))
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", str(whl[-1])], check=True)
    import torch
    import transformers
    import peft
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
    from transformers import AutoTokenizer

    if a.device == "auto":
        import importlib.util
        a.device = "xla" if importlib.util.find_spec("torch_xla") else "cpu"
    xla = a.device == "xla"
    env = {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__}
    if xla:
        import numpy as np
        import torch_xla
        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.spmd as xs
        import torch_xla.runtime as xr
        xr.use_spmd()
        n_dev = xr.global_runtime_device_count()
        mesh = xs.Mesh(np.arange(n_dev), (n_dev, 1), ("fsdp", "model"))
        device = torch_xla.device()
        env.update(torch_xla=torch_xla.__version__, device_type=xr.device_type(), devices=n_dev)
    else:
        n_dev, device = 1, torch.device("cpu")
        env.update(device_type="cpu", devices=1)
    batch = a.batch or n_dev
    stage("env", batch=batch, **env)

    REPORT["current"] = "data"
    tok = AutoTokenizer.from_pretrained(a.tokenizer or a.model)
    buckets = [int(b) for b in a.buckets.split(",")]
    train_turns, _ = split_by_task(coder_turns(a.data), 0.1, 20261005)
    by_variant: dict[str, list[dict]] = {}
    for t in train_turns:
        by_variant.setdefault(t["variant"], []).append(t)
    examples = [e for v in sorted(by_variant) for e in merge(tok, by_variant[v], buckets[-1])]
    lengths = sorted(len(e["input_ids"]) for e in examples)
    if not lengths:
        raise SystemExit(f"no merged example fits the largest bucket ({buckets[-1]} tokens)")
    per_row = sorted(sum(1 for x in e["labels"] if x != -100) for e in examples)
    loss_tokens = sum(per_row)
    stage("data", turns=len(train_turns), examples=len(examples), tokens=sum(lengths), loss_tokens=loss_tokens, max_labels_per_example=per_row[-1] if per_row else 0,
          examples_over_label_cap=sum(1 for n in per_row if n > a.labels_per_row),
          median_len=lengths[len(lengths) // 2], max_len=lengths[-1],
          per_bucket={b: sum(1 for n in lengths if lo < n <= b) for lo, b in zip([0] + buckets, buckets)})

    REPORT["current"] = "load"
    t0 = time.time()
    if a.tiny:
        model = tiny_model(a.tiny)
    else:
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    n_params = sum(p.numel() for p in model.parameters())
    stage("load", seconds=round(time.time() - t0), params_b=round(n_params / 1e9, 2),
          host_peak_rss_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2**30 if sys.platform == "darwin" else 2**20), 1))

    REPORT["current"] = "shard"
    t0 = time.time()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    import peft.tuners.lora.torchao as peft_torchao  # same workaround as train.py
    peft_torchao.is_torchao_available = lambda: False
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=0.0,
                                             target_modules=LORA_TARGETS, bias="none"))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ckpt_fn = "torch.utils.checkpoint"
    sharded = replicated = 0
    if xla:
        # torch_xla's checkpoint keeps XLA from fusing the recomputation away (the memory saving of checkpointing);
        # reentrant, as torch_xla's own FSDP wrapper uses it.
        import functools
        import torch_xla.utils.checkpoint as xla_ckpt
        for m in model.modules():
            if hasattr(m, "_gradient_checkpointing_func"):
                m._gradient_checkpointing_func = functools.partial(xla_ckpt.checkpoint, use_reentrant=True)
                ckpt_fn = "torch_xla.utils.checkpoint"
        # One tensor at a time: moving the whole 62 GB model first would replicate it on every 16 GB chip. A tensor
        # shared by two modules (tied embedding and lm_head) is moved once and stays shared.
        moved: dict[int, torch.nn.Parameter] = {}
        for m in model.modules():
            for name, p in list(m.named_parameters(recurse=False)):
                if id(p) not in moved:
                    q = torch.nn.Parameter(p.data.to(device), requires_grad=p.requires_grad)
                    dims = [i for i, s in enumerate(q.shape) if s % n_dev == 0]
                    if dims and q.numel() >= 2**16:
                        spec = [None] * q.dim()
                        spec[max(dims, key=lambda i: q.shape[i])] = "fsdp"
                        xs.mark_sharding(q, mesh, tuple(spec))
                        sharded += 1
                    else:
                        replicated += 1
                    moved[id(p)] = q
                setattr(m, name, moved[id(p)])
            for name, b in list(m.named_buffers(recurse=False)):
                m._buffers[name] = b.to(device)
            xm.mark_step()
    stage("shard", seconds=round(time.time() - t0), trainable_m=round(trainable / 1e6, 1), checkpoint=ckpt_fn,
          sharded=sharded, replicated=replicated, hbm=hbm(xla, device))

    hf = model.base_model.model
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    def step(tensors):
        ids, mask, pos, tgt, w = (t.to(device) for t in tensors)
        if xla:
            for t in (ids, mask, pos, tgt, w):
                xs.mark_sharding(t, mesh, ("fsdp",) + (None,) * (t.dim() - 1))
        loss = batch_loss(hf, ids, mask, pos, tgt, w)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad()
        if xla:
            xm.mark_step()
        return loss.item()

    def bucket_steps(low: int, length: int) -> None:
        name = f"step@{length}"
        REPORT["current"] = name
        if over_budget():
            stage(name, skipped="budget")
            return
        rows = pick(examples, low, length, batch) or pick(examples, 0, length, batch)
        if not rows:
            stage(name, skipped="no example fits")
            return
        tensors, dropped = make_batch(rows, length, a.labels_per_row, pad_id)
        times, losses = [], []
        try:
            for _ in range(3):
                t0 = time.time()
                losses.append(round(step(tensors), 4))
                times.append(round(time.time() - t0, 1))
        except Exception as e:  # a long bucket may run out of HBM: report it, keep the other answers
            stage(name, error=repr(e)[:1500], losses=losses, times=times, hbm=hbm(xla, device))
            return
        real = sum(len(r["input_ids"]) for r in rows)
        stage(name, k=a.labels_per_row, real_tokens=real, dropped_labels=dropped, first_s=times[0],
              step_s=round(sum(times[1:]) / 2, 1), tokens_per_s=round(real / max(0.1, sum(times[1:]) / 2)),
              losses=losses, finite=all(x == x and abs(x) < 1e4 for x in losses), hbm=hbm(xla, device))

    # The shortest bucket, learning and saving first: they answer whether G2 can train here at all. The longer
    # buckets come last because an out-of-memory there can leave the runtime unusable.
    bucket_steps(0, buckets[0])

    REPORT["current"] = "learn"
    rows = pick(examples, 0, buckets[0], batch)
    tensors, _ = make_batch(rows, buckets[0], a.labels_per_row, pad_id)
    losses = [round(step(tensors), 4) for _ in range(a.learn_steps)]
    stage("learn", losses=losses, falls=losses[-1] < losses[0])

    REPORT["current"] = "save"
    t0 = time.time()
    state = {k: v.detach().cpu() for k, v in get_peft_model_state_dict(model).items()}
    model.save_pretrained(a.out / "adapter", state_dict=state)
    cfg = json.loads((a.out / "adapter/adapter_config.json").read_text())
    b_norm = sum(float(v.float().norm()) for k, v in state.items() if "lora_B" in k)
    stage("save", seconds=round(time.time() - t0), tensors=len(state), sample_keys=sorted(state)[:2],
          finite=all(bool(torch.isfinite(v.float()).all()) for v in state.values()), lora_b_norm=round(b_norm, 4),
          target_modules=cfg.get("target_modules"), r=cfg.get("r"))

    for low, length in zip(buckets, buckets[1:]):
        bucket_steps(low, length)

    REPORT["current"] = "verdict"
    st = REPORT["stages"]
    tps = {length: st[f"step@{length}"].get("tokens_per_s") for length in buckets}
    first = st[f"step@{buckets[0]}"]
    per_bucket = st["data"]["per_bucket"]
    stage("verdict", trains=bool(first.get("finite")) and st["learn"]["falls"] and st["save"]["finite"]
          and st["save"]["lora_b_norm"] > 0,
          longest_bucket_ok=max((b for b in buckets if tps[b]), default=None), tokens_per_s=tps,
          # Examples in a bucket that failed have no throughput: the estimate is then missing, not optimistic.
          epoch_minutes=round(sum(math.ceil(per_bucket[b] / batch) * st[f"step@{b}"]["step_s"]
                                  for b in buckets if per_bucket[b]) / 60, 1)
          if all(tps[b] for b in buckets if per_bucket[b]) else None)
    REPORT.pop("current", None)


def hbm(xla: bool, device) -> dict | str | None:
    """Device memory, best effort. Under SPMD torch_xla has no memory info for the virtual device (checked in a local
    torch_xla 2.8 run), so this asks the TPU runtime through `tpu-info` when the image has it. An out-of-memory error
    reports usage in its own message, which the step stage keeps."""
    if not xla:
        return None
    import shutil
    if not shutil.which("tpu-info"):
        return "unavailable"
    try:
        return subprocess.run(["tpu-info"], capture_output=True, text=True, timeout=60).stdout[-1500:]
    except Exception as e:  # memory stats are informative only
        return f"error: {e!r}"[:200]

if __name__ == "__main__":
    main()
