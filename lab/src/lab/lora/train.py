"""Behavior LoRA: render logged harness turns with the checkpoint's chat template and train a LoRA adapter.

Each coder turn is a (prefix, target) pair: the exact request the harness sent (rendered as vLLM renders it,
with tool-call arguments as mappings) and the scripted response; loss only on targets. Turns of a variant that
are prefixes of each other are fused into one example (`merge`), token for token. The
same file runs locally on a tiny random Gemma 4 (pipeline test, `--tiny`) and in the Kaggle notebook.

Usage (local):  lab/.venv-train/bin/python lab/src/lab/lora/train.py --data data/lora/b --tokenizer \
                    data/models/gemma4-31b-tok --tiny data/models/e4b-bf16/config.json --out build/lora-tiny --max-steps 3
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import re
import time
from pathlib import Path

LORA_TARGETS = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
STOP_TOKENS = ("<|tool_response>", "<turn|>", "<eos>")  # generation_config.json eos_token_id [50, 106, 1]


def coder_turns(data_dir: Path) -> list[dict]:
    """Coder (request, response) pairs of the variants kept in plans.json (a rerun replaces its first try)."""
    kept = set(json.loads((data_dir / "plans.json").read_text()))
    rows = []
    for line in (data_dir / "requests.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("role") == "coder" and r.get("variant") in kept and not r.get("overrun") and not r.get("error") \
            and r.get("loss", True):
            rows.append(r)
    return rows


def as_vllm(messages: list[dict]) -> list[dict]:
    """The messages as vLLM 0.19.1 hands them to the chat template: tool-call arguments parsed into mappings, and
    string content wrapped as a list of text parts. The template renders a system message given as parts with a
    trailing space ("| trim + ' '"); without the parts our render was one token short at the system/tools boundary in
    42 of 42 turns checked against the 31B server's /tokenize (doc 03 §14.26, 7 Oct)."""
    out = copy.deepcopy(messages)
    for m in out:
        if isinstance(m.get("content"), str):
            m["content"] = [{"type": "text", "text": m["content"]}]
        for tc in m.get("tool_calls") or []:
            args = tc["function"].get("arguments")
            if isinstance(args, str):
                tc["function"]["arguments"] = json.loads(args) if args.strip() else {}
    return out


def render(tokenizer, turn: dict) -> tuple[str, str]:
    req = turn["request"]
    kw = dict(tools=req.get("tools"), tokenize=False, **(req.get("chat_template_kwargs") or {}))
    history = as_vllm(req["messages"])
    prompt = tokenizer.apply_chat_template(history, add_generation_prompt=True, **kw)
    full = tokenizer.apply_chat_template(history + as_vllm([turn["response"]]), add_generation_prompt=False, **kw)
    if not full.startswith(prompt):
        raise ValueError(f"{turn['variant']} turn {turn.get('turn')}: rendered prompt is not a prefix of prompt+response")
    target = full[len(prompt):]
    stops = [i + len(s) for s in STOP_TOKENS if (i := target.find(s)) >= 0]
    target = target[:min(stops)] if stops else target  # generation ends at the first stop token
    if turn["response"].get("tool_calls") and not target.endswith("<|tool_response>"):
        raise ValueError(f"{turn['variant']} turn {turn.get('turn')}: tool-call target does not end at <|tool_response>")
    return prompt, target


def tokenize(tokenizer, prompt: str, target: str, max_len: int) -> dict | None:
    p = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    t = tokenizer(target, add_special_tokens=False)["input_ids"]
    if len(p) + len(t) > max_len:
        return None
    return {"input_ids": p + t, "labels": [-100] * len(p) + t}


def merge(tokenizer, turns: list[dict], max_len: int) -> list[dict]:
    """Fuse the turns of one variant into as few examples as possible, losing nothing.

    Turn k can join the running example when its rendered prompt starts with everything rendered so far
    (prompt and target of the previous turn), and the token ids of the running example are exactly the ids of
    tokenizing turn k's prompt on its own (no merge across a BPE boundary). Otherwise it starts a new example.
    Thoughts before the last user message are dropped by the template, so the turn before the harness nudge never
    merges forward.
    """
    out, text, ids, labels = [], None, [], []
    for t in sorted(turns, key=lambda t: t["turn"]):
        prompt, target = render(tokenizer, t)
        p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        t_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
        if text is not None and prompt.startswith(text) and p_ids[:len(ids)] == ids and len(p_ids) + len(t_ids) <= max_len:
            labels += [-100] * (len(p_ids) - len(ids))
        else:
            if ids:
                out.append({"input_ids": ids, "labels": labels})
            if len(p_ids) + len(t_ids) > max_len:
                text, ids, labels = None, [], []
                continue
            labels = [-100] * len(p_ids)
        ids, labels, text = p_ids + t_ids, labels + t_ids, prompt + target
    if ids:
        out.append({"input_ids": ids, "labels": labels})
    return out


def labeled_positions(labels: list[int]) -> tuple[list[int], list[int]]:
    """(logit positions, labeled token positions): logit i predicts token i + 1.

    Merged examples carry several labeled segments (one per fused turn), not only a suffix. The first version of the
    loss took the last n tokens instead (rehearsal v6 and the cancelled 31B run, 6 Oct): 55% of what it trained
    should have been masked."""
    tokens = [i for i, x in enumerate(labels) if x != -100 and i > 0]
    return [i - 1 for i in tokens], tokens


def labeled_loss(model, ex: dict, device):
    """Cross-entropy on the labeled tokens only, with logits computed only at the positions that predict them
    (full-vocab logits of a 13k-token prompt take ~13 GB in float32)."""
    import torch

    keep, tokens = labeled_positions(ex["labels"])
    ids = torch.tensor([ex["input_ids"]], device=device)
    # A plain list, not a tensor: with the model split across GPUs a tensor index could sit on a device other than
    # the output layer's. The targets follow the logits' device for the same reason.
    logits = model(input_ids=ids, logits_to_keep=keep).logits[0].float()
    targets = torch.tensor([ex["input_ids"][i] for i in tokens], device=logits.device)
    return torch.nn.functional.cross_entropy(logits, targets)


def tiny_model(config_path: Path):
    """A random Gemma 4 with the real config's structure and toy sizes (local pipeline test only)."""
    from transformers import Gemma4Config, Gemma4ForConditionalGeneration

    c = json.loads(Path(config_path).read_text())
    c["text_config"].update(hidden_size=64, num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
                            head_dim=32, global_head_dim=32, intermediate_size=128, hidden_size_per_layer_input=16,
                            num_kv_shared_layers=2)
    # Both attention types before the shared-KV layers, which reuse the last KV of their type.
    c["text_config"]["layer_types"] = ["sliding_attention", "full_attention"] * 2
    for k in ("vision_config", "audio_config"):
        for key, val in (("hidden_size", 32), ("num_hidden_layers", 1), ("num_attention_heads", 2),
                         ("num_key_value_heads", 1), ("intermediate_size", 64), ("head_dim", 16)):
            if c.get(k) and key in c[k]:
                c[k][key] = val
    return Gemma4ForConditionalGeneration(Gemma4Config(**{k: v for k, v in c.items() if k != "architectures"}))


def split_by_task(turns: list[dict], val_frac: float, seed: int) -> tuple[list[dict], list[dict]]:
    tasks = sorted({re.sub(r"-\d+(-notest)?$", "", t["variant"]) for t in turns})
    random.Random(seed).shuffle(tasks)
    val = set(tasks[:max(1, math.ceil(len(tasks) * val_frac))]) if val_frac > 0 else set()
    return ([t for t in turns if re.sub(r"-\d+(-notest)?$", "", t["variant"]) not in val],
            [t for t in turns if re.sub(r"-\d+(-notest)?$", "", t["variant"]) in val])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True, help="dir with requests.jsonl and plans.json")
    ap.add_argument("--tokenizer", type=Path, required=True, help="dir with the checkpoint's tokenizer and chat template")
    ap.add_argument("--model", type=Path, help="bf16 checkpoint to train on")
    ap.add_argument("--tiny", type=Path, help="config.json for a tiny random model (local test)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--max-len", type=int, default=16384)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=20261005)
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--max-minutes", type=float, default=0,
                    help="stop training after this many minutes (from the first step) and still save the adapter")
    a = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer

    torch.manual_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    turns = coder_turns(a.data)
    train_turns, val_turns = split_by_task(turns, a.val_frac, a.seed)
    report = {"turns": len(turns), "train_turns": len(train_turns), "val_turns": len(val_turns), "dropped_long": 0}
    sets = {}
    for name, part in (("train", train_turns), ("val", val_turns)):
        by_variant: dict[str, list[dict]] = {}
        for t in part:
            by_variant.setdefault(t["variant"], []).append(t)
        sets[name] = [e for v in sorted(by_variant) for e in merge(tok, by_variant[v], a.max_len)]
        kept = sum(sum(1 for x in e["labels"] if x != -100) for e in sets[name])
        every = sum(len(tok(render(tok, t)[1], add_special_tokens=False)["input_ids"]) for t in part)
        report["dropped_long"] += every - kept  # target tokens of turns longer than --max-len
    report["train_examples"], report["val_examples"] = len(sets["train"]), len(sets["val"])
    report["train_tokens"] = sum(len(e["input_ids"]) for e in sets["train"])
    report["train_loss_tokens"] = sum(sum(1 for x in e["labels"] if x != -100) for e in sets["train"])
    print(json.dumps(report), flush=True)

    if a.tiny:
        model = tiny_model(a.tiny)
    else:
        from transformers import AutoModelForImageTextToText, BitsAndBytesConfig
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                                   bnb_4bit_use_double_quant=True) if a.load_4bit else None
        model = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16, device_map="auto",
                                                            quantization_config=quant)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    # The Kaggle image ships torchao 0.10.0 and peft 0.21.2 raises on any torchao below 0.16 instead of skipping its
    # dispatcher (rehearsal v3, 6 Oct). Nothing here uses torchao: report it as absent to peft's LoRA dispatch.
    import peft.tuners.lora.torchao as peft_torchao
    peft_torchao.is_torchao_available = lambda: False
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=0.0,
                                             target_modules=LORA_TARGETS, bias="none"))
    model.print_trainable_parameters()
    device = next(p for p in model.parameters() if p.requires_grad).device
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)

    def loss_of(ex):
        return labeled_loss(model, ex, device)

    @torch.no_grad()
    def val_loss():
        model.eval()
        ls = [loss_of(e).item() for e in sets["val"]]
        model.train()
        return sum(ls) / len(ls) if ls else None

    total = a.max_steps or math.ceil(len(sets["train"]) * a.epochs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, total // 10)) *
                                              max(0.0, 1 - s / total))
    history = {"val_initial": val_loss(), "steps": []}
    print("val_initial", history["val_initial"], flush=True)
    model.train()
    order, t0 = [], time.time()
    history["stopped_at_step"] = None
    for step in range(total):
        if a.max_minutes and time.time() - t0 > a.max_minutes * 60:
            history["stopped_at_step"] = step
            print(f"time budget reached ({a.max_minutes} min): stopping at step {step}/{total}", flush=True)
            break
        if not order:
            order = list(range(len(sets["train"])))
            random.Random(a.seed + step).shuffle(order)
        loss = loss_of(sets["train"][order.pop()])
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step(), sched.step(), opt.zero_grad()
        history["steps"].append(round(loss.item(), 4))
        if step % 10 == 0 or step == total - 1:
            print(f"step {step + 1}/{total} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
    history["val_final"] = val_loss()
    print("val_final", history["val_final"], flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(a.out)
    (a.out / "train_report.json").write_text(json.dumps({**report, **history, "args": {k: str(v) for k, v in vars(a).items()}}, indent=1))


if __name__ == "__main__":
    main()
