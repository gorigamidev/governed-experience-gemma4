"""Data audit for the behavior LoRA (doc 03 §14.25, rule 1) -> data/analysis/lora_data_audit.md.

Checks that run without vLLM: every coder turn renders with the checkpoint's chat template and its target ends at
a stop token; merged examples reproduce every turn token for token; no reserve-task text reaches any example; every
variant's submitted patch equals the reference. The `gemma4` tool-parser round trip needs vLLM and runs in the
rehearsal notebook unless `--vllm-parser` points at vLLM's `tool_parsers/gemma4_tool_parser.py`: its pure parsing
helpers (regex and argument parser) are loaded from that file and run here.

Usage: lab/.venv-train/bin/python lab/src/lab/lora/audit.py [--data data/lora/b] [--tokenizer data/models/gemma4-31b-tok]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(ROOT / "lab/src"))

from scripted import RESERVE  # noqa: E402
from train import coder_turns, merge, render, split_by_task  # noqa: E402

MIN_LINE = 30  # an added reserve line shorter than this is too generic to count as leaked code


def reserve_probes(tasks: list[dict]) -> dict[str, dict]:
    """Lines of each reserve task's text (and added code). Text lines that also occur in a non-reserve task's
    statement (repository PR templates) are kept apart in `shared`: they are not specific to the reserve task."""
    others = "\n".join(t["problem_statement"] + "\n" + (t.get("hints_text") or "") for t in tasks
                       if t["instance_id"] not in RESERVE)
    out = {}
    for t in tasks:
        if t["instance_id"] not in RESERVE:
            continue
        text = [ln.strip() for ln in (t["problem_statement"] + "\n" + (t.get("hints_text") or "")).splitlines()]
        added = [ln[1:].strip() for ln in t["patch"].splitlines() if ln.startswith("+") and not ln.startswith("+++")]
        long = [ln for ln in text if len(ln) >= 40]
        out[t["instance_id"]] = {"text": [ln for ln in long if ln not in others],
                                 "shared": [ln for ln in long if ln in others],
                                 "code": [ln for ln in added if len(ln) >= MIN_LINE]}
    return out


def load_gemma4_parser(path: Path):
    """The module-level helpers of vLLM's Gemma 4 tool parser (constants and argument parsing), without vLLM."""
    import re as stdlib_re
    src = path.read_text()
    start, end = src.index("# Gemma4 special tokens"), src.index("class Gemma4ToolParser")
    ns = {"re": stdlib_re, "json": json}
    exec(src[start:end], ns)
    regex = stdlib_re.compile(r"<\|tool_call>call:([\w\-\.]+)\{(.*?)\}<tool_call\|>", stdlib_re.DOTALL)  # line 341
    assert r'r"<\|tool_call>call:([\w\-\.]+)\{(.*?)\}<tool_call\|>"' in src, "tool_call_regex changed upstream"
    return regex, ns["_parse_gemma4_args"]


def parser_round_trip(tokenizer, turns: list[dict], parser_path: Path) -> list[str]:
    """Each tool-call target, as the model would emit it, parses back to the scripted call."""
    regex, parse_args = load_gemma4_parser(parser_path)
    bad = []
    for t in turns:
        calls = t["response"].get("tool_calls") or []
        _, target = render(tokenizer, t)
        content = target.split("<channel|>", 1)[1] if "<channel|>" in target else target
        found = [(name, parse_args(args)) for name, args in regex.findall(content.removesuffix("<|tool_response>"))]
        want = [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in calls]
        if found != want:
            bad.append(f"{t['variant']} turn {t['turn']}: {str(found)[:120]} != {str(want)[:120]}")
    return bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data/lora/b")
    ap.add_argument("--tokenizer", type=Path, default=ROOT / "data/models/gemma4-31b-tok")
    ap.add_argument("--out", type=Path, default=ROOT / "data/analysis/lora_data_audit.md")
    ap.add_argument("--vllm-parser", type=Path, help="vllm/tool_parsers/gemma4_tool_parser.py from the wheelhouse's vLLM")
    a = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    tasks = [json.loads(ln) for ln in (ROOT / "data/competition/tasks.jsonl").read_text().splitlines()]
    summary = json.loads((a.data / "summary.json").read_text())
    turns = coder_turns(a.data)
    by_variant = defaultdict(list)
    for t in turns:
        by_variant[t["variant"]].append(t)

    render_fail, stop_fail = [], []
    for t in turns:
        try:
            render(tok, t)
        except ValueError as e:
            (stop_fail if "stop" in str(e) or "tool_response" in str(e) else render_fail).append(str(e))
    examples, reproduced = [], 0
    for v, ts in by_variant.items():
        ex = merge(tok, ts, 16384)
        examples += ex
        for t in ts:
            p, g = render(tok, t)
            pi, gi = tok(p, add_special_tokens=False)["input_ids"], tok(g, add_special_tokens=False)["input_ids"]
            reproduced += any(e["input_ids"][:len(pi) + len(gi)] == pi + gi and
                              all(x != -100 for x in e["labels"][len(pi):len(pi) + len(gi)]) for e in ex)

    corpus = "\n".join(json.dumps(t["request"]) + json.dumps(t["response"]) for t in turns)
    corpus_plain = corpus.encode().decode("unicode_escape", errors="ignore")
    probes = reserve_probes(tasks)
    leaks_text = {i: [ln for ln in p["text"] if ln in corpus_plain] for i, p in probes.items()}
    leaks_code = {i: [ln for ln in p["code"] if ln in corpus_plain] for i, p in probes.items()}
    shared = sorted({ln for p in probes.values() for ln in p["shared"] if ln in corpus_plain})
    train, val = split_by_task(turns, 0.1, 20261005)
    n_tokens = sum(len(e["input_ids"]) for e in examples)
    n_loss = sum(sum(1 for x in e["labels"] if x != -100) for e in examples)
    repos = Counter(re.sub(r"_\d+.*$", "", v) for v in by_variant)
    tools = Counter(t["response"]["tool_calls"][0]["function"]["name"] if t["response"].get("tool_calls") else "DONE (text)"
                    for t in turns)
    patch_ok = sum(r.get("patch_equals_reference", False) for r in summary)
    parser_bad = parser_round_trip(tok, turns, a.vllm_parser) if a.vllm_parser else None
    for b in parser_bad or []:
        print("PARSER", b)

    checks = [
        ("Render con el chat template del 31B (prompt prefijo de prompt + respuesta)", not render_fail, f"{len(turns) - len(render_fail)} de {len(turns)}"),
        ("Objetivo de cada llamada termina en `<|tool_response>`", not stop_fail, f"{len(turns) - len(stop_fail)} de {len(turns)}"),
        ("Ejemplos fusionados reproducen cada turno token por token", reproduced == len(turns), f"{reproduced} de {len(turns)}"),
        ("Texto de las 16 tareas de reserva (enunciado, pistas) en algún ejemplo", not any(leaks_text.values()),
         f"{sum(map(len, leaks_text.values()))} líneas"),
        ("Parche enviado igual al de referencia", patch_ok == len(summary), f"{patch_ok} de {len(summary)}"),
        ("Ida y vuelta por el parser `gemma4` de vLLM", None if parser_bad is None else not parser_bad,
         "pendiente: corre en el notebook del ensayo" if parser_bad is None else
         f"{len(turns) - len(parser_bad)} de {len(turns)} (helpers de `{a.vllm_parser.name}`, vLLM del wheelhouse)"),
    ]
    mark = {True: "OK", False: "FALLA", None: "pendiente"}
    lines = [
        "# Auditoría de datos del LoRA de conducta (regla 1, doc 03 §14.25)", "",
        f"Generado por `lab/src/lab/lora/audit.py` sobre `{a.data.relative_to(ROOT)}`. Template: `{a.tokenizer.relative_to(ROOT)}`.", "",
        "| Comprobación | Estado | Detalle |", "|---|---|---|",
        *[f"| {name} | {mark[ok]} | {detail} |" for name, ok, detail in checks], "",
        "## Composición", "",
        f"- Variantes: {len(by_variant)} de {len({re.sub(r'-[0-9]+(-notest)?$', '', v) for v in by_variant})} tareas; por repo: {dict(repos)}.",
        f"- Mezcla A/B: 100% B (A no se generó; opcional por la enmienda 1).",
        f"- Turnos del coder: {len(turns)} (entrenamiento {len(train)}, validación {len(val)}, separados por tarea).",
        f"- Ejemplos fusionados: {len(examples)}; tokens {n_tokens:,}; con pérdida {n_loss:,} ({n_loss / n_tokens:.1%}).",
        f"- Acciones del coder (objetivos): {dict(tools)}.",
        f"- Con paso de tests: {sum(not v.endswith('notest') for v in by_variant)} variantes; sin él: {sum(v.endswith('notest') for v in by_variant)}.", "",
        "## Líneas compartidas con tareas fuera de la reserva", "",
        "Aparecen en los ejemplos y en enunciados de reserva, pero también en enunciados de tareas fuera de la reserva (plantillas de PR del repositorio), así que no cuentan como texto de una tarea de reserva:", "",
        *[f"- `{ln[:110]}`" for ln in shared], *(["- Ninguna."] if not shared else []), "",
        "## Código de las tareas de reserva en archivos leídos", "",
        "Líneas de al menos 30 caracteres que agregan los parches de reserva y aparecen en algún ejemplo (archivos de snapshots posteriores que ya traen ese arreglo). No es texto de la tarea; se reporta para medir el riesgo de memorizar una solución de la reserva.", "",
        *[f"- `{i}`: {len(v)} de {len(probes[i]['code'])} líneas" for i, v in sorted(leaks_code.items()) if v],
        *(["- Ninguna."] if not any(leaks_code.values()) else []), "",
    ]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
