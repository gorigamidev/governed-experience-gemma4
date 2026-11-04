"""The Gemma 4 checkpoint is a parameter, not a constant (handoff O2).

The scorer serves `gemma-4-31b-it-qat-w4a16-ct`; doc 03 §14.30 adds the 12B. Every generator and `make variant`
take `MODEL=<short>` (default `31b`), so a 31B and a 12B build never collide in a directory, a slug or an archive
name. `MODEL` accepts a short name (`31b`, `12b`, `e4b`) or the full checkpoint name.
"""

from __future__ import annotations

DEFAULT = "31b"
VERSION = "2"  # Kaggle Models version of every checkpoint below
NAMES = {
    "31b": "gemma-4-31b-it-qat-w4a16-ct",
    "12b": "gemma-4-12b-it-qat-w4a16-ct",
    "e4b": "gemma-4-e4b-it-qat-w4a16-ct",  # small-model prize probe (doc 03 §14 item 41) and LoRA smoke/rehearsal
}


def short(model: str) -> str:
    if model in NAMES:
        return model
    for key, name in NAMES.items():
        if model == name:
            return key
    raise SystemExit(f"unknown model {model!r}: use one of {', '.join(sorted(NAMES))} or a full checkpoint name")


def name(model: str) -> str:
    return NAMES[short(model)]


def token(model: str) -> str:
    """Checkpoint name without the `gemma-4-` prefix, as it appears inside a path or a glob filter."""
    return name(model).removeprefix("gemma-4-")


def source(model: str) -> str:
    """Kaggle Models source of the W4A16 served checkpoint."""
    return f"google/gemma-4/Other/{name(model)}/{VERSION}"


def path(model: str) -> str:
    """Mount path of the same checkpoint inside a Kaggle notebook."""
    return f"/kaggle/input/models/google/gemma-4/other/{name(model)}/{VERSION}"


def bf16_source(model: str) -> str:
    """Kaggle Models source of the unquantized checkpoint the LoRA training loads."""
    return f"google/gemma-4/Transformers/gemma-4-{short(model)}-it-qat-q4_0-unquantized/{VERSION}"


def bf16_path(model: str) -> str:
    return f"/kaggle/input/models/google/gemma-4/transformers/gemma-4-{short(model)}-it-qat-q4_0-unquantized/{VERSION}"


def variant_suffix(model: str) -> str:
    """`-12b` for a non-default model so `build/<variant><suffix>/` and its slug are unambiguous."""
    key = short(model)
    return "" if key == DEFAULT else f"-{key}"


# Extra vLLM flags a checkpoint needs on top of the defaults, keyed by short name. The 12B entry is the staff's
# (Ryan Holbrook, forum 748287, 11 oct: "The scorer has this update already"); the Getting Started notebook was
# still not updated on 12 oct. A participant reports the 12B still fails to load on wheelhouse v30 with these flags
# (`KeyError: 'vision_embedder.patch_dense.weight'`, same thread, 12 oct), so every 12B notebook loads it first.
VLLM_EXTRA_ARGS: dict[str, tuple[str, ...]] = {
    "12b": ("--hf-overrides", '{"architectures": ["Gemma4ForCausalLM"], "model_type": "gemma4"}'),
}


def extra_args(model: str) -> tuple[str, ...]:
    return VLLM_EXTRA_ARGS.get(short(model), ())
