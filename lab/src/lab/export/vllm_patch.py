"""Load the 12B W4A16 checkpoint on vLLM by skipping its vision embedder (forum 748287, 12 oct).

With the staff's flags (`--hf-overrides` selecting `Gemma4ForCausalLM`, `models.VLLM_EXTRA_ARGS`) a participant
still reports the 12B failing to load on the wheelhouse's vLLM: `KeyError: 'vision_embedder.patch_dense.weight'`
(same thread, 12 oct). `Gemma4ForCausalLM.load_weights` builds an `AutoWeightsLoader` with a `skip_substrs` list
of multimodal weights (`"audio_tower."`, `"vision_tower."`, `"embed_audio."`, `"embed_vision."`), but this
checkpoint also carries `vision_embedder.*`. The reported fix adds `"vision_embedder."` to that same list; the
agent is text-only, so the tower is never used.

The patch is applied to the installed `.py` on disk, not by monkeypatching in process: vLLM spawns the workers
(`VLLM_WORKER_MULTIPROC_METHOD=spawn`), and a fresh worker re-reads the module, so only the file matters. The
anchors are checked; if the class, the method or its `skip` list moved, `patch_source` raises instead of writing
a half patch.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

MODULE = "vllm.model_executor.models.gemma4"
CLASS = "class Gemma4ForCausalLM"
METHOD = "def load_weights(self, weights"
SKIP_OPEN = "skip = [\n"
INSERT = '            "vision_embedder.",\n'
MARK = '"vision_embedder.",'


class PatchError(RuntimeError):
    """The anchor the patch expects is gone: stop instead of writing a partial patch."""


def module_path() -> Path:
    spec = importlib.util.find_spec(MODULE)
    if spec is None or not spec.origin:
        raise PatchError(f"{MODULE} is not importable in this environment")
    return Path(spec.origin)


def patch_source(text: str) -> str:
    """Add `"vision_embedder."` to the `skip` list of `Gemma4ForCausalLM.load_weights`. Idempotent."""
    if MARK in text:
        return text
    start = text.find(CLASS)
    if start < 0:
        raise PatchError(f"{CLASS} not found")
    method = text.find(METHOD, start)
    if method < 0:
        raise PatchError(f"{METHOD!r} not found in {CLASS}")
    anchor = text.find(SKIP_OPEN, method)
    if anchor < 0:
        raise PatchError(f"{SKIP_OPEN.strip()!r} list not found in {CLASS}.load_weights")
    close = text.find("]", anchor)
    if close < 0:
        raise PatchError("the skip list is not closed")
    cut = anchor + len(SKIP_OPEN)
    return text[:cut] + INSERT + text[cut:]


def apply_to_file(path: Path) -> bool:
    """Patch `path` in place. Return True when it changed, False when it was already patched."""
    original = path.read_text()
    patched = patch_source(original)
    if patched == original:
        return False
    path.write_text(patched)
    return True


__all__ = ["PatchError", "apply_to_file", "module_path", "patch_source"]
