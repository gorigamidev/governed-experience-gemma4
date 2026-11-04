"""Shared runtime helpers: paths, compact JSON output with a size cap, invocation clock."""

from __future__ import annotations

import json
import os
import time

SCORER_WORKSPACE = "/workspace"


def _task_dirs() -> tuple[str, str]:
    """(repo, dir for the db). Docker sandbox: /workspace and /tmp. swegemma's subprocess sandbox has neither:
    the repo is a per-task dir put on PYTHONPATH, and TMPDIR is that task's own tmp."""
    if os.path.isdir(SCORER_WORKSPACE):
        return SCORER_WORKSPACE, "/tmp"
    for p in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        if p and os.path.basename(os.path.normpath(p)) == "workspace" and os.path.isdir(p):
            return p, os.environ.get("TMPDIR") or "/tmp"
    return SCORER_WORKSPACE, "/tmp"


_repo, _db_dir = _task_dirs()
WORKSPACE = os.environ.get("MINI_PALACE_WORKSPACE", _repo)
DB_PATH = os.environ.get("MINI_PALACE_DB", os.path.join(_db_dir, "mini-palace.db"))
# ADK runs scripts/<file>.py from a temp dir that also holds references/.
REFERENCES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "references")
MAX_OUTPUT_CHARS = 1500

_START = time.perf_counter()


def elapsed_ms() -> int:
    return int((time.perf_counter() - _START) * 1000)


def load_reference(name: str, default=None):
    try:
        with open(os.path.join(REFERENCES_DIR, name), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def dumps(obj: dict) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def emit(payload: dict, max_chars: int = MAX_OUTPUT_CHARS) -> str:
    """Serialize payload, dropping trailing items until it fits max_chars."""
    payload = dict(payload)
    payload["ms"] = elapsed_ms()
    out = dumps(payload)
    items = payload.get("items")
    while len(out) > max_chars and isinstance(items, list) and len(items) > 1:
        items.pop()
        payload["truncated"] = True
        out = dumps(payload)
    if len(out) > max_chars:
        out = out[: max_chars - 3] + "..."
    return out
