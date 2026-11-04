"""Consolidated habits (read-only at runtime): activation is checked without an LLM, the effect is a ranking boost.

Each pattern is one JSON line produced offline by lab/consolidate (never written here):
  {"id": "P1a2b3c4", "v": 3, "kind": "file_term", "repo": "fastapi", "term": "depend",
   "file": "fastapi/dependencies/utils.py", "symbol": null, "w": 2.7}
`file` is in module space (see indexer.module_path); `symbol` is the qualname inside that module.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict

from indexer import module_path


def load(path: str) -> list[dict]:
    try:
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    except (OSError, ValueError):
        return []


def is_active(p: dict, repo: str, tokens: set[str]) -> bool:
    return p.get("repo") == repo and (not p.get("term") or p["term"] in tokens)


def boosts(patterns: list[dict], repo: str, tokens: set[str], files: list[str], scale: float = 1.0):
    """Return (file_boost, symbol_boost, active_ids) for the patterns whose activation holds here."""
    by_module = {module_path(f): f for f in files}
    file_boost: dict[str, float] = defaultdict(float)
    symbol_boost: dict[str, dict[str, float]] = defaultdict(dict)
    active = []
    for p in patterns:
        if not is_active(p, repo, tokens):
            continue
        f = by_module.get(p["file"])
        if f is None:
            continue
        active.append(p["id"])
        w = scale * p["w"]
        if p.get("symbol"):
            symbol_boost[f][p["symbol"]] = symbol_boost[f].get(p["symbol"], 0.0) + w
        else:
            file_boost[f] += w
    return dict(file_boost), dict(symbol_boost), active


def detect_repo(workspace: str) -> str:
    """Project name from packaging metadata; patterns only fire in the repo they were learned from."""
    for name in ("pyproject.toml", "setup.cfg", "setup.py"):
        try:
            with open(os.path.join(workspace, name), encoding="utf-8") as f:
                for line in f:
                    s = line.strip().replace(" ", "")
                    if s.startswith("name="):
                        return s.split("=", 1)[1].strip("\"',").lower()
        except OSError:
            continue
    return ""
