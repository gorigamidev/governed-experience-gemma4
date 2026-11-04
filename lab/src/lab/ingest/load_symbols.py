"""Exact per-task symbol tables rebuilt by reconstruct_symbols (async included, def-less modules included)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from lab.ingest.load_tasks import FileChange

ROOT = Path(__file__).resolve().parents[4]
SYMBOLS_DIR = ROOT / "data/symbols"


@dataclass(frozen=True)
class SymbolRow:
    path: str  # repo path as in the patch
    qualname: str  # inside the module, nested definitions included (same as the runtime indexer)
    kind: str
    is_async: bool
    start: int
    end: int


@dataclass
class SymbolTable:
    instance_id: str
    status: str  # exact | approximate | not_found
    files: list[str]
    rows: list[SymbolRow]

    def in_file(self, path: str) -> list[SymbolRow]:
        return [r for r in self.rows if r.path == path]


@lru_cache(maxsize=4)
def load_symbol_table(instance_id: str, symbols_dir: str = str(SYMBOLS_DIR)) -> SymbolTable | None:
    path = Path(symbols_dir) / f"{instance_id}.json"
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    if not d.get("base"):
        return SymbolTable(instance_id, d.get("status", "not_found"), [], [])
    rows = [SymbolRow(p, q, k, bool(a), s, e) for p, syms in d["symbols"].items() for q, k, a, s, e in syms]
    return SymbolTable(instance_id, d.get("status", "exact"), d["files"], rows)


def gold_symbols_by_lines(table: SymbolTable, change: FileChange) -> set[str]:
    """Innermost definitions whose line span contains a line the fix touches (module-level lines are skipped)."""
    rows = table.in_file(change.path)
    found = set()
    for hunk in change.hunks:
        for line in hunk.changed_at:
            covering = [r for r in rows if r.start <= line <= r.end]
            if covering:
                found.add(min(covering, key=lambda r: r.end - r.start).qualname)
    return found
