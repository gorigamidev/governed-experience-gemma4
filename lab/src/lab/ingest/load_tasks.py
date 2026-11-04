"""Load competition tasks and parse their reference patches."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@ ?(.*)$")
DEF_RE = re.compile(r"^\s*(?:async\s+)?(def|class)\s+([A-Za-z_]\w*)")


@dataclass
class Hunk:
    old_start: int
    old_len: int
    new_start: int
    new_len: int
    context: str  # enclosing def/class reported by git in the hunk header
    old_lines: list[str] = field(default_factory=list)  # context + removed lines, i.e. the code before the fix
    removed: list[str] = field(default_factory=list)
    # Old-file line numbers the fix touches: removed lines, plus the line an addition is inserted after.
    changed_at: list[int] = field(default_factory=list)
    _cursor: int = 0


@dataclass
class FileChange:
    path: str
    is_new: bool = False
    is_deleted: bool = False
    hunks: list[Hunk] = field(default_factory=list)
    added: int = 0
    removed: int = 0
    defined: set[str] = field(default_factory=set)  # def/class names on added lines

    @property
    def kind(self) -> str:
        p = self.path
        if p.startswith(("tests/", "test/")) or "/tests/" in p or p.split("/")[-1].startswith("test_"):
            return "test"
        if p.startswith(("docs/", "docs_src/")) or p.endswith((".md", ".rst")):
            return "docs"
        if p.endswith(".py"):
            return "source"
        return "other"

    @property
    def context_symbols(self) -> set[str]:
        names = set()
        for h in self.hunks:
            m = DEF_RE.match(h.context)
            if m:
                names.add(m.group(2))
        return names


@dataclass
class Task:
    instance_id: str
    repo: str
    base_commit: str
    created_at: str
    problem_statement: str
    hints_text: str
    patch: str
    test_patch: str

    @property
    def repo_short(self) -> str:
        return self.instance_id.rsplit("_", 1)[0]


def parse_patch(patch: str) -> list[FileChange]:
    files: list[FileChange] = []
    current: FileChange | None = None
    old_path = ""
    for line in patch.splitlines():
        if line.startswith("--- "):
            old_path = line[4:].strip()
            continue
        if line.startswith("+++ "):
            new_path = line[4:].strip()
            deleted = new_path == "/dev/null"
            path = old_path if deleted else new_path
            path = path[2:] if path.startswith(("a/", "b/")) else path
            current = FileChange(path=path, is_new=old_path == "/dev/null", is_deleted=deleted)
            files.append(current)
            continue
        if current is None:
            continue
        m = HUNK_RE.match(line)
        if m:
            old_start, old_len = int(m.group(1)), int(m.group(2) or 1)
            hunk = Hunk(old_start, old_len, int(m.group(3)), int(m.group(4) or 1), m.group(5))
            hunk._cursor = old_start
            current.hunks.append(hunk)
            # Some patches mark new files only through a "-0,0" hunk, not /dev/null.
            if old_start == 0 and old_len == 0:
                current.is_new = True
        elif line.startswith("+"):
            current.added += 1
            if current.hunks:
                h = current.hunks[-1]
                h.changed_at.append(max(1, h._cursor - 1))
            d = DEF_RE.match(line[1:])
            if d:
                current.defined.add(d.group(2))
        elif line.startswith("-"):
            current.removed += 1
            if current.hunks:
                h = current.hunks[-1]
                h.old_lines.append(line[1:])
                h.removed.append(line[1:])
                h.changed_at.append(h._cursor)
                h._cursor += 1
        elif line.startswith(" ") and current.hunks:
            h = current.hunks[-1]
            h.old_lines.append(line[1:])
            h._cursor += 1
    return files


def load_tasks(path: str | Path) -> list[Task]:
    with open(path, encoding="utf-8") as f:
        return [Task(**{k: rec.get(k) or "" for k in Task.__dataclass_fields__}) for rec in map(json.loads, f)]
