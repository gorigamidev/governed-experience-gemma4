"""Level-1 index: every def/class in the repo (ast, with a line-scan fallback for files that do not parse)."""

from __future__ import annotations

import ast
import os
import re
from dataclasses import dataclass

DEF_RE = re.compile(r"^([ \t]*)(?:async[ \t]+)?(def|class)[ \t]+([A-Za-z_]\w*)")
SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", "build", "dist", ".tox", ".mypy_cache",
             ".pytest_cache", "site", ".eggs"}


@dataclass
class Symbol:
    qualname: str
    file: str
    lineno: int
    kind: str  # function | class | method


MODULE_PREFIXES = ("src/", "docs_src/")


def module_path(file_path: str) -> str:
    """Repo path -> shared module key used by patterns and precomputed graphs ("src/requests/utils.py" -> "requests/utils.py")."""
    for prefix in MODULE_PREFIXES:
        if file_path.startswith(prefix):
            file_path = file_path[len(prefix):]
            break
    return file_path.removesuffix("/__init__.py").removesuffix(".py") + ".py"


def is_test_path(path: str) -> bool:
    parts = path.split("/")
    return any(p in ("tests", "test", "testing") for p in parts[:-1]) or parts[-1].startswith(("test_", "conftest"))


def iter_py_files(root: str):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.endswith(".egg-info")]
        for name in filenames:
            if name.endswith(".py"):
                full = os.path.join(dirpath, name)
                yield os.path.relpath(full, root).replace(os.sep, "/")


def extract_symbols(source: str) -> list[list] | None:
    """[qualname, kind, is_async, start, end] for every def/class via ast; None if the file does not parse.

    Shared with lab/ (reconstruct_symbols), so the offline benchmark and the skill index identically.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return None
    out = []

    def walk(node, prefix, in_class):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                q = f"{prefix}.{ch.name}" if prefix else ch.name
                kind = "class" if isinstance(ch, ast.ClassDef) else ("method" if in_class else "function")
                out.append([q, kind, isinstance(ch, ast.AsyncFunctionDef), ch.lineno, ch.end_lineno])
                walk(ch, q, isinstance(ch, ast.ClassDef))
            else:
                walk(ch, prefix, in_class)

    walk(tree, "", False)
    return out


def _line_scan(source: str, rel: str) -> list[Symbol]:
    """Fallback for files ast cannot parse: indentation-based, blind to strings and non-def blocks."""
    symbols: list[Symbol] = []
    stack: list[tuple[int, str, str]] = []
    for lineno, line in enumerate(source.splitlines(), 1):
        m = DEF_RE.match(line)
        if not m:
            continue
        indent = len(m.group(1).expandtabs(4))
        while stack and stack[-1][0] >= indent:
            stack.pop()
        name = m.group(3)
        kind = "class" if m.group(2) == "class" else ("method" if stack and stack[-1][2] == "class" else "function")
        symbols.append(Symbol(".".join([s[1] for s in stack] + [name]), rel, lineno, kind))
        stack.append((indent, name, kind))
    return symbols


def scan_file(root: str, rel: str) -> list[Symbol]:
    try:
        with open(os.path.join(root, rel), encoding="utf-8", errors="replace") as f:
            source = f.read()
    except OSError:
        return []
    rows = extract_symbols(source)
    if rows is None:
        return _line_scan(source, rel)
    return [Symbol(q, rel, start, kind) for q, kind, _is_async, start, _end in rows]


def build_index(root: str, include_tests: bool = False) -> tuple[list[str], list[Symbol]]:
    files = sorted(p for p in iter_py_files(root) if include_tests or not is_test_path(p))
    symbols = [s for rel in files for s in scan_file(root, rel)]
    return files, symbols
