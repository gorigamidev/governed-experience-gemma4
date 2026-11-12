"""focus: find the functions that mention the issue's terms and return their source, in one call.

Usage: run_skill_script(skill_name="focus", file_path="scripts/focus.py", args={"terms": "name1, name2, error text"})
Searches every non-test .py file under /workspace for each term (literal match), ranks files by distinct terms
and hits, and returns the verbatim source of the enclosing functions in the best two files. Always prints JSON.
"""

import argparse
import ast
import json
import os
import re
import sys

WORKSPACE = os.environ.get("FOCUS_WORKSPACE", "/workspace")
SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", "build", "dist", ".tox", "tests", "test"}
MAX_TERMS, TOP_FILES, MAX_LINES, MAX_REGION, WINDOW = 8, 2, 160, 80, 20


def is_test(rel: str) -> bool:
    name = rel.rsplit("/", 1)[-1]
    return name.startswith("test_") or name.endswith("_test.py") or name == "conftest.py"


def split_terms(raw: str) -> list[str]:
    raw = raw.strip()
    parts = re.split(r"[,;\n]+", raw) if re.search(r"[,;\n]", raw) else raw.split()
    out = []
    for p in parts:
        t = p.strip().strip("`'\"()[]{}").removeprefix("/workspace/")
        if len(t) >= 3 and t not in out:
            out.append(t)
    return out[:MAX_TERMS]


def py_files(root: str):
    for d, dirs, files in os.walk(root):
        dirs[:] = sorted(x for x in dirs if x not in SKIP_DIRS and not x.startswith("."))
        for f in sorted(files):
            if f.endswith(".py"):
                rel = os.path.relpath(os.path.join(d, f), root)
                if not is_test(rel):
                    yield rel


def scan(root: str, terms: list[str]) -> list[dict]:
    ranked = []
    for rel in py_files(root):
        try:
            with open(os.path.join(root, rel), encoding="utf-8", errors="replace") as fh:
                lines = fh.read().split("\n")
        except OSError:
            continue
        hits = {t: [] for t in terms if t == rel or rel.endswith("/" + t)}  # a path ranks its file, no line hits
        for i, line in enumerate(lines, 1):
            for t in terms:
                if t in line:
                    hits.setdefault(t, []).append(i)
        if hits:
            ranked.append({"file": rel, "terms": sorted(hits), "hits": sum(len(v) for v in hits.values()),
                           "lines": lines, "at": sorted({n for v in hits.values() for n in v})})
    ranked.sort(key=lambda r: (-len(r["terms"]), -r["hits"], r["file"]))
    return ranked


def scopes(source: str) -> list[tuple[int, int, str]]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    out = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = min([node.lineno] + [d.lineno for d in node.decorator_list])
            out.append((start, node.end_lineno or node.lineno, node.name))
    return out


def regions(entry: dict) -> list[dict]:
    lines, spans = entry["lines"], scopes("\n".join(entry["lines"]))
    found = []
    for n in entry["at"]:
        inner = [s for s in spans if s[0] <= n <= s[1]]
        if inner:
            start, end, name = min(inner, key=lambda s: s[1] - s[0])
        else:
            start, end, name = max(1, n - 8), min(len(lines), n + 8), "<module>"
        if end - start + 1 > MAX_REGION:
            start, end = max(start, n - WINDOW), min(end, n + WINDOW)
        found.append([start, end, name])
    found.sort()
    merged = []
    for r in found:
        if merged and r[0] <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], r[1])
        else:
            merged.append(r)
    return [{"file": entry["file"], "start": s, "end": e, "symbol": name,
             "code": "\n".join(lines[s - 1:e])} for s, e, name in merged]


def focus(raw: str, root: str = WORKSPACE) -> dict:
    terms = split_terms(raw or "")
    if not terms:
        return {"ok": False, "error": "no usable terms", "usage": 'args={"terms": "identifier, error text, path"}'}
    ranked = scan(root, terms)
    out, budget = [], MAX_LINES
    for entry in ranked[:TOP_FILES]:
        for reg in regions(entry):
            size = reg["end"] - reg["start"] + 1
            if size > budget:
                continue
            out.append(reg)
            budget -= size
    return {"ok": True, "terms": terms, "regions": out,
            "more_files": [{"file": r["file"], "terms": r["terms"], "hits": r["hits"]} for r in ranked[TOP_FILES:10]],
            "note": "code is verbatim from the file: copy old_string from it for edit_file; start/end are line numbers"}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(add_help=False, exit_on_error=False)
    ap.add_argument("positional", nargs="*")
    ap.add_argument("--terms", default="")
    ap.add_argument("--text", default="")
    try:
        args, _ = ap.parse_known_args(argv)
        result = focus(args.terms or args.text or " ".join(args.positional))
    except Exception as exc:  # the agent must always get JSON back
        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(result))


if __name__ == "__main__":
    main(sys.argv[1:])
