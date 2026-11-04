"""Habit G1, verify before submit (doc 03, 14.16): does the agent's change still compile, import and pass the
module's existing tests? Never edits /workspace; the HEAD copy for the baseline lives next to the ledger db.

Verdicts, most severe first: broken (compile or import fails), regressions (existing tests of the changed
modules that pass on HEAD fail now), hygiene (test files touched, stray files), ok, no_changes.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys

from indexer import is_test_path
from util import DB_PATH, WORKSPACE

TEST_DIRS = ("tests", "test")
ROOT_CONFIG = ("pyproject.toml", "setup.cfg", "setup.py", "pytest.ini", "tox.ini", "conftest.py")
MAX_TEST_FILES = 3
PYTEST_TIMEOUT = 90
FAILED = re.compile(r"^(FAILED|ERROR) (\S+)")


def _env(root: str) -> dict:
    paths = [root] + ([os.path.join(root, "src")] if os.path.isdir(os.path.join(root, "src")) else [])
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(paths + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _run(cmd: list[str], cwd: str, timeout: int) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(cmd, cwd=cwd, env=_env(cwd), capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None


def _short(text: str, limit: int = 240) -> str:
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    where = [ln.strip() for ln in lines if ln.strip().startswith("File ")][-1:]
    return " | ".join(where + lines[-1:])[:limit]


def module_of(path: str) -> str:
    """Top-level importable name of a repo-relative .py path (src/ layouts included)."""
    parts = path.removeprefix("src/").split("/")
    return parts[0][:-3] if len(parts) == 1 else parts[0]


def test_files_for(root: str, path: str) -> list[str]:
    stem = os.path.basename(path)[:-3]
    if stem == "__init__":
        stem = os.path.basename(os.path.dirname(path)) or stem
    found = []
    for top in TEST_DIRS:
        for dirpath, _dirs, files in os.walk(os.path.join(root, top)):
            if f"test_{stem}.py" in files:
                found.append(os.path.relpath(os.path.join(dirpath, f"test_{stem}.py"), root).replace(os.sep, "/"))
    return sorted(found)[:MAX_TEST_FILES]


def failing_tests(root: str, files: list[str]) -> tuple[set[str], int, str]:
    """(failing node ids, exit code, short output); exit code -1 when pytest could not run."""
    res = _run([sys.executable, "-m", "pytest", "-q", "-rfE", "-p", "no:cacheprovider", *files], root, PYTEST_TIMEOUT)
    if res is None:
        return set(), -1, "pytest timed out or could not start"
    out = res.stdout + res.stderr
    return {m.group(2) for ln in out.splitlines() if (m := FAILED.match(ln))}, res.returncode, _short(out)


def head_copy(paths: list[str]) -> str | None:
    """Extract HEAD's version of `paths` (and root config) once per HEAD, outside the workspace."""
    head = _run(["git", "rev-parse", "HEAD"], WORKSPACE, 15)
    if head is None or head.returncode != 0:
        return None
    dest = os.path.join(os.path.dirname(DB_PATH), f"mini-palace-head-{head.stdout.strip()[:12]}")
    if os.path.isdir(dest):
        return dest
    listed = _run(["git", "ls-tree", "--name-only", "HEAD"], WORKSPACE, 15)
    top = set(listed.stdout.split()) if listed and listed.returncode == 0 else set()
    wanted = sorted({p for p in paths if p in top} | {c for c in ROOT_CONFIG if c in top})
    tmp = dest + ".tmp"
    os.makedirs(tmp, exist_ok=True)
    try:
        archive = subprocess.run(["git", "-C", WORKSPACE, "archive", "HEAD", "--", *wanted], capture_output=True,
                                 timeout=60, check=True).stdout
        subprocess.run(["tar", "-x", "-C", tmp], input=archive, capture_output=True, timeout=60, check=True)
        os.rename(tmp, dest)
    except (OSError, subprocess.SubprocessError):
        return None
    return dest


NO_CHANGES_ADVICE = ("Nothing to verify: /workspace has no edits, so the issue is not fixed yet. Do not submit an "
                     "empty patch. Find the code that causes the issue, edit it, then call check again.")


def action_check(_args, conn, changes=None) -> dict:
    import lineage

    changes = changes if changes is not None else lineage.workspace_changes(WORKSPACE)
    changed = [(code, p) for code, p, _size, _mtime in changes]
    # New files at the repo root are scratch work (repro.py...): reported, never imported or executed.
    stray = [p for code, p in changed if code == "??" and "/" not in p and not is_test_path(p)]
    sources = [p for code, p in changed
               if p.endswith(".py") and not is_test_path(p) and code.strip() != "D" and p not in stray]
    tests_touched = [p for _code, p in changed if is_test_path(p)]
    modules = sorted({module_of(p) for p in sources})
    if not changed:
        # An empty check read as a green light: the coder submitted an empty patch (top-check preflight, rich_4077).
        return {"ok": True, "verdict": "no_changes", "advice": NO_CHANGES_ADVICE, "items": []}

    broken = []
    for p in sources:
        try:
            with open(os.path.join(WORKSPACE, p), encoding="utf-8") as f:
                compile(f.read(), p, "exec")
        except SyntaxError as exc:
            broken.append({"file": p, "error": f"SyntaxError line {exc.lineno}: {exc.msg}"})
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            broken.append({"file": p, "error": f"{type(exc).__name__}: {exc}"[:200]})
    if not broken:
        for mod in modules:
            res = _run([sys.executable, "-c", f"import {mod}"], WORKSPACE, 60)
            if res is None or res.returncode != 0:
                broken.append({"import": mod, "error": _short(res.stderr) if res else "import timed out"})

    regressions, tests_run = [], []
    if not broken:
        test_files = sorted({t for p in sources for t in test_files_for(WORKSPACE, p)})
        if test_files:
            now, _code, detail = failing_tests(WORKSPACE, test_files)
            base_root = head_copy(sorted({p.split("/")[0] for p in sources + test_files})) if now else None
            before = failing_tests(base_root, test_files)[0] if base_root else set()
            tests_run = test_files
            regressions = [{"test": t} for t in sorted(now - before)]
            if regressions and len(regressions) < 3:
                regressions[0]["detail"] = detail

    hygiene = ([{"tests_touched": tests_touched}] if tests_touched else []) + ([{"stray_files": stray}] if stray else [])
    verdict = "broken" if broken else "regressions" if regressions else "hygiene" if hygiene else "ok"
    advice = {"broken": "Your change does not compile or import. Fix the error below, then call check again.",
              # Informational by the admission rule (doc 03, 14.16): 6 of 25 resolved patches had such tests.
              "regressions": "These existing tests passed before your change and fail now. Expected if they assert "
                             "the old behavior that the issue asks to change; otherwise review your change.",
              "hygiene": "Do not modify tests; remove scratch files you created in the repository.",
              "ok": "No breakage found. Submit with submit_patch."}[verdict]
    summary = json.dumps({"verdict": verdict, "n": len(broken) + len(regressions)})
    lineage.ledger_add(conn, "note", "check", verdict, summary)
    return {"ok": True, "verdict": verdict, "advice": advice, "changed": sources, "tests_run": tests_run,
            "items": broken + regressions[:8] + hygiene}
