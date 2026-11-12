"""look: read a file range, list a directory, or search the repo. Code comes back verbatim (no line-number prefix
in `content`), so `old_string` for edit_file can be copied from it.

  args={"path": "pkg/mod.py", "start": 120, "end": 180}   read lines 120-180 (max 150 lines per call)
  args={"path": "pkg"}                                     list a directory
  args={"search": "text or regex", "path": "pkg"}          git grep -n (path optional; default *.py, then all files)
Always prints JSON. Counts toward the read lock (scripts/lock.py).
"""

import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # runpy.run_path does not add the script dir

import lock  # noqa: E402

MAX_LINES, MAX_HITS, MAX_ENTRIES = 150, 40, 200


def rel(root: str, path: str) -> str:
    path = (path or "").strip().strip("'\"")
    for prefix in (lock.SCORER_WORKSPACE + "/", root.rstrip("/") + "/"):
        if path.startswith(prefix):
            path = path[len(prefix):]
    if path in (lock.SCORER_WORKSPACE, root, "."):
        path = ""
    while path.startswith("./"):
        path = path[2:]
    return path


def inside(root: str, path: str) -> str | None:
    full = os.path.realpath(os.path.join(root, path))
    base = os.path.realpath(root)
    return full if full == base or full.startswith(base + os.sep) else None


def top(root: str) -> list[str]:
    return sorted(e + ("/" if os.path.isdir(os.path.join(root, e)) else "") for e in os.listdir(root)
                  if e not in (".git", "__pycache__"))[:MAX_ENTRIES]


def to_int(v):
    """ADK sends str(value): 120, "120", "120.0"; None/null/empty mean unset."""
    if v is None or str(v).strip().lower() in ("", "none", "null"):
        return None
    return int(float(str(v).strip()))


def read(root: str, st: dict, path: str, start, end, only_seen: bool = False) -> dict | None:
    full = inside(root, path)
    if full is None:
        return {"ok": False, "error": f"path outside the repository: {path}"}
    if os.path.isdir(full):
        entries = sorted(e + ("/" if os.path.isdir(os.path.join(full, e)) else "") for e in os.listdir(full)
                         if e not in (".git", "__pycache__"))
        return {"ok": True, "dir": path or ".", "entries": entries[:MAX_ENTRIES], "truncated": len(entries) > MAX_ENTRIES}
    if not os.path.isfile(full):
        return {"ok": False, "error": f"no such file or directory: {path}", "repo_root": top(root)}
    with open(full, encoding="utf-8", errors="replace") as fh:
        lines = fh.read().split("\n")
    total = len(lines)
    start, end = to_int(start), to_int(end)
    s = max(1, start) if start is not None else 1
    e = min(total, end) if end is not None else total
    if e < s:
        return {"ok": False, "error": f"empty range {s}-{e} (file has {total} lines)"}
    truncated = e - s + 1 > MAX_LINES
    if truncated:
        e = s + MAX_LINES - 1
    if only_seen and not lock.was_seen(st, path, s, e):
        return None
    code = "\n".join(lines[s - 1:e])
    lock.remember(st, path, s, e, code)
    lock.mark_seen(st, path, s, e)
    return {"ok": True, "file": path, "start": s, "end": e, "total_lines": total, "truncated": truncated,
            "content": code}


def search(root: str, st: dict, pattern: str, path: str) -> dict:
    if path and not os.path.exists(os.path.join(root, path)):
        return {"ok": False, "error": f"no such file or directory: {path}", "repo_root": top(root)}
    spec = [path] if path else ["*.py"]
    base = ["git", "-c", "safe.directory=*", "--no-optional-locks", "grep", "-n", "-I", "--no-color"]
    out = subprocess.run(base + ["-E", "-e", pattern, "--"] + spec, cwd=root, capture_output=True, text=True,
                         timeout=60)
    mode = "regex"
    if out.returncode > 1:  # bad regex: fall back to a literal search
        out = subprocess.run(base + ["-F", "-e", pattern, "--"] + spec, cwd=root, capture_output=True, text=True,
                             timeout=60)
        mode = "literal"
    if out.returncode > 1:
        return {"ok": False, "error": (out.stderr or "git grep failed").strip()[:500]}
    hits = [h for h in out.stdout.split("\n") if h]
    scope = path or "*.py"
    if not hits and not path:  # e.g. a version string that only lives in pyproject.toml: search every tracked file
        out = subprocess.run(base + (["-F"] if mode == "literal" else ["-E"]) + ["-e", pattern], cwd=root,
                             capture_output=True, text=True, timeout=60)
        hits = [h for h in out.stdout.split("\n") if h] if out.returncode <= 1 else []
        scope = "all tracked files (no hits in *.py)"
    if hits:
        m = re.match(r"^(.*?):(\d+):(.*)$", hits[0])
        if m:
            lock.remember(st, m.group(1), int(m.group(2)), int(m.group(2)), m.group(3))
    return {"ok": True, "search": pattern, "mode": mode, "hits": [h[:300] for h in hits[:MAX_HITS]],
            "total_hits": len(hits), "truncated": len(hits) > MAX_HITS, "searched": scope,
            "note": "hits are path:line:text; read the range with look before copying old_string"}


def main(argv=None) -> None:
    root = st = None
    try:
        opts, rest = lock.parse(list(argv or []), ("path", "search", "start", "end", "file", "filepath", "query"))
        path_arg = opts.get("path") or opts.get("file") or opts.get("filepath") or ""
        search_arg = opts.get("search") or opts.get("query") or ""
        root = lock.repo()
        st, refusal = lock.gate(root)
        path = rel(root, path_arg or (rest[0] if rest and not search_arg else ""))
        if refusal is not None:
            result = refusal
            if path and not search_arg and inside(root, path) and os.path.isfile(inside(root, path)):
                again = read(root, st, path, opts.get("start"), opts.get("end"), only_seen=True)
                if again is not None and again.get("ok"):  # lines already read: re-reading is not new code
                    result = {**refusal, **again, "ok": True, "reread": True}
                    result.pop("last_read", None)
                elif st.get("grace", 0) < lock.GRACE:
                    again = read(root, st, path, opts.get("start"), opts.get("end"))
                    if again.get("ok"):  # a new file range: limited, so the edit is not a blind guess
                        st["grace"] = st.get("grace", 0) + 1
                        result = {**refusal, **again, "ok": True, "grace": True,
                                  "grace_reads_left": lock.GRACE - st["grace"]}
                        result.pop("last_read", None)
        else:
            if search_arg:
                result = search(root, st, search_arg, path)
            elif path or path_arg:
                result = read(root, st, path, opts.get("start"), opts.get("end"))
            else:
                result = {"ok": False, "error": "give path (to read or list) or search",
                          "usage": 'args={"path": "pkg/mod.py", "start": 1, "end": 80} or args={"search": "name"}'}
            result.update(lock.status(st))
    except Exception as exc:  # the agent must always get JSON back
        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    if root is not None and st is not None:
        lock.save(root, st)
    lock.emit(result)


if __name__ == "__main__":
    main(sys.argv[1:])
