"""Shared state of the `code` skill: where the repo is, how many reads happened before the first edit, and the lock.

Doc 03 §14 item 54. `look` and `run` count their calls while no tracked file is modified. After LIMIT calls with no
edit, they stop returning code until a tracked file changes; the first change opens the lock for good.
Fails open: if the edit check cannot run, nothing is locked.
"""

import hashlib
import json
import os
import subprocess

LIMIT = 10
GRACE = 3  # new line ranges of any file while locked (no searches, no listings), so the edit is not a blind guess
SCORER_WORKSPACE = "/workspace"
LAST_LINES = 60


def repo() -> str:
    """Docker sandbox: /workspace. swegemma's subprocess sandbox: the task dir named `workspace` on PYTHONPATH."""
    if os.environ.get("CODE_WORKSPACE"):
        return os.environ["CODE_WORKSPACE"]
    for p in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        if p and os.path.basename(os.path.normpath(p)) == "workspace" and os.path.isdir(p):
            return p
    return SCORER_WORKSPACE


def state_path(root: str) -> str:
    base = os.environ.get("CODE_STATE_DIR") or os.environ.get("TMPDIR") or "/tmp"
    if not os.path.isdir(base):
        base = "/tmp"
    key = hashlib.sha1(os.path.realpath(root).encode()).hexdigest()[:12]
    return os.path.join(base, f"code-skill-{key}.json")


def load(root: str) -> dict:
    try:
        with open(state_path(root), encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            return data
    except (OSError, ValueError):
        pass
    return {"calls": 0, "open": False, "refused": 0, "last": None}


def save(root: str, st: dict) -> None:
    path = state_path(root)
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(st, fh)
        os.replace(tmp, path)
    except OSError:
        pass


def diff_hash(root: str):
    """Hash of the tracked-file diff against HEAD, or None if git cannot tell (then the lock stays off)."""
    try:
        out = subprocess.run(["git", "-c", "safe.directory=*", "--no-optional-locks", "diff", "HEAD", "--no-color",
                              "--no-ext-diff", "--binary"], cwd=root, capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return hashlib.sha1(out.stdout).hexdigest()


def gate(root: str) -> tuple[dict, dict | None]:
    """Count this call and decide. Returns (state, refusal payload or None).

    The first call stores the diff as found (a sandbox copy may already differ from HEAD in modes or line ends);
    any later change to tracked files opens the lock for good. Untracked files (scratch, caches) never open it."""
    st = load(root)
    if not st.get("open"):
        h = diff_hash(root)
        if h is None:
            st["open"], st["why_open"] = True, "git unavailable"
        elif "base" not in st:
            st["base"] = h
        elif h != st["base"]:
            st["open"], st["why_open"] = True, "edited"
    if st.get("open"):
        return st, None
    if st.get("calls", 0) >= LIMIT:
        st["refused"] = st.get("refused", 0) + 1
        left = max(0, GRACE - st.get("grace", 0))
        payload = {"ok": False, "locked": True, "reads_before_edit": st["calls"], "grace_reads_left": left,
                   "message": (f"Locked: you made {st['calls']} look/run calls and have not edited any file. Reading "
                               "more does not help. Call edit_file NOW with your best fix in the code you already "
                               "read (old_string copied exactly from it). look and run work again after your edit. "
                               "Until then look only re-reads lines you already read, plus "
                               f"{left} new line range(s) of a file you name with path, start and end (to copy an "
                               "exact old_string); no searches, no directory listings, no run.")}
        if st.get("last"):
            payload["last_read"] = st["last"]
        return st, payload
    st["calls"] = st.get("calls", 0) + 1
    return st, None


def remember(st: dict, file: str, start: int, end: int, code: str) -> None:
    lines = code.split("\n")
    if len(lines) > LAST_LINES:
        lines = lines[:LAST_LINES]
        end = start + LAST_LINES - 1
    st["last"] = {"file": file, "start": start, "end": end, "code": "\n".join(lines)}


def mark_seen(st: dict, file: str, start: int, end: int) -> None:
    seen = st.setdefault("seen", {}).setdefault(file, [])
    seen.append([start, end])
    seen.sort()
    merged = []
    for a, b in seen:
        if merged and a <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    st["seen"][file] = merged


def was_seen(st: dict, file: str, start: int, end: int) -> bool:
    return any(a <= start and end <= b for a, b in st.get("seen", {}).get(file, []))


def status(st: dict) -> dict:
    if st.get("open"):
        return {"lock": "open"}
    return {"reads_before_edit": st.get("calls", 0), "reads_left_before_lock": max(0, LIMIT - st.get("calls", 0))}


def parse(argv: list[str], keys: tuple[str, ...]) -> tuple[dict, list[str]]:
    """ADK passes a dict as `--key value` pairs with no quoting rules, so a value may itself start with `-`.
    Take the token after a known `--key` literally; everything else is positional."""
    opts, rest, i = {}, [], 0
    while i < len(argv):
        tok = argv[i]
        if tok.startswith("--") and tok[2:] in keys and i + 1 < len(argv):
            opts[tok[2:]] = argv[i + 1]
            i += 2
            continue
        if tok != "--":
            rest.append(tok)
        i += 1
    return opts, rest


def emit(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False))
