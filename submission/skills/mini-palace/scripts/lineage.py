"""Exact memoization of skill results and a compact ledger that survives context compaction."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS lineage (query_key TEXT PRIMARY KEY, result_json TEXT, ts REAL);
CREATE TABLE IF NOT EXISTS ledger (id INTEGER PRIMARY KEY, kind TEXT, subject TEXT, status TEXT, note TEXT, ts REAL);
"""


def connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=5)
    conn.executescript(SCHEMA)
    return conn


def workspace_changes(workspace: str) -> list[tuple[str, str, int, int]]:
    """Uncommitted changes as (git status code, path, size, mtime_ns).

    Status alone is not enough: a second edit to an already-modified file leaves `git status` unchanged,
    so each changed path carries its size and mtime too.
    """
    try:
        out = subprocess.run(["git", "-C", workspace, "status", "--porcelain", "-uall"], capture_output=True,
                             text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    changes = []
    for line in out.splitlines():
        path = line[3:].split(" -> ")[-1].strip('"')
        # The harness drops a fresh .adk_exec_<id>.py wrapper into the workspace on every invocation.
        if ".adk_exec_" in path:
            continue
        try:
            st = os.stat(os.path.join(workspace, path))
            changes.append((line[:2], path, st.st_size, st.st_mtime_ns))
        except OSError:
            changes.append((line[:2], path, -1, 0))
    return changes


def workspace_state(changes: list[tuple[str, str, int, int]]) -> str:
    return hashlib.sha1(repr(changes).encode()).hexdigest()[:12]


def make_key(action: str, args: dict, state: str) -> str:
    raw = json.dumps({"a": action, "args": args, "s": state}, sort_keys=True)
    return hashlib.sha1(raw.encode()).hexdigest()


def lookup(conn: sqlite3.Connection, key: str) -> dict | None:
    row = conn.execute("SELECT result_json FROM lineage WHERE query_key = ?", (key,)).fetchone()
    return json.loads(row[0]) if row else None


def record(conn: sqlite3.Connection, key: str, result: dict) -> None:
    conn.execute("INSERT OR REPLACE INTO lineage VALUES (?, ?, ?)", (key, json.dumps(result), time.time()))
    conn.commit()


# Implicit updates (locate, outline, git edits) only move a file up this scale; "discarded" is explicit.
STATUS_RANK = {"open": 0, "checked": 1, "edited": 2}


def ledger_add(conn: sqlite3.Connection, kind: str, subject: str, status: str = "open", note: str = "") -> None:
    """Explicit entry: overwrites status and note of (kind, subject)."""
    cur = conn.execute("UPDATE ledger SET status = ?, note = ?, ts = ? WHERE kind = ? AND subject = ?",
                       (status, note, time.time(), kind, subject))
    if cur.rowcount == 0:
        conn.execute("INSERT INTO ledger (kind, subject, status, note, ts) VALUES (?, ?, ?, ?, ?)",
                     (kind, subject, status, note, time.time()))
    conn.commit()


def ledger_touch(conn: sqlite3.Connection, kind: str, subject: str, status: str) -> None:
    """Implicit entry: never downgrades a status and never revives a discarded subject."""
    row = conn.execute("SELECT status FROM ledger WHERE kind = ? AND subject = ?", (kind, subject)).fetchone()
    if row is None:
        conn.execute("INSERT INTO ledger (kind, subject, status, note, ts) VALUES (?, ?, ?, '', ?)",
                     (kind, subject, status, time.time()))
    elif row[0] in STATUS_RANK and STATUS_RANK.get(status, -1) > STATUS_RANK[row[0]]:
        conn.execute("UPDATE ledger SET status = ?, ts = ? WHERE kind = ? AND subject = ?",
                     (status, time.time(), kind, subject))
    conn.commit()


def file_states(conn: sqlite3.Connection) -> dict[str, str]:
    return dict(conn.execute("SELECT subject, status FROM ledger WHERE kind = 'file'").fetchall())


def ledger_view(conn: sqlite3.Connection, limit: int = 30) -> list[dict]:
    rows = conn.execute("SELECT kind, subject, status, note FROM ledger ORDER BY ts DESC LIMIT ?", (limit,))
    return [{"kind": k, "subject": s, "status": st, **({"note": n} if n else {})} for k, s, st, n in rows]
