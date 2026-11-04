"""mini-palace skill entry point. One invocation = one harness tool call, so each action does as much
useful work as possible and returns compact JSON.

Actions:
  locate  --text "<title + key lines of the statement>" [--k 5]
  outline --file <repo-relative path>
  ledger  [--subject <file or symbol> --status open|checked|edited|discarded] [--note "<text>"]

Edited files are detected from git automatically; mark wrong candidates with --status discarded.
"""

from __future__ import annotations

import argparse
import os
import sys

# runpy.run_path does not put the script's directory on sys.path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import lineage  # noqa: E402
import patterns  # noqa: E402
import router  # noqa: E402
from indexer import build_index, is_test_path, scan_file  # noqa: E402
from locator import locate_core  # noqa: E402
from resolver import LexicalIndex  # noqa: E402
from triage import profile_statement  # noqa: E402
from util import DB_PATH, REFERENCES_DIR, SCORER_WORKSPACE, WORKSPACE, emit, load_reference  # noqa: E402


def action_locate(args, conn) -> dict:
    profile = profile_statement(args.text)
    idx = LexicalIndex(*build_index(WORKSPACE))
    cfg = {**router.DEFAULTS, **load_reference("router_config.json", {})}
    habits = patterns.load(os.path.join(REFERENCES_DIR, "seed_patterns.txt"))
    state = lineage.file_states(conn)
    decision, shown, n_discarded, _, active = locate_core(idx, args.text, patterns.detect_repo(WORKSPACE), habits,
                                                          cfg, state)
    items = []
    for r in shown[: args.k]:
        lines = {s.qualname: s.lineno for s in idx.symbols if s.file == r.file}
        item = {"file": r.file, "score": round(r.score, 2), "why": r.parts,
                "symbols": [f"{q}:{lines.get(q, '?')}" for q in r.top_symbols]}
        if state.get(r.file) in ("checked", "edited"):
            item["state"] = state[r.file]
        items.append(item)
        lineage.ledger_touch(conn, "file", r.file, "open")
    set_aside = sorted(f for f, st in state.items() if st == "discarded")
    return {"ok": True, "decision": decision, "statement": profile.category, "kind": profile.change_kind,
            "items": items, "discarded": n_discarded, "habits": active[:8],
            **({"set_aside": set_aside[:10]} if set_aside else {}),
            "next": [f"outline --file {items[0]['file']}"] if items else ["ledger"]}


def action_outline(args, conn) -> dict:
    # The agent always sees the repo as /workspace, even when this process finds it elsewhere.
    rel = args.file.removeprefix(WORKSPACE).removeprefix(SCORER_WORKSPACE).lstrip("/")
    if not os.path.isfile(os.path.join(WORKSPACE, rel)):
        return {"ok": False, "error": f"file not found: {rel}"}
    symbols = scan_file(WORKSPACE, rel)
    lineage.ledger_touch(conn, "file", rel, "checked")
    return {"ok": True, "file": rel, "items": [f"{s.qualname}:{s.lineno}" for s in symbols]}


def action_ledger(args, conn) -> dict:
    if args.subject:
        lineage.ledger_add(conn, "file" if args.subject.endswith(".py") else "symbol", args.subject,
                           args.status or "checked", args.note or "")
    elif args.note:
        lineage.ledger_add(conn, "note", args.note[:80], "note", args.note)
    return {"ok": True, "items": lineage.ledger_view(conn)}


ACTIONS = {"locate": action_locate, "outline": action_outline, "ledger": action_ledger}
CACHEABLE = {"locate", "outline"}


def main(argv: list[str] | None = None) -> str:
    # exit_on_error=False plus no `choices`: a bad call must still return JSON, never argparse's SystemExit.
    ap = argparse.ArgumentParser(prog="entrypoint.py", exit_on_error=False, add_help=False)
    ap.add_argument("action", nargs="?")
    ap.add_argument("--action", dest="action_opt")
    ap.add_argument("--text", default="")
    ap.add_argument("--file", default="")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--subject", default="")
    ap.add_argument("--status", default="")
    ap.add_argument("--note", default="")
    try:
        args, _unknown = ap.parse_known_args(argv)
    except argparse.ArgumentError as exc:
        return emit({"ok": False, "error": f"bad arguments: {exc}", "usage": __doc__.split("Actions:")[1].strip()})
    # --action wins: a stray value from an unknown option (e.g. --line 143) would otherwise land in the positional.
    action = args.action_opt or args.action
    inferred = False
    if action not in ACTIONS and (args.text or args.file):
        # Gemma often drops "action" (2 Oct preflight with Gemma 4 E4B): infer it rather than spend a call on usage.
        action, inferred = ("locate" if args.text else "outline"), True
    if action not in ACTIONS:
        return emit({"ok": False, "error": f"unknown action: {action!r}", "usage": __doc__.split("Actions:")[1].strip()})

    conn = lineage.connect(DB_PATH)
    changes = lineage.workspace_changes(WORKSPACE)
    # Edits are read from git, so the ledger stays right even if the agent never reports them.
    for _code, path, _size, _mtime in changes:
        if path.endswith(".py") and not is_test_path(path):
            lineage.ledger_touch(conn, "file", path, "edited")
    key = None
    if action in CACHEABLE:
        # "open" never changes locate's output; keeping it in the key would make the first call's own marks
        # invalidate the cache for the next identical call.
        state = {f: st for f, st in lineage.file_states(conn).items() if st != "open"} if action == "locate" else {}
        key = lineage.make_key(action, {"text": args.text, "file": args.file, "k": args.k, "state": state},
                               lineage.workspace_state(changes))
        cached = lineage.lookup(conn, key)
        if cached is not None:
            return emit({**cached, "cached": True})
    try:
        result = ACTIONS[action](args, conn)
    except Exception as exc:  # the agent must always get JSON back, never a traceback
        return emit({"ok": False, "error": f"{type(exc).__name__}: {exc}"[:300]})
    if key and result.get("ok"):
        lineage.record(conn, key, result)
    if inferred:
        result = {**result, "inferred_action": action}
    return emit(result)


if __name__ == "__main__":
    print(main())
