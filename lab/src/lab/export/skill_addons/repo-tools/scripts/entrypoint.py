"""The one script the agent calls: args.action picks look (read, list, search) or run (one shell command).

Doc 03 §14 item 54, v3: the 31B corrupted ~half of its calls to a skill named `code` with scripts look.py/run.py
(a backtick or 「」 where the string should close); with `mini-palace` and scripts/entrypoint.py it corrupted none
in ~660 calls. So this skill copies that shape: one entrypoint, an action argument.

  args={"action": "look", "search": "text"}
  args={"action": "look", "path": "pkg/mod.py", "start": 120, "end": 180}
  args={"action": "look", "path": "pkg"}
  args={"action": "run", "cmd": "python -m pytest tests/test_x.py -q"}
Always prints JSON.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # runpy.run_path does not add the script dir

import lock  # noqa: E402
import look  # noqa: E402
import run  # noqa: E402

LOOK = ("look", "read", "search", "list", "grep", "find", "open")
RUN = ("run", "cmd", "command", "test", "exec", "shell")


def split_action(argv: list[str]) -> tuple[str, list[str]]:
    action, rest, i = "", [], 0
    while i < len(argv):
        if argv[i] == "--action" and i + 1 < len(argv):
            action = argv[i + 1].strip().strip("'\"").lower()
            i += 2
            continue
        rest.append(argv[i])
        i += 1
    return action, rest


def main(argv=None) -> None:
    argv = list(argv or [])
    action, rest = split_action(argv)
    if not action:  # infer it, so a call that only forgot the action still works
        keys = set(a[2:] for a in rest if a.startswith("--"))
        action = "run" if keys & {"cmd", "command"} else ("look" if keys & {"path", "search", "query", "file"} else "")
    if action in LOOK:
        look.main(rest)
    elif action in RUN:
        run.main(rest)
    else:
        lock.emit({"ok": False, "error": f"unknown action {action!r}: use look or run",
                   "usage": ['args={"action": "look", "search": "<text>"}',
                             'args={"action": "look", "path": "<file.py>", "start": 1, "end": 80}',
                             'args={"action": "run", "cmd": "python -m pytest <tests/test_file.py> -q"}']})


if __name__ == "__main__":
    main(sys.argv[1:])
