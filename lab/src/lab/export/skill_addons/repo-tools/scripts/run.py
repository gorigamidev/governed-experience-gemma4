"""run: one shell command in the repository root, stopped after 90 s. For targeted tests and `python -c`.

  args={"cmd": "python -m pytest tests/test_x.py -k name -q"}
Always prints JSON. Counts toward the read lock (scripts/lock.py) until the first edit.
"""

import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # runpy.run_path does not add the script dir

import lock  # noqa: E402

TIMEOUT, MAX_CHARS = 90, 8000


def clip(text: str) -> str:
    if len(text) <= MAX_CHARS:
        return text
    half = MAX_CHARS // 2
    return text[:half] + f"\n... [{len(text) - MAX_CHARS} chars cut] ...\n" + text[-half:]


def execute(root: str, cmd: str) -> dict:
    if root != lock.SCORER_WORKSPACE:  # subprocess sandbox: translate like the harness does for run_command
        tmp = os.environ.get("TMPDIR")
        if tmp:
            cmd = re.sub(r"(?<![\w.-])/tmp(?![\w.-])", lambda m: tmp, cmd)
        cmd = re.sub(r"(?<![\w.-])/workspace(?![\w.-])", lambda m: root, cmd)
    try:
        out = subprocess.run(["bash", "-c", cmd], cwd=root, capture_output=True, text=True, timeout=TIMEOUT,
                             stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired as exc:
        def txt(b):
            return b.decode("utf-8", "replace") if isinstance(b, bytes) else (b or "")
        return {"ok": False, "exit_code": None, "timed_out": True, "stdout": clip(txt(exc.stdout)),
                "stderr": clip(txt(exc.stderr)), "note": f"stopped after {TIMEOUT} s"}
    return {"ok": out.returncode == 0, "exit_code": out.returncode, "stdout": clip(out.stdout),
            "stderr": clip(out.stderr)}


def main(argv=None) -> None:
    root = st = None
    try:
        opts, rest = lock.parse(list(argv or []), ("cmd", "command"))
        cmd = (opts.get("cmd") or opts.get("command") or " ".join(rest)).strip()
        root = lock.repo()
        st, refusal = lock.gate(root)
        if refusal is not None:
            result = refusal
        elif not cmd:
            result = {"ok": False, "error": "no command", "usage": 'args={"cmd": "python -m pytest tests/test_x.py -q"}'}
        else:
            result = execute(root, cmd)
            result.update(lock.status(st))
    except Exception as exc:  # the agent must always get JSON back
        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    if root is not None and st is not None:
        lock.save(root, st)
    lock.emit(result)


if __name__ == "__main__":
    main(sys.argv[1:])
