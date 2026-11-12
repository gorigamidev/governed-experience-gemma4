"""The `code` skill of `bf-gov-tb-lk` (doc 03 §14 item 54) inside the official sandbox image, run the way the harness
runs a skill script (wrapper written to /workspace/.adk_exec_<id>.py, python3, removed). An agent-like session on a
real snapshot: search, read, run, the lock after 10 calls, re-read while locked, an edit that opens it, and a clean
workspace at the end.

Usage: PYTHONPATH=lab/src python3 -m lab.tools.sandbox_code_test --snapshot data/competition/snapshots/rich_3470.tgz
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from lab.tools.sandbox_skill_test import ROOT, Sandbox, skill_files_from_zip


def session(box: Sandbox, files: dict[str, str], term: str) -> list[tuple[str, bool, str]]:
    checks = []

    def check(name, ok, detail=""):
        checks.append((name, bool(ok), detail))

    def look(args):  # through the one entrypoint the agent calls, as ADK runs it
        return box.run_skill(files, {"action": "look", **args}, "entrypoint.py")

    def run(args):
        return box.run_skill(files, {"action": "run", **args}, "entrypoint.py")

    out, err, dt = look({"search": term})
    check("search returns hits", out and out.get("ok") and out.get("hits"), f"{dt:.2f}s {err[-200:]}")
    hit = out["hits"][0] if out and out.get("hits") else ""
    path, line = hit.split(":")[0], int(hit.split(":")[1]) if hit else 1
    out, _, dt = look({"path": f"/workspace/{path}", "start": max(1, line - 5), "end": line + 5})
    check("read is verbatim", out and out.get("ok") and out.get("content"), f"{dt:.2f}s {path}:{line}")
    snippet = out.get("content", "") if out else ""
    real = box.exec(f"sed -n '{max(1, line - 5)},{line + 5}p' '{path}'", check=False).stdout.rstrip("\n")
    check("read matches the file byte for byte", snippet == real, "")
    out, _, dt = run({"cmd": "python -c 'import sys; print(sys.version_info[:2])'"})
    check("run executes python", out and out.get("exit_code") == 0 and "(3," in out.get("stdout", ""),
          f"{dt:.2f}s {out.get('stdout', '').strip() if out else ''}")
    out, _, dt = run({"cmd": "timeout 60 python -m pytest --co -q tests 2>/dev/null | tail -1"})
    check("run reaches pytest", out and out.get("exit_code") == 0, f"{dt:.2f}s {out.get('stdout', '').strip()[-60:] if out else ''}")
    for _ in range(6):
        out, _, _ = look({"path": path, "start": 1, "end": 3})
    check("count before the lock", out and out.get("reads_before_edit") == 10 and out.get("reads_left_before_lock") == 0,
          str(out.get("reads_before_edit") if out else None))
    out, _, dt = look({"search": term})
    check("11th call is locked, no hits", out and out.get("locked") and "hits" not in out and out.get("last_read"),
          f"{dt:.2f}s")
    out, _, _ = look({"path": path, "start": max(1, line - 5), "end": line + 5})
    check("re-read of seen lines while locked", out and out.get("reread") and out.get("content") == real, "")
    out, _, _ = run({"cmd": f"cat {path}"})
    check("run is locked too", out and out.get("locked") and "stdout" not in out, "")
    box.exec(f"printf '\\n# lock probe edit\\n' >> '{path}'")  # what edit_file does to the tree
    out, _, dt = look({"search": term})
    check("an edit opens the lock", out and out.get("ok") and out.get("lock") == "open", f"{dt:.2f}s")
    box.exec(f"git checkout -q -- '{path}'")
    status = box.exec("git status --porcelain", check=False).stdout.strip()
    check("workspace clean (no wrappers, no state)", status == "", status[:200])
    st = box.exec("ls /tmp/code-skill-*.json", check=False)
    check("state lives in /tmp", st.returncode == 0, st.stdout.strip()[:120])
    bad, _, _ = look({"start": "x"})
    check("bad call still returns JSON", bad is not None and bad.get("ok") is False)
    return checks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True, type=Path)
    ap.add_argument("--zip", default=ROOT / "build/bf-gov-tb-lk/submission.zip", type=Path)
    ap.add_argument("--term", default="def ")
    ap.add_argument("--image", default="swebench-sandbox:latest")
    args = ap.parse_args()
    files = skill_files_from_zip(args.zip, "repo-tools")
    if sorted(files) != ["scripts/entrypoint.py", "scripts/lock.py", "scripts/look.py", "scripts/run.py"]:
        raise SystemExit(f"unexpected skill files in the zip: {sorted(files)}")
    box = Sandbox(args.image)
    try:
        t0 = time.perf_counter()
        box.load_snapshot(args.snapshot)
        print(f"container {box.name}: snapshot {args.snapshot.name} ready in {time.perf_counter() - t0:.1f}s")
        results = session(box, files, args.term)
    finally:
        box.close()
    for name, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name:42} {detail}")
    failed = [n for n, ok, _ in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
