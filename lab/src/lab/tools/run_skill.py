"""Run a skill script the way ADK's run_skill_script does inside the sandbox.

Mirrors google/adk/tools/skill_toolset.py (_build_wrapper_code, v1.36.1): every file of the skill is
embedded in a wrapper, extracted to a temp dir, cwd switched there and the script run with runpy.
The wrapper itself is written to the workspace and executed with python3, as AdkSandboxCodeExecutor does.

Usage:
  PYTHONPATH=lab/src python3 -m lab.tools.run_skill --workspace data/repos/rich_4079 \
      --script entrypoint.py --args '{"action": "locate", "text": "Fix escaping trailing backslash"}'
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SKILL = ROOT / "submission/skills/mini-palace"


def load_skill_files(skill_dir: Path) -> dict[str, str]:
    files = {}
    for sub in ("references", "assets", "scripts"):
        base = skill_dir / sub
        if base.is_dir():
            for p in sorted(base.rglob("*")):
                if p.is_file() and "__pycache__" not in p.parts:
                    files[f"{sub}/{p.relative_to(base).as_posix()}"] = p.read_text(encoding="utf-8")
    return files


def build_wrapper(files: dict[str, str], script: str, args: dict | list | None) -> str:
    if not script.startswith("scripts/"):
        script = f"scripts/{script}"
    argv = [script]
    if isinstance(args, list):
        argv.extend(str(v) for v in args)
    elif isinstance(args, dict):
        for k, v in args.items():
            argv.extend([f"--{k}", str(v)])
    return "\n".join([
        "import os", "import tempfile", "import sys", "import json as _json", "import subprocess", "import runpy",
        f"_files = {files!r}",
        "def _materialize_and_run():",
        "  _orig_cwd = os.getcwd()",
        "  with tempfile.TemporaryDirectory() as td:",
        "    for rel_path, content in _files.items():",
        "      full_path = os.path.join(td, rel_path)",
        "      os.makedirs(os.path.dirname(full_path), exist_ok=True)",
        "      mode = 'wb' if isinstance(content, bytes) else 'w'",
        "      with open(full_path, mode) as f:",
        "        f.write(content)",
        "    os.chdir(td)",
        "    try:",
        f"      sys.argv = {argv!r}",
        "      try:",
        f"        runpy.run_path({script!r}, run_name='__main__')",
        "      except SystemExit as e:",
        "        if e.code is not None and e.code != 0:",
        "          raise e",
        "    finally:",
        "      os.chdir(_orig_cwd)",
        "_materialize_and_run()",
    ])


def run(workspace: Path, script: str, args, skill_dir: Path = DEFAULT_SKILL, db: str | None = None,
        timeout: int = 300) -> dict:
    code = build_wrapper(load_skill_files(skill_dir), script, args)
    workspace = workspace.resolve()
    target = workspace / f".adk_exec_{uuid.uuid4().hex[:8]}.py"
    target.write_text(code)
    env = {**os.environ, "MINI_PALACE_WORKSPACE": str(workspace.resolve())}
    if db:
        env["MINI_PALACE_DB"] = db
    try:
        res = subprocess.run(["python3", str(target)], cwd=workspace, capture_output=True, text=True,
                             timeout=timeout, env=env)
    finally:
        target.unlink(missing_ok=True)
    return {"stdout": res.stdout, "stderr": res.stderr, "exit_code": res.returncode,
            "payload_bytes": len(code.encode())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True, type=Path)
    ap.add_argument("--script", default="entrypoint.py")
    ap.add_argument("--args", default="{}")
    ap.add_argument("--db", default=None)
    a = ap.parse_args()
    print(json.dumps(run(a.workspace, a.script, json.loads(a.args), db=a.db), indent=1))


if __name__ == "__main__":
    main()
