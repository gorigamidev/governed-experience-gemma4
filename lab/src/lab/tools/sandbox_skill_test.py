"""Run the packaged mini-palace skill inside the official sandbox image, the way the harness does.

Container limits mirror swegemma/sandbox/docker.py (network none, 4 GiB, 2 CPUs, /workspace). The skill is
taken from the built submission.zip, wrapped exactly like ADK's run_skill_script (lab.tools.run_skill), written
to /workspace/.adk_exec_<id>.py and run with python3, then removed, as AdkSandboxCodeExecutor does.

Needs the image: docker build -f data/competition/docker/Dockerfile.sandbox -t swebench-sandbox:local data/competition/docker
Usage: PYTHONPATH=lab/src python3 -m lab.tools.sandbox_skill_test --snapshot data/competition/snapshots/rich_4079.tgz
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
import uuid
import zipfile
from pathlib import Path

from lab.tools.run_skill import build_wrapper

ROOT = Path(__file__).resolve().parents[4]
IMAGE = "swebench-sandbox:local"


def sh(*args: str, check: bool = True, input_text: str | None = None) -> subprocess.CompletedProcess:
    res = subprocess.run(list(args), capture_output=True, text=True, input=input_text)
    if check and res.returncode != 0:
        raise RuntimeError(f"{' '.join(args[:4])}...: {res.stderr.strip()[:300]}")
    return res


def skill_files_from_zip(zip_path: Path) -> dict[str, str]:
    prefix = "skills/mini-palace/"
    files = {}
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.startswith(prefix) and not name.endswith("/") and not name.endswith("SKILL.md"):
                files[name[len(prefix):]] = zf.read(name).decode("utf-8")
    return files


class Sandbox:
    def __init__(self, image: str):
        self.name = f"mp-test-{uuid.uuid4().hex[:8]}"
        sh("docker", "run", "-d", "--name", self.name, "--network", "none", "--memory", "4g", "--cpus", "2",
           "-w", "/workspace", image, "sleep", "infinity")

    def exec(self, cmd: str, check: bool = True) -> subprocess.CompletedProcess:
        return sh("docker", "exec", "-w", "/workspace", self.name, "bash", "-c", cmd, check=check)

    def load_snapshot(self, tgz: Path) -> None:
        sh("docker", "cp", str(tgz), f"{self.name}:/tmp/snapshot.tgz")
        self.exec("tar xzf /tmp/snapshot.tgz -C /workspace && rm /tmp/snapshot.tgz")
        # Same idea as the harness baseline commit: a clean tree the agent's diff is measured against.
        self.exec("git config --global --add safe.directory /workspace && git add -A && "
                  "git -c user.email=agent@eval -c user.name=Agent commit -qm baseline --allow-empty")

    def run_skill(self, files: dict[str, str], args: dict) -> tuple[dict | None, str, float]:
        code = build_wrapper(files, "entrypoint.py", args)
        target = f".adk_exec_{uuid.uuid4().hex[:8]}.py"
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(code)
        sh("docker", "cp", f.name, f"{self.name}:/workspace/{target}")
        Path(f.name).unlink()
        t0 = time.perf_counter()
        res = self.exec(f"python3 /workspace/{target}", check=False)
        elapsed = time.perf_counter() - t0
        self.exec(f"rm -f /workspace/{target}")
        try:
            return json.loads(res.stdout), res.stderr, elapsed
        except ValueError:
            return None, res.stderr or res.stdout, elapsed

    def close(self) -> None:
        sh("docker", "rm", "-f", self.name, check=False)


def scenario(box: Sandbox, files: dict[str, str], text: str) -> list[tuple[str, bool, str]]:
    """A short agent-like session; returns (check, passed, detail)."""
    checks = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, bool(ok), detail))

    out, err, dt = box.run_skill(files, {"action": "locate", "text": text})
    check("locate returns JSON", out is not None and out.get("ok"), f"{dt:.2f}s {err[-200:]}")
    if not out:
        return checks
    first = out["items"][0]["file"] if out.get("items") else None
    check("locate ranks candidates", bool(first), f"top={first} decision={out.get('decision')} habits={len(out.get('habits', []))}")
    check("output under size cap", len(json.dumps(out)) <= 1600, f"{len(json.dumps(out))} chars")

    again, _, dt2 = box.run_skill(files, {"action": "locate", "text": text})
    check("identical call is cached", again and again.get("cached"), f"{dt2:.2f}s")

    outline, _, dt3 = box.run_skill(files, {"action": "outline", "file": first})
    check("outline lists symbols", outline and outline.get("ok") and outline.get("items"), f"{dt3:.2f}s")

    box.run_skill(files, {"action": "ledger", "subject": first, "status": "discarded", "note": "probe"})
    after, _, _ = box.run_skill(files, {"action": "locate", "text": text})
    check("discard sets file aside", after and first not in [i["file"] for i in after.get("items", [])]
          and first in after.get("set_aside", []), "")

    second = after["items"][0]["file"] if after and after.get("items") else None
    if second:
        box.exec(f"printf '\\n# probe edit\\n' >> '{second}'")
        edited, _, _ = box.run_skill(files, {"action": "locate", "text": text})
        states = {i["file"]: i.get("state") for i in edited.get("items", [])} if edited else {}
        check("git edit detected as edited", states.get(second) == "edited", f"{second}: {states.get(second)}")
        box.exec(f"git checkout -q -- '{second}'")

    status = box.exec("git status --porcelain", check=False).stdout.strip()
    check("workspace left clean (no db, no wrappers)", status == "", status[:200])
    db = box.exec("ls -la /tmp/mini-palace.db", check=False)
    check("ledger db lives in /tmp", db.returncode == 0, db.stdout.strip()[:120])

    bad, _, _ = box.run_skill(files, {"action": "bogus"})
    check("bad call still returns JSON", bad is not None and bad.get("ok") is False)
    return checks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True, type=Path)
    ap.add_argument("--zip", default=ROOT / "build/submission.zip", type=Path)
    ap.add_argument("--text", default="Fix escaping trailing backslash in markup")
    ap.add_argument("--image", default=IMAGE)
    args = ap.parse_args()

    files = skill_files_from_zip(args.zip)
    box = Sandbox(args.image)
    try:
        t0 = time.perf_counter()
        box.load_snapshot(args.snapshot)
        print(f"container {box.name}: snapshot {args.snapshot.name} ready in {time.perf_counter() - t0:.1f}s")
        results = scenario(box, files, args.text)
    finally:
        box.close()
    for name, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name:42} {detail}")
    failed = [n for n, ok, _ in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
