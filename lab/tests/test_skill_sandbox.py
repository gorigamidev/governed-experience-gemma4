import importlib.util
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SKILL = ROOT / "submission/skills/mini-palace/scripts/entrypoint.py"


def exec_code(*argv: str) -> str:
    # Mirrors ADK's wrapper: the script runs from a temp dir, never from the workspace.
    return f"""import os, runpy, sys, tempfile
os.chdir(tempfile.mkdtemp())
sys.argv = ["entrypoint.py", *{list(argv)!r}]
runpy.run_path({str(SKILL)!r}, run_name="__main__")
"""


@unittest.skipUnless(importlib.util.find_spec("swegemma"), "swegemma not installed (subprocess sandbox tests)")
class SubprocessSandboxSkillTest(unittest.TestCase):
    """The scorer may run swegemma's subprocess sandbox, where /workspace and /tmp exist only on command lines."""

    def setUp(self):
        from swegemma.sandbox.subprocess import SubprocessManager

        self.mgr = SubprocessManager()
        self.sid = self.mgr.start()
        self.addCleanup(self.mgr.stop, self.sid)
        self.mgr.initialize()
        self.mgr.write_file("pkg/tables.py", "def render_table(rows):\n    return rows\n\n\ndef column_width(c):\n    return 1\n")

    def run_skill(self, *argv: str) -> dict:
        self.mgr.write_file(".adk_exec_test.py", exec_code(*argv))
        return json.loads(self.mgr.exec(self.sid, "python3 /workspace/.adk_exec_test.py").stdout)

    def test_locate_sees_the_task_repo_and_keeps_its_db_per_task(self):
        out = self.run_skill("locate", "--text", "fix render_table column width")
        self.assertEqual(out["items"][0]["file"], "pkg/tables.py")
        self.assertTrue(any(self.mgr.sandboxes[self.sid]["tmp"].glob("mini-palace.db*")))

    def test_outline_accepts_the_agents_workspace_path(self):
        out = self.run_skill("outline", "--file", "/workspace/pkg/tables.py")
        self.assertEqual(out["items"], ["render_table:1", "column_width:5"])


if __name__ == "__main__":
    unittest.main()
