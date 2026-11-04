"""Habit G1 (`check`, b2-check variant) against a throwaway git repo, through the ADK run_skill_script simulator.

No pytest is installed here or in CI, so a fake `pytest` module on PYTHONPATH stands in for it: it reports one
test that always fails (pre-existing) and one that fails only when pkg/mod.py contains "BUG".
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lab/src"))

from lab.export.variants import build_variant  # noqa: E402
from lab.tools.run_skill import run  # noqa: E402

FAKE_PYTEST = """import sys
src = open("pkg/mod.py").read()
print("FAILED tests/test_mod.py::test_old - pre-existing")
if "BUG" in src:
    print("FAILED tests/test_mod.py::test_new - assert 1 == 2")
sys.exit(1)
"""


class CheckTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.build = tempfile.TemporaryDirectory()
        cls.skill = build_variant("b2-check", ROOT / "submission", Path(cls.build.name)) / "skills/mini-palace"

    @classmethod
    def tearDownClass(cls):
        cls.build.cleanup()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws = Path(self.tmp.name, "ws")
        (self.ws / "pkg").mkdir(parents=True)
        (self.ws / "tests").mkdir()
        (self.ws / "pkg/__init__.py").write_text("from .mod import helper\n")
        (self.ws / "pkg/mod.py").write_text("def helper():\n    return 1\n")
        (self.ws / "tests/test_mod.py").write_text("def test_old():\n    pass\n")
        git = ["git", "-C", str(self.ws)]
        subprocess.run(["git", "init", "-q", str(self.ws)], check=True)
        subprocess.run([*git, "add", "."], check=True)
        subprocess.run([*git, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "base"], check=True)
        fake = Path(self.tmp.name, "fake/pytest")
        fake.mkdir(parents=True)
        (fake / "__init__.py").write_text("")
        (fake / "__main__.py").write_text(FAKE_PYTEST)
        self.env = {"PYTHONPATH": str(fake.parent)}
        self.db = str(Path(self.tmp.name, "mp.db"))

    def tearDown(self):
        self.tmp.cleanup()

    def check(self):
        old = {k: os.environ.get(k) for k in self.env}
        os.environ.update(self.env)
        try:
            res = run(self.ws, "entrypoint.py", {"action": "check"}, skill_dir=self.skill, db=self.db)
        finally:
            for k, v in old.items():
                os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        self.assertEqual(res["exit_code"], 0, res["stderr"])
        return json.loads(res["stdout"])

    def write(self, rel, text):
        (self.ws / rel).write_text(text)

    def test_no_changes(self):
        out = self.check()
        self.assertEqual(out["verdict"], "no_changes")
        self.assertIn("not fixed", out["advice"])
        self.assertIn("Do not submit an empty patch", out["advice"])

    def test_syntax_error_is_broken(self):
        self.write("pkg/mod.py", "def helper():\n    return 1\\n    x = 2\n")
        out = self.check()
        self.assertEqual(out["verdict"], "broken")
        self.assertIn("SyntaxError", out["items"][0]["error"])

    def test_rename_without_updating_uses_is_broken(self):
        self.write("pkg/mod.py", "def helper_renamed():\n    return 1\n")
        out = self.check()
        self.assertEqual(out["verdict"], "broken")
        self.assertEqual(out["items"][0]["import"], "pkg")
        self.assertIn("ImportError", out["items"][0]["error"])

    def test_new_failure_is_a_regression_but_old_failure_is_not(self):
        self.write("pkg/mod.py", "def helper():\n    return 2  # BUG\n")
        out = self.check()
        self.assertEqual(out["verdict"], "regressions")
        self.assertEqual([i["test"] for i in out["items"]], ["tests/test_mod.py::test_new"])
        self.assertEqual(out["tests_run"], ["tests/test_mod.py"])

    def test_clean_change_is_ok_and_workspace_untouched(self):
        self.write("pkg/mod.py", "def helper():\n    return 3\n")
        self.assertEqual(self.check()["verdict"], "ok")
        status = subprocess.run(["git", "-C", str(self.ws), "status", "--porcelain"], capture_output=True,
                                text=True).stdout.split("\n")
        self.assertEqual([ln for ln in status if ln.strip()], [" M pkg/mod.py"])

    def test_scratch_files_and_test_edits_are_hygiene_and_never_run(self):
        self.write("pkg/mod.py", "def helper():\n    return 3\n")
        self.write("repro.py", "raise SystemExit('must never be imported')\n")
        self.write("tests/test_mod.py", "def test_old():\n    assert True\n")
        out = self.check()
        self.assertEqual(out["verdict"], "hygiene")
        self.assertIn({"stray_files": ["repro.py"]}, out["items"])
        self.assertIn({"tests_touched": ["tests/test_mod.py"]}, out["items"])

    def test_verdict_goes_to_the_ledger(self):
        self.write("pkg/mod.py", "def helper():\n    return 3\n")
        self.check()
        res = run(self.ws, "entrypoint.py", {"action": "ledger"}, skill_dir=self.skill, db=self.db)
        notes = [i for i in json.loads(res["stdout"])["items"] if i["subject"] == "check"]
        self.assertEqual(notes[0]["status"], "ok")


if __name__ == "__main__":
    unittest.main()
