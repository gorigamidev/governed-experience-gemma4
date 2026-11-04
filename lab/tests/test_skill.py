"""End-to-end: run the skill through the ADK run_skill_script simulator against a throwaway git repo."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lab/src"))

from lab.tools.run_skill import run  # noqa: E402

MARKUP = "def escape(markup):\n    def escape_backslashes(m):\n        return m\n    return markup\n"
TABLE = "class Table:\n    def add_row(self):\n        pass\n"


class SkillTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws = Path(self.tmp.name, "ws")
        (self.ws / "pkg").mkdir(parents=True)
        (self.ws / "pkg/markup.py").write_text(MARKUP)
        (self.ws / "pkg/table.py").write_text(TABLE)
        subprocess.run(["git", "init", "-q", str(self.ws)], check=True)
        self.db = str(Path(self.tmp.name, "mp.db"))

    def tearDown(self):
        self.tmp.cleanup()

    def call(self, args):
        res = run(self.ws, "entrypoint.py", args, db=self.db)
        self.assertEqual(res["exit_code"], 0, res["stderr"])
        return json.loads(res["stdout"])

    def test_locate_outline_ledger(self):
        out = self.call({"action": "locate", "text": "Fix escaping trailing backslash"})
        self.assertTrue(out["ok"])
        self.assertEqual(out["items"][0]["file"], "pkg/markup.py")
        self.assertIn("escape:1", out["items"][0]["symbols"])

        out = self.call({"action": "outline", "file": "pkg/markup.py"})
        self.assertEqual(out["items"], ["escape:1", "escape.escape_backslashes:2"])

        self.call({"action": "ledger", "subject": "pkg/markup.py", "status": "edited", "note": "done"})
        ledger = self.call({"action": "ledger"})["items"]
        self.assertIn({"kind": "file", "subject": "pkg/markup.py", "status": "edited", "note": "done"}, ledger)

    def test_cache_hit_and_invalidation(self):
        args = {"action": "locate", "text": "escape backslash"}
        self.assertNotIn("cached", self.call(args))
        self.assertTrue(self.call(args).get("cached"))
        (self.ws / "pkg/markup.py").write_text(MARKUP + "# edited\n")
        self.assertNotIn("cached", self.call(args))

    def test_cache_hit_on_clean_committed_tree(self):
        # Regression: the first locate marks candidates "open"; that must not change the next call's cache key.
        subprocess.run(["git", "-C", str(self.ws), "add", "."], check=True)
        subprocess.run(["git", "-C", str(self.ws), "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "base"],
                       check=True)
        args = {"action": "locate", "text": "escape backslash"}
        self.assertNotIn("cached", self.call(args))
        self.assertTrue(self.call(args).get("cached"))

    def test_extra_option_does_not_hijack_the_action(self):
        # Seen in the 1 Oct preflight: {"action": "check", "file": ..., "line": 143} became "unknown action: '143'".
        out = self.call({"action": "outline", "file": "pkg/markup.py", "line": 143})
        self.assertTrue(out["ok"], out)

    def test_missing_action_is_inferred_from_the_arguments(self):
        # Seen with Gemma 4 E4B on 2 Oct: {"text": ...} without an action, and {"file": ...} meaning outline.
        located = self.call({"text": "escape backslash"})
        self.assertTrue(located["ok"], located)
        self.assertEqual(located["inferred_action"], "locate")
        outlined = self.call({"file": "pkg/markup.py"})
        self.assertTrue(outlined["ok"], outlined)
        self.assertEqual(outlined["inferred_action"], "outline")

    def test_bad_calls_return_json(self):
        for args in ({}, {"action": "bogus"}, {"action": "locate", "k": "x"}, {"action": "outline", "file": "nope.py"}):
            out = self.call(args)
            self.assertFalse(out["ok"], args)

    def ledger(self):
        return {e["subject"]: e["status"] for e in self.call({"action": "ledger"})["items"] if e["kind"] == "file"}

    def test_discard_moves_file_aside_and_busts_cache(self):
        args = {"action": "locate", "text": "escape backslash table"}
        first = self.call(args)["items"][0]["file"]
        self.call({"action": "ledger", "subject": first, "status": "discarded"})
        again = self.call(args)
        self.assertNotIn("cached", again)
        self.assertNotIn(first, [i["file"] for i in again["items"]])
        self.assertEqual(again["set_aside"], [first])

    def test_implicit_updates_never_downgrade(self):
        self.call({"action": "ledger", "subject": "pkg/markup.py", "status": "discarded"})
        self.call({"action": "outline", "file": "pkg/markup.py"})
        self.call({"action": "locate", "text": "escape"})
        self.assertEqual(self.ledger()["pkg/markup.py"], "discarded")

        self.call({"action": "ledger", "subject": "pkg/table.py", "status": "edited"})
        self.call({"action": "outline", "file": "pkg/table.py"})
        self.assertEqual(self.ledger()["pkg/table.py"], "edited")

    def test_edits_detected_from_git(self):
        subprocess.run(["git", "-C", str(self.ws), "add", "."], check=True)
        subprocess.run(["git", "-C", str(self.ws), "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "base"],
                       check=True)
        (self.ws / "pkg/table.py").write_text(TABLE + "# fix\n")
        (self.ws / "tests").mkdir()
        (self.ws / "tests/test_x.py").write_text("def test_x(): pass\n")
        out = self.call({"action": "locate", "text": "table row"})
        self.assertEqual(self.ledger().get("pkg/table.py"), "edited")
        self.assertNotIn("tests/test_x.py", self.ledger())
        self.assertEqual(out["items"][0].get("state"), "edited")

    def test_workspace_left_clean(self):
        self.call({"action": "locate", "text": "table row"})
        leftovers = [p.name for p in self.ws.iterdir() if p.name.startswith(".adk_exec_")]
        self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main()
