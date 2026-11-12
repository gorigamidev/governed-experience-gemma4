"""The `code` skill (doc 03 §14 item 54), run the way ADK runs it: the skill copied to a temp dir, cwd there,
sys.argv built from the args dict as `--key str(value)`, runpy.run_path. Both sandbox layouts are covered."""

import json
import os
import runpy
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SKILL = Path(__file__).resolve().parents[1] / "src/lab/export/skill_addons/repo-tools"
HAS_GIT = shutil.which("git") is not None

WRAPPER = """
import os, runpy, sys
os.chdir(sys.argv[1])
sys.argv = sys.argv[2:]
runpy.run_path(sys.argv[0], run_name="__main__")
"""


def git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True,
                   env={**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
                        "GIT_COMMITTER_EMAIL": "t@t"})


@unittest.skipUnless(HAS_GIT, "git not available")
class CodeSkillTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.repo = self.tmp / "workspace"
        (self.repo / "pkg").mkdir(parents=True)
        (self.repo / "pkg/mod.py").write_text("def f():\n    return 1\n\n\ndef g(x):\n    return x + 1\n")
        (self.repo / "pkg/other.py").write_text("VALUE = '--timeout'\n")
        git(self.repo, "init", "-q")
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-q", "-m", "base")
        self.state = self.tmp / "tmp"
        self.state.mkdir()
        self.skill_copy = self.tmp / "skillcopy"
        shutil.copytree(SKILL, self.skill_copy)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def call(self, script, args, layout="subprocess", direct=False):
        if not direct:  # the agent calls scripts/entrypoint.py with an action, as in the prompt
            args = {"action": "look" if script == "look.py" else "run", **args}
            script = "entrypoint.py"
        argv = [f"scripts/{script}"]
        for k, v in args.items():
            argv += [f"--{k}", str(v)]
        env = {k: v for k, v in os.environ.items() if k not in ("CODE_WORKSPACE", "CODE_STATE_DIR", "PYTHONPATH")}
        if layout == "subprocess":  # swegemma subprocess sandbox: repo dir on PYTHONPATH, per-task TMPDIR
            env.update(PYTHONPATH=str(self.repo), TMPDIR=str(self.state))
        else:
            env.update(CODE_WORKSPACE=str(self.repo), CODE_STATE_DIR=str(self.state))
        out = subprocess.run([sys.executable, "-c", WRAPPER, str(self.skill_copy), *argv], capture_output=True,
                             text=True, env=env, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr)
        lines = out.stdout.strip().split("\n")
        self.assertEqual(len(lines), 1, out.stdout)
        return json.loads(lines[0])

    def test_read_is_verbatim_and_counts(self):
        r = self.call("look.py", {"path": "/workspace/pkg/mod.py", "start": 1, "end": 2})
        self.assertTrue(r["ok"], r)
        self.assertEqual(r["content"], "def f():\n    return 1")
        self.assertEqual(r["reads_before_edit"], 1)
        self.assertEqual(r["reads_left_before_lock"], 9)

    def test_lock_after_ten_then_edit_opens_for_good(self):
        for i in range(10):
            r = self.call("look.py" if i % 2 else "run.py", {"path": "pkg/mod.py", "start": 5, "end": 6}
                          if i % 2 else {"cmd": "echo hi"})
            self.assertTrue(r["ok"], r)
        r = self.call("look.py", {"search": "VALUE"})  # searches stay refused while locked
        self.assertTrue(r["locked"])
        self.assertFalse(r["ok"])
        self.assertEqual(r["last_read"]["code"], "def g(x):\n    return x + 1")
        self.assertNotIn("hits", r)
        r = self.call("run.py", {"cmd": "cat pkg/other.py"})
        self.assertTrue(r["locked"])
        (self.repo / "pkg/mod.py").write_text("def f():\n    return 2\n\n\ndef g(x):\n    return x + 1\n")
        r = self.call("look.py", {"path": "pkg/other.py"})
        self.assertTrue(r["ok"], r)
        self.assertEqual(r["lock"], "open")
        git(self.repo, "checkout", "--", "pkg/mod.py")  # a revert does not close it again
        for _ in range(12):
            self.assertEqual(self.call("run.py", {"cmd": "true"})["lock"], "open")

    def test_locked_rereads_seen_lines_and_three_grace_ranges_of_opened_files(self):
        (self.repo / "pkg/long.py").write_text("\n".join(f"y{i} = {i}" for i in range(100)) + "\n")
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-q", "-m", "long")
        self.call("look.py", {"path": "pkg/mod.py", "start": 1, "end": 6})
        self.call("look.py", {"path": "pkg/long.py", "start": 1, "end": 5})
        for _ in range(8):
            self.call("run.py", {"cmd": "true"})
        r = self.call("look.py", {"path": "pkg/mod.py", "start": 5, "end": 6})
        self.assertTrue(r["ok"] and r["locked"] and r["reread"], r)
        self.assertEqual(r["content"], "def g(x):\n    return x + 1")
        self.assertNotIn("last_read", r)
        for i, start in enumerate((10, 20, 30)):
            r = self.call("look.py", {"path": "pkg/long.py", "start": start, "end": start + 2})
            self.assertTrue(r["ok"] and r["locked"] and r["grace"], r)
            self.assertEqual(r["grace_reads_left"], 2 - i)
        r = self.call("look.py", {"path": "pkg/long.py", "start": 20, "end": 22})  # now seen: free re-read
        self.assertTrue(r["reread"], r)
        for args in ({"path": "pkg/long.py", "start": 50, "end": 52}, {"path": "pkg/other.py"}, {"search": "def"}):
            r = self.call("look.py", args)
            self.assertTrue(r["locked"] and not r["ok"], (args, r))
            self.assertEqual(r["grace_reads_left"], 0)
            self.assertNotIn("content", r)
            self.assertNotIn("hits", r)

    def test_grace_reads_serve_a_file_never_opened_but_not_dirs_or_searches(self):
        # E4B rich_3006 (preflight 8 Nov): locked before opening the file to fix, then guessed old_string 3 times
        for _ in range(10):
            self.call("look.py", {"search": "nothing-matches-this"})
        r = self.call("look.py", {"path": "pkg/mod.py", "start": 5, "end": 6})
        self.assertTrue(r["ok"] and r["locked"] and r["grace"], r)
        self.assertEqual(r["content"], "def g(x):\n    return x + 1")
        self.assertEqual(r["grace_reads_left"], 2)
        for args in ({"path": "pkg"}, {"search": "def"}):
            r = self.call("look.py", args)
            self.assertFalse(r["ok"], (args, r))
            self.assertEqual(r["grace_reads_left"], 2)
        self.assertTrue(self.call("run.py", {"cmd": "cat pkg/other.py"})["locked"])

    def test_missing_path_lists_the_repo_root(self):
        r = self.call("look.py", {"path": "pkgx"})
        self.assertFalse(r["ok"])
        self.assertIn("pkg/", r["repo_root"])
        r = self.call("look.py", {"search": "def", "path": "pkgx"})
        self.assertIn("pkg/", r["repo_root"])

    def test_untracked_files_never_open_the_lock(self):
        self.call("look.py", {"path": "pkg"})
        (self.repo / "scratch.py").write_text("x = 1\n")
        (self.repo / "pkg/__pycache__").mkdir()
        (self.repo / "pkg/__pycache__/mod.cpython-313.pyc").write_bytes(b"\0")
        (self.repo / ".adk_exec_1.py").write_text("")
        for _ in range(9):
            self.call("run.py", {"cmd": "python -c 'print(1)'"})
        self.assertTrue(self.call("look.py", {"path": "pkg"})["locked"])

    def test_dirty_copy_at_first_call_is_the_baseline(self):
        (self.repo / "pkg/other.py").write_text("VALUE = 'x'\n")  # e.g. a sandbox copy that differs from HEAD
        for _ in range(10):
            self.assertTrue(self.call("look.py", {"path": "pkg"})["ok"])
        self.assertTrue(self.call("look.py", {"path": "pkg"})["locked"])
        (self.repo / "pkg/mod.py").write_text("changed\n")
        self.assertEqual(self.call("look.py", {"path": "pkg"})["lock"], "open")

    def test_no_git_fails_open(self):
        shutil.rmtree(self.repo / ".git")
        for _ in range(14):
            r = self.call("look.py", {"path": "pkg/mod.py"})
            self.assertTrue(r["ok"], r)
            self.assertEqual(r["lock"], "open")

    def test_docker_layout_and_separate_state(self):
        r = self.call("look.py", {"search": "def g"}, layout="docker")
        self.assertEqual(r["hits"], ["pkg/mod.py:5:def g(x):"])
        self.assertEqual(r["reads_before_edit"], 1)

    def test_search_value_starting_with_dashes_and_bad_regex(self):
        r = self.call("look.py", {"search": "--timeout"})
        self.assertTrue(r["ok"], r)
        self.assertEqual(r["total_hits"], 1)
        r = self.call("look.py", {"search": "values[", "path": "pkg"})
        self.assertTrue(r["ok"], r)
        self.assertEqual(r["mode"], "literal")

    def test_errors_are_json_and_still_counted(self):
        r = self.call("look.py", {"path": "pkg/mod.py", "start": "abc"})
        self.assertFalse(r["ok"])
        self.assertIn("ValueError", r["error"])
        r = self.call("look.py", {"path": "../../etc/passwd"})
        self.assertIn("outside", r["error"])
        r = self.call("look.py", {"path": "pkg/mod.py", "start": "None", "end": "2.0"})
        self.assertEqual(r["content"], "def f():\n    return 1")
        self.assertEqual(r["reads_before_edit"], 3)
        self.assertFalse(self.call("look.py", {})["ok"])
        self.assertFalse(self.call("run.py", {})["ok"])

    def test_search_falls_back_to_all_tracked_files(self):
        (self.repo / "pyproject.toml").write_text('python = ">=3.8.0"\n')
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-q", "-m", "toml")
        for args in ({"search": "3.8"}, {"search": "3.8", "path": "/workspace"}, {"query": "3.8", "path": "."}):
            r = self.call("look.py", args)
            self.assertEqual(r["hits"], ['pyproject.toml:1:python = ">=3.8.0"'], (args, r))
            self.assertIn("all tracked files", r["searched"])
        r = self.call("look.py", {"search": "def g"})
        self.assertEqual(r["searched"], "*.py")
        r = self.call("look.py", {"search": "3.8", "path": "pkg"})  # an explicit directory is not widened
        self.assertEqual(r["total_hits"], 0)

    def test_run_translates_workspace_and_reports_exit(self):
        r = self.call("run.py", {"cmd": "cat /workspace/pkg/other.py && ls /tmp >/dev/null && exit 3"})
        self.assertEqual(r["exit_code"], 3)
        self.assertIn("--timeout", r["stdout"])

    def test_entrypoint_actions_inference_and_unknown(self):
        r = self.call("entrypoint.py", {"action": "LOOK", "path": "pkg/mod.py", "start": 1, "end": 2}, direct=True)
        self.assertEqual(r["content"], "def f():\n    return 1")
        r = self.call("entrypoint.py", {"cmd": "echo hi"}, direct=True)  # forgot the action: inferred from cmd
        self.assertEqual(r["stdout"], "hi\n")
        r = self.call("entrypoint.py", {"search": "def g"}, direct=True)
        self.assertEqual(r["hits"], ["pkg/mod.py:5:def g(x):"])
        r = self.call("entrypoint.py", {"action": "dance"}, direct=True)
        self.assertFalse(r["ok"])
        self.assertIn("unknown action", r["error"])
        self.assertEqual(self.call("look.py", {"path": "pkg"}, direct=True)["reads_before_edit"], 4)  # old scripts still work, same counter

    def test_read_caps_at_150_lines(self):
        (self.repo / "big.py").write_text("\n".join(f"x{i} = {i}" for i in range(400)))
        r = self.call("look.py", {"path": "big.py", "start": 10})
        self.assertEqual((r["start"], r["end"], r["truncated"]), (10, 159, True))


if __name__ == "__main__":
    unittest.main()
