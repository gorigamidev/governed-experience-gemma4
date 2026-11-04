import unittest

from lab.analysis.edit_timing import bucket, command_kind, ending, pre_edit


def _row(first=None, refused=0, error="", calls=10):
    return {"first_edit_call": first, "edits_after_budget": refused, "error": error, "tool_calls": calls}


def _call(name, **args):
    return {"tool_calls": [{"function_name": name, "arguments": args}]}


class EditTimingTest(unittest.TestCase):
    def test_buckets(self):
        self.assertEqual(bucket(_row(12), 28), "early")
        self.assertEqual(bucket(_row(13), 28), "mid")
        self.assertEqual(bucket(_row(24), 28), "mid")
        self.assertEqual(bucket(_row(25), 28), "late")
        self.assertEqual(bucket(_row(33), 36), "late")
        self.assertEqual(bucket(_row(None, refused=1), 28), "refused_only")
        self.assertEqual(bucket(_row(None), 28), "never")

    def test_endings(self):
        self.assertEqual(ending(_row(error="Agent exceeded session timeout (8.0 min)", calls=28), 28), "time")
        self.assertEqual(ending(_row(calls=28), 28), "calls")
        self.assertEqual(ending(_row(calls=12), 28), "submit")
        self.assertEqual(ending(_row(error="boom"), 28), "error")

    def test_pre_edit_skips_submit_and_stops_at_first_edit(self):
        trace = {"steps": [_call("run_command", command="grep -rn foo /workspace"),
                           _call("read_file", path="/workspace/src/a.py"),
                           _call("submit_patch"),
                           _call("run_command", command="python -c 'import a'"),
                           _call("edit_file", path="/workspace/src/a.py"),
                           _call("run_command", command="pytest tests/test_a.py")]}
        out = pre_edit(trace, 4, {"src/a.py"})
        self.assertEqual(out["gold_read"], 2)
        self.assertEqual(dict(out["kinds"]), {"shell_read": 1, "read_file": 1, "python_inline": 1})
        self.assertEqual(command_kind("pytest -x tests"), "pytest")


if __name__ == "__main__":
    unittest.main()
