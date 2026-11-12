import unittest

from lab.analysis.authority_split import compare, first_edit_file, mcnemar


def call(name, args, content='{"status": "ok"}'):
    return {"tool_calls": [{"function_name": name, "arguments": args}], "observation": {"content": content}}


class AuthoritySplitTest(unittest.TestCase):
    def test_first_successful_edit_skips_failed_edits(self):
        t = {"steps": [call("read_file", {"filepath": "a.py"}),
                       call("edit_file", {"filepath": "/workspace/b.py"}, '{"status": "error", "error_type": "FileEditError"}'),
                       call("edit_file", {"filepath": "/workspace/pkg/c.py"})]}
        self.assertEqual(first_edit_file(t), "pkg/c.py")
        self.assertIsNone(first_edit_file({"steps": [call("read_file", {"filepath": "a.py"})]}))

    def test_mcnemar_exact(self):
        self.assertAlmostEqual(mcnemar(7, 3), 0.34375)
        self.assertEqual(mcnemar(0, 0), 1.0)

    def test_compare_runs_and_task_majority(self):
        rows = [{"task": "t1", "model": True, "habits": False}, {"task": "t1", "model": True, "habits": False},
                {"task": "t2", "model": False, "habits": True}, {"task": "t3", "model": True, "habits": True}]
        c = compare(rows, "habits")
        self.assertEqual((c["runs"], c["model"], c["memory"], c["diff_pts"]), (4, 3, 2, -25.0))
        self.assertEqual((c["tasks"], c["model_tasks"], c["memory_tasks"], c["model_only"], c["memory_only"]), (3, 2, 2, 1, 1))


if __name__ == "__main__":
    unittest.main()
