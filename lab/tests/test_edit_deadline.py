import unittest

from lab.analysis.edit_deadline import after_warning, group, warning_call


def _call(name, observation=""):
    return {"source": "agent", "tool_calls": [{"function_name": name}], "observation": observation}


class EditDeadlineTest(unittest.TestCase):
    def test_warning_call_skips_submit(self):
        steps = [_call("read_file"), _call("submit_patch"), _call("run_command", '{"budget_warning": "Only 10"}')]
        self.assertEqual(warning_call({"steps": steps}), 2)
        self.assertIsNone(warning_call({"steps": [_call("read_file")]}))

    def test_after_warning(self):
        self.assertIsNone(after_warning(10, 18))
        self.assertIsNone(after_warning(None, 18))
        self.assertEqual(after_warning(15, 18), "before_warning")
        self.assertEqual(after_warning(21, 18), "within_3_after")
        self.assertEqual(after_warning(22, 18), "later")

    def test_group(self):
        self.assertEqual(group("early"), "early")
        self.assertEqual(group("mid"), "mid")
        self.assertEqual(group("refused_only"), "late_or_never")


if __name__ == "__main__":
    unittest.main()
