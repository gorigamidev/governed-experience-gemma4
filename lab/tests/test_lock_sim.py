import json
import unittest

from lab.analysis import lock_sim


def step(name, args, content):
    return {"tool_calls": [{"function_name": name, "arguments": args}], "observation": {"content": json.dumps(content)}}


class LockSimTest(unittest.TestCase):
    def test_final_edit_position_and_in_hand(self):
        trace = {"steps": [
            step("read_file", {"filepath": "a.py"}, {"status": "ok", "content": "def f():\n    return 1  \n"}),
            step("get_status", {}, {"status": "ok"}),
            step("run_command", {"command": "grep -n x a.py"}, {"status": "ok", "stdout": "1:x"}),
            step("edit_file", {"filepath": "a.py", "old_string": "def f():\n    return 1", "new_string": "y"},
                 {"status": "ok"}),
        ]}
        s = lock_sim.scan(trace)
        self.assertEqual(s["final"]["at"], 3)
        self.assertEqual(len(s["lock"]), 2)
        self.assertTrue(lock_sim.in_hand(s["final"]["old_string"], s["lock"][:1]))
        self.assertFalse(lock_sim.in_hand(s["final"]["old_string"], s["lock"][1:]))

    def test_budget_refused_edit_is_final(self):
        trace = {"steps": [step("read_file", {"filepath": "a.py"}, {"status": "ok", "content": "z"}),
                           step("edit_file", {"filepath": "a.py", "old_string": "z", "new_string": "w"},
                                {"status": "error", "error_type": "BudgetExceeded"})]}
        s = lock_sim.scan(trace)
        self.assertTrue(s["final"]["refused"])
        self.assertIsNone(s["first_ok"])

    def test_write_file_first_has_no_final(self):
        trace = {"steps": [step("write_file", {"filepath": "a.py", "content": "q"}, {"status": "ok"})]}
        s = lock_sim.scan(trace)
        self.assertIsNone(s["final"])
        self.assertEqual(s["first_ok"], 1)


if __name__ == "__main__":
    unittest.main()
