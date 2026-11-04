import unittest

from lab.analysis.second_opinion_sim import apply_policy, first_edit, rule_1


class SecondOpinionTest(unittest.TestCase):
    def test_policy_keeps_own_file_in_top_3_and_switches_otherwise(self):
        gold = {"rich.text"}
        self.assertEqual(apply_policy("rich.text", ["a", "b", "rich.text", "c"], gold), "keep_right")
        self.assertEqual(apply_policy("rich.text", ["a", "b", "c", "rich.text"], gold), "loss")
        self.assertEqual(apply_policy("x", ["rich.text", "b", "c"], gold), "gain")
        self.assertEqual(apply_policy("x", ["a", "b", "c"], gold), "switch_wrong")
        self.assertEqual(apply_policy("x", [], gold), "keep_wrong")
        self.assertEqual(apply_policy(None, ["rich.text"], gold), "none")

    def test_first_edit_and_rule(self):
        trace = {"steps": [{"tool_calls": [{"function_name": "read_file", "arguments": {"filepath": "a.py"}}]},
                           {"tool_calls": [{"function_name": "edit_file", "arguments": {"filepath": "b.py"}}]}]}
        self.assertEqual(first_edit(trace), "b.py")
        self.assertIn("NOT MET", rule_1(["gain"] * 5 + ["loss"] * 3))
        self.assertIn("MET |", rule_1(["gain"] * 5 + ["loss"]))
        self.assertIn("NOT MET", rule_1(["gain"] * 3))


if __name__ == "__main__":
    unittest.main()
