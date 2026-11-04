import unittest

from lab.analysis.lever_report import (complies, counter, first_edit, harness_warning_call, lever_metrics,
                                      looked_before_edit, read_to_edit_gap, readings)


def _turn(text, *calls):
    return {"source": "agent", "message": text,
            "tool_calls": [{"function_name": n, "arguments": a} for n, a in calls]}


TRACE = {"steps": [
    _turn("Step 1/28: look", ("run_command", {"command": "git log --oneline -5 -- rich/text.py"})),
    _turn("Step 2/28", ("read_file", {"filepath": "rich/text.py"})),
    _turn("thinking", ("read_file", {"filepath": "rich/console.py"})),
    _turn("Step 9/28 edit", ("edit_file", {"filepath": "rich/text.py"})),
    _turn("", ("submit_patch", {})),
]}


class LeverReportTest(unittest.TestCase):
    def test_counter_counts_turns_and_accuracy(self):
        self.assertEqual(counter(TRACE), (4, 3, 2))

    def test_first_edit_gap_and_looked(self):
        self.assertEqual(first_edit(TRACE), (4, "rich/text.py"))
        self.assertEqual(read_to_edit_gap(TRACE), 2)
        self.assertTrue(looked_before_edit(TRACE))

    def test_metrics_count_dropped_calls(self):
        broken = {"steps": TRACE["steps"] + [_turn('[step 5/28] run_command{command:<|"|>ls<|"|>}<tool_call|>')]}
        self.assertEqual(lever_metrics([("t", broken)], set())["dropped_calls"], 1)
        self.assertEqual(lever_metrics([("t", TRACE)], set())["dropped_calls"], 0)

    def test_harness_warning_call_skips_rejected_calls(self):
        rejected = {"content": '{"error": "... the following mandatory input parameters are not present:\\nold_string"}'}
        steps = [dict(_turn("", ("read_file", {})), observation={"content": "ok"}) for _ in range(3)]
        steps[1]["observation"] = rejected
        steps.append(dict(_turn("", ("read_file", {})), observation={"content": '{"budget_warning": "x"}'}))
        self.assertEqual(harness_warning_call({"steps": steps}), 3)
        self.assertEqual(lever_metrics([("t", {"steps": steps})], set())["rejected_calls"], 1)

    def test_complies(self):
        base = {"gap_p50": 6}
        self.assertTrue(complies("a", {"counter_turns": 10, "counter_tagged": 8, "counter_right": 7}, base))
        self.assertFalse(complies("a", {"counter_turns": 10, "counter_tagged": 7, "counter_right": 7}, base))
        self.assertTrue(complies("b", {"short_first_edit_p50": 8, "short_looked": 3, "short_traces": 6}, base))
        self.assertFalse(complies("b", {"short_first_edit_p50": 9, "short_looked": 6, "short_traces": 6}, base))
        self.assertTrue(complies("c", {"gap_p50": 4}, base))
        self.assertFalse(complies("c", {"gap_p50": 5}, base))
        self.assertTrue(complies("d", {"warned_traces": 3, "warned_at_12": 3}, base))
        self.assertFalse(complies("d", {"warned_traces": 3, "warned_at_12": 2}, base))

    def test_submission_order_defaults_to_structure(self):
        s = {a: {"infra_errors": 0, "late_or_never": 1} for a in ("bf-gov-to", "bf-gov-22", "bf-gov-ti")}
        m = {"bf-gov-to": {"gap_p50": 5}, "bf-gov-22": {"warned_traces": 2, "warned_at_12": 1},
             "bf-gov-ti": {"short_first_edit_p50": 6, "short_looked": 2, "short_traces": 2}}
        self.assertIn("Submission order: bf-gov-ti first", readings(s, m, 20)[-1])
        self.assertIn("Submission order: bf-gov-22 first", readings(s, m, 10)[-1])


if __name__ == "__main__":
    unittest.main()
