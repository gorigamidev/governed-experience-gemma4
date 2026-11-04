import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lab/src"))

from lab.analysis.pool_report import (apply_ed_rules, apply_gov_rules, apply_rules, apply_tb22_rules,  # noqa: E402
                                      apply_tb_rules,
                                      apply_to_rules, bucket, locator_metrics, replica_noise, trace_metrics)


def edit(args, ok, error_type="FileEditError"):
    content = '{"status": "ok"}' if ok else f'{{"status": "error", "error_type": "{error_type}"}}'
    return {"tool_calls": [{"function_name": "edit_file", "arguments": args}], "observation": {"content": content}}


def call(name):
    return {"tool_calls": [{"function_name": name, "arguments": {}}], "observation": {"content": "{}"}}


def arm(**kw):
    base = {"n": 24, "R": 6, "aux": 5, "mean_min": 4.5, "p90_min": 8.0, "infra_errors": 0}
    return {**base, **kw}


class TraceMetricsTest(unittest.TestCase):
    def test_identical_retry_after_a_failed_edit_counts_and_submit_is_free(self):
        a, b = {"old_string": "x"}, {"old_string": "y"}
        trace = {"steps": [call("run_command"), edit(a, False), edit(a, False), call("read_file"), edit(b, True),
                           call("submit_patch")]}
        self.assertEqual(trace_metrics(trace), {"first_edit_call": 5, "failed_edits": 2, "repeats": 1,
                                               "edits_after_budget": 0, "writes_outside": 0, "py_runs": 0,
                                               "wrapped": 0, "hangs": 0, "cut_by_timeout": 0,
                                               "max_turn_tokens": 0, "long_turn_seconds": [],
                                               "locator_used": False, "locator_calls": 0, "plan_file": None,
                                               "charged_calls": 5, "warning_call": None, "rejected_calls": 0})

    def test_a_retry_after_another_call_is_not_a_repeat(self):
        a = {"old_string": "x"}
        trace = {"steps": [edit(a, False), call("read_file"), edit(a, False)]}
        self.assertEqual(trace_metrics(trace)["repeats"], 0)


    def test_budget_and_path_refusals_are_not_failed_matches(self):
        trace = {"steps": [edit({"content": "x"}, False, "FileWriteError"), edit({"old_string": "x"}, False),
                           edit({"old_string": "y"}, False, "BudgetExceeded"),
                           edit({"old_string": "y"}, False, "BudgetExceeded")]}
        m = trace_metrics(trace)
        self.assertEqual((m["failed_edits"], m["writes_outside"], m["edits_after_budget"], m["repeats"]), (1, 1, 2, 0))


class RulesTest(unittest.TestCase):
    def test_h24_long_within_margin_and_not_worse_is_the_next_submission(self):
        rules = apply_rules({"bf": arm(), "h24": arm(R=7, aux=2), "h24-long": arm(R=7, mean_min=5.0, p90_min=9.1)})
        self.assertTrue(rules[1].startswith("Rule 2 (h24 base): YES"))
        self.assertTrue(rules[2].startswith("Rule 3 (h24-long): CANDIDATE"))
        self.assertEqual(rules[3], "Rule 4 (next submission): h24-long")

    def test_time_margin_blocks_h24_long_whatever_its_r(self):
        rules = apply_rules({"bf": arm(), "h24": arm(), "h24-long": arm(R=12, mean_min=5.1)})  # 5.1 > 1.12 x 4.5
        self.assertIn("NOT SENT (time margin)", rules[2])
        self.assertEqual(rules[3], "Rule 4 (next submission): h24")

    def test_h24_with_more_aux_than_bf_is_not_the_base(self):
        rules = apply_rules({"bf": arm(), "h24": arm(aux=6), "h24-long": arm(mean_min=6.0)})
        self.assertTrue(rules[1].startswith("Rule 2 (h24 base): NO"))
        self.assertEqual(rules[3], "Rule 4 (next submission): none")

    def test_too_few_common_tasks_applies_no_rule(self):
        rules = apply_rules({"bf": arm(n=17), "h24": arm(n=17), "h24-long": arm(n=17)})
        self.assertEqual(len(rules), 1)
        self.assertIn("no rule applies", rules[0])


class GovRulesTest(unittest.TestCase):
    def test_bf_gov_not_worse_on_r_aux_and_time_is_the_base(self):
        out = apply_gov_rules({"bf": arm(R=5, aux=6, mean_min=5.0), "bf-gov": arm(R=4, aux=0, mean_min=5.5)})
        self.assertTrue(out[1].startswith("Rule 2 (bf-gov base): YES"))
        self.assertEqual(out[2], "Rule 3 (next submission): bf-gov")

    def test_bf_gov_over_the_time_margin_sends_nothing(self):
        out = apply_gov_rules({"bf": arm(R=4, aux=8, mean_min=5.11), "bf-gov": arm(R=6, aux=0, mean_min=5.70)})
        self.assertTrue(out[1].startswith("Rule 2 (bf-gov base): NO"))
        self.assertEqual(out[2], "Rule 3 (next submission): none")

    def test_replica_noise_counts_agreement(self):
        a = [{"resolved": True, "minutes": 4.0}, {"resolved": False, "minutes": 6.0}]
        b = [{"resolved": True, "minutes": 5.0}, {"resolved": True, "minutes": 5.0}]
        self.assertEqual(replica_noise(a, b), "bf vs bf-prev: same resolved on 1/2 tasks, mean minutes differ by +0.00")


def run(cmd, content='{"status": "ok", "exit_code": 0}'):
    return {"tool_calls": [{"function_name": "run_command", "arguments": {"command": cmd}}],
            "observation": {"content": content}}


class TimeoutMetricsTest(unittest.TestCase):
    def test_wrapped_runs_hangs_and_timeout_cuts(self):
        trace = {"steps": [run("grep -n foo rich/text.py"), run("cd /workspace && timeout 90 python -c 'x'"),
                           run("timeout 90 pytest tests/test_text.py", '{"status": "ok", "exit_code": 124}'),
                           run("python3 repro.py", '{"status": "error", "error_type": "TimeoutExceeded"}')]}
        m = trace_metrics(trace)
        self.assertEqual((m["py_runs"], m["wrapped"], m["hangs"], m["cut_by_timeout"]), (3, 2, 1, 1))


class TbRulesTest(unittest.TestCase):
    def rows(self, timeouts=()):
        return {a: [{"task": t, "error": "Agent exceeded session timeout (8.0 min)" if t in timeouts else "",
                     "first_edit_call": 9} for t in ("rich_3521", "rich_3471", "rich_3006")]
                for a in ("bf-gov-to", "bf-gov-tb")}

    def test_budget_applied_and_not_worse_is_the_base(self):
        s = {"bf-gov-to": arm(session_timeouts=4, max_turn_tokens=4900, long_turns=9, long_turn_s_p50=120),
             "bf-gov-tb": arm(R=5, aux=6, session_timeouts=3, max_turn_tokens=2600, long_turns=5, long_turn_s_p50=62)}
        out = apply_tb_rules(s, self.rows())
        self.assertTrue(out[1].startswith("Rule 2 (compliance): MET"))
        self.assertTrue(out[2].startswith("Rule 3 (bf-gov-tb base): YES"))
        self.assertIn("timeouts 0/2", out[3])

    def test_budget_not_applied_is_not_read(self):
        s = {"bf-gov-to": arm(session_timeouts=4, max_turn_tokens=4900, long_turns=9, long_turn_s_p50=120),
             "bf-gov-tb": arm(R=9, session_timeouts=0, max_turn_tokens=3100, long_turns=9, long_turn_s_p50=65)}
        out = apply_tb_rules(s, self.rows())
        self.assertIn("NOT MET", out[1])
        self.assertEqual(out[2], "Rule 3 (bf-gov-tb base): NOT READ (rule 2 not met)")

    def test_more_session_timeouts_keeps_bf_gov_to(self):
        s = {"bf-gov-to": arm(session_timeouts=2, long_turns=0, long_turn_s_p50=None, max_turn_tokens=1900),
             "bf-gov-tb": arm(session_timeouts=3, long_turns=0, long_turn_s_p50=None, max_turn_tokens=1900)}
        out = apply_tb_rules(s, self.rows(["rich_3521"]))
        self.assertTrue(out[2].startswith("Rule 3 (bf-gov-tb base): NO"))
        self.assertIn("bf-gov-tb: timeouts 1/2", out[3])


class Tb22RulesTest(unittest.TestCase):
    def rows(self, warning_calls=(12, 12), resolved=(True, False)):
        row = lambda t, w, r: {"task": t, "error": "", "first_edit_call": 9, "charged_calls": 20, "warning_call": w,
                               "rejected_calls": 0, "resolved": r}
        return {a: [row(t, w, r) for t, w, r in zip(("rich_3521", "rich_3006"), warning_calls, resolved)]
                for a in ("bf-gov-tb", "bf-gov-tb-22")}

    def test_cap_applied_and_not_worse_is_the_base(self):
        s = {"bf-gov-tb": arm(session_timeouts=2, max_tool_calls=28, edit_by_12=10, late_or_never=6, call_cap=9),
             "bf-gov-tb-22": arm(R=5, aux=6, session_timeouts=1, max_tool_calls=22, edit_by_12=12, late_or_never=4,
                                 call_cap=12)}
        prev = self.rows(resolved=(False, False))["bf-gov-tb"]
        out = apply_tb22_rules(s, self.rows(), prev)
        self.assertTrue(out[1].startswith("Rule 2 (compliance): MET"))
        self.assertTrue(out[2].startswith("Rule 3 (bf-gov-tb-22 base): YES"))
        self.assertIn("1 tasks change (rich_3521)", out[4])

    def test_warning_off_12_is_not_read(self):
        s = {"bf-gov-tb": arm(session_timeouts=2, max_tool_calls=28, edit_by_12=10, late_or_never=6, call_cap=9),
             "bf-gov-tb-22": arm(R=9, session_timeouts=0, max_tool_calls=22, edit_by_12=12, late_or_never=4,
                                 call_cap=12)}
        out = apply_tb22_rules(s, self.rows(warning_calls=(12, 18)), None)
        self.assertIn("NOT MET", out[1])
        self.assertEqual(out[2], "Rule 3 (bf-gov-tb-22 base): NOT READ (rule 2 not met)")
        self.assertEqual(len(out), 4)

    def test_more_session_timeouts_keeps_bf_gov_tb(self):
        s = {"bf-gov-tb": arm(session_timeouts=1, max_tool_calls=28, edit_by_12=10, late_or_never=6, call_cap=9),
             "bf-gov-tb-22": arm(session_timeouts=2, max_tool_calls=22, edit_by_12=12, late_or_never=4, call_cap=12)}
        out = apply_tb22_rules(s, self.rows(), None)
        self.assertTrue(out[2].startswith("Rule 3 (bf-gov-tb-22 base): NO"))


class EdRulesTest(unittest.TestCase):
    def rows(self):
        r = {"task": "rich_3521", "locator_used": True, "locator_calls": 4, "plan_file": "rich/table.py",
             "first_edit_call": 7}
        return {"bf-gov-to": [{**r, "locator_used": False, "plan_file": None, "first_edit_call": 14}],
                "bf-gov-ed": [r]}

    def ed(self, **kw):
        return arm(**{"late_or_never": 3, "edited": 20, "plan_ok": 22, "undeclared": 0, "session_timeouts": 2,
                      "edits_after_budget": 1, "call_cap": 4, **kw})

    def test_fewer_late_edits_with_a_plan_and_not_worse_is_the_base(self):
        s = {"bf-gov-to": arm(late_or_never=8, edited=18, plan_ok=0, undeclared=0, session_timeouts=3,
                              edits_after_budget=4, call_cap=6), "bf-gov-ed": self.ed(R=5)}
        out = apply_ed_rules(s, self.rows(), "bf-gov-to", {"rich_3521": {"rich/table.py"}})
        self.assertTrue(out[1].startswith("Rule 2a (compliance): MET"))
        self.assertTrue(out[2].startswith("Rule 2b (structure): MET"))
        self.assertTrue(out[3].startswith("Rule 3 (bf-gov-ed base): YES"))
        self.assertIn("plan file in gold 1/22", out[4])

    def test_no_plan_is_not_read(self):
        s = {"bf-gov-to": arm(late_or_never=8, edited=18, plan_ok=0, undeclared=0, session_timeouts=3,
                              edits_after_budget=4, call_cap=6), "bf-gov-ed": self.ed(plan_ok=10)}
        out = apply_ed_rules(s, self.rows(), "bf-gov-to", {})
        self.assertIn("NOT MET", out[2])
        self.assertEqual(out[3], "Rule 3 (bf-gov-ed base): NOT READ (rule 2 not met)")

    def test_undeclared_tools_invalidate(self):
        s = {"bf-gov-to": arm(late_or_never=8, edited=18, plan_ok=0, undeclared=0), "bf-gov-ed": self.ed(undeclared=2)}
        self.assertIn("FAIL (undeclared 2 > 1)", apply_ed_rules(s, self.rows(), "bf-gov-to", {})[0])

    def test_bucket_uses_the_arm_budget(self):
        row = {"first_edit_call": 17, "edits_after_budget": 0}
        self.assertEqual(bucket(row, 20), "late")
        self.assertEqual(bucket(row, 28), "mid")


class LocatorMetricsTest(unittest.TestCase):
    def test_locator_calls_and_plan_file_and_the_agent_tool_call_is_free(self):
        trace = {"steps": [{"extra": {"author": "swe_agent"}, "tool_calls": [{"function_name": "locator"}]},
                           {"extra": {"author": "locator"}, "tool_calls": [{"function_name": "run_command"}]},
                           {"extra": {"author": "locator"}, "tool_calls": [{"function_name": "read_file"}]},
                           {"extra": {"author": "locator"}, "message": "FILE: /workspace/rich/table.py:10-20 - x\nCHANGE: y"},
                           {"extra": {"author": "swe_agent"},
                            "tool_calls": [{"function_name": "edit_file", "arguments": {}}],
                            "observation": {"content": "{\"ok\": true}"}}]}
        self.assertEqual(locator_metrics(trace), {"locator_used": True, "locator_calls": 2, "plan_file": "rich/table.py"})
        self.assertEqual(trace_metrics(trace)["first_edit_call"], 3)


class ToRulesTest(unittest.TestCase):
    def rows(self, cut=()):
        return {"bf-gov-to": [{"task": t, "cut_by_timeout": 1} for t in cut]}

    def test_complying_and_not_worse_is_the_base(self):
        s = {"bf-gov": arm(R=5, aux=0, mean_min=4.9, hangs=2, wrapped_pct=0),
             "bf-gov-to": arm(R=4, aux=1, mean_min=5.3, hangs=0, wrapped_pct=85)}
        out = apply_to_rules(s, self.rows(["rich_3480"]))
        self.assertIn("Rule 2 (compliance, no decision): MET", out[1])
        self.assertIn("rich_3480", out[1])
        self.assertTrue(out[2].startswith("Rule 3 (bf-gov-to base, GPU only): YES"))
        self.assertTrue(out[3].endswith("bf-gov-to"))

    def test_more_hangs_than_bf_gov_keeps_bf_gov(self):
        s = {"bf-gov": arm(hangs=0, wrapped_pct=0), "bf-gov-to": arm(hangs=1, wrapped_pct=90)}
        out = apply_to_rules(s, self.rows())
        self.assertIn("NOT MET", out[1])
        self.assertTrue(out[2].startswith("Rule 3 (bf-gov-to base, GPU only): NO"))
        self.assertTrue(out[3].endswith("bf-gov"))
