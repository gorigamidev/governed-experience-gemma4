import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lab/src"))

from lab.analysis.submit_loops import trace_submits  # noqa: E402


def step(name, content="{}", t=None):
    obs = {"content": content, "extra": {"elapsed_s": t} if t is not None else {}}
    return {"tool_calls": [{"function_name": name, "arguments": {}}], "observation": obs}


def submit(size, t=None):
    return step("submit_patch", f'{{"status": "ok", "patch_size": {size}, "files_changed": 0}}', t)


class SubmitLoopsTest(unittest.TestCase):
    def test_empty_submit_loop_after_budget(self):
        budget = '{"status": "error", "error_type": "BudgetExceeded"}'
        trace = {"steps": [step("run_command", t=1), step("edit_file", budget, t=2)]
                 + [submit(0, t=10 + i) for i in range(4)]}
        r = trace_submits(trace)
        self.assertEqual((r["submits"], r["empty_submits"], r["max_run"]), (4, 4, 4))
        self.assertTrue(r["budget_before_first"] and r["last_empty"])
        self.assertFalse(r["recovered"])
        self.assertEqual((r["calls_after_first"], r["seconds_after_first"]), (3, 3))

    def test_another_call_breaks_the_run_and_a_later_patch_recovers(self):
        trace = {"steps": [submit(0), step("edit_file"), submit(120), submit(120)]}
        r = trace_submits(trace)
        self.assertEqual(r["max_run"], 2)
        self.assertTrue(r["recovered"] and r["resubmit_after_patch"])
        self.assertFalse(r["budget_before_first"])

    def test_no_submit(self):
        r = trace_submits({"steps": [step("run_command")]})
        self.assertEqual((r["submits"], r["max_run"], r["seconds_after_first"]), (0, 0, None))


if __name__ == "__main__":
    unittest.main()
