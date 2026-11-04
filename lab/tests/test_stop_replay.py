import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lab/src"))

from lab.analysis.stop_replay import (  # noqa: E402
    Task, parse_trace, patch_changed, replay, rule_failed_checks, rule_first_submit, rule_max_turns, self_checks,
)


def agent_step(t, tool, args, obs, author="swe_coder", prompt=1000, completion=20):
    return {"source": "agent", "metrics": {"prompt_tokens": prompt, "completion_tokens": completion},
            "extra": {"elapsed_s": t, "author": author},
            "tool_calls": [{"function_name": tool, "arguments": args}],
            "observation": {"content": json.dumps(obs), "extra": {"elapsed_s": t + 0.1}}}


def trace(*steps):
    return {"steps": [{"source": "system", "extra": {"elapsed_s": 0.0}}, *steps]}


PASS = {"status": "ok", "exit_code": 0}
FAIL = {"status": "ok", "exit_code": 1}


class StopReplayTest(unittest.TestCase):
    def load(self, data, resolved=False, duration=None, llm_calls=None):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "trace.json"
            p.write_text(json.dumps(data))
            steps = parse_trace(p)
        return Task("demo", resolved, duration or steps[-1].end + 1, llm_calls or len(steps), steps)

    def test_submit_loop_is_cut_and_not_lost_when_patch_is_unchanged(self):
        loop = [agent_step(10 + i, "submit_patch", {}, {"status": "ok", "patch_size": 500}) for i in range(5)]
        t = self.load(trace(agent_step(1, "edit_file", {"path": "a.py"}, {"status": "ok"}), *loop), resolved=True)
        self.assertEqual(rule_first_submit(t), 1)
        self.assertTrue(t.steps[2].repeat)
        self.assertFalse(patch_changed(t))
        r = replay([t], rule_first_submit, patch_changed)
        self.assertEqual((r["fired"], r["lost"]), (1, []))
        self.assertGreater(r["saved_min"], 0)

    def test_clean_stop_does_not_fire_and_changed_patch_counts_as_lost(self):
        clean = self.load(trace(agent_step(1, "submit_patch", {}, {"status": "ok", "patch_size": 500})))
        self.assertIsNone(rule_first_submit(clean))
        changed = self.load(trace(agent_step(1, "submit_patch", {}, {"status": "ok", "patch_size": 500}),
                                  agent_step(2, "run_command", {"command": "git restore ."}, PASS),
                                  agent_step(3, "submit_patch", {}, {"status": "ok", "patch_size": 80})), resolved=True)
        self.assertTrue(patch_changed(changed))
        self.assertEqual(replay([changed], rule_first_submit, patch_changed)["lost"], ["demo"])

    def test_failed_self_checks_and_max_turns(self):
        runs = [agent_step(i, "run_command", {"command": "python -m pytest tests"}, FAIL) for i in range(1, 4)]
        t = self.load(trace(*runs, agent_step(5, "run_command", {"command": "pytest tests"}, PASS)))
        self.assertEqual(self_checks(t), "FFFP")
        self.assertEqual(rule_failed_checks(t, 3), 2)
        self.assertIsNone(rule_failed_checks(t, 4))
        self.assertEqual(rule_max_turns(t, 2), 1)
        self.assertIsNone(rule_max_turns(t, 4))


if __name__ == "__main__":
    unittest.main()
