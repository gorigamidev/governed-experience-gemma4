import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lab/src"))

from lab.analysis.long_turns import trace_row, trace_turns  # noqa: E402


def turn(tokens, t, obs_t, calls=1):
    return {"source": "agent", "metrics": {"completion_tokens": tokens}, "extra": {"elapsed_s": t},
            "tool_calls": [{"function_name": "run_command"}] * calls, "observation": {"extra": {"elapsed_s": obs_t}}}


class LongTurnsTest(unittest.TestCase):
    def test_turn_time_runs_from_the_previous_observation(self):
        trace = {"steps": [{"source": "user", "extra": {"elapsed_s": 1}}, turn(100, 5, 8), turn(4100, 130, 131)]}
        turns = trace_turns(trace)
        self.assertEqual([x["seconds"] for x in turns], [4, 122])
        self.assertEqual([x["calls_before"] for x in turns], [0, 1])

    def test_row_counts_long_and_capped_turns_and_the_timeout(self):
        turns = [{"tokens": 2500, "seconds": 70, "calls_before": 2}, {"tokens": 4100, "seconds": 120, "calls_before": 9},
                 {"tokens": 50, "seconds": 2, "calls_before": 10}]
        r = trace_row(turns, {"duration_seconds": 480, "error": "Agent exceeded session timeout (8.0 min)"})
        self.assertEqual((r["long"], r["capped"], r["long_seconds"], r["long_before_first_call_5"]), (2, 1, 190, 1))
        self.assertTrue(r["timed_out"])
        self.assertFalse(r["resolved"])


if __name__ == "__main__":
    unittest.main()
