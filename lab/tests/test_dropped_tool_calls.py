import unittest

from lab.analysis.dropped_tool_calls import drops, kind


def _agent(message, ts, calls=None, event="thinking"):
    return {"source": "agent", "message": message, "tool_calls": calls, "extra": {"event_type": event, "timestamp": ts}}


class DroppedToolCallsTest(unittest.TestCase):
    def test_kinds(self):
        self.assertEqual(kind('call:read_file{filepath:<|"|>a.py<|"|>}<tool_call|>'), "in_thought")
        self.assertEqual(kind('<|tool_call>call,run_command{command:<|"|>ls<|"|>}<tool_call|>'), "malformed")
        self.assertEqual(kind('<|tool_call>call:edit_file{old_string:<|"|>x'), "truncated")
        self.assertEqual(kind('[call 1/28] run_command{command:<|"|>ls<|"|>}<tool_call|>'), "malformed")
        self.assertIsNone(kind("I will call the tests next."))

    def test_drop_is_nudged_and_timed_to_next_executed_call(self):
        trace = {"steps": [
            _agent("look", 1.0, calls=[{"function_name": "read_file"}]),
            _agent('call:read_file{filepath:<|"|>a.py<|"|>}<tool_call|>', 10.0),
            {"source": "system", "message": "Please continue", "extra": {"event_type": "continuation_nudge"}},
            _agent("again", 14.5, calls=[{"function_name": "read_file"}]),
        ]}
        [d] = drops(trace)
        self.assertEqual(d["kind"], "in_thought")
        self.assertEqual(d["after_call"], 1)
        self.assertTrue(d["nudged"])
        self.assertEqual(d["seconds_to_next_call"], 4.5)

    def test_turns_with_parsed_calls_are_not_drops(self):
        trace = {"steps": [_agent('call:read_file{a:1}<tool_call|>', 1.0, calls=[{"function_name": "read_file"}])]}
        self.assertEqual(drops(trace), [])


if __name__ == "__main__":
    unittest.main()
