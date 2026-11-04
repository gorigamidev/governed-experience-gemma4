import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lab/src"))

from lab.lora import gen_a as G  # noqa: E402
from lab.lora import scripted as S  # noqa: E402


class GenATest(unittest.TestCase):
    def test_reserve_detection(self):
        self.assertTrue(G.is_reserve("rich_3105"))
        self.assertFalse(G.is_reserve("requests_7427"))

    def test_repo_of_uses_instance_id_prefix(self):
        self.assertEqual(G.repo_of("fastapi_14258"), "fastapi")
        self.assertEqual(G.repo_of("rich_3882"), "rich")
        self.assertEqual(G.repo_of("requests_6644"), "requests")
        self.assertEqual(G.repo_of("httpx_123"), "httpx")

    def test_good_turn_count_drops_immediate_repeat_and_adds_done(self):
        trace = {
            "steps": [
                {"source": "agent", "tool_calls": [{"function_name": "read_file", "arguments": {"a": 1}}]},
                {"source": "agent", "tool_calls": [{"function_name": "read_file", "arguments": {"a": 1}}]},
                {"source": "agent", "tool_calls": [{"function_name": "submit_patch", "arguments": {}}]},
            ]
        }
        self.assertEqual(G.good_turn_count(trace), (3, True))  # read + submit, repeat dropped, +DONE

    def test_negative_control_repeat_filter_matters(self):
        # Without the repeat filter every assistant step would be counted.
        trace = {
            "steps": [
                {"source": "agent", "tool_calls": [{"function_name": "read_file", "arguments": {"a": 1}}]},
                {"source": "agent", "tool_calls": [{"function_name": "read_file", "arguments": {"a": 1}}]},
            ]
        }
        filtered, _ = G.good_turn_count(trace)
        unfiltered = sum(1 for s in trace["steps"] if s.get("source") == "agent" and s.get("tool_calls"))
        self.assertNotEqual(filtered, unfiltered)
        self.assertEqual(filtered, 1)
        self.assertEqual(unfiltered, 2)

    def test_reserve_exclusion_command_verifiable(self):
        # Build a fake run dir with one reserve trace and one non-reserve trace.
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            run = base / "r"
            (run / "traces").mkdir(parents=True)
            for iid in ["rich_3105", "requests_7427"]:
                (run / "traces" / f"trace_{iid}.json").write_text(json.dumps({
                    "steps": [{"source": "agent", "tool_calls": [{"function_name": "submit_patch", "arguments": {}}]}]
                }))
            sel = G.collect(["r"], base=base)
            ids = {s["task"] for s in sel}
            self.assertNotIn("rich_3105", ids)
            self.assertIn("requests_7427", ids)
            # Negative control: without exclusion, the reserve task would be present.
            all_ids = {"rich_3105", "requests_7427"}
            self.assertNotEqual(ids, all_ids)

    def test_thinking_from_trace_events_when_unlabeled(self):
        def step(thinking):
            return {"source": "agent", "extra": {"event_type": "thinking" if thinking else "tool_call"},
                    "tool_calls": [{"function_name": "submit_patch", "arguments": {}}]}
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            for run, flags in [("bf", [True, True, False]), ("plain", [True, False, False])]:
                (base / run / "traces").mkdir(parents=True)
                for i, f in enumerate(flags):
                    (base / run / "traces" / f"trace_requests_74{i}0.json").write_text(json.dumps({"steps": [step(f)]}))
            self.assertTrue(G._thinking_of("bf", base)[0])
            self.assertFalse(G._thinking_of("plain", base)[0])
            # A label marker still wins over trace events.
            (base / "bf" / "run_meta.json").write_text(json.dumps({"label": "x-think-off"}))
            self.assertFalse(G._thinking_of("bf", base)[0])


if __name__ == "__main__":
    unittest.main()
