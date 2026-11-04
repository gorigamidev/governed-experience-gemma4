import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lab/src"))

from lab.tools.preflight import declared_model, trace_problems  # noqa: E402


def results_dir(root: Path, steps: list[dict], error: str | None = None) -> Path:
    (root / "traces").mkdir(parents=True)
    (root / "task_results.jsonl").write_text(json.dumps(
        {"instance_id": "t_1", "resolved": False, "duration_seconds": 1, "tool_calls": len(steps), "error": error}) + "\n")
    (root / "traces" / "trace_t_1.json").write_text(json.dumps({"steps": steps, "final_metrics": {}}))
    return root


def call(step_id: int, tool: str, content, args: dict | None = None) -> dict:
    return {"step_id": step_id, "source": "agent", "metrics": {"completion_tokens": 1},
            "tool_calls": [{"function_name": tool, "arguments": args or {}, "extra": {"elapsed_s": step_id}}],
            "observation": {"content": content}}


def locate(step_id: int, items: list) -> dict:
    out = json.dumps({"ok": True, "decision": "ESCALATE", "items": items, "habits": []})
    return call(step_id, "run_skill_script", json.dumps({"stdout": out, "status": "success"}),
                {"skill_name": "mini-palace", "file_path": "scripts/entrypoint.py", "args": {"action": "locate"}})


class DeclaredModelTest(unittest.TestCase):
    def test_reads_the_root_agent_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            agent = Path(d, "agent.yaml")
            agent.write_text("name: swe_agent\nmodel: gemma-4-e4b-it-qat-w4a16-ct\ninstruction: x\n")
            self.assertEqual(declared_model(Path(d)), "e4b")
            agent.write_text("name: swe_agent\nmodel: gemma-4-31b-it-qat-w4a16-ct\n")
            self.assertEqual(declared_model(Path(d)), "31b")


class TraceProblemsTest(unittest.TestCase):
    def check(self, steps, has_skill=True):
        with tempfile.TemporaryDirectory() as d:
            return trace_problems(results_dir(Path(d), steps), has_skill)

    def test_blind_skill_and_missing_modules_fail(self):
        infra, _ = self.check([locate(1, []), call(2, "search_similar_code", json.dumps(
            {"status": "error", "error_message": "No module named 'cachetools'"}))])
        self.assertEqual(infra, ["t_1: locate returned no candidates (is the skill seeing the repo?)",
                                 "t_1: search_similar_code -> No module named"])

    def test_file_contents_and_model_commands_are_not_infrastructure(self):
        infra, _ = self.check([locate(1, [{"file": "a.py"}]),
                               call(2, "read_file", json.dumps({"status": "ok", "content": "except ImportError:"})),
                               call(3, "run_command", json.dumps({"status": "error", "error_message": "No module named x"}))])
        self.assertEqual(infra, [])

    def test_malformed_skill_call_is_model_behavior(self):
        rejected = call(1, "run_skill_script", json.dumps({"error": "not found", "error_code": "SCRIPT_NOT_FOUND"}),
                        {"skill_name": "mini-palace", "file_path": "locate.py", "args": ["x"]})
        infra, notes = self.check([rejected, locate(2, [{"file": "a.py"}])])
        self.assertEqual(infra, [])
        self.assertEqual(notes, ["t_1: malformed skill call rejected by ADK"])

    def test_skill_call_without_a_valid_action_is_model_behavior(self):
        usage = json.dumps({"stdout": json.dumps({"ok": False, "error": "unknown action: 'X.y'"}), "status": "success"})
        invented = call(1, "run_skill_script", usage,
                        {"skill_name": "mini-palace", "file_path": "scripts/entrypoint.py", "args": {"file": "a.py"}})
        infra, notes = self.check([invented, locate(2, [{"file": "a.py"}])])
        self.assertEqual(infra, [])
        self.assertEqual(notes, ["t_1: model sent a skill call without a valid action"])

    def test_a_known_skill_action_failing_is_infrastructure(self):
        failed = call(1, "run_skill_script", json.dumps({"stdout": json.dumps({"ok": False, "error": "boom"})}),
                      {"skill_name": "mini-palace", "file_path": "scripts/entrypoint.py", "args": {"action": "outline"}})
        infra, _ = self.check([failed, locate(2, [{"file": "a.py"}])])
        self.assertEqual(infra, ["t_1: skill outline returned ok=false"])

    def test_an_arm_that_never_acted_verified_nothing(self):
        infra, notes = self.check([], has_skill=False)
        self.assertEqual(infra, ["arm not exercised: no tool call in any task, so nothing was verified"])
        self.assertTrue(notes[0].startswith("t_1: not exercised"))

    def test_a_skill_arm_must_run_the_skill_once(self):
        infra, _ = self.check([call(1, "read_file", json.dumps({"status": "ok", "content": "x"}))])
        self.assertEqual(infra, ["skill not exercised: no successful skill call in any task; rerun with more tasks"])


if __name__ == "__main__":
    unittest.main()
