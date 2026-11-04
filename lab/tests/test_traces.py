import importlib.util
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lab/src"))

from lab.ingest.load_traces import load_results  # noqa: E402

FIXTURE = ROOT / "lab/tests/fixtures/eval_run"


class ParseResultsTest(unittest.TestCase):
    def setUp(self):
        self.runs = {r.instance_id: r for r in load_results(FIXTURE)}

    def test_outcomes_and_patch(self):
        ok, bad = self.runs["demo_1"], self.runs["demo_2"]
        self.assertTrue(ok.resolved)
        self.assertEqual(ok.patch_files, ["src/demo/core.py"])
        self.assertFalse(bad.resolved)
        self.assertEqual(bad.error, "agent timed out")
        self.assertEqual(bad.patch_files, [])

    def test_skill_usage_and_habits(self):
        ok = self.runs["demo_1"]
        self.assertTrue(ok.skill_first)
        (call,) = ok.skill_calls
        self.assertEqual((call.action, call.ok, call.decision), ("locate", True, "ESCALATE"))
        self.assertEqual(call.files, ["src/demo/core.py", "src/demo/cache.py"])
        self.assertEqual(ok.habits, ["Pa1b2c3d4", "Pe5f60718"])

    def test_truncated_skill_output_still_yields_habits(self):
        bad = self.runs["demo_2"]
        self.assertEqual(bad.first_tool, "run_command")  # get_status does not count as a first action
        self.assertFalse(bad.skill_first)
        self.assertIsNone(bad.skill_calls[0].ok)
        self.assertEqual(bad.habits, ["Pa1b2c3d4"])


class VerifiedOverrideTest(unittest.TestCase):
    def test_local_reverification_overrides_resolved(self):
        with tempfile.TemporaryDirectory() as d:
            shutil.copytree(FIXTURE, Path(d, "run"))
            Path(d, "run/verified_task_results.jsonl").write_text('{"instance_id": "demo_1", "resolved": false}\n')
            runs = {r.instance_id: r for r in load_results(Path(d, "run"))}
        self.assertFalse(runs["demo_1"].resolved)
        self.assertFalse(runs["demo_2"].resolved)


class CostTest(unittest.TestCase):
    def test_thinking_generation_and_first_edit(self):
        from lab.ingest.load_traces import parse_cost

        def step(event, msg, tool=None, at=0.0):
            s = {"source": "agent", "message": msg, "extra": {"event_type": event}, "metrics": {"completion_tokens": 1}}
            if tool:
                s["tool_calls"] = [{"function_name": tool, "extra": {"elapsed_s": at}}]
            return s

        failed = step("thinking", "x", "edit_file", 30.0)
        failed["observation"] = {"content": {"status": "error", "error_message": "old_string not found"}}
        trace = {"steps": [{"source": "user", "message": "task"}, step("thinking", "abc", "read_file", 5.0), failed,
                           step("thinking", "de", "edit_file", 42.5), step("thinking", "f", "write_file", 60.0)],
                 "final_metrics": {"total_completion_tokens": 321}}
        cost = parse_cost(trace)
        self.assertEqual((cost.llm_calls, cost.completion_tokens, cost.thinking_chars, cost.first_edit_s,
                          cost.failed_edits), (4, 321, 7, 42.5, 1))


class UndeclaredToolTest(unittest.TestCase):
    def test_matches_the_adk_error_the_harness_reports(self):
        from lab.analysis.trace_report import UNDECLARED_TOOL

        self.assertTrue(UNDECLARED_TOOL.search("Sandbox execution error: Tool 'submit_patch' not found.\nAvailable"))
        self.assertFalse(UNDECLARED_TOOL.search("Agent exceeded session timeout (4.0 min)"))
        self.assertFalse(UNDECLARED_TOOL.search("File not found: rich/x.py"))


class HabitG1MetricsTest(unittest.TestCase):
    def test_b_k_and_rescue(self):
        from lab.analysis.trace_report import render

        base = {"verifiable": True, "resolved": False, "skill_used": True, "skill_first": True, "locate_gold_rank": None,
                "patch_touches_gold": False, "error": "", "undeclared_tool": False, "first_edit_s": None,
                "gen_tokens": None, "thinking_kchars": None, "llm_calls": None, "failed_edits": 0,
                "check_no_changes": False, "submitted": True}
        rows = [{**base, "task": "a", "has_patch": True, "check_calls": 2, "check_broken": True, "final_broken": False},
                {**base, "task": "b", "has_patch": True, "check_calls": 1, "check_broken": True, "final_broken": True},
                {**base, "task": "c", "has_patch": True, "check_calls": 0, "check_broken": False, "final_broken": False},
                {**base, "task": "d", "has_patch": False, "check_calls": 0, "check_broken": False, "final_broken": None},
                {**base, "task": "e", "has_patch": False, "check_calls": 1, "check_broken": False, "final_broken": None,
                 "check_no_changes": True},
                {**base, "task": "f", "has_patch": False, "check_calls": 1, "check_broken": False, "final_broken": None,
                 "check_no_changes": True, "submitted": False}]
        report = render(rows, {})
        self.assertIn("B, final patch `broken` when check is replayed offline: 1/3", report)
        self.assertIn("K, tasks with a patch that called check: 2/3", report)
        self.assertIn("is not broken: 1/2", report)
        self.assertIn("submitted an empty patch: 1/2; tasks without a patch: 3", report)


class UnpackArmsTest(unittest.TestCase):
    def test_single_and_multi_arm_archives(self):
        import tarfile

        from lab.eval.process_traces import unpack

        with tempfile.TemporaryDirectory() as d:
            for layout, arms in (("single", [""]), ("multi", ["on", "off"])):
                src = Path(d, layout, "results")
                for arm in arms:
                    shutil.copytree(FIXTURE, src / arm)
                archive = Path(d, f"{layout}.tar.gz")
                with tarfile.open(archive, "w:gz") as tar:
                    tar.add(src, arcname="results")
                targets = unpack(archive, Path(d, "eval"), layout)
                expected = {arm: Path(d, "eval", f"{layout}-{arm}" if arm else layout) for arm in arms}
                self.assertEqual(targets, expected)
                self.assertTrue(all((t / "task_results.jsonl").exists() for t in targets.values()))
            with self.assertRaises(SystemExit):
                unpack(archive, Path(d, "eval"), "multi")
            self.assertEqual(sorted(p.name for p in Path(d, "eval").iterdir()), ["multi-off", "multi-on", "single"])


class WheelSetTest(unittest.TestCase):
    def test_extra_wheels_only_for_fastapi(self):
        from lab.eval.verify_patches import DATA, wheels_for

        # inline-snapshot's plugin imports `attrs`, which rich shadows with examples/attrs.py.
        self.assertEqual(wheels_for("Textualize/rich", None), DATA / "wheels")
        self.assertEqual(wheels_for("Textualize/rich", Path("/w")), Path("/w"))
        self.assertIn(wheels_for("fastapi/fastapi", None).name, ("wheels_local", "wheels"))


@unittest.skipUnless(importlib.util.find_spec("linaldb"), "linaldb not installed (offline store tests)")
class IngestEvalTest(unittest.TestCase):
    def test_ingest_is_idempotent_and_links_habits(self):
        from lab.store.experience_store import ExperienceStore
        from lab.store.ingest_eval import ingest

        with tempfile.TemporaryDirectory() as d:
            store = ExperienceStore(d)
            first = ingest(store, FIXTURE, "fixture")
            self.assertFalse(first.get("skipped"))
            rows = store.query("SELECT pattern_id, resolved FROM habit_activation")
            self.assertEqual(sorted((r["pattern_id"], r["resolved"]) for r in rows),
                             [("Pa1b2c3d4", False), ("Pa1b2c3d4", True), ("Pe5f60718", True)])
            exp = store.query("SELECT file, outcome FROM experience")
            self.assertEqual(exp, [{"file": "demo/core.py", "outcome": "resolved"}])
            self.assertTrue(ingest(store, FIXTURE, "fixture").get("skipped"))
            self.assertEqual(first["kind"], "agent_eval")  # no run_meta.json: external run, assumed faithful

    def test_outcomes_outside_the_verifiable_set_are_unverified(self):
        from lab.consolidate.patterns import load_experience
        from lab.store.experience_store import ExperienceStore
        from lab.store.ingest_eval import ingest

        with tempfile.TemporaryDirectory() as d:
            store = ExperienceStore(d)
            out = ingest(store, FIXTURE, "fixture", verifiable={"demo_1": False})
            self.assertEqual(out["unverified"], 2)
            self.assertEqual({r["outcome"] for r in store.query("SELECT outcome FROM experience")}, {"unverified"})
            self.assertNotIn("demo_1", load_experience(store))

    def test_replace_takes_the_new_verification_without_duplicates(self):
        from lab.store.experience_store import ExperienceStore
        from lab.store.ingest_eval import ingest

        with tempfile.TemporaryDirectory() as d:
            run = Path(d, "run")
            shutil.copytree(FIXTURE, run)
            store = ExperienceStore(Path(d, "store"))
            ingest(store, run, "fixture")
            (run / "verified_task_results.jsonl").write_text('{"instance_id": "demo_1", "resolved": false}\n')
            self.assertTrue(ingest(store, run, "fixture").get("skipped"))
            ingest(store, run, "fixture", replace=True)
            self.assertEqual(len(store.query("SELECT run_id FROM run")), 1)
            self.assertEqual(store.query("SELECT outcome FROM experience"), [{"outcome": "unresolved"}])

    def test_non_scorer_model_is_kept_out_of_consolidation(self):
        from lab.consolidate.patterns import load_experience
        from lab.store.experience_store import ExperienceStore
        from lab.store.ingest_eval import ingest

        with tempfile.TemporaryDirectory() as d:
            run = Path(d, "run")
            shutil.copytree(FIXTURE, run)
            (run / "run_meta.json").write_text('{"scorer_faithful": false, "served_model": "qwen3:8b"}')
            store = ExperienceStore(Path(d, "store"))
            self.assertEqual(ingest(store, run, "local")["kind"], "pipeline_test")
            self.assertEqual(load_experience(store), {})


if __name__ == "__main__":
    unittest.main()


class VerifiableSetTest(unittest.TestCase):
    def test_empty_patch_pass_removes_a_task(self):
        import json
        import tempfile
        from pathlib import Path

        from lab.eval.verifiable import load_verifiable

        with tempfile.TemporaryDirectory() as d:
            gold, empty = Path(d, "gold.json"), Path(d, "empty.json")
            gold.write_text(json.dumps({"a": True, "b": True, "c": False}))
            self.assertEqual(load_verifiable(gold, empty), {"a": True, "b": True, "c": False})
            empty.write_text(json.dumps({"a": False, "b": True}))
            self.assertEqual(load_verifiable(gold, empty), {"a": True, "b": False, "c": False})
            self.assertIsNone(load_verifiable(Path(d, "missing.json"), empty))


class BehaviorFlagsTest(unittest.TestCase):
    def test_flags_from_a_trace(self):
        import json
        import tempfile
        from pathlib import Path

        from lab.tools.preflight import behavior_flags

        agent = lambda tool, args, obs: {"source": "agent", "tool_calls": [{"function_name": tool, "arguments": args}],
                                          "observation": {"content": obs}}
        steps = [{"source": "user", "message": "task"}]
        steps += [{"source": "agent", "message": "thinking out loud"}] * 5
        steps += [agent("run_command", {"command": "pytest"}, "{}")] * 4
        steps += [agent("run_skill_script", {"x": 1}, '{"error_code": "INVALID_ARGUMENTS"}')]
        steps += [agent("submit_patch", {}, '{"status": "ok", "patch_size": 0}')]
        with tempfile.TemporaryDirectory() as d:
            res = Path(d)
            (res / "traces").mkdir()
            (res / "traces/trace_t1.json").write_text(json.dumps({"steps": steps, "final_metrics": {}}))
            (res / "task_results.jsonl").write_text(json.dumps(
                {"instance_id": "t1", "resolved": False, "error": None, "duration_seconds": 10, "tool_calls": 6,
                 "agent_patch_size": 0}) + "\n")
            flags = " | ".join(behavior_flags(res))
        self.assertIn("empty patch submitted", flags)
        self.assertIn("5 text-only turns in a row", flags)
        self.assertIn("1 malformed skill call", flags)
        self.assertIn("the same tool call 4 times in a row", flags)
