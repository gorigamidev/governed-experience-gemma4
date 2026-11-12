import json
import unittest

from lab.analysis import lock_report


def skill(payload, stderr=""):
    return {"skill_name": "code", "file_path": "scripts/look.py", "stdout": json.dumps(payload) + "\n",
            "stderr": stderr, "status": "success"}


def step(name, content, args=None):
    if name == "run_skill_script" and args is None:
        args = {"skill_name": "code", "file_path": "scripts/look.py"}
    return {"tool_calls": [{"function_name": name, "arguments": args or {}}],
            "observation": {"content": json.dumps(content) if not isinstance(content, str) else content}}


class LockReportTest(unittest.TestCase):
    def trace(self):
        return {"steps": [
            step("run_skill_script", skill({"ok": True, "content": "x"})),
            step("run_skill_script", skill({"ok": False, "locked": True, "last_read": {}})),
            step("run_skill_script", skill({"ok": True, "locked": True, "reread": True, "content": "x"})),
            step("edit_file", {"status": "ok"}),
            step("run_skill_script", '{"error": "bad", "error_code": "INVALID_ARGUMENTS"}', {"file_path": "scripts/look.py"}),
            step("run_skill_script", '{"error": "Skill', {"skill_name": "code", "file_path": "scripts/look.py"}),
            step("read_file", {"status": "error"}),
            step("submit_patch", {"status": "ok"}),
        ]}

    def test_metrics(self):
        m = lock_report.task_metrics(self.trace())
        self.assertEqual(m["skill_calls"], 5)
        self.assertEqual(m["model_errors"], ["INVALID_ARGUMENTS"])
        self.assertEqual(m["skill_errors"], ["unparseable observation"])
        self.assertEqual(m["undeclared"], ["read_file"])
        self.assertEqual((m["activations"], m["edit_after_lock"], m["rereads"]), (1, 1, 1))
        self.assertEqual((m["first_lock_call"], m["first_edit_call"]), (2, 4))

    def test_no_json_and_traceback_are_errors(self):
        self.assertEqual(lock_report.skill_payload(json.dumps({"stdout": "oops\n"}))[1], "script printed no JSON")
        self.assertEqual(lock_report.skill_payload(json.dumps(skill({"ok": True}, "Traceback (most recent")))[1],
                         "script traceback")

    def test_budget_refusal_is_not_a_skill_error(self):
        # GPU probe rich_3006: the harness refused a well-formed call after 28 calls
        obs = {"skill_name": "code", "file_path": "scripts/look.py", "stdout": "", "status": "error",
               "stderr": '{"status": "error", "error_type": "BudgetExceeded", "error_message": "exhausted"}'}
        m = lock_report.task_metrics({"steps": [step("run_skill_script", obs)]})
        self.assertEqual((m["skill_errors"], m["model_errors"], m["budget_refused"]), ([], [], 1))

    def test_load_reads_instance_id(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "traces").mkdir()
            (Path(d) / "traces/trace_rich_3006.json").write_text(json.dumps({"steps": []}))
            (Path(d) / "task_results.jsonl").write_text(json.dumps({"instance_id": "rich_3006", "resolved": True}) + "\n")
            self.assertTrue(lock_report.load(Path(d))["rich_3006"]["resolved"])

    def test_corrupt_calls_c5(self):
        # probe v3: the 31B closed file_path with a backtick and swallowed skill_name, from the first call on
        bad = {"args": {"search": "x"}, "file_path": "scripts/entrypoint.py`,skill_name:"}
        good = {"skill_name": "repo-tools", "file_path": "scripts/entrypoint.py", "args": {"search": "x"}}
        self.assertTrue(lock_report.corrupt_call(bad))
        self.assertTrue(lock_report.corrupt_call({"skill_name": "repo-tools", "file_path": "x.py`"}))
        self.assertFalse(lock_report.corrupt_call(good))
        m = lock_report.task_metrics({"steps": [step("run_skill_script", "{}", bad), step("run_skill_script", "{}", good)]})
        self.assertEqual((m["corrupt"], m["first_corrupt"]), (1, True))
        base = {"skill_calls": 20, "skill_errors": [], "model_errors": [], "undeclared": [], "ended_by_undeclared": False,
                "activations": 0, "edit_after_lock": 0, "resolved": True, "corrupt": 1, "first_corrupt": False}
        rows = {t: dict(base) for t in lock_report.ALWAYS + lock_report.FLIPS}
        self.assertTrue(lock_report.rules(rows)[1]["C5"])  # 7/140 = 5%
        rows["rich_3006"]["first_corrupt"] = rows["rich_3470"]["first_corrupt"] = True
        self.assertFalse(lock_report.rules(rows)[1]["C5"])

    def test_rules(self):
        base = {"skill_calls": 10, "skill_errors": [], "model_errors": [], "undeclared": [], "ended_by_undeclared": False,
                "activations": 0, "edit_after_lock": 0, "resolved": True}
        rows = {t: dict(base) for t in lock_report.ALWAYS}
        rows.update({t: dict(base, resolved=False, activations=1, edit_after_lock=1) for t in lock_report.FLIPS})
        _, v = lock_report.rules(rows)
        self.assertEqual(v, {"C1": True, "C2": True, "C3": True, "C4": True, "C5": True})
        rows["rich_3006"]["resolved"] = rows["rich_3882"]["resolved"] = False
        rows["rich_3480"]["ended_by_undeclared"] = True
        _, v = lock_report.rules(rows)
        self.assertEqual((v["C2"], v["C3"]), (False, False))


if __name__ == "__main__":
    unittest.main()
