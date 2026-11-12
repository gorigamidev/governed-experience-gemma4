import json
import tempfile
import unittest
from pathlib import Path

from lab.analysis.self_imitation import compare, load_rows, run_features, signals


def run(cmd):
    return {"tool_calls": [{"function_name": "run_command", "arguments": {"command": cmd}}],
            "observation": {"content": '{"status": "ok", "exit_code": 0}'}}


def edit(ok=True, s="x"):
    content = '{"status": "ok"}' if ok else '{"status": "error", "error_type": "FileEditError"}'
    return {"tool_calls": [{"function_name": "edit_file", "arguments": {"old_string": s}}], "observation": {"content": content}}


def submit():
    return {"tool_calls": [{"function_name": "submit_patch", "arguments": {}}], "observation": {"content": "{}"}}


class RunFeaturesTest(unittest.TestCase):
    def test_repetition_without_rejected_calls(self):
        bad = {"tool_calls": [{"function_name": "edit_file", "arguments": {"x": 1}}],
               "observation": {"content": "mandatory input parameters are not present"}}
        f = run_features({"steps": [run("ls"), bad, bad, bad, run("ls")]})
        self.assertEqual((f["same_after_same"], f["valid_same_after_same"], f["valid_same_after_diff"]), ([True, False], [], []))

    def test_early_and_final_test_around_the_successful_edits(self):
        t = {"steps": [run("timeout 90 python -m pytest tests/test_a.py -x -q"), edit(False), edit(True), edit(True, "y"),
                       run("timeout 90 pytest tests/test_a.py"), submit(), run("pytest after submit")]}
        f = run_features(t)
        self.assertEqual((f["edited"], f["first_edit"], f["early_test"], f["final_test"]), (True, 3, True, True))
        self.assertEqual((f["first_cmd_timeout"], f["later_cmd_timeout"]), (True, [True, False]))

    def test_a_test_after_the_first_edit_is_not_early_and_no_edit_gives_none(self):
        f = run_features({"steps": [run("grep -n x a.py"), edit(True), run("pytest tests"), edit(True, "y"), submit()]})
        self.assertEqual((f["early_test"], f["final_test"]), (False, False))
        f = run_features({"steps": [run("pytest"), edit(False), submit()]})
        self.assertEqual((f["edited"], f["early_test"], f["final_test"]), (False, None, None))

    def test_identical_consecutive_calls(self):
        f = run_features({"steps": [run("ls"), run("ls"), run("ls"), run("pwd"), run("cat a")]})
        # pairs: same, same, diff, diff -> after same: [same, diff]; after diff: [diff]
        self.assertEqual((f["same_after_same"], f["same_after_diff"]), ([True, False], [False]))


def row(task, g, o, verifiable=True, first_edit=5):
    return {"task": task, "edited": True, "early_test": g, "final_test": o, "resolved": o, "verifiable": verifiable,
            "first_edit": first_edit, "source": "s", "first_cmd_timeout": None, "later_cmd_timeout": [],
            "same_after_same": [], "same_after_diff": []}


class SignalsTest(unittest.TestCase):
    def test_compare_counts_per_task_signs_only_where_both_groups_exist(self):
        rows = [row("a", True, True), row("a", False, False), row("b", True, False), row("b", False, True),
                row("c", True, True), row("c", False, True), row("d", True, True)]
        c = compare(rows, "early_test", "final_test")
        self.assertEqual((c["n1"], c["n0"], c["tasks_pos"], c["tasks_neg"], c["tasks_tie"]), (4, 3, 1, 1, 1))
        self.assertAlmostEqual(c["diff"], 0.75 - 2 / 3)

    def test_s1_needs_the_pooled_gap_the_group_size_and_the_per_task_majority(self):
        rows = []
        for i in range(8):
            rows += [row(f"t{i}", True, True), row(f"t{i}", True, True), row(f"t{i}", False, False),
                     row(f"t{i}", False, False)]
        s = signals(rows)
        self.assertTrue(s["s1"] and s["s2"])
        self.assertIn("S1 and S2", s["decision"])
        self.assertFalse(signals(rows[:20])["s1"])  # 10 per group < 15

    def test_neither_signal_archives(self):
        rows = [row(f"t{i}", g, o) for i in range(8) for g in (True, False) for o in (True, False)]
        for r in rows:
            r["resolved"] = r["task"] in ("t0", "t1")  # resolution independent of the final test
        s = signals(rows)
        self.assertFalse(s["s1"] or s["s2"])
        self.assertIn("archived", s["decision"])


class LoadRowsTest(unittest.TestCase):
    def test_reads_results_and_traces_and_fails_loudly_on_a_missing_source(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "run"
            (src / "traces").mkdir(parents=True)
            (src / "task_results.jsonl").write_text(json.dumps({"instance_id": "rich_1", "resolved": True}) + "\n")
            (src / "traces" / "trace_rich_1.json").write_text(json.dumps({"steps": [edit(True), run("pytest"), submit()]}))
            rows = load_rows(Path(d), {"rich_1": True}, corpus=("run",))
            self.assertEqual((len(rows), rows[0]["resolved"], rows[0]["final_test"], rows[0]["verifiable"]),
                             (1, True, True, True))
            with self.assertRaises(FileNotFoundError):
                load_rows(Path(d), {}, corpus=("missing",))


if __name__ == "__main__":
    unittest.main()
