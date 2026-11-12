import json
import unittest

from lab.analysis.authority_phase_b import locate_files, task_row, verdict


def call(name, args, content='{"status": "ok"}'):
    return {"tool_calls": [{"function_name": name, "arguments": args}], "observation": {"content": content}}


LOC = json.dumps({"stdout": json.dumps({"ok": True, "items": [{"file": "a.py"}, {"file": "g.py"}]})})


def row(gold, empty=False):
    return {"gold": gold, "empty": empty, "locate_first": True, "max_prompt": 1000, "gold_rank": None}


class PhaseBTest(unittest.TestCase):
    def test_locate_files_and_failed_call(self):
        self.assertEqual(locate_files(LOC), ["a.py", "g.py"])
        self.assertIsNone(locate_files('{"error": "Argument \'skill_name\' is required."}'))

    def test_task_row(self):
        t = {"steps": [call("run_skill_script", {"args": {"text": "bug"}}, LOC),
                       call("edit_file", {"filepath": "/workspace/g.py"})]}
        r = task_row(t, {"g.py"}, False)
        self.assertEqual((r["gold"], r["locate_first"], r["gold_rank"]), (True, True, 2))
        r = task_row({"steps": [call("edit_file", {"filepath": "g.py"}), call("run_skill_script", {"args": {"text": "x"}}, LOC)]},
                     {"g.py"}, True)
        self.assertFalse(r["locate_first"])

    def test_rules(self):
        base = {f"t{i}": row(i < 3, empty=i < 4) for i in range(15)}
        mp = {f"t{i}": row(i < 10, empty=i < 2) for i in range(15)}
        v = verdict(base, mp, 0)
        self.assertTrue(v["b1"] and v["b2"] and not v["b3"])
        self.assertEqual((v["only_base"], v["only_mp"], v["p"]), (0, 7, 0.0156))
        self.assertFalse(verdict(base, mp, 1)["b2"])  # B1 fails: B2 is not read


if __name__ == "__main__":
    unittest.main()
