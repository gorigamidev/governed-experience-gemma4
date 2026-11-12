import unittest

from lab.analysis.near_miss import bucket, refused_summary


def run(**kw):
    base = {"resolved": False, "files": set(), "gold": {"a.py", "b.py"}, "verifiable": True, "edits_after_budget": 0,
            "error": "", "minutes": 4.0, "arm": "x", "task": "t"}
    return {**base, **kw}


class BucketTest(unittest.TestCase):
    def test_outcomes(self):
        self.assertEqual(bucket(run(resolved=True)), "resolved")
        self.assertEqual(bucket(run()), "no_patch")
        self.assertEqual(bucket(run(files={"c.py"})), "outside_gold")
        self.assertEqual(bucket(run(files={"tests/test_a.py"})), "tests_only")
        self.assertEqual(bucket(run(files={"a.py"})), "part_gold")
        self.assertEqual(bucket(run(files={"a.py", "b.py", "c.py"})), "all_gold")

    def test_refused_summary_ignores_unverifiable_tasks(self):
        text = "\n".join(refused_summary([run(edits_after_budget=1), run(error="Agent exceeded session timeout"),
                                          run(verifiable=False, edits_after_budget=1)]))
        self.assertIn("2 runs: 1 tried an edit", text)
        self.assertIn("1 hit the session timeout", text)


if __name__ == "__main__":
    unittest.main()
