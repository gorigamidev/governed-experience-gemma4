import unittest

from lab.analysis.memory_e4b import locate_stats, placement, rules

GOLD = "--- a/pkg/mod.py\n+++ b/pkg/mod.py\n@@ -40,6 +40,7 @@ def f():\n+    x = 1\n"


def _patch(path, start):
    return f"--- a/{path}\n+++ b/{path}\n@@ -{start},3 +{start},4 @@\n+    y = 2\n"


def _step(name, args, obs=""):
    return {"source": "agent", "tool_calls": [{"function_name": name, "arguments": args}], "observation": obs}


class PlacementTest(unittest.TestCase):
    def test_region_file_miss_empty(self):
        self.assertEqual(placement(_patch("pkg/mod.py", 43), GOLD), "region")
        self.assertEqual(placement(_patch("pkg/mod.py", 200), GOLD), "file")
        self.assertEqual(placement(_patch("pkg/other.py", 40), GOLD), "miss")
        self.assertEqual(placement("", GOLD), "empty")
        # Test-only edits do not count as touching the gold region.
        self.assertEqual(placement(_patch("tests/test_mod.py", 40), GOLD), "miss")


class LocateStatsTest(unittest.TestCase):
    def test_locate_before_edit_and_candidates(self):
        hit = {"content": '{"stdout": "{\\"ok\\":true,\\"items\\":[{\\"file\\":\\"pkg/mod.py\\"}]}"}'}
        empty = {"content": '{"stdout": "{\\"ok\\":true,\\"items\\":[]}"}'}
        trace = {"steps": [_step("run_skill_script", {"args": {"action": "locate", "text": "x"}}, hit),
                           _step("run_skill_script", {"args": {"action": "outline", "file": "pkg/mod.py"}}, hit),
                           _step("run_skill_script", {"args": {"text": "y"}}, empty),
                           _step("edit_file", {"filepath": "pkg/mod.py"}),
                           _step("run_skill_script", {"args": {"action": "locate", "text": "z"}}, hit)]}
        self.assertEqual(locate_stats(trace), (True, 2, 1))
        self.assertEqual(locate_stats({"steps": [_step("edit_file", {})]}), (False, 0, 0))


class RulesTest(unittest.TestCase):
    def _s(self, **kw):
        s = {"n": 24, "misplaced": 10, "R": 3, "infra": 0, "call_cap": 0, "locate_first": 0, "locate_calls": 0,
             "locate_hits": 0, "max_prompt": 17000}
        s.update(kw)
        return s

    def test_rule_3_needs_compliance_and_three_fewer_misplaced(self):
        base = self._s()
        good = self._s(misplaced=7, locate_first=20, locate_calls=22, locate_hits=21)
        self.assertIn("YES", rules(base, good, 24)[2])
        self.assertIn("NO", rules(base, self._s(misplaced=8, locate_first=20, locate_calls=20, locate_hits=20), 24)[2])
        self.assertIn("NO", rules(base, self._s(misplaced=7, R=1, locate_first=20, locate_calls=20, locate_hits=20), 24)[2])
        self.assertEqual(rules(base, self._s(misplaced=2, locate_first=17, locate_calls=20, locate_hits=20), 24)[2],
                         "Rule 3: not read")
        self.assertEqual(rules(base, self._s(misplaced=2, locate_first=20, locate_calls=10, locate_hits=8), 24)[2],
                         "Rule 3: not read")

    def test_rule_5_flags_noise(self):
        out = rules(self._s(), self._s(locate_first=20, locate_calls=20, locate_hits=20), 24, prev=self._s(misplaced=6))
        self.assertIn("within the noise", out[-1])


if __name__ == "__main__":
    unittest.main()
