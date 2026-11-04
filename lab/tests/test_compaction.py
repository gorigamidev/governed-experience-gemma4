import unittest

from lab.analysis.compaction import cap_counterfactual, compactions, growth_fit, growth_pairs, timing


def _turn(prompt, *paths):
    calls = [{"function_name": "read_file", "arguments": {"filepath": p}} for p in paths]
    return {"source": "agent", "metrics": str({"prompt_tokens": prompt}), "tool_calls": calls}


class CompactionTest(unittest.TestCase):
    def test_prompt_drop_marks_compaction_and_counts_rereads_after_it(self):
        steps = [_turn(4000, "a.py"), _turn(9000, "b.py"), _turn(15000, "c.py"), _turn(6000, "a.py"),
                 _turn(7000, "d.py")]
        c = compactions({"steps": steps})
        self.assertEqual(c["compaction_calls"], [3])
        self.assertEqual(c["max_prompt"], 15000)
        self.assertEqual(c["after"], {"reads": 2, "rereads": 1})
        self.assertEqual(c["other"], {"reads": 3, "rereads": 0})

    def test_small_dips_are_not_compactions(self):
        c = compactions({"steps": [_turn(10000, "a.py"), _turn(9000, "a.py")]})
        self.assertEqual(c["compaction_calls"], [])
        self.assertEqual(c["other"], {"reads": 2, "rereads": 1})

    def test_timing(self):
        self.assertEqual(timing({"compaction_calls": [], "first_edit_call": 5}), "none")
        self.assertEqual(timing({"compaction_calls": [15], "first_edit_call": None}), "before_edit")
        self.assertEqual(timing({"compaction_calls": [15], "first_edit_call": 20}), "before_edit")
        self.assertEqual(timing({"compaction_calls": [15], "first_edit_call": 9}), "after_edit")

    def test_growth_fit_recovers_coefficients(self):
        pairs = [(c, o, 1.0 * c + 0.5 * o + 10) for c, o in [(100, 50), (300, 20), (50, 400), (800, 100)]]
        fit = growth_fit(pairs)
        self.assertEqual((fit["completion_coef"], fit["observation_coef"], fit["intercept"]), (1.0, 0.5, 10.0))

    def test_growth_pairs_skip_compaction(self):
        steps = [{"source": "agent", "metrics": str({"prompt_tokens": p, "completion_tokens": 10})}
                 for p in (1000, 1100, 500)]
        self.assertEqual(growth_pairs({"steps": steps}), [(10.0, 0.0, 100.0)])

    def test_cap_counterfactual(self):
        def turn(p, c):
            return {"source": "agent", "metrics": str({"prompt_tokens": p, "completion_tokens": c})}
        avoided = cap_counterfactual({"steps": [turn(10000, 4100), turn(14500, 100)]}, threshold=14336)
        self.assertEqual((avoided["excess"], avoided["still_triggers"]), (2000, False))
        kept = cap_counterfactual({"steps": [turn(10000, 2200), turn(14500, 100)]}, threshold=14336)
        self.assertTrue(kept["still_triggers"])


if __name__ == "__main__":
    unittest.main()
