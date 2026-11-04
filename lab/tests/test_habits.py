import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "submission/skills/mini-palace/scripts"))
sys.path.insert(0, str(ROOT / "lab/src"))

import patterns as habits  # noqa: E402
from indexer import Symbol  # noqa: E402
from locator import locate_core  # noqa: E402
from resolver import LexicalIndex  # noqa: E402

from lab.consolidate.patterns import mine, paired_permutation_p, sign_test_p  # noqa: E402

PATTERNS = [
    {"id": "Pa", "v": 1, "kind": "file_prior", "repo": "demo", "term": None, "file": "demo/core.py", "symbol": None, "w": 2.0},
    {"id": "Pb", "v": 1, "kind": "symbol_prior", "repo": "demo", "term": None, "file": "demo/core.py",
     "symbol": "Engine.run", "w": 5.0},
    {"id": "Pc", "v": 1, "kind": "file_term", "repo": "demo", "term": "cach", "file": "demo/cache.py", "symbol": None, "w": 1.0},
    {"id": "Pd", "v": 1, "kind": "file_prior", "repo": "other", "term": None, "file": "demo/cache.py", "symbol": None, "w": 9.0},
]


class RuntimeHabitsTest(unittest.TestCase):
    def test_activation_is_scoped_to_repo_and_term(self):
        files = ["src/demo/core.py", "src/demo/cache.py"]
        fb, sb, active = habits.boosts(PATTERNS, "demo", {"run"}, files)
        self.assertEqual(active, ["Pa", "Pb"])
        self.assertEqual(fb, {"src/demo/core.py": 2.0})
        self.assertEqual(sb, {"src/demo/core.py": {"Engine.run": 5.0}})
        _, _, active = habits.boosts(PATTERNS, "demo", {"cach"}, files)
        self.assertIn("Pc", active)
        self.assertEqual(habits.boosts(PATTERNS, "unknown", {"cach"}, files)[2], [])

    def test_symbol_habit_reorders_symbols_not_files(self):
        idx = LexicalIndex(["src/demo/core.py", "src/demo/cache.py"],
                           [Symbol("Engine", "src/demo/core.py", 1, "class"),
                            Symbol("Engine.run", "src/demo/core.py", 5, "method"),
                            Symbol("Engine.stop_engine", "src/demo/core.py", 9, "method"),
                            Symbol("Cache", "src/demo/cache.py", 1, "class")])
        _, _, _, ranked, active = locate_core(idx, "engine stop", "demo", PATTERNS, {"pattern_weight": 1.0})
        self.assertEqual(ranked[0].file, "src/demo/core.py")
        self.assertEqual(ranked[0].top_symbols[0], "Engine.run")
        self.assertIn("Pb", active)

    def test_detect_repo(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "pyproject.toml").write_text('[project]\nname = "FastAPI"\nversion = "1"\n')
            self.assertEqual(habits.detect_repo(d), "fastapi")
            Path(d, "pyproject.toml").unlink()
            self.assertEqual(habits.detect_repo(d), "")


class MiningTest(unittest.TestCase):
    def test_support_and_lift(self):
        exp = {f"t{i}": {"repo": "demo", "files": {"demo/a.py"} if i < 4 else {"demo/b.py"},
                         "syms": {"demo/a.py::f"} if i < 3 else set()} for i in range(8)}
        terms = {f"t{i}": {"alpha"} if i < 3 else {"beta"} for i in range(8)}
        pats = mine(list(exp), exp, terms, min_support=3, min_lift=1.5,
                    kinds=("file_prior", "symbol_prior", "file_term"))
        by = {(p["kind"], p["term"], p["file"], p["symbol"]): p for p in pats}
        self.assertEqual(by[("file_prior", None, "demo/a.py", None)]["support"], 4)
        self.assertEqual(by[("symbol_prior", None, "demo/a.py", "f")]["evidence"], ["t0", "t1", "t2"])
        term = by[("file_term", "alpha", "demo/a.py", None)]
        self.assertAlmostEqual(term["lift"], (3 / 3) / (4 / 8))
        self.assertNotIn(("file_prior", None, "demo/b.py", None), {k for k in by if k[2] == "demo/b.py" and by[k]["support"] < 3})

    def test_symbol_support_is_independent_of_file_support(self):
        exp = {f"t{i}": {"repo": "demo", "files": {"demo/a.py"}, "syms": {"demo/a.py::f"} if i < 2 else set()}
               for i in range(5)}
        kinds = ("file_prior", "symbol_prior")
        strict = mine(list(exp), exp, {}, min_support=5, min_lift=2.0, kinds=kinds)
        self.assertEqual({p["kind"] for p in strict}, {"file_prior"})
        loose = mine(list(exp), exp, {}, min_support=5, min_lift=2.0, kinds=kinds, min_support_symbol=2)
        self.assertEqual({p["kind"] for p in loose}, {"file_prior", "symbol_prior"})

    def test_paired_permutation(self):
        self.assertEqual(paired_permutation_p([0, 0, 0]), 1.0)
        self.assertGreater(paired_permutation_p([1, -1] * 20), 0.5)
        self.assertLess(paired_permutation_p([1.0] * 15 + [-0.1] * 2), 0.01)

    def test_sign_test(self):
        self.assertAlmostEqual(sign_test_p(10, 2), 0.0386, places=3)
        self.assertEqual(sign_test_p(0, 0), 1.0)

class TemporalSplitTest(unittest.TestCase):
    def test_only_strictly_earlier_tasks_train(self):
        from lab.analysis.temporal_split import earlier, paired

        created = {"a": "2026-01-01", "b": "2026-02-01", "c": "2026-02-01", "d": "2026-03-01"}
        self.assertEqual(earlier(created, "a"), [])
        self.assertEqual(sorted(earlier(created, "c")), ["a"])  # same timestamp is not "before"
        self.assertEqual(sorted(earlier(created, "d")), ["a", "b", "c"])
        base = {"x": {"rank": 3}, "y": {"rank": None}, "z": {"rank": 1}}
        new = {"x": {"rank": 1}, "y": {"rank": 7}, "z": {"rank": 2}}
        self.assertEqual(paired(base, new, ["x", "y", "z"]), (2, 1))

if __name__ == "__main__":
    unittest.main()
