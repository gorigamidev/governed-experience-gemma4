import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "submission/skills/mini-palace/scripts"))
sys.path.insert(0, str(ROOT / "lab/src"))

from indexer import build_index, scan_file  # noqa: E402
from resolver import LexicalIndex, normalize, split_identifier  # noqa: E402

from lab.ingest.load_tasks import parse_patch  # noqa: E402

SAMPLE = '''\
class Console:
    def print_json(self, data):
        pass

    async def _render_buffer(self):
        def inner():
            pass

def escape(markup: str) -> str:
    return markup
'''


class IndexerTest(unittest.TestCase):
    def test_qualnames_and_kinds(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "pkg").mkdir()
            Path(d, "pkg/console.py").write_text(SAMPLE)
            got = {(s.qualname, s.kind) for s in scan_file(d, "pkg/console.py")}
        self.assertEqual(got, {("Console", "class"), ("Console.print_json", "method"),
                               ("Console._render_buffer", "method"),
                               ("Console._render_buffer.inner", "function"), ("escape", "function")})

    def test_ast_ignores_docstring_code_and_non_def_blocks(self):
        source = (
            "class Palette:\n"
            "    def run(self):\n"
            '        """Example:\n\n        def not_a_function():\n            pass\n        """\n'
            "if __name__ == '__main__':\n"
            "    class ColorBox:\n"
            "        async def render(self):\n"
            "            pass\n"
        )
        with tempfile.TemporaryDirectory() as d:
            Path(d, "palette.py").write_text(source)
            got = {s.qualname for s in scan_file(d, "palette.py")}
        self.assertEqual(got, {"Palette", "Palette.run", "ColorBox", "ColorBox.render"})

    def test_line_scan_fallback_on_syntax_error(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "legacy.py").write_text("def ok():\n    print 'py2'\n")
            self.assertEqual([s.qualname for s in scan_file(d, "legacy.py")], ["ok"])

    def test_tests_excluded(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "tests").mkdir()
            Path(d, "tests/test_x.py").write_text("def test_a(): pass\n")
            Path(d, "mod.py").write_text("def a(): pass\n")
            files, _ = build_index(d)
        self.assertEqual(files, ["mod.py"])


class ResolverTest(unittest.TestCase):
    def test_split_and_normalize(self):
        self.assertEqual(split_identifier("HTTPResponse_code"), ["http", "response", "code"])
        self.assertEqual({normalize(w) for w in ("escape", "escaping", "escaped")}, {"escap"})

    def test_ranks_matching_file_first(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "markup.py").write_text(SAMPLE)
            Path(d, "table.py").write_text("class Table:\n    def add_row(self):\n        pass\n")
            idx = LexicalIndex(*build_index(d))
        self.assertEqual(idx.rank_files("Fix escaping trailing backslash")[0].file, "markup.py")
        self.assertEqual(idx.rank_files("`print_json` crashes")[0].file, "markup.py")


class ParsePatchTest(unittest.TestCase):
    def test_new_file_via_zero_hunk(self):
        patch = "--- a/x.py\n+++ b/x.py\n@@ -0,0 +1,2 @@\n+def f():\n+    pass\n"
        (c,) = parse_patch(patch)
        self.assertTrue(c.is_new)
        self.assertEqual(c.defined, {"f"})


if __name__ == "__main__":
    unittest.main()
