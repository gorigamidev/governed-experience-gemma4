import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lab/src"))

from lab.ingest.load_graphs import gold_symbols, load_graph, module_path  # noqa: E402
from lab.ingest.load_tasks import Task, parse_patch  # noqa: E402

GRAPH = {"nodes": [
    {"id": "pkg.markup.escape", "name": "pkg.markup.escape", "text": "def escape(markup):\n    return markup.replace('[', '\\\\[')"},
    {"id": "pkg.markup.Tag", "name": "pkg.markup.Tag", "text": "class Tag:\n    def render(self):\n        ..."},
    {"id": "pkg.markup.Tag.render", "name": "pkg.markup.Tag.render", "text": "def render(self):\n        return self.name.upper()"},
], "edges": [{"type": "calls", "source": "pkg.markup.Tag.render", "target": "pkg.markup.escape"}]}
PATCH = """--- a/src/pkg/markup.py
+++ b/src/pkg/markup.py
@@ -5,2 +5,2 @@ class Tag:
     def render(self):
-        return self.name.upper()
+        return escape(self.name.upper())
"""


class GraphTest(unittest.TestCase):
    def test_module_and_gold_symbols(self):
        task = Task("pkg_1", "o/pkg", "abc", "", "", "", PATCH, "")
        with tempfile.TemporaryDirectory() as d:
            Path(d, "pkg_abc.json").write_text(json.dumps(GRAPH))
            nodes, edges = load_graph(Path(d), task)
        self.assertEqual({n.module for n in nodes}, {"pkg/markup.py"})
        self.assertEqual(edges, [("pkg.markup.Tag.render", "pkg.markup.escape")])
        (change,) = parse_patch(PATCH)
        self.assertEqual(module_path(change.path), "pkg/markup.py")
        self.assertEqual(gold_symbols(nodes, change), {"pkg.markup.Tag.render"})

    def test_module_path_prefixes(self):
        self.assertEqual(module_path("docs_src/stream_data/tutorial002.py"), "stream_data/tutorial002.py")
        self.assertEqual(module_path("rich/_unicode_data/__init__.py"), "rich/_unicode_data.py")


if __name__ == "__main__":
    unittest.main()
