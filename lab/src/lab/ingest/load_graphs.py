"""Precomputed per-commit code graphs: symbols at each task's exact base_commit, without snapshots.

Graph nodes are {"name": dotted qualname, "text": source}. Names drop the "src/" and "docs_src/"
prefixes, so files are compared in a shared module space: "src/requests/utils.py" -> "requests/utils.py".
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from lab.ingest.load_tasks import FileChange, Task

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "submission/skills/mini-palace/scripts"))
from indexer import module_path  # noqa: E402,F401

MIN_MATCH_LEN = 12  # shorter code lines ("return x", "pass") match too many symbols


@dataclass
class GraphNode:
    name: str
    module: str  # module path in shared module space, e.g. "rich/markup.py"
    text: str

    @property
    def short(self) -> str:
        return self.name.rsplit(".", 1)[-1]

    @property
    def local_name(self) -> str:
        """Qualname inside its module ("rich.markup.Tag.render" -> "Tag.render"), as the runtime indexer sees it."""
        prefix = self.module.removesuffix(".py").replace("/", ".") + "."
        return self.name[len(prefix):] if self.name.startswith(prefix) else self.name


def graph_file(graphs_dir: Path, task: Task) -> Path:
    return graphs_dir / f"{task.repo_short}_{task.base_commit}.json"


@lru_cache(maxsize=8)
def _load(path: str) -> tuple[list[GraphNode], list[tuple[str, str]]]:
    with open(path, encoding="utf-8") as f:
        g = json.load(f)
    names = {n["name"] for n in g["nodes"]}
    nodes = []
    for n in g["nodes"]:
        # Strip the node's own segment, then enclosing classes/functions (which are nodes themselves).
        mod = n["name"].rsplit(".", 1)[0]
        while mod in names and "." in mod:
            mod = mod.rsplit(".", 1)[0]
        nodes.append(GraphNode(n["name"], mod.replace(".", "/") + ".py", n.get("text", "")))
    edges = [(e["source"], e["target"]) for e in g["edges"]]
    return nodes, edges


def load_graph(graphs_dir: Path, task: Task) -> tuple[list[GraphNode], list[tuple[str, str]]]:
    return _load(str(graph_file(graphs_dir, task)))


def gold_symbols(nodes: list[GraphNode], change: FileChange) -> set[str]:
    """Innermost graph nodes of the changed module whose source contains a line the patch touches."""
    module = module_path(change.path)
    candidates = [n for n in nodes if n.module == module]
    found = set()
    for hunk in change.hunks:
        lines = [ln.strip() for ln in (hunk.removed or hunk.old_lines)]
        for line in lines:
            if len(line) < MIN_MATCH_LEN:
                continue
            hits = [n for n in candidates if line in n.text]
            if hits:
                found.add(max(hits, key=lambda n: n.name.count(".")).name)
    return found
