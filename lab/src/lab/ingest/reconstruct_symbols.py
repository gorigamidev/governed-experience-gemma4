"""Exact per-task symbol tables (async included) rebuilt from one newer snapshot per repo, no extra downloads.

Snapshots carry history with rewritten hashes, so base_commit cannot be looked up directly. Instead:
  1. the commit whose committer date equals the task's created_at (+-2 s) is the fix commit;
  2. its first parent is the base, and the reference patch must apply there (git apply --recount);
  3. the base tree's sync defs must match the precomputed graph for the task (the graphs omit async).
The result is written to data/symbols/<instance_id>.json; the git objects are only read, never checked out.

Usage: PYTHONPATH=lab/src python3 -m lab.ingest.reconstruct_symbols [--tasks-filter fastapi]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from lab.ingest.load_graphs import load_graph, module_path
from lab.ingest.load_tasks import Task, load_tasks

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "submission/skills/mini-palace/scripts"))
from indexer import extract_symbols as runtime_extract  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
DATE_TOLERANCE_S = 2


class Repo:
    def __init__(self, path: Path):
        self.path = str(path)
        log = self.git("log", "--all", "--format=%H %ct %P").stdout.splitlines()
        self.commits = [(parts[0], int(parts[1]), parts[2:]) for parts in (ln.split() for ln in log) if len(parts) >= 2]

    def git(self, *args, env=None, input_text=None) -> subprocess.CompletedProcess:
        return subprocess.run(["git", "-C", self.path, *args], capture_output=True, text=True, env=env,
                              input=input_text)

    def fix_candidates(self, created_at: str) -> list[tuple[str, str]]:
        ts = int(datetime.fromisoformat(created_at.replace("Z", "+00:00")).timestamp())
        # Every parent: when the fix is a merge, the base can be the second parent (e.g. rich_3052).
        return [(sha, parent) for sha, ct, parents in self.commits
                if abs(ct - ts) <= DATE_TOLERANCE_S for parent in parents]

    def patch_applies(self, commit: str, patch: str) -> bool:
        """Same fallback order as the harness: strict git apply, then GNU patch with fuzz.

        GNU patch runs on a temp dir holding only the files the patch touches, so nothing is checked out.
        """
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "fix.patch")
            Path(p).write_text(patch if patch.endswith("\n") else patch + "\n")
            env = {**os.environ, "GIT_INDEX_FILE": os.path.join(d, "index")}
            if self.git("read-tree", commit, env=env).returncode != 0:
                return False
            if self.git("apply", "--cached", "--check", "--recount", p, env=env).returncode == 0:
                return True
            tree = os.path.join(d, "tree")
            for rel in {m.group(1) for m in re.finditer(r"^--- a/(\S+)", patch, re.M)}:
                blob = self.git("show", f"{commit}:{rel}")
                if blob.returncode != 0:
                    continue  # file created by the patch
                os.makedirs(os.path.dirname(os.path.join(tree, rel)), exist_ok=True)
                Path(tree, rel).write_text(blob.stdout)
            os.makedirs(tree, exist_ok=True)
            res = subprocess.run(["patch", "-p1", "--dry-run", "--batch", "-d", tree, "-i", p],
                                 capture_output=True, text=True)
            return res.returncode == 0

    def py_sources(self, commit: str) -> dict[str, str]:
        listing = self.git("ls-tree", "-r", "--format=%(objectname) %(path)", commit).stdout.splitlines()
        blobs = [(ln.split(" ", 1)[0], ln.split(" ", 1)[1]) for ln in listing if ln.endswith(".py")]
        batch = self.git("cat-file", "--batch", input_text="\n".join(b for b, _ in blobs) + "\n").stdout
        sources, pos = {}, 0
        for _, path in blobs:
            header_end = batch.index("\n", pos)
            size = int(batch[pos:header_end].split()[2])
            start = header_end + 1
            # Sizes are in bytes; slice on the encoded text to stay exact with non-ASCII sources.
            raw = batch[start:].encode("utf-8")[:size].decode("utf-8", errors="replace")
            sources[path] = raw
            pos = start + len(raw) + 1
        return sources


def extract_symbols(path: str, source: str) -> list[list]:
    return runtime_extract(source) or []


def graph_agreement(task: Task, graphs: Path, symbols: dict[str, list]) -> dict:
    """Compare sync defs against the graph (which has no async nodes).

    Matching is on (module, short name): the graph flattens nested definitions (APIRouter.api_route.decorator
    appears as APIRouter.decorator), so full qualnames differ by convention, not by content.
    """
    nodes, _ = load_graph(graphs, task)
    graph_keys = {(n.module, n.short) for n in nodes}
    ours = set()
    for path, syms in symbols.items():
        mod = module_path(path)
        async_prefixes = [s[0] for s in syms if s[2]]
        for q, _kind, is_async, _s, _e in syms:
            if not is_async and not any(q.startswith(a + ".") for a in async_prefixes):
                ours.add((mod, q.rsplit(".", 1)[-1]))
    common = graph_keys & ours
    return {"graph_nodes": len(graph_keys), "matched": len(common),
            "graph_recall": round(len(common) / max(1, len(graph_keys)), 4)}


MIN_GRAPH_RECALL = 0.98


def reconstruct(task: Task, repo: Repo, graphs: Path, own_snapshot: bool) -> dict:
    """Pick the candidate base where the patch applies and the graph agrees best.

    status is "exact" when sync defs match the graph at >= MIN_GRAPH_RECALL, "approximate" otherwise
    (e.g. the base lived on a PR branch the exported history does not contain).
    """
    # A task's own snapshot stops at its base_commit, so the fix commit is not in its history.
    candidates = [(None, "HEAD")] if own_snapshot else repo.fix_candidates(task.created_at)
    best = None
    for fix, base in candidates:
        if not repo.patch_applies(base, task.patch):
            continue
        sources = repo.py_sources(base)
        symbols = {p: extract_symbols(p, src) for p, src in sources.items()}
        check = graph_agreement(task, graphs, symbols)
        if best is None or check["graph_recall"] > best[3]["graph_recall"]:
            best = (fix, base, (sources, symbols), check)
    if best is None:
        return {"instance_id": task.instance_id, "base": None, "status": "not_found",
                "verification": {"patch_applies": False}}
    fix, base, (sources, symbols), check = best
    return {"instance_id": task.instance_id, "base": repo.git("rev-parse", base).stdout.strip(), "fix": fix,
            "status": "exact" if check["graph_recall"] >= MIN_GRAPH_RECALL else "approximate",
            "files": sorted(sources), "symbols": symbols, "verification": {"patch_applies": True, **check}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=ROOT / "data/competition/tasks.jsonl", type=Path)
    ap.add_argument("--graphs", default=ROOT / "data/competition/graphs", type=Path)
    ap.add_argument("--repos", default=ROOT / "data/repos", type=Path)
    ap.add_argument("--out", default=ROOT / "data/symbols", type=Path)
    ap.add_argument("--tasks-filter", default="")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    repos: dict[str, Repo] = {}
    ok = 0
    tasks = [t for t in load_tasks(args.tasks) if args.tasks_filter in t.instance_id]
    for t in tasks:
        snap = max(args.repos.glob(f"{t.repo_short}_*"), key=lambda p: int(p.name.rsplit("_", 1)[1]), default=None)
        if snap is None:
            print(f"{t.instance_id}: no snapshot for {t.repo_short}")
            continue
        own = args.repos / t.instance_id
        if own.is_dir():
            res = reconstruct(t, Repo(own), args.graphs, own_snapshot=True)
        else:
            repos.setdefault(t.repo_short, Repo(snap))
            res = reconstruct(t, repos[t.repo_short], args.graphs, own_snapshot=False)
        (args.out / f"{t.instance_id}.json").write_text(json.dumps(res, separators=(",", ":")))
        v = res["verification"]
        ok += res["status"] == "exact"
        n_async = sum(s[2] for syms in res.get("symbols", {}).values() for s in syms)
        print(f"{t.instance_id}: " + (f"{res['status']} base {res['base'][:10]} | graph recall {v['graph_recall']:.3f} "
                                       f"({v['matched']}/{v['graph_nodes']}) | async defs {n_async}"
                                       if res["base"] else "NOT FOUND"))
    print(f"\nexact {ok}/{len(tasks)}")


if __name__ == "__main__":
    main()
