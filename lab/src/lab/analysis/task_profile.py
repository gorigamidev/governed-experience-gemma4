"""Profile every task: statement type, reference patch shape and lexical localization signals.

Usage: PYTHONPATH=lab/src python3 -m lab.analysis.task_profile
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from lab.analysis.issue_types import profile_statement
from lab.ingest.load_tasks import load_tasks, parse_patch

ROOT = Path(__file__).resolve().parents[4]
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def statement_tokens(text: str) -> set[str]:
    return {t for t in TOKEN_RE.findall(text) if len(t) >= 3}


def module_path(path: str) -> str:
    p = path.removeprefix("src/").removesuffix(".py").removesuffix("/__init__")
    return p.replace("/", ".")


def profile_task(task) -> dict:
    sp = profile_statement(task.problem_statement)
    changes = parse_patch(task.patch)
    source = [c for c in changes if c.kind == "source"]
    text = task.problem_statement + "\n" + task.hints_text
    lower = text.lower()
    tokens = statement_tokens(text)

    file_mentioned = False
    for c in source:
        stem = c.path.rsplit("/", 1)[-1].removesuffix(".py")
        if c.path.lower() in lower or module_path(c.path).lower() in lower:
            file_mentioned = True
        elif len(stem) >= 4 and stem not in {"__init__", "utils", "main"} and stem.lower() in lower:
            file_mentioned = True

    context_syms = set().union(*(c.context_symbols for c in source)) if source else set()
    new_syms = set().union(*(c.defined for c in source)) if source else set()
    return {
        "instance_id": task.instance_id,
        "repo": task.repo_short,
        "category": sp.category,
        "change_kind": sp.change_kind,
        "words": sp.words,
        "effective_words": sp.effective_words,
        "has_traceback": sp.has_traceback,
        "has_error_name": sp.has_error_name,
        "has_code_block": sp.has_code_block,
        "mentions_test": sp.mentions_test,
        "has_issue_ref": sp.has_issue_ref,
        "has_hints": bool(task.hints_text.strip()),
        "n_inline_code": len(sp.inline_code),
        "n_source_files": len(source),
        "n_test_files": sum(c.kind == "test" for c in changes),
        "n_docs_files": sum(c.kind == "docs" for c in changes),
        "n_other_files": sum(c.kind == "other" for c in changes),
        "only_new_source_files": bool(source) and all(c.is_new for c in source),
        "n_hunks": sum(len(c.hunks) for c in source),
        "churn": sum(c.added + c.removed for c in source),
        "source_files": ";".join(c.path for c in source),
        "context_symbols": ";".join(sorted(context_syms)),
        "file_mentioned": file_mentioned,
        "context_symbol_mentioned": bool(context_syms & tokens),
        "new_symbol_mentioned": bool(new_syms & tokens),
        "any_symbol_mentioned": bool((context_syms | new_syms) & tokens),
    }


def pct(n: int, d: int) -> str:
    return f"{n}/{d} ({100 * n / d:.0f}%)" if d else "0/0"


def summarize(rows: list[dict]) -> str:
    n = len(rows)
    out = [f"# Task profile ({n} tasks)", ""]
    by_repo = defaultdict(list)
    for r in rows:
        by_repo[r["repo"]].append(r)

    def table(title: str, key: str) -> None:
        values = sorted(Counter(r[key] for r in rows), key=lambda v: -sum(r[key] == v for r in rows))
        out.append(f"## {title}")
        out.append("| value | total | " + " | ".join(by_repo) + " |")
        out.append("|---|---|" + "---|" * len(by_repo))
        for v in values:
            cells = [str(sum(r[key] == v for r in rs)) for rs in by_repo.values()]
            out.append(f"| {v} | {pct(sum(r[key] == v for r in rows), n)} | " + " | ".join(cells) + " |")
        out.append("")

    table("Statement category", "category")
    table("Change kind (from title)", "change_kind")

    out.append("## Signals")
    out.append("| signal | total | " + " | ".join(by_repo) + " |")
    out.append("|---|---|" + "---|" * len(by_repo))
    for key in ["has_traceback", "has_error_name", "has_code_block", "mentions_test", "has_issue_ref",
                "has_hints", "only_new_source_files", "file_mentioned", "context_symbol_mentioned",
                "new_symbol_mentioned", "any_symbol_mentioned"]:
        cells = [pct(sum(bool(r[key]) for r in rs), len(rs)) for rs in by_repo.values()]
        out.append(f"| {key} | {pct(sum(bool(r[key]) for r in rows), n)} | " + " | ".join(cells) + " |")
    out.append("")

    out.append("## Patch shape (source files only)")
    for key in ["n_source_files", "n_hunks", "churn", "effective_words"]:
        vals = sorted(r[key] for r in rows)
        out.append(f"- {key}: median {vals[n // 2]}, p90 {vals[int(n * 0.9)]}, max {vals[-1]}, zero {sum(v == 0 for v in vals)}")
    out.append("")

    files = Counter(f for r in rows for f in r["source_files"].split(";") if f)
    out.append("## Most edited source files")
    for f, c in files.most_common(15):
        out.append(f"- {c} × `{f}`")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=ROOT / "data/competition/tasks.jsonl", type=Path)
    ap.add_argument("--out", default=ROOT / "data/analysis", type=Path)
    args = ap.parse_args()

    rows = [profile_task(t) for t in load_tasks(args.tasks)]
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "task_profile.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = summarize(rows)
    (args.out / "task_profile.md").write_text(summary)
    (args.out / "task_profile.json").write_text(json.dumps(rows, indent=1))
    print(summary)


if __name__ == "__main__":
    main()
