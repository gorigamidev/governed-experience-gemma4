"""Experience store on LINALDB: runs, experiences, consolidated patterns and their evidence, with lineage.

Offline plane only (needs the lab venv). The runtime never imports this module.
"""

from __future__ import annotations

import os
from pathlib import Path

import linaldb

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DIR = ROOT / "data/store"

SCHEMA = {
    "task": "task_id: String, repo: String, base_commit: String, category: String, change_kind: String",
    # kind: reference_patch (expert demonstrations) | agent_eval (swegemma eval) | consolidation
    "run": "run_id: String, kind: String, created: String, source: String, note: String",
    # outcome: demonstration | resolved | unresolved. symbol is NULL for file-level rows.
    "experience": "run_id: String, task_id: String, file: String, symbol: String?, outcome: String",
    "pattern": ("pattern_id: String, version: Int, run_id: String, kind: String, repo: String, term: String?, "
                "target_file: String, target_symbol: String?, weight: Float, support: Int, activations: Int, "
                "precision: Float, lift: Float"),
    "pattern_evidence": "pattern_id: String, version: Int, task_id: String, hit: Bool",
    # Agent runs (swegemma eval): one row per task, and which consolidated habits locate reported active.
    "task_result": ("run_id: String, task_id: String, resolved: Bool, error: String?, duration_s: Float, "
                    "tool_calls: Int, patch_size: Int, skill_calls: Int, skill_first: Bool, n_habits: Int"),
    "habit_activation": "run_id: String, task_id: String, pattern_id: String, resolved: Bool",
}


def literal(value) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    # LINALDB strings cannot escape double quotes.
    return '"' + str(value).replace('"', "'") + '"'


class ExperienceStore:
    def __init__(self, data_dir: str | Path = DEFAULT_DIR):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db = linaldb.Db(str(self.data_dir))
        for name, columns in SCHEMA.items():
            if os.path.isdir(self.db.dataset_dir(name)):
                self.db.execute(f"LOAD DATASET {name}")
            else:
                self.db.execute(f"DATASET {name} COLUMNS ({columns})")

    def insert(self, table: str, row: dict) -> None:
        assigns = ", ".join(f"{k} = {literal(v)}" for k, v in row.items())
        self.db.execute(f"INSERT INTO {table} ({assigns})")

    def query(self, dsl: str) -> list[dict]:
        res = self.db.execute(dsl)
        return [dict(zip(res.columns, r)) for r in res.rows]

    def execute(self, dsl: str):
        return self.db.execute(dsl)

    def has_run(self, run_id: str) -> bool:
        return bool(self.query(f"SELECT run_id FROM run WHERE run_id = {literal(run_id)}"))

    def delete_run(self, table: str, run_id: str) -> None:
        self.db.execute(f"DELETE FROM {table} WHERE run_id = {literal(run_id)}")

    def snapshot(self, name: str, source: str, where: str) -> None:
        """Materialize a filtered view as its own dataset, so EXPLAIN LINEAGE ties it to its source."""
        self.db.execute(f"DATASET {name} FROM {source} FILTER {where}")
        self.db.execute(f"SAVE DATASET {name}")

    def lineage(self, name: str) -> str:
        return self.db.execute(f"EXPLAIN LINEAGE {name} AS JSON")

    def save(self) -> None:
        for name in SCHEMA:
            self.db.execute(f"SAVE DATASET {name}")
