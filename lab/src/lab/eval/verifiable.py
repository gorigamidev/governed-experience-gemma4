"""The verifiable set: tasks whose reference patch passes and whose empty patch fails in our Docker environment.

Both registries are written by lab.eval.verify_patches (--gold and --empty). A task that passes with no fix would
count any patch as resolved, so it is excluded even when its reference patch passes.
"""

from __future__ import annotations

import json
from pathlib import Path

ANALYSIS = Path(__file__).resolve().parents[4] / "data/analysis"
GOLD = ANALYSIS / "verifiable_tasks.json"
EMPTY = ANALYSIS / "empty_patch_passes.json"


def load_verifiable(gold: Path = GOLD, empty: Path = EMPTY) -> dict[str, bool] | None:
    if not gold.exists():
        return None
    passes_gold = json.loads(gold.read_text())
    passes_empty = json.loads(empty.read_text()) if empty.exists() else {}
    return {iid: ok and not passes_empty.get(iid, False) for iid, ok in passes_gold.items()}
