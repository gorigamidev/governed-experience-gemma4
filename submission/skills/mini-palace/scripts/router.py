"""Router v0: RESOLVE / ESCALATE / DISCARD over ranked files.

Defaults below are fallbacks; the shipped values come from lab/calibrate/router_config.py (k-fold CV). DISCARD never hides a file from
the agent for good: discarded candidates are only left out of this answer.
"""

from __future__ import annotations

DEFAULTS = {"pattern_weight": 1.0, "resolve_ratio": 1.8, "resolve_min_score": 4.0, "discard_ratio": 0.15, "max_items": 5}


def decide(ranked: list, config: dict | None = None) -> tuple[str, list, int]:
    """Return (decision, shown_items, n_discarded)."""
    cfg = {**DEFAULTS, **(config or {})}
    if not ranked:
        return "ESCALATE", [], 0
    top = ranked[0].score
    second = ranked[1].score if len(ranked) > 1 else 0.0
    kept = [r for r in ranked if r.score >= cfg["discard_ratio"] * top]
    n_discarded = len(ranked) - len(kept)
    if top >= cfg["resolve_min_score"] and (second == 0 or top / second >= cfg["resolve_ratio"]):
        # Top-1 is right only ~53% of the time (file-level bench), so RESOLVE still shows two alternates.
        return "RESOLVE", kept[:3], n_discarded
    return "ESCALATE", kept[: cfg["max_items"]], n_discarded
