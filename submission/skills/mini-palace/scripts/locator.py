"""Pure core of the `locate` action, shared verbatim by the skill, the offline benchmark and consolidation."""

from __future__ import annotations

import patterns as habits
import router
from resolver import LexicalIndex, clean_query, tokenize


def locate_core(idx: LexicalIndex, text: str, repo: str, pattern_list: list[dict], cfg: dict,
                state: dict[str, str] | None = None):
    """Return (decision, shown, n_discarded, ranked, active_pattern_ids).

    `state` is the in-task ledger (file -> status). Files the agent discarded move behind every other
    candidate instead of disappearing, so a wrong "discard" can still be recovered by asking for more items.
    """
    tokens = set(tokenize(clean_query(text)))
    file_boost, symbol_boost, active = habits.boosts(pattern_list, repo, tokens, idx.files,
                                                     scale=cfg.get("pattern_weight", 1.0))
    ranked = idx.rank_files(text, boosts=file_boost, symbol_boosts=symbol_boost)
    if state:
        ranked = [r for r in ranked if state.get(r.file) != "discarded"] + \
                 [r for r in ranked if state.get(r.file) == "discarded"]
        live = [r for r in ranked if state.get(r.file) != "discarded"]
    else:
        live = ranked
    decision, shown, n_discarded = router.decide(live, cfg)
    return decision, shown, n_discarded, ranked, active
