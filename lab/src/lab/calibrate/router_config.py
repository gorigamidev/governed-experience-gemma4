"""Calibrate the router thresholds (RESOLVE/DISCARD) with k-fold cross-validation.

Habits are mined on each training fold with the deployed consolidation parameters, so the rankings the
router sees are out-of-sample. The habit weight itself belongs to lab.consolidate, not to this script.
Selection rule on each training fold, in order:
  1. DISCARD false negatives (gold file ranked but dropped by the threshold) <= MAX_DISCARD_FN
  2. RESOLVE precision (top-1 is gold when deciding RESOLVE) >= MIN_RESOLVE_PRECISION
  3. maximize recall of the files shown, then the RESOLVE rate
The final config is fit on all tasks.

Usage: PYTHONPATH=lab/src lab/.venv/bin/python -m lab.calibrate.router_config [--write]
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

from lab.analysis.localization_bench import router_cfg
from lab.consolidate.patterns import FOLDS, _context, _folds, load_deployed_params, mine

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "submission/skills/mini-palace/scripts"))
import router  # noqa: E402
from locator import locate_core  # noqa: E402

MAX_DISCARD_FN = 0.02
MIN_RESOLVE_PRECISION = 0.80
GRID = {
    "resolve_ratio": [1.3, 1.5, 1.8, 2.2, 3.0],
    "resolve_min_score": [2.0, 4.0, 6.0, 8.0, 12.0],
    "discard_ratio": [0.0, 0.05, 0.1, 0.15, 0.25],
}


def ranked_cases(cases, pats, cfg) -> list[dict]:
    out = []
    for c in cases:
        _, _, _, ranked, _ = locate_core(c.idx, c.task.problem_statement, c.task.repo_short, pats, cfg)
        out.append({"gold": c.gold, "ranked": ranked})
    return out


def score_config(items: list[dict], cfg: dict) -> dict:
    n = len(items)
    shown_hit = discard_fn = resolve_n = resolve_ok = n_items = 0
    for it in items:
        ranked, gold = it["ranked"], it["gold"]
        decision, shown, _ = router.decide(ranked, cfg)
        files = [r.file for r in shown]
        n_items += len(files)
        shown_hit += any(g in files for g in gold)
        top = ranked[0].score if ranked else 0.0
        kept = {r.file for r in ranked if r.score >= cfg["discard_ratio"] * top}
        discard_fn += any(g in {r.file for r in ranked} for g in gold) and not any(g in kept for g in gold)
        if decision == "RESOLVE":
            resolve_n += 1
            resolve_ok += bool(files) and files[0] in gold
    return {"n": n, "shown_recall": shown_hit / n, "discard_fn": discard_fn / n, "resolve_rate": resolve_n / n,
            "resolve_precision": resolve_ok / resolve_n if resolve_n else 1.0, "avg_items": n_items / n}


def select(items: list[dict], base: dict) -> tuple[dict, dict]:
    best = None
    for values in itertools.product(*GRID.values()):
        cfg = {**base, **dict(zip(GRID, values))}
        m = score_config(items, cfg)
        if m["discard_fn"] > MAX_DISCARD_FN or m["resolve_precision"] < MIN_RESOLVE_PRECISION:
            continue
        key = (round(m["shown_recall"], 4), round(m["resolve_rate"], 4))
        if best is None or key > best[0]:
            best = (key, cfg, m)
    return best[1], best[2]


def fmt(m: dict) -> str:
    return (f"shown_recall {100 * m['shown_recall']:.0f}% | discard_fn {100 * m['discard_fn']:.1f}% | "
            f"resolve_rate {100 * m['resolve_rate']:.0f}% @ precision {100 * m['resolve_precision']:.0f}% | "
            f"avg_items {m['avg_items']:.1f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=ROOT / "data/competition/tasks.jsonl", type=Path)
    ap.add_argument("--write", action="store_true", help="write the all-task fit to references/router_config.json")
    args = ap.parse_args()

    cases, exp, terms = _context(args.tasks)
    params = load_deployed_params()
    base = {**router_cfg(), "pattern_weight": params["pattern_weight"]}
    by_id = {c.task.instance_id: c for c in cases}
    folds = _folds(cases, FOLDS)

    pooled_test, pooled_default = [], []
    for i, test in enumerate(folds):
        train = [t for j, f in enumerate(folds) if j != i for t in f]
        pats = mine(train, exp, terms, params["min_support"], params["min_lift"], params["kinds"],
                    params.get("min_support_symbol"))
        cfg, _ = select(ranked_cases([by_id[t] for t in train], pats, base), base)
        test_items = ranked_cases([by_id[t] for t in test], pats, base)
        pooled_test.append(score_config(test_items, cfg))
        pooled_default.append(score_config(test_items, {**base, **router.DEFAULTS, "pattern_weight": base["pattern_weight"]}))
        print(f"fold {i}: { {k: cfg[k] for k in GRID} } | test {fmt(pooled_test[-1])}")

    def pool(ms):
        total = sum(m["n"] for m in ms)
        return {k: sum(m[k] * m["n"] for m in ms) / total for k in ms[0] if k != "n"}
    calibrated = pool(pooled_test)
    print(f"\nCV pooled, calibrated: {fmt(calibrated)}")
    print(f"CV pooled, defaults:   {fmt(pool(pooled_default))}")

    if calibrated["discard_fn"] > MAX_DISCARD_FN or calibrated["resolve_precision"] < MIN_RESOLVE_PRECISION:
        # The calibrated thresholds do not hold their own constraints out of sample: keep the conservative defaults.
        cfg = {**base, **{k: router.DEFAULTS[k] for k in GRID}}
        print(f"calibration fails its constraints out of sample -> keeping defaults { {k: cfg[k] for k in GRID} }")
    else:
        all_pats = mine(list(by_id), exp, terms, params["min_support"], params["min_lift"], params["kinds"],
                        params.get("min_support_symbol"))
        # In-sample habits inflate scores; this fit only picks thresholds, the estimate is the CV above.
        cfg, m = select(ranked_cases(cases, all_pats, base), base)
        print(f"all-task fit: { {k: cfg[k] for k in GRID} } | {fmt(m)}")
    if args.write:
        out = ROOT / "submission/skills/mini-palace/references/router_config.json"
        out.write_text(json.dumps({k: cfg[k] for k in (*GRID, "max_items", "pattern_weight")}, indent=1) + "\n")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
