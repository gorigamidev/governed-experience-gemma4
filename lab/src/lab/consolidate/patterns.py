"""Consolidate experience into habits: mine, verify with nested cross-validation, deploy with lineage.

A habit = activation (repo [+ statement term]) -> effect (ranking boost on a file or a symbol),
carrying its evidence (support, activations, precision, lift, task ids). Candidate sets are mined
from the experience store (LINALDB); a set is deployed only if it beats lexical-only ranking out of sample.

Usage: PYTHONPATH=lab/src lab/.venv/bin/python -m lab.consolidate.patterns [--deploy]
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from lab.analysis.localization_bench import Case, build_case, evaluate_case, router_cfg, summarize
from lab.ingest.load_tasks import load_tasks

ROOT = Path(__file__).resolve().parents[4]
REFS = ROOT / "submission/skills/mini-palace/references"
sys.path.insert(0, str(ROOT / "submission/skills/mini-palace/scripts"))
from resolver import clean_query, tokenize  # noqa: E402

ALL_KINDS = ("file_prior", "symbol_prior", "file_term", "symbol_term")
GRID = {
    "min_support": [2, 3, 5],  # file-level habits
    "min_support_symbol": [2, 3, 5],
    "min_lift": [2.0],
    "kinds": [("file_prior",), ("file_prior", "symbol_prior"), ("file_prior", "symbol_prior", "symbol_term"), ALL_KINDS],
    "pattern_weight": [1.0, 2.0, 3.0],
}
# Which habit kinds each parameter belongs to. A ladder step only searches the parameters of the kinds it
# adds; the others stay at the values each fold chose when those kinds were accepted.
PARAM_KINDS = {
    "min_support": {"file_prior", "file_term"},
    # Scales every habit boost; owned by the file kinds so adding symbol habits cannot move file ranks.
    "pattern_weight": {"file_prior", "file_term"},
    "min_support_symbol": {"symbol_prior", "symbol_term"},
    "min_lift": {"file_term", "symbol_term"},
}
# Rule history is part of the lineage: v1 re-selected every parameter at every step. Fixing the accepted
# ones was motivated by the v2 run (a symbol step rejected on MRR noise from re-selected file parameters),
# so the new rule first takes effect at the next consolidation that includes new experience.
LADDER_RULE = "fixed-accepted-params/2026-09-28"
MAX_TERM_DF = 0.5  # terms present in more than half of a repo's tasks carry no signal
FOLDS = 5
TRUSTED_RUN_KINDS = ("reference_patch", "agent_eval")


# ---------------------------------------------------------------- experience

def load_experience(store, run_kinds=TRUSTED_RUN_KINDS, outcomes=("demonstration", "resolved")) -> dict:
    kinds = ", ".join(f'"{k}"' for k in run_kinds)
    outs = ", ".join(f'"{o}"' for o in outcomes)
    rows = store.query(
        "SELECT e.task_id, t.repo, e.file, e.symbol FROM experience e "
        "JOIN run r ON e.run_id = r.run_id JOIN task t ON e.task_id = t.task_id "
        f"WHERE r.kind IN ({kinds}) AND e.outcome IN ({outs})")
    exp: dict[str, dict] = {}
    for r in rows:
        e = exp.setdefault(r["task_id"], {"repo": r["repo"], "files": set(), "syms": set()})
        (e["syms"].add(f"{r['file']}::{r['symbol']}") if r["symbol"] else e["files"].add(r["file"]))
    return exp


def statement_terms(text: str) -> set[str]:
    return set(tokenize(clean_query(text)))


# ---------------------------------------------------------------- mining

def pattern_id(kind: str, repo: str, term: str | None, file: str, symbol: str | None) -> str:
    return "P" + hashlib.sha1(f"{kind}|{repo}|{term}|{file}|{symbol}".encode()).hexdigest()[:8]


def _make(kind, repo, term, target, support_ids, act_ids, lift):
    file, _, symbol = target.partition("::")
    support = len(support_ids)
    weight = math.log1p(support) * (1.0 if term is None else math.log2(lift))
    return {"id": pattern_id(kind, repo, term, file, symbol or None), "kind": kind, "repo": repo, "term": term,
            "file": file, "symbol": symbol or None, "w": round(weight, 4), "support": support,
            "activations": len(act_ids), "precision": round(support / len(act_ids), 4), "lift": round(lift, 4),
            "evidence": sorted(support_ids)}


def mine(task_ids, exp: dict, terms: dict, min_support: int, min_lift: float, kinds,
         min_support_symbol: int | None = None) -> list[dict]:
    support_for = {"file": min_support, "symbol": min_support_symbol or min_support}
    patterns = []
    by_repo = defaultdict(list)
    for t in task_ids:
        if t in exp:
            by_repo[exp[t]["repo"]].append(t)
    for repo, ids in by_repo.items():
        n = len(ids)
        has = {"file": defaultdict(set), "symbol": defaultdict(set)}
        for t in ids:
            for f in exp[t]["files"]:
                has["file"][f].add(t)
            for s in exp[t]["syms"]:
                has["symbol"][s].add(t)
        for level in ("file", "symbol"):
            if f"{level}_prior" in kinds:
                for target, sup in has[level].items():
                    if len(sup) >= support_for[level]:
                        patterns.append(_make(f"{level}_prior", repo, None, target, sup, set(ids), 1.0))
        if not any(k.endswith("_term") for k in kinds):
            continue
        term_ids = defaultdict(set)
        for t in ids:
            for term in terms.get(t, ()):
                term_ids[term].add(t)
        for term, act in term_ids.items():
            if len(act) < min(support_for.values()) or len(act) > MAX_TERM_DF * n:
                continue
            for level in ("file", "symbol"):
                if f"{level}_term" not in kinds:
                    continue
                for target, holders in has[level].items():
                    sup = act & holders
                    if len(sup) < support_for[level]:
                        continue
                    lift = (len(sup) / len(act)) / (len(holders) / n)
                    if lift >= min_lift:
                        patterns.append(_make(f"{level}_term", repo, term, target, sup, act, lift))
    return patterns


# ---------------------------------------------------------------- verification

def folds_by_repo(ids_by_repo: dict[str, list[str]], k: int, seed: int = 13) -> list[list[str]]:
    out = [[] for _ in range(k)]
    rng = random.Random(seed)
    i = 0
    for repo in sorted(ids_by_repo):
        items = sorted(ids_by_repo[repo])
        rng.shuffle(items)
        for t in items:
            out[i % k].append(t)
            i += 1
    return out


def _folds(cases: list[Case], k: int) -> list[list[str]]:
    by_repo = defaultdict(list)
    for c in cases:
        by_repo[c.task.repo_short].append(c.task.instance_id)
    return folds_by_repo(by_repo, k)


def objective(m: dict) -> float:
    return m["r@5"] + m["sym_shown"]


def run_fold(train_ids, test_cases, exp, terms, params, cfg) -> dict[str, dict]:
    pats = mine(train_ids, exp, terms, params["min_support"], params["min_lift"], params["kinds"],
                params.get("min_support_symbol"))
    fold_cfg = {**cfg, "pattern_weight": params["pattern_weight"]}
    return {c.task.instance_id: evaluate_case(c, pats, fold_cfg) for c in test_cases}


def cv_results(cases, exp, terms, params, cfg, k=FOLDS) -> dict[str, dict]:
    by_id = {c.task.instance_id: c for c in cases}
    folds = _folds(cases, k)
    out = {}
    for i, test in enumerate(folds):
        train = [t for j, f in enumerate(folds) if j != i for t in f]
        out.update(run_fold(train, [by_id[t] for t in test], exp, terms, params, cfg))
    return out


def grid(kinds_options=None):
    keys = list(GRID)
    values = [GRID[k] if k != "kinds" or kinds_options is None else kinds_options for k in keys]
    return [dict(zip(keys, v)) for v in itertools.product(*values)]


def select_params(cases, exp, terms, cfg, candidates) -> tuple[dict, dict]:
    best = None
    for params in candidates:
        m = summarize(list(cv_results(cases, exp, terms, params, cfg, k=4).values()))
        if best is None or objective(m) > objective(best[1]):
            best = (params, m)
    return best


def nested_cv(cases, exp, terms, cfg, candidates, fixed_per_fold: list[dict] | None = None):
    """Outer 5 folds for an unbiased estimate; parameters chosen by inner CV on each outer training part.

    fixed_per_fold[i] pins parameters for fold i (only candidates agreeing with it are searched).
    Returns (per-task results, parameters chosen in each fold).
    """
    by_id = {c.task.instance_id: c for c in cases}
    out, chosen = {}, []
    for i, test in enumerate(_folds(cases, FOLDS)):
        train_cases = [c for c in cases if c.task.instance_id not in set(test)]
        fixed = fixed_per_fold[i] if fixed_per_fold else {}
        fold_candidates = [p for p in candidates if all(p[k] == v for k, v in fixed.items())] or candidates
        params, _ = select_params(train_cases, exp, terms, cfg, fold_candidates)
        chosen.append(params)
        out.update(run_fold([c.task.instance_id for c in train_cases], [by_id[t] for t in test],
                            exp, terms, params, cfg))
    return out, chosen


# ---------------------------------------------------------------- shared entry points

def _context(tasks_path: Path):
    from lab.store.experience_store import ExperienceStore
    tasks = load_tasks(tasks_path)
    cases = [c for c in (build_case(t) for t in tasks) if c]
    exp = load_experience(ExperienceStore())
    terms = {t.instance_id: statement_terms(t.problem_statement) for t in tasks}
    return cases, exp, terms


def load_deployed_params() -> dict:
    path = REFS / "consolidation.json"
    if path.exists():
        p = json.loads(path.read_text())["params"]
        return {**p, "kinds": tuple(p["kinds"])}
    return {"min_support": 3, "min_lift": 2.0, "kinds": ("file_prior",), "pattern_weight": 1.0}


def cross_validated_results(cases, params, cfg, tasks_path: Path = ROOT / "data/competition/tasks.jsonl"):
    _, exp, terms = _context(tasks_path)
    return cv_results(cases, exp, terms, params, cfg)


# ---------------------------------------------------------------- deployment

def deploy(store, patterns: list[dict], params: dict, estimates: dict) -> int:
    version = max([r["v"] for r in store.query("SELECT MAX(version) AS v FROM pattern") if r["v"] is not None] or [0]) + 1
    run_id = f"cons-v{version}"
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    sources = ",".join(r["run_id"] for r in store.query("SELECT run_id FROM run WHERE kind IN (\"reference_patch\", \"agent_eval\")"))
    note = " ".join(f"{k}={v}" for k, v in params.items()) + f" ladder={LADDER_RULE}"
    store.insert("run", {"run_id": run_id, "kind": "consolidation", "created": now, "source": sources, "note": note})
    for p in patterns:
        store.insert("pattern", {"pattern_id": p["id"], "version": version, "run_id": run_id, "kind": p["kind"],
                                 "repo": p["repo"], "term": p["term"], "target_file": p["file"],
                                 "target_symbol": p["symbol"], "weight": p["w"], "support": p["support"],
                                 "activations": p["activations"], "precision": p["precision"], "lift": p["lift"]})
        for t in p["evidence"]:
            store.insert("pattern_evidence", {"pattern_id": p["id"], "version": version, "task_id": t, "hit": True})
    store.save()
    store.snapshot(f"patterns_v{version}", "pattern", f"version = {version}")

    REFS.mkdir(parents=True, exist_ok=True)
    with open(REFS / "seed_patterns.txt", "w") as f:
        for p in patterns:
            f.write(json.dumps({"id": p["id"], "v": version, "kind": p["kind"], "repo": p["repo"], "term": p["term"],
                                "file": p["file"], "symbol": p["symbol"], "w": p["w"]}, separators=(",", ":")) + "\n")
    meta = {"version": version, "run_id": run_id, "created": now, "sources": sources.split(","),
            "params": {**params, "kinds": list(params["kinds"])}, "ladder_rule": LADDER_RULE, "n_patterns": len(patterns),
            "nested_cv": estimates}
    (REFS / "consolidation.json").write_text(json.dumps(meta, indent=1) + "\n")
    cfg_path = REFS / "router_config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    cfg.pop("prior_weight", None)
    cfg["pattern_weight"] = params["pattern_weight"]
    cfg_path.write_text(json.dumps(cfg, indent=1) + "\n")
    return version


def fmt(m: dict) -> str:
    return f"R@1 {100 * m['r@1']:.0f}% | R@5 {100 * m['r@5']:.0f}% | R@10 {100 * m['r@10']:.0f}% | MRR {m['mrr']:.3f} | symbol shown {100 * m['sym_shown']:.0f}%"


def _task_objective(r: dict) -> float:
    return (1.0 if r["rank"] and r["rank"] <= 5 else 0.0) + (1.0 if r["symbol_shown"] else 0.0)


def sign_test_p(better: int, worse: int) -> float:
    """Two-sided exact sign test."""
    n = better + worse
    if n == 0:
        return 1.0
    k = min(better, worse)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def mrr_diff_ci(a: dict, b: dict, ids, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    rr = lambda r: 1 / r["rank"] if r["rank"] else 0.0  # noqa: E731
    d = [rr(b[i]) - rr(a[i]) for i in ids]
    rng = random.Random(seed)
    boots = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(n_boot))
    return boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)]


def paired_permutation_p(diffs: list[float], n_perm: int = 10000, seed: int = 0) -> float:
    """Two-sided sign-flip permutation test on the mean of paired differences (zeros carry no information)."""
    d = [x for x in diffs if x != 0]
    if not d:
        return 1.0
    observed = abs(sum(d))
    rng = random.Random(seed)
    hits = sum(abs(sum(x if rng.random() < 0.5 else -x for x in d)) >= observed - 1e-12 for _ in range(n_perm))
    return (hits + 1) / (n_perm + 1)


MAX_P = 0.10          # sign test on per-task (top5 + symbol shown) to accept a more complex habit set
MIN_MRR_DIFF_LO = -0.01  # lower 95% bound of the paired MRR change tolerated


def complexity_ladder(cases, exp, terms, cfg, lexical_results: dict) -> tuple[tuple, dict, list[str]]:
    """Walk kind-sets from simple to complex; keep a step only if it wins paired against the current one.

    Each step searches only the parameters of the kinds it adds (PARAM_KINDS); the rest stay at the values
    each outer fold chose when the current kinds were accepted, so a step is judged on what it adds.
    """
    ids = [c.task.instance_id for c in cases]
    current_kinds, current, current_params, log = None, lexical_results, None, []
    for kinds in GRID["kinds"]:
        added = set(kinds) - set(current_kinds or ())
        fixed_keys = [k for k, owners in PARAM_KINDS.items() if current_params and not (owners & added)]
        fixed = [{k: p[k] for k in fixed_keys} for p in current_params] if current_params else None
        cand, chosen = nested_cv(cases, exp, terms, cfg, grid([kinds]), fixed)
        better = sum(_task_objective(cand[i]) > _task_objective(current[i]) for i in ids)
        worse = sum(_task_objective(cand[i]) < _task_objective(current[i]) for i in ids)
        p = sign_test_p(better, worse)
        lo, hi = mrr_diff_ci(current, cand, ids)
        accept = better > worse and p <= MAX_P and lo >= MIN_MRR_DIFF_LO
        # Reported only (external review, 2026-09-28); the admission rule above is unchanged.
        rr = lambda r: 1 / r["rank"] if r["rank"] else 0.0  # noqa: E731
        p_perm_obj = paired_permutation_p([_task_objective(cand[i]) - _task_objective(current[i]) for i in ids])
        p_perm_rr = paired_permutation_p([rr(cand[i]) - rr(current[i]) for i in ids])
        log.append(f"{'+'.join(kinds):45} vs {('+'.join(current_kinds) if current_kinds else 'lexical'):30} "
                   f"better {better:3} worse {worse:3} p={p:.3f} perm_p(obj)={p_perm_obj:.3f} "
                   f"perm_p(RR)={p_perm_rr:.3f} MRR diff CI [{lo:+.3f},{hi:+.3f}] "
                   f"fixed {fixed_keys or '-'} -> {'ACCEPT' if accept else 'reject'} | "
                   f"{fmt(summarize(list(cand.values())))}")
        if accept:
            current_kinds, current, current_params = kinds, cand, chosen
    return current_kinds, current, log


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=ROOT / "data/competition/tasks.jsonl", type=Path)
    ap.add_argument("--deploy", action="store_true", help="write the store version and references/ if a habit set is accepted")
    args = ap.parse_args()

    cases, exp, terms = _context(args.tasks)
    cfg = router_cfg()
    lexical_results = {c.task.instance_id: evaluate_case(c, [], cfg) for c in cases}
    lexical = summarize(list(lexical_results.values()))
    print(f"lexical only: {fmt(lexical)}\n")
    kinds, chosen, log = complexity_ladder(cases, exp, terms, cfg, lexical_results)
    print("\n".join(log))
    if kinds is None:
        print("\nno habit set beats lexical out of sample: nothing to deploy")
        return
    estimate = summarize(list(chosen.values()))
    print(f"\naccepted kinds: {kinds} | nested CV: {fmt(estimate)}")
    params, _ = select_params(cases, exp, terms, cfg, grid([kinds]))
    print(f"parameters for deployment (4-fold CV on all tasks): {params}")
    if args.deploy:
        from lab.store.experience_store import ExperienceStore
        pats = mine([c.task.instance_id for c in cases], exp, terms, params["min_support"], params["min_lift"],
                    params["kinds"], params.get("min_support_symbol"))
        version = deploy(ExperienceStore(), pats, params, {"lexical": lexical, "habits_nested_cv": estimate})
        print(f"deployed patterns v{version}: {len(pats)} habits "
              f"({dict(Counter(p['kind'] for p in pats))}) -> references/seed_patterns.txt")


if __name__ == "__main__":
    main()
