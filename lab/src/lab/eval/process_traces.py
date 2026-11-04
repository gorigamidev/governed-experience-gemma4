"""One command from a finished Kaggle GPU notebook to evidence: download, extract, re-verify, ingest, report.

Usage: PYTHONPATH=lab/src lab/.venv/bin/python -m lab.eval.process_traces --kernel gorigamidev/gemma4-agent-traces --label kaggle-v1
Needs KAGGLE_API_TOKEN in the environment (the Makefile loads .env) and Docker for the re-verification.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


def step(title: str, cmd: list[str]) -> None:
    print(f"\n== {title}", flush=True)
    if subprocess.run(cmd, cwd=ROOT).returncode != 0:
        sys.exit(f"step failed: {title}")


def unpack(archive: Path, eval_root: Path, label: str) -> dict[str, Path]:
    """Extract results.tar.gz into eval_root/<label>, or eval_root/<label>-<arm> for each arm of a multi-arm
    notebook (results/<arm>/). Refuses to overwrite a processed run."""
    tmp = eval_root / f".{label}.tmp"
    with tarfile.open(archive) as tar:
        tar.extractall(tmp, filter="data")
    extracted = tmp / "results"
    arms = ({"": extracted} if (extracted / "task_results.jsonl").exists()
            else {d.name: d for d in sorted(extracted.iterdir()) if (d / "task_results.jsonl").exists()})
    targets = {arm: eval_root / (f"{label}-{arm}" if arm else label) for arm in arms}
    clash = [str(t) for t in targets.values() if t.exists()]
    if not arms or clash:
        shutil.rmtree(tmp)
        sys.exit(f"{clash[0]} already exists; remove it or use another label" if clash
                 else f"no task_results.jsonl in {archive}")
    for arm, src in arms.items():
        src.rename(targets[arm])
    shutil.rmtree(tmp)
    return targets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", help="Kaggle kernel whose output holds results.tar.gz")
    ap.add_argument("--from-archive", type=Path, help="use an already downloaded results.tar.gz instead")
    ap.add_argument("--label", required=True)
    ap.add_argument("--skip-verify", action="store_true", help="keep the notebook's own (subprocess) outcomes")
    args = ap.parse_args()

    raw = ROOT / "data/eval" / f"{args.label}-raw"
    results = ROOT / "data/eval" / args.label
    raw.mkdir(parents=True, exist_ok=True)
    if args.from_archive:
        archive = args.from_archive
    else:
        if not args.kernel:
            sys.exit("give --kernel or --from-archive")
        step("download notebook output", ["kaggle", "kernels", "output", args.kernel, "-p", str(raw)])
        archive = raw / "results.tar.gz"
    if not archive.exists():
        sys.exit(f"no results.tar.gz in the notebook output ({raw}); check the kernel log there")
    targets = unpack(archive, results.parent, args.label)

    py = [sys.executable, "-m"]
    for target in targets.values():
        label = target.name
        if not args.skip_verify:
            step(f"re-verify patches with the local Docker sandbox ({label})",
                 py + ["lab.eval.verify_patches", "--results", str(target), "--download-missing"])
        step(f"ingest into LINALDB ({label})", py + ["lab.store.ingest_eval", "--results", str(target), "--label", label])
        if not args.skip_verify:
            step(f"replay habit G1's check on the final patches ({label})",
                 py + ["lab.eval.replay_check", "--results", str(target)])
        step(f"trace report ({label})", py + ["lab.analysis.trace_report", "--results", str(target)])


if __name__ == "__main__":
    main()
