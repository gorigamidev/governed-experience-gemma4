"""Stage, validate (with the official adk_submission validator) and zip submission/.

Needs the lab venv (google-adk + adk_submission from the wheelhouse):
  PYTHONPATH=lab/src lab/.venv/bin/python -m lab.export.package
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import zipfile
from pathlib import Path

from adk_submission import (
    GenerationConstraints,
    NumericRange,
    SubmissionLimits,
    compile_submission,
    discover_adapters,
    validate_directory,
)
from adk_submission.registry import ModelRegistry

from lab.export import models

ROOT = Path(__file__).resolve().parents[4]
MODEL = models.name(models.DEFAULT)  # gemma-4-31b-it-qat-w4a16-ct; --model overrides it per build
TEXT_EXT = frozenset({".yaml", ".yml", ".json", ".md", ".txt"})
# Mirrors swegemma/config.py (0.2.7): ALLOWED_SUBMISSION_EXTENSIONS and build_submission_limits().
ALLOWED_EXT = frozenset({".yaml", ".yml", ".md", ".txt", ".py", ".json", ".safetensors"})
LIMITS = SubmissionLimits(
    max_total_size_bytes=3 * 1024**3, max_yaml_size_bytes=50 * 1024**2, max_skill_size_bytes=50 * 1024**2,
    max_file_count=10_000, max_yaml_files=1_000, max_instruction_chars=1_000_000,
    max_total_instruction_chars=10_000_000, max_agents=500, max_sub_agent_depth=50, max_skills=1_000,
    max_loop_iterations=500, allowed_file_extensions=ALLOWED_EXT, adapter_extensions=frozenset({".safetensors"}),
)
GEN = GenerationConstraints(
    allowed_fields=None, max_output_tokens=NumericRange(1, 32768), thinking_budget=NumericRange(0, 32768),
    defaults={"max_output_tokens": 16384, "thinking_config": {"thinking_budget": 4096}},
)
HARNESS_TOOLS = ["run_command", "read_file", "write_file", "edit_file", "submit_patch", "get_status",
                 "get_code_neighbors", "search_similar_code", "get_code_subgraph"]
SKILL_SUBDIRS = {"scripts", "references", "assets"}
MAX_SKILL_PAYLOAD = 1024**2  # our own target; ADK only warns at 16 MB


def stage(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"))


def check_skills(root: Path) -> list[str]:
    errors = []
    for y in list(root.rglob("*.yaml")) + list(root.rglob("*.yml")):
        if re.search(r"^\s*(before|after)_(agent|model|tool)_callbacks\s*:", y.read_text(), re.M):
            errors.append(f"{y.relative_to(root)}: callbacks are silently ignored (swegemma passes no CallbackRegistry)")
    for skill_md in root.glob("skills/*/SKILL.md"):
        skill = skill_md.parent
        m = re.search(r"^name:\s*(\S+)", skill_md.read_text(), re.M)
        if not m or m.group(1) != skill.name:
            errors.append(f"{skill.name}: SKILL.md name must equal the directory name")
        if not re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", skill.name):
            errors.append(f"{skill.name}: name must be kebab-case (snake_case is off by default in ADK)")
        for child in skill.iterdir():
            if child.is_dir() and child.name not in SKILL_SUBDIRS:
                errors.append(f"{skill.name}/{child.name}/ is ignored by ADK (only {sorted(SKILL_SUBDIRS)} load)")
        payload = sum(p.stat().st_size for d in SKILL_SUBDIRS for p in (skill / d).rglob("*") if p.is_file())
        if payload > MAX_SKILL_PAYLOAD:
            errors.append(f"{skill.name}: {payload:,} bytes are re-sent on every run_skill_script call")
    return errors


def retarget(root: Path, model: str) -> int:
    """Rewrite the default checkpoint name to `model` on every staged text file (a 12B build differs only there)."""
    target = models.name(model)
    if target == MODEL:
        return 0
    changed = 0
    for p in root.rglob("*"):
        if p.is_file() and p.suffix in TEXT_EXT and MODEL in p.read_text():
            p.write_text(p.read_text().replace(MODEL, target))
            changed += 1
    return changed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=ROOT / "submission", type=Path)
    ap.add_argument("--build", default=ROOT / "build/submission", type=Path)
    ap.add_argument("--zip", default=ROOT / "build/submission.zip", type=Path)
    ap.add_argument("--model", default=models.DEFAULT, help="checkpoint short name (31b, 12b, e4b) or full name")
    args = ap.parse_args()
    model_name = models.name(args.model)

    stage(args.src, args.build)
    # a source already on the target checkpoint (re-validating a 12B or E4B zip) has nothing left to retarget
    if model_name != MODEL and retarget(args.build, args.model) == 0 and model_name not in (args.build / "agent.yaml").read_text():
        sys.exit(f"Submission rejected: no {MODEL} in {args.build} to retarget to {model_name}")
    validate_directory(args.build, LIMITS)
    adapters = discover_adapters(args.build, LIMITS.adapter_extensions)

    models_registry = ModelRegistry()
    models_registry.register(model_name, model_name)
    for name in adapters.adapters:
        models_registry.register(name, model_name)
    agent = compile_submission(
        submission_dir=args.build,
        tool_registry={name: (lambda **_: "") for name in HARNESS_TOOLS},
        model_registry=models_registry,
        limits=LIMITS,
        generation_constraints=GEN,
        adapter_manifest=adapters,
    )

    errors = check_skills(args.build)
    if errors:
        sys.exit("Submission rejected:\n  " + "\n  ".join(errors))

    files = sorted(p for p in args.build.rglob("*") if p.is_file())
    args.zip.parent.mkdir(parents=True, exist_ok=True)
    # Deterministic archive: fixed timestamps and order, so the same source always yields the same sha256.
    with zipfile.ZipFile(args.zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in files:
            info = zipfile.ZipInfo(p.relative_to(args.build).as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, p.read_bytes())
    total = sum(p.stat().st_size for p in files)
    print(f"OK: root agent '{agent.name}', {len(files)} files, {total:,} bytes unpacked -> {args.zip} "
          f"({args.zip.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
