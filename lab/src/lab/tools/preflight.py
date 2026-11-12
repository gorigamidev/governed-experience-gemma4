"""Local gate before anything goes to Kaggle (AGENTS.md, workflow step 6): each Kaggle cycle costs GPU hours or the
daily submission, and pipeline bugs show up locally in minutes.

For every arm (a submission.zip) it checks that
  1. the zip is exactly what the official validator builds from its own contents (re-packaged byte for byte),
  2. the compiled agent has the expected configuration (skill present, enable_thinking on or off, thinking budget),
  3. a GPU notebook, if given, embeds exactly these arms and the current runner (its materialize cell runs locally),
  4. the agent runs end to end on a few tasks with the notebook's sandbox (subprocess) and a local model, and the
     traces carry no infrastructure errors (missing modules, a skill that returns nothing, bad skill JSON), and
  5. trace_report computes every metric the decision rules use.
Model behavior (failed edits, timeouts) is reported, not failed: a small local model says nothing about Gemma's
quality or speed. Each arm also runs with a small Gemma 4 (--behavior-model, same chat template and tool-call format
as the scorer's model), and a checklist of behaviors to read is printed for both runs: invented tools, empty
submissions, text-only streaks, malformed skill calls, repeated calls. Submission 5 shipped with an empty-submission
trace in its preflight that was read as "the action ran".

Usage: PYTHONPATH=lab/src lab/.venv/bin/python -m lab.tools.preflight --arms b2=build/submission.zip \
           [--notebook build/kaggle_eval] [--tasks rich_4077,rich_3067,fastapi_14372] [--model qwen3:8b]
           [--behavior-model gemma4-e4b-32k | --behavior-model none]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OLLAMA = "http://localhost:11434"
# Tool errors that mean the environment is broken rather than the model being wrong.
INFRA_ERROR = re.compile(r"No module named|ModuleNotFoundError|MissingRepoContext|ImportError")
ERROR_STATUS = re.compile(r"""["']status["']:\s*["']error["']""")
ERROR_MESSAGE = re.compile(r"""["']error_message["']:\s*["']([^"']*)""")
SKILL_ACTIONS = {"locate", "outline", "ledger", "check"}
METRICS = ("first_edit_s", "gen_tokens", "thinking_kchars", "llm_calls", "failed_edits", "patch_touches_gold",
           "locate_gold_rank", "undeclared_tool", "check_calls", "check_broken")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def declared_model(src: Path) -> str:
    """Checkpoint named by the package's root agent, so a 12B or E4B zip is validated against its own model."""
    from lab.export import models

    found = re.search(r"^model:\s*(\S+)", (src / "agent.yaml").read_text(), re.M)
    return models.short(found.group(1)) if found else models.DEFAULT


def repackage_matches(zip_path: Path, work: Path) -> tuple[bool, str]:
    """The zip must be what `lab.export.package` produces from its own contents: validated and deterministic."""
    src, build, out = work / "src", work / "build", work / "repacked.zip"
    zipfile.ZipFile(zip_path).extractall(src)
    res = subprocess.run([sys.executable, "-W", "ignore", "-m", "lab.export.package", "--src", str(src),
                          "--model", declared_model(src), "--build", str(build), "--zip", str(out)],
                         capture_output=True, text=True, cwd=ROOT)
    if res.returncode != 0:
        return False, (res.stderr or res.stdout).strip()[-300:]
    return sha(out) == sha(zip_path), f"validator OK, sha {sha(zip_path)[:12]}"


def compiled_config(build_dir: Path) -> dict:
    from adk_submission import compile_submission
    from swegemma.config import build_submission_limits

    from lab.eval.run_eval import model_registry
    from lab.export.package import HARNESS_TOOLS

    limits, gen = build_submission_limits()
    agent = compile_submission(submission_dir=build_dir, tool_registry={n: (lambda **_: "") for n in HARNESS_TOOLS},
                               model_registry=model_registry("http://127.0.0.1:1/v1", "x", "openai/", "EMPTY"),
                               limits=limits, generation_constraints=gen)
    thinking, budget = {}, {}
    stack = [agent]
    while stack:  # composite roots (SequentialAgent) have no model; report every LlmAgent in the tree
        node = stack.pop()
        stack.extend(node.sub_agents)
        # Agents used as tools (agent_tool) are not sub_agents; their thinking can differ from the root's.
        stack.extend(t.agent for t in getattr(node, "tools", []) if hasattr(t, "agent"))
        if hasattr(node, "model") and hasattr(node.model, "_additional_args"):
            extra = node.model._additional_args.get("extra_body") or {}
            thinking[node.name] = (extra.get("chat_template_kwargs") or {}).get("enable_thinking")
            budget[node.name] = extra.get("thinking_token_budget")
    values, budgets = set(thinking.values()), set(budget.values())
    return {"skill": (build_dir / "skills").is_dir(),
            "enable_thinking": values.pop() if len(values) == 1 else thinking,
            "thinking_token_budget": budgets.pop() if len(budgets) == 1 else budget}


def check_notebook(notebook_dir: Path, arms: dict[str, Path], work: Path, max_minutes: float | None = None) -> list[str]:
    """Run the notebook's materialize cell with WORK redirected; it asserts every embedded hash itself."""
    nb = json.loads((notebook_dir / "eval.ipynb").read_text())
    cell = next("".join(c["source"]) for c in nb["cells"] if "ARMS = " in "".join(c["source"]))
    scope: dict = {}
    exec(cell.replace("Path('/kaggle/working')", f"Path({str(work)!r})"), scope)
    problems = []
    embedded = {name: v[0] for name, v in scope["ARMS"].items()}
    expected = {name: sha(p) for name, p in arms.items()}
    if embedded != expected:
        problems.append(f"notebook arms {embedded} != preflight arms {expected}")
    if scope["RUN_EVAL_SHA256"] != sha(ROOT / "lab/src/lab/eval/run_eval.py"):
        problems.append("notebook embeds an older lab/eval/run_eval.py; rebuild it")
    if scope.get("MAX_MINUTES") != max_minutes:
        problems.append(f"notebook per-task cap {scope.get('MAX_MINUTES')} != preflight --max-minutes {max_minutes}")
    return problems


def trace_problems(results: Path, has_skill: bool, require_skill: bool = True) -> tuple[list[str], list[str]]:
    """(infrastructure problems that fail the gate, model-behavior notes)."""
    from lab.ingest.load_traces import load_results

    infra, notes = [], []
    runs = load_results(results)
    for run in runs:
        iid = run.instance_id
        if not run.has_trace:
            infra.append(f"{iid}: no trace")
            continue
        if run.error and not run.error.startswith("Agent exceeded"):
            notes.append(f"{iid}: {run.error[:80]}")
        if run.cost and run.cost.llm_calls == 0:
            notes.append(f"{iid}: not exercised (no model call finished in the budget; local model too slow?)")
        steps = json.loads((results / "traces" / f"trace_{iid}.json").read_text()).get("steps", [])
        rejected = {st.get("step_id") for st in steps if "error_code" in str((st.get("observation") or {}).get("content", ""))}
        for step in steps:
            tool = (step.get("tool_calls") or [{}])[0].get("function_name")
            content = str((step.get("observation") or {}).get("content", ""))
            # Only a harness tool failing counts; file contents and the model's own commands can mention imports.
            if tool and tool != "run_command" and ERROR_STATUS.search(content):
                message = ERROR_MESSAGE.search(content)
                hit = INFRA_ERROR.search(message.group(1) if message else "")
                if hit:
                    infra.append(f"{iid}: {tool} -> {hit.group(0)}")
        for call in run.skill_calls:
            if call.step in rejected:  # ADK refused the call itself (e.g. a script name that does not exist)
                notes.append(f"{iid}: malformed skill call rejected by ADK")
            elif call.ok is None:
                infra.append(f"{iid}: skill output is not JSON")
            elif not call.ok and call.action not in SKILL_ACTIONS:
                # The skill answered with its usage; the model invented the call (seen 30 Sep: args without action).
                notes.append(f"{iid}: model sent a skill call without a valid action")
            elif not call.ok:
                infra.append(f"{iid}: skill {call.action} returned ok=false")
            elif call.action == "locate" and not call.files:
                infra.append(f"{iid}: locate returned no candidates (is the skill seeing the repo?)")
        if has_skill and not run.skill_calls:
            notes.append(f"{iid}: the model did not call the skill")
    if not any(run.tool_calls for run in runs):
        infra.append("arm not exercised: no tool call in any task, so nothing was verified")
    code_skill = any((results / f"submission/skills/{n}").is_dir() for n in ("code", "repo-tools"))
    if code_skill:  # the read-lock skill of bf-gov-tb-lk (doc 03 §14 item 54); load_traces only knows mini-palace
        from lab.analysis.lock_report import task_metrics

        ok_calls = 0
        for path in sorted((results / "traces").glob("trace_*.json")):
            iid = path.stem.removeprefix("trace_")
            m = task_metrics(json.loads(path.read_text()))
            ok_calls += m["skill_calls"] - len(m["skill_errors"])
            ok_calls -= len(m["model_errors"])
            infra += [f"{iid}: well-formed code-skill call failed -> {err}" for err in m["skill_errors"]]
            notes += [f"{iid}: malformed code-skill call rejected by ADK ({err})" for err in m["model_errors"]]
            if m["undeclared"]:
                notes.append(f"{iid}: undeclared tools called {m['undeclared']} (ends the task in the harness)")
            if m["activations"]:
                notes.append(f"{iid}: lock fired {m['activations']}x, edit right after {m['edit_after_lock']}, "
                             f"first edit at call {m['first_edit_call']}")
        notes = [n for n in notes if not n.endswith("the model did not call the skill")]
        if require_skill and not ok_calls:
            infra.append("code skill not exercised: no successful skill call in any task")
    elif require_skill and has_skill and not any(c.ok for run in runs for c in run.skill_calls):
        infra.append("skill not exercised: no successful skill call in any task; rerun with more tasks")
    return sorted(set(infra)), notes


TEXT_STREAK = 5  # consecutive agent turns with text and no tool call
REPEAT_STREAK = 4  # the same tool call with the same arguments, back to back


def behavior_flags(results: Path) -> list[str]:
    """Behaviors to read before shipping; none of them fails the gate."""
    from lab.analysis.trace_report import UNDECLARED_TOOL
    from lab.ingest.load_traces import load_results

    flags = []
    for run in load_results(results):
        iid = run.instance_id
        if not run.has_trace:
            continue
        steps = json.loads((results / "traces" / f"trace_{iid}.json").read_text()).get("steps", [])
        agent = [st for st in steps if st.get("source") == "agent"]
        if UNDECLARED_TOOL.search(run.error or ""):
            flags.append(f"{iid}: invented or undeclared tool ended the task ({run.error[:70]})")
        submitted = run.tools_used.get("submit_patch", 0) > 0
        if submitted and run.patch_size == 0:
            after = " after check said no_changes" if any(c.verdict == "no_changes" for c in run.skill_calls) else ""
            flags.append(f"{iid}: empty patch submitted{after}")
        if "without calling submit_patch" in (run.error or ""):
            flags.append(f"{iid}: ended without submit_patch")
        streak = best = 0
        for st in agent:
            streak = streak + 1 if (not st.get("tool_calls") and st.get("message")) else 0
            best = max(best, streak)
        if best >= TEXT_STREAK:
            flags.append(f"{iid}: {best} text-only turns in a row")
        malformed = sum("INVALID_ARGUMENTS" in str((st.get("observation") or {}).get("content", ""))
                        for st in agent if any(tc.get("function_name") == "run_skill_script"
                                               for tc in st.get("tool_calls") or []))
        if malformed:
            flags.append(f"{iid}: {malformed} malformed skill call(s) rejected by ADK")
        calls = [json.dumps([tc.get("function_name"), tc.get("arguments")], sort_keys=True)
                 for st in agent for tc in st.get("tool_calls") or []]
        streak = best = 1 if calls else 0
        for prev, cur in zip(calls, calls[1:]):
            streak = streak + 1 if cur == prev else 1
            best = max(best, streak)
        if best >= REPEAT_STREAK:
            flags.append(f"{iid}: the same tool call {best} times in a row")
    return flags


def report_metrics(results: Path) -> tuple[dict, list[str]]:
    from lab.analysis.localization_bench import gold_files
    from lab.analysis.trace_report import task_row
    from lab.ingest.load_tasks import load_tasks
    from lab.ingest.load_traces import load_results

    tasks = {t.instance_id: t for t in load_tasks(ROOT / "data/competition/tasks.jsonl")}
    rows = [task_row(r, gold_files(tasks[r.instance_id].patch), None) for r in load_results(results)]
    missing = [f"{r['task']}: {k} not computed" for r in rows for k in METRICS if k not in r]
    missing += [f"{r['task']}: no cost data" for r in rows if r["llm_calls"] is None]
    summary = {k: [r[k] for r in rows] for k in ("gen_tokens", "thinking_kchars", "llm_calls", "first_edit_s",
                                                 "failed_edits", "check_calls")}
    return summary, missing


def ollama_up() -> bool:
    try:
        urllib.request.urlopen(f"{OLLAMA}/api/tags", timeout=2)
        return True
    except OSError:
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=f"main={ROOT / 'build/submission.zip'}", help="name=zip,...")
    ap.add_argument("--notebook", type=Path, help="GPU notebook dir to check against the arms (build/kaggle_eval)")
    ap.add_argument("--tasks", default="rich_4077,rich_3067,fastapi_14372")
    ap.add_argument("--model", default="qwen3:8b")
    # gemma4:e4b-it-qat runs at Ollama's default 4096-token context, which silently drops the task
    # from the prompt; gemma4-e4b-32k is the same weights with num_ctx 32768 (correction of Nov 2).
    ap.add_argument("--behavior-model", default="gemma4-e4b-32k",
                    help="second local run with a small Gemma 4 for the behavior checklist; 'none' to skip")
    ap.add_argument("--skip-run", action="store_true", help="static checks only (no local agent run)")
    ap.add_argument("--max-minutes", type=float, help="per-task cap for local runs of packages without eval_config")
    args = ap.parse_args()

    arms = {n: Path(p) for n, p in (item.split("=", 1) for item in args.arms.split(",") if item)}
    stamp = time.strftime("%Y%m%d-%H%M%S")
    failures: list[str] = []
    behavior_model = None if args.behavior_model in ("", "none") else args.behavior_model
    lines = [f"# Preflight {stamp}", "", f"Arms: {', '.join(arms)} | tasks: {args.tasks} | local model: {args.model}"
             + (f" | behavior model: {behavior_model}" if behavior_model else ""), ""]
    checklist: list[str] = []

    with tempfile.TemporaryDirectory() as tmp:
        builds = {}
        for name, zip_path in arms.items():
            work = Path(tmp, name)
            ok, detail = repackage_matches(zip_path, work)
            builds[name] = work / "build"
            cfg = compiled_config(builds[name]) if ok else {}
            lines.append(f"- `{name}`: {detail}; compiled {cfg}")
            if not ok:
                failures.append(f"{name}: zip is not the validated build ({detail})")
        if args.notebook:
            problems = check_notebook(args.notebook, arms, Path(tmp, "nb"), args.max_minutes)
            failures += problems
            lines.append(f"- notebook `{args.notebook}`: " + ("; ".join(problems) or "arms and runner match"))

        started = False
        if not args.skip_run and not failures:
            if not ollama_up():
                subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                started = True
                for _ in range(30):
                    if ollama_up():
                        break
                    time.sleep(1)
            try:
                runs = [(name, args.model, "") for name in arms]
                runs += [(name, behavior_model, "-gemma") for name in arms] if behavior_model else []
                for name, model, suffix in runs:
                    results = ROOT / "data/eval" / f"preflight-{stamp}-{name}{suffix}"
                    subprocess.run([sys.executable, "-W", "ignore", "-m", "lab.eval.run_eval", "--submission",
                                    str(builds[name]), "--task-ids", args.tasks, "--sandbox", "subprocess",
                                    "--api-base", f"{OLLAMA}/v1", "--served-model", model,
                                    "--label", f"preflight-{name}{suffix}", "--results", str(results),
                                    "--ollama-thinking-off",
                                    *(["--max-minutes", str(args.max_minutes)] if args.max_minutes else [])],
                                   cwd=ROOT, capture_output=True, text=True)
                    # The skill-exercised gate belongs to the main run; the behavior run may never reach the skill.
                    infra, notes = trace_problems(results, compiled_config(builds[name])["skill"],
                                                  require_skill=not suffix)
                    summary, missing = report_metrics(results)
                    failures += [f"{name}{suffix}: {p}" for p in infra + missing]
                    lines += [f"- run `{name}` with `{model}` ({results.relative_to(ROOT)}): "
                              + ("OK" if not infra and not missing else f"{len(infra) + len(missing)} problems"),
                              f"  - metrics per task: {summary}"]
                    lines += [f"  - infrastructure: {p}" for p in infra + missing]
                    lines += [f"  - model behavior (not a failure): {n}" for n in notes]
                    checklist += [f"`{name}` / `{model}`: {flag}" for flag in behavior_flags(results)]
            finally:
                if started:
                    subprocess.run(["pkill", "-x", "ollama"])

    if not args.skip_run:
        lines += ["", "## Behaviors to read before shipping (not failures)", ""]
        lines += [f"- {c}" for c in checklist] or ["- none"]
    lines += ["", "**Not verifiable locally**: Gemma's quality and speed, the scorer's hardware and sandbox.", "",
              f"**Result: {'FAIL' if failures else 'PASS'}**"] + [f"- {f}" for f in failures]
    out = ROOT / "data/eval" / f"preflight-{stamp}.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"-> {out}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
