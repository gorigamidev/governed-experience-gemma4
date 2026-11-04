"""Turn a `swegemma eval` results directory into agent experience (pure parsing, standard library only).

Layout read (swegemma 0.2.7): task_results.jsonl, patches/<id>.patch, traces/trace_<id>.json (ATIF v1.7).
For each task it recovers the outcome, what the agent's patch touched, and what the mini-palace skill did:
calls per action, whether `locate` came first, and the habit ids `locate` reported as active. Those ids
are the link from a verified outcome back to the consolidated patterns in LINALDB.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from lab.ingest.load_tasks import parse_patch

SKILL_NAME = "mini-palace"
HABITS_RE = re.compile(r'"habits"\s*:\s*\[([^\]]*)\]')


@dataclass
class SkillCall:
    step: int
    action: str
    ok: bool | None
    decision: str | None = None
    files: list[str] = field(default_factory=list)
    habits: list[str] = field(default_factory=list)
    cached: bool = False
    verdict: str | None = None


@dataclass
class Cost:
    """What the model spent deliberating: most task time is generation (doc 02, section 8.7)."""
    llm_calls: int = 0
    completion_tokens: int = 0
    thinking_chars: int = 0
    first_edit_s: float | None = None  # seconds from task start to the first successful edit_file/write_file
    failed_edits: int = 0


EDIT_TOOLS = ("edit_file", "write_file")


def _is_error(content) -> bool:
    text = content if isinstance(content, str) else json.dumps(content, default=str)
    return bool(re.search(r"""["']status["']:\s*["']error["']""", text))


def parse_cost(trace: dict) -> Cost:
    cost = Cost()
    for step in trace.get("steps", []):
        if step.get("source") != "agent":
            continue
        extra = step.get("extra") or {}
        if extra.get("event_type") == "thinking":
            cost.thinking_chars += len(step.get("message") or "")
        for tc in step.get("tool_calls") or []:
            if tc.get("function_name") not in EDIT_TOOLS:
                continue
            # A failed edit (old_string not found, path outside the workspace...) changes nothing.
            if _is_error((step.get("observation") or {}).get("content")):
                cost.failed_edits += 1
            elif cost.first_edit_s is None:
                cost.first_edit_s = (tc.get("extra") or {}).get("elapsed_s")
    fm = trace.get("final_metrics") or {}
    cost.completion_tokens = int(fm.get("total_completion_tokens") or 0)
    cost.llm_calls = sum(1 for s in trace.get("steps", []) if s.get("source") == "agent" and s.get("metrics"))
    return cost


@dataclass
class TaskRun:
    instance_id: str
    resolved: bool
    error: str | None
    duration_s: float
    tool_calls: int
    patch_size: int
    patch_files: list[str] = field(default_factory=list)
    patch: str = ""
    tools_used: dict[str, int] = field(default_factory=dict)
    skill_calls: list[SkillCall] = field(default_factory=list)
    first_tool: str | None = None
    has_trace: bool = False
    cost: Cost | None = None

    @property
    def habits(self) -> list[str]:
        seen: dict[str, None] = {}
        for c in self.skill_calls:
            for h in c.habits:
                seen.setdefault(h)
        return list(seen)

    @property
    def skill_first(self) -> bool:
        return bool(self.skill_calls) and self.first_tool == "run_skill_script"


def _skill_payload(content) -> tuple[dict | None, str]:
    """observation.content of run_skill_script -> (our emitted JSON, raw stdout)."""
    if isinstance(content, list):
        content = " ".join(str(c.get("text", c)) if isinstance(c, dict) else str(c) for c in content)
    try:
        outer = json.loads(content) if isinstance(content, str) else content
    except (TypeError, ValueError):
        return None, str(content)
    stdout = outer.get("stdout", "") if isinstance(outer, dict) else ""
    try:
        return json.loads(stdout), stdout
    except (TypeError, ValueError):
        return None, stdout  # emit() may cut an oversized payload; habits are still recoverable by regex


def parse_trace(trace: dict) -> tuple[dict[str, int], list[SkillCall], str | None]:
    tools: dict[str, int] = {}
    skill_calls: list[SkillCall] = []
    first_tool = None
    for step in trace.get("steps", []):
        calls = step.get("tool_calls") or []
        for tc in calls:
            name = tc.get("function_name", "")
            tools[name] = tools.get(name, 0) + 1
            if first_tool is None and name not in ("get_status", "list_skills", "load_skill"):
                first_tool = name
        skill = [tc for tc in calls if tc.get("function_name") == "run_skill_script"
                 and (tc.get("arguments") or {}).get("skill_name") == SKILL_NAME]
        if not skill:
            continue
        args = skill[0].get("arguments") or {}
        script_args = args.get("args") if isinstance(args.get("args"), dict) else {}
        action = script_args.get("action") or next(iter(args.get("positional_args") or []), None) or "?"
        payload, raw = _skill_payload((step.get("observation") or {}).get("content", ""))
        call = SkillCall(step=step.get("step_id", 0), action=action, ok=None)
        if payload is not None:
            call.ok = bool(payload.get("ok"))
            call.decision = payload.get("decision")
            call.files = [i["file"] for i in payload.get("items", []) if isinstance(i, dict) and "file" in i]
            call.habits = list(payload.get("habits", []))
            call.cached = bool(payload.get("cached"))
            call.verdict = payload.get("verdict")
        else:
            m = HABITS_RE.search(raw)
            call.habits = re.findall(r'"(P[0-9a-f]{8})"', m.group(1)) if m else []
        skill_calls.append(call)
    return tools, skill_calls, first_tool


def load_results(results_dir: Path) -> list[TaskRun]:
    """Per-task runs; a local Docker re-verification (verified_task_results.jsonl) overrides `resolved`."""
    verified_path = results_dir / "verified_task_results.jsonl"
    verified = {}
    if verified_path.exists():
        for line in verified_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                v = json.loads(line)
                verified[v["instance_id"]] = bool(v["resolved"])
    runs = []
    for line in (results_dir / "task_results.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        iid = r["instance_id"]
        safe = iid.replace("/", "__")
        patch_path = results_dir / "patches" / f"{safe}.patch"
        patch = patch_path.read_text(encoding="utf-8") if patch_path.exists() else ""
        run = TaskRun(instance_id=iid, resolved=verified.get(iid, bool(r.get("resolved"))), error=r.get("error"),
                      duration_s=float(r.get("duration_seconds") or 0.0), tool_calls=int(r.get("tool_calls") or 0),
                      patch_size=int(r.get("agent_patch_size") or len(patch)), patch=patch,
                      patch_files=[c.path for c in parse_patch(patch)])
        trace_path = results_dir / "traces" / f"trace_{safe}.json"
        if trace_path.exists():
            trace = json.loads(trace_path.read_text())
            run.tools_used, run.skill_calls, run.first_tool = parse_trace(trace)
            run.cost = parse_cost(trace)
            run.has_trace = True
        runs.append(run)
    return runs
