"""Ablation variants derived from submission/ (never hand-maintained copies).

b2           : submission/ as is (agent + mini-palace skill).
b1-no-skill  : same agent, prompt, sampling and budgets; the skill and the prompt lines that call it removed,
               step 1 rewritten to locate with the native tools. B2 vs B1 isolates the skill.
b2-think-off : b2 with Gemma 4 thinking switched off (include_thoughts: false -> enable_thinking=false).
b1-think-off : b1-no-skill with thinking off. The 2x2 {b1, b2} x {thinking on, off} asks whether governed
               experience can stand in for deliberation.

Usage: PYTHONPATH=lab/src python3 -m lab.export.variants b1-no-skill   (writes build/variants/<name>/)
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]

NO_SKILL_LOCATE = """### 1. Locate the code
- Extract identifiers, file paths and error messages from the issue, then search for them with `run_command`, e.g. `grep -rn "name" --include=*.py <package dir>`, leaving tests out.
- `search_similar_code` expects symbol names (for example `get_openapi`), never sentences.
- Read only the relevant ranges with `read_file`.

"""


def _replace_once(text: str, pattern: str, repl: str, what: str) -> str:
    new, n = re.subn(pattern, repl, text, count=1, flags=re.S | re.M)
    if n != 1:
        raise SystemExit(f"b1-no-skill: could not find {what} in submission/; update variants.py")
    return new


def b1_no_skill(out: Path) -> None:
    shutil.rmtree(out / "skills")
    original = (out / "agent.yaml").read_text()
    agent = _replace_once(original, r"^skills:\n(?:  - [^\n]*\n)+", "", "the skills block of agent.yaml")
    # Everything but the skill lines must survive untouched (sampling, tools, instruction...).
    removed = [ln for ln in original.splitlines() if ln not in agent.splitlines()]
    if removed != ["skills:", "  - skills/mini-palace"]:
        raise SystemExit(f"b1-no-skill: agent.yaml lost more than the skill: {removed}")
    (out / "agent.yaml").write_text(agent)

    prompt = (out / "prompts/system.md").read_text()
    prompt = _replace_once(prompt, r"^### 1\. Locate with the `mini-palace` skill first\n.*?(?=^### 2\.)",
                           NO_SKILL_LOCATE, "prompt step 1 (skill-first locate)")
    prompt = _replace_once(prompt, r'^- If the context was compacted[^\n]*"ledger"[^\n]*\n', "", "the ledger bullet")
    if "mini-palace" in prompt or "run_skill_script" in prompt:
        raise SystemExit("b1-no-skill: prompt still mentions the skill")
    (out / "prompts/system.md").write_text(prompt)


def thinking_off(out: Path) -> None:
    # adk_submission maps include_thoughts: false to chat_template_kwargs.enable_thinking=false before looking at
    # thinking_budget, and Gemma 4 thinking has no levels (doc 01, section 6.5).
    path = out / "configs/sampling.yaml"
    original = path.read_text()
    sampling = _replace_once(original, r"^  include_thoughts: true$", "  include_thoughts: false", "include_thoughts: true")
    changed = [ln for ln in sampling.splitlines() if ln not in original.splitlines()]
    if changed != ["  include_thoughts: false"]:
        raise SystemExit(f"thinking_off: sampling.yaml changed more than include_thoughts: {changed}")
    path.write_text(sampling)


def b1_think_off(out: Path) -> None:
    b1_no_skill(out)
    thinking_off(out)


PHASES_ROOT = """agent_class: SequentialAgent
name: swe_agent
description: Locate with the governed memory, then edit from that plan.
sub_agents:
  - config_path: agents/locator.yaml
  - config_path: agents/editor.yaml
"""

LOCATOR = """name: locator
model: {model}
instruction: !include ../prompts/locate.md
tools:
  - read_file
  - get_status
skills:
  - skills/mini-palace
output_key: plan
disallow_transfer_to_parent: true
disallow_transfer_to_peers: true
generate_content_config: !include ../configs/sampling.yaml
"""

EDITOR = """name: editor
model: {model}
instruction: !include ../prompts/edit.md
tools:
{tools}skills:
  - skills/mini-palace
disallow_transfer_to_parent: true
disallow_transfer_to_peers: true
generate_content_config: !include ../configs/sampling.yaml
"""

LOCATE_PROMPT = """You are the locator. Your only job is to find where the issue in `/workspace` must be fixed and write a short plan. Another agent reads your plan and makes the change; you never edit, test or submit.

Plan already written in this session (empty if none): <<{plan?}>>
If it is not empty, reply with exactly that text and call no tool.

## Tools
You only have `run_skill_script` (the `mini-palace` skill), `read_file` and `get_status`. The task text may mention other tools (`run_command`, `edit_file`, `submit_patch`, code-graph tools); they belong to the editor and calling them fails the whole task.

## Steps
1. Call `run_skill_script(skill_name="mini-palace", file_path="scripts/entrypoint.py", args={"action": "locate", "text": "<issue title + identifiers, paths, error names>"})`. Keep `text` under ~400 characters.
2. `RESOLVE`: take the first file. `ESCALATE`: compare the candidates' `symbols` and pick the most likely file.
3. Call `{"action": "outline", "file": "<path>"}` for that file, then `read_file` only the range you need to confirm it. Use at most 6 tool calls in total.
4. Reply with the plan as plain text and no tool call, in this format:

FILES: <path>:<start>-<end> - <why>   (one line per file, most likely first)
CHANGE: <one or two sentences on what to change>
TEST: <test file or test name to run, if you saw one>
"""

EDIT_STEP = """### 1. Start from the locator's plan
The locator already searched with the governed memory (`mini-palace`) and wrote this plan:

<plan>
{plan?}
</plan>

- Open the files it names with `read_file` on the given ranges. Do not search the repository again unless the plan is empty or the code contradicts it.
- For other parts of a file, call the skill with `{"action": "outline", "file": "<path>"}` and read only that range.
- `search_similar_code` expects symbol names (for example `get_openapi`), never sentences.

"""


def phases(out: Path) -> None:
    """B2.2 (thinking off) split into a locator and an editor phase; one change: the structure."""
    thinking_off(out)
    agent = (out / "agent.yaml").read_text()
    model = re.search(r"^model: (\S+)$", agent, re.M)
    tools = re.search(r"^tools:\n((?:  - [^\n]*\n)+)", agent, re.M)
    if not model or not tools:
        raise SystemExit("phases: could not read model and tools from agent.yaml; update variants.py")
    (out / "agents").mkdir()
    (out / "agents/locator.yaml").write_text(LOCATOR.format(model=model.group(1)))
    (out / "agents/editor.yaml").write_text(EDITOR.format(model=model.group(1), tools=tools.group(1)))
    (out / "agent.yaml").write_text(PHASES_ROOT)

    system = (out / "prompts/system.md").read_text()
    edit = _replace_once(system, r"^### 1\. Locate with the `mini-palace` skill first\n.*?(?=^### 2\.)",
                         EDIT_STEP.replace("\\", "\\\\"), "prompt step 1 (skill-first locate)")
    (out / "prompts/edit.md").write_text(edit)
    (out / "prompts/locate.md").write_text(LOCATE_PROMPT)
    (out / "prompts/system.md").unlink()


GRAPH_TOOLS = ["get_code_neighbors", "search_similar_code", "get_code_subgraph"]


def no_graph(out: Path) -> None:
    """B2.2 (thinking off) without the code-graph tools, so the skill is the only way to navigate besides grep.
    The harness prompt still lists them when a graph exists; calling an undeclared tool aborts the task."""
    thinking_off(out)
    original = (out / "agent.yaml").read_text()
    agent = original
    for tool in GRAPH_TOOLS:
        agent = _replace_once(agent, rf"^  - {tool}\n", "", f"tool {tool} in agent.yaml")
    removed = [ln for ln in original.splitlines() if ln not in agent.splitlines()]
    if removed != [f"  - {t}" for t in GRAPH_TOOLS]:
        raise SystemExit(f"no_graph: agent.yaml lost more than the graph tools: {removed}")
    (out / "agent.yaml").write_text(agent)
    prompt = (out / "prompts/system.md").read_text()
    prompt = _replace_once(prompt, r"^- `search_similar_code` expects[^\n]*\n",
                           "- Only use the tools you were given; the code-graph tools are not available.\n",
                           "the search_similar_code bullet")
    (out / "prompts/system.md").write_text(prompt)


LOCATOR_TOOL_STEP = """### 1. Get the plan from the locator
- Your first action is to call the `locator` tool with `request` set to the issue title plus identifiers, paths and error names. It searches with the governed memory (`mini-palace`) and returns a plan: `FILES`, `CHANGE`, `TEST`.
- Open the files it names with `read_file` on the given ranges. Do not search the repository again unless the plan is empty or the code contradicts it.
- For other parts of a file, call the skill with `{"action": "outline", "file": "<path>"}` and read only that range.
- `search_similar_code` expects symbol names (for example `get_openapi`), never sentences.

"""


def locator_tool(out: Path) -> None:
    """B2.2 with the locator as an agent_tool of the root editor. Unlike b2-phases, a harness nudge returns to the
    editor instead of restarting the sequence (seen in the local preflight: the locator re-ran and rewrote the plan)."""
    thinking_off(out)
    agent = (out / "agent.yaml").read_text()
    model = re.search(r"^model: (\S+)$", agent, re.M)
    if not model:
        raise SystemExit("locator_tool: no model in agent.yaml; update variants.py")
    (out / "agents").mkdir()
    locator = LOCATOR.format(model=model.group(1)).replace("output_key: plan\n", "")
    locator = locator.replace("name: locator\n", "name: locator\ndescription: Finds where to fix the issue with the "
                              "governed memory and returns a plan (FILES, CHANGE, TEST). Never edits.\n")
    (out / "agents/locator.yaml").write_text(locator)
    agent = _replace_once(agent, r"^(tools:\n(?:  - [^\n]*\n)+)",
                          r"\1  - agent_tool:\n      config_path: agents/locator.yaml\n      skip_summarization: false\n",
                          "tools block of agent.yaml")
    (out / "agent.yaml").write_text(agent)
    prompt = _replace_once(LOCATE_PROMPT, r"^Plan already written.*?call no tool\.\n",
                           "## Issue\n{problem_description}\n", "the plan lines of the locator prompt")
    (out / "prompts/locate.md").write_text(prompt)
    system = (out / "prompts/system.md").read_text()
    system = _replace_once(system, r"^### 1\. Locate with the `mini-palace` skill first\n.*?(?=^### 2\.)",
                           LOCATOR_TOOL_STEP.replace("\\", "\\\\"), "prompt step 1 (skill-first locate)")
    (out / "prompts/system.md").write_text(system)


REREAD_BULLET = ("- If `edit_file` fails (for example `old_string not found`), `read_file` the exact lines again and copy "
                 "`old_string` from that output before retrying. Never retry the same `old_string`.\n")


def b2_3(out: Path) -> None:
    """B2.2 (thinking off) plus one prompt line: re-read the exact lines after a failed edit (doc 03, 14.11)."""
    thinking_off(out)
    path = out / "prompts/system.md"
    original = path.read_text()
    prompt = _replace_once(original, r"^(- Edit source files only, with small `edit_file` calls\.[^\n]*\n)",
                           lambda m: m.group(1) + REREAD_BULLET, "the edit_file bullet of step 2")
    added = [ln for ln in prompt.splitlines() if ln not in original.splitlines()]
    if added != [REREAD_BULLET.rstrip("\n")]:
        raise SystemExit(f"b2.3: the prompt changed more than the re-read line: {added}")
    path.write_text(prompt)


EXPLICIT_PROMPT = Path(__file__).parent / "prompts/system_explicit.md"


def b2_explicit(out: Path) -> None:
    """B2.2 (thinking off) with system.md replaced by the explicit step-by-step prompt; nothing else changes."""
    thinking_off(out)
    (out / "prompts/system.md").write_text(EXPLICIT_PROMPT.read_text())


def b2_explicit_short(out: Path) -> None:
    """Same rules and order as b2-explicit in minimal wording: isolates prompt length at constant style."""
    thinking_off(out)
    (out / "prompts/system.md").write_text((Path(__file__).parent / "prompts/system_explicit_short.md").read_text())


EXPLICIT_NO_GRAPH_LINE = "- Never run the whole test suite. Never install packages (no network). The code-graph tools are not available.\n"


def b2_explicit_no_graph(out: Path) -> None:
    """b2-no-graph (the submitted package) with the explicit prompt; the graph line adapts to the missing tools."""
    no_graph(out)
    prompt = _replace_once(EXPLICIT_PROMPT.read_text(), r"^- Never use `search_similar_code`\.[^\n]*\n",
                           EXPLICIT_NO_GRAPH_LINE, "the search_similar_code line of the explicit prompt")
    (out / "prompts/system.md").write_text(prompt)


CHECK_ADDON = Path(__file__).parent / "skill_addons/check.py"
# Lineage of habit G1, shipped with the skill (doc 03, 14.16; replay in data/analysis/replay-check-g1.jsonl).
G1_RECORD = {"id": "G1", "v": 1, "kind": "guard", "action": "check", "admitted": "2026-10-01",
             "evidence": "autopsy 2026-10-01: 9 agent patches on the right file broken by the agent itself",
             "verification": {"rule": "doc 03 14.16", "flagged": "8/9", "broken_import_syntax": "6/6",
                              "false_broken_on_resolved": "0/25", "regressions_on_resolved": "6/25 (informational)",
                              "median_seconds": 0.9}}
CHECK_SKILL_ROW = ('| `{"action": "check"}` | Before `submit_patch`: compiles and imports your changed modules and runs their existing '
                   'tests against HEAD. `verdict`: `broken`, `regressions`, `hygiene`, `ok` or `no_changes` (nothing edited yet: '
                   'the issue is not fixed, do not submit), with the evidence |\n')
CHECK_PROMPT = ('- Before `submit_patch`, call the skill with `{"action": "check"}`. If `verdict` is `broken`, fix the error it '
                'shows and call `check` again. If it is `regressions`, read which tests fail: that is expected when they assert '
                'the old behavior that the issue asks to change.\n')


def b2_check(out: Path) -> None:
    """b2-no-graph plus habit G1 (doc 03, 14.16): the skill's `check` action and one prompt line to call it."""
    no_graph(out)
    skill = out / "skills/mini-palace"
    b2_check_skill(skill)
    doc = _replace_once((skill / "SKILL.md").read_text(), r'^(\| `\{"action": "ledger", "subject"[^\n]*\n)',
                        lambda m: m.group(1) + CHECK_SKILL_ROW, "ledger row of SKILL.md")
    (skill / "SKILL.md").write_text(doc)
    prompt = _replace_once((out / "prompts/system.md").read_text(), r"^(- Call `submit_patch` as your last action\.[^\n]*\n)",
                           lambda m: CHECK_PROMPT + m.group(1), "the submit_patch bullet")
    (out / "prompts/system.md").write_text(prompt)


def b2_check_skill(skill: Path) -> None:
    """Install habit G1 into a mini-palace skill dir: the check addon, its lineage record and the action wiring."""
    shutil.copy(CHECK_ADDON, skill / "scripts/check.py")
    (skill / "references/guards.json").write_text(json.dumps(G1_RECORD, indent=1) + "\n")
    entry = (skill / "scripts/entrypoint.py").read_text()
    entry = _replace_once(entry, r"^(  ledger  \[[^\n]*\n)", lambda m: m.group(1) + "  check\n", "ledger usage line")
    entry = _replace_once(entry, r"^(import lineage  # noqa: E402\n)", lambda m: "import check  # noqa: E402\n" + m.group(1),
                          "import lineage line")
    entry = _replace_once(entry, r'"ledger": action_ledger\}', '"ledger": action_ledger, "check": check.action_check}',
                          "ACTIONS dict")
    (skill / "scripts/entrypoint.py").write_text(entry)


TOP_BUNDLE = ROOT / "external/top-bundle"
BUDGET_FIT = ROOT / "external/budget-fit"
H24 = ROOT / "external/h24"
TOP_SKILL_DOC = """---
name: mini-palace
description: Habit G1, verify before submit. Before calling submit_patch, call run_skill_script with skill_name "mini-palace", file_path "scripts/entrypoint.py" and args {"action":"check"}. It compiles and imports your changed modules and runs their existing tests against HEAD.
---

# mini-palace (habit G1)

`run_skill_script(skill_name="mini-palace", file_path="scripts/entrypoint.py", args={"action": "check"})`

| `verdict` | Meaning |
|---|---|
| `broken` | Your change does not compile or import. Fix the error shown, then call check again |
| `regressions` | Existing tests that passed before your change fail now. Expected if they assert the old behavior the issue asks to change |
| `hygiene` | You touched test files or left scratch files in the repository |
| `ok` | No breakage found |
| `no_changes` | You have not edited anything yet: the issue is not fixed. Do not submit an empty patch |

The habit and its verification are recorded in `references/guards.json`.
"""
TOP_PROMPT_LINE = ('   - Before `submit_patch`, call the `mini-palace` skill with `{"action": "check"}`. If `verdict` is `broken`, '
                   'fix the error it shows and call `check` again.\n')


def top_check(out: Path) -> None:
    """The public top bundle (external/top-bundle) plus habit G1 for the coder; one change: the check gate."""
    skill = out / "skills/mini-palace"
    shutil.copytree(ROOT / "submission/skills/mini-palace", skill,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"))
    b2_check_skill(skill)
    (skill / "SKILL.md").write_text(TOP_SKILL_DOC)
    agent = (out / "agent.yaml").read_text()
    if "skills:" in agent:
        raise SystemExit("top_check: the top bundle already declares skills; update variants.py")
    (out / "agent.yaml").write_text(agent.rstrip("\n") + "\nskills:\n  - skills/mini-palace\n")
    prompt = _replace_once((out / "prompts/system.md").read_text(), r"^(6\. \*\*Submit\*\*:[^\n]*\n)",
                           lambda m: m.group(1) + TOP_PROMPT_LINE, "step 6 (Submit) of the top prompt")
    (out / "prompts/system.md").write_text(prompt)


PROBE_ADAPTER = "edit_lora"


def write_probe_adapter(dest: Path) -> None:
    """A PEFT directory with rank-1 zero tensors, stdlib only. It tests packaging and routing (which agent asks the
    server for which model name); its weights are meaningless and it must never reach a scored submission."""
    dest.mkdir(parents=True)
    (dest / "adapter_config.json").write_text(json.dumps(
        {"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": 1, "lora_alpha": 1, "lora_dropout": 0.0, "bias": "none",
         "target_modules": ["q_proj"], "base_model_name_or_path": "probe"}, indent=1))
    names = ["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
             "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"]
    header, offset = {}, 0
    for name in names:
        header[name] = {"dtype": "F32", "shape": [1, 1], "data_offsets": [offset, offset + 4]}
        offset += 4
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    (dest / "adapter_model.safetensors").write_bytes(struct.pack("<Q", len(blob)) + blob + bytes(offset))


def phases_adapter(out: Path) -> None:
    """b2-phases with the probe adapter on the editor only: local wiring test for per-phase adapters."""
    phases(out)
    write_probe_adapter(out / "adapters" / PROBE_ADAPTER)
    editor = (out / "agents/editor.yaml").read_text()
    editor = _replace_once(editor, r"^(model: [^\n]*\n)", rf"\1adapter: {PROBE_ADAPTER}\n", "model line of editor.yaml")
    (out / "agents/editor.yaml").write_text(editor)


def top_think(out: Path) -> None:
    """The public top bundle with thinking on (scorer of 30 Sep: thoughts kept across calls, thinking_budget is a cap)."""
    sampling = (out / "configs/sampling.yaml").read_text()
    sampling = _replace_once(sampling, r"include_thoughts: false", lambda m: "include_thoughts: true",
                             "include_thoughts of sampling.yaml")
    (out / "configs/sampling.yaml").write_text(sampling)


MIX_EVAL_CONFIG = """# The scorer reads only these four fields and runs ~120 hidden tasks sequentially; a run past 12 h errors as a whole
# (forum 743063). The top bundle at 4.9 min per task finished within 12 h, so sandbox setup is < ~1.1 min per task:
# worst case 120 x (4.5 + 1.1) = 11.2 h (doc 03 14.24, E4).
evaluation:
  max_time_minutes: 4.5
  # Phase 2 runs the official pytest with this same timeout (swegemma harness/verification.py).
  timeout_seconds: 300
  max_tool_calls: 100
  # submit_patch is not counted by max_tool_calls; this cap ends submit loops (doc 02 8.14, E3).
  max_turns: 150
"""
MIX_SUBMIT_STEP = ("6. **Submit**: delete scratch files you created inside /workspace with `rm` (never `git restore`, "
                   "`git checkout`, `git reset` or `git stash`), run `git diff --stat` once, then call `submit_patch`. "
                   "`submit_patch` is your last tool call: when it returns `\"status\": \"ok\"`, reply with the single "
                   "word DONE and call no more tools.\n")


def top_mix(budget: int):
    """The top bundle with thinking on and a smaller thinking_budget, a stop after submit_patch, and an eval_config
    that fits ~120 tasks in 12 h (doc 03 14.24)."""
    def build(out: Path) -> None:
        top_think(out)
        sampling = _replace_once((out / "configs/sampling.yaml").read_text(), r"thinking_budget: 4096",
                                 lambda m: f"thinking_budget: {budget}", "thinking_budget of sampling.yaml")
        (out / "configs/sampling.yaml").write_text(sampling)
        prompt = _replace_once((out / "prompts/system.md").read_text(), r"^6\. \*\*Submit\*\*:[^\n]*\n",
                               lambda m: MIX_SUBMIT_STEP, "step 6 (Submit) of the top prompt")
        (out / "prompts/system.md").write_text(prompt)
        if (out / "eval_config.yaml").exists():
            raise SystemExit("top_mix: the top bundle already has an eval_config.yaml; update variants.py")
        (out / "eval_config.yaml").write_text(MIX_EVAL_CONFIG)
    return build



def top_mix_analyzer_off(budget: int):
    """top-mix with the analyzer back on the top bundle's own sampling (thinking off): one change. The mix shares
    configs/sampling.yaml between coder and analyzer, so it also turned thinking on for the analyzer, which then
    took a median 176 s per call against 23-35 s without thinking (doc 03 §14.26)."""
    def build(out: Path) -> None:
        original = (TOP_BUNDLE / "configs/sampling.yaml").read_text()
        top_mix(budget)(out)
        (out / "configs/sampling_analyzer.yaml").write_text(original)
        analyzer = _replace_once((out / "sub_agents/code_analyzer.yaml").read_text(),
                                 r"generate_content_config: !include \.\./configs/sampling\.yaml",
                                 lambda m: "generate_content_config: !include ../configs/sampling_analyzer.yaml",
                                 "sampling include of the analyzer")
        (out / "sub_agents/code_analyzer.yaml").write_text(analyzer)
    return build


def top_mix_analyzer_off_tb0(budget: int):
    """`top-mix-1k-an-off` with the analyzer's thinking switched off by `thinking_budget: 0`. `adk_submission`
    0.2.13 (wheelhouse 2026-10-09, forum 747914) reads a positive `thinking_budget` before `include_thoughts`,
    so the analyzer of `top-mix-1k-an-off` (`thinking_budget: 4096, include_thoughts: false`) would think again;
    `thinking_budget: 0` turns it off in both 0.2.12 and 0.2.13 without touching the coder."""
    def build(out: Path) -> None:
        top_mix_analyzer_off(budget)(out)
        sa = (out / "configs/sampling_analyzer.yaml").read_text()
        (out / "configs/sampling_analyzer.yaml").write_text(
            _replace_once(sa, r"thinking_budget:\s*4096", lambda m: "thinking_budget: 0",
                          "thinking_budget of sampling_analyzer.yaml"))
    return build

DIRECTED_ANALYZER = """name: code_analyzer
model: {model}
description: Reasons over governed evidence about the repository. Given an issue, returns the exact location, root cause and a fix plan.
instruction: !include ../prompts/analyzer.md
generate_content_config: !include ../configs/sampling.yaml
include_contents: none
tools:
  - read_file
skills:
  - skills/mini-palace
"""
DIRECTED_PROMPT = """You are `code_analyzer`. You reason only over evidence; you never explore blindly and never modify files.

## Evidence (at most 4 tool calls; your only tools are `run_skill_script` and `read_file`)
1. `run_skill_script(skill_name="mini-palace", file_path="scripts/entrypoint.py", args={{"action": "locate", "text": "<issue title and key terms, at most 400 characters>"}})`
2. `run_skill_script(skill_name="mini-palace", file_path="scripts/entrypoint.py", args={{"action": "outline", "file": "<best candidate path>"}})`
3. `read_file(filepath="<that path>", start_line=<start>, end_line=<end>)` for the function that explains the issue.

## Answer from that evidence only (at most 220 words, nothing else), as visible text
REASONING: <two or three sentences that tie the evidence to the issue>
LOCATION: <path>:<start>-<end> (<function>), the line that explains the issue
NOW / EXPECTED: <what the code does today> / <what the issue asks>
FIX PLAN: <the minimal change>
TESTS: <existing test file and the pytest command>
CONFIDENCE: high | medium | low
MISSING: <the one fact you would still need, or "none">

Your final message is only those seven lines.
"""
DIRECTED_CODER_LINE = ("   - Call the `code_analyzer` tool exactly once, first, passing the full issue text verbatim. It returns "
                       "LOCATION / NOW / EXPECTED / FIX PLAN / TESTS. Read those exact lines, then follow its FIX PLAN; "
                       "do not call it again.\n")
def top_directed(out: Path) -> None:
    """The top bundle whose analyzer reasons over governed evidence: no conversation history, the skill and read_file as
    its only tools, and the reasoning written into a typed answer (REASONING, LOCATION, ...). Thinking stays off."""
    shutil.copytree(ROOT / "submission/skills/mini-palace", out / "skills/mini-palace",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"))
    analyzer = (out / "sub_agents/code_analyzer.yaml").read_text()
    model = re.search(r"^model: (\S+)$", analyzer, re.M)
    if not model:
        raise SystemExit("top_directed: no model in code_analyzer.yaml; update variants.py")
    (out / "sub_agents/code_analyzer.yaml").write_text(DIRECTED_ANALYZER.format(model=model.group(1)))
    (out / "prompts/analyzer.md").write_text(DIRECTED_PROMPT.format())
    # The reasoning is written in the typed answer, not in hidden thinking: with thinking on, the local E4B test put the
    # whole answer in its thoughts and the coder received {"raw": ""} (3 Oct). The analyzer keeps the top's sampling.
    # With skip_summarization the analyzer's answer ends the coder's turn, and the next harness nudge went to an
    # analyzer without history: 8 of 12 tasks ended without submit_patch (local E4B test, 3 Oct).
    agent = (out / "agent.yaml").read_text()
    agent = _replace_once(agent, r"skip_summarization: true", lambda m: "skip_summarization: false",
                          "skip_summarization of the code_analyzer agent_tool")
    (out / "agent.yaml").write_text(agent)
    system = (out / "prompts/system.md").read_text()
    system = _replace_once(system, r"^   - Call the `code_analyzer` tool with the full issue text first\.[^\n]*\n",
                           lambda m: DIRECTED_CODER_LINE, "the code_analyzer bullet of the coder prompt")
    (out / "prompts/system.md").write_text(system)


BF_EDIT_RETRY = ("- If `edit_file` returns an error, never repeat the same call: re-read the exact lines with `read_file` "
                 "and copy `old_string` from that output, with every mandatory parameter.\n")
BF_CLEANUP = ("  1. Delete scratch files you created inside /workspace (for example `repro.py` or `reproduce_issue.py`) "
              "with `rm`; never use `git restore`, `git checkout`, `git reset` or `git stash`.\n"
              "  2. Call `submit_patch`.\n"
              "  3. Verify `patch_size > 0` and `files_changed > 0`.\n"
              "  4. Output a short summary of the fix to end the session.\n")


def bf_habits(out: Path) -> None:
    """`bf` plus two autopsy habits as prompt rules (doc 03 §14.32, outcome 1): re-read after a failed edit (R1) and
    remove scratch files before submitting (aux). Sampling and budgets stay as in `bf`; O (`max_output_tokens` 4096)
    is left out because `bf` never overflowed and 7 of its completions exceed 4096 tokens with thinking."""
    prompt = (out / "prompts/system.md").read_text()
    prompt = _replace_once(prompt, r"^- Apply the minimal necessary fix[^\n]*\n",
                           lambda m: m.group(0) + BF_EDIT_RETRY, "the first bullet of step 2 of the bf prompt")
    prompt = _replace_once(prompt, r"^  1\. Call `submit_patch` immediately\.\n.*?^  3\. Output a short summary[^\n]*\n",
                           lambda m: BF_CLEANUP, "the submission steps of the bf prompt")
    (out / "prompts/system.md").write_text(prompt)


BF_NO_SCRATCH = ("- Never create scratch or reproduction files: anything left under /workspace becomes part of the graded "
                 "patch, and `write_file` cannot write outside /workspace. To try code, run it inline with "
                 "`run_command` (`python -c \"...\"` or `python - <<'EOF'`).\n")


def bf_gov(out: Path) -> None:
    """`bf` plus the two procedure rules the 19 Oct pool showed the 31B follows (doc 03 §14.34): the failed-edit rule
    of `bf-habits`, and scratch-file prevention that, unlike h24's "/tmp only", does not send `write_file` outside
    /workspace. The submission steps are unchanged, so no extra `rm` call."""
    prompt = (out / "prompts/system.md").read_text()
    prompt = _replace_once(prompt, r"^- Apply the minimal necessary fix[^\n]*\n",
                           lambda m: m.group(0) + BF_EDIT_RETRY + BF_NO_SCRATCH, "the first bullet of step 2 of the bf prompt")
    (out / "prompts/system.md").write_text(prompt)


BF_TIMEOUT = ("- Run every `python`, `python -c`, heredoc or `pytest` command with `timeout 90` in front (for example "
              "`timeout 90 python -c \"...\"`): a reproduction that hangs otherwise blocks until the 240-second "
              "command limit.\n")


def bf_gov_to(out: Path) -> None:
    """`bf-gov` plus one rule (doc 03 §14.35): bound every execution with `timeout 90`. No finished command in 582
    traces took over 66 s; 15 traces hung on the 240 s limit, mostly reproductions that loop."""
    bf_gov(out)
    prompt = (out / "prompts/system.md").read_text()
    prompt = _replace_once(prompt, r"^- Never create scratch or reproduction files[^\n]*\n",
                           lambda m: m.group(0) + BF_TIMEOUT, "the scratch-file rule of bf-gov")
    (out / "prompts/system.md").write_text(prompt)


def bf_gov_tb(out: Path) -> None:
    """`bf-gov-to` with `thinking_budget` 2048 instead of 4096 (doc 03 §14.37): 55 turns in 211 base traces stop at
    the 4096 budget, ~2 min each, and their traces hit the session timeout 46% of the time against 11%."""
    bf_gov_to(out)
    path = out / "configs/sampling.yaml"
    sampling = _replace_once(path.read_text(), r"^  thinking_budget: 4096$", "  thinking_budget: 2048",
                             "thinking_budget of the bf sampling")
    path.write_text(sampling)


ED_LOCATOR = """name: locator
description: Reads the repository and returns where to fix the issue (FILE, CHANGE, TEST). Never edits.
model: {model}
instruction: !include ../prompts/locate.md
tools:
  - read_file
  - run_command
  - get_status
disallow_transfer_to_parent: true
disallow_transfer_to_peers: true
generate_content_config: !include ../configs/sampling.yaml
"""

ED_LOCATE_PROMPT = """You are the locator. Your only job is to find where the issue in `/workspace` must be fixed and say what to change. Another agent reads your answer and makes the change; you never edit, test or submit.

## Issue
{problem_description}

## Rules
- You only have `read_file`, `run_command` and `get_status`. Use `run_command` only to read (`git grep -n "text" -- '*.py' | head -20`, `ls`, `sed -n`); never modify files and never run tests.
- Use at most 6 tool calls in total. Stop as soon as you have read the lines that must change.
- Reply with plain text and no tool call, in exactly this format:

FILE: <path>:<start>-<end> - <why>   (one line per file, most likely first, at most 2)
CHANGE: <one or two sentences on what to change in those lines>
TEST: <test file or test name to run, if you saw one>
"""

ED_STEP = """### 1. Get the plan from the locator, then edit
- Your first action is to call the `locator` tool with `request` set to the issue title plus identifiers, paths and error names. It reads the code and returns `FILE`, `CHANGE` and `TEST`. Its tool calls count against your budget.
- `read_file` only the lines it names, then make the edit within your next 3 tool calls. Do not search the repository again unless the plan is empty or the code contradicts it.
- Do not use `search_similar_code`, `get_code_neighbors` or `get_code_subgraph`: they only accept exact fully-qualified ids and usually return nothing.

"""

ED_BUDGET = """## Budget: 20 tool calls, 8 minutes
- You have at most 20 tool calls and 8 minutes for this task, including the locator's calls. When either runs out, the working tree is graded as it is, so an edit made in time counts even if you never reach `submit_patch`.
- By your 10th tool call you must have edited a source file. If you have not, stop exploring and make your best-guess edit in the lines the locator named now. Runs that are still only reading after 12 calls almost never succeed; more reading does not help.
- After your edit: one compile check or one targeted test, a correction if needed, then `submit_patch`. Do not start new exploration after your 15th tool call.

"""


def bf_gov_ed(out: Path) -> None:
    """`bf-gov-to` with two structural changes for the edit-timing lineage (doc 03 §14.38): 20 tool calls instead of
    28, and a locator `agent_tool` (no skill) that hands the editor a FILE/CHANGE/TEST plan. 52 of 211 base traces
    edit in the last 4 calls or never and resolve none, although 49 of them had read a gold file by call 2-4."""
    bf_gov_to(out)
    config = (out / "eval_config.yaml").read_text()
    config = _replace_once(config, r"^  max_tool_calls: 28\n", "  max_tool_calls: 20\n", "max_tool_calls of bf")
    (out / "eval_config.yaml").write_text(config)
    agent = (out / "agent.yaml").read_text()
    model = re.search(r"^model: (\S+)$", agent, re.M)
    if not model:
        raise SystemExit("bf_gov_ed: no model in agent.yaml; update variants.py")
    agent = _replace_once(agent, r"^(tools:\n(?:  - [^\n]*\n)+)",
                          r"\1  - agent_tool:\n      config_path: agents/locator.yaml\n      skip_summarization: false\n",
                          "tools block of agent.yaml")
    (out / "agent.yaml").write_text(agent)
    (out / "agents").mkdir()
    (out / "agents/locator.yaml").write_text(ED_LOCATOR.format(model=model.group(1)))
    (out / "prompts/locate.md").write_text(ED_LOCATE_PROMPT)
    prompt = (out / "prompts/system.md").read_text()
    prompt = _replace_once(prompt, r"^### 1\. Identify Target Files Immediately\n.*?(?=^### 2\.)",
                           lambda m: ED_STEP, "step 1 of the bf prompt")
    prompt = _replace_once(prompt, r"^## Budget: 28 tool calls, 8 minutes\n.*?(?=^## Anti-Patterns)",
                           lambda m: ED_BUDGET, "the budget section of the bf prompt")
    (out / "prompts/system.md").write_text(prompt)


# Levers of §14.40 take their base as an argument: built on `bf-gov-to` first, rebuilt on `bf-gov-tb` once the
# GPU pool of §14.37 made it the base (31 Oct).
# The count goes in the reasoning, never at the start of the reply: on Gemma 4 E4B a `[call N/28]` or `[step N/28]`
# prefix took the place of the tool-call token and the parser dropped every call (doc 03 §14.40).
LEVER_COUNTER = ("- Count your tool calls. In your reasoning before each tool call, write `Step N/28`, where N is the number "
                 "of the tool call you are about to make (1 for the first). Write it only in your reasoning, never next to "
                 "or inside the tool call. If you have not edited a source file yet, tool call 12 must be `edit_file` or "
                 "`write_file` on a source file.\n")
LEVER_TITLE = ("- If the problem statement has no code, traceback, error message or expected output (often only a title "
               "and a link you cannot open), you cannot read the intended behaviour from it, and reading more code will "
               "not reveal it. Within your first 6 tool calls: find the code the title names, check its existing tests "
               "(`git grep -n \"<name>\" -- tests | head -20`) and recent history (`git log --oneline -5 -- <file>`), "
               "state your hypothesis of the change in one line, and make that edit. Then run the related test and "
               "adjust the edit if needed.\n")
LEVER_CHEAP = ("- Editing is cheap and reversible. As soon as you have read the lines you believe cause the issue, edit "
               "them, then verify with one targeted test or inline run and correct the edit if it is wrong. Do not keep "
               "tracing callers or other files to confirm your hypothesis before the first edit.\n")


def bf_gov_ct(out: Path, base=bf_gov_to) -> None:
    """`bf-gov-to` plus a self-kept call counter (doc 03 §14.40, lever a): the harness shows no count before
    `budget_warning`, which with 28 calls first arrives at call 18, and no base trace mentions the call-12 deadline."""
    base(out)
    prompt = (out / "prompts/system.md").read_text()
    prompt = _replace_once(prompt, r"^- By your 12th tool call you must have edited[^\n]*\n",
                           lambda m: m.group(0) + LEVER_COUNTER, "the edit-by-12 rule of the bf prompt")
    (out / "prompts/system.md").write_text(prompt)


def bf_gov_ti(out: Path, base=bf_gov_to) -> None:
    """`bf-gov-to` plus a procedure for title-only statements (doc 03 §14.40, lever b): 35 of 77 runs on statements
    of at most 243 characters edit late or never, against 4 of 57 on long ones."""
    base(out)
    prompt = (out / "prompts/system.md").read_text()
    prompt = _replace_once(prompt, r"^- If the problem statement does not provide explicit file paths[^\n]*\n",
                           lambda m: m.group(0) + LEVER_TITLE, "the search bullet of step 1 of the bf prompt")
    (out / "prompts/system.md").write_text(prompt)


def bf_gov_ec(out: Path, base=bf_gov_to) -> None:
    """`bf-gov-to` plus "editing is cheap" (doc 03 §14.40, lever c): late editors name the culprit and keep tracing
    callers to confirm it before any edit."""
    base(out)
    prompt = (out / "prompts/system.md").read_text()
    prompt = _replace_once(prompt, r"^- Apply the minimal necessary fix[^\n]*\n",
                           lambda m: LEVER_CHEAP + m.group(0), "the first bullet of step 2 of the bf prompt")
    (out / "prompts/system.md").write_text(prompt)


def bf_gov_22(out: Path, base=bf_gov_to) -> None:
    """`bf-gov-to` with 22 tool calls instead of 28 (doc 03 §14.40, lever d): swegemma attaches `budget_warning` when
    10 or fewer calls remain and the cap is at least 20, so with 22 the first warning arrives at call 12, the deadline
    of the prompt. Only the stated limit and the late-exploration call (20 -> 16) follow the new cap."""
    base(out)
    config = (out / "eval_config.yaml").read_text()
    config = _replace_once(config, r"^  max_tool_calls: 28\n", "  max_tool_calls: 22\n", "max_tool_calls of bf")
    (out / "eval_config.yaml").write_text(config)
    prompt = (out / "prompts/system.md").read_text()
    prompt = _replace_once(prompt, r"^## Budget: 28 tool calls, 8 minutes$", "## Budget: 22 tool calls, 8 minutes",
                           "the budget heading of the bf prompt")
    prompt = _replace_once(prompt, r"You have at most 28 tool calls", "You have at most 22 tool calls",
                           "the budget line of the bf prompt")
    prompt = _replace_once(prompt, r"after your 20th tool call", "after your 16th tool call",
                           "the late-exploration line of the bf prompt")
    (out / "prompts/system.md").write_text(prompt)


def bf_gov_tb_sl(out: Path) -> None:
    """`bf-gov-tb` with structural slack (doc 03 §14 item 50): a real cap of 32 calls while the prompt keeps saying 28,
    so an edit that arrives just past the stated budget still lands. Nothing else changes."""
    bf_gov_tb(out)
    config = (out / "eval_config.yaml").read_text()
    config = _replace_once(config, r"^  max_tool_calls: 28\n", "  max_tool_calls: 32\n", "max_tool_calls of bf")
    (out / "eval_config.yaml").write_text(config)


LK_TOOLS = """## Tools
You have exactly these tools: run_skill_script (skill "repo-tools"), edit_file, write_file, get_status and submit_patch. There is no read_file and no run_command: calling them ends the task with an empty patch.

Every run_skill_script call has three arguments: skill_name is "repo-tools", file_path is "scripts/entrypoint.py" (never a repository file; the repository file or directory goes inside args as "path"), and args is an object. Replace each <...> with a real value:

```
search:      run_skill_script(skill_name="repo-tools", file_path="scripts/entrypoint.py", args={"search": "<identifier or error text from the issue>"})
read lines:  run_skill_script(skill_name="repo-tools", file_path="scripts/entrypoint.py", args={"path": "<file.py from a hit>", "start": <line - 20>, "end": <line + 40>})
list a dir:  run_skill_script(skill_name="repo-tools", file_path="scripts/entrypoint.py", args={"path": "<directory>"})
run tests:   run_skill_script(skill_name="repo-tools", file_path="scripts/entrypoint.py", args={"cmd": "python -m pytest <tests/test_file.py> -k <name> -q"})
```

- search: hits are path:line:text. It searches *.py first and every tracked file if that finds nothing. Add "path": "<directory>" to narrow it.
- read lines: content is the file verbatim (max 150 lines); copy old_string from it.
- run: runs in the repository root and stops after 90 seconds.
- If a call returns INVALID_ARGUMENTS or "skill_name is required", do not repeat it: write the next call exactly like the lines above, with skill_name="repo-tools".

"""

LK_BUDGET_OLD = (r"^- Use at most about 10 tool calls to find and read the code\.\n"
                 r"- By your 12th tool call you must have edited a source file\.[^\n]*\n")
LK_BUDGET = """- **Read lock**: before your first edit, the look and run actions together allow 10 calls. After that they return `locked` and no new code until you change a source file with `edit_file` or `write_file`. Find the place, read the exact lines and edit within those 10 calls. Your first edit can be a best guess: you can correct it afterwards, because any edit opens the lock for good.
- If a response says `locked`, your next call must be `edit_file`, with `old_string` copied from `last_read` or from code you already read. While locked, the look action only re-reads lines you already read, plus up to 3 new line ranges of any file you name with `path`, `start` and `end` (to copy an exact `old_string`); no searches, no directory listings and no `run`.
"""

LK_REMOVED_TOOLS = {"run_command", "read_file", "get_code_neighbors", "search_similar_code", "get_code_subgraph"}
LK_KEPT_TOOLS = ["edit_file", "write_file", "get_status", "submit_patch"]


def bf_gov_tb_lk(out: Path) -> None:
    """`bf-gov-tb` with a read lock (doc 03 §14 item 54): reading and running go only through the `code` skill, which
    stops returning new code after 10 calls with no edit. 89 of 133 late first edits in 264 GPU traces already had
    their `old_string` on screen by call 10. Budget, sampling and thinking are those of `bf-gov-tb`."""
    bf_gov_tb(out)
    agent = (out / "agent.yaml").read_text()
    block = re.search(r"^tools:\n((?:  - [^\n]*\n)+)", agent, re.M)
    if not block:
        raise SystemExit("bf-gov-tb-lk: no tools block in agent.yaml; update variants.py")
    tools = [t.strip()[2:] for t in block.group(1).splitlines()]
    if set(tools) - set(LK_KEPT_TOOLS) != LK_REMOVED_TOOLS or not set(LK_KEPT_TOOLS) <= set(tools):
        raise SystemExit(f"bf-gov-tb-lk: unexpected tools {tools}; update variants.py")
    new_block = "tools:\n" + "".join(f"  - {t}\n" for t in LK_KEPT_TOOLS)
    agent = agent[:block.start()] + new_block + agent[block.end():]
    agent = _replace_once(agent, r"^generate_content_config:", lambda m: "skills:\n  - skills/repo-tools\n" + m.group(0),
                          "generate_content_config of bf")
    (out / "agent.yaml").write_text(agent)
    shutil.copytree(ROOT / "lab/src/lab/export/skill_addons/repo-tools", out / "skills/repo-tools",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"))
    p = out / "prompts/system.md"
    t = p.read_text()
    t = _replace_once(t, r"^## Workflow\n", lambda m: LK_TOOLS + m.group(0), "Workflow heading of the bf prompt")
    t = _replace_once(t, r"^- Read only the specific target files and lines using `read_file` or search tools\.",
                      "- Read only the specific target files and lines with the look action.", "read bullet of step 1")
    t = _replace_once(t, r"with `git grep -n \"text\" -- '\*\.py' \| head -20`\.",
                      'with the search call shown under Tools.', "git grep bullet of step 1")
    t = _replace_once(t, r"^- Do not use `search_similar_code`[^\n]*\n", "", "graph-tools bullet of step 1")
    t = _replace_once(t, r"re-read the exact lines with `read_file` and copy",
                      "re-read the exact lines with the look action and copy", "failed-edit rule of bf-gov")
    t = _replace_once(t, r"run it inline with `run_command` \(`python -c \"\.\.\.\"` or `python - <<'EOF'`\)\.",
                      'run it inline with the run action (args={"cmd": "python -c \\"...\\""}).', "scratch-file rule of bf-gov")
    t = _replace_once(t, r"^- Run every `python`, `python -c`, heredoc or `pytest` command with `timeout 90`[^\n]*\n",
                      "- The run action stops every command after 90 seconds: keep reproductions and tests short and targeted.\n",
                      "timeout rule of bf-gov-to")
    t = _replace_once(t, r"find it explicitly with `find tests -name \"\*<name>\*\.py\"`",
                      'find it explicitly with the run action and `find tests -name "*<name>*.py"`', "find-tests bullet")
    t = _replace_once(t, LK_BUDGET_OLD, LK_BUDGET, "edit-by-12 rule of the bf budget")
    for word in ("read_file`", "run_command`", "git grep", "timeout 90"):
        if word in t.replace("There is no `read_file` and no `run_command`", ""):
            raise SystemExit(f"bf-gov-tb-lk: prompt still mentions {word}")
    p.write_text(t)


def h24_long(out: Path) -> None:
    """`h24` with a larger per-task budget (doc 03 §14.33): 36 tool calls and 9 minutes instead of 28 and 8. The call
    plan of the prompt (1-8 / 9-12 / 13-20) is unchanged; only the stated limit follows the new budget."""
    config = (out / "eval_config.yaml").read_text()
    config = _replace_once(config, r"^  max_tool_calls: 28\n", "  max_tool_calls: 36\n", "max_tool_calls of h24")
    config = _replace_once(config, r"^  max_time_minutes: 8\n", "  max_time_minutes: 9\n", "max_time_minutes of h24")
    (out / "eval_config.yaml").write_text(config)
    prompt = (out / "prompts/system.md").read_text()
    prompt = _replace_once(prompt, r"28 tool calls and 8 minutes in total", "36 tool calls and 9 minutes in total",
                           "the HARD LIMITS line of the h24 prompt")
    (out / "prompts/system.md").write_text(prompt)


MP_LOCATE = """- First call `run_skill_script(skill_name="mini-palace", file_path="scripts/entrypoint.py", args={"action": "locate", "text": "<issue title + identifiers, paths, error names>"})`, with `text` under ~400 characters. Its first file is where the fix usually goes, and its `symbols` (`function:line`) point to the function to change: read that range first.
- `ESCALATE` means several files are plausible: compare their `symbols`, then call `args={"action": "outline", "file": "<path>"}` on the most likely one to get every `function:line`, and read only the relevant range.
"""


def bf_gov_tb_mp(out: Path) -> None:
    """`bf-gov-tb` plus the `mini-palace` skill as in `submission/` (doc 03 §14.44): E4B edits early but edits the
    wrong file or region in 10 of 23 patches, where the 31B almost never does. Only the skill, its `skills` line and
    a locate-first bullet at the top of step 1 change."""
    bf_gov_tb(out)
    shutil.copytree(ROOT / "submission/skills/mini-palace", out / "skills/mini-palace",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"))
    agent = _replace_once((out / "agent.yaml").read_text(), r"^generate_content_config:",
                          lambda m: "skills:\n  - skills/mini-palace\n" + m.group(0), "generate_content_config of bf")
    (out / "agent.yaml").write_text(agent)
    prompt = _replace_once((out / "prompts/system.md").read_text(), r"^### 1\. Identify Target Files Immediately\n",
                           lambda m: m.group(0) + MP_LOCATE, "step 1 heading of the bf prompt")
    (out / "prompts/system.md").write_text(prompt)


def bf_gov_tb_mp_nohab(out: Path) -> None:
    """`bf-gov-tb-mp` with no consolidated habits: the 22 file priors were learned from every public task, including
    the 24 of the local test, so the local arm runs on the runtime localization alone (no leak). Local use only."""
    bf_gov_tb_mp(out)
    seeds = out / "skills/mini-palace/references/seed_patterns.txt"
    if not seeds.read_text().strip():
        raise SystemExit("bf-gov-tb-mp-nohab: seed_patterns.txt was already empty")
    seeds.write_text("")


def bf_gov_tb_lk_nt(out: Path) -> None:
    """Lock probe v4 (doc 03 §14 item 54): bf-gov-tb-lk with thinking off and nothing else. Tests whether the 31B's
    corrupted skill calls (66% in probe v3, thinking on) come from thinking; v3-v5 runs with thinking off had 0-6%."""
    bf_gov_tb_lk(out)
    thinking_off(out)


VR_RULE = ("- After your last edit and before `submit_patch`, run the existing test file of the module you changed, "
           "once: find it with `find tests -name \"test_<module>*.py\"`, then run "
           "`timeout 90 python -m pytest <that test file> -x -q` (add `-k <name>` when one test covers the change). "
           "If there is no such file, run `timeout 90 python -c \"...\"` that imports the changed module and calls the "
           "changed code with the input from the issue. If it fails because of your edit, fix the edit once, then submit.\n")


def bf_gov_to_nt(out: Path) -> None:
    """Doc 03 §14 item 55: `bf-gov-to` with thinking off and nothing else (doc 01 row 32 left thinking off to be
    revalidated after the 30 Sep scorer; the 31B with thinking off made hundreds of clean tool calls in v4-v5)."""
    bf_gov_to(out)
    thinking_off(out)


def bf_gov_to_nt_vr(out: Path) -> None:
    """Doc 03 §14 item 55: `bf-gov-to-nt` plus one rule whose action is visible in the call (run the changed module's
    existing tests after the last edit). In 48 `bf-gov-to` runs the model ran pytest after its last edit in 3."""
    bf_gov_to_nt(out)
    path = out / "prompts/system.md"
    prompt = _replace_once(path.read_text(), r"^- After your edit: one compile check or one targeted test, a correction if "
                           r"needed, then `submit_patch`\. (Do not start new exploration after your 20th tool call\.)$",
                           lambda m: VR_RULE.rstrip("\n") + " " + m.group(1), "the after-edit line of the bf budget section")
    path.write_text(prompt)


VARIANTS = {"b2": lambda out: None, "b1-no-skill": b1_no_skill, "b2-think-off": thinking_off,
            "b1-think-off": b1_think_off, "b2-phases": phases, "b2-no-graph": no_graph,
            "b2-phases-adapter": phases_adapter, "b2-locator-tool": locator_tool, "b2.3": b2_3, "b2-explicit": b2_explicit, "b2-explicit-short": b2_explicit_short,
            "b2-explicit-no-graph": b2_explicit_no_graph,
            "b2-check": b2_check, "top": lambda out: None, "top-check": top_check,
            "top-think": top_think, "top-directed": top_directed,
            "top-mix-1k": top_mix(1024), "top-mix-2k": top_mix(2048),
            "top-mix-1k-an-off": top_mix_analyzer_off(1024),
            "top-mix-1k-an-off-tb0": top_mix_analyzer_off_tb0(1024), "bf": lambda out: None,
            "bf-habits": bf_habits, "bf-gov": bf_gov, "bf-gov-to": bf_gov_to, "bf-gov-tb": bf_gov_tb, "bf-gov-ed": bf_gov_ed,
            "bf-gov-ct": bf_gov_ct, "bf-gov-ti": bf_gov_ti, "bf-gov-ec": bf_gov_ec, "bf-gov-22": bf_gov_22,
            **{f"bf-gov-tb-{k}": (lambda f: lambda out: f(out, bf_gov_tb))(f)
               for k, f in (("ct", bf_gov_ct), ("ti", bf_gov_ti), ("ec", bf_gov_ec), ("22", bf_gov_22))},
            "bf-gov-tb-mp": bf_gov_tb_mp, "bf-gov-tb-mp-nohab": bf_gov_tb_mp_nohab, "bf-gov-tb-sl": bf_gov_tb_sl,
            "bf-gov-tb-lk": bf_gov_tb_lk, "bf-gov-tb-lk-nt": bf_gov_tb_lk_nt,
            "bf-gov-to-nt": bf_gov_to_nt, "bf-gov-to-nt-vr": bf_gov_to_nt_vr,
            "h24": lambda out: None, "h24-long": h24_long}
SOURCES = {"top": TOP_BUNDLE, "top-check": TOP_BUNDLE, "top-think": TOP_BUNDLE, "top-directed": TOP_BUNDLE,
           "top-mix-1k": TOP_BUNDLE, "top-mix-2k": TOP_BUNDLE,
           "top-mix-1k-an-off": TOP_BUNDLE, "top-mix-1k-an-off-tb0": TOP_BUNDLE, "bf": BUDGET_FIT,
           "bf-habits": BUDGET_FIT, "bf-gov": BUDGET_FIT, "bf-gov-to": BUDGET_FIT, "bf-gov-tb": BUDGET_FIT, "bf-gov-ed": BUDGET_FIT,
           "bf-gov-ct": BUDGET_FIT, "bf-gov-ti": BUDGET_FIT, "bf-gov-ec": BUDGET_FIT, "bf-gov-22": BUDGET_FIT,
           **{f"bf-gov-tb-{k}": BUDGET_FIT for k in ("ct", "ti", "ec", "22")},
           "bf-gov-tb-mp": BUDGET_FIT, "bf-gov-tb-mp-nohab": BUDGET_FIT, "bf-gov-tb-sl": BUDGET_FIT,
           "bf-gov-tb-lk": BUDGET_FIT, "bf-gov-tb-lk-nt": BUDGET_FIT,
           "bf-gov-to-nt": BUDGET_FIT, "bf-gov-to-nt-vr": BUDGET_FIT, "h24": H24, "h24-long": H24}


def build_variant(name: str, src: Path | None = None, dest_root: Path = ROOT / "build/variants") -> Path:
    """Build in a temp dir and publish only if every transformation succeeded (no half-built variants)."""
    out = dest_root / name
    tmp = dest_root / f".{name}.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.rmtree(out, ignore_errors=True)
    src = src or SOURCES.get(name, ROOT / "submission")
    shutil.copytree(src, tmp, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"))
    try:
        VARIANTS[name](tmp)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    tmp.rename(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name", choices=sorted(VARIANTS))
    args = ap.parse_args()
    print(build_variant(args.name))


if __name__ == "__main__":
    main()
