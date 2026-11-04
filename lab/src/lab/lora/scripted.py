"""Plan a gold-guided coder trajectory (core B, doc 03 §14.25, amendment 1) from a task's reference patch.

The plan is the list of coder responses a scripted model server returns, in order. Tool outputs are never
written here: the real harness executes every call. Localization is a real `code_analyzer` answer taken from
our own traces, so nothing in the trajectory knows where the bug is before the analyzer says so.
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / "data/competition"
RESERVE = frozenset(
    "fastapi_14258 fastapi_14372 fastapi_15280 fastapi_15588 rich_2725 rich_3043 rich_3063 rich_3067 rich_3105 "
    "rich_3180 rich_3278 rich_3506 rich_3676 rich_3934 rich_3944 rich_4077".split())  # doc 03 §14.24
IMPOSSIBLE = frozenset(
    "fastapi_15661 fastapi_15030 rich_3930 fastapi_14609 fastapi_15785 fastapi_15745".split())  # doc 01, row 46
MAX_CHANGED_LINES = 60
NEW_SCORER_RUNS = ("kaggle-v7-top-a", "kaggle-v7-top-b", "kaggle-v7-top-check", "kaggle-think-probe-top",
                   "kaggle-think-probe-top-think", "kaggle-thinking-top", "kaggle-thinking-top-think")
READ_MARGIN = 3
MAX_READ_SPAN = 120


@dataclass
class Hunk:
    path: str
    old_start: int
    old_lines: list[str]  # context and removed lines, as in the base file
    new_lines: list[str]  # context and added lines


@dataclass
class Step:
    thought: str
    tool: str | None = None
    args: dict = field(default_factory=dict)
    content: str | None = None  # final text answer (no tool)
    loss: bool = True  # whether this coder turn carries gradient in the LoRA data (recovery turns that provoke an error carry False)

    def to_json(self) -> dict:
        return {"thought": self.thought, "tool": self.tool, "args": self.args, "content": self.content, "loss": self.loss}


def parse_patch(patch: str) -> list[Hunk]:
    hunks: list[Hunk] = []
    path = None
    cur: Hunk | None = None
    for line in patch.splitlines():
        if line.startswith("+++ "):
            path = line[6:] if line.startswith("+++ b/") else None
            cur = None
        elif line.startswith("--- "):
            cur = None
        elif line.startswith("@@"):
            m = re.match(r"@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@", line)
            cur = Hunk(path, int(m.group(1)), [], [])
            hunks.append(cur)
        elif cur is not None and line.startswith("\\"):
            continue
        elif cur is not None and line[:1] in (" ", "-", "+", ""):
            body = line[1:]
            if line[:1] in (" ", "", "-"):
                cur.old_lines.append(body)
            if line[:1] in (" ", "", "+"):
                cur.new_lines.append(body)
    return hunks


def changed_lines(patch: str) -> int:
    return len(re.findall(r"^[+-](?![+-]{2} )", patch, re.M))


def creates_files(patch: str) -> bool:
    return bool(re.search(r"^--- /dev/null$", patch, re.M))


def eligible(task: dict) -> str | None:
    """Reason a task is excluded from core B, or None."""
    iid = task["instance_id"]
    if iid in RESERVE:
        return "reserve"
    if iid in IMPOSSIBLE:
        return "impossible"
    if creates_files(task["patch"]):
        return "new_file"
    if changed_lines(task["patch"]) > MAX_CHANGED_LINES:
        return "big_patch"
    return None


def analyzer_answers(task: dict, eval_root: Path = ROOT / "data/eval") -> list[dict]:
    """First code_analyzer answer of each new-scorer run that names a file of the reference patch."""
    gold = {h.path for h in parse_patch(task["patch"])}
    out = []
    for run in NEW_SCORER_RUNS:
        f = eval_root / run / "traces" / f"trace_{task['instance_id']}.json"
        if not f.exists():
            continue
        for s in json.loads(f.read_text())["steps"]:
            obs = s.get("observation") or {}
            if (obs.get("extra") or {}).get("tool_name") == "code_analyzer":
                text = _analyzer_text(obs.get("content") or "")
                if any(g in text for g in gold):
                    out.append({"run": run, "text": text})
                break
    return out


def _analyzer_text(content: str) -> str:
    try:
        return json.loads(content).get("raw", content)
    except (json.JSONDecodeError, AttributeError):
        return content


def _unique_old_string(src_lines: list[str], h: Hunk) -> str:
    """The hunk's old side, widened with neighbouring base lines until it occurs once in the file."""
    text = "\n".join(src_lines)
    start = h.old_start - 1 if h.old_lines else h.old_start
    end = start + len(h.old_lines)
    assert src_lines[start:end] == h.old_lines, f"{h.path}:{h.old_start} hunk does not match the base file"
    lo, hi = start, end
    while text.count("\n".join(src_lines[lo:hi])) != 1 or not "".join(src_lines[lo:hi]).strip():
        if lo == 0 and hi == len(src_lines):
            raise ValueError(f"{h.path}:{h.old_start} cannot be made unique")
        lo, hi = max(0, lo - 1), min(len(src_lines), hi + 1)
    old = src_lines[lo:hi]
    new = src_lines[lo:start] + h.new_lines + src_lines[end:hi]
    return "\n".join(old), "\n".join(new)


def _read_ranges(hunks: list[Hunk]) -> list[tuple[int, int]]:
    ranges: list[list[int]] = []
    for h in hunks:
        a = max(1, h.old_start - READ_MARGIN)
        b = h.old_start + max(len(h.old_lines), 1) + READ_MARGIN
        if ranges and a - ranges[-1][1] <= 10 and b - ranges[-1][0] <= MAX_READ_SPAN:
            ranges[-1][1] = max(ranges[-1][1], b)
        else:
            ranges.append([a, b])
    return [(a, b) for a, b in ranges]


def title(problem: str) -> str:
    first = next((ln.strip(" #*") for ln in problem.splitlines() if ln.strip(" #*")), "the issue")
    return first[:160]


def plan(task: dict, snapshot_dir: Path, test_file: str | None) -> list[Step]:
    """Coder responses for one trajectory. `snapshot_dir` is the extracted base snapshot."""
    hunks = parse_patch(task["patch"])
    files = list(dict.fromkeys(h.path for h in hunks))
    steps = [Step(f"Issue: {title(task['problem_statement'])}. I'll ask code_analyzer for the location first.",
                  "code_analyzer", {"request": task["problem_statement"]})]
    edits: list[Step] = []
    for path in files:
        fh = [h for h in hunks if h.path == path]
        src = (snapshot_dir / path).read_text().split("\n")
        for a, b in _read_ranges(fh):
            lead = "The analyzer's answer points here." if not edits and not steps[1:] else "Next change is in this file."
            steps.append(Step(f"{lead} Read {path} lines {a}-{b} to confirm before editing.",
                              "read_file", {"filepath": path, "start_line": a, "end_line": b}))
        for h in fh:
            old, new = _unique_old_string(src, h)
            edits.append(Step(f"Edit {path} near line {h.old_start}, old_string copied exactly from what I read.",
                              "edit_file", {"filepath": path, "old_string": old, "new_string": new}))
        steps += edits
        edits = []
    py = [f for f in files if f.endswith(".py")]
    if py:
        steps.append(Step("Check that the edited files compile.", "run_command",
                          {"command": "python -m py_compile " + " ".join(py)}))
    if test_file:
        steps.append(Step("Run the existing tests for this module once.", "run_command",
                          {"command": f"python -m pytest {test_file} -x -q 2>&1 | tail -5"}))
    steps.append(Step("Check that the diff holds only the intended change.", "run_command", {"command": "git diff --stat"}))
    steps.append(Step("Submit the patch.", "submit_patch", {}))
    steps.append(Step("", content="DONE"))
    return steps


def apply_edits(snapshot_dir: Path, steps: list[Step]) -> None:
    for s in steps:
        if s.tool == "edit_file":
            p = snapshot_dir / s.args["filepath"]
            text = p.read_text()
            assert text.count(s.args["old_string"]) == 1, s.args["filepath"]
            p.write_text(text.replace(s.args["old_string"], s.args["new_string"]))


def extract(task_id: str, dest: Path) -> Path:
    subprocess.run(["tar", "xzf", str(DATA / "snapshots" / f"{task_id}.tgz"), "-C", str(dest)], check=True)
    return dest


def edits_reproduce_reference(task: dict, steps: list[Step]) -> bool:
    """Applying the planned edits to the base snapshot yields exactly the reference patch's files."""
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        ours, ref = extract(task["instance_id"], Path(a)), extract(task["instance_id"], Path(b))
        apply_edits(ours, steps)
        subprocess.run(["git", "apply", "--whitespace=nowarn", "-"], cwd=ref, input=task["patch"].rstrip("\n") + "\n", text=True, check=True)
        files = {h.path for h in parse_patch(task["patch"])}
        return all((ours / f).read_text() == (ref / f).read_text() for f in files)


def module_test_file(snapshot_dir: Path, files: list[str]) -> str | None:
    """tests/test_<stem>.py for the first edited module that has one."""
    for f in files:
        stem = Path(f).stem
        for cand in (f"tests/test_{stem}.py", f"tests/test_{stem.lstrip('_')}.py"):
            if (snapshot_dir / cand).exists():
                return cand
    return None


def _perturb_old_string(src_lines: list[str], old: str) -> str:
    """Return a near copy of `old` that does not occur in the file (recovery R1).

    Whitespace-only changes are avoided on purpose: the harness's edit_file tries a flexible
    match that ignores trailing whitespace, so they could still apply and silently mutate the
    file instead of raising the real error we want to show.
    """
    text = "\n".join(src_lines)
    marker = old + "\n# __broken_old_string__"
    assert text.count(marker) == 0, "marker already present"
    return marker


def plan_r1(task: dict, snapshot_dir: Path, test_file: str | None) -> list[Step]:
    """R1: one edit_file carries an old_string that is absent -> real error, then re-read and correct edit."""
    hunks = parse_patch(task["patch"])
    files = list(dict.fromkeys(h.path for h in hunks))
    steps = [Step(f"Issue: {title(task['problem_statement'])}. I'll ask code_analyzer for the location first.",
                   "code_analyzer", {"request": task["problem_statement"]})]
    for path in files:
        fh = [h for h in hunks if h.path == path]
        src = (snapshot_dir / path).read_text().split("\n")
        for a, b in _read_ranges(fh):
            steps.append(Step(f"Read {path} lines {a}-{b} to confirm before editing.", "read_file",
                               {"filepath": path, "start_line": a, "end_line": b}))
    first_file = files[0]
    first_hunks = [h for h in hunks if h.path == first_file]
    src0 = (snapshot_dir / first_file).read_text().split("\n")
    broken_old, _ = _unique_old_string(src0, first_hunks[0])
    broken_old = _perturb_old_string(src0, broken_old)
    # The turn that provokes the error carries no loss (doc 03 §14.31).
    steps.append(Step("Apply the first change (this edit should be rejected because old_string is absent).",
                       "edit_file", {"filepath": first_file, "old_string": broken_old,
                                     "new_string": _unique_old_string(src0, first_hunks[0])[1]}, loss=False))
    for a, b in _read_ranges(first_hunks):
        steps.append(Step(f"Re-read {first_file} lines {a}-{b} after the failed edit to copy the exact text.",
                           "read_file", {"filepath": first_file, "start_line": a, "end_line": b}))
    edits: list[Step] = []
    for path in files:
        fh = [h for h in hunks if h.path == path]
        src = (snapshot_dir / path).read_text().split("\n")
        for h in fh:
            old, new = _unique_old_string(src, h)
            edits.append(Step(f"Edit {path} near line {h.old_start} with the exact old_string copied from the file.",
                               "edit_file", {"filepath": path, "old_string": old, "new_string": new}))
        steps += edits
        edits = []
    py = [f for f in files if f.endswith(".py")]
    if py:
        steps.append(Step("Check that the edited files compile.", "run_command",
                           {"command": "python -m py_compile " + " ".join(py)}))
    if test_file:
        steps.append(Step("Run the existing tests for this module once.", "run_command",
                           {"command": f"python -m pytest {test_file} -x -q 2>&1 | tail -5"}))
    steps.append(Step("Check that the diff holds only the intended change.", "run_command", {"command": "git diff --stat"}))
    steps.append(Step("Submit the patch.", "submit_patch", {}))
    steps.append(Step("", content="DONE"))
    return steps


def plan_r2(task: dict, snapshot_dir: Path, test_file: str | None) -> list[Step]:
    """R2: an edit_file missing a mandatory argument -> ADK rejection, then a complete resend."""
    hunks = parse_patch(task["patch"])
    files = list(dict.fromkeys(h.path for h in hunks))
    steps = [Step(f"Issue: {title(task['problem_statement'])}. I'll ask code_analyzer for the location first.",
                   "code_analyzer", {"request": task["problem_statement"]})]
    for path in files:
        fh = [h for h in hunks if h.path == path]
        src = (snapshot_dir / path).read_text().split("\n")
        for a, b in _read_ranges(fh):
            steps.append(Step(f"Read {path} lines {a}-{b} to confirm before editing.", "read_file",
                               {"filepath": path, "start_line": a, "end_line": b}))
    first_file = files[0]
    first_hunks = [h for h in hunks if h.path == first_file]
    src0 = (snapshot_dir / first_file).read_text().split("\n")
    old0, new0 = _unique_old_string(src0, first_hunks[0])
    # Mandatory argument (`new_string`) absent -> ADK rejects without executing. No loss.
    steps.append(Step("Send an incomplete edit_file call (missing new_string) to reproduce the rejection.",
                       "edit_file", {"filepath": first_file, "old_string": old0}, loss=False))
    steps.append(Step("Resend the edit_file call in full after the rejection.", "edit_file",
                       {"filepath": first_file, "old_string": old0, "new_string": new0}))
    edits: list[Step] = []
    for path in files[1:]:
        fh = [h for h in hunks if h.path == path]
        src = (snapshot_dir / path).read_text().split("\n")
        for h in fh:
            old, new = _unique_old_string(src, h)
            edits.append(Step(f"Edit {path} near line {h.old_start} with the exact old_string copied from the file.",
                               "edit_file", {"filepath": path, "old_string": old, "new_string": new}))
        steps += edits
        edits = []
    # If the first file has more than one hunk, apply the rest now.
    if len(first_hunks) > 1:
        src = (snapshot_dir / first_file).read_text().split("\n")
        for h in first_hunks[1:]:
            old, new = _unique_old_string(src, h)
            steps.append(Step(f"Edit {first_file} near line {h.old_start}.", "edit_file",
                               {"filepath": first_file, "old_string": old, "new_string": new}))
    py = [f for f in files if f.endswith(".py")]
    if py:
        steps.append(Step("Check that the edited files compile.", "run_command",
                           {"command": "python -m py_compile " + " ".join(py)}))
    if test_file:
        steps.append(Step("Run the existing tests for this module once.", "run_command",
                           {"command": f"python -m pytest {test_file} -x -q 2>&1 | tail -5"}))
    steps.append(Step("Check that the diff holds only the intended change.", "run_command", {"command": "git diff --stat"}))
    steps.append(Step("Submit the patch.", "submit_patch", {}))
    steps.append(Step("", content="DONE"))
    return steps


def plan_r3(task: dict, snapshot_dir: Path, test_file: str | None) -> list[Step] | None:
    """R3: apply only the first hunk -> failing real test -> apply the rest -> passing test -> submit. None if not applicable."""
    hunks = parse_patch(task["patch"])
    if len(hunks) < 2 or not test_file:
        return None
    files = list(dict.fromkeys(h.path for h in hunks))
    steps = [Step(f"Issue: {title(task['problem_statement'])}. I'll ask code_analyzer for the location first.",
                   "code_analyzer", {"request": task["problem_statement"]})]
    for path in files:
        fh = [h for h in hunks if h.path == path]
        src = (snapshot_dir / path).read_text().split("\n")
        for a, b in _read_ranges(fh):
            steps.append(Step(f"Read {path} lines {a}-{b} to confirm before editing.", "read_file",
                               {"filepath": path, "start_line": a, "end_line": b}))
    # First hunk only.
    first = hunks[0]
    src0 = (snapshot_dir / first.path).read_text().split("\n")
    old0, new0 = _unique_old_string(src0, first)
    steps.append(Step(f"Apply only the first hunk of {first.path}.", "edit_file",
                       {"filepath": first.path, "old_string": old0, "new_string": new0}, loss=False))
    py = [f for f in files if f.endswith(".py")]
    if py:
        steps.append(Step("Check that the edited files compile after the first hunk.", "run_command",
                           {"command": "python -m py_compile " + " ".join(py)}))
    steps.append(Step("Run the module tests after the partial patch.", "run_command",
                       {"command": f"python -m pytest {test_file} -x -q 2>&1 | tail -8"}))
    # Remaining hunks.
    edits: list[Step] = []
    for h in hunks[1:]:
        src = (snapshot_dir / h.path).read_text().split("\n")
        old, new = _unique_old_string(src, h)
        edits.append(Step(f"The test still fails: the remaining change in {h.path} is missing; apply it now.",
                           "edit_file", {"filepath": h.path, "old_string": old, "new_string": new}))
    steps += edits
    if py:
        steps.append(Step("Check that all edited files compile.", "run_command",
                           {"command": "python -m py_compile " + " ".join(py)}))
    steps.append(Step("Re-run the module tests after the complete patch.", "run_command",
                       {"command": f"python -m pytest {test_file} -x -q 2>&1 | tail -8"}))
    steps.append(Step("Check that the diff holds only the intended change.", "run_command", {"command": "git diff --stat"}))
    steps.append(Step("Submit the patch.", "submit_patch", {}))
    steps.append(Step("", content="DONE"))
    return steps
