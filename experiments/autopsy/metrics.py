#!/usr/bin/env python3
"""Mechanical metrics for the failure autopsy (METHOD.md step 3 evidence).

Prints one markdown table row per attempt in scope (failure_split.md, minus
resolved and timeout): steps, tool counts, longest identical call run (loop),
whether anything ran after the last successful edit (V), whether the last edit
failed and nothing ran after it (G), and auxiliary files shipped in the patch.

Usage: python experiments/autopsy/metrics.py
"""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCOPE_FILE = ROOT / "data" / "analysis" / "failure_split.md"
LOOP_MIN = 5
AUX_RE = re.compile(r"(repro|_issue|test_graphemes)")


def scope_attempts():
    for line in SCOPE_FILE.read_text().splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) > 4 and parts[1].startswith("data/"):
            run = parts[1].split("/")[2]
            task, bucket = parts[2], parts[3]
            if bucket == "resolved" or bucket.startswith("timeout"):
                continue
            yield task, run, bucket


def norm_obs(obs):
    obs = str(obs)
    obs = re.sub(r"'elapsed_s': [0-9.]+, ", "", obs)
    obs = re.sub(r"'timestamp': [0-9.]+, ", "", obs)
    obs = re.sub(r"'budget_warning': \"[^\"]*\",?", "", obs)
    return obs


def metrics(task, run):
    trace = json.loads((ROOT / "data" / "eval" / run / "traces" / f"trace_{task}.json").read_text())
    counts = Counter()
    groups = defaultdict(list)
    first_submit = None
    for step in trace["steps"]:
        for call in step.get("tool_calls") or []:
            if call.get("function_name") == "submit_patch":
                first_submit = step["step_id"]
                break
        if first_submit is not None:
            break
    last_edit = last_edit_ok = None
    exec_after_last_edit = False
    for step in trace["steps"]:
        pre_submit = first_submit is None or step["step_id"] < first_submit
        for call in step.get("tool_calls") or []:
            fn = call.get("function_name")
            counts[fn] += 1
            args = json.dumps(call.get("arguments"), sort_keys=True)
            groups[(fn, args, norm_obs(step.get("observation")))].append(step["step_id"])
            if not pre_submit:
                continue
            if fn == "edit_file":
                last_edit = step["step_id"]
                if re.search(r"'status': 'ok'|\"status\": \"ok\"", str(step.get("observation"))):
                    last_edit_ok = step["step_id"]
            elif fn == "run_command" and last_edit is not None and step["step_id"] > last_edit:
                exec_after_last_edit = True
    loop = max(groups.values(), key=len)
    loop_len = len(loop) if len(loop) >= LOOP_MIN else 0
    v = last_edit is not None and first_submit is not None and not exec_after_last_edit
    g = v and last_edit != last_edit_ok
    patch = ROOT / "data" / "eval" / run / "patches" / f"{task}.patch"
    aux = []
    if patch.exists() and patch.stat().st_size:
        for m in re.finditer(r"^diff --git a/(\S+)", patch.read_text(), re.M):
            aux.append(m.group(1))
    aux = [f for f in aux if AUX_RE.search(f.split("/")[-1])]
    return counts, loop_len, v, g, aux, len(trace["steps"])


def main():
    print("| Tarea | Corrida | Bucket | Pasos | edit | F | V | G | Aux en parche |")
    print("|---|---|---|---|---|---|---|---|---|")
    for task, run, bucket in sorted(scope_attempts()):
        counts, loop_len, v, g, aux, steps = metrics(task, run)
        print(f"| {task} | {run.replace('kaggle-', '')} | {bucket} | {steps} | "
              f"{counts['edit_file']} | {loop_len or '-'} | {'x' if v else ''} | "
              f"{'x' if g else ''} | {', '.join(aux) or '-'} |")


if __name__ == "__main__":
    sys.exit(main())
