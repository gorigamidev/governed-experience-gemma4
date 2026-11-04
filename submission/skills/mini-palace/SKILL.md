---
name: mini-palace
description: Locates the source files and functions to edit for the current issue, and keeps a ledger of what was already checked. Call run_skill_script with skill_name "mini-palace", file_path "scripts/entrypoint.py" and args {"action":"locate","text":"<issue title + key terms>"} as your FIRST step, before any grep or read_file.
---

# mini-palace

Fast, deterministic code localization for the repository in `/workspace`. It never edits files.
Every call costs one tool call, so use it instead of exploratory `grep`/`find`, not in addition to them.

## Actions

All calls: `run_skill_script(skill_name="mini-palace", file_path="scripts/entrypoint.py", args={...})`.

| args | Returns |
|---|---|
| `{"action": "locate", "text": "<title + identifiers, paths, error messages from the issue>"}` | Ranked candidate files with scores, best matching functions as `name:line`, and a `decision` |
| `{"action": "outline", "file": "<repo-relative path>"}` | Every class/function in the file as `qualname:line`, so you can `read_file` just the right range |
| `{"action": "ledger"}` | Candidates and files already checked or edited, plus your notes. Use it after the context gets compacted |
| `{"action": "ledger", "subject": "<file>", "status": "checked\|edited\|discarded", "note": "<short finding>"}` | Records progress |

## How to read `locate`

- `decision: RESOLVE`: the first file is a clear winner. Go straight to it; the other items are fallbacks.
- `decision: ESCALATE`: several files are plausible. Compare the `symbols` listed for each, `outline` the most likely one, and decide.
- `why` shows the evidence per component: `sym` (function/class names match), `path` (file path matches), `exact` (an identifier from the issue exists in that file), `prior` (file often changed by similar fixes).
- Pass `text` with the issue title plus concrete terms (identifiers in backticks, file paths, error names). Keep it under ~400 characters; do not paste the whole issue.
