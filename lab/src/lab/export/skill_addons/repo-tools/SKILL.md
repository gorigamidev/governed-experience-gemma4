---
name: repo-tools
description: Your only way to read the repository and run commands. Call run_skill_script with skill_name "repo-tools", file_path "scripts/entrypoint.py" and args - look (with "search" or "path") reads a file range, lists a directory or searches (git grep); run (with "cmd") executes one shell command (tests, python -c). After 10 look/run calls without an edit, both stop returning new code until you edit a file.
---

# repo-tools

Every call has three arguments: skill_name is "repo-tools", file_path is "scripts/entrypoint.py", and args is an object. Each call costs one tool call. With "search" or "path" in args it looks; with "cmd" it runs. The repository file or directory goes in args as "path", never in file_path.

```
search:      run_skill_script(skill_name="repo-tools", file_path="scripts/entrypoint.py", args={"search": "<identifier or regex>"})
read lines:  run_skill_script(skill_name="repo-tools", file_path="scripts/entrypoint.py", args={"path": "<file.py>", "start": 120, "end": 180})
list a dir:  run_skill_script(skill_name="repo-tools", file_path="scripts/entrypoint.py", args={"path": "<directory>"})
run tests:   run_skill_script(skill_name="repo-tools", file_path="scripts/entrypoint.py", args={"cmd": "python -m pytest <tests/test_file.py> -k <name> -q"})
```

- search: git grep -n hits as path:line:text (*.py first; with no hits there, every tracked file). Add "path": "<directory>" to narrow it.
- read lines: content is lines 120-180 verbatim (max 150 lines), to copy old_string from.
- run: exit_code, stdout, stderr; stopped after 90 s.

## Read lock

Before your first edit, look and run allow 10 calls. After that they return "locked": true with the last lines you read instead of new ones; look still re-reads line ranges you already read, plus up to 3 new line ranges of any file you name with path, start and end (to copy an exact old_string); no searches, no directory listings and no run. Then call edit_file with your best fix (copy old_string exactly from what you already read). After any edit to a source file, look and run work again without limit, for testing and corrections.
