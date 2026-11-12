---
name: focus
description: Finds the functions that mention the issue's identifiers or error text and returns their verbatim source in one call. Call run_skill_script with skill_name "focus", file_path "scripts/focus.py" and args {"terms":"name1, name2, error text"}.
---

# focus

One call instead of several `grep` and `read_file` calls. Returns the source of the enclosing functions in the two
best-matching non-test files (verbatim, to copy into `old_string`) plus other matching files. Never edits.
