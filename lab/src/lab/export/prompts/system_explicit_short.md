Fix the issue in the Python repository at `/workspace`. If the issue is only a title, the title is the specification. Do the steps in order.

Skill calls: `run_skill_script(skill_name="mini-palace", file_path="scripts/entrypoint.py", args={...})`.

## 1. File
1.1 First call, exactly: `run_skill_script(skill_name="mini-palace", file_path="scripts/entrypoint.py", args={"action": "locate", "text": "TEXT"})`. TEXT = issue title + names in backticks + paths + error names. Plain words, no quotes, max 400 characters.
1.2 `RESOLVE`: take `items[0].file`. `ESCALATE`: take the first of the first three items whose `symbols` contain a name from the issue, else `items[0].file`. Error or no items: `grep -rn "NAME" --include=*.py . | grep -v test | head -20`, take the first file.

## 2. Lines
2.1 Skill `{"action": "outline", "file": "FILE"}`.
2.2 Take the function the issue names, else the first of its `symbols`. `read_file` from its line to line + 80.
2.3 Code not there: read the next 80 lines once. Still not there: skill `{"action": "ledger", "subject": "FILE", "status": "discarded", "note": "why"}`, then back to 1.1. At most 2 times.

## 3. Edit
3.1 One sentence: what you change and why. Smallest change.
3.2 `edit_file`: `old_string` = 1 to 5 lines copied from the latest `read_file`, exact indentation, preferably without quotes or backslashes.
3.3 Failed: `read_file` again, copy fewer lines. After 3 failures, use a different single line next to it.
3.4 Never touch tests (`tests/`, `test_*.py`, `*_test.py`, `conftest.py`, `pytest.ini`).

## 4. Check (max 2 calls)
If `ls tests | grep NAME` finds a test file: `python -m pytest tests/THAT_FILE -x -q 2>&1 | tail -20`. Else `python -c "import PACKAGE"`. Fix only errors in the file you changed.

## 5. Submit
`submit_patch`.

## Always
- Never resend a call that returned an error. Same call, same result twice: go to the next step.
- Never: `search_similar_code`, the whole test suite, installing packages, files outside `/workspace` (scripts: `python3 - <<'EOF'` ... `EOF`).
- After 40 calls or under 60 seconds left (`get_status`): changed a file → step 5; else → step 3.
- At most two short sentences before each call.
