# Errata

`paper.md` is kept as submitted to the Paper Track (v0.9). This page corrects it.

## 2026-11-25: resolved counts in Table 2

**What the paper says.** The R columns of Table 2 count resolved tasks out of 24, using the verdict of the harness in our evaluation session.

**What was wrong.** For 11 of those 24 tasks the reference patch does not pass in our Docker environment. In the session log we opened, 6 fastapi tasks fail at import because `inline_snapshot` or `dirty_equals` is missing, and 3 requests tasks fail on the `httpbin` fixture. On those 9 tasks any patch was counted as unresolved.

**What we did.** We rebuilt one test environment per repository (Python 3.12; versions in `experiments/reverify/`) and re-ran the stored patches. A task counts only if the reference patch passes and the empty patch fails: 23 of the 24 pool tasks now do. No new model runs were made. Scripts: `experiments/autopsy/reverify_base.py` and `reverify_patches.py`.

| Pool | Base | Variant | R as published (24 tasks) | R re-verified (23 tasks) |
|---|---|---|---|---|
| 1 | `bf` | `h24` | 4 / 6 | not re-verified |
| 2 | `bf` | `bf-gov` | 6 / 5 | 8 / 11 |
| 3 | `bf-gov` | `bf-gov-to` | 4 / 6 | 8 / 10 |
| 4 | `bf-gov-to` | `bf-gov-tb` | 6 / 8 | 9 / 11 |
| 5 | `bf-gov-tb` | `bf-gov-tb-22` | 5 / 5 | 8 / 10 |

**What changes.** Absolute resolved counts rise. Over the 324 stored 31B runs on the 23 tasks (seven pools, including ones not in Table 2), 129 are resolved instead of 80. No run counted as resolved by the harness is lost.

**What does not change.** In pools 2 to 5 the resolved-task part of the "not worse" rule still holds. The other parts (stray files, minutes, session timeouts) do not depend on the tests, so the Table 2 decisions in those pools stand. The noise paragraph uses only the 11 tasks that were already verifiable. Public scores come from the competition scorer. Differences of two or three tasks from one session per arm are still within noise.

**Limits.** The environments use Python 3.12, not the 3.13 of the official sandbox image, and package versions were chosen with the reference patches, not with the agent's patches.
