# External agent bundles

Third-party submission bundles used as a base for variants. Kept byte-identical to the source; our changes are
applied by `lab/export/variants.py`, never by editing these files.

| Folder | Source | Original `submission.zip` sha256 | Public LB |
|---|---|---|---|
| `top-bundle/` | Kaggle notebook `romanrozen/gemma-eda-baseline-for-a-start-lb-top-1` (Roman Rozen), output of 2026-09-23 | `f3534769cd8c…` | 0.12 and 0.15 (byte-identical copy by another team) |
| `budget-fit/` | Kaggle notebook `verracodeguacas/gemma-4-agent-budget-fit-single-agent` (Wilmer E. Henao), output downloaded 2026-10-15 | `1cb73986bd9a…` | 0.18 |
| `h24/` | Kaggle notebook `honghanhhh/gemma-4-baseline-lb-0-24`, output downloaded 2026-10-17. Same `agent.yaml`, `eval_config.yaml` and `configs/sampling.yaml` as `budget-fit/` byte for byte; only `prompts/system.md` differs | `277b20147e89…` | 0.24 |

Public Kaggle notebooks are released under the Apache 2.0 license. Cite the author in the paper when a result builds on this base.
