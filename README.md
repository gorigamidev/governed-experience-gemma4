# Governed Experience for an Offline Gemma 4 Coding Agent

Code for the Kaggle *Gemma 4 Developer Agent* Paper Track writeup ([`paper/paper.md`](paper/paper.md)): admitting an
agent's experience through rules written before the data, measuring each rule by whether the quantized model follows
it, and deciding when memory may override the model.

Competition Data is not redistributed (competition rules, 2.4.b): no tasks, snapshots, traces or patches are in this
repository. Agent traces embed task statements, so the pool results are reproduced by re-running the pools, not from
shipped traces. The paper reports only aggregates.

## Layout

| Path | What it is |
|---|---|
| `submission/` | Runtime plane: the agent configuration and the `mini-palace` skill (standard library only, no network) |
| `lab/src/lab/` | Offline plane: localization benchmark, habit consolidation (with LINALDB lineage), variant builder, Kaggle notebook builders, local evaluation, trace analyses |
| `lab/tests/` | Unit tests (CI: Python 3.13 with `linaldb==0.1.22`; tests that need the competition wheels skip) |
| `external/` | Third-party public bundles used as bases, byte-identical to their sources (see `external/README.md`) |
| `experiments/autopsy/` | Failure-autopsy metrics used to derive procedure rules |

Comments of the form `doc 03 §14.N` point to entries of our private design log, where each decision rule was written
before its run; the rules themselves are implemented in the analysis module that applies them.

## Setup

```bash
kaggle competitions download gemma-4-developer-agent -p data/competition   # tasks.jsonl, snapshots/, graphs/, wheels/
python3 -m venv lab/.venv && lab/.venv/bin/pip install -e "lab[store]"
# For local agent runs, also install the competition wheelhouse (google-adk, adk_submission, swegemma) from data/competition/wheels.
make test
```

## Reproducing the paper

| Paper | Command | Needs |
|---|---|---|
| Table 1 (localization, nested CV) | `make symbols && make bench` → `data/analysis/localization_bench.md` | Competition Data |
| §4.1 temporal split | `make temporal` | Competition Data |
| Table 2 pools: arms | `make variant VARIANT=<arm>` for `bf`, `bf-gov`, `bf-gov-to`, `bf-gov-tb`, `bf-gov-tb-22`; `h24` = `external/h24` | |
| Table 2 pools: runs | `python -m lab.export.kaggle_eval_notebook --arms a=zip,b=zip --interleave --task-ids ...`, run on 4× L4 | Kaggle GPU quota |
| Table 2 pools: rules | `python -m lab.analysis.pool_report --arms a=<run>,b=<run>`; mechanisms: `lever_report`, `command_timeouts`, `long_turns`, `edit_deadline` | Pool outputs |
| §4.2 E4B diagnostic and memory test | `make preflight` with Ollama `gemma4-e4b-32k` (Gemma 4 E4B, `num_ctx` 32768); `python -m lab.analysis.memory_e4b`, `failure_split` | Ollama, Competition Data |
| §4.2 second-opinion simulation | `python -m lab.analysis.second_opinion_sim` | Two E4B base runs, Competition Data |
| §4.2 conditional authority (31B and E4B) | phase A: `python -m lab.analysis.authority_split`; phase B: `make preflight` (E4B, base against `bf-gov-tb-mp-nohab`) then `python -m lab.analysis.authority_phase_b --stamp <run>` | Pool traces; Ollama, Competition Data |
| §4.4 hidden thinking with `enable_thinking: False` | `python -m lab.analysis.hidden_thinking` | Pool outputs |
| §4.4 repetition inside an episode | `python -m lab.analysis.self_imitation` | Pool outputs |
| §4.4 where tasks are lost, edits after the warning | `python -m lab.analysis.near_miss`, `refused_edits` | Pool outputs, Docker |
| §4.4 read lock (archived) | `python -m lab.analysis.lock_sim` (offline), `lock_report` (probe runs) | Pool outputs, Kaggle GPU quota |
| Table 3 defects | `lab.analysis.graph_coverage`, `make verify-patches`, `lab.analysis.pool_report` (failed-edit split) | Competition Data, Docker |

All commands run from the repository root with `PYTHONPATH=lab/src` (the `Makefile` sets it).

## License

Apache License 2.0 (`LICENSE`). The bundles in `external/` keep their authors' attribution (`NOTICE`).
LINALDB (https://github.com/gorigami/linaldb) is Apache 2.0 from version 0.1.20.
