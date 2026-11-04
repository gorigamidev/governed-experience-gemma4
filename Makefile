PY := lab/.venv/bin/python
# Kaggle CLI with the API token from .env (never printed, never committed)
KAGGLE := set -a; [ -f .env ] && . ./.env; set +a; kaggle
export PYTHONPATH := lab/src
export PYTHONDONTWRITEBYTECODE := 1

.PHONY: test analyze symbols store ingest-eval consolidate calibrate bench package variant sandbox-image sandbox-test eval-local verify-patches eval-notebook traces wheels-local preflight temporal stop-replay lora-b lora-reserve lora-audit lora-smoke tpu-smoke lora-rehearsal lora-31b lora-effect failure-split all

test:
	$(PY) -m unittest discover -s lab/tests

analyze:
	$(PY) -m lab.analysis.task_profile > /dev/null && echo "data/analysis/task_profile.md"
	$(PY) -m lab.analysis.graph_coverage > /dev/null && echo "data/analysis/graph_coverage.md"

# Offline plane: experience -> LINALDB -> verified habits -> references/
# Exact symbols per task from the history of the newest snapshot of each repo (no extra downloads)
symbols:
	$(PY) -m lab.ingest.reconstruct_symbols

store:
	$(PY) -m lab.store.ingest_reference

# Agent experience from a swegemma eval results dir: make ingest-eval RESULTS=path/to/results LABEL=name
ingest-eval:
	$(PY) -m lab.store.ingest_eval --results $(RESULTS) --label $(or $(LABEL),local)

consolidate: store
	$(PY) -m lab.consolidate.patterns --deploy

calibrate:
	$(PY) -m lab.calibrate.router_config --write

bench:
	$(PY) -m lab.analysis.localization_bench

package:
	$(PY) -W ignore -m lab.export.package

# Ablation variant, validated and wrapped in its own Kaggle notebook: make variant VARIANT=b1-no-skill
# MODEL=12b builds the 12B package under build/<variant>-12b/ and its own slug; default 31b keeps every name as-is.
MODEL ?= 31b
MODEL_SUFFIX := $(shell PYTHONPATH=lab/src $(PY) -c "from lab.export.models import variant_suffix; print(variant_suffix('$(MODEL)'))" 2>/dev/null)
variant:
	@test -n "$(MODEL_SUFFIX)" -o "$(MODEL)" = 31b -o "$(MODEL)" = gemma-4-31b-it-qat-w4a16-ct || { echo "MODEL=$(MODEL): unknown or suffix lookup failed; refusing to overwrite the 31B build"; exit 1; }
	$(PY) -m lab.export.variants $(VARIANT)
	$(PY) -W ignore -m lab.export.package --src build/variants/$(VARIANT) --model $(MODEL) --build build/$(VARIANT)$(MODEL_SUFFIX)/submission --zip build/$(VARIANT)$(MODEL_SUFFIX)/submission.zip
	$(PY) -m lab.export.kaggle_notebook --zip build/$(VARIANT)$(MODEL_SUFFIX)/submission.zip --out build/$(VARIANT)$(MODEL_SUFFIX)/kaggle_kernel --slug gemma4-agent-$(VARIANT)$(MODEL_SUFFIX)

# Skill inside the official sandbox image (Docker): network none, 4 GiB, 2 CPUs
sandbox-image:
	docker build -q -f data/competition/docker/Dockerfile.sandbox -t swebench-sandbox:local data/competition/docker

sandbox-test: package
	$(PY) -m lab.tools.sandbox_skill_test --snapshot $(or $(SNAPSHOT),data/competition/snapshots/rich_4079.tgz)

# Agent on a few tasks with a local OpenAI-compatible server (pipeline test, stored as pipeline_test):
#   make eval-local TASKS=rich_4079 MODEL=qwen3:8b   (needs `ollama serve` and the amd64 sandbox image)
eval-local:
	$(PY) -m lab.eval.run_eval --task-ids $(TASKS) --api-base $(or $(API),http://localhost:11434/v1) --served-model $(or $(MODEL),qwen3:8b) --label local-$(subst :,-,$(or $(MODEL),qwen3:8b))

# Public wheels + the test deps they lack (typing_inspection for pydantic, inline-snapshot for fastapi tests)
wheels-local:
	rm -rf data/competition/wheels_local && cp -R data/competition/wheels data/competition/wheels_local
	$(PY) -m pip download -q --only-binary=:all: --python-version 3.13 --platform manylinux2014_x86_64 --platform any -d data/competition/wheels_local typing_inspection inline-snapshot

# Phase 2 again with the local Docker sandbox: make verify-patches RESULTS=data/eval/<run>
verify-patches:
	$(PY) -m lab.eval.verify_patches --results $(RESULTS) --download-missing

# GPU notebook (4x L4, scorer stack) on a stratified task sample, pushed to the Kaggle queue
eval-notebook: package
	$(PY) -m lab.export.kaggle_eval_notebook $(if $(TASK_IDS),--task-ids $(TASK_IDS),) $(if $(ARMS),--arms $(ARMS),)
	$(KAGGLE) kernels push -p build/kaggle_eval

# Localization with habits mined only from earlier tasks, against the shuffled CV:
temporal:
	$(PY) -W ignore -m lab.analysis.temporal_split

# Stop rules (max_turns, first submit, failed self-checks) replayed on processed runs in data/eval:
stop-replay:
	$(PY) -m lab.analysis.stop_replay

# Behavior LoRA core B (doc 03 §14.25): scripted trajectories through the real harness -> data/lora/b
lora-b:
	$(PY) -m lab.lora.gen_b $(if $(TASKS),--tasks $(TASKS),)

# Turn-0 requests of the 16 reserve tasks (evaluation prompts for the rehearsal) -> data/lora/reserve
lora-reserve:
	$(PY) -m lab.lora.gen_b --reserve-prompts

# Rule 1 data audit (needs lab/.venv-train with transformers) -> data/analysis/lora_data_audit.md
lora-audit:
	lab/.venv-train/bin/python lab/src/lab/lora/audit.py $(if $(VLLM_PARSER),--vllm-parser $(VLLM_PARSER),)

# E4B pipeline smoke notebook (needs data/lora/b for the harness's tool schemas) -> build/kaggle_lora_smoke
lora-smoke:
	$(PY) -m lab.export.kaggle_lora_smoke_notebook

# TPU smoke test of the LoRA recipe on the 31B (script kernel): make tpu-smoke -> build/tpu-smoke/
tpu-smoke:
	$(PY) -m lab.export.kaggle_tpu_smoke

# Local gate before anything goes to Kaggle (AGENTS.md, step 6). Each arm is a submission.zip:
#   make preflight ARMS=b2=build/submission.zip,b1=build/b1-no-skill/submission.zip NOTEBOOK=build/kaggle_eval
preflight: test package
	$(PY) -W ignore -m lab.tools.preflight $(if $(ARMS),--arms $(ARMS),) $(if $(NOTEBOOK),--notebook $(NOTEBOOK),) $(if $(TASKS),--tasks $(TASKS),) $(if $(SKIP_RUN),--skip-run,) $(if $(MAX_MINUTES),--max-minutes $(MAX_MINUTES),)

# Finished GPU notebook -> download, re-verify (Docker), ingest, report:
#   make traces KERNEL=gorigamidev/gemma4-agent-traces LABEL=kaggle-v1
traces:
	set -a; [ -f .env ] && . ./.env; set +a; $(PY) -m lab.eval.process_traces --kernel $(KERNEL) --label $(LABEL)

all: test analyze bench package

# E4B rehearsal notebook, rule 2 (needs the private datasets gemma4-lora-wheels and gemma4-lora-data) -> build/kaggle_lora_rehearsal
#   make lora-rehearsal PACKAGE=build/top-mix-1k-an-off/submission.zip   (default: build/top-mix-1k/submission.zip)
lora-rehearsal:
	$(PY) -m lab.export.kaggle_lora_rehearsal_notebook $(if $(PACKAGE),--package $(PACKAGE),)

# 31B behavior LoRA, rule 3 (launch only after the E4B rehearsal passes) -> build/kaggle_lora_31b
lora-31b:
	$(PY) -m lab.export.kaggle_lora_rehearsal_notebook --model 31b $(if $(PACKAGE),--package $(PACKAGE),)

# Rule 4 only, with the adapter from the 31B notebook's output (no retraining) -> build/kaggle_lora_effect
lora-effect:
	$(PY) -m lab.export.kaggle_lora_effect_notebook $(if $(PACKAGE),--package $(PACKAGE),)

# First pass of the failure autopsy (experiments/autopsy/METHOD.md) -> data/analysis/failure_split.md
failure-split:
	$(PY) -m lab.analysis.failure_split $(if $(RUNS),--runs $(RUNS),) $(if $(RESERVE),--include-reserve,)
