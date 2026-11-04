# Governed Experience for an Offline Gemma 4 Coding Agent: What Gets Obeyed, and What Earns Authority

**Subtitle:** Rules written before the data, compliance as the measure of a rule, and memory that may override the model only where it is measurably more accurate.

**Code:** https://github.com/gorigamidev/governed-experience-gemma4

## Abstract

Offline coding agents built on quantized open models work under tight budgets: a 31B model with 4-bit weights, a 32K-token context and about 120 hidden tasks in 12 hours. We treat what such an agent learns from past runs as governed data, admitted only through rules written before the evidence, with lineage and open corrections, and we separate three questions: is the experience true, does the model follow it, and should it override the model. In five paired pools on the scorer's hardware, structure enforced by the harness (a thinking cap, a tool-call cap) was followed in every pool. Prompt rules were followed when the action was visible in the call (wrapping commands in `timeout`: 94%; stray files in the patch: 10 to 0) and not when they needed state the model could not see (the call number), and the same prompt failed on another serving stack. Content memory (where fixes land) raised the gold file into the top five from 64% to 85% of 128 tasks offline and did not move the 31B agent in two early runs. Gemma 4 E4B alone picked the right file in 18 of 24 tasks, more often than the memory ranked it first (14), and told to follow the memory it fell to 15 in one local run. As a second opinion the memory lost more files than it gained (9 against 5). Same-zip noise of three tasks bounds every comparison; we report the defects that would have changed our decisions.

## 1. Introduction

Under the constraints of the Gemma 4 Developer Agent competition (Gemma 4 31B in W4A16, four L4 GPUs, no network, about 120 hidden tasks within 12 hours) every model call is expensive and the model is hard to change: no adapter reported in the competition, ours included, beat the base model.

We study a second lever: the experience the agent works with, such as where fixes usually land or which habits waste the budget. Memory updated without quality control degrades [Zhang et al., 2026], a sound rule can be ignored by the model or conflict with its tools, and a memory right on average can be worse than the model it advises.

Contributions:

1. **A governance protocol** for agent experience: decision rules written before the evidence, admission through paired tests, lineage, and corrections recorded as corrections.
2. **Compliance as the measure of procedural experience**: five paired pools showing which kinds of rules a quantized model follows, and that compliance depends on the serving stack.
3. **Authority as a separate decision**: content memory that helps offline made a small model worse when it led; comparing the memory's accuracy with the model's own, which we did only afterwards, shows this offline before any run.
4. **An audit of defects** in data, harness, local testbed and our own metrics, each tied to the decision it would have changed.

## 2. Setting

**Harness.** A submission is a declarative ADK agent configuration (prompt, sampling, per-task budgets, optional skills) served by vLLM on `gemma-4-31b-it-qat-w4a16-ct`; Gemma 4 E4B is also allowed. The model edits a repository through nine tools; the working tree becomes the patch, which passes only if all hidden tests pass. There are no callbacks: nothing runs between turns unless the model calls a tool. The scorer changed on 30 September; agent results below are dated.

**Budget.** The strongest public configuration, Budget-Fit [Henao, 2026], uses one agent, a 4,096-token thinking budget, 28 tool calls, 8 minutes per task and a prompt rule to edit by the twelfth call. It is our base (`bf`).

**Noise.** The public score is the floor of the percentage of 58 tasks. The same `bf` zip scored 10 and 9 of 58 on our account, and other teams report deviations of 2.5 to 3 tasks. In our pools the same arm changed by up to three resolved tasks between sessions. We decide only from paired runs with "not worse" thresholds, and read public scores only against an alarm written in advance.

**Tasks.** The 129 public tasks come from four repositories; the hidden ones come from private repositories.

## 3. Method

### 3.1 Governance protocol

Every decision is written before the data that decides it: arms, tasks, metrics, thresholds, and the next submission under each outcome. A rule is never changed after seeing the result that would change it. Admitted experience keeps its lineage (support, task ids, runs, parameters) in LINALDB, a store with content-hashed lineage [Gorigami, 2026]. A wrong number is corrected by stating what it said, what it says and why.

### 3.2 Two kinds of experience

**Content.** Habits mined per repository from fixes (a file that recurs across tasks, or one that co-occurs with a statement term) adjust a lexical ranking computed by a deterministic standard-library skill inside the sandbox. A habit type is admitted through nested cross-validation and a complexity ladder: it must beat the last accepted configuration on a paired per-task objective (sign test, p ≤ 0.10) with the 95% bootstrap interval of the MRR change above −0.01.

**Procedure.** Candidate rules come from an autopsy of failed runs coded with a fixed taxonomy (no edit before the budget ends, stray scripts in the patch, hung commands, failed edits retried). A code becomes a candidate only if its mechanism can be checked on recorded traces; the vehicle (prompt rule, budget field, tool) follows the mechanism.

### 3.3 Measurement

**Paired pools.** Two arms run in one session on four L4 GPUs, interleaved per task, on the same 24 tasks. A variant replaces the base if resolved tasks R drop by at most one, stray files (aux) rise by at most one (none in pools 1 and 2), mean minutes stay within 10%, and the failure the variant targets (hangs, time cuts or session timeouts) does not rise. Each pool also defines a compliance metric for the rule it tests, computed from the traces. Patches are re-run in the official sandbox image; a task is verifiable if its reference patch passes and an empty patch fails.

## 4. Experiments

### 4.1 Content experience helps localization offline

On exact symbol tables at each task's base commit, file habits raise the rank of the gold file (Table 1). Term-conditioned habits looked plausible and were rejected by the ladder (5 tasks better, 16 worse, on the released graphs). With a temporal split, R@5 is 80%.

**Table 1.** Localization without an LLM, nested cross-validation (128 tasks). R@k: share of tasks with a gold file in the top k; MRR: mean reciprocal rank.

| Configuration | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|
| Lexical only | 42% | 64% | 71% | 0.52 |
| + file habits | 51% | 85% | 92% | 0.65 |

### 4.2 Content experience does not earn authority

**31B.** With the old scorer, the agent with and without the skill resolved the same number of tasks in two runs (2 and 2, 3 and 3). In one run it touched the right file in 8 of 10 verifiable tasks and resolved 2: finding the file was not the bottleneck. Repository habits also cannot fire on the private hidden repositories.

**E4B.** The small model failed differently. In a local diagnostic on the pool tasks it edited by call 12 in 22 of 24 tasks (31B on the scorer's hardware: 11), but 10 of its 23 patches missed the region the reference patch changes (31B: 1 of 36). A paired local run with a rule written in advance tested whether calling `locate` first fixes this. It could not be read (one infrastructure error, and the skill returned candidates in 84% of calls against a 90% threshold); descriptively the arm with memory made more empty patches (6 against 1) and resolved 0 against 2. E4B alone puts its first edit in a gold file in 18 of 24 tasks; `locate` ranks a gold file first in 12 of 24 lexically and 14 of 24 with out-of-sample habits. The prompt told the model to trust the memory's first file; with it, E4B's first edit landed in a gold file in 15 of 24.

Where E4B picks the wrong file, the memory has the right one first in 3 of 6, so we tested the memory as a second opinion: keep the model's file if it is in the memory's top three, otherwise switch. With a rule written before the simulation (net gain of at least 4 of 48 first edits, at most 2 losses), the policy gained 5 and lost 9, assuming perfect compliance. Here the comparison with the model came first, and it stopped the build.

### 4.3 Procedural experience: an offered hint was not used

A verification habit admitted offline (a `check` command that flagged 8 of 9 broken patches in recorded traces, 0 of 25 false alarms) was offered in the prompt; the 31B called it in 0 of 14 tasks with a patch. Adopting the Budget-Fit package gave our best score at the time, 10 of 58 against a mean of 5.2 over nine earlier submissions, in a single submission.

### 4.4 Paired pools: which rules are followed

Each pool in Table 2 changes one thing in the current base.

**Table 2.** Paired pools, 24 tasks each, scorer hardware, 0 infrastructure errors.

| Pool | Change | Compliance | R base / variant | Aux | Min | Adopted | Public |
|---|---|---|---|---|---|---|---|
| 1 | Public 0.24 prompt, longer, with two of our rules | aux 8 → 0; tool refused 1 write (6 with 36 calls) | 4 / 6 | 8 / 0 | 5.11 / 5.70 | No (time +11.5%) | |
| 2 | `bf` + only those two rules, reworded to fit the tools | aux 10 → 0, writes outside repo 0 | 6 / 5 | 10 / 0 | 4.80 / 4.92 | Yes | |
| 3 | + wrap long commands in `timeout 90` | 94% of commands wrapped; hangs 2 → 0 | 4 / 6 | 1 / 0 | 5.24 / 5.12 | Yes | 0.18 |
| 4 | + thinking cap 2,048 tokens | longest turn 4,268 → 2,555 | 6 / 8 | 1 / 1 | 5.15 / 4.60 | Yes | 0.13 |
| 5 | + tool-call cap 22 (harness warning at call 12) | warning at call 12 in 22 of 22 | 5 / 5 | 0 / 0 | 4.46 / 4.54 | No (session timeouts 1 → 3) | 0.15 |

In these pools, rules were followed when the action was visible in the call the model was writing: a command wrapped in `timeout`, a file not created. Rules that asked the model to track state it cannot see were not. "Edit by call 12" depends on the call number, which the model does not see before the harness warns it: in 211 traces of the base the model never mentions call 12, and a prompt-requested counter appeared in 0 of 482 turns of a TPU pool. A locator sub-agent told to use at most 6 calls used all 20 of its calls in both probed tasks; that submission scored 0.10. Structure the harness enforces was followed: under the thinking cap the longest turn fell from 4,268 to 2,555 tokens (a later 3,470-token turn was mostly code in an edit), and the tool-call cap moved the warning to call 12 in all 22 traces that reached it. Structure governs only what the harness counts: malformed calls rejected before execution are not charged, and one task made 49 of them before its session timed out.

Compliance depends on the stack. The pool 2 prompt, run on TPU in bf16 with another vLLM build, raised aux from 10 to 11 where the scorer's stack lowered it to 0; weights and server changed together, so compliance has to be measured on the deployed stack.

Noise bounds Table 2: the pool 4 variant resolved 8 tasks on 31 October and 5 on 2 November with the same zip, and the public scores (0.18, 0.13, 0.15) are within the noise of one zip. Two E4B versions of the base scored 0.05, against 0.18 for the 31B.

### 4.5 Defects that changed decisions

**Table 3.** Defects we found, and the decision each one would have changed.

| Defect | Where | Effect on a decision |
|---|---|---|
| Released code graphs omit async functions and modules without definitions: 56 gold functions invisible in 20 tasks | Data | Function habits were admitted on the graphs and rejected on exact tables recovered from git |
| The skill found no files in the harness's subprocess mode | Harness | "The skill does not help" would have been read from 0 candidates in 12 calls |
| Three requests tasks pass with an empty patch | Measurement | The verifiable set shrank and two runs lost one resolved task per arm |
| Our failed-edit metric counted calls after the budget and writes refused by the tool | Our metric | 7 of `bf`'s 9 failures were calls after the budget; the comparison between prompts changed once split by cause |
| Identical zips got no score or 0.01 between 21 and 25 October | Scorer | Two replicas would have entered the noise estimate |
| The local testbed ran E4B at Ollama's default 4,096-token context, silently dropping the task in 34 of 49 traces | Our testbed | A month of "the small model forgets the task" was this artifact |
| The second-opinion simulation first compared file paths with module names | Our analysis | It showed 28 gains; the baseline of 0 correct files exposed it before reading |

## 5. Related work

SWE-bench [Jimenez et al., 2024] defines the task; SWE-agent [Yang et al., 2024] and Agentless [Xia et al., 2025] shape the agent and separate localization from repair. Procedural memory for agents includes Voyager [Wang et al., 2023], ExpeL [Zhao et al., 2024], Agent Workflow Memory [Wang et al., 2024] and ProcMEM [Mi et al., 2026], which admits skills through a verification gate. Jev-Mem [Jiang et al., 2026] puts a non-LLM controller in charge of memory; Hindsight [Latimer et al., 2025] lets the model revise its opinions, which we keep outside the model. Reasoning models overthink [Chen et al., 2024]; on SWE-bench Verified, more reasoning with less interaction goes with lower success [Cuadron et al., 2025]. A time budget stated only in the prompt does not control time use without feedback from the harness [Wang et al., 2026], which matches our split between structure and prompt rules. The rules in our pools are not new: the public 0.24 prompt already contains some of them [honghanhhh, 2026]. Our contribution is the method that admits rules from traces, measures whether they are followed, and decides when memory may override the model.

## 6. Limitations

- Pools of 24 tasks cannot separate differences of one to three resolved tasks; we use them for "not worse" decisions and compliance.
- Compliance is measured on four public repositories; hidden tasks come from private ones.
- The E4B memory results are local (Ollama GGUF, uncapped thinking, slower hardware) and post hoc on the 24 tasks that generated the idea.

## 7. Conclusion

In our runs, the experience that changed a quantized agent's behavior was procedure, and its value depended on whether the model followed it. Structure enforced by the harness was followed; prompt rules were followed when the action was visible in the call and not when they required state the model could not see; both have to be measured on the stack that serves the model. Content memory raised offline accuracy and made a small model worse when it led, so authority over the model should be its own decision, taken against the model's measured accuracy before any run. Written rules, paired runs and recorded corrections made these conclusions reachable under noise as large as every effect we measured.

## References

- Chen, X., et al. (2024). Do NOT think that much for 2+3=? On the overthinking of o1-like LLMs. arXiv:2412.21187.
- Cuadron, A., et al. (2025). The danger of overthinking: Examining the reasoning-action dilemma in agentic tasks. arXiv:2502.08235.
- Gorigami (2026). LINALDB, version 0.1.22. Apache-2.0. https://github.com/gorigami/linaldb
- Henao, W. E. (2026). Gemma 4 Agent: Budget-Fit Single Agent. Kaggle notebook. https://www.kaggle.com/code/verracodeguacas/gemma-4-agent-budget-fit-single-agent
- honghanhhh (2026). Gemma 4 baseline, LB 0.24. Kaggle notebook. https://www.kaggle.com/code/honghanhhh/gemma-4-baseline-lb-0-24
- Jiang, D., et al. (2026). Jev-Mem: System-one-controlled agentic memory for efficient AI agents. arXiv:2609.23986.
- Jimenez, C. E., et al. (2024). SWE-bench: Can language models resolve real-world GitHub issues? ICLR 2024.
- Latimer, C., et al. (2025). Hindsight is 20/20: Building agent memory that retains, recalls, and reflects. arXiv:2512.12818.
- Mi, Q., et al. (2026). ProcMEM: Learning reusable procedural memory from experience via non-parametric PPO for LLM agents. ICML 2026.
- Wang, A., et al. (2026). On the Clock: Towards punctual and productive time-budgeted AI agents. NeurIPS 2026.
- Wang, G., et al. (2023). Voyager: An open-ended embodied agent with large language models. arXiv:2305.16291.
- Wang, Z. Z., et al. (2024). Agent workflow memory. arXiv:2409.07429.
- Xia, C. S., et al. (2025). Agentless: Demystifying LLM-based software engineering agents. FSE 2025.
- Yang, J., et al. (2024). SWE-agent: Agent-computer interfaces enable automated software engineering. arXiv:2405.15793.
- Zhang, D., et al. (2026). Useful memories become faulty when continuously updated by LLMs. arXiv:2605.12978.
- Zhao, A., et al. (2024). ExpeL: LLM agents are experiential learners. AAAI 2024.
