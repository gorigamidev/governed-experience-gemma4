# Memory as Evidence, Not Authority: Governing the Experience of an Offline Gemma 4 Coding Agent

**Subtitle:** Rules written before the data, compliance as the measure of a rule, and paired tests of when memory may override the model.

**Code:** https://github.com/gorigamidev/governed-experience-gemma4

## Abstract

What an offline coding agent learns from past runs should enter as evidence for the model, not as authority over it, unless measurement says otherwise. We govern the experience of a Gemma 4 agent (31B, 4-bit weights, 32K context) with rules written before the data, lineage and open corrections, and ask: is the experience true, does the model follow it, and should it override the model. Offline, content memory raised the gold file into the top five from 64% to 85% of 128 tasks. Yet the 31B's first edit landed in a gold file in 79% of 322 recorded runs, against 59% for the memory's first file, even on title-only statements. Out of sample, the memory moved a small model's (E4B) first edit to a gold file in 10 of 15 tasks against 3, though its first file explained only 2 of the 7 gains. In five paired pools, prompt rules were followed when the action was visible in the call and not when they needed state the model could not see, and serving stacks and switches changed behavior unseen. Same-zip noise of up to three tasks bounds every comparison.

## 1. Introduction

Under the competition's constraints every model call is expensive and the model is hard to change: no adapter reported in the competition, ours included, beat the base model. We study a second lever: the experience the agent works with, such as where fixes usually land or which habits waste the budget. Memory updated without quality control degrades [Zhang et al., 2026], a sound rule can be ignored by the model, and a memory right on average can be worse than the model it advises. External memory helps where the model's own knowledge is weak and hurts where it is strong [Mallen et al., 2023]; we test this for an agent's file choice.

Contributions:

1. **The governance protocol we used**: decision rules written before the evidence, admission through paired tests, lineage, and corrections recorded as corrections.
2. **Compliance as the measure of procedural experience**: five paired pools showing which kinds of rules a quantized model follows, and that compliance depends on the serving stack.
3. **Authority as a separate, measured decision**: on 322 runs the 31B's own file choice beat the memory's, even on title-only statements; out of sample the memory helped a weaker model as evidence it could use, not as a ranking it obeyed.
4. **An audit of defects** in data, harness and our local testbed, each tied to the decision it would have changed.

## 2. Setting

**Harness.** A submission is a declarative ADK agent configuration (prompt, sampling, budgets, skills) served by vLLM on `gemma-4-31b-it-qat-w4a16-ct` (or Gemma 4 E4B). The model edits a repository through nine tools; the working tree becomes the patch, which passes only if all hidden tests pass. There are no callbacks: nothing runs between turns unless the model calls a tool. The scorer changed on 30 September; agent results below are dated.

**Budget.** The strongest public configuration, Budget-Fit [Henao, 2026], uses one agent, a 4,096-token thinking budget, 28 tool calls, 8 minutes per task and a prompt rule to edit by the twelfth call. It is our base (`bf`).

**Noise.** The public score is the floor of the percentage of 58 tasks. Identical zips scored 11, 10 and 9, and 8, 11 and 9 of 58 on our account; concurrent work measures a re-run deviation of about 1.5 tasks [Mobassir, 2026]. We decide only from paired runs with "not worse" thresholds, and read public scores only against an alarm written in advance.

## 3. Method

### 3.1 Governance protocol

Every decision is written before its data: arms, tasks, metrics, thresholds and the next step per outcome. A rule is never changed after seeing the result it decides. Admitted experience keeps its lineage in LINALDB, a store with content-hashed lineage [Gorigami, 2026]. A wrong number is corrected by stating what it said, what it says and why.

### 3.2 Two kinds of experience

**Content.** Habits mined per repository from fixes (a file that recurs across tasks, or one that co-occurs with a statement term) adjust a lexical ranking computed by a deterministic standard-library skill (`locate`) inside the sandbox. A habit type is admitted through nested cross-validation and a complexity ladder: it must beat the last accepted configuration on a paired per-task objective (sign test, p ≤ 0.10) with the 95% bootstrap interval of the MRR change above −0.01.

**Procedure.** Candidate rules come from failed runs coded with a fixed taxonomy (no edit before the budget ends, stray scripts, hung commands, retried failed edits). A code becomes a candidate only if its mechanism can be checked on recorded traces; the vehicle (prompt rule, budget field, tool) follows the mechanism.

### 3.3 Measurement

**Paired pools.** Two arms run in one session on four L4 GPUs, interleaved per task, on the same 24 tasks. A variant counts as not worse than the base if resolved tasks (R) drop by at most one, stray files left in the patch (aux) rise by at most one, mean minutes stay within 10%, and the failure the variant targets does not rise. Each pool also defines a compliance metric for the rule it tests, computed from the traces. Patches are re-run in the official sandbox; a task is verifiable if its reference patch passes and an empty one fails.

**Authority.** For a run, the model is right if its first successful edit lands in a file the reference patch changes (a gold file); the memory is right if `locate` ranks a gold file first. Per-task pairs use an exact McNemar test.

## 4. Experiments

### 4.1 Content experience helps localization offline

On exact symbol tables at each task's base commit, file habits raise the rank of the gold file (Table 1). Term-conditioned habits were rejected by the ladder (5 tasks better, 16 worse).

**Table 1.** Localization without an LLM, nested cross-validation (128 tasks). R@k: share of tasks with a gold file in the top k; MRR: mean reciprocal rank.

| Configuration | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|
| Lexical only | 42% | 64% | 71% | 0.52 |
| + file habits | 51% | 85% | 92% | 0.65 |

### 4.2 Content experience does not earn authority

**31B.** In 322 recorded runs on 43 tasks, the 31B's first edit landed in a gold file in 255 (79%); the memory's first file, with habits learned out of sample, was gold in 191 (59%). Per task, the model alone was right in 11 and the memory alone in 5 (p = 0.21; lexical memory alone, p = 0.008). Following Mallen et al., we wrote a rule that memory should win on title-only statements (243 characters or fewer), where the model has least to go on. It did not: 42 against 63 of 90 runs. This test is post hoc on traces we had seen in aggregate.

**E4B.** On the 24 pool tasks, E4B's first edit was in a gold file in 18; told that `locate`'s first file is where the fix usually goes, it landed 15, with more empty patches (6 against 1), in a paired local run that failed its validity rule (per task 7 against 3, p = 0.34). Used as a second opinion, the memory lost more first edits than it gained (9 against 5 of 48) in a simulation with a rule written first, which stopped the build.

**Out of sample.** We ran the same two zips on the 15 title-only tasks outside those 24, one local session per arm, with rules written first. Without memory, E4B's first edit was in a gold file in 3 of 15; with it, 10 of 15, and empty patches fell from 4 to 2 (per task 7 against 0, p = 0.016). Validity held: no infrastructure errors, and `locate` came before the first edit in 13 of 15. But `locate` ranked a gold file first in only 2 of the 7 gains, and in 3 the gold file was not in its list. The help came from the arm (a search step and its candidate list), not from obeying the ranking; this design cannot separate the two.

Read together, after the fact, help tracked the model's own accuracy, not statement length: the memory helped where E4B started at 3 of 15, not where either model already picked well.

### 4.3 Procedural experience: an offered hint was not used

A verification habit admitted offline (a `check` command flagging 8 of 9 broken patches, 0 of 25 false alarms) was offered in the prompt; the 31B called it in 0 of 14 tasks with a patch. Our public mean rose from about 4.9 to 9.2 of 58 with Budget-Fit; that gain is Budget-Fit's, not ours.

### 4.4 Paired pools: which rules are followed

**Table 2.** Paired pools (24 tasks, scorer hardware, 0 infrastructure errors); each changes one thing in the current base. Public: tasks of 58, one entry per submission of the same zip.

| Pool | Change | Compliance | R base / variant | Aux | Min | Not worse (rule) | Public |
|---|---|---|---|---|---|---|---|
| 1 | Public 0.24 prompt, longer, with two of our rules | aux 8 → 0 | 4 / 6 | 8 / 0 | 5.11 / 5.70 | No (time +11.5%) | |
| 2 | `bf` + only those two rules, reworded to fit the tools | aux 10 → 0, writes outside repo 0 | 6 / 5 | 10 / 0 | 4.80 / 4.92 | Yes | |
| 3 | + wrap long commands in `timeout 90` | 94% of commands wrapped; hangs 2 → 0 | 4 / 6 | 1 / 0 | 5.24 / 5.12 | Yes | 11, 10, 9 |
| 4 | + thinking cap 2,048 tokens | longest turn 4,268 → 2,555 | 6 / 8 | 1 / 1 | 5.15 / 4.60 | Yes | 8, 11, 9 |
| 5 | + tool-call cap 22 (harness warning at call 12) | warning at call 12 in 22 of 22 | 5 / 5 | 0 / 0 | 4.46 / 4.54 | No (session timeouts 1 → 3) | 9 |

Rules were followed when the action was visible in the call the model was writing: a command wrapped in `timeout`, a file not created. Rules that asked the model to track state it cannot see were not. "Edit by call 12" depends on the call number, which the model does not see before the harness warns it: in 211 base traces the model never mentions call 12. Caps and warnings in pools 4 and 5 are applied by the harness, so their compliance holds by construction.

Compliance depends on the stack. The pool 2 prompt, run on TPU in bf16 with another vLLM build, raised aux from 10 to 11 where the scorer's stack lowered it to 0 (weights and server changed together).

**A switch is a rule too.** With thinking turned off (`enable_thinking: False`, a 16-task pool), the 31B kept thinking in a channel the trace does not record: turns of up to 4,500 tokens with an empty message and a short tool call carried 62% and 69% of the generated tokens in two arms, against 3% in the base. Sessions cut by the timeout rose from 1 to 4 and 5, and the arms resolved 4 and 3 of 11 against 5.

Noise bounds Table 2. On the 11 verifiable tasks, 11 sessions of seven `bf` arms resolved 4.18 tasks on average (standard deviation 0.98, against 0.81 if all arms shared per-task rates). One session per arm detects only effects of about 3 of 11, and no variant left that band.

**The edit follows the warning, not the finding.** All 19 runs that ended without a patch named the gold file within four calls; 14 tried their first edit after the cap and were refused. In 89 of 133 first edits after call 10, the edited text had appeared verbatim in the first ten calls. With the cap at 32 the warning came at call 22, and both late tasks still edited after the cap.

**A structural lock failed on call format.** To force the edit, we replaced `read_file` and `run_command` with a skill that stops serving reads after ten calls without an edit. In four probes of 7 tasks, the 31B corrupted 25% to 95% of its skill calls (a backtick closing `file_path` swallows `skill_name`), against none in about 6,700 native tool calls. ADK rejects these calls without charge, so the model repeated them until the session ended. A rule written before the last probe archived the lever.

**Repetition sustains itself.** In 395 runs of the base family, after two identical consecutive calls the next was identical in 76% of 250 cases, against 1% of 8,470 after two different calls; leaving out the calls ADK rejects without charge, 55% of 97 against 1%. Seeding a good habit this way was untestable: testing before the first edit appeared in 6 of 326 runs.

### 4.5 Defects that changed decisions

**Table 3.** Defects we found, and the decision each one would have changed.

| Defect | Where | Effect on a decision |
|---|---|---|
| Released code graphs omit async functions and modules without definitions: 56 gold functions invisible in 20 tasks | Data | Function habits were admitted on the graphs and rejected on exact tables recovered from git |
| The local testbed ran E4B at Ollama's default 4,096-token context, silently dropping the task in 34 of 49 traces | Our testbed | A month of "the small model forgets the task" was this artifact |

## 5. Related work

SWE-bench [Jimenez et al., 2024] defines the task; SWE-agent [Yang et al., 2024] and Agentless [Xia et al., 2025] shape the agent and separate localization from repair. Procedural memory for agents includes Voyager [Wang et al., 2023], ExpeL [Zhao et al., 2024], Agent Workflow Memory [Wang et al., 2024] and ProcMEM [Mi et al., 2026], which gates skills by verification. Adaptive retrieval uses external memory only where the model's own knowledge is weak [Mallen et al., 2023]. On SWE-bench Verified, more reasoning with less interaction goes with lower success [Cuadron et al., 2025]. A time budget stated only in the prompt does not control time use without feedback from the harness [Wang et al., 2026], which matches our split between structure and prompt rules. Concurrent work measures the same leaderboard noise [Mobassir, 2026], and the public 0.24 prompt already contains some of our rules [honghanhhh, 2026]. Our contribution is the method that admits experience, measures whether it is followed, and decides by measurement when memory may override the model.

## 6. Limitations

- Pools of 24 tasks cannot separate differences of one to three resolved tasks. What survives is the "not worse" decisions and the compliance measures, not rankings of variants.
- Compliance is measured on four public repositories, and the split between visible and hidden state rests on a few rules.
- The 31B authority comparison is post hoc on recorded runs. It survives as a measured gap (79% against 59%), not as a tested prediction.
- The E4B runs are local (Ollama GGUF), one session per arm on 15 tasks out of sample, and the arm changes the search step and the memory together. What survives is that the memory helped without being obeyed, not how much.

## 7. Conclusion

The value of experience given to a quantized agent is the product of two measured terms: how often the model follows it, and how much it helps when followed. Prompt rules were followed when the action was visible in the call and not when they required state the model could not see, and compliance, like any configuration switch, has to be measured on the serving stack. Content memory raised offline accuracy, but the 31B's own file choice beat it, even where the statement said least, while a weaker model gained from the memory without obeying its ranking. Memory should enter as evidence, and override the model only where a measurement against that model says so. Written rules, paired runs and recorded corrections made this reachable under noise as large as most effects.

## References

- Cuadron, A., et al. (2025). The danger of overthinking: Examining the reasoning-action dilemma in agentic tasks. arXiv:2502.08235.
- Gorigami (2026). LINALDB, version 0.1.22. Apache-2.0. https://github.com/gorigami/linaldb
- Henao, W. E. (2026). Gemma 4 Agent: Budget-Fit Single Agent. Kaggle notebook. https://www.kaggle.com/code/verracodeguacas/gemma-4-agent-budget-fit-single-agent
- honghanhhh (2026). Gemma 4 baseline, LB 0.24. Kaggle notebook. https://www.kaggle.com/code/honghanhhh/gemma-4-baseline-lb-0-24
- Jimenez, C. E., et al. (2024). SWE-bench: Can language models resolve real-world GitHub issues? ICLR 2024.
- Mallen, A., et al. (2023). When not to trust language models: Investigating effectiveness of parametric and non-parametric memories. ACL 2023.
- Mi, Q., et al. (2026). ProcMEM: Learning reusable procedural memory from experience via non-parametric PPO for LLM agents. ICML 2026.
- Mobassir [S. M. Hossain] (2026). Why your leaderboard can't see a 3-point gain. Kaggle notebook, companion to a Paper Track writeup. https://www.kaggle.com/code/mobassir/why-your-leaderboard-can-t-see-a-3-point-gain
- Wang, A., et al. (2026). On the Clock: Towards punctual and productive time-budgeted AI agents. NeurIPS 2026.
- Wang, G., et al. (2023). Voyager: An open-ended embodied agent with large language models. arXiv:2305.16291.
- Wang, Z. Z., et al. (2024). Agent workflow memory. arXiv:2409.07429.
- Xia, C. S., et al. (2025). Agentless: Demystifying LLM-based software engineering agents. FSE 2025.
- Yang, J., et al. (2024). SWE-agent: Agent-computer interfaces enable automated software engineering. arXiv:2405.15793.
- Zhang, D., et al. (2026). Useful memories become faulty when continuously updated by LLMs. arXiv:2605.12978.
- Zhao, A., et al. (2024). ExpeL: LLM agents are experiential learners. AAAI 2024.
