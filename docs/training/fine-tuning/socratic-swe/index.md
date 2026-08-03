---
title: "Socratic-SWE: Self-Evolving Coding Agents via Trace-Derived Skills"
summary: "Closed-loop self-evolution framework that distills SWE agent solving traces into structured skills, uses them to guide targeted task generation, and aligns the curriculum via solver-gradient reward — reaching 50.40% on SWE-bench Verified."
layout: default
confidence: high
sources:
  - raw/training/socratic-swe-self-evolving-coding-agents--arxiv-2606.07412v1.pdf
  - derived/pdf-markdown/training/socratic-swe-self-evolving-coding-agents.md
updated: 2026-07-23
---

# Socratic-SWE: Self-Evolving Coding Agents via Trace-Derived Skills

**Paper:** Socratic-SWE: Self-Evolving Coding Agents via Trace-Derived Agent Skills  
**Authors:** Chuan Xiao, Zhengbo Jiao, Shaobo Wang, Wei Wang, Bing Zhao, HU WEI, Linfeng Zhang, Lin Qu  
**arXiv:** 2606.07412v1 — June 8, 2026

**Related pages:** [Intrinsic Dimensionality Fine-Tuning](../intrinsic-dimensionality-fine-tuning/index.md)

## TL;DR

**What:** A closed-loop self-play framework where a single SWE agent alternates between generating repair tasks and solving them, with the curriculum adapting to the agent's own weaknesses across iterations.

**How:** Historical solving traces are distilled into an Agent Skill Registry — structured documents that capture recurring failures and repair patterns. These skills guide a Generator to create targeted tasks; tasks are filtered through a four-stage verifier gate and scored by gradient alignment with a held-out validation set.

**The number:** 50.40% on SWE-bench Verified after 3 iterations (12k validated instances each), outperforming five self-evolving baselines (SPIRAL, R-Zero, Absolute-Zero, Socratic-Zero, SSR) under identical compute budgets.

## The Big Picture

![Socratic-SWE Framework](socratic-swe-big-picture.drawio.svg)

*① The Solver consumes tasks from the curriculum pool, interacts with real repository sandboxes, and produces patches evaluated by executable verifiers. ② Successful and failed traces are distilled into the Agent Skill Registry — structured skills capturing recurring failures, repair patterns, and verifier designs. ③ The Generator samples skills from the registry and constructs targeted repair tasks conditioned on the Solver's current evidence (weaknesses). ④ A four-stage Verifier Gate (format → grounding → execution → semantics) filters candidate tasks so only executable, reproducible, and solvable ones enter the curriculum. ⑤ Accepted tasks receive a gradient-aligned Generator reward: cosine similarity between the candidate task's policy gradient and the aggregated validation gradient, favoring tasks whose learning direction aligns with improving on trusted held-out tasks.*

## Why This Exists

Imagine you are training a coding agent that can fix bugs in real repositories. You have a handful of high-quality SWE tasks (like SWE-bench issues), but not nearly enough to drive reinforcement learning at scale. Existing approaches try to fix this by synthetically generating more tasks — a mutation engine randomly injects bugs, or an LLM rewrites existing issues. But these synthetic pipelines are **open-loop**: they produce tasks independently of what the agent actually struggles with. You end up with a curriculum full of tasks the agent either trivially solves or fails at for irrelevant reasons (obscure domain knowledge, broken environments), neither of which improves real-world capability.

**Concrete failure example:** Suppose your Solver agent consistently fails on multi-file refactors that require coordinating edits across three modules. A static mutation pipeline keeps generating single-function bug injections because that's what its AST rules produce. The agent trains on these, gets good at single-function fixes, but still fails the multi-file case. After three rounds of training, the Solver has wasted compute on tasks that don't address its actual capability gap.

Socratic-SWE closes this loop: the agent's own failure traces become the signal for what to generate next. If the Solver struggles with multi-file refactors, the Skill Registry captures that pattern, and the Generator produces tasks targeting exactly that weakness.

## The Landscape

```mermaid
flowchart TD
    subgraph Static["Static Data Pipelines"]
        A1["SWE-bench<br>Human-curated issues"]
        A2["SWE-smith<br>AST-level mutation"]
        A3["BugLab<br>Learned bug injection"]
        A4["SWE-Gym<br>Software evolution data"]
    end

    subgraph SelfPlay["Self-Play / Co-Evolution"]
        B1["SPIRAL<br>Zero-sum game: injector vs repairer"]
        B2["R-Zero<br>Challenger-Solver, majority-vote reward"]
        B3["Absolute-Zero<br>Single-model propose+solve"]
        B4["Socratic-Zero<br>Teacher-Solver-Generator tri-role"]
        B5["SSR<br>Bug injection via code removal/revert"]
    end

    subgraph SkillBased["Skill-Guided Evolution"]
        C1["SkillRL / SKILL0<br>Distill interaction into reusable skills"]
        C2["AgenticProposing<br>Modular reasoning skills for harder examples"]
        C3["Socratic-SWE ★<br>Trace→Skill→Task→Trace closed loop<br>+ gradient-aligned Generator reward"]
    end

    A1 --> A4
    B1 --> B5
    B2 --> B3 --> B4 --> B5
    A4 --> C1
    B5 --> C3
    C1 --> C2 --> C3

    classDef highlight fill:#d5f5e3,stroke:#27ae60,stroke-width:3px
    class C3 highlight
```

The evolutionary path shows three generations:

1. **Static pipelines** produce tasks once, independently of training. They scale well but produce irrelevant or trivial tasks.
2. **Self-play methods** introduce co-evolution between task proposers and solvers, but reward design is coarse (binary pass/fail, majority vote) and doesn't capture whether a task is *useful* for learning.
3. **Skill-guided evolution** (Socratic-SWE's contribution) adds structured skill extraction from traces and gradient-aligned task quality scoring, making the curriculum both adaptive and quality-aware.

## The Core Idea

Socratic-SWE treats the agent's own solving traces — both successes and failures — as the primary source of training signal. Rather than just using traces for binary reward (pass/fail), the framework distills them into a structured **Agent Skill Registry**: programmable documents that capture *what went wrong*, *what repair strategy worked*, and *how to design a verifier*. These skills then guide a Generator to produce targeted tasks in real repositories. The quality of each generated task is measured not just by executability, but by **gradient alignment**: would training on this task move the Solver's parameters in a direction that also improves performance on trusted held-out validation tasks? This closes the loop from trace → skill → task → trace.

## Deep Dive

### Agent Skill Registry

**What it does:** Distills raw interaction traces from the Solver into a structured, retrievable set of reusable skills that the Generator can condition on.

**Why it matters:** Without structured skills, the Generator must generate tasks from scratch using only free-form text prompts, making it hard to target specific Solver weaknesses consistently.

**How it works:** A three-stage pipeline:

1. **Trace Collection:** Deploy the current Solver checkpoint on seed tasks. Collect both successful traces $T^+$ (showing effective strategies) and failed traces $T^-$ (exposing capability gaps).
2. **Skill Extraction:** A distillation model $M_{\text{distill}}$ (Qwen3.6-27B) processes the trace corpus and extracts recurring behavioral patterns. Success traces yield generalizable repair strategies; failure traces yield corrective principles.
3. **Registry Construction:** Candidate skills are deduplicated by semantic similarity and filtered by trace coverage: $S = \text{Dedup}(\hat{S}, \delta_{\text{sim}}) = \{s_1, \ldots, s_M\}$.

Each skill has four fields: **name**, **natural-language description**, **applicability conditions**, and an **ordered list of operations** — enabling programmatic retrieval rather than free-form prompting.

**The intuition:** Instead of saying "generate a hard task," skills tell the Generator: "the Solver fails on multi-file refactors where function signatures change — here's a template for constructing such tasks."

**A concrete example:** From the failure scenario above, the Skill Registry might capture a skill called `multi-file-signature-refactor`: conditions = "task involves >2 files with interdependent API changes," operations = [identify call sites across modules, update imports, verify all callers compile]. The Generator uses this to construct a task specifically exercising multi-file coordination.

**Remember:** Skills are distilled from the agent's own behavior, not hand-designed — they evolve with the agent's capability.

---

### Skill-Guided Task Generator

**What it does:** Constructs executable SWE repair tasks in real repositories, conditioned on skills from the registry and current Solver evidence.

**Why it matters:** This is where the "skill" becomes a "task" — turning abstract behavioral patterns into concrete, executable training instances.

**How it works:** Given a repository $r$ and a skill $s$, the Generator samples:
$$(\tau, v) \sim \pi_\theta(\cdot \mid r, s, E_t, \text{role}=G)$$

where $\tau$ is the repair objective, $v$ is the executable verifier, and $E_t$ is Solver-side evidence collected from the current curriculum $D_t$.

**The intuition:** The Generator doesn't just inject random bugs — it constructs tasks that specifically exercise the behavioral pattern described by the skill, in a real repository where the verifier is guaranteed to be executable.

**Remember:** Generator output is a $(task, verifier)$ pair — both are needed for training.

---

### Four-Stage Verifier Gate

**What it does:** Filters generated tasks before they enter the curriculum, ensuring only executable, reproducible, and solvable tasks survive.

**Why it matters:** Without filtering, the curriculum fills with broken environments, unsolvable tasks, or tasks where the verifier doesn't meaningfully separate correct from incorrect patches.

**How it works:** Four sequential checks in the repository sandbox $r$:

| Stage | Name | What it checks | Failure means |
|-------|------|---------------|---------------|
| 1 | **Format** ($f_1$) | $\tau$ and $v$ are well-formed, parseable, syntactically valid | Task can't even be loaded |
| 2 | **Grounding** ($f_2$) | $\tau$ references artifacts that actually exist in $r$ | Task points to non-existent files/functions |
| 3 | **Execution** ($f_3$) | $v$ runs without infrastructure errors, stable across repeated runs | Verifier is flaky or broken |
| 4 | **Semantics** ($f_4$) | $v$ separates failing from repaired states; at least one valid repair exists | Task is either always-passing or impossible |

Each stage is evaluated only if all preceding stages pass (short-circuit evaluation). The overall validation function is:

$$V_{\text{alid}}(\tau, v, r) = \prod_{l=1}^{4} f_l(\tau, v, r) \in \{0, 1\}$$

**The intuition:** Think of this as a CI pipeline for training data — format/lint, then dependency check, then integration test, then semantic correctness. Only green builds enter the curriculum.

**A concrete example:** A Generator produces a task claiming to fix a `calculate_tax` function in `billing.py`. The gate checks: (1) is the task JSON well-formed? (2) does `billing.py` and `calculate_tax` actually exist? (3) does the test suite run without crashing? (4) does the test actually fail before the fix and pass after?

**Remember:** Validation is about executability, not usefulness — that's the reward's job.

---

### Gradient-Aligned Generator Reward

**What it does:** Scores generated tasks by how well training on them would align the Solver's parameter updates with improving on trusted held-out validation tasks.

**Why it matters:** Validation (the gate) ensures a task is *runnable*. This reward ensures it's *useful*. A task that's trivially easy or impossibly hard may pass validation but provides zero learning signal.

**How it works:**

1. Maintain a held-out validation set $V_{\text{val}}$ of trusted SWE tasks.
2. For each validation task, roll out $K$ Solver trajectories and compute the per-task policy gradient $g^v_j$ using GRPO. Average across all validation tasks to get the target gradient direction $G_v = \frac{1}{|V_{\text{val}}|} \sum_j g^v_j$.
3. For each candidate Generator task, similarly compute its Solver policy gradient $g_\tau$ from $K$ rollouts.
4. The Generator reward is:

$$R_G(\tau, v, r) = V_{\text{alid}}(\tau, v, r) \cdot \cos(g_\tau, G_v)$$

The validation factor zeros out invalid tasks, and the cosine term measures directional alignment.

**Why cosine (not inner product):** A first-order Taylor expansion shows the validation improvement from one gradient step on a candidate task is $\Delta J_{\text{val}} \approx \eta \|g_\tau\| \|G_v\| \cos(g_\tau, G_v)$. Cosine normalizes out $\|g_\tau\|$, which can be confounded by task length and patch complexity — a long task with a large gradient magnitude isn't necessarily more useful than a short one.

**The intuition:** Imagine you have a set of trusted benchmark tasks you care about. Before adding a generated task to training, ask: "If I train the Solver on this task, will the parameter update also push performance up on my trusted tasks?" The cosine similarity answers this by measuring whether the gradient directions point the same way.

**A concrete example:** The Generator produces two candidate tasks. Task A (multi-file refactor) has $g_\tau$ strongly aligned with $G_v$ ($\cos \approx 0.85$). Task B (obscure numerical edge case) has $g_\tau$ nearly orthogonal to $G_v$ ($\cos \approx 0.05$). Both pass validation, but only Task A gets a high reward and contributes meaningfully to the curriculum.

**Ablation result (Table 3):**

| Generator Reward | SWE-bench Verified (%) | $\Delta$ |
|---|---|---|
| Hardness ($1 - p$) | 47.40 | ↓3.00 |
| Uncertainty ($1 - 2\|p - 0.5\|$) | 48.20 | ↓2.20 |
| Variance (Gaussian) | 48.80 | ↓1.60 |
| **Gradient-aligned (Ours)** | **50.40** | – |
| Gradient + Difficulty hybrid | 50.60 | ↑0.20 |

**Remember:** Gradient alignment decouples task difficulty from task usefulness — a moderate-difficulty task teaching transferable patterns beats a near-impossible one every time.

---

### Repository Repair Solver

**What it does:** The Solver consumes accepted tasks from the curriculum, interacts with real repository sandboxes, and produces patches evaluated by executable verifiers.

**Why it matters:** The Solver is both the consumer of the curriculum and the producer of traces that feed back into skill extraction — it drives the self-evolution loop.

**How it works:** Given a task $(\tau, v)$ and repository $r$, the Solver samples a trajectory $y \sim \pi_\theta(\cdot \mid \tau, r, v, \text{role}=S)$ that may include inspection, code localization, file edits, and validation attempts.

The Solver reward has three components ($\lambda_1, \lambda_2, \lambda_3$ are weighting coefficients):

$$r_S = \lambda_1 \mathbf{1}[F^\checkmark = F \land P^\checkmark = P] + \lambda_2 \frac{|F^\checkmark|}{|F|} + \lambda_3 \frac{|P^\checkmark|}{|P|}$$

| Term | Meaning | Penalizes |
|------|---------|-----------|
| Full-suite pass | All originally failing tests pass AND all originally passing tests still pass | – |
| Partial repair rate | Fraction of originally failing tests now passing | Incomplete fixes |
| Regression avoidance | Fraction of originally passing tests still passing | Breaking existing functionality |

These three heterogeneous components are normalized via GDPO (Group-wise Dimension-wise Preference Optimization), which normalizes each component within its own group before aggregation — preventing any single component from dominating the gradient.

**The intuition:** A perfect fix gets full credit. A patch that fixes 3/5 failing tests but introduces no regressions gets partial credit. A patch that fixes everything but breaks 2 existing tests gets penalized for regressions.

**Remember:** The Solver never sees the reference solution or verifier internals — it learns purely from executable feedback.

---

### Role-Specific Training

**What it does:** Jointly optimizes both Generator and Solver roles with shared weights, using role-specific advantage estimation.

**Why it matters:** Sharing weights allows skill knowledge from the Generator and repair capability from the Solver to cross-pollinate within a single model.

**How it works:** The joint objective is:

$$J(\theta) = \mathbb{E}_{r,s}\big[\mathbb{E}_{(\tau,v) \sim \pi_G}[R_G(\tau,v,r)]\big] + \mathbb{E}_{\tau,r,v}\big[\mathbb{E}_{y \sim \pi_S}[r_S(y,\tau,v,r)]\big]$$

The Generator uses GRPO with the scalar gradient-aligned reward $R_G$. The Solver uses GDPO — normalizing each of the three reward components (pass, repair, regression) independently within its group before aggregation:

$$\hat{A}^{(m)}_i = \frac{r_S^{(m,i)} - \text{mean}(\{r_S^{(m,j)}\}_{j=1}^K)}{\text{std}(\{r_S^{(m,j)}\}_{j=1}^K) + \delta}, \quad m \in \{1,2,3\}$$

$$\hat{A}^S_i = \text{BatchNorm}\!\left(\sum_{m=1}^{3} \hat{A}^{(m)}_i\right)$$

Both roles use the same clipped surrogate objective with KL regularization toward a reference policy.

**The intuition:** The Generator learns which tasks are worth creating (gradient-aligned), and the Solver learns how to fix them (multi-component reward). Sharing weights means the model develops an internal understanding of both what makes a good training task and how to solve one.

**Remember:** The roles are distinguished by a `role` token in the input, not by separate models — a single Qwen3.5-9B handles both.

---

## What This Buys You

### The headline claim

Trace-derived skill curricula produce stronger and more sustained self-evolution than any prior self-play method. Socratic-SWE reaches **50.40% on SWE-bench Verified** after 3 iterations (+7.80 over the base agent and +3.40 over SSR), while baselines either saturate early or regress.

### How we know: head-to-head at Iteration 3

All methods use identical Solver (Qwen3.5-9B), harness (mini-swe-agent for SWE, little-coder for TB2), and budget (12k instances × 3 iterations = 36k total).

| Method | SWE-bench Verified | SWE-bench Lite | SWE-bench Pro | Terminal-Bench 2.0 | Overall |
|---|---|---|---|---|---|
| Base Agent | 42.60 | 29.67 | 17.24 | 10.11 | 24.91 |
| R-Zero | 41.80 | 29.00 | 16.28 | 8.99 | 24.02 |
| SPIRAL | 44.00 | 31.00 | 17.78 | 10.11 | 25.72 |
| Absolute-Zero | 44.40 | 31.33 | 18.19 | 10.11 | 26.01 |
| Socratic-Zero | 45.80 | 32.67 | 19.43 | 12.36 | 27.57 |
| SSR | 47.00 | 34.00 | 20.66 | 12.36 | 28.51 |
| **Socratic-SWE** | **50.40** | **36.67** | **22.85** | **14.61** | **31.13** |

### The mechanism behind the numbers

Three patterns explain why Socratic-SWE pulls ahead:

1. **R-Zero degrades after Iteration 1** (−0.80 vs. Base). Majority-vote rewards are too noisy for partial SWE repairs — the Challenger learns to propose tasks the Solver sometimes flukes, rather than tasks that teach transferable skills.

2. **SPIRAL and Absolute-Zero peak at Iteration 2 then regress.** Self-play without execution-grounded validation leads to reward hacking: the Generator learns to produce tasks that look hard but whose verifiers are gamed by surface-level patches.

3. **Socratic-Zero saturates despite a 397B Teacher.** The Teacher's domain understanding bounds what tasks can be generated, and distilling the Teacher into a smaller Generator loses information at each iteration. Socratic-SWE avoids this bottleneck by extracting skills directly from the Solver's own traces — no external Teacher needed.

4. **SSR improves steadily but plateaus earlier.** Bug injection via code removal/revert produces valid tasks, but without skill-guided targeting, it exhausts low-hanging bug patterns and can't sustain acceleration.

### ⚠️ How to read these numbers

The base agent here (Qwen3.5-9B + mini-swe-agent) starts at 42.60% on Verified — substantially below the 50%+ reported by stronger base models. The +7.80 gain is measured against this starting point. Gains are likely smaller when starting from a stronger base policy. The Overall column is a simple mean across four benchmarks; it does not weight by benchmark importance.

The scaling analysis (Figure 4) shows that Socratic-SWE still gains +1.20 in Iteration 4 (reaching 51.60%) and plateaus only at Iteration 5 (52.00%), while SSR plateaus at Iteration 4 (48.00%). Socratic-SWE saturates later and higher.

### Ablation: what each component contributes

On SWE-bench Verified at Iteration 3:

| Variant | Verified (%) | $\Delta$ vs. Full |
|---|---|---|
| Socratic-SWE (full) | 50.40 | – |
| w/o Skill Registry | 46.20 | ↓4.20 |
| w/o Trace Distillation (manual skills) | 48.00 | ↓2.40 |
| w/o GDPO (use GRPO) | 48.60 | ↓1.80 |
| Qwen3.5-9B self-extraction | 49.80 | ↓0.60 |
| Claude Opus 4.5 extraction | 51.00 | ↑0.60 |

The Skill Registry is the single largest contributor (−4.20), confirming that curriculum design drives the gains. Skill extraction is robust to the extractor model — even 9B self-extraction reaches 49.80%.

---

## Putting It Together

A walk through one complete iteration of Socratic-SWE:

1. **Start of iteration $t$.** The Solver $\pi_\theta$ has just finished training on curriculum $D_t$. The Generator needs to produce $D_{t+1}$: 12k validated tasks for the next round.

2. **Trace collection.** Deploy the Solver on $D_t$ in repository sandboxes. Collect all trajectories — both passes ($T^+$) and failures ($T^-$). A failed trajectory on `oauthlib` might show: the Solver searched for `scope_to_list`, found the right file, but applied generic `sorted()` that broke OAuthLib's scope ordering contract.

3. **Skill distillation.** $M_{\text{distill}}$ (Qwen3.6-27B) processes the trace corpus. It extracts a skill: `scope-helper-contract-violation` — "Solver repairs scope conversion by adding sorting/filtering that violates OAuthLib's exact-order contract. Condition: task involves `list_to_scope`/`scope_to_list`. Operations: inject a mutation that breaks exact scope ordering while keeping syntax valid."

4. **Task generation.** The Generator samples skill $s \sim S$ and repository $r \sim R$. Conditioned on $s$, it navigates `oauthlib/oauth2/rfc6749/utils.py`, injects a targeted mutation in `scope_to_list` that reorders tokens during conversion, and writes a verifier using existing pytest tests. The Generator proposes 8 candidates per repository (group size $K=8$).

5. **Verifier Gate.** Each $(task, verifier)$ pair runs through the four-stage pipeline in the sandbox: (1) format — is the task JSON parseable? ✓ (2) grounding — does `scope_to_list` exist in the repo? ✓ (3) execution — does `pytest tests/test_scope.py` run without crashing, and are results stable across 3 reruns? ✓ (4) semantics — does the test fail on the buggy code and pass on the reference fix? ✓

6. **Gradient scoring.** For surviving candidates, roll out the Solver $K$ times, compute the task-specific policy gradient $g_\tau$, and compare against the pre-computed validation gradient $G_v$ via cosine similarity. The `scope-helper` candidate gets $\cos = 0.82$ — strongly aligned, because the validation set also contains scope-manipulation tasks.

7. **Curriculum update.** Accepted and high-scoring tasks enter $D_{t+1}$. The Generator receives its reward $R_G = \cos(g_\tau, G_v)$ and updates via GRPO. Meanwhile, accepted tasks are queued for the Solver phase.

8. **Solver training.** The Solver consumes $D_{t+1}$, producing patches evaluated by the three-component reward (full pass, partial repair, regression avoidance). GDPO normalizes these components and updates $\pi_\theta$. New traces are collected — and the next iteration begins.

---

## Where It Breaks

| Failure mode | When it happens | Impact |
|---|---|---|
| **Closed-world saturation** | After ~5 iterations with a fixed repository pool. The Skill Registry covers most gaps, and novel tasks become redundant. | Gains plateau (52.00% ceiling in experiments). Expand the repository pool or enable cross-repo transfer to sustain improvement. |
| **Validation set bias** | The held-out $V_{\text{val}}$ is not representative of the target deployment distribution (e.g., all Python when target includes JS/TS). | $G_v$ points in the wrong direction, and gradient-aligned tasks optimize for the wrong skills. Stratify $V_{\text{val}}$ by language and domain. |
| **Verifier quality collapse** | Generated verifiers are too weak — e.g., a test that passes on both buggy and correct code because it only checks surface behavior. | Invalid tasks pass the semantics gate ($f_4$) and enter the curriculum, providing zero learning signal. Stronger verifier design (e.g., requiring behavior delta) mitigates this. |
| **Skill extraction noise** | $M_{\text{distill}}$ produces overly generic skills ("fix bugs better") that don't constrain the Generator. | Generator produces untargeted tasks, reverting to static-mutation quality. Deduplication and coverage filtering (Eq. 4) help, but extraction quality is the bottleneck. |
| **Mode collapse in self-play** | The Generator and Solver co-adapt to a narrow task distribution — e.g., both optimize for single-function string-manipulation bugs. | Performance on diverse real-world SWE tasks degrades even as training metrics improve. Diversity in the repository pool and periodic resampling of skills help. |
| **Compute overhead of gradient alignment** | Every candidate task requires $K$ Solver rollouts and gradient computation, adding ~8.4% wall-clock time. | Acceptable at current scale (1.3 h/iteration vs. 15 h total). Becomes prohibitive if $K$ or $\|V_{\text{val}}\|$ must scale significantly. |
| **No executable verification available** | The target domain lacks deterministic, sandboxed tests (e.g., UI tasks, open-ended code review). | The entire pipeline — verifier gate, gradient alignment, Solver reward — collapses. Not applicable outside SWE-like settings with reliable test suites. |

---

## One Thing to Remember

Socratic-SWE proves that **an agent's own solving traces are a richer training signal than any static task pipeline.** By distilling traces into structured skills, generating tasks that target the Solver's actual capability gaps, and scoring those tasks by whether their gradients point toward better validation performance, the framework creates a curriculum that adapts as the agent improves. The key insight is not that self-play works — it's that *what you measure determines what you learn*: gradient alignment beats difficulty heuristics, execution-grounded validation beats majority vote, and trace-derived skills beat hand-designed taxonomies. Six months from now, remember: **close the loop from trace to skill to task, and align the task gradient with the validation gradient.**

---

## Go Deeper

- **Read:** [arXiv 2606.07412v1](https://arxiv.org/abs/2606.07412v1)
- **Build on:** [SSR](https://arxiv.org/abs/2505.21521) (strongest baseline, bug injection via code removal/revert), [SWE-smith](https://arxiv.org/abs/2505.21211) (static mutation baseline), [SWE-Gym](https://arxiv.org/abs/2505.19171) (software evolution training data)
- **Understand the context:** [Intrinsic Dimensionality Fine-Tuning](../intrinsic-dimensionality-fine-tuning/index.md), [SWE-bench](https://www.swebench.com/)
- **Reproduce:** Code not available at time of writing.

## Self-Test

- [ ] Can I explain why trace-derived skills beat static mutation pipelines?
- [ ] Can I describe each of the four verifier gate stages and what they prevent?
- [ ] Can I articulate why cosine similarity (not inner product) is the right Generator reward?
- [ ] Can I explain how GDPO normalizes the Solver's three-component reward?
- [ ] Can I name the five self-evolving baselines and Socratic-SWE's advantage over each?
- [ ] Can I explain when Socratic-SWE's closed-world assumption breaks and what happens?
- [ ] Can I trace one full iteration from trace collection through curriculum update?
