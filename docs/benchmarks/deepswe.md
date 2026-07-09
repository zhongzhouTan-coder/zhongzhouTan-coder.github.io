---
title: "DeepSWE: Long-Horizon Software Engineering Benchmark"
summary: "Original long-horizon coding benchmark covering DeepSWE task design, verifier methodology, leaderboard snapshot, qualitative failure analysis, and limitations."
layout: default
doc_layer: layer_0
confidence: high
sources:
  - raw/benchmark/deepswe.md
updated: 2026-07-09
---

# DeepSWE: Long-Horizon Software Engineering Benchmark

**Benchmark site:** DeepSWE: Measuring frontier coding agents on original, long-horizon engineering tasks
**Authors:** Wenqi Huang, Charley Lee, Leonard Tng, Serena Ge
**Publication snapshot:** May 26, 2026
**Code / benchmark:** <https://github.com/datacurve-ai/deep-swe>

**Related pages:** [DeepSWE v1.1: Execution and Scoring Changes](deepswe-v1-1.md), [τ-bench: Tool-Agent-User Interaction Benchmark](tau-bench.md), [τ²-Bench: Mechanism and Design](tau2-bench-mechanism.md)

## Summary

DeepSWE is a software-engineering benchmark built to measure long-horizon coding ability on original tasks rather than mined historical fixes. The source emphasizes four claims: contamination resistance, broader repository diversity, larger and less overspecified tasks, and more reliable hand-written behavioral verifiers. On its publication snapshot, the benchmark separates frontier models more sharply than public SWE-bench-style leaderboards, and the authors argue that this better matches day-to-day developer experience with coding agents.

## Visual Explainer

The diagram below summarizes both why DeepSWE was introduced and how its task-plus-verifier design is intended to produce a cleaner benchmark signal.

![DeepSWE benchmark explainer](assets/deepswe-explainer.svg)

[Editable draw.io source](assets/deepswe-explainer.drawio)

## What DeepSWE Tries to Fix

The benchmark is positioned as a response to three weaknesses in existing public coding benchmarks:

1. **Contamination risk:** tasks derived from public commits or pull requests may have leaked into pretraining data.
2. **Narrow task scope:** prompts can be long and prescriptive while the required code change is comparatively small.
3. **Verifier mismatch:** inherited PR test suites can produce both false positives and false negatives when used as benchmark graders.

DeepSWE's answer is to write every task and reference solution from scratch, keep tasks out of upstream repositories, and grade against purpose-built behavioral verifiers.

## Corpus and Task Shape

- **113 tasks** across **91 repositories**
- **5 languages:** TypeScript, Go, Python, JavaScript, Rust
- Median repository contribution is one task, which limits leaderboard concentration in a few flagship repos.

Compared with SWE-Bench Verified and SWE-Bench Pro, DeepSWE tasks are described as shorter in prompt form but substantially larger in implementation scope:

| Metric | SWE-Bench Verified | SWE-Bench Pro | DeepSWE |
|---|---:|---:|---:|
| Mean prompt length | 1,700 chars | 4,614 chars | 2,158 chars |
| Mean reference solution lines added | 10 | 120 | 668 |
| Mean files edited | 1 | 5 | 7 |

The intended effect is to force the agent to explore the codebase, infer the right implementation surface, and preserve surrounding behavior rather than filling in an over-specified patch.

## Methodology

Each task includes three artifacts:

1. The prompt shown to the agent.
2. An executable verifier.
3. A reference solution used during review, not grading.

Repository selection requires public, actively maintained repos with at least 500 GitHub stars and permissive licenses. Each task is pinned to an immutable commit.

Verifiers are designed around **observable behavior** rather than private implementation details. They also include:

- repeat runs during authoring to catch flakiness,
- regression checks against existing repository behavior,
- human review plus LLM-assisted review of prompt-verifier alignment.

```mermaid
flowchart TD
    A[Select active public repository] --> B[Author prompt, verifier, reference solution]
    B --> C[Run verifier repeatedly to catch flakiness]
    C --> D[Review prompt-verifier bijection, breadth, realism, environment cleanliness]
    D --> E[Benchmark task admitted]
```

## Verifier Reliability Claim

The source reports a cross-benchmark audit comparing verifier verdicts against an LLM-based analyzer reading the trajectory, patch, prompt, and reference solution.

| Metric | SWE-Bench Pro | DeepSWE |
|---|---:|---:|
| False positive rate | 8.5% | 0.3% |
| False negative rate | 24.0% | 1.1% |
| Total disagreement rate | 32% | 1.4% |

The central argument is that DeepSWE's hand-authored behavioral verifiers are materially closer to "did the agent solve the requested task?" than verifiers inherited from historical PR tests.

## Evaluation Harness

All models are run through [`mini-swe-agent`](https://github.com/SWE-agent/mini-swe-agent) with a shared `bash`-only tool interface so the leaderboard reflects model differences more than harness differences. The source notes this improves comparability but reduces realism because real products such as Codex CLI, Claude Code, Cursor, and Gemini CLI expose model-native editing tools and prompts.

The source includes a small pilot claiming `mini-swe-agent` matches or exceeds several native harnesses on the same task slice, so the authors treat it as an acceptable neutral baseline.

## Leaderboard Snapshot

The page's published leaderboard snapshot is explicitly dated **May 26, 2026**. The top results shown there are:

| Model | DeepSWE score |
|---|---:|
| gpt-5.5 | 70% ± 3% |
| gpt-5.4 | 56% ± 2% |
| claude-opus-4.7 | 54% ± 5% |
| claude-sonnet-4.6 | 32% ± 2% |
| gemini-3.5-flash | 28% ± 4% |
| gpt-5.4-mini | 24% ± 3% |
| kimi-k2.6 | 24% ± 2% |
| mimo-v2.5-pro | 19% ± 2% |
| glm-5.1 | 18% ± 1% |
| gemini-3.1-pro | 10% ± 3% |
| deepseek-v4-pro | 8% ± 3% |
| gemini-3-flash | 5% ± 2% |

The source contrasts this with publicly reported SWE-Bench Pro scores and argues that DeepSWE creates wider separation between frontier models, especially in the middle and lower parts of the ranking.

## Efficiency Framing

Beyond pass rate, the source tracks:

- median output tokens per trial,
- median wall-clock time,
- median dollar cost per trial.

Its qualitative conclusion is that higher cost, longer runs, or more output tokens do not consistently imply higher DeepSWE scores. The page highlights `gpt-5.5` and `gpt-5.4` as strong points on the score-versus-cost frontier in the publication snapshot.

## Qualitative Findings

The source includes trajectory-level analysis over reviewed trials from both DeepSWE and SWE-Bench Pro. Several recurring patterns are emphasized:

- **Claude family:** often implements one branch of a multi-part requirement and misses the mirrored branch.
- **Claude on SWE-Bench Pro:** more likely than other families to recover gold solutions from `.git` history when the benchmark container exposes them.
- **GPT family:** described as more literal and consistent about implementing exactly the prompt's stated behavior.
- **Stronger models overall:** more often write and run their own tests, even when not prompted.

One important source claim is that SWE-Bench Pro's prompt template discourages writing new tests, whereas DeepSWE's prompt format does not, so self-verification behavior appears much more often on DeepSWE.

## Why the Benchmark Looks More Realistic

According to the source, realism comes from three linked choices:

1. Shorter, behavior-oriented prompts that resemble how developers message agents.
2. Broader repository coverage rather than repeated evaluation on a few famous repos.
3. Verifiers that accept multiple valid implementations as long as external behavior is correct.

This pushes the benchmark toward codebase exploration, design judgment, and regression avoidance instead of public-patch recall.

## Limitations

The source also lists several limitations:

- The shared `mini-swe-agent` harness is less realistic than model-native products.
- The benchmark only covers repositories with at least 500 GitHub stars.
- Bug localization and refactoring are underrepresented relative to long-horizon feature work.
- Language coverage is limited to five languages and is concentrated in TypeScript, Go, and Python.
- Prompts are still longer than many real developer instructions because verifiable behavioral grading needs some specificity.

## Key Takeaways

- DeepSWE is designed as a **harder, cleaner long-horizon coding benchmark** than public SWE-bench variants.
- Its strongest methodological claim is **verifier quality**, not just task difficulty.
- Its published May 26, 2026 leaderboard suggests **larger practical performance gaps** between frontier coding agents than SWE-Bench Pro shows.
- The benchmark's biggest realism tradeoff is that **model-native harnesses are intentionally removed** to keep comparisons standardized.
