---
title: "Agent Skill"
summary: "A reusable, structured procedural artifact that guides an AI agent on a class of tasks without retraining the underlying model."
tooltip: "An agent skill packages procedures, pitfalls, and sometimes references or scripts so an agent can reuse experience at inference time. A skill can be hand-authored or extracted from trajectories, but it still needs validation because plausible instructions may cause negative transfer."
layout: default
confidence: medium
category: general
sources:
  - raw/benchmarks/from-raw-experience-to-skill-consumption--arxiv-2605.23899v1.pdf
  - raw/training/socratic-swe-self-evolving-coding-agents--arxiv-2606.07412v1.pdf
  - raw/benchmarks/skillopt-executive-strategy-for-self-evolving-agent-skills--arxiv-2605.23904v2.pdf
  - raw/benchmarks/autoskill-experience-driven-lifelong-learning--arxiv-2603.01145v2.pdf
aliases:
  - agent skills
  - model-generated skill
  - procedural skill
mention_lint: off
appears_in:
  - docs/benchmarks/agent-eval/skilllens/index.md
  - docs/benchmarks/agent-eval/skillopt/index.md
  - docs/training/fine-tuning/socratic-swe/index.md
  - docs/benchmarks/agent-eval/autoskill/index.md
updated: 2026-08-31
---

# Agent Skill

**Agent Skill** is a reusable, structured procedural artifact that guides an AI agent through a class of tasks without retraining the underlying model.

## Why It Exists

Agents repeatedly encounter the same procedures, tool semantics, and failure modes. A skill stores that reusable knowledge so a later run can start with tested guidance instead of rediscovering it through trial and error.

## How It Works

A skill can be written by a person or extracted from successful and failed interaction trajectories. It normally contains a short description, procedural instructions, applicability conditions, pitfalls, and optional references or scripts. At inference time it is either injected into the agent prompt or retrieved from a library when the task matches its scope.

The important validation target is behavior: a skill is useful when it improves held-out task performance for the intended consumer. Fluent wording, a tidy format, or broad-sounding advice is not sufficient; a skill can encode an unsafe shortcut or a procedure that a particular model cannot execute.

## Tradeoffs

Skills reduce repeated reasoning and enable fast adaptation, but they can transfer errors and create interference. A skill extracted from a weak experience pool may cause negative transfer, while a skill that helps one model may be neutral or harmful for another. Large libraries add retrieval, composition, and versioning problems that a single prompt-injected skill does not expose.

## Common Confusions

- **Agent skill vs. tool:** A tool performs an operation; a skill explains when and how to combine operations for a class of tasks.
- **Agent skill vs. memory:** Memory records past information or events; a skill distills reusable procedure and decision rules.
- **Agent skill vs. fine-tuning:** A skill changes the runtime context without changing model parameters; fine-tuning changes the parameters or learned weights.

## Where It Appears

- [From Raw Experience to Skill Consumption](../benchmarks/agent-eval/skilllens/index.md) — Studies the complete experience-generation, extraction, and consumption lifecycle and validates a failure-aware extraction rubric against downstream utility.
- [SkillOpt: Executive Strategy for Self-Evolving Agent Skills](../benchmarks/agent-eval/skillopt/index.md) — Treats one skill document as the trainable state of a frozen agent and validates bounded text edits on held-out tasks.
- [Socratic-SWE](../training/fine-tuning/socratic-swe/index.md) — Distills coding-agent traces into a registry that guides targeted task generation in a self-evolution loop.
- [AutoSkill: Experience-Driven Lifelong Learning](../benchmarks/agent-eval/autoskill/index.md) — Extracts user-side behavioral rules into versioned `SKILL.md` artifacts and retrieves them for later requests without updating model weights.
