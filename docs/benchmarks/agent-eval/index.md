---
title: "Agent Evaluation Benchmarks"
summary: "Benchmarks and harnesses for evaluating coding agents and tool-use agents: DeepSWE, Pier, τ-bench family, and AutoJudger."
layout: default
confidence: high
sources:
  - logs/index.md
updated: 2026-08-03
---

# Agent Evaluation Benchmarks

- [Pier: Coding-Agent Evaluation Harness](pier/index.md) — Harbor-compatible coding-agent evaluation harness focused on installed agents in sandboxed tasks, stricter ATIF trajectory conversion, `mini-swe-agent` integration, a DeepSWE-style local-dataset execution path, and a visual architecture explainer.
- [DeepSWE: Long-Horizon Software Engineering Benchmark](deepswe/index.md) — Long-horizon coding benchmark with original tasks, 91 repositories across five languages, behavioral verifiers, a visual explainer, and limitations.
- [DeepSWE v1.1: Execution and Scoring Changes](deepswe-v1-1/index.md) — DeepSWE execution-and-grading update with committed-patch isolated verification, CTRF structured test reports, and v1 versus v1.1 impact.
- [τ-bench: Tool-Agent-User Interaction Benchmark](tau-bench.md) — Original benchmark: two customer-service domains (retail and airline), pass^k metric, empirical results across 12 models, and failure analysis.
- [τ²-Bench: Mechanism and Design](tau2-bench-mechanism.md) — Dec-POMDP formalism, dual-control domain, task generation, and evaluation methodology.
- [τ-Voice: Full-Duplex Voice Agent Benchmark](tau-voice.md) — Extends τ²-bench to voice: tick-based orchestrator, controllable voice user simulator, 278 tasks, empirical results, and error analysis.
- [AutoJudger: Agent-Driven Efficient MLLM Benchmarking](autojudger.md) — Agent-driven framework for adaptive MLLM evaluation using IRT difficulty estimation and semantic-aware retrieval.
- [HORIZON: Agentic Hardware Design as Repository-Level Code Evolution](agentic-hardware-design/index.md) — Git-traced RTL self-evolution from Markdown harness to executable evaluator, with 100% best-so-far completion but large convergence-cost and reward-hacking caveats.
