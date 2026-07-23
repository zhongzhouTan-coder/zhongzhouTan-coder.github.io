---
title: "Wiki Log"
summary: "Chronological log of knowledge-base updates and documentation maintenance."
layout: default
confidence: high
sources:
  - AGENTS.md
  - .github/instructions/logs-maintenance.instructions.md
updated: 2026-07-23
---

# Wiki Log

## 2026-07-23

- Added [Socratic-SWE: Self-Evolving Coding Agents via Trace-Derived Skills](../training/socratic-swe/index.md) to `docs/training/socratic-swe/index.md`, sourced from `raw/training/2606.07412v1.pdf` (arXiv:2606.07412v1, Jun 2026). Covers the closed-loop self-evolution framework: Agent Skill Registry (trace collection → skill extraction → deduplication), skill-guided task Generator with four-stage Verifier Gate (format/grounding/execution/semantics), gradient-aligned Generator reward ($R_G = \cos(g_\tau, G_v)$), GDPO-normalized three-component Solver reward, role-specific GRPO+GDPO training with shared weights, evolutionary landscape from static pipelines through self-play to skill-guided methods, benchmark results (50.40% SWE-bench Verified, 52.33% Lite, 47.20% Pro, 35.60% Terminal-Bench 2.0), Generator reward ablation, and an editable Draw.io big-picture diagram.

## 2026-07-17

- Added [EvalScope Perf: LLM Inference Stress Testing](../benchmarks/evalscope-perf.md) to `docs/benchmarks/evalscope-perf.md`, sourced from the EvalScope documentation at evalscope.readthedocs.io. Covers the full stress-testing tool architecture: closed-loop vs. open-loop load generation, SLA binary-search auto-tuning with AND/OR constraint logic, multi-turn conversation benchmarking with cache-hit metrics, dataset flexibility (random, ShareGPT, OpenQA, embedding, rerank, multi-modal), comprehensive metrics coverage (TTFT/TPOT/ITL/latency with full percentile breakdowns), parameter-aligned vLLM bench comparison showing statistical consistency, warmup support, WandB/SwanLab/ClearML visualization, and documented failure modes.
- Added [AISBench Benchmark vs. EvalScope Perf](../benchmarks/aisbench-vs-evalscope-perf.md), sourced from both projects' current documentation and repositories. Compares positioning, load semantics, traffic and trace workloads, metrics, extensibility, user experience, result operations, and risks; records an AISBench SWOT, prioritized roadmap, and reproducible head-to-head experiment design at medium confidence.
- Added an editable Draw.io competitive big-picture diagram to [AISBench Benchmark vs. EvalScope Perf](../benchmarks/aisbench-vs-evalscope-perf.md), showing the shared baseline, each product's differentiated strengths, the central product gap, and AISBench's recommended winning path.
- Added [AISBench First-Class Performance Roadmap](../benchmarks/aisbench-first-class-roadmap/index.md), a presentation-ready decision document defining metric, load-model, SLA-capacity, evidence-store, and one-command UX capabilities as the five first-class priorities. Includes a detailed EvalScope feature explanation, implementation contracts, acceptance criteria, delivery phases, and an editable Draw.io roadmap.
- Added the complete Chinese presentation edition, [AISBench First-Class 性能能力路线图](../benchmarks/aisbench-first-class-roadmap/zh-cn.md), preserving `First-Class` as the product and architecture term while retaining the five priorities, detailed EvalScope analysis, implementation contracts, acceptance criteria, architecture, delivery phases, and reciprocal language navigation.

## 2026-05-30

- Added [τ-Voice: Full-Duplex Voice Agent Benchmark](../benchmarks/tau-voice.md) to `docs/benchmarks/tau-voice.md`, sourced from the τ-Voice preprint (arXiv:2603.13686v1, Mar 2026). Covers tick-based orchestrator design, voice user simulator pipeline (7 personas, audio environment, turn-taking policy), three domains (278 total tasks), three audio-native providers, Clean vs Realistic conditions, acoustic ablation results (accents most damaging at -10pp avg), voice interaction quality metrics, qualitative error analysis (79-90% agent errors), and key conclusions.
- Added [τ-bench: Tool-Agent-User Interaction Benchmark](../benchmarks/tau-bench.md) to `docs/benchmarks/tau-bench.md`, sourced from `raw/benchmark/tau-bench.pdf` (arXiv:2406.12045v1). Covers benchmark architecture, two domains (τ-retail 115 tasks, τ-airline 50 tasks), three-stage construction, pass^k metric definition, full empirical results table for 12 models, method comparison, domain policy ablation, and failure analysis.

## 2026-05-29

- Added [Knowledge Base Introduction](../README.md) to `docs/README.md` as a meta page sourced from `AGENTS.md`. Describes the repo structure, category organization, navigation workflow, and the docs lint command.
- Added [τ²-Bench: Mechanism and Design](../benchmarks/tau2-bench-mechanism.md) to `docs/benchmarks/tau2-bench-mechanism.md`, sourced from `raw/benchmark/tau2-bench.pdf`. Covers Dec-POMDP formalism, dual-control architecture, five-stage domain construction, programmatic task generation, user simulator design, ablation modes, and empirical results.

## 2026-06-02

- Migrated GitHub-only agent guidance into `AGENTS.md` so Codex can read repository rules directly, including docs front matter, category handling, logs maintenance, and rendering constraints.
- Copied the docs ingest agent and instruction files from `.github/` into `.codex/agents/` and `.codex/instructions/` so the repository has a Codex-local agent setup.
- Copied the PDF skill assets from `.github/skills/pdf/` into `.codex/skills/pdf/` as a Codex-facing repo-local skill mirror.

## 2026-06-11

- Added [NVFP4: Blackwell 4-Bit Floating Point](../hardware/nvfp4.md) to `docs/hardware/nvfp4.md`, sourced from NVIDIA's NVFP4 inference blog and Transformer Engine NVFP4 documentation captured in `raw/nvidia/nvfp4-references.md`. Covers E2M1 encoding, FP8 E4M3 block scaling, FP32 tensor scaling, memory reduction claims, Transformer Engine recipe defaults, stochastic rounding, Random Hadamard Transform, GEMM layout, distributed-training behavior, and supported hardware.

## 2026-06-15

- Added [FlashAttention: IO-Aware Exact Attention](../algorithms/flashattention.md) to `docs/algorithms/flashattention.md`, sourced from `raw/infer-algorithm/2205.14135v2.pdf`. Covers the original tiled exact attention algorithm, online softmax statistics, recomputation in backward, IO complexity, block-sparse FlashAttention, empirical training speedups, long-context results, limitations, and relationship to later FlashAttention versions.
- Added [FlashAttention-2: Better Parallelism and Work Partitioning](../algorithms/flashattention-2.md) to `docs/algorithms/flashattention-2.md`, sourced from `raw/infer-algorithm/2307.08691v1.pdf`. Covers reduced non-matmul overhead, logsumexp softmax bookkeeping, sequence-level thread-block parallelism, backward atomic accumulation, warp-level work partitioning, causal block skipping, A100/H100 attention-kernel results, and GPT-style training throughput.
- Added [FlashAttention-3: Hopper Asynchrony and FP8 Attention](../algorithms/flashattention-3.md) to `docs/algorithms/flashattention-3.md`, sourced from `raw/infer-algorithm/2407.08608v2.pdf`. Covers Hopper-specific warp specialization, TMA/WGMMA producer-consumer scheduling, ping-pong and two-stage GEMM-softmax overlap, FP8 layout handling, block quantization, incoherent processing, backward-pass acceleration, empirical speedups, and limitations.
- Added [SGLang: Structured Language Model Programs](../frameworks/sglang-framework.md) to `docs/frameworks/sglang-framework.md`, sourced from `raw/sglang/2312.07104v2.pdf`. Covers the SGLang frontend primitives, interpreter/compiler execution modes, RadixAttention KV cache reuse, cache-aware scheduling, compressed finite-state-machine decoding, API speculative execution, evaluation setup, throughput/latency results, production deployment observations, and limitations.
- Added [vLLM: PagedAttention Serving Framework](../frameworks/vllm-framework.md) to `docs/frameworks/vllm-framework.md`, sourced from `raw/vllm/2309.06180v1.pdf`. Covers the PagedAttention algorithm, KV-cache block tables, virtual-memory analogy, copy-on-write sharing for parallel sampling and beam search, shared-prefix reuse, scheduling/preemption, distributed execution, implementation details, evaluation setup, throughput results, ablations, and limitations.
- Added [FlashAttention-4: Blackwell Attention Kernel Co-Design](../algorithms/flashattention-4.md) to `docs/algorithms/flashattention-4.md`, sourced from `raw/infer-algorithm/2603.05451v1.pdf`. Covers FA4's Blackwell hardware motivation, forward pipeline, software-emulated exponentials, conditional online-softmax rescaling, backward TMEM pipeline, 2-CTA MMA mode, deterministic backward pass, LPT scheduling, CuTe-DSL implementation, benchmark results, and the source's B200/B100 hardware-name inconsistency.
- Refactored [Wiki Index](../logs/index.md) categories, grouping benchmark pages, inference systems/frameworks, attention/kernel algorithms, and hardware/numerics.
- Refactored `docs/` from the previous layer folders into category folders: `docs/benchmarks/`, `docs/frameworks/`, `docs/algorithms/`, and `docs/hardware/`; added category index pages, removed legacy classification front matter, and updated Jekyll navigation/configuration.

## 2026-06-16

- Added [FlatQuant: Fast Learnable Affine Quantization](../hardware/flatquant.md) to `docs/hardware/flatquant.md`, sourced from `raw/quantization/2410.09426v4.pdf`. Covers FlatQuant's flatness motivation, learnable affine transformations, Kronecker factorization, per-channel scaling, learnable clipping thresholds, Transformer integration, fused Triton kernel design, W4A4 accuracy results, latency speedups, ablations, and limitations.

## 2026-06-24

- Added [Matrix Exponentiation for Linear Transitions](../algorithms/matrix-exponentiation.md) at the time under the legacy path `docs/layer_0/matrix-exponentiation.md`, sourced from `raw/algorithm/Matrix exponentiation | HackerEarth.pdf`. Covers binary matrix exponentiation, row-vector transition-matrix construction, Fibonacci and general linear recurrences, prefix sums, coupled sequences, fixed linear dynamic-programming transitions, modular arithmetic, and repeated-query optimization in layer 0.

## 2026-07-06

- Added [DeepSWE: Long-Horizon Software Engineering Benchmark](../benchmarks/deepswe/index.md) to `docs/benchmarks/deepswe/index.md`, sourced from `raw/benchmark/deepswe.md`. Covers benchmark motivation against contamination and verifier mismatch, corpus scope (113 tasks, 91 repositories, 5 languages), behavioral-verifier methodology, shared `mini-swe-agent` harness, the publication leaderboard snapshot dated 2026-05-26, qualitative failure patterns across model families, and stated limitations.
- Added [DeepSWE v1.1: Execution and Scoring Changes](../benchmarks/deepswe-v1-1/index.md) to `docs/benchmarks/deepswe-v1-1/index.md`, sourced from `raw/benchmark/deepswev1dot1.md`. Covers the v1.1 execution-and-grading update for the same 113 tasks: committed-diff isolated verification, CTRF per-test reporting, the cleaner `main`-branch git environment, updated July 1, 2026 leaderboard snapshot, removal of wall-clock reporting, and the reported similarity between v1 and v1.1 results.
- Added [vLLM Code Learning Path and Request Flow](../frameworks/vllm-code-learning-path.md) to `docs/frameworks/vllm-code-learning-path.md`, sourced from the current `vllm` codebase entrypoint, engine, scheduler, KV-cache, executor, and worker files. Covers the current request path from `/v1/chat/completions` through `AsyncLLM`, `EngineCore`, `Scheduler`, `KVCacheManager`, `Executor`, `GPUWorker`, `GPUModelRunner`, output processing, and an achievement-driven staged plan to build a mini vLLM.
- Updated [Frameworks](../frameworks/index.md) and [Wiki Index](../logs/index.md) to include the new code-oriented vLLM learning page alongside the existing paper-oriented framework notes.

## 2026-07-09

- Added [Pier: Coding-Agent Evaluation Harness](../benchmarks/pier/index.md) to `docs/benchmarks/pier/index.md`, sourced from `raw/benchmark/pier.md`. Covers why Pier exists as a smaller Harbor-compatible evaluation harness, its job/trial/agent/environment architecture, filtered-network installed-agent support, stricter ATIF trajectory conversion, the `mini-swe-agent` adapter path, the constraints of the current local-dataset-only build, and a DeepSWE-style execution recipe plus visual explainer assets.
- Updated [Benchmarks](../benchmarks/index.md) and [Wiki Index](../logs/index.md) to include the new Pier evaluation-harness page.
- Updated [DeepSWE: Long-Horizon Software Engineering Benchmark](../benchmarks/deepswe/index.md), sourced from `raw/benchmark/deepswe.md`, with a richer visual explainer section and added editable/source assets at `docs/benchmarks/assets/deepswe-explainer.drawio` and `docs/benchmarks/assets/deepswe-explainer.svg` to show why DeepSWE was introduced and how its benchmark pipeline works.
- Updated [Wiki Index](../logs/index.md) to note the new DeepSWE visual explainer on the benchmark page.
- Updated [DeepSWE v1.1: Execution and Scoring Changes](../benchmarks/deepswe-v1-1/index.md), sourced from `raw/benchmark/deepswev1dot1.md`, with a visual explainer section and added editable/source assets at `docs/benchmarks/assets/deepswe-v1-1-explainer.drawio` and `docs/benchmarks/assets/deepswe-v1-1-explainer.svg` to show the committed-diff verification flow, CTRF reporting, and why v1.1 is a measurement cleanup rather than a new benchmark corpus.
- Updated [Wiki Index](../logs/index.md) to note the new DeepSWE v1.1 visual explainer on the benchmark page.
- Added [Harbor: Agent Evaluation Framework Design](../frameworks/harbor-framework/index.md) to `docs/frameworks/harbor-framework/index.md`, sourced from the inspected Harbor README, docs, and runtime code copied under `raw/harbor/`. Covers why Harbor exists, its task-centric design philosophy, `Job -> Trial -> Agent/Environment/Verifier` architecture, registry and distribution model, policy-driven sandboxing, verifier isolation, and the editable draw.io asset at `docs/frameworks/assets/harbor-architecture.drawio`.
- Updated [Frameworks](../frameworks/index.md) and [Wiki Index](../logs/index.md) to include the new Harbor framework page.
- Updated [Harbor: Agent Evaluation Framework Design](../frameworks/harbor-framework/index.md) with a second, philosophy-first visual explainer and added complementary assets at `docs/frameworks/assets/harbor-design-philosophy.drawio` and `docs/frameworks/assets/harbor-design-philosophy.svg` to show Harbor's motivating pressures, core principles, architecture consequences, and intended outcomes without replacing the existing runtime architecture visual.
- Refactored [Pier: Coding-Agent Evaluation Harness](../benchmarks/pier/index.md), [DeepSWE: Long-Horizon Software Engineering Benchmark](../benchmarks/deepswe/index.md), [DeepSWE v1.1: Execution and Scoring Changes](../benchmarks/deepswe-v1-1/index.md), and [Harbor: Agent Evaluation Framework Design](../frameworks/harbor-framework/index.md) into folder-backed `index.md` pages so VS Code Markdown preview can resolve relative image paths while GitHub Pages keeps pretty trailing-slash URLs.
- Removed the repo-local Codex mirror files `.codex/AGENTS.md` and `.codex/agents/docs-ingest.toml`, and updated [Knowledge Base Introduction](../README.md) to point back to `AGENTS.md` for the docs workflow source of truth.
- Moved [Matrix Exponentiation for Linear Transitions](../algorithms/matrix-exponentiation.md) from the legacy `docs/layer_0/` path into `docs/algorithms/`, updated [Algorithms](../algorithms/index.md) and [Wiki Index](../logs/index.md) to match the category-based layout, and revised `AGENTS.md` plus the `.github` docs-ingest/front-matter instructions so `doc_layer` now consistently represents confidence rather than folder placement.
- Simplified the docs front matter schema to use `confidence: high|medium|low` directly, removed the redundant `doc_layer` field from pages and instructions, and updated the docs lint script so every page under `docs/` now requires an explicit readable confidence value.

## 2026-07-14

- Added [Intrinsic Dimensionality and Language Model Fine-Tuning](../training/intrinsic-dimensionality-fine-tuning/index.md) to `docs/training/intrinsic-dimensionality-fine-tuning/index.md`, sourced from `raw/training/2012.13255v1.pdf`. Covers DID/SAID intrinsic-dimension fine-tuning, MRPC/QQP `d90` findings, pretraining as task compression, model-size trends, generalization correlations, the intrinsic-dimension compression bound, and an editable Draw.io explainer at `docs/training/intrinsic-dimensionality-fine-tuning/intrinsic-dimensionality-fine-tuning.drawio`.
- Added the new [Training](../training/index.md) category and updated [Wiki Index](../logs/index.md), [Knowledge Base Introduction](../README.md), and the site landing page to include training and fine-tuning pages.

## 2026-07-15

- Added [DSpark: Confidence-Scheduled Speculative Decoding](../frameworks/dspark/index.md) to `docs/frameworks/dspark/index.md`, sourced from `raw/sp-infer/2607.05147v1.pdf`. Covers DSpark's semi-autoregressive drafter, Markov/RNN sequential heads, calibrated confidence head, hardware-aware prefix scheduler, offline accepted-length results, DeepSeek-V4 production deployment, live traffic throughput/interactivity results, limitations, and an editable Draw.io asset at `docs/frameworks/dspark/dspark-decoding-cycle.drawio`.
- Updated [Frameworks](../frameworks/index.md) and [Wiki Index](../logs/index.md) to include the new DSpark speculative decoding page.

- Restyled all docs pages to comply with the updated `docs-content-structure.instructions.md`: converted `## Summary` sections to `## TL;DR` with three-sentence (What/How/The number) format, added `## The Core Idea`, `## Why This Exists`, `## Where It Breaks` (failure mode table), `## One Thing to Remember` (bold key phrase paragraph), and `## Go Deeper` sections across all paper-insight pages. Updated all front matter `updated` dates to 2026-07-15. DSpark page already conformed; nvfp4 and vllm-code-learning-path received structural additions appropriate for their format-reference and learning-guide roles.
