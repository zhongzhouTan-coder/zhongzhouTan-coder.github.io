---
title: "Wiki Index"
summary: "Top-level index of knowledge-base pages grouped by topic."
layout: default
confidence: high
sources:
  - AGENTS.md
  - .github/instructions/logs-maintenance.instructions.md
updated: 2026-07-23
---

# Wiki Index

## Meta

- [Knowledge Base Introduction](../README.md)

## Benchmarks

- [Benchmarks](../benchmarks/index.md) — Category overview for benchmark pages.
- [Pier: Coding-Agent Evaluation Harness](../benchmarks/pier/index.md) — Harbor-compatible coding-agent evaluation harness focused on installed agents in sandboxed tasks, stricter ATIF trajectory conversion, `mini-swe-agent` integration, a DeepSWE-style local-dataset execution path, and a visual architecture explainer.
- [DeepSWE: Long-Horizon Software Engineering Benchmark](../benchmarks/deepswe/index.md) — Long-horizon coding benchmark with original tasks, 91 repositories across five languages, behavioral verifiers, a visual explainer of benchmark rationale and workflow, publication leaderboard snapshot, qualitative failure analysis, and limitations.
- [DeepSWE v1.1: Execution and Scoring Changes](../benchmarks/deepswe-v1-1/index.md) — DeepSWE execution-and-grading update with a visual explainer of committed-patch isolated verification, CTRF structured test reports, cleaner `main`-branch git environment, updated July 1, 2026 leaderboard snapshot, and v1 versus v1.1 impact.
- [τ-bench: Tool-Agent-User Interaction Benchmark](../benchmarks/tau-bench.md) — Original benchmark: two customer-service domains (retail and airline), pass^k metric, empirical results across 12 models, and failure analysis.
- [τ²-Bench: Mechanism and Design](../benchmarks/tau2-bench-mechanism.md) — Dec-POMDP formalism, dual-control domain, task generation, and evaluation methodology.
- [τ-Voice: Full-Duplex Voice Agent Benchmark](../benchmarks/tau-voice.md) — Extends τ²-bench to voice: tick-based orchestrator, controllable voice user simulator, 278 tasks across retail/airline/telecom, empirical results for Google/OpenAI/xAI, acoustic ablations, and error analysis.
- [EvalScope Perf: LLM Inference Stress Testing](../benchmarks/evalscope-perf.md) — Comprehensive model inference stress-testing tool: closed-loop and open-loop modes, SLA binary-search auto-tuning, multi-turn conversation benchmarking, embedding/rerank/multi-modal dataset support, vLLM bench parity comparison, and metrics coverage analysis.
- [AISBench Benchmark vs. EvalScope Perf](../benchmarks/aisbench-vs-evalscope-perf.md) — Competitive analysis for AISBench across load semantics, traffic and trace workloads, metrics, extensibility, operations, product gaps, validation methodology, and prioritized roadmap.
- [AISBench First-Class Performance Roadmap](../benchmarks/aisbench-first-class-roadmap/index.md) — Presentation-ready decision document defining five core AISBench implementation priorities, detailed EvalScope feature lessons, architecture, acceptance criteria, and phased delivery plan.
- [AISBench First-Class 性能能力路线图（中文）](../benchmarks/aisbench-first-class-roadmap/zh-cn.md) — 中文汇报版本，完整说明五项 First-Class 能力、EvalScope 功能分析、实现契约、验收标准、架构和分阶段交付计划。
- [AutoJudger: Agent-Driven Efficient MLLM Benchmarking](../benchmarks/autojudger.md) — Agent-driven framework for adaptive MLLM evaluation using IRT difficulty estimation and semantic-aware retrieval, achieving 90%+ ranking accuracy at ~4% data usage.

## Inference Systems and Frameworks

- [Frameworks](../frameworks/index.md) — Category overview for LLM serving and programming framework pages.
- [DSpark: Confidence-Scheduled Speculative Decoding](../frameworks/dspark/index.md) — DeepSeek speculative decoding framework that combines semi-autoregressive draft generation, calibrated confidence estimates, hardware-aware verification scheduling, production DeepSeek-V4 deployment results, and an editable Draw.io decoding-cycle visual.
- [Harbor: Agent Evaluation Framework Design](../frameworks/harbor-framework/index.md) — Why Harbor exists beyond Terminal-Bench, its task-centric design philosophy, `Job -> Trial -> Agent/Environment/Verifier` architecture, artifact and verifier isolation model, and the editable draw.io architecture asset.
- [SGLang: Structured Language Model Programs](../frameworks/sglang-framework.md) — Framework architecture, Python-embedded programming model, RadixAttention KV cache reuse, compressed FSM decoding, API speculative execution, and performance results.
- [vLLM Code Learning Path and Request Flow](../frameworks/vllm-code-learning-path.md) — Current vLLM codebase map, request lifecycle from OpenAI API entrypoint to worker execution, and an achievement-driven path to build a mini vLLM.
- [vLLM: PagedAttention Serving Framework](../frameworks/vllm-framework.md) — LLM serving framework design, PagedAttention KV-cache paging, block tables, copy-on-write sharing, scheduling/preemption, distributed execution, and throughput results.

## Algorithms

- [Algorithms](../algorithms/index.md) — Category overview for inference algorithm and kernel pages.
- [Matrix Exponentiation for Linear Transitions](../algorithms/matrix-exponentiation.md) — Binary matrix exponentiation, transition-matrix construction, linear recurrences, augmented state vectors, and fixed linear dynamic programming.
- [FlashAttention: IO-Aware Exact Attention](../algorithms/flashattention.md) — Original IO-aware exact attention algorithm: tiling, online softmax, recomputation, IO complexity, block-sparse extension, and training/runtime results.
- [FlashAttention-2: Better Parallelism and Work Partitioning](../algorithms/flashattention-2.md) — Exact attention kernel optimization: reduced non-matmul overhead, sequence-parallel thread blocks, warp-level work partitioning, causal block skipping, and A100/H100 performance results.
- [FlashAttention-3: Hopper Asynchrony and FP8 Attention](../algorithms/flashattention-3.md) — Hopper attention kernel design: warp specialization, TMA/WGMMA asynchrony, GEMM-softmax overlap, FP8 block quantization, incoherent processing, and speed/accuracy results.
- [FlashAttention-4: Blackwell Attention Kernel Co-Design](../algorithms/flashattention-4.md) — Exact attention algorithm and Blackwell kernel design: asynchronous MMA/TMEM pipeline, exponential emulation, conditional softmax rescaling, 2-CTA backward pass, LPT scheduling, and performance results.

## Training

- [Training](../training/index.md) — Category overview for model training, fine-tuning, transfer learning, and generalization pages.
- [Intrinsic Dimensionality and Language Model Fine-Tuning](../training/intrinsic-dimensionality-fine-tuning/index.md) — Intrinsic-dimension view of pretrained language model fine-tuning: DID/SAID subspace training, low `d90` task dimensions, pretraining as downstream task compression, model-size trends, generalization correlations, and an editable Draw.io explainer.
- [Socratic-SWE: Self-Evolving Coding Agents via Trace-Derived Skills](../training/socratic-swe/index.md) — Closed-loop self-evolution framework: trace-derived Agent Skill Registry, skill-guided Generator with four-stage Verifier Gate, gradient-aligned Generator reward via cosine similarity to validation gradient, GDPO-normalized Solver reward, and 50.40% on SWE-bench Verified across three iterations.

## Hardware and Numerics

- [Hardware and Numerics](../hardware/index.md) — Category overview for hardware and numerics pages.
- [FlatQuant: Fast Learnable Affine Quantization](../hardware/flatquant.md) — Post-training LLM quantization method: learnable affine transformations, Kronecker factorization, per-channel scaling, learnable clipping, fused kernels, W4A4 accuracy, and inference latency results.
- [NVFP4: Blackwell 4-Bit Floating Point](../hardware/nvfp4.md) — NVIDIA Blackwell NVFP4 format, hierarchical FP8/FP32 scaling, memory benefits, Transformer Engine training recipe, RHT, stochastic rounding, and hardware support.

## Sources
