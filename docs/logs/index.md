---
title: "Wiki Index"
summary: "Top-level index of knowledge-base pages grouped by topic."
layout: default
confidence: high
sources:
  - AGENTS.md
  - .github/instructions/logs-maintenance.instructions.md
updated: 2026-07-27
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
- [The Transformer: Attention Is All You Need](../algorithms/transformer.md) — The foundational architecture: scaled dot-product attention, multi-head self-attention, sinusoidal positional encoding, encoder-decoder stacks, and the training recipe that launched modern LLMs.
- [Collaborative Multi-Head Attention](../algorithms/collaborative-attention.md) — Redesigns MHA with shared key/query projections and per-head mixing vectors, enabling 4× compression of Q/K dimensions; CP tensor decomposition for post-hoc conversion of pretrained models.
- [Multi-Query Attention: One Write-Head is All You Need](../algorithms/multi-query-attention.md) — Shares one K/V across all attention heads, shrinking incremental decoder memory bandwidth 8× for a 12× inference speedup with negligible quality loss.
- [Grouped-Query Attention in Llama 2](../algorithms/grouped-query-attention/index.md) — Llama 2's 34B/70B GQA decision: 8 KV groups, 30B MHA/MQA/GQA ablation, higher large-batch throughput, and simpler 8-GPU tensor-parallel serving than MQA.
- [DeepSeek-V2 Multi-Head Latent Attention](../algorithms/deepseek-v2-mla.md) — DeepSeek-V2's MLA design: low-rank joint K/V latent cache, decoupled RoPE, query compression, MoE serving context, 93.3% KV-cache reduction, and 5.76× maximum generation throughput versus DeepSeek 67B.
- [Matrix Exponentiation for Linear Transitions](../algorithms/matrix-exponentiation.md) — Binary matrix exponentiation, transition-matrix construction, linear recurrences, augmented state vectors, and fixed linear dynamic programming.
- [DeepSeek-V3.2: Sparse Attention, Scaled RL, and Thinking in Tool-Use](../algorithms/deepseek-v3.2/index.md) — Three innovations: DSA sparse attention with lightning indexer and top-k token selection, scaled GRPO with four MoE stabilization tricks, and cold-start + synthetic agentic task pipeline unifying reasoning with tool-use.
- [FlashAttention: IO-Aware Exact Attention](../algorithms/flashattention.md) — Original IO-aware exact attention algorithm: tiling, online softmax, recomputation, IO complexity, block-sparse extension, landscape evolutionary tree, and training/runtime results.
- [FlashAttention-2: Better Parallelism and Work Partitioning](../algorithms/flashattention-2.md) — Exact attention kernel optimization: reduced non-matmul overhead, sequence-parallel thread blocks, warp-level work partitioning, causal block skipping, Big Picture FA1→FA2 comparison diagram, landscape of GPU utilization gap closure, and A100/H100 performance results.
- [FlashAttention-3: Hopper Asynchrony and FP8 Attention](../algorithms/flashattention-3.md) — Hopper attention kernel design: warp specialization, TMA/WGMMA asynchrony, GEMM-softmax overlap, FP8 block quantization, incoherent processing, Big Picture async pipeline diagram, landscape of generation-specific hardware exploitation, and speed/accuracy results.
- [FlashAttention-4: Blackwell Attention Kernel Co-Design](../algorithms/flashattention-4.md) — Exact attention algorithm and Blackwell kernel design: asymmetric scaling response, exponential emulation, conditional softmax rescaling, TMEM-based pipelining, 2-CTA backward pass, LPT scheduling, landscape of shifting hardware bottlenecks, and performance results.
- [The Softmax Function: Properties, Motivation, and Interpretation](../algorithms/softmax.md) — Tutorial covering score-difference semantics, α parameter interpretation, three conceptual justifications (Gumbel noise, maximum entropy, exploration-exploitation), IO vs. IM model taxonomy, and complete mathematical properties.

## Training

- [Training](../training/index.md) — Category overview for model training, fine-tuning, transfer learning, and generalization pages.
- [GPT-1: Improving Language Understanding by Generative Pre-Training](../training/gpt-1.md) — Introduces the decoder-only Transformer, the pre-train + fine-tune paradigm, task-agnostic input transformations, long-contiguous-text motivation, and transfer ablations.
- [GPT-2: Language Models are Unsupervised Multitask Learners](../training/gpt-2.md) — Scales to 1.5B on WebText; demonstrates zero-shot task transfer, byte-level BPE evaluation, prompt-only task conditioning, and contamination analysis.
- [GPT-3: Language Models are Few-Shot Learners](../training/gpt-3.md) — Scales to 175B; demonstrates in-context few-shot learning, scaling-law behavior, prompt-format evaluation settings, and limitations of context-only adaptation.
- [Megatron-LM: GPU-Cluster Training Parallelism](../training/megatron-lm/index.md) — PTD-P training-system recipe for trillion-parameter GPT models: tensor parallelism inside nodes, pipeline parallelism across nodes, data parallelism across replicas, interleaved 1F1B scheduling, scatter/gather communication, and fused kernels.
- [GPipe: Micro-Batch Pipeline Parallelism](../training/gpipe/index.md) — Synchronous micro-batch pipeline parallelism with activation recomputation: splits mini-batches into micro-batches, pipelines them through partitioned layers, and applies synchronous gradient updates for near-linear speedup.
- [Sequence Parallelism: Splitting Sequences Across GPUs](../training/sequence-parallelism/index.md) — Distributes input sequence chunks across GPUs with Ring Self-Attention (RSA), introducing the fourth parallelism dimension alongside data, pipeline, and tensor parallelism. Achieves 13.7× larger batch size and 3.0× longer sequences than tensor parallelism.
- [LLaMA: Open and Efficient Foundation Language Models](../training/llama.md) — Original LLaMA family: 7B-65B decoder-only models trained on 1.0T-1.4T public-data tokens, inference-budget motivation, architecture defaults, efficient training implementation, benchmark comparisons, and safety limitations.
- [Intrinsic Dimensionality and Language Model Fine-Tuning](../training/intrinsic-dimensionality-fine-tuning/index.md) — Intrinsic-dimension view of pretrained language model fine-tuning: DID/SAID subspace training, low `d90` task dimensions, pretraining as downstream task compression, model-size trends, generalization correlations, and an editable Draw.io explainer.
- [Socratic-SWE: Self-Evolving Coding Agents via Trace-Derived Skills](../training/socratic-swe/index.md) — Closed-loop self-evolution framework: trace-derived Agent Skill Registry, skill-guided Generator with four-stage Verifier Gate, gradient-aligned Generator reward via cosine similarity to validation gradient, GDPO-normalized Solver reward, and 50.40% on SWE-bench Verified across three iterations.

## Hardware and Numerics

- [Hardware and Numerics](../hardware/index.md) — Category overview for hardware and numerics pages.
- [FlatQuant: Fast Learnable Affine Quantization](../hardware/flatquant.md) — Post-training LLM quantization method: learnable affine transformations, Kronecker factorization, per-channel scaling, learnable clipping, fused kernels, W4A4 accuracy, and inference latency results.
- [NVFP4: Blackwell 4-Bit Floating Point](../hardware/nvfp4.md) — NVIDIA Blackwell NVFP4 format, hierarchical FP8/FP32 scaling, memory benefits, Transformer Engine training recipe, RHT, stochastic rounding, and hardware support.

## Terms

- [Terms Glossary](../terms/index.md) — Alphabetical glossary of cross-paper technical terms with concise definitions and backlinks to the papers that use them.
- [Microbatch](../terms/microbatch.md) — A small chunk of a training batch used to enable pipeline parallelism; the unit of work in a pipeline schedule.
- [Scatter/Gather](../terms/scatter-gather.md) — Cross-node communication optimization that avoids redundant activation transfers over slow inter-node links.

## Sources
