---
title: "Wiki Index"
summary: "Top-level navigation for the knowledge base, organized by category and major topic hub."
layout: default
confidence: high
sources:
  - AGENTS.md
  - .github/instructions/logs-maintenance.instructions.md
updated: 2026-10-10
---

# Wiki Index

## Start Here

- [Knowledge Base Introduction](../README.md)
- [Terms Glossary](../terms/index.md)
- [Chronological Wiki Log](log.md)

## Algorithms

- [Algorithms](../algorithms/index.md) — Category overview.
- [Foundations](../algorithms/foundations/index.md) — Transformer, normalization, probability, sequence, and linear algebra.
- [Attention Algorithms](../algorithms/attention/index.md) — Exact, approximate, architecture-specific, and compressed-KV attention.
- [Distributed Inference](../algorithms/distributed-inference/index.md) — Multi-device inference algorithms.
- [Length Generalization](../algorithms/length-generalization/index.md) — Extrapolating computation beyond training lengths.
- [Model Architecture Algorithms](../algorithms/model-architectures/index.md) — Model-wide algorithmic reports.

## Benchmarks

- [Benchmarks](../benchmarks/index.md) — Category overview.
- [Agent Evaluation](../benchmarks/agent-eval/index.md) — Coding, tool-use, interaction, and skill-learning benchmarks.
- [Serving Performance](../benchmarks/serving-performance/index.md) — Inference stress testing and performance roadmaps.

## Frameworks

- [Frameworks](../frameworks/index.md) — Category overview.
- [Inference Serving Systems](../frameworks/serving/index.md) — Runtimes, caches, and deployment pipelines.
- [Model Architecture and KV Cache Paths](../frameworks/serving/model-kv-cache/index.md) — Five checkpoint-specific architecture and vLLM GPU/Ascend code readings.
- [Compilers and Kernel Frameworks](../frameworks/compilers-kernels/index.md) — GPU and NPU programming stacks.
- [Programming Languages](../frameworks/languages/index.md) — C++ and Python semantics and runtimes.
- [Runtime Infrastructure](../frameworks/infrastructure/index.md) — Sandboxing, orchestration, and observability.
- [Speculative Decoding Systems](../frameworks/speculative-decoding/index.md) — Draft-model and diffusion-based acceleration.

## Hardware and Numerics

- [Hardware and Numerics](../hardware/index.md) — Category overview.
- [Accelerator Architectures](../hardware/accelerators/index.md) — NVIDIA GPU and Ascend NPU families.
- [Numerics and Quantization](../hardware/numerics/index.md) — Low-precision formats and quantization.
- [Hardware Kernels and Dataflow](../hardware/kernels-dataflow/index.md) — Communication-aware kernels and spatial execution.

## Training

- [Training](../training/index.md) — Category overview.
- [Training Parallelism](../training/parallelism/index.md) — Tensor, pipeline, and sequence parallelism.
- [Foundation Models](../training/foundation-models/index.md) — GPT and LLaMA papers.
- [Model Families](../training/model-families/index.md) — DeepSeek and Kimi reports.
- [Efficient Attention Training](../training/efficient-attention/index.md) — Sparse, recurrent, and shared-cache attention.
- [Model Adaptation](../training/adaptation/index.md) — Fine-tuning, transfer, and agent self-evolution.
- [Training Architectures](../training/architectures/index.md) — Training-motivated architectural mechanisms.
