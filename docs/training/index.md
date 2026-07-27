---
title: "Training"
summary: "Training and fine-tuning pages covering optimization behavior, transfer learning, and generalization in large models."
layout: default
confidence: high
sources:
  - logs/index.md
updated: 2026-07-27
---

# Training

- [GPT-1: Improving Language Understanding by Generative Pre-Training](gpt-1.md) — Introduces the decoder-only Transformer, the pre-train + fine-tune paradigm, task-agnostic input transformations, long-contiguous-text motivation, and transfer ablations.
- [GPT-2: Language Models are Unsupervised Multitask Learners](gpt-2.md) — Scales to 1.5B parameters on WebText; demonstrates zero-shot task transfer, byte-level BPE evaluation, prompt-only task conditioning, and contamination analysis.
- [GPT-3: Language Models are Few-Shot Learners](gpt-3.md) — Scales to 175B parameters; demonstrates in-context few-shot learning, scaling-law behavior, prompt-format evaluation settings, and limitations of context-only adaptation.
- [Megatron-LM: GPU-Cluster Training Parallelism](megatron-lm/) — Composes tensor, pipeline, and data parallelism with interleaved scheduling, scatter/gather communication, activation recomputation, and fused kernels to train trillion-parameter GPT models on thousands of GPUs.
- [GPipe: Micro-Batch Pipeline Parallelism](gpipe/) — Introduces synchronous micro-batch pipeline parallelism with activation recomputation, achieving near-linear speedup when training models across multiple accelerators by splitting mini-batches into micro-batches and streaming them through partitioned model layers.
- [Sequence Parallelism: Splitting Sequences Across GPUs](sequence-parallelism/) — Distributes input sequence chunks across GPUs with Ring Self-Attention (RSA), enabling longer-sequence Transformer training by splitting along the sequence dimension instead of model layers or hidden dimensions. Achieves 13.7× larger batch size and 3.0× longer sequences than tensor parallelism.
- [LLaMA: Open and Efficient Foundation Language Models](llama.md) — Trains 7B-65B decoder-only foundation models on 1.0T-1.4T public-data tokens, showing that smaller long-trained models can rival much larger closed models under practical inference budgets.
- [Intrinsic Dimensionality and Language Model Fine-Tuning](intrinsic-dimensionality-fine-tuning/index.md)
- [Socratic-SWE: Self-Evolving Coding Agents via Trace-Derived Skills](socratic-swe/index.md)
