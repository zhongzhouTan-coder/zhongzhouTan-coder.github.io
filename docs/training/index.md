---
title: "Training"
summary: "Training and fine-tuning pages covering optimization behavior, transfer learning, and generalization in large models."
layout: default
confidence: high
sources:
  - logs/index.md
updated: 2026-07-28
---

# Training

- [GPT-1: Improving Language Understanding by Generative Pre-Training](gpt-1.md) — Introduces the decoder-only Transformer, the pre-train + fine-tune paradigm, task-agnostic input transformations, long-contiguous-text motivation, and transfer ablations.
- [GPT-2: Language Models are Unsupervised Multitask Learners](gpt-2.md) — Scales to 1.5B parameters on WebText; demonstrates zero-shot task transfer, byte-level BPE evaluation, prompt-only task conditioning, and contamination analysis.
- [GPT-3: Language Models are Few-Shot Learners](gpt-3.md) — Scales to 175B parameters; demonstrates in-context few-shot learning, scaling-law behavior, prompt-format evaluation settings, and limitations of context-only adaptation.
- [Megatron-LM: GPU-Cluster Training Parallelism](megatron-lm/) — Covers both Megatron-LM papers: intra-layer tensor model parallelism with `f`/`g` conjugate operators and BERT LayerNorm rearrangement (2019, 8.3B GPT on 512 V100), and the PTD-P recipe composing tensor, pipeline, and data parallelism with interleaved 1F1B and scatter/gather (2021, 1T GPT on 3072 A100).
- [GPipe: Micro-Batch Pipeline Parallelism](gpipe/) — Introduces synchronous micro-batch pipeline parallelism with activation recomputation, achieving near-linear speedup when training models across multiple accelerators by splitting mini-batches into micro-batches and streaming them through partitioned model layers.
- [Sequence Parallelism: Splitting Sequences Across GPUs](sequence-parallelism/) — Distributes input sequence chunks across GPUs with Ring Self-Attention (RSA), enabling longer-sequence Transformer training by splitting along the sequence dimension instead of model layers or hidden dimensions. Achieves 13.7× larger batch size and 3.0× longer sequences than tensor parallelism.
- [LLaMA: Open and Efficient Foundation Language Models](llama.md) — Trains 7B-65B decoder-only foundation models on 1.0T-1.4T public-data tokens, showing that smaller long-trained models can rival much larger closed models under practical inference budgets.
- [Intrinsic Dimensionality and Language Model Fine-Tuning](intrinsic-dimensionality-fine-tuning/index.md)
- [Socratic-SWE: Self-Evolving Coding Agents via Trace-Derived Skills](socratic-swe/index.md)
- [MiniMax Sparse Attention (MSA)](minimax-sparse-attention/index.md) — Blockwise sparse attention co-designed with GQA: lightweight Index Branch selects top-k KV blocks per group, Main Branch computes exact block-sparse softmax attention, trained with KL alignment loss. 28.4× FLOPs reduction and 14.2× prefill speedup at 1M context on a 109B MoE model.
