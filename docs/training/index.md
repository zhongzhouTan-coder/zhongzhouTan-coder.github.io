---
title: "Training"
summary: "Training and fine-tuning pages covering optimization behavior, transfer learning, and generalization in large models."
layout: default
confidence: high
sources:
  - logs/index.md
updated: 2026-08-13
---

# Training

## Training Parallelism

- [Training Parallelism](parallelism/index.md) — Category hub for data, tensor, pipeline, and [sequence parallelism](../terms/sequence-parallelism.md) techniques.
- [Megatron-LM: GPU-Cluster Training Parallelism](parallelism/megatron-lm/index.md) — Covers both Megatron-LM papers: intra-layer tensor model parallelism with `f`/`g` conjugate operators and BERT [LayerNorm](../terms/layer-normalization.md) rearrangement (2019, 8.3B GPT on 512 V100), and the PTD-P recipe composing tensor, pipeline, and data parallelism with interleaved 1F1B and [scatter/gather](../terms/scatter-gather.md) (2021, 1T GPT on 3072 A100).
- [GPipe: Micro-Batch Pipeline Parallelism](parallelism/gpipe/index.md) — Introduces synchronous [micro-batch](../terms/microbatch.md) pipeline parallelism with activation recomputation, achieving near-linear speedup when training models across multiple accelerators.
- [Sequence Parallelism: Splitting Sequences Across GPUs](parallelism/sequence-parallelism/index.md) — Distributes input sequence chunks across GPUs with Ring Self-Attention (RSA), achieving 13.7× larger batch size and 3.0× longer sequences than tensor parallelism.

## Foundation Models

- [Foundation Models](foundation-models/index.md) — Category hub for classic decoder-only foundation model papers.
- [GPT-1: Improving Language Understanding by Generative Pre-Training](foundation-models/gpt-1.md) — Introduces the decoder-only Transformer, the pre-train + fine-tune paradigm, task-agnostic input transformations, long-contiguous-text motivation, and transfer ablations.
- [GPT-2: Language Models are Unsupervised Multitask Learners](foundation-models/gpt-2.md) — Scales to 1.5B parameters on WebText; demonstrates zero-shot task transfer, byte-level BPE evaluation, prompt-only task conditioning, and contamination analysis.
- [GPT-3: Language Models are Few-Shot Learners](foundation-models/gpt-3.md) — Scales to 175B parameters; demonstrates in-context few-shot learning, scaling-law behavior, prompt-format evaluation settings, and limitations of context-only adaptation.
- [LLaMA: Open and Efficient Foundation Language Models](foundation-models/llama.md) — Trains 7B-65B decoder-only foundation models on 1.0T-1.4T public-data tokens.

## DeepSeek

- [DeepSeek](deepseek/index.md) — Category hub for DeepSeek model training papers.
- [DeepSeek-V4: Million-Token Context via Hybrid Compressed Attention](deepseek/deepseek-v4/index.md) — 1.6T/284B [MoE](../terms/mixture-of-experts.md) models with CSA+HCA hybrid attention, [mHC](../terms/hyper-connections.md), and Muon optimizer; achieves 27% FLOPs and 10% [KV cache](../terms/kv-cache.md) of V3.2 at 1M-token contexts.

## Residual and Hyper-Connections

- [mHC: Manifold-Constrained Hyper-Connections](mhc/index.md) — Widens the residual stream into n parallel streams and constrains the mixing matrix to be doubly stochastic via Sinkhorn-Knopp, restoring the identity-mapping property; 27B MoE final loss 0.021 lower than baseline at 6.7% overhead.

## Kimi

- [Kimi](kimi/index.md) — Category hub for the Kimi model family.
- [Kimi Linear: Expressive Efficient Attention Architecture](kimi/kimi-linear/index.md) — Hybrid [linear attention](../terms/linear-attention.md) that for the first time outperforms full MLA across short-context, long-context, and RL: [KDA](../terms/kimi-delta-attention.md) extends Gated DeltaNet with channel-wise gating, 3:1 KDA-to-MLA layer ratio with NoPE, 48B [MoE](../terms/mixture-of-experts.md) with 3B active.
- [Kimi K3: Open 3T-Class Frontier Model](kimi/kimi-k3/index.md) — 2.8T-parameter native multimodal MoE with 104B active parameters, hybrid KDA/MLA attention, 1M-token context, Stable LatentMoE, multi-effort agentic RL, MoonEP balanced expert training, and long-rollout cache/sandbox infrastructure.

## Efficient Attention Training

- [Efficient Attention Training](efficient-attention/index.md) — Category hub for training approaches built around efficient attention mechanisms.
- [MiniMax Sparse Attention (MSA)](efficient-attention/minimax-sparse-attention/index.md) — Blockwise sparse attention co-designed with GQA: lightweight [Index Branch](../terms/lightning-indexer.md) selects top-k KV blocks per group, Main Branch computes exact block-sparse softmax attention, trained with KL alignment loss.
- [SWAT: Sliding Window Attention Training](efficient-attention/swat-sliding-window-attention/index.md) — Trains Transformers from scratch with sigmoid-based sliding window attention: replaces softmax with sigmoid to eliminate attention sink, combines balanced bidirectional ALiBi with RoPE for training stability.
- [Gated Delta Networks: Improving Mamba2 with Delta Rule](efficient-attention/gated-delta-networks/index.md) — Combines Mamba2-style global decay with DeltaNet's key-targeted correction ([Delta Rule](../terms/delta-rule.md)), preserving hardware-efficient chunkwise training.

## Fine-Tuning and Adaptation

- [Fine-Tuning and Adaptation](fine-tuning/index.md) — Category hub for fine-tuning, transfer learning, and self-evolution methods.
- [Intrinsic Dimensionality and Language Model Fine-Tuning](fine-tuning/intrinsic-dimensionality-fine-tuning/index.md)
- [Socratic-SWE: Self-Evolving Coding Agents via Trace-Derived Skills](fine-tuning/socratic-swe/index.md)
