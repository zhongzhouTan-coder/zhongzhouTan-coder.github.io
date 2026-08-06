---
title: "Kimi Delta Attention"
summary: "A gated linear-attention mechanism that extends delta-rule recurrent memory with channel-wise decay and hardware-efficient chunkwise computation."
tooltip: "Kimi Delta Attention stores history in a fixed-size recurrent state instead of a full KV cache for every layer. It improves Gated DeltaNet with per-channel retention and Kimi K3 further bounds decay so Tensor Core chunk kernels stay numerically safe."
layout: default
confidence: high
category: algorithms
sources:
  - raw/training/kimi-linear-expressive-efficient-attention--paper.pdf
  - raw/training/k3-technical-report--paper.pdf
aliases:
  - KDA
appears_in:
  - docs/frameworks/index.md
  - docs/frameworks/triton/triton-in-vllm.md
  - docs/frameworks/vllm-ascend/kimi-k3-moe-forward.md
  - docs/frameworks/vllm/index.md
  - docs/frameworks/vllm/vllm-kimi-k3-code-reading.md
  - docs/training/efficient-attention/gated-delta-networks/index.md
  - docs/training/index.md
  - docs/training/kimi/index.md
  - docs/training/kimi/kimi-k3/index.md
  - docs/training/kimi/kimi-linear/index.md
updated: 2026-08-06
---

# Kimi Delta Attention

**Kimi Delta Attention** is a gated linear-attention mechanism that maintains a fixed-size key–value recurrent memory updated with channel-wise decay and a delta-rule correction.

## Why It Exists

Full attention preserves exact token access but makes cache memory and per-token reads grow with sequence length. Earlier gated delta models use fixed recurrent state, but scalar decay is too coarse for selective long-context memory. KDA gives each key channel its own retention factor so different memory dimensions can forget at different rates.

## How It Works

For state $\mathbf S_t$, key $\mathbf k_t$, value $\mathbf v_t$, retention vector $\alpha_t$, and write strength $\beta_t$, KDA applies per-channel decay, a key-targeted delta correction, and a new key–value write:

$$
\mathbf S_t = (I-\beta_t \mathbf k_t \mathbf k_t^\top)\operatorname{Diag}(\alpha_t)\mathbf S_{t-1}+\beta_t\mathbf k_t\mathbf v_t^\top.
$$

Kimi Linear makes this practical with chunkwise parallel computation and a hardware-efficient DPLR constraint. Kimi K3 keeps the same hybrid role but changes the log-decay mapping to a lower-bounded sigmoid so reciprocal cumulative decay remains inside BF16 dynamic range and causal tiles can use dense Tensor Core matrix multiplications.

## Tradeoffs

KDA is still a compressed recurrent memory, so it can blur exact token identities. Kimi models therefore interleave KDA with periodic MLA layers, using KDA for cheap long-context mixing and MLA for exact global content interaction.

## Common Confusions

- **KDA vs. linear attention:** KDA is a specific gated delta-rule linear-attention mechanism.
- **KDA vs. MLA:** KDA stores fixed recurrent state; MLA still stores compressed token-level KV cache.

## Where It Appears

- [Kimi Linear](../training/kimi/kimi-linear/index.md) — Introduces KDA as the central 3:1 hybrid attention operator.
- [Kimi K3](../training/kimi/kimi-k3/index.md) — Scales KDA to a 2.8T MoE and modifies decay numerics for large-scale training kernels.
- [vLLM Kimi K3 Code Reading](../frameworks/vllm/vllm-kimi-k3-code-reading.md) — Traces the deployed KDA path in vLLM.
- [Frameworks](../frameworks/index.md) — Framework pages covering LLM serving systems and structured language-model programming runtimes.
- [Triton in Practice: How vLLM and vllm-ascend Use Triton](../frameworks/triton/triton-in-vllm.md) — A codebase-driven tour of Triton kernel development in vLLM (NVIDIA GPU) and vllm-ascend (Ascend NPU), covering infrastructure.
- [vLLM](../frameworks/vllm/index.md) — vLLM serving framework pages: PagedAttention paper, request-flow code learning path, continuous batching, and Kimi K3 code.
- [vLLM-Ascend Kimi K3 MoE Forward Insight](../frameworks/vllm-ascend/kimi-k3-moe-forward.md) — Fresh code-reading insight for how the latest vllm-ascend routed-MoE substrate would execute a Kimi K3-style forward pass.
- [Gated Delta Networks: Improving Mamba2 with Delta Rule](../training/efficient-attention/gated-delta-networks/index.md) — Gated DeltaNet combines global state decay with key-targeted delta updates to improve fixed-state sequence memory while.
- [Training](../training/index.md) — Training and fine-tuning pages covering optimization behavior, transfer learning, and generalization in large models.
- [Kimi](../training/kimi/index.md) — Kimi model family: the Kimi Linear attention architecture and the Kimi K3 frontier model.

## Related Terms

- [Linear Attention](linear-attention.md) — The broader fixed-state attention family.
- [Delta Rule](delta-rule.md) — The key-targeted correction used in KDA's recurrent update.
- [KV Cache](kv-cache.md) — The explicit token-history store that KDA partially avoids.
