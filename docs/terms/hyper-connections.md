---
title: "Hyper-Connections"
summary: "A residual-connection family that widens the residual stream into n parallel streams mixed by learnable mappings; mHC constrains that mixing to a doubly stochastic manifold to keep it trainable at scale."
tooltip: "Hyper-Connections (HC) expand the classic single residual path into n parallel streams mixed by learnable matrices. Unconstrained, the composite mixing amplifies or shrinks signal so training destabilizes; Manifold-Constrained Hyper-Connections (mHC) project the mixing matrix to be doubly stochastic, restoring the identity-mapping property. Not to be confused with a plain residual connection."
layout: default
confidence: high
category: training
sources:
  - raw/training/mhc-manifold-constrained-hyper-connections--arxiv-2512.24880.pdf
aliases:
  - hyper-connection
  - hc
  - mhc
  - manifold-constrained-hyper-connections
appears_in:
  - docs/training/mhc/index.md
  - docs/training/deepseek/deepseek-v4/index.md
updated: 2026-08-06
---

# Hyper-Connections

**Hyper-Connections** is a residual-connection family that widens the residual stream from a single path to $n$ parallel streams and mixes those streams with learnable mappings, parameterized so that performance can improve without adding layer FLOPs.

## Why It Exists

The classic residual connection $\mathbf{x}_{l+1} = \mathbf{x}_l + \mathcal{F}(\mathbf{x}_l, \mathcal{W}_l)$ carries exactly one stream of width $C$ from layer to layer. Its identity-mapping property keeps gradients stable, but it also means the residual stream's information capacity is locked to the layer's hidden width — the same dimension that drives FLOPs. Hyper-Connections break that coupling: widen the stream to $n \times C$ and let learnable matrices decide how streams mix, read out into a layer, and write back. The knowledge base uses this term for both the original Hyper-Connections (HC) design and its stabilized Manifold-Constrained Hyper-Connections (mHC) variant.

## How It Works

HC keeps the layer function $\mathcal{F}$ unchanged but replaces the single stream with an $n \times C$ hidden matrix $\mathbf{x}_l$. Three learnable mappings govern the stream: $\mathcal{H}_l^{\text{res}} \in \mathbb{R}^{n \times n}$ mixes the streams, $\mathcal{H}_l^{\text{pre}} \in \mathbb{R}^{1 \times n}$ aggregates streams into the layer input, and $\mathcal{H}_l^{\text{post}} \in \mathbb{R}^{1 \times n}$ writes the layer output back onto the stream:

$$
\mathbf{x}_{l+1} = \mathcal{H}_l^{\text{res}} \mathbf{x}_l + \mathcal{H}_l^{\text{post}^\top} \mathcal{F}(\mathcal{H}_l^{\text{pre}} \mathbf{x}_l, \mathcal{W}_l).
$$

Each mapping is a sum of a dynamic (input-dependent) term and a static bias term. mHC constrains $\mathcal{H}_l^{\text{res}}$ to be doubly stochastic (non-negative, rows and columns sum to 1) via the Sinkhorn-Knopp algorithm, and constrains $\mathcal{H}_l^{\text{pre}}, \mathcal{H}_l^{\text{post}}$ to be non-negative. With expansion rate $n = 4$, mHC adds only 6.7% wall-clock overhead in large-scale MoE training.

## Tradeoffs

- The widened stream multiplies memory-access (I/O) cost by roughly $n$; fused kernels and recomputation are required to keep training fast.
- Approximate Sinkhorn-Knopp (20 iterations) leaves the backward gain slightly above 1, but bounded near ~1.6 instead of ~3000 for unconstrained HC.
- Non-negative pre/post gating trades some HC expressivity for cancellation-free signal flow.
- For $n = 1$ the design degenerates to a plain residual connection with no benefit.

## Common Confusions

- **Hyper-Connections vs. residual connection:** A residual connection is the single-stream identity-plus-function form; Hyper-Connections is a family that widens that stream into multiple mixed streams. mHC is the constrained member of that family, not a separate mechanism.
- **mHC vs. adding capacity:** mHC is about *stabilizing* signal flow through deep stacks, not adding parameters or FLOPs.

## Where It Appears

- [mHC: Manifold-Constrained Hyper-Connections](../training/mhc/index.md) — The paper that introduces mHC and analyzes HC's instability.
- [DeepSeek-V4: Million-Token Context via Hybrid Compressed Attention](../training/deepseek/deepseek-v4/index.md) — Deploys mHC as the residual mechanism in a 1.6T/284B MoE.

## Related Terms

- [Sequence Parallelism](sequence-parallelism.md) — A different training-side mechanism for scaling, orthogonal to residual topology.
- [Microbatch](microbatch.md) — The pipeline-parallel unit that interacts with DualPipe scheduling, which mHC's infrastructure extends.
