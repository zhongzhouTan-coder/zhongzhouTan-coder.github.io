---
title: "Kronecker Product"
summary: "The block-structured matrix product A⊗B (matrix direct product), foundational to tensor factorization and the Kronecker factorization trick behind FlatQuant's learnable affine transforms."
layout: default
confidence: high
sources:
  - raw/algorithms/kronecker-product--web-2026-08-04-41ef5673a67a.html
  - raw/algorithms/kronecker-product--web-2026-08-04-41ef5673a67a.metadata.json
  - derived/web-markdown/algorithms/kronecker-product--web-2026-08-04-41ef5673a67a.md
updated: 2026-08-04
---

# Kronecker Product

**Source:** [Kronecker Product — from Wolfram MathWorld](https://mathworld.wolfram.com/KroneckerProduct.html)

**Related pages:** [Kronecker Product (term)](../terms/kronecker-product.md) · [FlatQuant: Fast Learnable Affine Quantization](../hardware/quantization/flatquant/index.md) · [Quantization hub](../hardware/quantization/index.md)

## TL;DR

**What:** The Kronecker product (a.k.a. matrix direct product) `A ⊗ B` builds one large block-structured matrix from two smaller matrices.
**How:** Every entry `a_ij` of `A` is replaced by the scaled block `a_ij · B`; if `A` is `m×n` and `B` is `p×q`, the result is an `mp×nq` matrix.
**Why it matters here:** The vectorization identity `vec(A X B) = (Bᵀ ⊗ A) vec(X)` lets a full matrix transformation be re-expressed as a Kronecker product of two small factors — the trick FlatQuant uses to make learned per-layer affine transforms affordable at inference.

## Definition

Given an `m×n` matrix `A` and a `p×q` matrix `B`, their **Kronecker product** `A ⊗ B` is the `mp×nq` matrix whose elements are defined by

$$(A \otimes B)_{(i-1)p + k,\; (j-1)q + l} = a_{ij}\, b_{kl}$$

In block form, each entry of `A` is replaced by a scaled copy of the whole matrix `B`:

$$A \otimes B = \begin{bmatrix} a_{11}B & a_{12}B & \cdots & a_{1n}B \\ a_{21}B & a_{22}B & \cdots & a_{2n}B \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1}B & a_{m2}B & \cdots & a_{mn}B \end{bmatrix}$$

For example, the direct product of a `2×2` matrix `A` with a `2×3` matrix `B` yields the `4×6` block matrix `[[a11·B, a12·B], [a21·B, a22·B]]` expanded entry by entry. This is the definition and block form given in the [MathWorld source](https://mathworld.wolfram.com/KroneckerProduct.html), which also notes the operation is implemented in the [Wolfram Language](https://reference.wolfram.com/language/ref/KroneckerProduct.html) as `KroneckerProduct[a, b]`.

## Tensor-Product Interpretation

The matrix direct product gives the matrix of the linear transformation induced by the [tensor product](https://mathworld.wolfram.com/VectorSpaceTensorProduct.html) of the original vector spaces. If `A` and `B` describe linear maps on two spaces, then `A ⊗ B` describes their tensor product acting on the product space — which is why the Kronecker product is the standard way to turn a "small" pair of operators into one "big" operator.

## Why It Matters for LLM Quantization

FlatQuant ([insight page](../hardware/quantization/flatquant/index.md)) needs a per-layer invertible transform `P ∈ ℝ^{n×n}` that flattens weights and activations before 4-bit quantization. A full `P` doubles matmul cost, memory traffic, and storage. Instead, FlatQuant constrains `P = P1 ⊗ P2` with `P1 ∈ ℝ^{n1×n1}`, `P2 ∈ ℝ^{n2×n2}`, `n = n1·n2`, and uses the vectorization identity

$$\operatorname{vec}(V)\,(P_1 \otimes P_2) = \operatorname{vec}(P_1^{\top} V P_2)$$

to apply the transform as two small matrix multiplications on a reshaped tensor — `P1ᵀ ×1 X̃ ×2 P2` — instead of one large one. Storage drops by up to `n/2` and compute by up to `√n/2` when `n1 = n2 = √n`. For hidden size 8192, the optimal factors are `(64, 128)`; on LLaMA-2-7B all online transforms together cost only ~2.61% of the FP16 model's FLOPs and ~3.41 MB of parameters. This identity is exactly the mechanism documented in the [FlatQuant paper page](../hardware/quantization/flatquant/index.md)'s Kronecker factorization deep dive.

## Implementation Notes

- Wolfram Language: `KroneckerProduct[a, b]` ([reference](https://reference.wolfram.com/language/ref/KroneckerProduct.html)).
- NumPy/Torch: `numpy.kron` / `torch.kron`.
- In vLLM-Ascend's FlatQuant W4A4 path, the two factors are stored as `left_trans`/`right_trans` and applied by the `npu_kronecker_quant` activation quantizer ([MiniMax GQA W4A4 path](../frameworks/vllm/minimax-gqa-w4a4-quantization-path.md)).

## Go Deeper

- **Read:** [Kronecker Product — Wolfram MathWorld](https://mathworld.wolfram.com/KroneckerProduct.html)
- **Understand the term:** [Kronecker Product (term page)](../terms/kronecker-product.md)
- **See it in action:** [FlatQuant: Fast Learnable Affine Quantization](../hardware/quantization/flatquant/index.md), [MiniMax GQA W4A4 Quantization Path](../frameworks/vllm/minimax-gqa-w4a4-quantization-path.md)
