---
title: "Kronecker Product"
summary: "The block-structured matrix product A⊗B (matrix direct product) that builds a large matrix from two smaller ones by scaling copies of B by the entries of A."
tooltip: "The Kronecker product A⊗B replaces every entry a_ij of A with the block a_ij·B, producing an mp×nq matrix from an m×n and a p×q matrix. It is the matrix of the tensor product of two linear maps, and its vectorization identity vec(A X B) = (Bᵀ⊗A) vec(X) is the trick FlatQuant uses to run a learned affine transform as two small matrix multiplications."
layout: default
confidence: high
category: algorithms
sources:
  - raw/algorithms/kronecker-product--web-2026-08-04-41ef5673a67a.html
  - raw/algorithms/kronecker-product--web-2026-08-04-41ef5673a67a.metadata.json
  - derived/web-markdown/algorithms/kronecker-product--web-2026-08-04-41ef5673a67a.md
aliases:
  - matrix direct product
  - tensor product of matrices
appears_in:
  - docs/algorithms/kronecker-product.md
  - docs/hardware/quantization/flatquant/index.md
  - docs/frameworks/vllm/minimax-gqa-w4a4-quantization-path.md
updated: 2026-08-06
---

# Kronecker Product

**Kronecker Product** is the block-structured product `A ⊗ B` of two matrices that forms a larger matrix by scaling copies of `B` by the entries of `A`.

## Why It Exists

Sometimes you need one big linear operator that behaves like two small ones acting together — the tensor product of two maps. Writing out that big operator naively costs `(mn)²` storage for `m×n` times `p×q`. The Kronecker product gives the big matrix *and* a way to compute with it through the small factors, which is what makes structured transforms (like FlatQuant's per-layer affine maps) affordable.

## How It Works

If `A` is `m×n` and `B` is `p×q`, then `A ⊗ B` is the `mp×nq` matrix with elements

$$(A \otimes B)_{(i-1)p + k,\; (j-1)q + l} = a_{ij}\, b_{kl}$$

Equivalently, replace every entry `a_ij` of `A` with the scaled block `a_ij·B` (block form `[[a11·B, a12·B, …], [a21·B, …]]`). The key computational identity is

$$\operatorname{vec}(A X B) = (B^{\top} \otimes A)\,\operatorname{vec}(X)$$

which lets a full matrix transformation on `X` be re-expressed as a Kronecker product of two smaller matrices. For example, FlatQuant constrains each learned transform to `P = P1 ⊗ P2`, so `P` acts on a reshaped tensor as `P1ᵀ X̃ P2` — two small matmuls instead of one large one.

## Tradeoffs

- **Parameter savings come from structure.** `P1 ⊗ P2` uses `n1² + n2²` parameters versus `n²` for a full `n×n` matrix — up to `n/2` less storage at `n1 = n2 = √n` — but only spans the Kronecker-structured subset of all matrices.
- **Factor choice affects speed, not just size.** Speedup peaks when `n1` and `n2` are balanced; once one factor grows past `√n`, irregular memory access degrades performance.
- **Invertibility transfers.** `(A ⊗ B)⁻¹ = A⁻¹ ⊗ B⁻¹` when both factors are invertible, which is what keeps FlatQuant's offline inverse practical.

## Common Confusions

- **Kronecker product vs. outer product:** The [outer product](outer-product.md) of two vectors is the rank-1 matrix `u vᵀ`; the Kronecker product is the block-structured product of two full matrices. The outer product is the Kronecker product's rank-1 special case.
- **Kronecker product vs. regular matrix product:** The ordinary product `AB` reuses dimensions and needs `A.columns = B.rows`; the Kronecker product `A ⊗ B` accepts any two matrices and never contracts a shared dimension.

## Where It Appears

- [Kronecker Product reference](../algorithms/kronecker-product.md) — Hosts the MathWorld source and the full definition, identities, and FlatQuant tie-in.
- [FlatQuant: Fast Learnable Affine Quantization](../hardware/quantization/flatquant/index.md) — Factorizes each layer's learned affine transform `P` as `P1 ⊗ P2` so it runs online as two small matrix multiplications (the Kronecker factorization deep dive).
- [MiniMax GQA W4A4 Quantization Path](../frameworks/vllm/minimax-gqa-w4a4-quantization-path.md) — The Ascend NPU stores FlatQuant's Kronecker factors as `left_trans`/`right_trans` and applies them via the `npu_kronecker_quant` activation quantizer.

## Related Terms

- [Outer Product](outer-product.md) — The rank-1 special case of a Kronecker product.
- [General Matrix Multiply (GEMM)](gemm.md) — The kernel that executes the two small Kronecker-factor matmuls.
- [Inner Product](inner-product.md) — The element-wise dual of the outer-product view.
- [Matrix Tiling](matrix-tiling.md) — How the small-factor matmuls are blocked onto on-chip memory.
