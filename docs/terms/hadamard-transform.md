---
title: "Hadamard Transform"
summary: "An orthogonal matrix with ±1 entries that redistributes a vector's outlier mass across all coordinates, making quantized LLM tensors easier to compress."
tooltip: "A Hadamard transform is a norm-preserving rotation built from ±1 entries. Applied to a weight or activation tensor before quantization, it spreads large outlier values across all channels, so a uniform low-bit grid no longer needs to span the outlier range. QuaRot and QuIP# use it to enable accurate 4-bit LLM quantization; its fast Walsh-Hadamard form runs in O(d log d)."
layout: default
confidence: high
category: algorithms
sources:
  - raw/hardware/quarot-outlier-free-4bit-inference-rotated-llms--arxiv-2404.00456v2.pdf
aliases:
  - Walsh-Hadamard Transform
  - Hadamard rotation
  - randomized Hadamard transform
  - Hadamard incoherence processing
mention_aliases:
  - Hadamard rotation
  - randomized Hadamard transform
mention_lint: canonical
appears_in:
  - docs/hardware/quantization/quarot/index.md
  - docs/hardware/quantization/flatquant/index.md
  - docs/hardware/quantization/nvfp4.md
  - docs/algorithms/kronecker-product.md
updated: 2026-08-21
---

# Hadamard Transform

**Hadamard Transform** is an orthogonal matrix multiply built from entries of $\pm 1$ that rotates a vector so its magnitude is spread evenly across all coordinates, removing coordinate-wise outliers.

## Why It Exists

Quantized tensors fail when a few channels carry values much larger than the rest: the quantization scale must cover the outliers, so ordinary values collapse onto a few usable grid points. A Hadamard rotation changes the coordinate basis without changing the underlying information, so a tensor that is outlier-heavy in the standard basis becomes uniform in the rotated basis — and therefore easy to quantize.

## How It Works

The $2 \times 2$ Walsh-Hadamard matrix is

$$
H_2 = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix},
$$

and larger sizes recurse with a [Kronecker product](kronecker-product.md): $H_{2^n} = H_2 \otimes H_{2^{n-1}}$. Because $H$ is orthogonal ($HH^\top = I$), multiplying activations by $H$ preserves norms and does not change a model's output once the adjacent weight matrices are transformed to cancel it. A randomized Hadamard matrix $\tilde H = H \operatorname{diag}(s)$ with random signs $s \in \{\pm1\}^d$ is also orthogonal and is what QuaRot applies in practice. The matrix-vector product costs $O(d \log d)$ via the fast Walsh-Hadamard transform.

## Tradeoffs

- Hadamard matrices of size $d$ exist only for certain $d$; when $d \neq 2^n$, QuaRot factors $d = 2^n m$ and builds $H_d = H_{2^n} \otimes H_m$ from a known $m$-sized Hadamard matrix, costing $O(d(m+n))$.
- The rotation is fixed and data-independent. Later methods such as SpinQuant and FlatQuant learn the rotation or affine transform per layer to do better than a fixed Hadamard.
- Applying the transform online adds a small runtime overhead (QuaRot reports at most 7%).

## Common Confusions

- **Hadamard vs. random orthogonal rotation:** Any orthogonal matrix spreads outliers, but Hadamard matrices use only $\pm 1$ entries and admit the fast $O(d \log d)$ transform, so they are much cheaper at runtime than a general dense rotation.
- **Hadamard vs. per-channel scaling:** Scaling (SmoothQuant) reshapes the distribution between weights and activations but can steepen the weight envelope; rotation redistributes outlier mass without changing either envelope.

## Where It Appears

- [QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs](../hardware/quantization/quarot/index.md) — Source-defining use: fused Hadamard rotations make weights, activations, and KV cache all quantizable to 4 bits.
- [FlatQuant: Fast Learnable Affine Quantization](../hardware/quantization/flatquant/index.md) — Replaces the fixed Hadamard rotation with a per-layer learned affine transform, citing QuaRot as the baseline.
- [NVFP4: Blackwell 4-Bit Floating Point](../hardware/quantization/nvfp4.md) — Uses a Random Hadamard Transform to smooth outliers before FP4 block quantization.
- [Kronecker Product](../algorithms/kronecker-product.md) — Builds large Hadamard matrices from small ones via Kronecker products.

## Related Terms

- [Kronecker Product](kronecker-product.md) — The construction primitive for large Hadamard matrices.
- [Post-Training Quantization](post-training-quantization.md) — The workflow in which Hadamard transforms remove outliers.
