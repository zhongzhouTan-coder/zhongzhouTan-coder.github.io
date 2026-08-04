---
kind: repository-analysis
repository_id: github:vllm-project/vllm-ascend@8645122088f5cad1701205310573c5ee05c809f5
commit: 8645122088f5cad1701205310573c5ee05c809f5
source_record: raw/frameworks/vllm-ascend-codebase--github-8645122088f5.md
generated: 2026-07-28
---

# vLLM Ascend Triton Reading Notes

## Required Code Evidence

| Docs page | Finding | File | Symbol | Start | End |
|---|---|---|---|---:|---:|
| `docs/frameworks/triton/triton-in-vllm.md` | ascend-triton-wrapper | `vllm_ascend/ops/triton/triton_utils.py` | `_resolve_triton_ascend_op` | 17 | — |
| `docs/frameworks/triton/triton-in-vllm.md` | ascend-fused-qkv | `vllm_ascend/ops/triton/linearnorm/split_qkv_rmsnorm_rope.py` | `split_qkv_rmsnorm_rope_kernel` | 26 | — |
| `docs/frameworks/triton/triton-in-vllm.md` | ascend-kda | `vllm_ascend/ops/triton/kda/kda.py` | `fused_recurrent_kda_fwd` | 36 | — |
| `docs/frameworks/triton/triton-in-vllm.md` | ascend-rmsnorm | `vllm_ascend/ops/triton/rms_norm.py` | `triton_rms_kernel` | 6 | — |
| `docs/frameworks/triton/triton-in-vllm.md` | ascend-rope | `vllm_ascend/ops/triton/rope.py` | `_triton_rope` | 24 | — |
| `docs/frameworks/triton/triton-in-vllm.md` | ascend-mul-add | `vllm_ascend/ops/triton/mul_add.py` | `muls_add_kernel` | 8 | — |

## Runtime Flow Evidence

1. Ascend Triton infrastructure — `ascend-triton-wrapper`.
2. Fused QKV, normalization, and rotary — `ascend-fused-qkv`, `ascend-rmsnorm`, `ascend-rope`.
3. Linear-attention and element-wise kernels — `ascend-kda`, `ascend-mul-add`.

## Evidence Map

- `vllm_ascend/ops/triton/triton_utils.py` imports Triton through vLLM,
  resolves `insert_slice`, `extract_slice`, and `get_element` from the CANN
  extension when available, and queries AI-core and vector-core counts.
- `vllm_ascend/ops/triton/linearnorm/` contains fused QKV, normalization, and
  rotary-embedding implementations.
- `vllm_ascend/ops/triton/fla/` and `vllm_ascend/ops/triton/kda/` contain
  multi-kernel implementations for linear and kernelized dynamic attention.
- `vllm_ascend/ops/triton/activation/`, `mamba/`, and the top-level Triton
  modules contain activation, state-space, normalization, sampling, and
  utility kernels.
- `csrc/` contains the separate AscendC implementation surface for operations
  that are not implemented through the Triton path.

## Scope Note

The repository uses several launch patterns rather than one universal rule.
The page highlights the vector-core-count grid-stride pattern where the
inspected kernels use it, without treating it as a property of every Ascend
Triton kernel.
