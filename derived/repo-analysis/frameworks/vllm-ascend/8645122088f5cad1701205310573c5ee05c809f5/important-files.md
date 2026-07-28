---
kind: repository-analysis
repository_id: github:vllm-project/vllm-ascend@8645122088f5cad1701205310573c5ee05c809f5
commit: 8645122088f5cad1701205310573c5ee05c809f5
source_record: raw/frameworks/vllm-ascend-codebase--github-8645122088f5.md
generated: 2026-07-28
---

# vLLM Ascend Triton Reading Notes

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
