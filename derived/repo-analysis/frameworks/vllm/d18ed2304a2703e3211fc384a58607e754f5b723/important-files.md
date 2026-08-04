---
kind: repository-analysis
repository_id: github:vllm-project/vllm@d18ed2304a2703e3211fc384a58607e754f5b723
commit: d18ed2304a2703e3211fc384a58607e754f5b723
source_record: raw/frameworks/vllm-codebase--github-d18ed2304a27.md
generated: 2026-07-28
---

# vLLM Triton Reading Notes

## Required Code Evidence

| Docs page | Finding | File | Symbol | Start | End |
|---|---|---|---|---:|---:|
| `docs/frameworks/triton/triton-in-vllm.md` | triton-import | `vllm/triton_utils/importing.py` | `TritonPlaceholder` | 94 | — |
| `docs/frameworks/triton/triton-in-vllm.md` | clone-elimination | `vllm/compilation/passes/ir/clone_elimination.py` | `clone_preserves_layout` | 19 | — |
| `docs/frameworks/triton/triton-in-vllm.md` | decode-attention | `vllm/v1/attention/ops/triton_decode_attention.py` | `_fwd_kernel_stage1` | 54 | — |
| `docs/frameworks/triton/triton-in-vllm.md` | attention-helpers | `vllm/v1/attention/ops/triton_attention_helpers.py` | `find_seq_idx` | 22 | — |
| `docs/frameworks/triton/triton-in-vllm.md` | moe-dispatcher | `vllm/model_executor/layers/fused_moe/fused_moe.py` | `dispatch_fused_moe_kernel` | 42 | — |
| `docs/frameworks/triton/triton-in-vllm.md` | moe-align | `vllm/model_executor/layers/fused_moe/moe_align_block_size.py` | `moe_align_block_size` | 11 | — |
| `docs/frameworks/triton/triton-in-vllm.md` | awq-triton | `vllm/model_executor/layers/quantization/awq_triton.py` | `awq_dequantize_kernel` | 12 | — |
| `docs/frameworks/triton/triton-in-vllm.md` | fp8-utils | `vllm/model_executor/layers/quantization/utils/fp8_utils.py` | `is_fp8` | 42 | — |
| `docs/frameworks/triton/triton-in-vllm.md` | activation | `vllm/model_executor/layers/activation.py` | `swiglustep_and_mul_triton` | 27 | — |
| `docs/frameworks/triton/triton-in-vllm.md` | sampling | `vllm/v1/sample/ops/topk_topp_triton.py` | `apply_top_k_top_p_triton` | 71 | — |
| `docs/frameworks/triton/triton-in-vllm.md` | kv-offload | `vllm/v1/kv_offload/cpu/swap_blocks_triton.py` | `_swap_blocks_kernel` | 25 | — |

## Runtime Flow Evidence

1. Kernel availability and graph boundaries — `triton-import`, `clone-elimination`.
2. Attention kernels — `decode-attention`, `attention-helpers`.
3. MoE kernels — `moe-dispatcher`, `moe-align`.
4. Quantization and fused element-wise ops — `awq-triton`, `fp8-utils`, `activation`, `sampling`, `kv-offload`.

## Evidence Map

- `vllm/triton_utils/importing.py` detects usable Triton backends and supplies
  placeholder modules when Triton is unavailable.
- `vllm/triton_utils/force_first_config.py` implements the
  `VLLM_TRITON_FORCE_FIRST_CONFIG` behavior.
- `vllm/utils/torch_utils.py` defines `direct_register_custom_op`.
- `vllm/compilation/passes/ir/clone_elimination.py` checks
  `TritonKernelWrapperFunctional` users.
- `vllm/model_executor/layers/mamba/ops/mamba_ssm.py` loads optional
  device-and-shape-specific JSON launch configurations and otherwise uses a
  fallback configuration.
- `vllm/model_executor/layers/quantization/awq_triton.py` uses two-dimensional
  program IDs, masked two-dimensional tiles, packed `int32` weights and zeros,
  a runtime `group_size`, three repeated `tl.interleave` operations, and the
  explicit AWQ shift order `[0, 4, 1, 5, 2, 6, 3, 7]`.

## Decorator Inventory

At the pinned commit, repository searches under `vllm/` found:

- 163 Python files containing `@triton.jit`.
- 16 of those files also containing `@triton.autotune`.
- 408 `@triton.jit` decorator occurrences.
- 28 `@triton.autotune` decorator occurrences.

These counts show that explicit `@triton.autotune` use is a minority pattern,
not the default for most Triton kernel files.

## Reproduction Commands

```bash
rg -l '@triton\.jit' external-repos/vllm/vllm -g '*.py' | wc -l
rg -l '@triton\.autotune' external-repos/vllm/vllm -g '*.py' | wc -l
rg -o '@triton\.jit' external-repos/vllm/vllm -g '*.py' | wc -l
rg -o '@triton\.autotune' external-repos/vllm/vllm -g '*.py' | wc -l
```
