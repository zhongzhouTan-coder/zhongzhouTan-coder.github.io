---
kind: repository-analysis
repository_id: github:vllm-project/vllm@d18ed2304a2703e3211fc384a58607e754f5b723
commit: d18ed2304a2703e3211fc384a58607e754f5b723
source_record: raw/frameworks/vllm-codebase--github-d18ed2304a27.md
generated: 2026-07-28
---

# vLLM Triton Reading Notes

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
