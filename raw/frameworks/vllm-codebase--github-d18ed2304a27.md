---
kind: repository-source
repository_url: https://github.com/vllm-project/vllm
local_checkout: external-repos/vllm/
commit: d18ed2304a2703e3211fc384a58607e754f5b723
ref: main
inspected: 2026-07-28
checkout_state: clean
---

# vLLM Codebase Source Record

## Reading Scope

The inspection focused on vLLM's Triton integration and kernel implementations:

- `vllm/triton_utils/`
- `vllm/model_executor/layers/quantization/awq_triton.py`
- `vllm/model_executor/layers/mamba/ops/mamba_ssm.py`
- `vllm/model_executor/layers/`
- `vllm/v1/attention/ops/`
- `vllm/compilation/passes/ir/clone_elimination.py`
- `vllm/utils/torch_utils.py`

Repository-wide searches counted Python files and decorators containing
`@triton.jit` and `@triton.autotune`.

## Important Entry Files

- `README.md` — project scope and supported execution features.
- `vllm/triton_utils/importing.py` — Triton availability and placeholder imports.
- `vllm/triton_utils/force_first_config.py` — first-configuration override.
- `vllm/model_executor/layers/quantization/awq_triton.py` — AWQ dequantization and GEMM kernels.
- `vllm/model_executor/layers/mamba/ops/mamba_ssm.py` — selective-state-update configuration loading.

## Limitations

- Static code reading only; kernels were not benchmarked or executed on GPU
  hardware.
- Performance descriptions are limited to implementation structure and
  repository-provided configuration behavior.
