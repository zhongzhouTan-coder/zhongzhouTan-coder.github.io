---
kind: repository-source
repository_url: https://github.com/vllm-project/vllm-ascend
local_checkout: external-repos/vllm-ascend/
commit: 8645122088f5cad1701205310573c5ee05c809f5
ref: main
inspected: 2026-07-28
checkout_state: clean
---

# vLLM Ascend Codebase Source Record

## Reading Scope

The inspection focused on the Ascend plugin's Triton and AscendC kernel
surfaces:

- `vllm_ascend/ops/triton/`
- `vllm_ascend/ops/triton/triton_utils.py`
- `vllm_ascend/ops/triton/linearnorm/`
- `vllm_ascend/ops/triton/fla/`
- `vllm_ascend/ops/triton/kda/`
- `csrc/`

## Important Entry Files

- `README.md` — plugin purpose, supported hardware, and version policy.
- `vllm_ascend/ops/triton/triton_utils.py` — CANN extension resolution and device-property queries.
- `vllm_ascend/ops/triton/linearnorm/split_qkv_rmsnorm_rope.py` — fused linearnorm kernel.
- `vllm_ascend/ops/triton/activation/swiglu_quant.py` — element-wise fusion and quantization example.

## Limitations

- Static code reading only; kernels were not built or run on Ascend hardware.
- Performance claims are not independently benchmark-validated.
