---
kind: repository-source
provider: github
clone_url: https://github.com/vllm-project/vllm-ascend
repository_url: https://github.com/vllm-project/vllm-ascend
local_checkout: external-repos/vllm-ascend-61221e9add8c/
commit: 61221e9add8c717b304005bd9d48d6215d035be7
ref: detached
inspected: 2026-08-04
checkout_state: clean
---

# vllm-ascend Codebase Source Record

## Reading Scope

- MiniMax M2 GQA model W4A4 quantization path (ModelSlim W4A4 FLATQUANT/LAOS/MXFP4) for Ascend NPU serving

## Important Entry Files

- `vllm_ascend/quantization/modelslim_config.py` — ModelSlim quant config parsing and packed module mapping
- `vllm_ascend/quantization/methods/w4a4_mxfp4.py` — W4A4 MXFP4 linear and MoE methods
- `vllm_ascend/quantization/methods/w4a4_flatquant.py` — W4A4 FLATQUANT dynamic linear method
- `vllm_ascend/quantization/methods/w4a4_laos_dynamic.py` — W4A4 LAOS dynamic method
- `vllm_ascend/quantization/methods/registry.py` — Scheme registry for ModelSlim quantization methods
- `vllm_ascend/patch/platform/patch_minimax_m2_config.py` — MiniMax M2 fp8-to-bf16 disable patch on NPU

## Limitations

- Static code reading only; runtime behavior was not executed.
