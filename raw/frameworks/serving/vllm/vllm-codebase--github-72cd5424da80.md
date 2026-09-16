---
kind: repository-source
provider: github
clone_url: https://github.com/vllm-project/vllm
repository_url: https://github.com/vllm-project/vllm
local_checkout: external-repos/vllm-72cd5424da80/
commit: 72cd5424da80a4a9caa3f42fd65bc0b94e61cbf0
ref: detached
inspected: 2026-08-04
checkout_state: clean
---

# vllm Codebase Source Record

## Reading Scope

- MiniMax M2 GQA attention model and W4A4 quantization path (MXFP4/NVFP4/ModelOpt) for GPU serving

## Important Entry Files

- `vllm/model_executor/models/minimax_m2.py` — MiniMax M2 model with GQA attention layers
- `vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_mxfp4.py` — CompressedTensors W4A4 MXFP4 linear scheme
- `vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py` — CompressedTensors W4A4 NVFP4 linear scheme
- `vllm/model_executor/layers/quantization/mxfp4.py` — MXFP4 W4A4 config and MoE method
- `vllm/model_executor/layers/quantization/modelopt.py` — ModelOpt NVFP4 W4A4 and MXFP8 (MiniMax-style) configs

## Limitations

- Static code reading only; runtime behavior was not executed.
