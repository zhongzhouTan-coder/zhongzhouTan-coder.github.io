---
kind: repository-source
provider: github
clone_url: https://github.com/vllm-project/vllm
repository_url: https://github.com/vllm-project/vllm
local_checkout: external-repos/vllm-ca90b9e7d4e3/
commit: ca90b9e7d4e3ec670143e4b1822bb856ab0260cc
ref: detached
inspected: 2026-08-27
checkout_state: clean
---

# vLLM Codebase (weight-loading lifecycle for Qwen3.8 FP8 to MXFP8) Source Record

## Reading Scope

- Generic vLLM model-loader lifecycle that invokes the vllm-ascend quantization method after checkpoint weights are loaded

## Important Entry Files

- `vllm/model_executor/model_loader/base_loader.py` — Runs the generic load_weights lifecycle and invokes post-load processing
- `vllm/model_executor/model_loader/default_loader.py` — Loads checkpoint tensors through AutoWeightsLoader before post-load processing
- `vllm/model_executor/model_loader/utils.py` — Walks quantized layers and calls quant_method.process_weights_after_loading
- `vllm/model_executor/model_loader/weight_utils.py` — Resolves the quantization config used to construct AscendFp8Config

## Limitations

- Static code reading only; this source is used only for the generic loader boundary, not for Ascend kernel behavior.
