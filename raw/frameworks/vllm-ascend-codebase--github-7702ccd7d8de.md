---
kind: repository-source
provider: github
clone_url: https://github.com/vllm-project/vllm-ascend
repository_url: https://github.com/vllm-project/vllm-ascend
local_checkout: external-repos/vllm-ascend-7702ccd7d8de/
commit: 7702ccd7d8dea6b4dabdacb0118adb522dedbec7
ref: detached
inspected: 2026-08-27
checkout_state: clean
---

# vLLM Ascend Codebase (Qwen3.8-27B FP8 to MXFP8 on Ascend 950) Source Record

## Reading Scope

- Native block-wise FP8 checkpoint loading for Qwen/Qwen3.8-27B-FP8 and the Ascend 950-only conversion to MXFP8 weight layout

## Important Entry Files

- `vllm_ascend/quantization/fp8_config.py` — Registers native block-wise FP8 quantization and selects the 950 MXFP8 wrapper
- `vllm_ascend/quantization/methods/fp8_block.py` — Loads block-wise FP8 weights, reconstructs them with checkpoint scales, requantizes them to MXFP8 on 950, and dispatches the wrapper
- `vllm_ascend/quantization/methods/w8a8_mxfp8.py` — Defines MXFP8 weight buffers, E8M0 scales, layout transformation, and Ascend MXFP8 operators
- `vllm_ascend/quantization/utils.py` — Defines MXFP8 group-scale calculation and scale-dtype helpers
- `vllm_ascend/quantization/modelopt_mxfp8_config.py` — Registers already-MXFP8 ModelOpt checkpoints and selects the same Ascend MXFP8 methods

## Limitations

- Static code reading only; no Ascend 950 runtime, numerical equivalence, or Qwen/Qwen3.8-27B-FP8 checkpoint load was executed.
