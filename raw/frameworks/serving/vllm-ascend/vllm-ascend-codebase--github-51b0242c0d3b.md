---
kind: repository-source
provider: github
clone_url: https://github.com/vllm-project/vllm-ascend
repository_url: https://github.com/vllm-project/vllm-ascend
local_checkout: external-repos/vllm-ascend-51b0242c0d3b/
commit: 51b0242c0d3be1d3bd59e763a7d78b8553cd45e1
ref: detached
inspected: 2026-10-10
checkout_state: clean
---

# vllm-ascend Codebase Source Record

## Reading Scope

- Ascend model-specific cache paths for GLM-5.3, DeepSeek-V4.1-Flash, MiniMax-M3, Qwen3.8-Flash-Next, and Kimi-K3

## Important Entry Files

- `vllm_ascend/models/deepseek_mtp.py` — GLM-5.3 family registration
- `vllm_ascend/models/deepseek_v41/model.py` — DeepSeek-V4.1 cache path
- `vllm_ascend/models/minimax_m3/minimax_m3.py` — MiniMax-M3 sparse attention
- `vllm_ascend/models/kimi_k3.py` — Kimi-K3 NPU override

## Limitations

- No Qwen4Exp model override found in this pinned vLLM Ascend revision; support cannot be confirmed from this checkout.
