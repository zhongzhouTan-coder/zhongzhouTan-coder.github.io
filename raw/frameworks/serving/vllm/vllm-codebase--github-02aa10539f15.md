---
kind: repository-source
provider: github
clone_url: https://github.com/vllm-project/vllm
repository_url: https://github.com/vllm-project/vllm
local_checkout: external-repos/vllm-02aa10539f15/
commit: 02aa10539f15c8c73aa7fd4a5ee0f691cbf88dcb
ref: detached
inspected: 2026-10-10
checkout_state: clean
---

# vllm Codebase Source Record

## Reading Scope

- GPU model-specific cache paths for GLM-5.3, DeepSeek-V4.1-Flash, MiniMax-M3, Qwen3.8-Flash-Next, and Kimi-K3

## Important Entry Files

- `vllm/models/deepseek_v32/nvidia/model.py` — GLM-5.3 family model path
- `vllm/models/deepseek_v41/attention.py` — DeepSeek-V4.1 cache path
- `vllm/models/minimax_m3/nvidia/model.py` — MiniMax-M3 sparse attention
- `vllm/models/qwen4_exp/nvidia/qsa.py` — Qwen sparse attention
- `vllm/models/kimi_k3/nvidia/mla.py` — Kimi-K3 MLA cache

## Limitations

- Static code reading only; runtime behavior was not executed.
