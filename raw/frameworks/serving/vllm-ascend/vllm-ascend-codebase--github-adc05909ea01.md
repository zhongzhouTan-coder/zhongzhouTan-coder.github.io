---
kind: repository-source
provider: github
clone_url: https://github.com/shaopeng-666/vllm-ascend.git
repository_url: https://github.com/shaopeng-666/vllm-ascend
local_checkout: external-repos/vllm-ascend-qwen38-pr/
commit: adc05909ea0125d058c2c9eb3131e761929f2551
ref: feat/qwen38-flash-next-main-v1
inspected: 2026-10-26
checkout_state: clean
---

# vllm-ascend Codebase Source Record

## Reading Scope

- Qwen3.8 Flash Next A3 model wrapper, QSA indexer and GBSA, PLE, six-region cache layout, runner, MTP, and ModelSlim W8A8

## Important Entry Files

- `vllm_ascend/models/qwen4_exp/model.py` — Ascend model wrapper and upstream substitutions
- `vllm_ascend/models/qwen4_exp/qsa.py` — QSA write and sparse read path
- `vllm_ascend/core/six_region_kv_cache_layout.py` — Physical cache layout

## Limitations

- Static code reading only; runtime behavior was not executed.
