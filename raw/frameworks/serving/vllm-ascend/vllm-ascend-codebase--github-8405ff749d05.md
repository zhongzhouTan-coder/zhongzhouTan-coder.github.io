---
kind: repository-source
provider: github
clone_url: https://github.com/vllm-project/vllm-ascend
repository_url: https://github.com/vllm-project/vllm-ascend
local_checkout: external-repos/vllm-ascend-8405ff749d05/
commit: 8405ff749d05aee320923d7d1dd9ad24c9246d4c
ref: detached
inspected: 2026-10-27
checkout_state: clean
---

# vLLM Ascend Codebase (DeepSeek-V4 C4/C128 main-KV compression) Source Record

## Reading Scope

- DeepSeek-V4 C4/C128 compression of main KV and the separate C4 indexer selection path

## Important Entry Files

- `vllm_ascend/models/deepseek_v4/model.py` — Creates a compressor for all compressed ratios and an indexer only for ratio 4
- `vllm_ascend/models/deepseek_v4/compressor.py` — Runs the model main-KV compressor with cmp_ratio 4 or 128 and returns compressed KV
- `vllm_ascend/attention/dsa_v1.py` — Scatters compressed main KV for both ratios and only passes sparse indices on C4 attention reads
- `vllm_ascend/attention/context_parallel/dsa_cp.py` — Context-parallel main-KV compressed writes for both ratios and C4-only top-k sparse indices

## Limitations

- Static code reading only; runtime behavior was not executed.
