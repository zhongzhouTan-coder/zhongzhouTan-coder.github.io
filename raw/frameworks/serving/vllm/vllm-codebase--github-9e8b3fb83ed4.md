---
kind: repository-source
provider: github
clone_url: https://github.com/vllm-project/vllm
repository_url: https://github.com/vllm-project/vllm
local_checkout: external-repos/vllm-9e8b3fb83ed4/
commit: 9e8b3fb83ed4f3da1ed5376083027b14994939fc
ref: detached
inspected: 2026-11-25
checkout_state: clean
---

# vLLM HiSparse Runtime Source Record

## Reading Scope

- Upstream vLLM HiSparse runtime and sparse MLA index-group interfaces consumed by vLLM-Ascend

## Important Entry Files

- `vllm/v1/hisparse/runtime.py` — Common HiSparse resident, host, and hot cache runtime
- `vllm/v1/attention/backends/mla/index_group.py` — HiSparse index groups own per-layer cache handles and swap-in path
- `vllm/v1/hisparse/layout.py` — HiSparse cache layout and host/source/resident/hot allocation contract

## Limitations

- Static code reading only. The upstream HiSparse host path was not run on GPU.
