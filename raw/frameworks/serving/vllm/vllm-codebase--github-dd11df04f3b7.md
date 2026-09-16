---
kind: repository-source
provider: github
clone_url: https://github.com/vllm-project/vllm
repository_url: https://github.com/vllm-project/vllm
local_checkout: external-repos/vllm-dd11df04f3b7/
commit: dd11df04f3b7046c40f13e586ac38a3725bc3c03
ref: detached
inspected: 2026-08-03
checkout_state: clean
---

# vLLM Codebase Source Record

## Reading Scope

- vLLM V1 block table / KV cache management mechanism

## Important Entry Files

- `vllm/v1/core/block_pool.py` — Physical KV block pool: fixed-size block allocation, free-list, block refcounts
- `vllm/v1/core/kv_cache_manager.py` — KV cache manager: block allocator, logical-to-physical mapping, copy-on-write, prefix caching
- `vllm/v1/core/single_type_kv_cache_manager.py` — Default single-type KV cache manager: hash-based prefix caching, eviction, block table construction
- `vllm/v1/core/kv_cache_coordinator.py` — Cross-GPU block coordination: logical-to-physical global mapping, swap in/out

## Limitations

- Static code reading only; runtime behavior was not executed.
