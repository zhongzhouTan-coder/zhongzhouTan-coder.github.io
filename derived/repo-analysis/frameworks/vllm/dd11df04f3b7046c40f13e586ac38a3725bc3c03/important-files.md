---
kind: repository-analysis
repository_id: github:vllm-project/vllm@dd11df04f3b7046c40f13e586ac38a3725bc3c03
commit: dd11df04f3b7046c40f13e586ac38a3725bc3c03
source_record: raw/frameworks/vllm-codebase--github-dd11df04f3b7.md
generated: 2026-08-03
---

# vLLM Codebase Important Files

## Evidence Map

- `vllm/v1/core/block_pool.py` — Physical KV block pool: fixed-size block allocation, free-list, block refcounts
- `vllm/v1/core/kv_cache_manager.py` — KV cache manager: block allocator, logical-to-physical mapping, copy-on-write, prefix caching
- `vllm/v1/core/single_type_kv_cache_manager.py` — Default single-type KV cache manager: hash-based prefix caching, eviction, block table construction
- `vllm/v1/core/kv_cache_coordinator.py` — Cross-GPU block coordination: logical-to-physical global mapping, swap in/out

## Reproduction Commands

Record exact read-only search or counting commands here when the docs make
quantitative codebase claims.
