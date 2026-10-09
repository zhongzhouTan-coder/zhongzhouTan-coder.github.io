---
kind: repository-source
provider: github
clone_url: https://github.com/sgl-project/sglang.git
repository_url: https://github.com/sgl-project/sglang
local_checkout: external-repos/sglang/
commit: 08fc290f48a865f2cd0b9cf7a90f5a1c44dc0bd1
ref: main
inspected: 2026-10-09
checkout_state: clean
---

# SGLang HiCache and Mooncake Integration Source Record

## Reading Scope

- HiCache L3 request prefetch, prefix lookup, asynchronous transfer, radix insertion, and L2 load-back.
- Mooncake storage adapter configuration, host-buffer registration, and batch get/set contracts.
- Upstream deployment documentation and registered integration-test boundaries.

## Important Entry Files

- `python/sglang/srt/managers/scheduler.py` — Selects the unmatched request suffix for HiCache storage prefetch and polls completion before admission.
- `python/sglang/srt/mem_cache/unified_radix_cache.py` — Builds page-aligned prefetch keys, reconciles storage results into the radix tree, and loads host hits back to device.
- `python/sglang/srt/mem_cache/hybrid_cache/hybrid_cache_controller.py` — Attaches the storage backend, queries L3 prefix availability, and coordinates hybrid-pool transfers and write-back.
- `python/sglang/srt/managers/cache_controller.py` — Runs asynchronous prefetch/backup queues, storage page transfers, and rank-synchronized acknowledgements.
- `python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py` — Configures the Mooncake client, registers host buffers, and dispatches page existence/read/write operations.
- `python/sglang/srt/mem_cache/storage/mooncake_store/README.md` — Documents Mooncake service roles, deployment options, and SGLang configuration.
- `test/registered/hicache/test_hicache_storage_mooncake_backend.py` — Shows the registered backend integration-test setup and its CUDA/Mooncake service requirements.

## Limitations

- Static code reading only; no SGLang server, GPU, Mooncake service, RDMA/TCP transfer, or model request was executed.
- Mooncake client/server internals are outside this SGLang snapshot; transport claims are limited to SGLang adapter calls and its checked-in deployment documentation.
