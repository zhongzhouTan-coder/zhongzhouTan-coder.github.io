---
kind: repository-analysis
repository_id: github:sgl-project/sglang@08fc290f48a865f2cd0b9cf7a90f5a1c44dc0bd1
commit: 08fc290f48a865f2cd0b9cf7a90f5a1c44dc0bd1
source_record: raw/frameworks/serving/sglang/sglang-codebase--github-08fc290f48a8.md
generated: 2026-10-09
---

# SGLang HiCache and Mooncake Integration Important Files

## Evidence Map

- `python/sglang/srt/managers/scheduler.py` — Selects the unmatched request suffix for HiCache storage prefetch and polls completion before admission.
- `python/sglang/srt/mem_cache/unified_radix_cache.py` — Builds page-aligned prefetch keys, reconciles storage results into the radix tree, and loads host hits back to device.
- `python/sglang/srt/mem_cache/hybrid_cache/hybrid_cache_controller.py` — Attaches the storage backend, queries L3 prefix availability, and coordinates hybrid-pool transfers and write-back.
- `python/sglang/srt/managers/cache_controller.py` — Runs asynchronous prefetch/backup queues, storage page transfers, and rank-synchronized acknowledgements.
- `python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py` — Configures the Mooncake client, registers host buffers, and dispatches page existence/read/write operations.
- `python/sglang/srt/mem_cache/storage/mooncake_store/README.md` — Documents Mooncake service roles, deployment options, and SGLang configuration.
- `test/registered/hicache/test_hicache_storage_mooncake_backend.py` — Shows the registered backend integration-test setup and its CUDA/Mooncake service requirements.
- `docs/docs/advanced_features/hicache_design.mdx` — Defines HiCache tier scope, prefetch behavior, and write-back policies.

Consuming page: `docs/frameworks/serving/sglang/hicache-mooncake.md`

| docs page | finding | file | symbol | start | end |
|---|---|---|---|---:|---:|
| `docs/frameworks/serving/sglang/hicache-mooncake.md` | request-suffix | `python/sglang/srt/managers/scheduler.py` | `Scheduler._prefetch_kvcache` | 3139 | |
| `docs/frameworks/serving/sglang/hicache-mooncake.md` | cache-prefetch | `python/sglang/srt/mem_cache/unified_radix_cache.py` | `UnifiedRadixCache.prefetch_from_storage` | 2011 | |
| `docs/frameworks/serving/sglang/hicache-mooncake.md` | controller-submit | `python/sglang/srt/mem_cache/hybrid_cache/hybrid_cache_controller.py` | `HybridCacheController.submit_prefetch` | 1035 | |
| `docs/frameworks/serving/sglang/hicache-mooncake.md` | query-prefix | `python/sglang/srt/mem_cache/hybrid_cache/hybrid_cache_controller.py` | `HybridCacheController._storage_hit_query` | 1349 | |
| `docs/frameworks/serving/sglang/hicache-mooncake.md` | async-prefetch | `python/sglang/srt/managers/cache_controller.py` | `HiCacheController.prefetch_thread_func` | 1274 | |
| `docs/frameworks/serving/sglang/hicache-mooncake.md` | page-transfer | `python/sglang/srt/managers/cache_controller.py` | `HiCacheController._page_transfer` | 1095 | |
| `docs/frameworks/serving/sglang/hicache-mooncake.md` | ack-sync | `python/sglang/srt/managers/cache_controller.py` | `HiCacheController.prefetch_sync_thread_func` | 1394 | |
| `docs/frameworks/serving/sglang/hicache-mooncake.md` | backend-attach | `python/sglang/srt/managers/cache_controller.py` | `HiCacheController.attach_storage_backend` | 532 | |
| `docs/frameworks/serving/sglang/hicache-mooncake.md` | host-registration | `python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py` | `MooncakeStore.register_mem_pool_host` | 703 | |
| `docs/frameworks/serving/sglang/hicache-mooncake.md` | mooncake-exists | `python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py` | `MooncakeStore.batch_exists` | 1302 | |
| `docs/frameworks/serving/sglang/hicache-mooncake.md` | mooncake-get | `python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py` | `MooncakeStore.batch_get_v1` | 1092 | |
| `docs/frameworks/serving/sglang/hicache-mooncake.md` | zero-copy-get | `python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py` | `MooncakeStore._get_batch_zero_copy_impl` | 1368 | |
| `docs/frameworks/serving/sglang/hicache-mooncake.md` | result-insert | `python/sglang/srt/mem_cache/unified_radix_cache.py` | `UnifiedRadixCache._handle_prefetch_result` | 2289 | |
| `docs/frameworks/serving/sglang/hicache-mooncake.md` | hybrid-sidecars | `python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py` | `MooncakeStore.batch_exists_v2` | 857 | |
| `docs/frameworks/serving/sglang/hicache-mooncake.md` | sidecar-validation | `python/sglang/srt/mem_cache/unified_radix_cache.py` | `UnifiedRadixCache._check_hybrid_prefetch_result` | 2403 | |
| `docs/frameworks/serving/sglang/hicache-mooncake.md` | scheduler-progress | `python/sglang/srt/managers/scheduler.py` | `Scheduler` | 4007 | |
| `docs/frameworks/serving/sglang/hicache-mooncake.md` | load-back | `python/sglang/srt/mem_cache/unified_radix_cache.py` | `UnifiedRadixCache._load_back_transfers` | 1787 | |
| `docs/frameworks/serving/sglang/hicache-mooncake.md` | backup-queue | `python/sglang/srt/managers/cache_controller.py` | `HiCacheController._page_backup` | 1355 | |
| `docs/frameworks/serving/sglang/hicache-mooncake.md` | mooncake-set | `python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py` | `MooncakeStore.batch_set_v1` | 1121 | |
| `docs/frameworks/serving/sglang/hicache-mooncake.md` | zero-copy-set | `python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py` | `MooncakeStore._put_batch_zero_copy_impl` | 1339 | |
| `docs/frameworks/serving/sglang/hicache-mooncake.md` | tier-scope | `docs/docs/advanced_features/hicache_design.mdx` | HiCache L1/L2/L3 tier-sharing table | 27 | 29 |
| `docs/frameworks/serving/sglang/hicache-mooncake.md` | integration-guide | `python/sglang/srt/mem_cache/storage/mooncake_store/README.md` | Mooncake as L3 KV Cache | 1 | |
| `docs/frameworks/serving/sglang/hicache-mooncake.md` | integration-test | `test/registered/hicache/test_hicache_storage_mooncake_backend.py` | Mooncake backend integration test module | 1 | |

## Reproduction Commands

No quantitative codebase claims are made. The integration test was inspected but not executed.
