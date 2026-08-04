---
kind: repository-analysis
repository_id: github:vllm-project/vllm@dd11df04f3b7046c40f13e586ac38a3725bc3c03
commit: dd11df04f3b7046c40f13e586ac38a3725bc3c03
source_record: raw/frameworks/vllm-codebase--github-dd11df04f3b7.md
generated: 2026-08-03
---

# vLLM V1 Block Table Management — Reading Notes

Scope: `vllm/v1/core/` (block pool, KV cache managers, coordinator) plus the
worker-side block table in `vllm/v1/worker/block_table.py`. Static reading of a
clean detached checkout at `dd11df04f3b7046c40f13e586ac38a3725bc3c03` (2026-08-03).

## Layout

The legacy `vllm/core/block/` package no longer exists. V1 block management now
lives under `vllm/v1/core/`:

- `block_pool.py` — `BlockPool` + `BlockHashToBlockMap` + `FreeKVCacheBlockQueue`.
- `kv_cache_manager.py` — `KVCacheManager` facade used by the scheduler.
- `kv_cache_coordinator.py` — `KVCacheCoordinator` (abstract), no-prefix-cache,
  `UnitaryKVCacheCoordinator` (1 group), `HybridKVCacheCoordinator` (≥2 groups).
- `single_type_kv_cache_manager.py` — per-spec managers:
  `FullAttentionManager` (also TQ/MLA/HiddenState), `RSWAManager`,
  `SlidingWindowManager`, `ChunkedLocalAttentionManager`, `MambaManager`
  (align / non-align), `CrossAttentionManager`, `SinkFullAttentionManager`.
- `kv_cache_utils.py` — `KVCacheBlock`, `FreeKVCacheBlockQueue`, block hashing
  (`hash_block_tokens`, `get_request_block_hasher`), `resolve_block_hashes`,
  `BlockHashListWithBlockSize`, pool sizing (`get_num_blocks`).
- `kv_cache_metrics.py`, `encoder_cache_manager.py` — support.
- `vllm/v1/worker/block_table.py` — `BlockTable` / `MultiGroupBlockTable`, the
  actual int32 device tensor consumed by attention kernels.

## Key structures

- `KVCacheBlock`: `block_id`, `ref_cnt`, `_block_hash` (full hash + group id),
  `_block_hash_num_tokens`, doubly-linked `prev_free_block`/`next_free_block`,
  `is_null`. A `null_block` (block 0) pads skipped positions; never freed.
- `FreeKVCacheBlockQueue`: O(1) middle removal over the blocks' own links;
  eviction order = LRU at the front, tail-of-chain first on ties; `prepend_n`
  puts unhashed (never-APC) blocks at the front.
- `BlockHashToBlockMap`: `{hash_with_group_id: KVCacheBlock | dict[int, KVCacheBlock]}`
  prefix-cache index; no de-duplication so block tables stay append-only.
- `Request.block_hashes`: chained hashes computed at `hash_block_size`
  granularity (`get_request_block_hasher`), each hash covers its full prefix.

## Allocation flow (scheduler → manager → pool)

1. Scheduler calls `KVCacheManager.get_computed_blocks(request)` →
   `coordinator.find_longest_cache_hit(block_hashes, num_tokens - 1)`.
2. `KVCacheManager.allocate_slots(...)` computes token budget, applies watermark
   (waiting/preempted only when other requests are scheduled), calls
   `coordinator.remove_skipped_blocks`, `get_num_blocks_to_allocate`,
   `allocate_new_computed_blocks`, `allocate_new_blocks`, then `cache_blocks`.
3. `BlockPool.get_new_blocks(n)` pops from the free queue, evicts any cached
   hash on the block (`_maybe_evict_cached_block`), bumps `ref_cnt`.
4. `cache_blocks` → `BlockPool.cache_full_blocks` registers `hash+group_id` →
   block in the prefix map; `num_cached_block` per request tracks the frontier.
5. Worker `MultiGroupBlockTable.add_row(block_ids)` materializes the per-group
   logical block id lists; `compute_slot_mapping` fills per-token slots; the
   int32 block-table tensor is what PagedAttention kernels index.

## Prefix caching

- Hash chain: `hash_block_tokens(hash_fn, parent_hash, tokens, extra_keys)`;
  `hash_block_size` = GCD of group block sizes (or `prefix_match_unit`), each
  group's block size is a multiple of it.
- `resolve_block_hashes` reuses/views hashes at a group's block size; the last
  `hash_block_size` hash in a `block_size` block is that block's hash.
- `FullAttentionManager.find_longest_cache_hit`: left-to-right run of cached
  full blocks; fine-grained mode (`alignment_tokens == hash_block_size`)
  probes interior boundaries high-to-low for partial hits; eagle drops one unit.
- `SlidingWindowManager`: right-to-left search for a run of
  `window/blocksize` contiguous cached blocks; returns null-padded lists.
- `HybridKVCacheCoordinator.find_longest_cache_hit`: fixed-point iteration over
  per-spec groups until the hit length converges; reports uncached shared prefix.
- Eviction: `free_blocks` decrements `ref_cnt`; blocks at 0 with a hash go to
  the LRU tail (eviction candidates), unhashed ones `prepend_n` to be reused
  first; allocation of a cached block calls `_maybe_evict_cached_block`.
- `touch` removes a ref_cnt==0 block from the free queue and bumps refcount —
  the prefix-hit fast path.
- Partial hits (fine-grained) redirect the shared tail block to a private CoW
  block (`_apply_cow`), queueing `KVCacheBlockCopy` for the worker.

## Refcounts / sharing

- A block's `ref_cnt` counts requests sharing it (prefix hits and CoW). Blocks
  shared by all allocated requests are "common prefix" blocks
  (`get_num_common_prefix_blocks`, ref_cnt == len(req_to_blocks)).

## Sliding window / Mamba recycling

- `remove_skipped_blocks` frees blocks outside the attention window and pads
  with `null_block`; `get_num_skipped_tokens` differs per spec (SWA:
  `num_computed - window + 1`; chunked-local: chunk floor; mamba: `-1`).
- `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` sparsifies SWA/Mamba cached
  checkpoints (`reachable_block_mask`), keeping replay-boundary and shared
  junction tails reachable.

## Revisions compared

- Same layout in `a0c092ee72c0` (2026-07-29) and `dd11df04f3b7` (2026-08-03).
- Changed in scope: `kv_cache_utils.py` (removed deprecated `need_extra_keys`
  and `is_kv_cache_page_size_uniform`; improved same-type group merging),
  `sched/interface.py` (+4 lines), `sched/scheduler.py` (42-line change).
  Core block table mechanism is unchanged.

## Limitations

- Static reading only; no runtime tracing of refcounts or eviction.
- `KVCacheBlockCopy`, zeroing, and KV-connector paths read from source without
  execution.
