---
kind: repository-analysis
repository_id: github:vllm-project/vllm@5c9ff5366b039a69b344773bdfead8466ed9a097
commit: 5c9ff5366b039a69b344773bdfead8466ed9a097
source_record: raw/frameworks/vllm-codebase--github-5c9ff5366b03.md
generated: 2026-08-24
---

# vLLM Mamba2 and Linear-Attention Prefix-Cache Evidence

Consuming page:
`docs/frameworks/vllm/mamba2-linear-attention-prefix-cache/index.md`

## Reader Contract

- **Audience:** vLLM users and developers who understand ordinary attention
  KV caching but have not followed recurrent-state prefix reuse through V1.
- **Question:** how does a matching token prefix become a usable Mamba2 or
  linear-attention initial state, and how is the resulting state published for
  the next request?
- **Mental model:** vLLM hashes token prefixes, but a Mamba-backed cache group
  maps each reusable boundary to a recurrent-state snapshot; a hit resumes the
  recurrence at that boundary instead of replaying the prefix.
- **Offline/load time:** every Mamba-like layer publishes a `MambaSpec` with
  state shapes, dtypes, block size, cache mode, and speculative-state count.
- **Runtime:** the scheduler finds and reconciles a checkpoint, adopts its
  physical block, allocates a writable running block when needed, the worker
  maps that block to state indices, the kernel scans only the suffix, and the
  cache manager hashes the resulting checkpoint.
- **Limits:** clean static reading at the pinned commit. No Mamba2, Gated Delta
  Network (GDN), or linear-attention model was executed. The 2026-08-24 scoped
  freshness check returned `decision: defer` because the newest snapshot is
  not eligible for promotion until 2026-09-01.

## Representation Plan

| Reader question | Evidence | Representation | Teaching job |
|---|---|---|---|
| What does prefix caching retain for recurrent layers instead of normal attention? | `MambaSpec`, block-pool hash publication, Mamba hit lookup | Exact comparison table | Separate token-range K/V pages from boundary-state snapshots. |
| How does one producer request make a state reusable by a later consumer? | Scheduler, cache manager, block pool, worker, metadata builder, layer | Mermaid sequence plus numbered state trace | Preserve ownership and direction across scheduler and GPU boundaries. |
| Why can a hybrid full-attention + Mamba model report a shorter hit than either group alone? | Hybrid coordinator fixed-point lookup | Reconciliation table | Show that one safe resume boundary must be valid for every cache group. |
| What changes between `none`, `all`, and `align` modes? | Cache configuration, Mamba manager, metadata builder, mixer | Mode comparison table | Explain memory cost, checkpoint density, and state-copy behavior without conflating modes. |
| Do Mamba2, GDN, and other linear-attention layers use the same compute kernel? | Mamba2 and linear metadata/layer paths | Shared-plumbing/divergent-kernel table | Make the architectural reuse boundary explicit. |

## Required Code Evidence

| Docs page | Finding | File | Symbol | Start | End |
|---|---|---|---|---:|---:|
| `docs/frameworks/vllm/mamba2-linear-attention-prefix-cache/index.md` | cache-modes | `vllm/config/cache.py` | `CacheConfig.mamba_cache_mode` | 145 | 165 |
| `docs/frameworks/vllm/mamba2-linear-attention-prefix-cache/index.md` | state-spec | `vllm/model_executor/layers/mamba/abstract.py` | `MambaBase.get_kv_cache_spec` | 63 | 81 |
| `docs/frameworks/vllm/mamba2-linear-attention-prefix-cache/index.md` | mamba-capacity | `vllm/v1/kv_cache_interface.py` | `MambaSpec` | 667 | 708 |
| `docs/frameworks/vllm/mamba2-linear-attention-prefix-cache/index.md` | scheduler-hit-admission | `vllm/v1/core/sched/scheduler.py` | waiting-request prefix lookup | 805 | 818 |
| `docs/frameworks/vllm/mamba2-linear-attention-prefix-cache/index.md` | lookup-cap | `vllm/v1/core/kv_cache_manager.py` | `KVCacheManager.get_computed_blocks` | 232 | 298 |
| `docs/frameworks/vllm/mamba2-linear-attention-prefix-cache/index.md` | hybrid-reconciliation | `vllm/v1/core/kv_cache_coordinator.py` | `HybridKVCacheCoordinator.find_longest_cache_hit` | 757 | 889 |
| `docs/frameworks/vllm/mamba2-linear-attention-prefix-cache/index.md` | mamba-checkpoint-hit | `vllm/v1/core/single_type_kv_cache_manager.py` | `MambaManager.find_longest_cache_hit` | 1294 | 1371 |
| `docs/frameworks/vllm/mamba2-linear-attention-prefix-cache/index.md` | adopt-hit | `vllm/v1/core/single_type_kv_cache_manager.py` | `add_local_computed_blocks` | 229 | 286 |
| `docs/frameworks/vllm/mamba2-linear-attention-prefix-cache/index.md` | same-step-visibility-gate | `vllm/v1/core/single_type_kv_cache_manager.py` | `MambaManager.get_num_blocks_to_allocate` | 1477 | 1486 |
| `docs/frameworks/vllm/mamba2-linear-attention-prefix-cache/index.md` | align-running-block | `vllm/v1/core/single_type_kv_cache_manager.py` | `MambaManager.allocate_new_blocks` | 1547 | 1667 |
| `docs/frameworks/vllm/mamba2-linear-attention-prefix-cache/index.md` | checkpoint-publication | `vllm/v1/core/single_type_kv_cache_manager.py` | `MambaManager.cache_blocks` | 1691 | 1714 |
| `docs/frameworks/vllm/mamba2-linear-attention-prefix-cache/index.md` | hash-index-publication | `vllm/v1/core/block_pool.py` | `BlockPool.cache_full_blocks` | 225 | 299 |
| `docs/frameworks/vllm/mamba2-linear-attention-prefix-cache/index.md` | cached-free-retention | `vllm/v1/core/block_pool.py` | `BlockPool.free_blocks` | 719 | 743 |
| `docs/frameworks/vllm/mamba2-linear-attention-prefix-cache/index.md` | worker-admission | `vllm/v1/worker/gpu_model_runner.py` | `CachedRequestState` construction | 1319 | 1363 |
| `docs/frameworks/vllm/mamba2-linear-attention-prefix-cache/index.md` | align-state-migration | `vllm/v1/worker/mamba_utils.py` | `preprocess_mamba` | 1229 | 1333 |
| `docs/frameworks/vllm/mamba2-linear-attention-prefix-cache/index.md` | mamba-state-addressing | `vllm/v1/attention/backends/mamba_attn.py` | `_compute_common_metadata` state lookup | 504 | 570 |
| `docs/frameworks/vllm/mamba2-linear-attention-prefix-cache/index.md` | mamba2-resume-scan | `vllm/model_executor/layers/mamba/mamba_mixer2.py` | prefill initial-state load and scan | 815 | 893 |
| `docs/frameworks/vllm/mamba2-linear-attention-prefix-cache/index.md` | mamba2-state-write | `vllm/model_executor/layers/mamba/mamba_mixer2.py` | prefill final-state write | 970 | 983 |
| `docs/frameworks/vllm/mamba2-linear-attention-prefix-cache/index.md` | linear-state-addressing | `vllm/v1/attention/backends/linear_attn.py` | `LinearAttentionMetadataBuilder.build` | 63 | 94 |
| `docs/frameworks/vllm/mamba2-linear-attention-prefix-cache/index.md` | linear-state-consumption | `vllm/model_executor/layers/mamba/linear/minimax_linear_attn.py` | `MiniMaxText01LinearAttention._forward` | 277 | 318 |

## Reproduction Commands

The investigation used read-only exact-string searches over the declared
subsystems, followed by numbered-line inspection:

```bash
rg -n -i "prefix cach|mamba_cache_mode|find_longest_cache_hit|state_indices" \
  external-repos/vllm-5c9ff5366b03/vllm/v1 \
  external-repos/vllm-5c9ff5366b03/vllm/model_executor/layers/mamba
```

Freshness was checked with:

```bash
./scripts/run-in-workspace.sh python scripts/repositories/worktree.py sync \
  vllm-5c9ff5366b03 \
  --path vllm/model_executor/models/mamba2.py \
  --path vllm/model_executor/layers/mamba \
  --path vllm/v1/attention/backends \
  --path vllm/v1/core \
  --path vllm/v1/worker/mamba_utils.py \
  --path vllm/v1/worker/gpu_model_runner.py
```

Result: `decision: defer`; pinned evidence remains
`5c9ff5366b039a69b344773bdfead8466ed9a097`.
