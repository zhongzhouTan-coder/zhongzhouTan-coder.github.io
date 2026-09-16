# DCP and PCP implementation evidence

**Repository:** `github:vllm-project/vllm`
**Commit:** `a0c092ee72c0dcefbb3b3e74f97ac62d842e5f4b`
**Inspected:** 2026-08-10
**Method:** Static reading of the clean pinned checkout; no multi-GPU end-to-end run was performed.

## Reader contract

- **Audience:** Engineers who know tensor parallel inference and paged KV caches and want to trace V1 context parallelism.
- **Question:** How do decode context parallelism (DCP) and prefill context parallelism (PCP) change process groups, batch layout, KV ownership, and attention output reconstruction?
- **Mental model:** PCP partitions transient prefill work and restores the global batch; DCP partitions persistent decode KV ownership and restores exact attention across local shards.
- **Offline:** `ParallelConfig` validates the TP/PCP/DCP topology before workers start.
- **Load time:** workers create TP, DCP, PCP, and other model-parallel groups; KV-cache managers scale attention block size for DCP.
- **Runtime:** `PCPManager` rewrites batches, attention layers gather or merge tensors, and DCP metadata localizes sequence lengths and block ownership.
- **Limits:** The pinned MRV2 PCP path is MLA-only and rejects several features; DCP support is cache-type and backend dependent.

## Evidence map

| Finding | File and symbol | Exact role |
|---|---|---|
| `config-topology` | `vllm/config/parallel.py::ParallelConfig` | Declares PCP/DCP sizes and validates compatible combinations. |
| `group-topology` | `vllm/distributed/parallel_state.py::initialize_model_parallel` | Builds TP, DCP, and PCP rank groups from the TP x PCP layout. |
| `pcp-manager` | `vllm/v1/worker/gpu/pcp_manager.py::PCPManager` | Splits prefills into mirrored chunks, replicates decodes, pads local batches, and restores hidden states. |
| `pcp-runner` | `vllm/v1/worker/gpu/model_runner.py::GPUModelRunner` | Creates the manager and inserts partition/attention/restore calls into each step. |
| `pcp-cache-inputs` | `vllm/model_executor/layers/attention/pcp.py::_gather_prefill_cache_inputs` | Gathers prefill KV inputs across PCP while keeping decode writes local. |
| `mla-integration` | `vllm/model_executor/layers/attention/mla_attention.py::MLAImpl.forward_impl` | Selects DCP merge mode and finalizes PCP decode head layout. |
| `dcp-local-lengths` | `vllm/v1/attention/backends/utils.py::get_dcp_local_seq_lens` | Computes each rank's local KV length under interleaving. |
| `dcp-ag-rs` | `vllm/v1/attention/ops/common.py::cp_lse_ag_out_rs` | All-gathers LSE, merges exact partial attention, and reduce-scatters heads. |
| `dcp-a2a` | `vllm/v1/attention/ops/dcp_alltoall.py::dcp_a2a_lse_reduce` | Packs output plus LSE, performs one all-to-all, and runs exact LSE-weighted reduction. |
| `dcp-prefill` | `vllm/v1/attention/backends/flashinfer.py::BatchDCPPrefillWrapper` | Gathers prefill queries across DCP heads and merges context results. |
| `cache-scaling` | `vllm/v1/core/single_type_kv_cache_manager.py::SingleTypeKVCacheManager` | Multiplies attention block size by DCP world size; leaves PCP out of KV shard scaling. |
| `cache-limits` | `vllm/v1/core/single_type_kv_cache_manager.py::SlidingWindowManager`, `ChunkedLocalAttentionManager`, `MambaManager` | Explicitly rejects DCP/PCP for unsupported cache types. |

## Runtime trace

1. `GPUWorker.init_device()` calls `ensure_model_parallel_initialized()` with TP, PCP, and DCP sizes.
2. `GPUModelRunner` creates `PCPManager` only when PCP is greater than one.
3. Before model execution, `PCPManager.partition_batch()` maps each prefill to two mirrored chunks per PCP rank and leaves decode requests replicated.
4. The manager gathers rank-local block tables and slot mappings, while `prepare_dcp_local_seq_lens()` describes the local KV length for DCP attention.
5. MLA and FlashInfer attention either use AG+RS, AG+AR, or the A2A reducer to merge exact partial attention using log-sum-exp values.
6. After the forward pass, PCP all-gathers hidden states and indexes them back into the global batch before sampling.

## Verification boundary

The checkout contains CPU-only DCP A2A reference tests in `tests/distributed/test_dcp_a2a.py`, distributed context-parallel integration tests in `tests/distributed/test_context_parallel.py`, and focused DCP localization tests in `tests/v1/attention/test_indexer_dcp_localize.py`. This note records their existence and the assertions they target; this workspace did not run a multi-GPU serving test.
