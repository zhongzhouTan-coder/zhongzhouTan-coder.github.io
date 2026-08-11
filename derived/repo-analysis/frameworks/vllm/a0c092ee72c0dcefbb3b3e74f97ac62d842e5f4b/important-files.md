---
kind: repository-analysis
repository_id: github:vllm-project/vllm@a0c092ee72c0dcefbb3b3e74f97ac62d842e5f4b
commit: a0c092ee72c0dcefbb3b3e74f97ac62d842e5f4b
source_record: raw/frameworks/vllm-codebase--github-a0c092ee72c0.md
generated: 2026-07-29
---

# vllm Codebase Important Files

## Evidence Map

- `vllm/models/kimi_k3/__init__.py` — Hardware-isolated Kimi K3 model entrypoint
- `vllm/models/kimi_k3/nvidia/model.py` — NVIDIA Kimi K3 multimodal model, decoder, attention, MoE, and weight-loading path
- `vllm/models/kimi_k3/nvidia/kda.py` — Kimi K3 Delta Attention implementation
- `vllm/models/kimi_k3/nvidia/mla.py` — Kimi K3 Multi-head Latent Attention implementation
- `vllm/model_executor/layers/fused_moe/runner/latent_moe_runner.py` — Generic latent-MoE runner used by Kimi K3 routed experts
- `vllm/models/kimi_k3/nvidia/ops/latent_moe_tail.py` — Optional Kimi K3 latent-MoE tail fusion
- `vllm/models/kimi_k3/nvidia/mtp.py` — Kimi K3 MTP draft model
- `vllm/parser/kimi_k3.py` — Kimi K3 XTML parser composition

## Qwen3.5 MTP and target verification extension

Consuming page: `docs/frameworks/vllm-ascend/qwen3.5-mtp.md`

| Docs page | Finding | File | Symbol | Start | End |
|---|---|---|---|---:|---:|
| `docs/frameworks/vllm-ascend/qwen3.5-mtp.md` | mtp-model | `vllm/model_executor/models/qwen3_5_mtp.py` | `Qwen3_5MultiTokenPredictor` | 64 | 189 |
| `docs/frameworks/vllm-ascend/qwen3.5-mtp.md` | mtp-head | `vllm/model_executor/models/qwen3_5_mtp.py` | `Qwen3_5MTP.compute_logits` | 212 | 299 |
| `docs/frameworks/vllm-ascend/qwen3.5-mtp.md` | spec-metadata | `vllm/v1/spec_decode/metadata.py` | `SpecDecodeMetadata` | 8 | 31 |
| `docs/frameworks/vllm-ascend/qwen3.5-mtp.md` | sampler-entry | `vllm/v1/worker/gpu_model_runner.py` | `GPUModelRunner._sample` | 3692 | 3719 |
| `docs/frameworks/vllm-ascend/qwen3.5-mtp.md` | rejection-forward | `vllm/v1/sample/rejection_sampler.py` | `RejectionSampler.forward` | 38 | 181 |
| `docs/frameworks/vllm-ascend/qwen3.5-mtp.md` | greedy-verify | `vllm/v1/sample/rejection_sampler.py` | `rejection_greedy_sample_kernel` | 715 | 769 |
| `docs/frameworks/vllm-ascend/qwen3.5-mtp.md` | random-verify | `vllm/v1/sample/rejection_sampler.py` | `rejection_random_sample_kernel` | 774 | 845 |

## DCP and PCP implementation

Consuming page: `docs/frameworks/vllm/vllm-context-parallelism.md`

| Docs page | Finding | File | Symbol | Start | End |
|---|---|---|---|---:|---:|
| `docs/frameworks/vllm/vllm-context-parallelism.md` | pcp-config | `vllm/config/parallel.py` | `ParallelConfig.prefill_context_parallel_size` | 126 | 128 |
| `docs/frameworks/vllm/vllm-context-parallelism.md` | dcp-config | `vllm/config/parallel.py` | `decode_context_parallel_size` | 342 | 345 |
| `docs/frameworks/vllm/vllm-context-parallelism.md` | config-validation | `vllm/config/parallel.py` | `ParallelConfig.__post_init__` | 524 | 543 |
| `docs/frameworks/vllm/vllm-context-parallelism.md` | group-topology | `vllm/distributed/parallel_state.py` | `initialize_model_parallel` | 1746 | 1746 |
| `docs/frameworks/vllm/vllm-context-parallelism.md` | group-validation | `vllm/distributed/parallel_state.py` | `ensure_model_parallel_initialized` | 1992 | 2041 |
| `docs/frameworks/vllm/vllm-context-parallelism.md` | worker-groups | `vllm/v1/worker/gpu_worker.py` | `ensure_model_parallel_initialized` | 1380 | 1385 |
| `docs/frameworks/vllm/vllm-context-parallelism.md` | pcp-manager | `vllm/v1/worker/gpu/pcp_manager.py` | `PCPManager` | 37 | 123 |
| `docs/frameworks/vllm/vllm-context-parallelism.md` | pcp-validation | `vllm/v1/worker/gpu/pcp_manager.py` | `PCPManager.validate_config` | 125 | 180 |
| `docs/frameworks/vllm/vllm-context-parallelism.md` | pcp-rank-segments | `vllm/v1/worker/gpu/pcp_manager.py` | `_get_rank_segments` | 195 | 250 |
| `docs/frameworks/vllm/vllm-context-parallelism.md` | pcp-batch-layout | `vllm/v1/worker/gpu/pcp_manager.py` | `_build_batch_layout` | 252 | 317 |
| `docs/frameworks/vllm/vllm-context-parallelism.md` | pcp-init | `vllm/v1/worker/gpu/model_runner.py` | `GPUModelRunner` PCP initialization | 476 | 485 |
| `docs/frameworks/vllm/vllm-context-parallelism.md` | pcp-partition | `vllm/v1/worker/gpu/model_runner.py` | `maybe_partition_pcp_batch` | 1092 | 1098 |
| `docs/frameworks/vllm/vllm-context-parallelism.md` | pcp-restore | `vllm/v1/worker/gpu/model_runner.py` | `maybe_restore_pcp_for_sampling` | 1472 | 1474 |
| `docs/frameworks/vllm/vllm-context-parallelism.md` | pcp-cache-inputs | `vllm/model_executor/layers/attention/pcp.py` | `_gather_prefill_cache_inputs` | 11 | 45 |
| `docs/frameworks/vllm/vllm-context-parallelism.md` | mla-cache-integration | `vllm/model_executor/layers/attention/mla_attention.py` | `maybe_gather_mla_latent_cache_inputs` | 634 | 649 |
| `docs/frameworks/vllm/vllm-context-parallelism.md` | mla-merge-integration | `vllm/model_executor/layers/attention/mla_attention.py` | `MLAImpl.forward_impl` | 898 | 921 |
| `docs/frameworks/vllm/vllm-context-parallelism.md` | dcp-local-lengths | `vllm/v1/attention/backends/utils.py` | `get_dcp_local_seq_lens` | 887 | 920 |
| `docs/frameworks/vllm/vllm-context-parallelism.md` | dcp-ag-rs | `vllm/v1/attention/ops/common.py` | `cp_lse_ag_out_rs` | 213 | 236 |
| `docs/frameworks/vllm/vllm-context-parallelism.md` | dcp-ag-ar | `vllm/v1/attention/ops/common.py` | `cp_lse_ag_out_ar` | 238 | 261 |
| `docs/frameworks/vllm/vllm-context-parallelism.md` | dcp-a2a | `vllm/v1/attention/ops/dcp_alltoall.py` | `dcp_a2a_lse_reduce` | 392 | 460 |
| `docs/frameworks/vllm/vllm-context-parallelism.md` | dcp-prefill | `vllm/v1/attention/backends/flashinfer.py` | `BatchDCPPrefillWrapper` | 230 | 326 |
| `docs/frameworks/vllm/vllm-context-parallelism.md` | cache-scaling | `vllm/v1/core/single_type_kv_cache_manager.py` | `SingleTypeKVCacheManager` | 36 | 84 |
| `docs/frameworks/vllm/vllm-context-parallelism.md` | sliding-window-limit | `vllm/v1/core/single_type_kv_cache_manager.py` | `SlidingWindowManager` | 906 | 913 |
| `docs/frameworks/vllm/vllm-context-parallelism.md` | chunked-local-limit | `vllm/v1/core/single_type_kv_cache_manager.py` | `ChunkedLocalAttentionManager` | 1110 | 1156 |
| `docs/frameworks/vllm/vllm-context-parallelism.md` | mamba-limit | `vllm/v1/core/single_type_kv_cache_manager.py` | `MambaManager` | 1253 | 1296 |

## Focused DCP attention derivation

Consuming page: `docs/frameworks/vllm/dcp-attention/index.md`

| Docs page | Finding | File | Symbol | Start | End |
|---|---|---|---|---:|---:|
| `docs/frameworks/vllm/dcp-attention/index.md` | dcp-merge-selection | `vllm/model_executor/layers/attention/mla_attention.py` | `MLAImpl.forward_impl` | 896 | 921 |
| `docs/frameworks/vllm/dcp-attention/index.md` | dcp-lse-gather | `vllm/v1/attention/ops/common.py` | `_cp_lse_common` | 182 | 210 |
| `docs/frameworks/vllm/dcp-attention/index.md` | dcp-correction-kernel | `vllm/v1/attention/ops/common.py` | `_correct_attn_cp_out_kernel` | 10 | |
| `docs/frameworks/vllm/dcp-attention/index.md` | dcp-correction-helper | `vllm/v1/attention/ops/common.py` | `correct_attn_out` | 111 | 178 |
| `docs/frameworks/vllm/dcp-attention/index.md` | dcp-ag-rs | `vllm/v1/attention/ops/common.py` | `cp_lse_ag_out_rs` | 213 | 236 |
| `docs/frameworks/vllm/dcp-attention/index.md` | dcp-ag-ar | `vllm/v1/attention/ops/common.py` | `cp_lse_ag_out_ar` | 238 | 261 |
| `docs/frameworks/vllm/dcp-attention/index.md` | dcp-a2a | `vllm/v1/attention/ops/dcp_alltoall.py` | `dcp_a2a_lse_reduce` | 392 | 460 |

## Reproduction Commands

Record exact read-only search or counting commands here when the docs make
quantitative codebase claims.
