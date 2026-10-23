---
kind: repository-analysis
repository_id: github:vllm-project/vllm@02aa10539f15c8c73aa7fd4a5ee0f691cbf88dcb
commit: 02aa10539f15c8c73aa7fd4a5ee0f691cbf88dcb
source_record: raw/frameworks/serving/vllm/vllm-codebase--github-02aa10539f15.md
generated: 2026-10-21
---

# GLM-5.3-Flash GPU Inference Path Evidence

## Reader Contract

- **Audience:** vLLM serving engineers familiar with decoder layers, paged KV caches, and GPU backend selection.
- **Question:** how does one ordinary decode row traverse GLM5Next KDA/MLA layers and return as a sampled token ID?
- **Mental model:** KDA layers update recurrent state; sparse MLA layers use kpool keys to choose token positions, then read the selected latent records from a distinct main cache.
- **Boundary:** `SchedulerOutput` into the V1 GPU worker through `sample_tokens()` returning `ModelRunnerOutput`.
- **Selected branch:** one already-prefetched text request; one decode row; context longer than `index_topk`; CUDA SM90; BF16 model/cache; explicit `FLASHMLA_SPARSE`; TP=1, PP=1, DCP/PCP=1; no speculative decoding or async scheduling.
- **Limits:** checkpoint config and weights were not loaded, and no GPU execution was performed. Layer indices, `index_topk`, `index_kpool`, MoE quantization, cache block size, and automatic backend choice remain deployment-specific.

## Revision and Freshness

The inspected checkout is the clean detached GitHub revision
`02aa10539f15c8c73aa7fd4a5ee0f691cbf88dcb`, originally recorded on
2026-10-10. A scoped freshness check on 2026-10-21 found relevant upstream
changes through `fc377f77714b42f706640d5857a0f2a5f233758d`; the 14-day evidence
promotion interval defers a new revision until 2026-10-24. The newer candidate
was not inspected.

## Representation Plan

| Reader question | Evidence | Representation | Teaching job |
|---|---|---|---|
| Which state is recurrent, selector-only, or read by attention? | GLM5Next cache specs and layer forwards | State-store table | Prevent KDA state, pooled index keys, and main MLA KV from being conflated. |
| How do pool IDs become attention-cache rows? | Kpool expansion and block-table conversion | Coordinate formulas | Make the pool/token/physical-row units explicit. |
| How does a token return from the worker? | Scheduler, V1 runner, sampler, and worker wrapper | Mermaid sequence plus numbered trace | Show both attention branches and complete the worker round trip. |

## Evidence Map

- `vllm/model_executor/models/registry.py` — GLM5Next and GLM DSA architecture routes.
- `vllm/models/glm5next/common/model.py` — sparse config validation, shared top-k buffer, KDA/MLA layer construction, residual/MLP selection, model output, and logits.
- `vllm/models/glm5next/common/attention.py` — pooled-key and tail-cache specs, indexer record width, and indexer projection.
- `vllm/models/glm5next/common/kda.py` — KDA state spec, state shape, gate and decode update.
- `vllm/models/glm5next/nvidia/sparse_indexer.py` — CUDA pool update, paged scoring, top-k dispatch, token-index expansion, and DeepGEMM requirement.
- `vllm/models/glm5next/nvidia/ops/kpool_compress.py` — tail-ring update and pool-to-token mapping.
- `vllm/model_executor/layers/mla.py` — indexer-before-MLA ordering and logical-top-k readiness.
- `vllm/model_executor/layers/attention/mla_attention.py` — main-cache spec/write, sparse decode orchestration, and value-up projection.
- `vllm/v1/attention/backends/mla/index_group.py` and `sparse_utils.py` — event-ordered top-k remap and block-table conversion.
- `vllm/v1/attention/backends/mla/flashmla_sparse.py` — SM90 NoPE-512 eligibility and sparse kernel call.
- `vllm/v1/worker/gpu_model_runner.py` and `gpu_worker.py` — V1 worker input, slots, logits, sampling, and output.

## Required Code Evidence

| Docs page | Finding | File | Symbol | Start | End |
|---|---|---|---|---:|---:|
| `docs/frameworks/serving/vllm/glm-5.3-flash-gpu-inference-path/index.md` | glm5next-architecture | `vllm/model_executor/models/registry.py` | `Glm5NextForCausalLM` mapping | 119 | — |
| `docs/frameworks/serving/vllm/glm-5.3-flash-gpu-inference-path/index.md` | glm-dsa-contrast | `vllm/model_executor/models/registry.py` | `GlmMoeDsaForCausalLM` mapping | 118 | — |
| `docs/frameworks/serving/vllm/glm-5.3-flash-gpu-inference-path/index.md` | glm53-flash-identity | `vllm/models/glm5next/common/kda.py` | GLM-5.3-Flash module docstring | 3 | — |
| `docs/frameworks/serving/vllm/glm-5.3-flash-gpu-inference-path/index.md` | sparse-tail-config-gate | `vllm/models/glm5next/common/model.py` | `_validate_supported_config` | 144 | — |
| `docs/frameworks/serving/vllm/glm-5.3-flash-gpu-inference-path/index.md` | model-topk-buffer | `vllm/models/glm5next/common/model.py` | `Glm5NextModel.__init__` | 701 | — |
| `docs/frameworks/serving/vllm/glm-5.3-flash-gpu-inference-path/index.md` | layer-kind-selection | `vllm/models/glm5next/common/model.py` | `Glm5NextDecoderLayer.__init__` | 345 | — |
| `docs/frameworks/serving/vllm/glm-5.3-flash-gpu-inference-path/index.md` | layer-residual-forward | `vllm/models/glm5next/common/model.py` | `Glm5NextDecoderLayer.forward` | 518 | — |
| `docs/frameworks/serving/vllm/glm-5.3-flash-gpu-inference-path/index.md` | model-layer-loop | `vllm/models/glm5next/common/model.py` | `Glm5NextModel.forward` | 793 | — |
| `docs/frameworks/serving/vllm/glm-5.3-flash-gpu-inference-path/index.md` | model-logits | `vllm/models/glm5next/common/model.py` | `Glm5NextForCausalLM.compute_logits` | 1126 | — |
| `docs/frameworks/serving/vllm/glm-5.3-flash-gpu-inference-path/index.md` | scheduler-output | `vllm/v1/core/sched/scheduler.py` | `Scheduler.schedule` | 1500 | — |
| `docs/frameworks/serving/vllm/glm-5.3-flash-gpu-inference-path/index.md` | v1-runner-selection | `vllm/v1/worker/gpu_worker.py` | `GPUWorker._make_model_runner` | 542 | — |
| `docs/frameworks/serving/vllm/glm-5.3-flash-gpu-inference-path/index.md` | scheduled-token-inputs | `vllm/v1/worker/gpu_model_runner.py` | `GPUModelRunner._prepare_inputs` | 1943 | — |
| `docs/frameworks/serving/vllm/glm-5.3-flash-gpu-inference-path/index.md` | paged-slot-mapping | `vllm/v1/worker/gpu_model_runner.py` | `GPUModelRunner._get_slot_mappings` | 4024 | — |
| `docs/frameworks/serving/vllm/glm-5.3-flash-gpu-inference-path/index.md` | layer-attention-metadata | `vllm/v1/worker/gpu_model_runner.py` | `GPUModelRunner._build_attention_metadata` | 2272 | — |
| `docs/frameworks/serving/vllm/glm-5.3-flash-gpu-inference-path/index.md` | v1-forward-context | `vllm/v1/worker/gpu_model_runner.py` | `GPUModelRunner.execute_model` | 4108 | — |
| `docs/frameworks/serving/vllm/glm-5.3-flash-gpu-inference-path/index.md` | execute-state-save | `vllm/v1/worker/gpu_model_runner.py` | `ExecuteModelState` assignment | 4446 | — |
| `docs/frameworks/serving/vllm/glm-5.3-flash-gpu-inference-path/index.md` | runner-logits | `vllm/v1/worker/gpu_model_runner.py` | `GPUModelRunner.execute_model` logits call | 4421 | — |
| `docs/frameworks/serving/vllm/glm-5.3-flash-gpu-inference-path/index.md` | ordinary-sampler | `vllm/v1/worker/gpu_model_runner.py` | `GPUModelRunner._sample` | 3617 | — |
| `docs/frameworks/serving/vllm/glm-5.3-flash-gpu-inference-path/index.md` | sample-token-phase | `vllm/v1/worker/gpu_model_runner.py` | `GPUModelRunner.sample_tokens` | 4483 | — |
| `docs/frameworks/serving/vllm/glm-5.3-flash-gpu-inference-path/index.md` | runner-output | `vllm/v1/worker/gpu_model_runner.py` | `ModelRunnerOutput` construction | 4698 | — |
| `docs/frameworks/serving/vllm/glm-5.3-flash-gpu-inference-path/index.md` | worker-token-return | `vllm/v1/worker/gpu_worker.py` | `GPUWorker.sample_tokens` | 1260 | — |
| `docs/frameworks/serving/vllm/glm-5.3-flash-gpu-inference-path/index.md` | kda-cache-spec | `vllm/models/glm5next/common/kda.py` | `Glm5NextLinearAttention.get_kv_cache_spec` | 168 | — |
| `docs/frameworks/serving/vllm/glm-5.3-flash-gpu-inference-path/index.md` | kda-state-shape | `vllm/models/glm5next/common/kda.py` | `Glm5NextLinearAttention.get_state_shape` | 187 | — |
| `docs/frameworks/serving/vllm/glm-5.3-flash-gpu-inference-path/index.md` | kda-prefill-backend | `vllm/models/glm5next/common/kda.py` | `_resolve_kda_prefill_backend` | 137 | — |
| `docs/frameworks/serving/vllm/glm-5.3-flash-gpu-inference-path/index.md` | kda-bounded-gate | `vllm/models/glm5next/common/kda.py` | bounded-gate initialization | 333 | — |
| `docs/frameworks/serving/vllm/glm-5.3-flash-gpu-inference-path/index.md` | kda-conv-state-write | `vllm/models/glm5next/common/kda.py` | `Glm5NextLinearAttention._forward` | 640 | — |
| `docs/frameworks/serving/vllm/glm-5.3-flash-gpu-inference-path/index.md` | kda-recurrent-state-update | `vllm/models/glm5next/common/kda.py` | `Glm5NextLinearAttention._forward` | 758 | — |
| `docs/frameworks/serving/vllm/glm-5.3-flash-gpu-inference-path/index.md` | kda-triton-kernel | `vllm/models/glm5next/nvidia/ops/third_party/kda/kernels.py` | `fused_recurrent_gated_delta_rule_fwd_kernel` launch | 101 | — |
| `docs/frameworks/serving/vllm/glm-5.3-flash-gpu-inference-path/index.md` | moe-forward | `vllm/models/glm5next/common/model.py` | `Glm5NextMoE.forward` | 317 | — |
| `docs/frameworks/serving/vllm/glm-5.3-flash-gpu-inference-path/index.md` | pooled-index-cache-spec | `vllm/models/glm5next/common/attention.py` | `Glm5NextIndexerCache.get_kv_cache_spec` | 112 | — |
| `docs/frameworks/serving/vllm/glm-5.3-flash-gpu-inference-path/index.md` | tail-ring-cache-spec | `vllm/models/glm5next/common/attention.py` | `Glm5NextTailCache.get_kv_cache_spec` | 156 | — |
| `docs/frameworks/serving/vllm/glm-5.3-flash-gpu-inference-path/index.md` | pooled-index-record-width | `vllm/models/glm5next/common/attention.py` | `Indexer` cache construction | 261 | — |
| `docs/frameworks/serving/vllm/glm-5.3-flash-gpu-inference-path/index.md` | indexer-projection | `vllm/models/glm5next/common/attention.py` | `Indexer.forward` | 297 | — |
| `docs/frameworks/serving/vllm/glm-5.3-flash-gpu-inference-path/index.md` | kpool-decode-update | `vllm/models/glm5next/nvidia/sparse_indexer.py` | `sparse_attn_indexer_kpool` | 516 | — |
| `docs/frameworks/serving/vllm/glm-5.3-flash-gpu-inference-path/index.md` | kpool-paged-scorer | `vllm/models/glm5next/nvidia/sparse_indexer.py` | `fp8_fp4_paged_mqa_logits` call | 582 | — |
| `docs/frameworks/serving/vllm/glm-5.3-flash-gpu-inference-path/index.md` | kpool-topk-dispatch | `vllm/models/glm5next/nvidia/sparse_indexer.py` | `get_indexer_topk` call | 609 | — |
| `docs/frameworks/serving/vllm/glm-5.3-flash-gpu-inference-path/index.md` | short-decode-path | `vllm/models/glm5next/nvidia/sparse_indexer.py` | `_fill_short_decode_causal_indices` gate | 529 | — |
| `docs/frameworks/serving/vllm/glm-5.3-flash-gpu-inference-path/index.md` | indexer-token-expansion | `vllm/models/glm5next/nvidia/sparse_indexer.py` | decode pool expansion | 646 | — |
| `docs/frameworks/serving/vllm/glm-5.3-flash-gpu-inference-path/index.md` | pool-to-token-formula | `vllm/models/glm5next/nvidia/ops/kpool_compress.py` | `expand_pools_and_append_tail` | 882 | — |
| `docs/frameworks/serving/vllm/glm-5.3-flash-gpu-inference-path/index.md` | kpool-tail-kernel | `vllm/models/glm5next/nvidia/ops/kpool_compress.py` | `_kpool_decode_update_batched_kernel` | 457 | — |
| `docs/frameworks/serving/vllm/glm-5.3-flash-gpu-inference-path/index.md` | deepgemm-requirement | `vllm/models/glm5next/nvidia/sparse_indexer.py` | `SparseAttnIndexerKpool.__init__` | 661 | — |
| `docs/frameworks/serving/vllm/glm-5.3-flash-gpu-inference-path/index.md` | indexer-topk-auto | `vllm/model_executor/layers/indexer_topk.py` | `SparseIndexerTopk._resolve_auto` | 239 | — |
| `docs/frameworks/serving/vllm/glm-5.3-flash-gpu-inference-path/index.md` | topk-ready-event | `vllm/model_executor/layers/mla.py` | `record_logical_topk_ready` | 235 | — |
| `docs/frameworks/serving/vllm/glm-5.3-flash-gpu-inference-path/index.md` | topk-side-stream-wait | `vllm/v1/attention/backends/mla/index_group.py` | `SparseMLAIndexGroup._convert_once` | 142 | — |
| `docs/frameworks/serving/vllm/glm-5.3-flash-gpu-inference-path/index.md` | topk-consumer-wait | `vllm/v1/attention/backends/mla/index_group.py` | physical-top-k event wait | 160 | — |
| `docs/frameworks/serving/vllm/glm-5.3-flash-gpu-inference-path/index.md` | physical-cache-row-map | `vllm/v1/attention/backends/mla/sparse_utils.py` | `triton_convert_req_index_to_global_index` | 462 | — |
| `docs/frameworks/serving/vllm/glm-5.3-flash-gpu-inference-path/index.md` | mla-backend-config | `vllm/model_executor/layers/attention/mla_attention.py` | `get_attn_backend` call | 513 | — |
| `docs/frameworks/serving/vllm/glm-5.3-flash-gpu-inference-path/index.md` | mla-cache-spec | `vllm/model_executor/layers/attention/mla_attention.py` | `MLAAttention.get_kv_cache_spec` | 1350 | — |
| `docs/frameworks/serving/vllm/glm-5.3-flash-gpu-inference-path/index.md` | mla-cache-write | `vllm/model_executor/layers/attention/mla_attention.py` | `MLAAttention.update_kv_cache` | 769 | — |
| `docs/frameworks/serving/vllm/glm-5.3-flash-gpu-inference-path/index.md` | mla-cache-write-op | `vllm/v1/attention/backend.py` | `MLAAttentionImpl.do_kv_cache_update` | 1105 | — |
| `docs/frameworks/serving/vllm/glm-5.3-flash-gpu-inference-path/index.md` | mla-decode-query-absorption | `vllm/model_executor/layers/attention/mla_attention.py` | NoPE query projection through `W_UK_T` | 1071 | — |
| `docs/frameworks/serving/vllm/glm-5.3-flash-gpu-inference-path/index.md` | mla-prefill-decode-dispatch | `vllm/model_executor/layers/attention/mla_attention.py` | `MLAAttention.forward_impl` | 884 | — |
| `docs/frameworks/serving/vllm/glm-5.3-flash-gpu-inference-path/index.md` | flashmla-backend-class | `vllm/v1/attention/backends/mla/flashmla_sparse.py` | `FlashMLASparseBackend` | 147 | — |
| `docs/frameworks/serving/vllm/glm-5.3-flash-gpu-inference-path/index.md` | flashmla-nope-eligibility | `vllm/v1/attention/backends/mla/flashmla_sparse.py` | `supports_combination` | 207 | — |
| `docs/frameworks/serving/vllm/glm-5.3-flash-gpu-inference-path/index.md` | flashmla-bf16-decode | `vllm/v1/attention/backends/mla/flashmla_sparse.py` | `FlashMLASparseImpl._forward_bf16_kv` | 844 | — |
| `docs/frameworks/serving/vllm/glm-5.3-flash-gpu-inference-path/index.md` | flashmla-kernel | `vllm/v1/attention/backends/mla/flashmla_sparse.py` | `flash_mla_sparse_fwd` call | 1388 | — |
| `docs/frameworks/serving/vllm/glm-5.3-flash-gpu-inference-path/index.md` | mla-value-up-projection | `vllm/model_executor/layers/attention/mla_attention.py` | `MLAAttention._v_up_proj` | 1381 | — |

## Request Round-Trip Evidence

1. Schedule a decode row and emit its `SchedulerOutput` — `scheduler-output`.
2. Select the V1 worker, gather token IDs and positions, and build slots and
   attention metadata — `v1-runner-selection`, `scheduled-token-inputs`,
   `paged-slot-mapping`, `layer-attention-metadata`, `v1-forward-context`.
3. Resolve GLM5Next, choose each layer's KDA/MLA and MLP branches, and produce
   the final normalized hidden state — `glm5next-architecture`,
   `layer-kind-selection`, `layer-residual-forward`, `model-layer-loop`,
   `moe-forward`.
4. For a configured KDA layer, mutate convolution and recurrent state and
   return its attention output — `kda-cache-spec`, `kda-state-shape`,
   `kda-conv-state-write`, `kda-recurrent-state-update`, `kda-triton-kernel`.
5. For a configured sparse MLA layer, update tail/pool state, score pooled
   history, select pool IDs, and expand token positions — `pooled-index-cache-spec`,
   `tail-ring-cache-spec`, `pooled-index-record-width`, `indexer-projection`,
   `kpool-decode-update`, `kpool-paged-scorer`, `kpool-topk-dispatch`,
   `indexer-token-expansion`, `pool-to-token-formula`.
6. Write the new main MLA record, remap selected token positions, run sparse
   attention, and project its latent result back to the hidden width —
   `topk-ready-event`, `topk-side-stream-wait`, `topk-consumer-wait`,
   `physical-cache-row-map`, `mla-backend-config`, `mla-cache-spec`,
   `mla-cache-write`, `mla-cache-write-op`, `mla-decode-query-absorption`,
   `mla-prefill-decode-dispatch`, `flashmla-backend-class`,
   `flashmla-nope-eligibility`, `flashmla-bf16-decode`, `flashmla-kernel`,
   `mla-value-up-projection`.
7. Compute vocabulary logits from the selected hidden row — `model-logits`,
   `runner-logits`.
8. Sample the next token, update request state, and return `ModelRunnerOutput`
   to the worker caller — `ordinary-sampler`, `sample-token-phase`,
   `runner-output`, `worker-token-return`.

## Reproduction Commands

From the repository root:

```bash
rg -n "Glm5NextForCausalLM|index_kpool|index_topk|linear_attention" \
  external-repos/vllm-02aa10539f15/vllm/models/glm5next
rg -n "FlashMLASparseBackend|flash_mla_sparse_fwd|convert_logical_to_physical_topk" \
  external-repos/vllm-02aa10539f15/vllm/v1/attention/backends/mla
```

## Verification Boundary

All findings are static reads of the pinned checkout. The code path is not a
runtime trace; the SM90/BF16/NoPE-512 configuration is a selected eligible
branch, not an observed default. The actual checkpoint config, backend
autoselection, cache allocation, and kernel execution must be verified in the
deployment environment.
