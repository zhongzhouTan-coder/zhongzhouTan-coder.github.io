---
kind: repository-analysis
repository_id: github:vllm-project/vllm-ascend@8405ff749d05aee320923d7d1dd9ad24c9246d4c
commit: 8405ff749d05aee320923d7d1dd9ad24c9246d4c
source_record: raw/frameworks/serving/vllm-ascend/vllm-ascend-codebase--github-8405ff749d05.md
generated: 2026-10-27
---

# GLM-5.3-Flash A3 and A5 Evidence

The selected boundary is the V1 runner's scheduled batch through synchronous
sampled-token output. The example uses one ordinary decode row, no speculative
decoding or context parallelism, a NoPE sparse layer, and C8 main KV. Hardware
and checkpoint claims are static source statements, not measured execution.

## Required Code Evidence

| Docs page | Finding | File | Symbol | Start | End |
|---|---|---|---|---:|---:|
| `docs/frameworks/serving/vllm-ascend/glm-5.3-flash-a3-a5-inference/index.md` | deployment-guide | `docs/source/tutorials/models/GLM5.3-Flash.md` | Checkpoints and devices | 21 | — |
| `docs/frameworks/serving/vllm-ascend/glm-5.3-flash-a3-a5-inference/index.md` | hardware-profiles | `vllm_ascend/device/hardware_profile.py` | A3 and A5 profiles | 296 | — |
| `docs/frameworks/serving/vllm-ascend/glm-5.3-flash-a3-a5-inference/index.md` | registry | `vllm_ascend/models/__init__.py` | GLM registrations | 89 | — |
| `docs/frameworks/serving/vllm-ascend/glm-5.3-flash-a3-a5-inference/index.md` | example-config | `tests/e2e/pull_request/four_card/glm53flash_assets/config.json` | Nine-layer fixture | 17 | — |
| `docs/frameworks/serving/vllm-ascend/glm-5.3-flash-a3-a5-inference/index.md` | layer-select | `vllm_ascend/models/glm5next/model.py` | Decoder construction | 300 | — |
| `docs/frameworks/serving/vllm-ascend/glm-5.3-flash-a3-a5-inference/index.md` | batch-positions | `vllm_ascend/worker/v1/model_runner.py` | _prepare_inputs | 1399 | — |
| `docs/frameworks/serving/vllm-ascend/glm-5.3-flash-a3-a5-inference/index.md` | slot-mapping | `vllm_ascend/worker/v1/model_runner.py` | compute_slot_mapping | 1721 | — |
| `docs/frameworks/serving/vllm-ascend/glm-5.3-flash-a3-a5-inference/index.md` | forward-handoff | `vllm_ascend/worker/v1/model_runner.py` | execute_model forward context | 2662 | — |
| `docs/frameworks/serving/vllm-ascend/glm-5.3-flash-a3-a5-inference/index.md` | decoder-loop | `vllm_ascend/models/glm5next/model.py` | Glm5NextModel.forward | 680 | — |
| `docs/frameworks/serving/vllm-ascend/glm-5.3-flash-a3-a5-inference/index.md` | mhc-attention | `vllm_ascend/models/glm5next/model.py` | Decoder mHC and attention | 435 | — |
| `docs/frameworks/serving/vllm-ascend/glm-5.3-flash-a3-a5-inference/index.md` | kda-projection | `vllm_ascend/models/glm5next/kda.py` | Glm5NextLinearAttention.forward | 397 | — |
| `docs/frameworks/serving/vllm-ascend/glm-5.3-flash-a3-a5-inference/index.md` | kda-decode | `vllm_ascend/models/glm5next/kda.py` | Decode convolution and recurrent dispatch | 584 | — |
| `docs/frameworks/serving/vllm-ascend/glm-5.3-flash-a3-a5-inference/index.md` | kda-kernel | `vllm_ascend/ops/kda.py` | run_recurrent_kda | 16 | — |
| `docs/frameworks/serving/vllm-ascend/glm-5.3-flash-a3-a5-inference/index.md` | kda-prefill | `vllm_ascend/models/glm5next/ops/kda.py` | chunk_kda | 56 | — |
| `docs/frameworks/serving/vllm-ascend/glm-5.3-flash-a3-a5-inference/index.md` | sfa-selection | `vllm_ascend/platform.py` | _select_attn_backend_cls | 299 | — |
| `docs/frameworks/serving/vllm-ascend/glm-5.3-flash-a3-a5-inference/index.md` | index-cache-owners | `vllm_ascend/models/glm5next/attention.py` | Indexer cache construction | 104 | — |
| `docs/frameworks/serving/vllm-ascend/glm-5.3-flash-a3-a5-inference/index.md` | compressed-slot | `vllm_ascend/models/glm5next/kv_cache.py` | format_indexer_kpool_slot_mapping | 45 | — |
| `docs/frameworks/serving/vllm-ascend/glm-5.3-flash-a3-a5-inference/index.md` | indexer-gate | `vllm_ascend/attention/indexer_kpool.py` | Glm5NextKPoolIndexerBackend.__init__ | 574 | — |
| `docs/frameworks/serving/vllm-ascend/glm-5.3-flash-a3-a5-inference/index.md` | indexer-ops | `vllm_ascend/attention/indexer_kpool.py` | CANN key pool and indexer | 709 | — |
| `docs/frameworks/serving/vllm-ascend/glm-5.3-flash-a3-a5-inference/index.md` | pooled-write | `vllm_ascend/attention/indexer_kpool.py` | _forward_cann | 785 | — |
| `docs/frameworks/serving/vllm-ascend/glm-5.3-flash-a3-a5-inference/index.md` | token-selection | `vllm_ascend/attention/indexer_kpool.py` | pool_key_indexer and pack | 839 | — |
| `docs/frameworks/serving/vllm-ascend/glm-5.3-flash-a3-a5-inference/index.md` | tail-pack | `vllm_ascend/ops/triton/glm5_next_cann_keypool.py` | pack tail token positions | 110 | — |
| `docs/frameworks/serving/vllm-ascend/glm-5.3-flash-a3-a5-inference/index.md` | sfa-producer-consumer | `vllm_ascend/attention/sfa_v1.py` | Indexer and sparse attention calls | 2059 | — |
| `docs/frameworks/serving/vllm-ascend/glm-5.3-flash-a3-a5-inference/index.md` | c8-packing | `vllm_ascend/attention/utils.py` | get_sfa_qsfa_packed_head_dim | 184 | — |
| `docs/frameworks/serving/vllm-ascend/glm-5.3-flash-a3-a5-inference/index.md` | c8-write | `vllm_ascend/attention/sfa_v1.py` | _store_parallel_kv | 1715 | — |
| `docs/frameworks/serving/vllm-ascend/glm-5.3-flash-a3-a5-inference/index.md` | quant-sparse-read | `vllm_ascend/device/device_op.py` | _execute_kv_quant_sparse_flash_attention_vllm | 594 | — |
| `docs/frameworks/serving/vllm-ascend/glm-5.3-flash-a3-a5-inference/index.md` | float-branches | `vllm_ascend/attention/sfa_v1.py` | sparse_mla | 170 | — |
| `docs/frameworks/serving/vllm-ascend/glm-5.3-flash-a3-a5-inference/index.md` | float-gate | `vllm_ascend/attention/sfa_v1.py` | SparseMLAMetadataState.__init__ | 254 | — |
| `docs/frameworks/serving/vllm-ascend/glm-5.3-flash-a3-a5-inference/index.md` | moe-router | `vllm_ascend/models/glm5next/model.py` | Glm5NextMoE.forward | 242 | — |
| `docs/frameworks/serving/vllm-ascend/glm-5.3-flash-a3-a5-inference/index.md` | moe-policy | `vllm_ascend/ascend_forward_context.py` | Device-specific MoE selectors | 333 | — |
| `docs/frameworks/serving/vllm-ascend/glm-5.3-flash-a3-a5-inference/index.md` | logits | `vllm_ascend/worker/v1/model_runner.py` | Hidden row to logits | 2744 | — |
| `docs/frameworks/serving/vllm-ascend/glm-5.3-flash-a3-a5-inference/index.md` | sample-return | `vllm_ascend/worker/v1/model_runner.py` | sample_tokens | 2874 | — |
| `docs/frameworks/serving/vllm-ascend/glm-5.3-flash-a3-a5-inference/index.md` | output | `vllm_ascend/worker/v1/model_runner.py` | ModelRunnerOutput | 3019 | — |
| `docs/frameworks/serving/vllm-ascend/glm-5.3-flash-a3-a5-inference/index.md` | snapshot-slot-mapping | `vllm_ascend/worker/v1/model_runner.py` | compute_slot_mapping | 1721 | 1725 |
| `docs/frameworks/serving/vllm-ascend/glm-5.3-flash-a3-a5-inference/index.md` | snapshot-decoder-loop | `vllm_ascend/models/glm5next/model.py` | Glm5NextModel.forward | 701 | 702 |
| `docs/frameworks/serving/vllm-ascend/glm-5.3-flash-a3-a5-inference/index.md` | snapshot-kda-recurrent | `vllm_ascend/ops/kda.py` | run_recurrent_kda | 33 | 56 |
| `docs/frameworks/serving/vllm-ascend/glm-5.3-flash-a3-a5-inference/index.md` | snapshot-main-kv-write | `vllm_ascend/attention/sfa_v1.py` | _store_parallel_kv | 1744 | 1752 |
| `docs/frameworks/serving/vllm-ascend/glm-5.3-flash-a3-a5-inference/index.md` | snapshot-pool-selection | `vllm_ascend/attention/indexer_kpool.py` | _forward_cann | 839 | 857 |
| `docs/frameworks/serving/vllm-ascend/glm-5.3-flash-a3-a5-inference/index.md` | snapshot-c8-kernel | `vllm_ascend/device/device_op.py` | _execute_kv_quant_sparse_flash_attention_vllm | 611 | 632 |
| `docs/frameworks/serving/vllm-ascend/glm-5.3-flash-a3-a5-inference/index.md` | snapshot-moe-router | `vllm_ascend/models/glm5next/model.py` | Glm5NextMoE.forward | 254 | 267 |
| `docs/frameworks/serving/vllm-ascend/glm-5.3-flash-a3-a5-inference/index.md` | snapshot-sampling | `vllm_ascend/worker/v1/model_runner.py` | sample_tokens | 2922 | 2923 |
| `docs/frameworks/serving/vllm-ascend/glm-5.3-flash-a3-a5-inference/index.md` | snapshot-output | `vllm_ascend/worker/v1/model_runner.py` | ModelRunnerOutput | 3019 | 3030 |
| `docs/frameworks/serving/vllm-ascend/glm-5.3-flash-a3-a5-inference/index.md` | snapshot-float-gate | `vllm_ascend/attention/sfa_v1.py` | SparseMLAMetadataState.__init__ | 259 | 262 |

## Round Trip

Batch metadata and slots -> model forward -> KDA state or sparse KV and index
state -> attention output -> mHC and MoE -> normalized hidden row -> logits ->
sampler -> synchronous ModelRunnerOutput. API detokenization is out of scope.

## Reproduction

The clean sparse checkout was read at the pinned commit with `git show` for
files absent from the sparse working tree and normal file reads for attention.
The fixture's four-token pool and 512-wide latent cache are test configuration
values; the byte arithmetic in the page is derived from these values, not a
hardware measurement.
