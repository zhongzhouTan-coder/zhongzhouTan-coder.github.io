---
kind: repository-analysis
repository_id: github:vllm-project/vllm@fce64bc5c2c939a4521bfb9b462c8c74c1c13893
commit: fce64bc5c2c939a4521bfb9b462c8c74c1c13893
source_record: raw/frameworks/serving/vllm/vllm-codebase--github-fce64bc5c2c9.md
generated: 2026-10-20
---

# vLLM Codebase Important Files

## Important Files

These files cover model registration, the selected NVIDIA implementation, its
separate recurrent and paged-cache state, and the V2 worker boundary.

- `vllm/model_executor/models/registry.py` — Maps the architecture string to Qwen4Exp.
- `vllm/model_executor/models/config.py` — Enforces Qwen4Exp parallelism and feature gates.
- `vllm/models/qwen4_exp/__init__.py` — Selects the NVIDIA or AMD model implementation.
- `vllm/models/qwen4_exp/nvidia/model.py` — Builds the Qwen4Exp decoder, PLE layers, MoE, and state specs.
- `vllm/models/qwen4_exp/nvidia/model_state.py` — Prepares per-request n-gram context for PLE.
- `vllm/models/qwen4_exp/nvidia/ngram_embedding.py` — Hashes n-grams and runs the NVIDIA PLE lookup.
- `vllm/models/qwen4_exp/common/ngram_embedding.py` — Defines device and pinned-host PLE storage.
- `vllm/models/qwen4_exp/nvidia/ple_layer.py` — Applies PLE and updates its short-convolution state.
- `vllm/models/qwen4_exp/nvidia/hyperconnection.py` — Mixes the model's gated residual streams.
- `vllm/models/qwen4_exp/nvidia/qsa.py` — Defines the QSA backend, main KV spec, and attention handoff.
- `vllm/models/qwen4_exp/nvidia/indexer_qsa.py` — Writes QSA side caches and selects context positions.
- `vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py` — Scores compressed groups and expands selected indices.
- `vllm/models/qwen4_exp/nvidia/ops/qsa.py` — Implements sparse paged attention over main K/V.
- `vllm/models/qwen4_exp/common/qsa_cache.py` — Defines QSA cache metadata and logical-to-physical slot maps.
- `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` — Implements the CUDA GDN path.
- `vllm/v1/kv_cache_interface.py` — Defines Mamba state specifications.
- `vllm/v1/worker/gpu_worker.py` — Selects the V1 or V2 GPU model runner.
- `vllm/v1/worker/gpu/model_runner.py` — Prepares scheduled rows, executes the model, and returns async output.
- `vllm/v1/worker/gpu/model_states/__init__.py` — Resolves model-specific state classes.
- `vllm/v1/worker/gpu/block_table.py` — Converts logical token positions to physical cache slots.
- `vllm/model_executor/models/qwen3_vl.py` — Supplies the inherited conditional-generation logits handoff.

## Required Code Evidence

| Docs page | Finding | File | Symbol | Start | End |
|---|---|---|---|---:|---:|
| `docs/frameworks/serving/vllm/qwen3-8-flash-next-gpu-inference-path/index.md` | architecture-registration | `vllm/model_executor/models/registry.py` | `Qwen4ExpForConditionalGeneration` registry entry | 609 | 612 |
| `docs/frameworks/serving/vllm/qwen3-8-flash-next-gpu-inference-path/index.md` | platform-dispatch | `vllm/models/qwen4_exp/__init__.py` | `__getattr__` | 28 | 48 |
| `docs/frameworks/serving/vllm/qwen3-8-flash-next-gpu-inference-path/index.md` | qwen4exp-config-gates | `vllm/model_executor/models/config.py` | `Qwen4ExpForConditionalGenerationConfig.verify_and_update_config` | 981 | 1032 |
| `docs/frameworks/serving/vllm/qwen3-8-flash-next-gpu-inference-path/index.md` | v2-runner-selection | `vllm/v1/worker/gpu_worker.py` | `GPUWorker._make_model_runner` | 567 | 583 |
| `docs/frameworks/serving/vllm/qwen3-8-flash-next-gpu-inference-path/index.md` | model-state-init | `vllm/v1/worker/gpu/model_runner.py` | `GPUModelRunner.__init__` | 438 | 440 |
| `docs/frameworks/serving/vllm/qwen3-8-flash-next-gpu-inference-path/index.md` | model-state-resolution | `vllm/v1/worker/gpu/model_states/__init__.py` | `resolve_model_state_cls` | 28 | 34 |
| `docs/frameworks/serving/vllm/qwen3-8-flash-next-gpu-inference-path/index.md` | qwen4exp-state-hook | `vllm/models/qwen4_exp/nvidia/model.py` | `Qwen4ExpForConditionalGeneration.get_model_state_cls` | 958 | 961 |
| `docs/frameworks/serving/vllm/qwen3-8-flash-next-gpu-inference-path/index.md` | scheduled-coordinates | `vllm/v1/worker/gpu/model_runner.py` | `GPUModelRunner.prepare_inputs` | 1446 | 1488 |
| `docs/frameworks/serving/vllm/qwen3-8-flash-next-gpu-inference-path/index.md` | physical-slot-map | `vllm/v1/worker/gpu/block_table.py` | `_compute_slot_mappings_kernel` | 381 | 413 |
| `docs/frameworks/serving/vllm/qwen3-8-flash-next-gpu-inference-path/index.md` | ngram-context-input | `vllm/models/qwen4_exp/nvidia/model_state.py` | `Qwen4ExpModelState.prepare_inputs` | 67 | 113 |
| `docs/frameworks/serving/vllm/qwen3-8-flash-next-gpu-inference-path/index.md` | model-input-assembly | `vllm/v1/worker/gpu/model_runner.py` | `GPUModelRunner.execute_model` model input map | 1984 | 1992 |
| `docs/frameworks/serving/vllm/qwen3-8-flash-next-gpu-inference-path/index.md` | decoder-layer-build | `vllm/models/qwen4_exp/nvidia/model.py` | `Qwen4ExpDecoderLayer.__init__` | 191 | 279 |
| `docs/frameworks/serving/vllm/qwen3-8-flash-next-gpu-inference-path/index.md` | decoder-layer-forward | `vllm/models/qwen4_exp/nvidia/model.py` | `Qwen4ExpDecoderLayer.forward` | 281 | 354 |
| `docs/frameworks/serving/vllm/qwen3-8-flash-next-gpu-inference-path/index.md` | decoder-layer-loop | `vllm/models/qwen4_exp/nvidia/model.py` | `Qwen4ExpModel.forward` layer loop and final mixer | 554 | 640 |
| `docs/frameworks/serving/vllm/qwen3-8-flash-next-gpu-inference-path/index.md` | ngram-id-generation | `vllm/models/qwen4_exp/nvidia/ngram_embedding.py` | `Qwen4ExpNGramEmbedding.compute_ngram_ids` | 293 | 366 |
| `docs/frameworks/serving/vllm/qwen3-8-flash-next-gpu-inference-path/index.md` | ple-state-spec | `vllm/models/qwen4_exp/nvidia/ple_layer.py` | `Qwen4ExpPLELayer.get_state_shape` | 189 | 220 |
| `docs/frameworks/serving/vllm/qwen3-8-flash-next-gpu-inference-path/index.md` | ple-runtime | `vllm/models/qwen4_exp/nvidia/ple_layer.py` | `Qwen4ExpPLELayer.forward` | 438 | 484 |
| `docs/frameworks/serving/vllm/qwen3-8-flash-next-gpu-inference-path/index.md` | mamba-state-specs | `vllm/models/qwen4_exp/nvidia/model.py` | `Qwen4ExpForCausalLM.get_mamba_specs_from_config` | 880 | 898 |
| `docs/frameworks/serving/vllm/qwen3-8-flash-next-gpu-inference-path/index.md` | gdn-cuda-path | `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` | `QwenGatedDeltaNetAttention.forward_cuda` | 1117 | 1193 |
| `docs/frameworks/serving/vllm/qwen3-8-flash-next-gpu-inference-path/index.md` | qsa-backend-contract | `vllm/models/qwen4_exp/nvidia/qsa.py` | `Qwen4ExpQSAFlashAttentionBackend` | 146 | 203 |
| `docs/frameworks/serving/vllm/qwen3-8-flash-next-gpu-inference-path/index.md` | qsa-init-gates | `vllm/models/qwen4_exp/nvidia/qsa.py` | `Qwen4ExpQSAAttention.__init__` | 342 | 385 |
| `docs/frameworks/serving/vllm/qwen3-8-flash-next-gpu-inference-path/index.md` | qsa-main-kv-spec | `vllm/models/qwen4_exp/nvidia/qsa.py` | `Qwen4ExpQSAAttention.get_kv_cache_spec` | 552 | 560 |
| `docs/frameworks/serving/vllm/qwen3-8-flash-next-gpu-inference-path/index.md` | qsa-side-cache-specs | `vllm/models/qwen4_exp/common/qsa_cache.py` | `QSAKeyStateCache.get_kv_cache_spec` and `QSACompressedKeyCache.get_kv_cache_spec` | 842 | 875 |
| `docs/frameworks/serving/vllm/qwen3-8-flash-next-gpu-inference-path/index.md` | qsa-compressed-slot-map | `vllm/models/qwen4_exp/common/qsa_cache.py` | `compressed_qsa_slot_mapping` | 161 | 190 |
| `docs/frameworks/serving/vllm/qwen3-8-flash-next-gpu-inference-path/index.md` | qsa-circular-slot-map | `vllm/models/qwen4_exp/common/qsa_cache.py` | `circular_qsa_slot_mapping` | 108 | 159 |
| `docs/frameworks/serving/vllm/qwen3-8-flash-next-gpu-inference-path/index.md` | qsa-fused-prepare | `vllm/models/qwen4_exp/nvidia/indexer_qsa.py` | `QSAIndexer.forward` fused `qsa_prepare` branch | 310 | 350 |
| `docs/frameworks/serving/vllm/qwen3-8-flash-next-gpu-inference-path/index.md` | qsa-side-cache-writes | `vllm/models/qwen4_exp/nvidia/indexer_qsa.py` | `QSAIndexer.forward` cache stores | 392 | 407 |
| `docs/frameworks/serving/vllm/qwen3-8-flash-next-gpu-inference-path/index.md` | qsa-decode-selection | `vllm/models/qwen4_exp/nvidia/indexer_qsa.py` | `QSAIndexer.forward` decode selection and expansion | 424 | 474 |
| `docs/frameworks/serving/vllm/qwen3-8-flash-next-gpu-inference-path/index.md` | qsa-topk-hardware-gate | `vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py` | `_topk` | 477 | 505 |
| `docs/frameworks/serving/vllm/qwen3-8-flash-next-gpu-inference-path/index.md` | qsa-selection-buffer | `vllm/models/qwen4_exp/nvidia/qsa.py` | `Qwen4ExpQSAAttention.topk_indices_buffer` | 519 | 531 |
| `docs/frameworks/serving/vllm/qwen3-8-flash-next-gpu-inference-path/index.md` | qsa-main-kv-update | `vllm/models/qwen4_exp/nvidia/qsa.py` | `Qwen4ExpQSAAttention._run_qsa` | 563 | 627 |
| `docs/frameworks/serving/vllm/qwen3-8-flash-next-gpu-inference-path/index.md` | qsa-sparse-call | `vllm/models/qwen4_exp/nvidia/qsa.py` | `Qwen4ExpQSAFlashAttentionImpl.forward_qsa` | 264 | 334 |
| `docs/frameworks/serving/vllm/qwen3-8-flash-next-gpu-inference-path/index.md` | qsa-sparse-kernel | `vllm/models/qwen4_exp/nvidia/ops/qsa.py` | `qsa_sparse_paged_attention` | 637 | 735 |
| `docs/frameworks/serving/vllm/qwen3-8-flash-next-gpu-inference-path/index.md` | qwen4exp-wrapper-forward | `vllm/models/qwen4_exp/nvidia/model.py` | `Qwen4ExpForConditionalGeneration.forward` | 1089 | 1110 |
| `docs/frameworks/serving/vllm/qwen3-8-flash-next-gpu-inference-path/index.md` | qwen4exp-raw-token-requirement | `vllm/models/qwen4_exp/nvidia/model.py` | `Qwen4ExpForConditionalGeneration.requires_raw_input_tokens` | 937 | 945 |
| `docs/frameworks/serving/vllm/qwen3-8-flash-next-gpu-inference-path/index.md` | qsa-indexer-dtype | `vllm/models/qwen4_exp/nvidia/indexer_qsa.py` | `QSAIndexer.__init__` | 141 | 164 |
| `docs/frameworks/serving/vllm/qwen3-8-flash-next-gpu-inference-path/index.md` | qsa-packed-index-contract | `vllm/models/qwen4_exp/nvidia/indexer_qsa.py` | `QSAIndexer.forward` packed output contract | 234 | 254 |
| `docs/frameworks/serving/vllm/qwen3-8-flash-next-gpu-inference-path/index.md` | qsa-index-width | `vllm/models/qwen4_exp/nvidia/indexer_qsa.py` | `QSAIndexer.output_width` and `packed_output_width` | 184 | 197 |
| `docs/frameworks/serving/vllm/qwen3-8-flash-next-gpu-inference-path/index.md` | qsa-fused-prepare-gate | `vllm/models/qwen4_exp/nvidia/qsa.py` | `Qwen4ExpQSAAttention.use_fused_qsa_prepare` | 475 | 513 |
| `docs/frameworks/serving/vllm/qwen3-8-flash-next-gpu-inference-path/index.md` | ple-storage-selection | `vllm/models/qwen4_exp/nvidia/ngram_embedding.py` | `Qwen4ExpNGramEmbedding.__init__` | 216 | 226 |
| `docs/frameworks/serving/vllm/qwen3-8-flash-next-gpu-inference-path/index.md` | ple-host-prefetch | `vllm/models/qwen4_exp/common/ngram_embedding.py` | `Qwen4ExpPLEPinnedHostEmbedding.start_prefetch` and `finalize_prefetch` | 739 | 817 |
| `docs/frameworks/serving/vllm/qwen3-8-flash-next-gpu-inference-path/index.md` | hyperconnection-mix | `vllm/models/qwen4_exp/nvidia/hyperconnection.py` | `GatedResidual.combine_and_mix` | 177 | 212 |
| `docs/frameworks/serving/vllm/qwen3-8-flash-next-gpu-inference-path/index.md` | logits-delegation | `vllm/model_executor/models/qwen3_vl.py` | `Qwen3VLForConditionalGeneration.compute_logits` | 3221 | 3225 |
| `docs/frameworks/serving/vllm/qwen3-8-flash-next-gpu-inference-path/index.md` | qwen4exp-lm-head | `vllm/models/qwen4_exp/nvidia/model.py` | `Qwen4ExpForCausalLM.compute_logits` | 902 | 904 |
| `docs/frameworks/serving/vllm/qwen3-8-flash-next-gpu-inference-path/index.md` | v2-logits-sampling | `vllm/v1/worker/gpu/model_runner.py` | `GPUModelRunner.sample` | 1615 | 1665 |
| `docs/frameworks/serving/vllm/qwen3-8-flash-next-gpu-inference-path/index.md` | async-output-creation | `vllm/v1/worker/gpu/model_runner.py` | `GPUModelRunner.sample_tokens` | 2210 | 2218 |
| `docs/frameworks/serving/vllm/qwen3-8-flash-next-gpu-inference-path/index.md` | async-output-return | `vllm/v1/worker/gpu/model_runner.py` | `GPUModelRunner.sample_tokens` | 2311 | — |

## Request Round-Trip Evidence

1. Load-time model registration and CUDA implementation selection — `architecture-registration`, `platform-dispatch`.
2. V2 worker selection and Qwen4Exp state initialization — `v2-runner-selection`, `model-state-init`, `model-state-resolution`, `qwen4exp-state-hook`.
3. Scheduled rows, logical positions, physical KV slots, and PLE context — `scheduled-coordinates`, `physical-slot-map`, `ngram-context-input`, `model-input-assembly`.
4. Qwen4Exp wrapper, configured layer branches, PLE, and GDN state — `decoder-layer-build`, `decoder-layer-forward`, `decoder-layer-loop`, `ngram-id-generation`, `ple-state-spec`, `ple-runtime`, `mamba-state-specs`, `gdn-cuda-path`.
5. QSA side-cache writes, compressed-group selection, main K/V read, and attention output — `qsa-backend-contract`, `qsa-init-gates`, `qsa-main-kv-spec`, `qsa-side-cache-specs`, `qsa-compressed-slot-map`, `qsa-circular-slot-map`, `qsa-fused-prepare`, `qsa-side-cache-writes`, `qsa-decode-selection`, `qsa-topk-hardware-gate`, `qsa-selection-buffer`, `qsa-main-kv-update`, `qsa-sparse-call`, `qsa-sparse-kernel`.
6. Final hidden state, logits, sampling, and async worker return — `qwen4exp-wrapper-forward`, `logits-delegation`, `qwen4exp-lm-head`, `v2-logits-sampling`, `async-output-creation`, `async-output-return`.

## Reproduction Commands

No repository-wide counts, performance results, or memory totals are claimed.

## Verification Boundary

The model configuration was not loaded from the Hub, and no checkpoint,
CUDA kernel, or GPU runtime was executed. Layer indices, QSA compression ratio,
indexer budget, TP size, and cache byte totals therefore remain unspecified.
The CUDA SM100 branch is a static code-selection example, not a measured run.
