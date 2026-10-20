---
kind: repository-analysis
repository_id: github:vllm-project/vllm@fce64bc5c2c939a4521bfb9b462c8c74c1c13893
commit: fce64bc5c2c939a4521bfb9b462c8c74c1c13893
source_record: raw/frameworks/serving/vllm/vllm-codebase--github-fce64bc5c2c9.md
generated: 2026-10-20
---

# Qwen3.8-Flash-Next GPU Inference Path: Evidence Notes

## Scope

This note records the NVIDIA path in the pinned `vllm-project/vllm` checkout
from V2 GPU-worker model-state initialization through one text decode step,
Qwen4Exp layers, cache updates, logits, sampling, and `AsyncOutput`. The trace
assumes `use_v2_model_runner`, `PP=1`, BF16 activations, no speculative decoding,
no context parallelism, no DBO or microbatching, a resident PLE table, and a
CUDA SM100 device. Four requests each schedule one decode token. These are
illustrative static assumptions, not an executed launch configuration.

The Qwen3.8-Flash-Next model page currently advertises the architecture
`Qwen4ExpForConditionalGeneration` and `model_type: qwen4_exp`; this identity is
external model metadata. The implementation evidence below is pinned to the
vLLM commit in this note. The Hub config was not downloaded into `raw/`.

## Stable repository findings

- `vllm/model_executor/models/registry.py` maps the conditional-generation
  architecture name to `vllm.models.qwen4_exp`.
- `vllm/models/qwen4_exp/__init__.py` rejects XPU/TPU, imports the AMD model on
  ROCm, and otherwise imports the NVIDIA implementation.
- `GPUWorker._make_model_runner` selects `vllm.v1.worker.gpu.model_runner` when
  `use_v2_model_runner` is true; otherwise it selects the separate V1 runner in
  `vllm.v1.worker.gpu_model_runner`.
- The V2 runner calls `init_model_state` after model loading. The resolver uses
  the model's `get_model_state_cls` hook, which returns `Qwen4ExpModelState`.
- `Qwen4ExpModelState.prepare_inputs` gathers each active request's prior
  `ngram_size - 1` token IDs from request state and supplies `query_start_loc`
  plus `ngram_context` to the model. Negative history positions are filled with
  the configured EOS ID.
- `GPUModelRunner.prepare_inputs` constructs `query_start_loc` from scheduled
  token counts and positions from the request's computed-token count plus the
  within-query offset. `prepare_attn` gathers block-table rows and computes
  per-token physical slot mappings.
- `Qwen4ExpDecoderLayer` optionally adds PLE for configured `ple_layer_ids`.
  `layer_types` choose GDN linear attention or full attention; when QSA indexer
  heads are configured, the latter uses `Qwen4ExpQSAAttention`. Every decoder
  layer constructs a sparse MoE block and gated hyperconnection modules.
- Qwen4Exp provides two separate `MambaSpec` contracts: the GDN recurrent state
  and the PLE short-convolution state. PLE marks its state `tp_replicated=True`.
- QSA uses a `FullAttentionSpec` for its main paged K/V cache. Its raw-key side
  cache is a `CircularBufferSpec`; its compressed-key side cache is an
  `MLAAttentionSpec` with `tokens_per_state=compress_ratio`.
- `compressed_qsa_slot_mapping` maps logical positions to compressed group
  positions with `floor(position / compress_ratio)` and emits a cache slot only
  at complete-group boundaries. `circular_qsa_slot_mapping` maps raw keys into
  the request's ring buffer.
- `QSAIndexer.forward` updates raw/compressed side caches, selects compressed
  blocks for decode, and expands them to request-relative token positions. The
  packed selection buffer has a final valid-entry-count column; that count is
  not a token index.
- The QSA attention owner writes the current main K/V through either fused
  `qsa_prepare` or a separate `do_kv_cache_update`, then calls
  `qsa_sparse_paged_attention` with selected logical indices, a block table,
  and `token_to_req`. The custom kernel reads the full-precision main K/V cache;
  compressed keys are used for selection, not as the attention values.
- On CUDA, QSA's decode selector uses `cooperative_topk` only when the row count
  is at most 64, the logits row stride is divisible by four, and device
  capability is at least 9.0 but not capability family 12.0. Otherwise it uses
  `persistent_topk`.
- The V2 runner selects logits rows, calls the model's logits path, samples, and
  returns `AsyncOutput`; it does not return detokenized text.

## State and coordinate contracts

| Object | Unit or layout | Writer and reader | Lifetime |
|---|---|---|---|
| `positions` | Logical token position in a request | V2 input preparation; rotary embedding and attention metadata | Current scheduled rows |
| Main K/V `slot_mapping` | Physical KV slot derived from request block table, logical position, and KV block size | GPU block-table kernel; QSA K/V writer | Persistent attention history until request/cache release |
| Raw QSA key cache | One BF16 index key per raw token in an open compression group; optional RoPE positions for MRoPE | QSA indexer writes and group compressor reads | Circular ring sized for the open group plus configured speculative rows |
| Compressed QSA key cache | One normalized index key per complete token group; BF16 by default, FP8 if configured | QSA compressor writes; paged top-k scorer reads | Compressed history for the request |
| QSA `logical_indices` | Request-relative token positions, padded with `-1`, plus a trailing valid-entry count | QSA block-index expansion writes; sparse attention kernel reads | One forward step |
| GDN state | Config-derived recurrent and convolution tensors in a `MambaSpec` | GDN CUDA core reads and updates | Per request and GDN layer |
| PLE state | Config-derived short-convolution tensors in a replicated `MambaSpec` | PLE short-convolution op reads and updates | Per request and configured PLE layer |
| PLE context/table | `int32` prior-token context plus hashed n-gram IDs; embedding table is a model weight | Qwen4Exp model state prepares context; PLE embedding reads the table | Context rebuilt per step; weights resident or optionally pinned-host |

No total bytes-per-token estimate is made: the model config's dimensions,
compression ratio, indexer budget, KV block size, and cache dtype overrides were
not read from a checkpoint.

## Branch contracts and limitations

- Qwen4Exp's config rejects DBO or `ubatch_size > 1` when PLE or QSA is active.
  PLE requires pipeline parallel size 1. QSA requires BF16 model dtype, causal
  attention, no DCP/PCP, and a BF16 or FP8-e4m3 main KV cache.
- QSA binds `QWEN4_EXP_QSA_TRITON`; it is not a generic automatic switch to
  FlashAttention paged attention. FlashAttention varlen availability is checked
  by the implementation, but the sparse K/V read is the QSA Triton kernel.
- The main cache's `auto` dtype resolves to BF16 for the assumed BF16 model.
  The indexer side-cache dtype defaults to BF16 and can be configured to FP8.
  Main-cache FP8 uses uint8 storage and device-capability-dependent e4m3 views.
- `engram_config.cpu_offload` switches PLE weights from the device embedding to
  a pinned-host/UVA implementation. The host implementation starts lookups on
  a side stream and finalizes before the PLE layer consumes them; this note's
  selected branch keeps the table device-resident.
- Speculative decoding is absent from the selected step. The config accepts
  Qwen4Exp's native MTP and linear n-gram proposer methods; QSA can reuse its
  step-zero index buffer on later MTP steps.
- No checkpoint configuration was loaded, no GPU kernels were run, and no
  throughput, memory fit, correctness, or runtime backend selection was
  measured. Layer indices and architecture field values remain conditional on
  the linked public model config.
