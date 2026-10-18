---
kind: repository-analysis
repository_id: github:vllm-project/vllm@02aa10539f15c8c73aa7fd4a5ee0f691cbf88dcb
commit: 02aa10539f15c8c73aa7fd4a5ee0f691cbf88dcb
source_record: raw/frameworks/serving/vllm/vllm-codebase--github-02aa10539f15.md
generated: 2026-10-18
---

# vLLM DeepSeek-V4.1 GPU Inference Path

## Reader Contract

- **Audience:** vLLM serving engineers familiar with model registries, paged KV
  caches, and CUDA attention backends.
- **Question:** how does a scheduled DeepSeek-V4.1-Flash batch move through the
  NVIDIA vLLM worker to sparse attention, token sampling, and the worker output?
- **Mental model:** the checkpoint config defines cache and index ownership,
  the GPU runner binds those logical roles to a scheduled token batch, and the
  selected attention backend consumes the local sliding window plus sparse
  global KV.
- **Offline/load time:** the architecture and model type resolve to the V4.1
  wrapper and config; model construction creates cache owners and attention
  backends from checkpoint fields and runtime flags.
- **Runtime:** the runner prepares slot mappings and attention metadata; the
  decoder projects, compresses, indexes, attends, mixes residual streams, runs
  MoE, and returns hidden states for logits and sampling.
- **Limits:** this is static reading of a clean detached GitHub checkout at
  `02aa10539f15c8c73aa7fd4a5ee0f691cbf88dcb`. The checkpoint's actual
  `compress_ratios`, `kv_source_layer_ids`, and `index_source_layer_ids` were
  not loaded. No GPU, model weights, or runtime tests were used. The scoped
  upstream refresh is deferred until 2026-10-24.

## Representation Plan

| Reader question | Evidence | Representation | Teaching job |
|---|---|---|---|
| How does one scheduled batch reach the GPU model and return sampled IDs? | GPU runner, model wrapper, attention, and sampler call sites | Mermaid sequence plus numbered trace | Show the worker boundary, model handoffs, backend branch, and return |
| What changes for one ordinary decode token? | Runner, V4.1 attention, CUDA indexer, FlashMLA decode, and sampled-output call sites | Seven-step input/state/output trace with short code excerpts | Make the top-k buffer, cache reads, attention output, and token return followable without opening every file |
| Which layer owns each cache or sparse selection? | V4.1 attention source IDs and cache specs | Compact ownership table | Distinguish persistent SWA, compressed KV, indexer K, and current-step top-k |
| Where does the selected GPU backend change behavior? | Backend selector and NVIDIA implementations | Backend decision table | Make hardware, cache layout, and dependency gates explicit |

## Required Code Evidence

| Docs page | Finding | File | Symbol | Start | End |
|---|---|---|---:|---:|---:|
| `docs/frameworks/serving/deepseek/v4.1-gpu-inference-path/index.md` | hf-config | `vllm/transformers_utils/config.py` | `deepseek_v41` config mapping | 129 | — |
| `docs/frameworks/serving/deepseek/v4.1-gpu-inference-path/index.md` | architecture-registry | `vllm/model_executor/models/registry.py` | `DeepseekV41ForCausalLM` | 370 | — |
| `docs/frameworks/serving/deepseek/v4.1-gpu-inference-path/index.md` | platform-entry | `vllm/models/deepseek_v41/__init__.py` | `current_platform.is_rocm` branch | 9 | — |
| `docs/frameworks/serving/deepseek/v4.1-gpu-inference-path/index.md` | multimodal-wrapper | `vllm/models/deepseek_v41/nvidia/vl_model.py` | `DeepseekV41ForCausalLM` | 117 | — |
| `docs/frameworks/serving/deepseek/v4.1-gpu-inference-path/index.md` | image-embeddings | `vllm/models/deepseek_v41/nvidia/vl_model.py` | `_process_image_input` | 207 | — |
| `docs/frameworks/serving/deepseek/v4.1-gpu-inference-path/index.md` | runner-entry | `vllm/v1/worker/gpu/model_runner.py` | `GPUModelRunner.execute_model` | 1711 | — |
| `docs/frameworks/serving/deepseek/v4.1-gpu-inference-path/index.md` | model-state-selection | `vllm/v1/worker/gpu/model_states/__init__.py` | `resolve_model_state_cls` | 28 | — |
| `docs/frameworks/serving/deepseek/v4.1-gpu-inference-path/index.md` | v41-state-preparation | `vllm/models/deepseek_v41/nvidia/model_state.py` | `DeepseekV41ModelState.prepare_attn` | 308 | — |
| `docs/frameworks/serving/deepseek/v4.1-gpu-inference-path/index.md` | model-input-assembly | `vllm/v1/worker/gpu/model_runner.py` | `GPUModelRunner.execute_model` model inputs | 1949 | 1957 |
| `docs/frameworks/serving/deepseek/v4.1-gpu-inference-path/index.md` | runner-model-call | `vllm/v1/worker/gpu/model_runner.py` | `self.model(**model_inputs)` | 2034 | — |
| `docs/frameworks/serving/deepseek/v4.1-gpu-inference-path/index.md` | text-model | `vllm/models/deepseek_v41/nvidia/model.py` | `DeepseekV41LLMForCausalLM` | 1537 | — |
| `docs/frameworks/serving/deepseek/v4.1-gpu-inference-path/index.md` | text-layer-stack | `vllm/models/deepseek_v41/nvidia/model.py` | `DeepseekV4Model` layer construction | 765 | — |
| `docs/frameworks/serving/deepseek/v4.1-gpu-inference-path/index.md` | decoder-layer | `vllm/models/deepseek_v41/nvidia/model.py` | `DeepseekV4DecoderLayer.forward` | 455 | — |
| `docs/frameworks/serving/deepseek/v4.1-gpu-inference-path/index.md` | attention-topology | `vllm/models/deepseek_v41/attention.py` | `DeepseekV4Attention` source-layer setup | 304 | — |
| `docs/frameworks/serving/deepseek/v4.1-gpu-inference-path/index.md` | parallel-projections | `vllm/models/deepseek_v41/attention.py` | `_run_parallel_input_projections` | 932 | — |
| `docs/frameworks/serving/deepseek/v4.1-gpu-inference-path/index.md` | attention-overlap | `vllm/models/deepseek_v41/attention.py` | `_prepare_and_attn` | 832 | — |
| `docs/frameworks/serving/deepseek/v4.1-gpu-inference-path/index.md` | compressor-latent | `vllm/models/deepseek_v41/compressor.py` | `DeepseekCompressor.forward` | 250 | — |
| `docs/frameworks/serving/deepseek/v4.1-gpu-inference-path/index.md` | compressor-kernel | `vllm/models/deepseek_v41/common/ops/fused_compress_quant_cache.py` | `fused_save_compress_norm` | 22 | — |
| `docs/frameworks/serving/deepseek/v4.1-gpu-inference-path/index.md` | compressed-kv-write | `vllm/models/deepseek_v41/common/ops/fused_compress_quant_cache.py` | `rope_quant_insert` | 224 | — |
| `docs/frameworks/serving/deepseek/v4.1-gpu-inference-path/index.md` | indexer-k-store | `vllm/models/deepseek_v41/common/ops/indexer_k_store.py` | `indexer_k_norm_rope_store` | 29 | — |
| `docs/frameworks/serving/deepseek/v4.1-gpu-inference-path/index.md` | indexer-topk | `vllm/models/deepseek_v41/attention.py` | `DeepseekV4Indexer.forward` | 1470 | — |
| `docs/frameworks/serving/deepseek/v4.1-gpu-inference-path/index.md` | indexer-to-attention | `vllm/models/deepseek_v41/attention.py` | `DeepseekV4Attention._sparse_indexer_and_attn` | 986 | 1009 |
| `docs/frameworks/serving/deepseek/v4.1-gpu-inference-path/index.md` | cuda-topk-buffer | `vllm/model_executor/layers/sparse_attn_indexer.py` | CUDA top-k output into shared buffer | 769 | 780 |
| `docs/frameworks/serving/deepseek/v4.1-gpu-inference-path/index.md` | main-kv-spec | `vllm/models/deepseek_v41/attention.py` | `DeepseekV4Attention.get_kv_cache_spec` | 1138 | — |
| `docs/frameworks/serving/deepseek/v4.1-gpu-inference-path/index.md` | swa-kv-spec | `vllm/v1/attention/backends/mla/sparse_swa.py` | `DeepseekV4SWACache.get_kv_cache_spec` | 116 | — |
| `docs/frameworks/serving/deepseek/v4.1-gpu-inference-path/index.md` | cache-record-format | `vllm/models/deepseek_v41/attention.py` | V4.1 packed KV record widths | 512 | — |
| `docs/frameworks/serving/deepseek/v4.1-gpu-inference-path/index.md` | cache-family-switch | `vllm/models/deepseek_v41/attention.py` | `_use_v41_mxfp8_kv_record` | 128 | — |
| `docs/frameworks/serving/deepseek/v4.1-gpu-inference-path/index.md` | cache-dtype-resolver | `vllm/models/deepseek_v41/attention.py` | `_resolve_dsv4_kv_cache_dtype` | 132 | — |
| `docs/frameworks/serving/deepseek/v4.1-gpu-inference-path/index.md` | backend-selection | `vllm/models/deepseek_v41/nvidia/model.py` | `_select_dsv4_attn_cls` | 197 | — |
| `docs/frameworks/serving/deepseek/v4.1-gpu-inference-path/index.md` | flashmla-prefill | `vllm/models/deepseek_v41/nvidia/flashmla.py` | `_forward_prefill` | 248 | — |
| `docs/frameworks/serving/deepseek/v4.1-gpu-inference-path/index.md` | flashmla-decode | `vllm/models/deepseek_v41/nvidia/flashmla.py` | `_forward_decode` | 167 | — |
| `docs/frameworks/serving/deepseek/v4.1-gpu-inference-path/index.md` | flashmla-decode-call | `vllm/models/deepseek_v41/nvidia/flashmla.py` | `flash_mla_with_kvcache` | 230 | 246 |
| `docs/frameworks/serving/deepseek/v4.1-gpu-inference-path/index.md` | flashinfer-sm120 | `vllm/models/deepseek_v41/nvidia/flashinfer_sparse.py` | `DeepseekV4FlashInferSM120Attention` | 590 | — |
| `docs/frameworks/serving/deepseek/v4.1-gpu-inference-path/index.md` | mega-attention | `vllm/models/deepseek_v41/nvidia/flash_mla_mega_attn.py` | `DeepseekV4MegaAttnAttention.forward_mqa` | 240 | — |
| `docs/frameworks/serving/deepseek/v4.1-gpu-inference-path/index.md` | mega-cache-contract | `vllm/models/deepseek_v41/nvidia/flash_mla_mega_attn.py` | `packed_kv_cache_dtype` | 202 | — |
| `docs/frameworks/serving/deepseek/v4.1-gpu-inference-path/index.md` | sm120-cache-read | `vllm/models/deepseek_v41/nvidia/flashinfer_sparse.py` | SM120 FlashInfer decode cache arguments | 809 | — |
| `docs/frameworks/serving/deepseek/v4.1-gpu-inference-path/index.md` | fused-kv-insert | `csrc/libtorch_stable/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu` | `fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert` | 1134 | — |
| `docs/frameworks/serving/deepseek/v4.1-gpu-inference-path/index.md` | swa-replay-spec | `vllm/v1/kv_cache_interface.py` | `SlidingWindowMLASpec.prefix_replay_tokens` | 973 | — |
| `docs/frameworks/serving/deepseek/v4.1-gpu-inference-path/index.md` | replay-config-gates | `vllm/models/deepseek_v41/attention.py` | `swa_bounded_replay` configuration gates | 529 | — |
| `docs/frameworks/serving/deepseek/v4.1-gpu-inference-path/index.md` | replay-batch | `vllm/models/deepseek_v41/nvidia/model_state.py` | `_prepare_replay_batch` | 438 | — |
| `docs/frameworks/serving/deepseek/v4.1-gpu-inference-path/index.md` | hc-collapse | `vllm/models/deepseek_v41/nvidia/model.py` | `hc_collapse_triton` call | 1070 | — |
| `docs/frameworks/serving/deepseek/v4.1-gpu-inference-path/index.md` | logits | `vllm/models/deepseek_v41/nvidia/model.py` | `DeepseekV41LLMForCausalLM.compute_logits` | 1609 | — |
| `docs/frameworks/serving/deepseek/v4.1-gpu-inference-path/index.md` | sampler | `vllm/v1/worker/gpu/model_runner.py` | `GPUModelRunner.sample` | 1607 | — |
| `docs/frameworks/serving/deepseek/v4.1-gpu-inference-path/index.md` | async-output-assembly | `vllm/v1/worker/gpu/model_runner.py` | `AsyncOutput` construction | 2174 | 2182 |
| `docs/frameworks/serving/deepseek/v4.1-gpu-inference-path/index.md` | output-postprocess | `vllm/v1/worker/gpu/model_runner.py` | `postprocess_sampled` | 2207 | — |
| `docs/frameworks/serving/deepseek/v4.1-gpu-inference-path/index.md` | kv-connector-postforward | `vllm/v1/worker/gpu/model_runner.py` | `kv_connector.post_forward` | 2266 | — |
| `docs/frameworks/serving/deepseek/v4.1-gpu-inference-path/index.md` | worker-return | `vllm/v1/worker/gpu/model_runner.py` | `GPUModelRunner.sample_tokens` async return | 2270 | — |

## Request Round Trip Evidence

1. **Scheduled worker boundary:** `runner-entry`.
2. **Batch and model-specific metadata:** `model-state-selection`,
   `v41-state-preparation`.
3. **Model handoff and layer execution:** `model-input-assembly`,
   `runner-model-call`, `text-model`,
   `text-layer-stack`, `decoder-layer`.
4. **Attention state and computation:** `attention-topology`,
   `parallel-projections`, `attention-overlap`, `compressor-latent`,
   `compressor-kernel`, `compressed-kv-write`, `indexer-k-store`,
   `indexer-topk`, `indexer-to-attention`, `cuda-topk-buffer`, `main-kv-spec`,
   `swa-kv-spec`.
5. **Backend result:** `backend-selection`, `flashmla-prefill`,
   `flashmla-decode`, `flashmla-decode-call`, `flashinfer-sm120`,
   `mega-attention`, `fused-kv-insert`.
6. **Hidden-state and token return:** `hc-collapse`, `logits`, `sampler`,
   `async-output-assembly`, `output-postprocess`, `kv-connector-postforward`,
   `worker-return`.

## Evidence Map

### Model identity and worker entry

- The HF model type is mapped to `DeepseekV41Config`, and the architecture
  registry selects `vllm.models.deepseek_v41.DeepseekV41ForCausalLM`.
- The non-ROCm package branch imports the NVIDIA vision-language wrapper. It
  creates an optional vision/aligner path and forwards to the text model.
- `GPUModelRunner.execute_model` receives `SchedulerOutput`, updates request
  state, prepares input rows and cache mappings, calls the custom model state
  for attention metadata, and invokes the model through eager or graph execution.
- `GPUModelRunner.sample_tokens` consumes saved hidden states, computes logits,
  samples, postprocesses request state, and returns `AsyncOutput`/worker output.

### Text layers, compression, and sparse indexing

- `DeepseekV4Model` builds `config.num_hidden_layers` instances of
  `DeepseekV4DecoderLayer`. The layer forward applies mHC around attention and
  the routed/shared MoE FFN.
- `compress_ratios` accepts 0, 1, or 2: 0 is SWA-only; 1 writes one compressed
  entry per token without a pooling gate; 2 pools token pairs and uses
  compressor state for incomplete groups.
- `kv_source_layer_ids` own compressed main KV and its compressor. The latest
  KV source is read by consumer layers. `index_source_layer_ids` own sparse
  index updates; non-owning index sources reuse the latest KV-source K cache.
- The indexer K comes from the compressor latent (`wk(latent)`), followed by
  K RMSNorm, RoPE, quantization, and a paged store. Indexer queries and weights
  feed the shared top-k buffer; short contexts can select all candidates
  without scoring.
- SWA, main compressed KV, indexer K, and the ratio-2 compressor ring state
  are separate logical cache objects/specs. The V4.1 code does not implement
  the V4-only compression ratios 4 and 128 in this compressor.

### NVIDIA backend and cache gates

- `_select_dsv4_attn_cls` routes SM12 to the V4.1 FlashInfer implementation,
  chooses FlashMLA Mega on eligible SM100 configurations, and otherwise uses
  FlashMLA sparse attention. Explicit generic FlashInfer sparse enums are
  rejected in favor of the V4.1-specific backend.
- FlashMLA prefill gathers/dequantizes compressed KV and SWA in chunks and
  merges selected global indices with local-window indices. Decode maps
  top-k entries to paged storage offsets and calls the cache-backed sparse
  attention path.
- FlashInfer SM120 requires a matching sparse-MLA specialization at model
  initialization. Mega attention requires the SM100 kernel extension and a
  compatible local-head/TP topology.
- With generic `cache_dtype` auto/fp8, the packed `fp8_ds_mla` record is 528 B
  on SM100 (V4.1 block-scaled FP8) but 584 B on SM120 (V4 mixed FP8/BF16).
  Mega instead prefers `nvfp4_ds_mla`: its compressed main-KV record is 288 B,
  while its layer-local SWA record stays 528 B FP8. SM100 FlashMLA uses 528 B
  for both. These are per-state record widths; ratio-2 compression and page
  alignment change the total bytes per input token. Explicit SM100 FlashInfer
  uses plain BF16 or per-tensor FP8 rows instead of the packed DS-MLA layout.
- The packed CUDA query/KV insertion operator requires SM80+. On SM90+ it
  enables programmatic stream serialization. The exact Q transformations
  differ for backends that consume unnormalized/un-RoPE'd queries themselves.

### Prefix reuse and the CED boundary

- With `swa_bounded_replay`, `SlidingWindowMLASpec` marks SWA non-prefix-cacheable
  and requests replay of the window. `DeepseekV41ModelState` pads replayed-token
  writes into cacheable groups, prepares a trailing-window sub-batch, and
  supplies replay metadata. `DeepseekV4Model` starts this replay after the last
  KV-source layer when its runner/config gates allow it.
- The V4.1 code path constructs one text-model layer stack and the wrapper's
  explicit encoder is the image ViT. No explicit causal-text encoder or CED
  stage was found in the scoped V4.1 model package. This is a scoped static
  negative finding, not proof that all CED behavior is unsupported; the
  checkpoint config was not loaded and implementation may be elsewhere.

## Reproduction Commands

Run from the repository root:

```bash
rg -n 'DeepseekV41ForCausalLM|deepseek_v41' \
  external-repos/vllm-02aa10539f15/vllm/model_executor/models/registry.py \
  external-repos/vllm-02aa10539f15/vllm/transformers_utils/config.py
rg -n 'compress_ratios|kv_source_layer_ids|index_source_layer_ids' \
  external-repos/vllm-02aa10539f15/vllm/models/deepseek_v41/attention.py
rg -n 'def execute_model|model_state.prepare_attn|self.model\\(\\*\\*model_inputs\\)|def sample_tokens' \
  external-repos/vllm-02aa10539f15/vllm/v1/worker/gpu/model_runner.py
rg -n 'fused_save_compress_norm|rope_quant_insert' \
  external-repos/vllm-02aa10539f15/vllm/models/deepseek_v41/common/ops/fused_compress_quant_cache.py
rg -n -i 'ced|causal.?encoder.?decoder|causal.?encoder' \
  external-repos/vllm-02aa10539f15/vllm/models/deepseek_v41
```

## Scope and Verification

- Repository: `https://github.com/vllm-project/vllm`, provider `github`,
  checkout origin `https://github.com/vllm-project/vllm`, detached ref,
  clean state, commit `02aa10539f15c8c73aa7fd4a5ee0f691cbf88dcb`.
- Inspected: 2026-10-17. The scoped `worktree.py sync` result is `defer` against
  upstream `237d99b4aff67ecfedacca7acb0ee93f299fa54e`; refresh becomes eligible
  on 2026-10-24. The newer upstream candidate was not inspected.
- Static code reading only. No model checkpoint config was loaded; no GPU,
  FlashMLA/FlashInfer kernel, or runtime behavior was executed.
