---
kind: repository-analysis
repository_id: github:vllm-project/vllm@02aa10539f15c8c73aa7fd4a5ee0f691cbf88dcb
commit: 02aa10539f15c8c73aa7fd4a5ee0f691cbf88dcb
source_record: raw/frameworks/serving/vllm/vllm-codebase--github-02aa10539f15.md
generated: 2026-10-10
---

# vllm Codebase Important Files

## Reader Contract

- **Audience:** serving engineers familiar with transformer inference and paged KV caches.
- **Question:** what exact state does each of the five selected checkpoints write, retain, index, and read on GPU?
- **Mental model:** the model chooses state classes and ownership; the cache specs choose logical pages; the selected CUDA backend determines physical packing and kernel reads.
- **Offline/load time:** the checkpoint architecture resolves to a model class, and config plus flags choose cache specs and backends.
- **Runtime:** each scheduled token updates durable pages or recurrent state, may produce a sparse selection, and returns a hidden state to the decoder.
- **Limits:** checkpoint configs are external to the pinned code checkout; the pages cross-check them against released configuration files. No device execution was performed. Live validation must establish physical bytes, kernel selection, and prefix restoration.

## Representation Plan

| Reader question | Evidence | Representation | Teaching job |
|---|---|---|---|
| Which objects persist versus exist for one step? | Model cache specs and forward paths | Per-model state tables | Prevent sparse top-k from being mistaken for eviction. |
| Where do GPU and NPU differ? | Pinned model implementations and plugin registry | Paired platform sections | Keep logical model state separate from backend layout. |
| What happens to one new token? | Model and attention forward paths | Numbered token traces | Connect projection, cache write, selection, read, and return. |

## Evidence Map

- `vllm/models/deepseek_v32/nvidia/model.py` — GLM-5.3 family model path
- `vllm/models/deepseek_v41/attention.py` — DeepSeek-V4.1 cache path
- `vllm/models/minimax_m3/nvidia/model.py` — MiniMax-M3 sparse attention
- `vllm/models/qwen4_exp/nvidia/qsa.py` — Qwen sparse attention
- `vllm/models/kimi_k3/nvidia/mla.py` — Kimi-K3 MLA cache

## Reproduction Commands

To revisit model-specific cache owners, run
`rg -n 'get_kv_cache_spec|CacheSpec|MambaSpec|IndexerCache' external-repos/vllm-02aa10539f15/vllm/models/{deepseek_v32,deepseek_v41,minimax_m3,qwen4_exp,kimi_k3}`.
Layer counts and dimensions in the docs come from the released checkpoint
configurations rather than code-tree counts.

## Required Code Evidence

| Docs page | Finding | File | Symbol | Start | End |
|---|---|---|---|---:|---:|
| `docs/frameworks/serving/model-kv-cache/qwen3.8-flash-next.md` | architecture-decoder | `vllm/models/qwen4_exp/nvidia/model.py` | `Qwen4ExpDecoderLayer` | 175 | — |
| `docs/frameworks/serving/model-kv-cache/qwen3.8-flash-next.md` | architecture-model | `vllm/models/qwen4_exp/nvidia/model.py` | `Qwen4ExpModel` | 366 | — |
| `docs/frameworks/serving/model-kv-cache/qwen3.8-flash-next.md` | architecture-qsa-indexer | `vllm/models/qwen4_exp/nvidia/indexer_qsa.py` | `QSAIndexer` | 90 | — |
| `docs/frameworks/serving/model-kv-cache/qwen3.8-flash-next.md` | architecture-ple | `vllm/models/qwen4_exp/nvidia/ple_layer.py` | `Qwen4ExpPLELayer` | 64 | — |
| `docs/frameworks/serving/model-kv-cache/kimi-k3.md` | architecture-decoder | `vllm/models/kimi_k3/nvidia/model.py` | `KimiDecoderLayer` | 857 | — |
| `docs/frameworks/serving/model-kv-cache/kimi-k3.md` | architecture-moe | `vllm/models/kimi_k3/nvidia/model.py` | `KimiMoE` | 540 | — |
| `docs/frameworks/serving/model-kv-cache/glm-5.3.md` | architecture-decoder | `vllm/models/deepseek_v32/nvidia/model.py` | `DeepseekV32DecoderLayer` | 57 | — |
| `docs/frameworks/serving/model-kv-cache/glm-5.3.md` | architecture-mla | `vllm/models/deepseek_v32/attention.py` | `DeepseekV32Attention` | 121 | — |
| `docs/frameworks/serving/model-kv-cache/glm-5.3.md` | architecture-indexer | `vllm/models/deepseek_v32/attention.py` | `DeepseekV32Indexer` | 50 | — |
| `docs/frameworks/serving/model-kv-cache/deepseek-v4.1-flash.md` | architecture-decoder | `vllm/models/deepseek_v41/nvidia/model.py` | `DeepseekV4DecoderLayer` | 283 | — |
| `docs/frameworks/serving/model-kv-cache/deepseek-v4.1-flash.md` | architecture-source-map | `vllm/models/deepseek_v41/attention.py` | `KV and index source mapping` | 305 | — |
| `docs/frameworks/serving/model-kv-cache/deepseek-v4.1-flash.md` | architecture-moe | `vllm/models/deepseek_v41/nvidia/model.py` | `DeepseekV4MoE` | 125 | — |
| `docs/frameworks/serving/model-kv-cache/minimax-m3.md` | architecture-decoder | `vllm/models/minimax_m3/nvidia/model.py` | `MiniMaxM3DecoderLayer` | 678 | — |
| `docs/frameworks/serving/model-kv-cache/minimax-m3.md` | architecture-moe | `vllm/models/minimax_m3/nvidia/model.py` | `MiniMaxM3MoE` | 195 | — |
| `docs/frameworks/serving/model-kv-cache/glm-5.3.md` | gpu-model | `vllm/models/deepseek_v32/nvidia/model.py` | `DeepseekV32Model` | 169 | — |
| `docs/frameworks/serving/model-kv-cache/glm-5.3.md` | gpu-topk | `vllm/models/deepseek_v32/nvidia/model.py` | `shared top-k buffer` | 190 | — |
| `docs/frameworks/serving/model-kv-cache/glm-5.3.md` | gpu-alias | `vllm/models/deepseek_v32/__init__.py` | `GlmMoeDsa alias` | 18 | — |
| `docs/frameworks/serving/model-kv-cache/deepseek-v4.1-flash.md` | gpu-model | `vllm/models/deepseek_v41/nvidia/model.py` | `DeepseekV41LLMForCausalLM` | 1537 | — |
| `docs/frameworks/serving/model-kv-cache/deepseek-v4.1-flash.md` | gpu-source-cache | `vllm/models/deepseek_v41/attention.py` | `compressed KV cache spec` | 1138 | — |
| `docs/frameworks/serving/model-kv-cache/deepseek-v4.1-flash.md` | gpu-index-cache | `vllm/models/deepseek_v41/attention.py` | `DeepseekV4IndexerCache` | 1175 | — |
| `docs/frameworks/serving/model-kv-cache/deepseek-v4.1-flash.md` | gpu-compressor-state | `vllm/models/deepseek_v41/compressor.py` | `CompressorStateCache` | 136 | — |
| `docs/frameworks/serving/model-kv-cache/minimax-m3.md` | gpu-model | `vllm/models/minimax_m3/nvidia/model.py` | `MiniMaxM3SparseAttention` | 392 | — |
| `docs/frameworks/serving/model-kv-cache/minimax-m3.md` | gpu-main-cache | `vllm/models/minimax_m3/nvidia/model.py` | `main KV cache spec` | 562 | — |
| `docs/frameworks/serving/model-kv-cache/minimax-m3.md` | gpu-index-cache | `vllm/models/minimax_m3/common/indexer.py` | `MiniMaxM3IndexerCache` | 102 | — |
| `docs/frameworks/serving/model-kv-cache/minimax-m3.md` | gpu-sparse-kernel | `vllm/models/minimax_m3/nvidia/sparse_attention_msa.py` | `MiniMaxM3SparseMSAImpl` | 185 | — |
| `docs/frameworks/serving/model-kv-cache/qwen3.8-flash-next.md` | gpu-layer | `vllm/models/qwen4_exp/nvidia/model.py` | `Qwen4ExpDecoderLayer` | 175 | — |
| `docs/frameworks/serving/model-kv-cache/qwen3.8-flash-next.md` | gpu-qsa | `vllm/models/qwen4_exp/nvidia/qsa.py` | `Qwen4ExpQSAAttention` | 333 | — |
| `docs/frameworks/serving/model-kv-cache/qwen3.8-flash-next.md` | gpu-main-spec | `vllm/models/qwen4_exp/nvidia/qsa.py` | `QSA main cache spec` | 541 | — |
| `docs/frameworks/serving/model-kv-cache/qwen3.8-flash-next.md` | gpu-raw-index | `vllm/models/qwen4_exp/common/qsa_cache.py` | `QSAKeyStateCache` | 811 | — |
| `docs/frameworks/serving/model-kv-cache/qwen3.8-flash-next.md` | gpu-compressed-index | `vllm/models/qwen4_exp/common/qsa_cache.py` | `QSACompressedKeyCache` | 865 | — |
| `docs/frameworks/serving/model-kv-cache/qwen3.8-flash-next.md` | gpu-hybrid-state | `vllm/models/qwen4_exp/nvidia/model_state.py` | `Qwen4ExpModelState` | 17 | — |
| `docs/frameworks/serving/model-kv-cache/kimi-k3.md` | gpu-model | `vllm/models/kimi_k3/nvidia/model.py` | `KimiLinearForCausalLM` | 1581 | — |
| `docs/frameworks/serving/model-kv-cache/kimi-k3.md` | gpu-mla | `vllm/models/kimi_k3/nvidia/mla.py` | `MultiHeadLatentAttention` | 123 | — |
| `docs/frameworks/serving/model-kv-cache/kimi-k3.md` | gpu-cache-spec | `vllm/models/kimi_k3/nvidia/mla.py` | `MLA cache spec` | 417 | — |
| `docs/frameworks/serving/model-kv-cache/kimi-k3.md` | gpu-kda | `vllm/models/kimi_k3/nvidia/kda.py` | `KimiK3DeltaAttention` | 544 | — |
| `docs/frameworks/serving/model-kv-cache/kimi-k3.md` | gpu-kda-spec | `vllm/models/kimi_k3/nvidia/kda.py` | `KDA MambaSpec` | 829 | — |

## Scope and verification

Five checkpoint pages consume this revision. The findings above are from static source reading; no model, GPU, or Ascend NPU was executed. Checkpoint dimensions in the pages were compared with the released configuration files, while physical layouts still need deployment validation.

## Model-specific findings

### GLM-5.3

- `vllm/models/deepseek_v32/__init__.py:18` aliases the GLM sparse-MLA architecture to the V3.2 implementation on CUDA. The alias makes the checkpoint architecture field a prerequisite for interpreting the runtime path.
- `vllm/models/deepseek_v32/nvidia/model.py:190` allocates one model-level `int32` top-k buffer, then passes it to decoder layers. This buffer is a current-batch selection product, separate from long-lived MLA and indexer caches.
- `vllm/models/glm5next/` describes GLM-5.3-Flash, including KDA. The similarly named directory is not evidence that the selected `zai-org/GLM-5.3` has KDA.

### DeepSeek-V4.1-Flash

- `vllm/models/deepseek_v41/attention.py:1138` returns no global compressed cache spec for consumer layers; only source layers own it. The spec records compression ratio via `tokens_per_state` and conditionally uses packed cache formats.
- `vllm/models/deepseek_v41/attention.py:1175` implements a distinct indexer-key cache, and `vllm/models/deepseek_v41/compressor.py:136` implements incomplete-group state. A report-level bytes/token number cannot describe all three objects.

### MiniMax-M3

- `vllm/models/minimax_m3/nvidia/model.py:392` combines QKV and index Q/K projection with the sparse attention owner. `vllm/models/minimax_m3/common/indexer.py:102` has its own cache owner.
- `vllm/models/minimax_m3/nvidia/sparse_attention_msa.py:185` is a selected-block read path. Sparse selection is not an eviction operation on the full main K/V history.

### Qwen3.8-Flash-Next

- `vllm/models/qwen4_exp/nvidia/model.py:175` chooses between recurrent and QSA layer behavior. `vllm/models/qwen4_exp/nvidia/model_state.py:17` integrates hybrid recurrent state with the runner.
- `vllm/models/qwen4_exp/nvidia/qsa.py:333` has explicit model/KV dtype and context-parallel gates. `vllm/models/qwen4_exp/common/qsa_cache.py:811` and `:865` separate raw/open-group and compressed index-key ownership.

### Kimi-K3

- `vllm/models/kimi_k3/nvidia/kda.py:829` exposes recurrent state through a `MambaSpec`, while `vllm/models/kimi_k3/nvidia/mla.py:417` declares paged MLA state. These are separate cache groups and need coherent prefix boundaries.
- The MLA cache spec's 656-byte `fp8_ds_mla` record is conditional; the brief's BF16 576-element payload is a logical comparison floor rather than a statement about this packed format.

## Reproduction of negative findings

Scoped reading used `rg --files external-repos/vllm-02aa10539f15/vllm/models` and `rg -n` on the model files named in the evidence table. No quantitative count of runtime allocations was made.
