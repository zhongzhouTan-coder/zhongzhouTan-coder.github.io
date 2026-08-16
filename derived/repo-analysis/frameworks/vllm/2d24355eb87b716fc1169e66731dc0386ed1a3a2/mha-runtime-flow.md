# vLLM MHA Runtime Flow Evidence

## Reader Contract

- **Audience:** readers who know Transformer attention math and basic PyTorch,
  but do not yet know how vLLM connects a model definition to paged serving
  state and hardware kernels.
- **Question:** where does decoder multi-head attention actually happen in
  vLLM, and how does one scheduled token move from QKV projection through the
  paged KV cache to the output projection?
- **Mental model:** the model produces Q/K/V, the generic attention layer is a
  serving adapter, the model runner injects dynamic cache metadata, and the
  selected backend performs the cache write plus attention kernel.
- **Lifecycle:** backend selection and KV-cache binding happen during model/cache
  initialization; block tables, slot mappings, and attention metadata are built
  once per engine iteration; QKV projection, cache update, attention, and output
  projection happen once per layer during the model forward.
- **Limits:** static reading of decoder self-attention at the pinned commit; the
  Llama module and FlashAttention backend are concrete examples, not universal
  choices for every model or platform. No GPU execution or performance
  measurement was performed. A freshness check on 2026-08-16 found relevant
  upstream changes, but the repository's 14-day revision interval deferred a
  new snapshot until 2026-08-27.

## Rich-Content Plan

| Reader question | Evidence | Representation | Teaching job |
|---|---|---|---|
| Which component owns each part of attention? | Model module, generic layer, model runner, backend | Lifecycle table | Separate model math from serving state and hardware execution. |
| How does one decode token change state? | Slot mapping, forward context, cache scatter, FlashAttention call | Numbered worked trace | Make the cache write/read ordering and handoff boundaries explicit. |
| How do MHA, GQA, and MQA share one path? | Head-count calculations and QKV tensor shapes | Compact comparison table | Show that `num_heads` versus `num_kv_heads` changes geometry, not the control path. |

No synthesized diagram is needed: the lifecycle table and state-changing trace
answer the reader questions without duplicating the prose.

## Required Code Evidence

| Docs page | Finding | File | Symbol | Start | End |
|---|---|---|---|---:|---:|
| `docs/frameworks/vllm/vllm-mha-code-path.md` | model-head-layout | `vllm/model_executor/models/llama.py` | `LlamaAttention.__init__` | 123 | 160 |
| `docs/frameworks/vllm/vllm-mha-code-path.md` | model-attention-assembly | `vllm/model_executor/models/llama.py` | `LlamaAttention.__init__` | 163 | 220 |
| `docs/frameworks/vllm/vllm-mha-code-path.md` | model-forward | `vllm/model_executor/models/llama.py` | `LlamaAttention.forward` | 222 | 232 |
| `docs/frameworks/vllm/vllm-mha-code-path.md` | backend-request | `vllm/model_executor/layers/attention/attention.py` | `Attention.__init__` | 317 | 350 |
| `docs/frameworks/vllm/vllm-mha-code-path.md` | backend-implementation | `vllm/model_executor/layers/attention/attention.py` | `Attention.__init__` | 409 | 423 |
| `docs/frameworks/vllm/vllm-mha-code-path.md` | selector-constraints | `vllm/v1/attention/selector.py` | `get_attn_backend` | 153 | 188 |
| `docs/frameworks/vllm/vllm-mha-code-path.md` | selector-platform-dispatch | `vllm/v1/attention/selector.py` | `_cached_get_attn_backend` | 193 | 223 |
| `docs/frameworks/vllm/vllm-mha-code-path.md` | kv-cache-bind | `vllm/v1/worker/gpu_model_runner.py` | `GPUModelRunner.initialize_kv_cache_tensors` | 7660 | 7713 |
| `docs/frameworks/vllm/vllm-mha-code-path.md` | slot-mapping | `vllm/v1/worker/gpu_model_runner.py` | `GPUModelRunner._get_slot_mappings` | 4225 | 4255 |
| `docs/frameworks/vllm/vllm-mha-code-path.md` | iteration-metadata | `vllm/v1/worker/gpu_model_runner.py` | `GPUModelRunner.execute_model` | 4478 | 4503 |
| `docs/frameworks/vllm/vllm-mha-code-path.md` | forward-context | `vllm/v1/worker/gpu_model_runner.py` | `GPUModelRunner.execute_model` | 4537 | 4560 |
| `docs/frameworks/vllm/vllm-mha-code-path.md` | layer-shape-adapter | `vllm/model_executor/layers/attention/attention.py` | `Attention.forward` | 514 | 529 |
| `docs/frameworks/vllm/vllm-mha-code-path.md` | layer-custom-op-dispatch | `vllm/model_executor/layers/attention/attention.py` | `Attention.forward` | 530 | 570 |
| `docs/frameworks/vllm/vllm-mha-code-path.md` | forward-context-lookup | `vllm/model_executor/layers/attention/attention.py` | `get_attention_context` | 695 | 713 |
| `docs/frameworks/vllm/vllm-mha-code-path.md` | kv-update-dispatch | `vllm/model_executor/layers/attention/attention.py` | `unified_kv_cache_update` | 716 | 739 |
| `docs/frameworks/vllm/vllm-mha-code-path.md` | backend-forward-dispatch | `vllm/model_executor/layers/attention/attention.py` | `unified_attention_with_output` | 758 | 787 |
| `docs/frameworks/vllm/vllm-mha-code-path.md` | flash-separate-cache-update | `vllm/v1/attention/backends/flash_attn.py` | `FlashAttentionBackend.forward_includes_kv_cache_update` | 77 | 97 |
| `docs/frameworks/vllm/vllm-mha-code-path.md` | flash-cache-write | `vllm/v1/attention/backends/flash_attn.py` | `FlashAttentionImpl.do_kv_cache_update` | 1179 | 1213 |
| `docs/frameworks/vllm/vllm-mha-code-path.md` | flash-cache-view | `vllm/v1/attention/backends/flash_attn.py` | `FlashAttentionImpl.forward` | 982 | 1013 |
| `docs/frameworks/vllm/vllm-mha-code-path.md` | flash-kernel-call | `vllm/v1/attention/backends/flash_attn.py` | `FlashAttentionImpl.forward` | 1122 | 1148 |

## Runtime Flow Evidence

1. Load-time geometry and dispatch — `model-head-layout`,
   `model-attention-assembly`, `backend-request`, `selector-constraints`,
   `selector-platform-dispatch`, `backend-implementation`.
2. Cache materialization — `kv-cache-bind`.
3. Per-iteration coordination — `slot-mapping`, `iteration-metadata`,
   `forward-context`.
4. Per-layer model computation — `model-forward`, `layer-shape-adapter`,
   `layer-custom-op-dispatch`.
5. Cache state transition — `forward-context-lookup`, `kv-update-dispatch`,
   `flash-separate-cache-update`, `flash-cache-write`.
6. Backend attention and result handoff — `backend-forward-dispatch`,
   `flash-cache-view`, `flash-kernel-call`, `model-forward`.

## Link Completion

- [x] Every Required Code Evidence row has a matching code link.
- [x] Every runtime-flow step names at least one declared finding.
- [x] The first meaningful occurrence of every major implementation symbol is linked.
- [x] Symbol maps link important implementation types and operations.
- [x] Repeated mentions and generic variables remain ordinary inline code.
- [x] `./scripts/lint-docs.sh` passes.
