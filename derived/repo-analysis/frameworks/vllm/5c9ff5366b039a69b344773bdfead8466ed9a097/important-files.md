---
kind: repository-analysis
repository_id: github:vllm-project/vllm@5c9ff5366b039a69b344773bdfead8466ed9a097
commit: 5c9ff5366b039a69b344773bdfead8466ed9a097
source_record: raw/frameworks/vllm-codebase--github-5c9ff5366b03.md
generated: 2026-08-18
---

# GLM-5.2 Upstream vLLM Inference-Path Evidence

Consuming page: `docs/frameworks/vllm/glm-5.2-inference-path.md`

## Reader Contract

- **Audience:** vLLM users and developers who know the transformer inference
  loop but have not followed sparse Multi-head Latent Attention (MLA) through
  the V1 runtime.
- **Question:** how does one OpenAI-compatible GLM-5.2 request reach the
  upstream NVIDIA GPU kernels, and how does the sampled token return?
- **Mental model:** vLLM schedules token rows and paged-cache slots; the GLM
  model shell turns rows into latent attention and expert work; the selected
  sparse backend maps indexer-selected history positions to GPU cache pages.
- **Offline/load time:** checkpoint architecture selects
  `GlmMoeDsaForCausalLM`; configuration fields construct sparse indexers and
  the CUDA platform validates one attention backend.
- **Runtime:** the scheduler builds a mixed token batch, the GPU runner builds
  addressing metadata, each owning layer refreshes shared top-k indices,
  sparse MLA and MoE transform the rows, and sampling returns token IDs.
- **Limits:** clean static reading only. No GLM-5.2 checkpoint, HTTP server, or
  NVIDIA device was executed. Exact model dimensions, index pattern, top-k,
  quantization, and selected backend remain checkpoint/configuration dependent.

## Representation Plan

| Reader question | Evidence | Representation | Teaching job |
|---|---|---|---|
| Which process or subsystem owns the request at each point, including the return path? | API, `AsyncLLM`, `EngineCore`, runner, model, output code | Mermaid sequence diagram plus numbered trace | Make asynchronous handoffs, GPU execution, and cleanup visible in one round trip. |
| Why is there no single universal “GLM backend”? | CUDA priorities, backend validation, backend-specific constraints | Platform decision table | Separate model identity from hardware- and configuration-driven kernel selection. |
| What state changes inside sparse MLA? | MLA wrapper, indexer, paged cache, sparse kernels | Worked decode-token trace and exact state table | Distinguish latent KV, indexer K, top-k indices, physical cache slots, and attention output. |
| Where does MoE branch and reconverge? | DeepSeek-family MoE and generic runner | Compact prose with one call-chain table | Show routing, optional dispatch, routed/shared expert compute, combine, and reduction without pretending one quantized kernel is universal. |

## Required Code Evidence

| Docs page | Finding | File | Symbol | Start | End |
|---|---|---|---|---:|---:|
| `docs/frameworks/vllm/glm-5.2-inference-path.md` | api-admission | `vllm/entrypoints/openai/chat_completion/serving.py` | `OpenAIServingChat._create_chat_completion` | 241 | 396 |
| `docs/frameworks/vllm/glm-5.2-inference-path.md` | async-admission | `vllm/v1/engine/async_llm.py` | `AsyncLLM.generate` | 550 | 624 |
| `docs/frameworks/vllm/glm-5.2-inference-path.md` | engine-step | `vllm/v1/engine/core.py` | `EngineCore.step` | 583 | 613 |
| `docs/frameworks/vllm/glm-5.2-inference-path.md` | scheduler-budget | `vllm/v1/core/sched/scheduler.py` | `Scheduler.schedule` token model | 476 | 487 |
| `docs/frameworks/vllm/glm-5.2-inference-path.md` | scheduler-kv | `vllm/v1/core/sched/scheduler.py` | running-request KV allocation and preemption | 628 | 701 |
| `docs/frameworks/vllm/glm-5.2-inference-path.md` | gpu-forward | `vllm/v1/worker/gpu_model_runner.py` | metadata construction, model forward, and logits | 4494 | 4655 |
| `docs/frameworks/vllm/glm-5.2-inference-path.md` | gpu-sampling | `vllm/v1/worker/gpu_model_runner.py` | `GPUModelRunner.sample_tokens` | 4673 | 4713 |
| `docs/frameworks/vllm/glm-5.2-inference-path.md` | runner-output | `vllm/v1/worker/gpu_model_runner.py` | `ModelRunnerOutput` assembly | 4887 | 4913 |
| `docs/frameworks/vllm/glm-5.2-inference-path.md` | model-registration | `vllm/model_executor/models/registry.py` | `GlmMoeDsaForCausalLM` registry entry | 118 | — |
| `docs/frameworks/vllm/glm-5.2-inference-path.md` | model-shell | `vllm/model_executor/models/deepseek_v2.py` | `DeepseekV2ForCausalLM` construction, forward, and logits | 1794 | 1904 |
| `docs/frameworks/vllm/glm-5.2-inference-path.md` | glm-alias | `vllm/model_executor/models/deepseek_v2.py` | `GlmMoeDsaForCausalLM` | 1931 | 1932 |
| `docs/frameworks/vllm/glm-5.2-inference-path.md` | decoder-stack | `vllm/model_executor/models/deepseek_v2.py` | `DeepseekV2Model.forward` | 1427 | 1504 |
| `docs/frameworks/vllm/glm-5.2-inference-path.md` | decoder-layer | `vllm/model_executor/models/deepseek_v2.py` | `DeepseekV2DecoderLayer.forward` | 1286 | 1357 |
| `docs/frameworks/vllm/glm-5.2-inference-path.md` | sparse-construction | `vllm/model_executor/models/deepseek_v2.py` | index pattern and MLA wrapper construction | 1080 | 1179 |
| `docs/frameworks/vllm/glm-5.2-inference-path.md` | shared-index-buffer | `vllm/model_executor/models/deepseek_v2.py` | model-wide top-k buffer and layer construction | 1373 | 1402 |
| `docs/frameworks/vllm/glm-5.2-inference-path.md` | mla-wrapper | `vllm/model_executor/layers/mla.py` | `MultiHeadLatentAttentionWrapper.forward` | 150 | 226 |
| `docs/frameworks/vllm/glm-5.2-inference-path.md` | backend-selection | `vllm/v1/attention/selector.py` | `get_attn_backend` | 105 | 226 |
| `docs/frameworks/vllm/glm-5.2-inference-path.md` | backend-validation | `vllm/v1/attention/backend.py` | `AttentionBackend.validate_configuration` | 367 | 452 |
| `docs/frameworks/vllm/glm-5.2-inference-path.md` | cuda-priority | `vllm/platforms/cuda.py` | CUDA MLA backend priorities | 83 | 143 |
| `docs/frameworks/vllm/glm-5.2-inference-path.md` | cuda-choice | `vllm/platforms/cuda.py` | CUDA candidate validation and selection | 363 | 490 |
| `docs/frameworks/vllm/glm-5.2-inference-path.md` | indexer-projection | `vllm/model_executor/models/deepseek_v2.py` | `Indexer` construction and forward | 633 | 808 |
| `docs/frameworks/vllm/glm-5.2-inference-path.md` | indexer-kernels | `vllm/model_executor/layers/sparse_attn_indexer.py` | paged indexer cache, scoring, and top-k | 295 | 689 |
| `docs/frameworks/vllm/glm-5.2-inference-path.md` | sparse-mla-dispatch | `vllm/model_executor/layers/attention/mla_attention.py` | MHA/MQA split and sparse backend call | 780 | 980 |
| `docs/frameworks/vllm/glm-5.2-inference-path.md` | hopper-backend | `vllm/v1/attention/backends/mla/flashattn_mla_sparse.py` | Hopper validation and sparse decode | 33 | 104 |
| `docs/frameworks/vllm/glm-5.2-inference-path.md` | hopper-kernel | `vllm/v1/attention/backends/mla/flashattn_mla_sparse.py` | request-index conversion and FlashAttention call | 210 | 260 |
| `docs/frameworks/vllm/glm-5.2-inference-path.md` | blackwell-flashinfer | `vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py` | Blackwell FlashInfer sparse decode | 380 | 469 |
| `docs/frameworks/vllm/glm-5.2-inference-path.md` | blackwell-flashmla | `vllm/v1/attention/backends/mla/flashmla_sparse.py` | FlashMLA BF16 sparse kernel and dispatch | 802 | 869 |
| `docs/frameworks/vllm/glm-5.2-inference-path.md` | moe-model | `vllm/model_executor/models/deepseek_v2.py` | routed/shared MoE construction and forward | 279 | 416 |
| `docs/frameworks/vllm/glm-5.2-inference-path.md` | moe-selection | `vllm/model_executor/layers/fused_moe/runner/moe_runner.py` | modular/monolithic routing and expert invocation | 570 | 619 |
| `docs/frameworks/vllm/glm-5.2-inference-path.md` | moe-distribution | `vllm/model_executor/layers/fused_moe/runner/moe_runner.py` | dispatch, expert execution, and combine | 781 | 884 |
| `docs/frameworks/vllm/glm-5.2-inference-path.md` | scheduler-return | `vllm/v1/core/sched/scheduler.py` | sampled-token update and `EngineCoreOutput` creation | 1733 | 2036 |
| `docs/frameworks/vllm/glm-5.2-inference-path.md` | scheduler-cleanup | `vllm/v1/core/sched/scheduler.py` | request and KV-block cleanup | 2388 | 2442 |
| `docs/frameworks/vllm/glm-5.2-inference-path.md` | output-processing | `vllm/v1/engine/output_processor.py` | detokenization, stop checks, queue emission, frontend cleanup | 598 | 724 |
| `docs/frameworks/vllm/glm-5.2-inference-path.md` | response-stream | `vllm/entrypoints/openai/chat_completion/serving.py` | delta parsing and SSE serialization | 603 | 795 |
| `docs/frameworks/vllm/glm-5.2-inference-path.md` | stream-finish | `vllm/entrypoints/openai/chat_completion/serving.py` | streaming error and `[DONE]` emission | 879 | 886 |

## Static Findings

- The GLM class is a zero-override subclass of the DeepSeek-V2-family causal-LM
  shell. Behavior comes from checkpoint configuration and shared vLLM
  components, not a GLM-specific forward method.
- `index_topk` makes the attention sparse. `index_topk_pattern` or
  `index_topk_freq` decides which layers own an indexer; all layers receive one
  model-wide top-k buffer, so a skip layer consumes indices left by a preceding
  owner layer.
- The indexer has its own quantized paged K cache. Its output is an `int32`
  matrix of history positions; the selected sparse attention backend converts
  those positions to the physical main-MLA cache layout.
- CUDA backend selection is conditional. Hopper, datacenter Blackwell, and
  consumer Blackwell do not share one default sparse-MLA implementation.
- Long-prefill and decode execution can diverge: the common MLA wrapper may use
  dense or masked-MHA prefill when valid, while decode uses the selected sparse
  MQA kernel.
- MoE execution is also conditional: modular methods select experts before the
  expert kernel; monolithic quantized methods receive router logits and route
  internally. Parallel configuration decides dispatch/combine collectives.

## Verification Boundary

The checkout was clean at the pinned commit. Findings were established by
static code reading only. No model weights or configuration were downloaded,
so this evidence does not independently prove GLM-5.2's exact dimensions,
`index_topk`, index-sharing pattern, quantization, or supported GPU. No HTTP
request, CUDA graph, sparse-attention kernel, MoE kernel, sampler, or parser was
executed.
