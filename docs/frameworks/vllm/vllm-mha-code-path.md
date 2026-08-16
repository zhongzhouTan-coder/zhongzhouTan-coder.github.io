---
title: "vLLM MHA Code Path: From QKV to Paged KV Cache"
summary: "A code-reading trace of decoder multi-head attention in vLLM, from model-side QKV projection and RoPE through runtime metadata, paged KV writes, backend dispatch, and the output projection."
layout: default
confidence: medium
sources:
  - raw/frameworks/vllm-codebase--github-2d24355eb87b.md
  - derived/repo-analysis/frameworks/vllm/2d24355eb87b716fc1169e66731dc0386ed1a3a2/mha-runtime-flow.md
updated: 2026-08-16
code_links: strict
code_evidence: strict
---

# vLLM MHA Code Path: From QKV to Paged KV Cache

**Repository:** [vllm-project/vllm](https://github.com/vllm-project/vllm)

**Inspected commit:** `2d24355eb87b716fc1169e66731dc0386ed1a3a2`

**Scope:** decoder self-attention in V1, using Llama's model module and the
FlashAttention backend as a concrete path. The same generic layer also carries
grouped-query attention (GQA) and multi-query attention (MQA); encoder,
cross-attention, multi-head latent attention (MLA), sparse attention, and
platform-specific alternatives are outside this trace.

**Verification boundary:** clean, commit-pinned static reading. No model was
served and no GPU kernel or performance benchmark was run. A scoped freshness
check on 2026-08-16 found relevant upstream changes, but the repository's
14-day evidence interval deferred a new snapshot until 2026-08-27.

## The Short Version

vLLM's multi-head attention (MHA) is not one function. It is a four-part
handoff:

1. A model-specific module projects hidden states into Q/K/V, applies rotary
   position embedding (RoPE), and later applies the output projection.
2. The reusable `Attention` layer reshapes flat tensors into heads and crosses
   a compilation-safe custom-op boundary.
3. `GPUModelRunner` supplies the changing serving state: request lengths,
   block tables, and the physical slot assigned to every new K/V token.
4. The selected hardware backend writes new K/V into the paged cache and runs
   the actual attention kernel over that cache.

> **Intuition:** the model file describes the Transformer layer; the model
> runner describes *this iteration's batch*; the backend describes *this
> device's execution*. The generic attention layer is the seam joining all
> three.

For the attention equation itself, start with [The Transformer](../../algorithms/foundations/transformer.md).
For the memory mapping underneath the kernel, see [vLLM Block Table Management](vllm-block-management/index.md).

## The Most Important Distinction

The model-side module and the reusable attention layer are different objects.
In the Llama example,
<a class="code-link" href="../../../external-repos/vllm-2d24355eb87b/vllm/model_executor/models/llama.py#L163" data-code-repo="vllm-2d24355eb87b" data-code-path="vllm/model_executor/models/llama.py" data-code-line="163" data-code-end-line="220"><code>LlamaAttention.__init__</code></a>
owns the learned QKV/output projections and constructs a generic `Attention`
instance. Its
<a class="code-link" href="../../../external-repos/vllm-2d24355eb87b/vllm/model_executor/models/llama.py#L222" data-code-repo="vllm-2d24355eb87b" data-code-path="vllm/model_executor/models/llama.py" data-code-line="222" data-code-end-line="232"><code>LlamaAttention.forward()</code></a>
is correspondingly small: fused QKV projection, split, RoPE on Q/K, generic
attention, then the row-parallel output projection.

The generic `Attention` object owns no QKV projection weights. It owns the
head geometry, cache contract, backend implementation, and forward dispatch.
That separation lets many model files reuse the same serving machinery while
platforms replace the backend without rewriting the model.

## When Each Piece Runs

| Time | Owner | State produced or changed | Why it exists |
|---|---|---|---|
| Model construction | Model module | Local Q/K/V head counts, fused projection modules, RoPE, generic `Attention` | Express the checkpoint architecture and tensor-parallel layout. |
| Model construction | Generic attention + selector | One validated backend class and its implementation object | Choose a kernel family compatible with dtype, cache format, mask, platform, and configuration. |
| KV-cache initialization | Model runner | Allocated cache tensors bound to layer names | Give every attention layer its persistent paged storage. |
| Every engine iteration | Model runner + metadata builder | Packed query boundaries, sequence lengths, block tables, slot mappings | Translate scheduled requests into the tensors kernels understand. |
| Every layer forward | Model module + generic attention | Q/K/V heads, new cache entries, attention output | Perform the Transformer computation for the scheduled tokens. |

The persistent-cache step is explicit in
<a class="code-link" href="../../../external-repos/vllm-2d24355eb87b/vllm/v1/worker/gpu_model_runner.py#L7660" data-code-repo="vllm-2d24355eb87b" data-code-path="vllm/v1/worker/gpu_model_runner.py" data-code-line="7660" data-code-end-line="7713"><code>GPUModelRunner.initialize_kv_cache_tensors()</code></a>:
the runner allocates or reshapes cache storage, handles cross-layer sharing,
then binds layer-name-specific views into the static forward context.

## One Path Covers MHA, GQA, and MQA

The control path does not branch on three separate attention classes. It is
parameterized by query-head count $H_q$ and key/value-head count $H_{kv}$.
<a class="code-link" href="../../../external-repos/vllm-2d24355eb87b/vllm/model_executor/models/llama.py#L123" data-code-repo="vllm-2d24355eb87b" data-code-path="vllm/model_executor/models/llama.py" data-code-line="123" data-code-end-line="160"><code>LlamaAttention.__init__</code></a>
partitions query heads across tensor-parallel (TP) ranks; KV heads are either
partitioned or replicated so every rank has at least one.

| Variant | Head relation | Local Q shape | Local K/V shape | Cache consequence |
|---|---|---|---|---|
| MHA | $H_q = H_{kv}$ | `[tokens, Hq/TP, D]` | `[tokens, Hkv/TP, D]` | One K/V head per query head. |
| [GQA](../../terms/grouped-query-attention.md) | $H_q > H_{kv} > 1$ | `[tokens, Hq/TP, D]` | Partitioned or replicated `[tokens, Hkv_local, D]` | Several query heads share each K/V head. |
| MQA | $H_{kv} = 1$ | `[tokens, Hq/TP, D]` | One replicated K/V head per TP rank | Smallest K/V width; all query heads share it. |

Here $D$ is `head_dim`. The generic layer enforces that `num_heads` is divisible
by `num_kv_heads`, so the kernel can infer the number of query heads sharing
each K/V head. This is why a Llama-family GQA model and an ordinary MHA model
can use the same Python forward path.

## How the Backend Is Chosen

The generic layer first records head and cache geometry, then requests a backend
with dtype, KV-cache dtype, mask features, attention type, and optional
quantization constraints in
<a class="code-link" href="../../../external-repos/vllm-2d24355eb87b/vllm/model_executor/layers/attention/attention.py#L317" data-code-repo="vllm-2d24355eb87b" data-code-path="vllm/model_executor/layers/attention/attention.py" data-code-line="317" data-code-end-line="350"><code>Attention.__init__</code></a>.
The selector adds configuration-level constraints—including explicit backend
overrides, cache block size, KV transfer, prefix masks, and prefill context
parallelism—in
<a class="code-link" href="../../../external-repos/vllm-2d24355eb87b/vllm/v1/attention/selector.py#L153" data-code-repo="vllm-2d24355eb87b" data-code-path="vllm/v1/attention/selector.py" data-code-line="153" data-code-end-line="188"><code>get_attn_backend()</code></a>.

The final choice is delegated to the active platform by
<a class="code-link" href="../../../external-repos/vllm-2d24355eb87b/vllm/v1/attention/selector.py#L193" data-code-repo="vllm-2d24355eb87b" data-code-path="vllm/v1/attention/selector.py" data-code-line="193" data-code-end-line="223"><code>_cached_get_attn_backend()</code></a>.
Only after that choice does
<a class="code-link" href="../../../external-repos/vllm-2d24355eb87b/vllm/model_executor/layers/attention/attention.py#L409" data-code-repo="vllm-2d24355eb87b" data-code-path="vllm/model_executor/layers/attention/attention.py" data-code-line="409" data-code-end-line="423"><code>Attention.__init__</code></a>
instantiate the backend-specific implementation.

> **Important:** FlashAttention below is a worked backend, not a promise that
> every NVIDIA run selects it. Hardware generation, dtype, head size, cache
> layout, configured overrides, sliding windows, sinks, and other features can
> select or reject a backend before the first forward.

## Worked Trace: One Decode Token

Assume one request is decoding one new token in one decoder layer. Its old K/V
already occupies non-contiguous physical pages.

1. **Map the new token to a physical cache slot.** The scheduler has already
   allocated blocks. The worker's
   <a class="code-link" href="../../../external-repos/vllm-2d24355eb87b/vllm/v1/worker/gpu_model_runner.py#L4225" data-code-repo="vllm-2d24355eb87b" data-code-path="vllm/v1/worker/gpu_model_runner.py" data-code-line="4225" data-code-end-line="4255"><code>GPUModelRunner._get_slot_mappings()</code></a>
   exposes a token-to-physical-slot vector for each KV-cache group and maps it
   back to every layer in that group. Padding slots become `-1`.

2. **Build the batch's attention metadata.** In
   <a class="code-link" href="../../../external-repos/vllm-2d24355eb87b/vllm/v1/worker/gpu_model_runner.py#L4478" data-code-repo="vllm-2d24355eb87b" data-code-path="vllm/v1/worker/gpu_model_runner.py" data-code-line="4478" data-code-end-line="4503"><code>GPUModelRunner.execute_model()</code></a>,
   the runner combines slot mappings with query lengths, sequence lengths, and
   block-table tensors, then lets each backend's metadata builder produce the
   per-layer structure it needs.

3. **Inject metadata without changing every model signature.** The runner enters
   <a class="code-link" href="../../../external-repos/vllm-2d24355eb87b/vllm/v1/worker/gpu_model_runner.py#L4537" data-code-repo="vllm-2d24355eb87b" data-code-path="vllm/v1/worker/gpu_model_runner.py" data-code-line="4537" data-code-end-line="4560"><code>set_forward_context(...)</code></a>
   around the model call. Model modules therefore pass only Q/K/V; the generic
   attention op retrieves dynamic serving state by layer name.

4. **Project and position the token.** `LlamaAttention.forward()` computes one
   fused QKV projection, splits the local Q/K/V widths, applies RoPE to Q and K,
   and calls `self.attn(q, k, v)`. No block table appears in this model-level
   method.

5. **Turn flat widths into head tensors.** The generic
   <a class="code-link" href="../../../external-repos/vllm-2d24355eb87b/vllm/model_executor/layers/attention/attention.py#L514" data-code-repo="vllm-2d24355eb87b" data-code-path="vllm/model_executor/layers/attention/attention.py" data-code-line="514" data-code-end-line="529"><code>Attention.forward()</code></a>
   preallocates output and views Q as `[tokens, Hq_local, D]` while viewing K/V
   as `[tokens, Hkv_local, D]`.

6. **Cross the compilation boundary.** The remainder of
   <a class="code-link" href="../../../external-repos/vllm-2d24355eb87b/vllm/model_executor/layers/attention/attention.py#L530" data-code-repo="vllm-2d24355eb87b" data-code-path="vllm/model_executor/layers/attention/attention.py" data-code-line="530" data-code-end-line="570"><code>Attention.forward()</code></a>
   chooses a direct call or opaque `torch.ops.vllm` call. When cache update is a
   separate operation, a dummy tensor creates a data dependency so
   `torch.compile` cannot reorder attention before the cache write.

7. **Resolve this layer's serving state.** The custom op uses
   <a class="code-link" href="../../../external-repos/vllm-2d24355eb87b/vllm/model_executor/layers/attention/attention.py#L695" data-code-repo="vllm-2d24355eb87b" data-code-path="vllm/model_executor/layers/attention/attention.py" data-code-line="695" data-code-end-line="713"><code>get_attention_context()</code></a>
   to retrieve per-layer metadata, the bound KV-cache tensor, and this layer's
   slot mapping from the forward context.

8. **Scatter the new K/V into its page.** FlashAttention declares the cache write
   to be separate from its forward in
   <a class="code-link" href="../../../external-repos/vllm-2d24355eb87b/vllm/v1/attention/backends/flash_attn.py#L77" data-code-repo="vllm-2d24355eb87b" data-code-path="vllm/v1/attention/backends/flash_attn.py" data-code-line="77" data-code-end-line="97"><code>FlashAttentionBackend</code></a>.
   The generic
   <a class="code-link" href="../../../external-repos/vllm-2d24355eb87b/vllm/model_executor/layers/attention/attention.py#L716" data-code-repo="vllm-2d24355eb87b" data-code-path="vllm/model_executor/layers/attention/attention.py" data-code-line="716" data-code-end-line="739"><code>unified_kv_cache_update()</code></a>
   delegates to the selected implementation, and
   <a class="code-link" href="../../../external-repos/vllm-2d24355eb87b/vllm/v1/attention/backends/flash_attn.py#L1179" data-code-repo="vllm-2d24355eb87b" data-code-path="vllm/v1/attention/backends/flash_attn.py" data-code-line="1179" data-code-end-line="1213"><code>FlashAttentionImpl.do_kv_cache_update()</code></a>
   performs the `reshape_and_cache_flash` scatter using `slot_mapping`.

9. **Read the whole logical history through the block table.** The generic
   <a class="code-link" href="../../../external-repos/vllm-2d24355eb87b/vllm/model_executor/layers/attention/attention.py#L758" data-code-repo="vllm-2d24355eb87b" data-code-path="vllm/model_executor/layers/attention/attention.py" data-code-line="758" data-code-end-line="787"><code>unified_attention_with_output()</code></a>
   calls `self.impl.forward(...)`. FlashAttention first views the paged tensor as
   separate K and V caches and extracts query boundaries, sequence lengths, and
   the block table in
   <a class="code-link" href="../../../external-repos/vllm-2d24355eb87b/vllm/v1/attention/backends/flash_attn.py#L982" data-code-repo="vllm-2d24355eb87b" data-code-path="vllm/v1/attention/backends/flash_attn.py" data-code-line="982" data-code-end-line="1013"><code>FlashAttentionImpl.forward()</code></a>.

10. **Launch attention and return to the model.** The normal non-cascade path
    calls
    <a class="code-link" href="../../../external-repos/vllm-2d24355eb87b/vllm/v1/attention/backends/flash_attn.py#L1122" data-code-repo="vllm-2d24355eb87b" data-code-path="vllm/v1/attention/backends/flash_attn.py" data-code-line="1122" data-code-end-line="1148"><code>flash_attn_varlen_func(...)</code></a>
    with the paged K/V views, query/sequence lengths, causal/window rules, and
    block table. The result returns through the preallocated output tensor;
    Llama flattens local heads and applies `o_proj`.

The state change happens at step 8. Steps 9-10 read the newly updated cache, so
the current token can attend to itself while causal masking excludes future
positions.

## Why vLLM Is Structured This Way

### Model code stays close to checkpoint semantics

The Llama module still reads like ordinary inference code: projection, split,
RoPE, attention, output projection. Cache paging and batch packing do not leak
into every model implementation.

### Prefill and decode share an interface

Both phases pass packed token tensors through `Attention.forward()`. Query
boundaries and sequence lengths in metadata tell the backend whether a request
contributes a long prompt chunk, a short extension, or one decode token. This
keeps the model path stable while the batch composition changes every
iteration.

### Cache write and cache read use different maps

- `slot_mapping` answers: **where should each new K/V token be written?**
- [Block table](../../terms/block-table.md) answers: **which physical pages make
  up each request's logical history?**

Conflating them is a common source-reading mistake. The first is token-to-slot
scatter metadata; the second is request-to-page gather metadata consumed by
[PagedAttention](../../terms/pagedattention.md)-style kernels.

### Backend replacement is a load-time decision

The hot model path does not repeatedly ask which kernel family to use. Backend
selection and implementation construction happen before execution; per-step
work is limited to metadata creation, cache update, and kernel dispatch.

## Failure and Debugging Surfaces

| Symptom | First boundary to inspect | Typical mismatch |
|---|---|---|
| Wrong output only with TP > 1 | Model head layout and fused QKV projection | Query heads partitioned but KV replication/partition count is wrong. |
| Corruption after preemption or prefix reuse | Worker `slot_mapping` and block table | New K/V scattered to one page while attention reads a different logical page list. |
| Backend rejected at startup | Selector constraints | Unsupported head size, dtype, cache dtype/layout, mask feature, or block size. |
| Failure only under `torch.compile` or CUDA graphs | Generic custom-op boundary | Cache side effect or output buffer dependency is not preserved. |
| Correct prefill but wrong decode | Per-step metadata and cache write | Query/sequence lengths or current-token slot differs between phases. |
| Different behavior across GPUs/platforms | Platform-selected backend | The model path is shared, but kernel support and cache-update ownership differ. |

## Recommended Reading Order

1. <a class="code-link" href="../../../external-repos/vllm-2d24355eb87b/vllm/model_executor/models/llama.py#L123" data-code-repo="vllm-2d24355eb87b" data-code-path="vllm/model_executor/models/llama.py" data-code-line="123" data-code-end-line="160"><code>vllm/model_executor/models/llama.py</code></a>
   — recognize familiar Transformer math.
2. <a class="code-link" href="../../../external-repos/vllm-2d24355eb87b/vllm/model_executor/layers/attention/attention.py#L317" data-code-repo="vllm-2d24355eb87b" data-code-path="vllm/model_executor/layers/attention/attention.py" data-code-line="317" data-code-end-line="350"><code>vllm/model_executor/layers/attention/attention.py</code></a>
   — understand the serving adapter and compile-safe dispatch.
3. <a class="code-link" href="../../../external-repos/vllm-2d24355eb87b/vllm/v1/worker/gpu_model_runner.py#L4478" data-code-repo="vllm-2d24355eb87b" data-code-path="vllm/v1/worker/gpu_model_runner.py" data-code-line="4478" data-code-end-line="4503"><code>vllm/v1/worker/gpu_model_runner.py</code></a>
   — see where batch and cache metadata enter the forward.
4. <a class="code-link" href="../../../external-repos/vllm-2d24355eb87b/vllm/v1/attention/selector.py#L153" data-code-repo="vllm-2d24355eb87b" data-code-path="vllm/v1/attention/selector.py" data-code-line="153" data-code-end-line="188"><code>vllm/v1/attention/selector.py</code></a>
   — understand why a backend was selected.
5. One file in `vllm/v1/attention/backends/` — follow cache layout, metadata,
   and the kernel call for the actual device.

## One Thing to Remember

**vLLM's MHA path is a stateful serving pipeline around a familiar Transformer
core:** the model creates Q/K/V, the worker describes where the current batch
lives, the generic attention layer preserves the handoff, and the backend
writes and reads the paged [KV cache](../../terms/kv-cache.md).

## Go Deeper

- [vLLM Architecture and Code Organization Overview](vllm-overview.md) — where
  this path sits in the full engine.
- [vLLM Continuous Batching](vllm-continuous-batching/index.md) — how requests
  and token counts are selected before the model runner sees them.
- [vLLM Block Table Management](vllm-block-management/index.md) — how the
  scheduler allocates the physical pages consumed here.
- [vLLM DCP Attention](dcp-attention/index.md) — how the backend path changes
  when context is distributed across ranks.
