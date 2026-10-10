---
title: "DeepSeek-V4.1-Flash cache path on vLLM GPU and Ascend"
summary: "Code-backed analysis of DeepSeek-V4.1 compressed global KV, indexer, compressor state, and local SWA on both platforms."
layout: default
confidence: medium
code_links: strict
code_evidence: strict
sources:
  - raw/frameworks/serving/vllm/vllm-codebase--github-02aa10539f15.md
  - derived/repo-analysis/frameworks/serving/vllm/vllm/02aa10539f15c8c73aa7fd4a5ee0f691cbf88dcb/important-files.md
  - raw/frameworks/serving/vllm-ascend/vllm-ascend-codebase--github-51b0242c0d3b.md
  - derived/repo-analysis/frameworks/serving/vllm-ascend/vllm-ascend/51b0242c0d3be1d3bd59e763a7d78b8553cd45e1/important-files.md
updated: 2026-10-10
---

# DeepSeek-V4.1-Flash cache path on vLLM GPU and Ascend

**Pinned code:** [vLLM](https://github.com/vllm-project/vllm/tree/02aa10539f15c8c73aa7fd4a5ee0f691cbf88dcb) `02aa10539f15c8c73aa7fd4a5ee0f691cbf88dcb`; [vLLM Ascend](https://github.com/vllm-project/vllm-ascend/tree/51b0242c0d3be1d3bd59e763a7d78b8553cd45e1) `51b0242c0d3be1d3bd59e763a7d78b8553cd45e1`. Both checkouts were clean. This is static code analysis; no model or accelerator was run.

[Model cache study](index.md) · [vLLM](../vllm/index.md) · [vLLM Ascend](../vllm-ascend/index.md)

## Model identity and memory classes

The [presentation brief](../../../../presentation-notes/open-model-architecture-kv-cache-gpu-ascend-2026-10.md)'s `deepseek-ai/DeepSeek-V4.1-Flash` checkpoint has a causal encoder plus decoder, compressed sparse attention, a 128-token local sliding window, and a 512-entry global selection budget. The claimed **890 B/token global cache** is a report-level model figure, not the allocation of one physical page and not the total live attention state. The GPU shell is <a class="code-link" href="../../../../external-repos/vllm-02aa10539f15/vllm/models/deepseek_v41/nvidia/model.py#L1537" data-code-repo="vllm-02aa10539f15" data-code-path="vllm/models/deepseek_v41/nvidia/model.py" data-code-line="1537"><code>DeepseekV41LLMForCausalLM</code></a>; the NPU plugin explicitly registers the V4.1 architecture at <a class="code-link" href="../../../../external-repos/vllm-ascend-51b0242c0d3b/vllm_ascend/models/__init__.py#L38" data-code-repo="vllm-ascend-51b0242c0d3b" data-code-path="vllm_ascend/models/__init__.py" data-code-line="38"><code>DeepseekV41 registration</code></a>.

## Architecture blueprint

The [released V4.1-Flash config](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/config.json) defines a 40-layer, 5,120-wide text trunk with a 20-layer causal encoder and 20-layer decoder. A local 128-token sliding window (SWA) accompanies compressed sparse global attention. The compression schedule is heterogeneous: the first two and final three listed groups have ratio zero, the next encoder-side groups use ratio two, and decoder-side groups use ratio one. Four layers (2, 8, 14, 20) own global KV, while eight layers (2, 8, 14, 20, 24, 28, 32, 36) own index sources. `index_topk: 512` is the final sparse global read budget; a separate 2,048-block candidate stage with block size eight narrows later search.

| Path | Produced state | Consumer or lifetime |
| --- | --- | --- |
| Causal encoder and compressor | Global latent rows, then compressed records | Source-owned pages reused by later layers. |
| Local SWA | Recent uncompressed K/V | Bounded attention window and replay-sensitive state. |
| Sparse indexer | Index keys, candidate blocks, top-k positions | Historical keys persist; candidate/selection buffers are step state. |
| Four-stream mHC | Mixed/residual hidden streams | Activation path, not a KV cache. |
| Engram and MoE | N-gram lookup at layers 1 and 14; six of 384 routed plus one shared expert | Weight and communication budgets outside the attention cache. |
| DSpark and vision | Three prediction layers; 32-layer vision encoder | Optional draft/multimodal work requiring separate accounting. |

The GPU <a class="code-link" href="../../../../external-repos/vllm-02aa10539f15/vllm/models/deepseek_v41/nvidia/model.py#L283" data-code-repo="vllm-02aa10539f15" data-code-path="vllm/models/deepseek_v41/nvidia/model.py" data-code-line="283"><code>DeepseekV4DecoderLayer</code></a> wires attention, MoE, normalization, Engram when configured, and four-stream hyper-connection (mHC) mixing. <a class="code-link" href="../../../../external-repos/vllm-02aa10539f15/vllm/models/deepseek_v41/attention.py#L305" data-code-repo="vllm-02aa10539f15" data-code-path="vllm/models/deepseek_v41/attention.py" data-code-line="305"><code>KV and index source mapping</code></a> decides source ownership per layer and maps consumers to the latest preceding KV and index source. Thus a consumer may run its own attention computation while borrowing the source cache; multiplying one cache record by 40 layers is wrong. <a class="code-link" href="../../../../external-repos/vllm-02aa10539f15/vllm/models/deepseek_v41/nvidia/model.py#L125" data-code-repo="vllm-02aa10539f15" data-code-path="vllm/models/deepseek_v41/nvidia/model.py" data-code-line="125"><code>DeepseekV4MoE</code></a> routes only the selected experts for the token, then the decoder returns its updated hidden stream. The released weight config's FP8/FP4 settings describe weights and experts; they do **not** by themselves specify the runtime KV-cache dtype.

### Prefill to decode boundary

During prompt processing, the causal encoder and source layers establish compressed global records while each layer has a local SWA view. At decode, the current token updates an incomplete compressor group, may finalize a new global record, scores source index keys, attends to selected global positions plus its local window, then passes through mHC and MoE. A prefix hit must restore the completed global pages **and** enough unfinished-group/local state to continue exactly. The NPU <a class="code-link" href="../../../../external-repos/vllm-ascend-51b0242c0d3b/vllm_ascend/models/deepseek_v41/model.py#L767" data-code-repo="vllm-ascend-51b0242c0d3b" data-code-path="vllm_ascend/models/deepseek_v41/model.py" data-code-line="767"><code>DeepseekV41DecoderLayer</code></a> builds corresponding attention, MoE, mHC, and optional Engram objects, but uses Ascend operators and cache formats; its physical page bytes must be measured independently.

The [DeepSeek-V4.1 report note](../../../training/model-families/deepseek/deepseek-v4.1/index.md) explains the report-level CSA2 and causal encoder-decoder design. The 890 B/token claim there applies to global compressed KV under report assumptions, not to every live cache and workspace listed above.

## One token through the GPU cache path

1. A KV-source layer forms candidate global latent records. <a class="code-link" href="../../../../external-repos/vllm-02aa10539f15/vllm/models/deepseek_v41/compressor.py#L136" data-code-repo="vllm-02aa10539f15" data-code-path="vllm/models/deepseek_v41/compressor.py" data-code-line="136"><code>CompressorStateCache</code></a> keeps an incomplete compressor group until a compressed record can be emitted.
2. <a class="code-link" href="../../../../external-repos/vllm-02aa10539f15/vllm/models/deepseek_v41/attention.py#L1138" data-code-repo="vllm-02aa10539f15" data-code-path="vllm/models/deepseek_v41/attention.py" data-code-line="1138"><code>compressed KV cache spec</code></a> allocates completed compressed KV only for source layers. Its `tokens_per_state` records the compression ratio; consumer layers retrieve their source's cache through the forward context. This prevents treating each consumer as another full global-cache owner.
3. <a class="code-link" href="../../../../external-repos/vllm-02aa10539f15/vllm/models/deepseek_v41/attention.py#L1175" data-code-repo="vllm-02aa10539f15" data-code-path="vllm/models/deepseek_v41/attention.py" data-code-line="1175"><code>DeepseekV4IndexerCache</code></a> owns the separate index-key page type. A current query scores that history to produce a transient top-k list. Sparse MLA reads selected global records while local sliding-window attention reads recent uncompressed tokens. The query result returns to the decoder; neither the top-k list nor the open compressor group is the durable global cache.

GPU physical bytes depend on `cache_dtype`: the cache spec can select packed `nvfp4_ds_mla` or `fp8_ds_mla` records and aligns their pages for the selected kernel. A token-level report figure cannot be obtained by multiplying one compressed record by 40 layers because only source layers own records and compression ratios differ.

The compressor's incomplete group uses a small FP32 circular-buffer cache, whereas the completed global rows use an MLA page spec. That difference matters at a prefix boundary: saving completed pages without the unfinished group can change the next compressed record. The indexer page has its own alignment because the sparse-logit kernel and main MLA record can share a physical block stride.

## Ascend ownership and divergence

<a class="code-link" href="../../../../external-repos/vllm-ascend-51b0242c0d3b/vllm_ascend/models/deepseek_v41/model.py#L491" data-code-repo="vllm-ascend-51b0242c0d3b" data-code-path="vllm_ascend/models/deepseek_v41/model.py" data-code-line="491"><code>DeepseekV41SharedAttentionState</code></a> coordinates cross-layer state, <a class="code-link" href="../../../../external-repos/vllm-ascend-51b0242c0d3b/vllm_ascend/models/deepseek_v41/model.py#L565" data-code-repo="vllm-ascend-51b0242c0d3b" data-code-path="vllm_ascend/models/deepseek_v41/model.py" data-code-line="565"><code>AscendDeepseekV41SWACache</code></a> defines a distinct local window cache, and <a class="code-link" href="../../../../external-repos/vllm-ascend-51b0242c0d3b/vllm_ascend/models/deepseek_v41/model.py#L675" data-code-repo="vllm-ascend-51b0242c0d3b" data-code-path="vllm_ascend/models/deepseek_v41/model.py" data-code-line="675"><code>DeepseekV41Attention</code></a> joins the sparse/global and local paths. The NPU implementation has its own compressor and indexer code; it does not inherit the GPU record packing verbatim. A colocated run must budget completed global pages, index history, active SWA rows, and compressor state. Prefill/decode separation may require replay or transfer according to the deployed connector and cache policy.

### NPU cache ABI by hardware profile

The Ascend cache layer does more than relabel GPU pages. <a class="code-link" href="../../../../external-repos/vllm-ascend-51b0242c0d3b/vllm_ascend/models/deepseek_v41/cache_config.py#L36" data-code-repo="vllm-ascend-51b0242c0d3b" data-code-path="vllm_ascend/models/deepseek_v41/cache_config.py" data-code-line="36"><code>make_swa_cache_spec</code></a> creates a sliding-window MLA spec; on a hardware profile with packed-cache support it stores FP8 values with BF16 scales. <a class="code-link" href="../../../../external-repos/vllm-ascend-51b0242c0d3b/vllm_ascend/models/deepseek_v41/cache_config.py#L55" data-code-repo="vllm-ascend-51b0242c0d3b" data-code-path="vllm_ascend/models/deepseek_v41/cache_config.py" data-code-line="55"><code>make_mla_cache_spec</code></a> creates source-owned global MLA pages: the packed path stores two FP4 values per byte plus BF16 scales, while the other branch uses BF16 data with a different scale contract. <a class="code-link" href="../../../../external-repos/vllm-ascend-51b0242c0d3b/vllm_ascend/models/deepseek_v41/cache_config.py#L76" data-code-repo="vllm-ascend-51b0242c0d3b" data-code-path="vllm_ascend/models/deepseek_v41/cache_config.py" data-code-line="76"><code>make_index_cache_spec</code></a> separately chooses index-key and scale types; its packed branch is not the same record as the main MLA branch. These are code-level ABI choices, not universal sizes for every Ascend generation.

The `DeepseekV41Attention` constructor uses a topology plan to instantiate long-KV cache and compressor **only** at KV-source layers, an indexer **only** at index-source layers, and references to the earlier source for consumers. This mirrors the checkpoint's cross-layer sharing without requiring consumers to allocate duplicate long histories. A physical allocation trace should list each `long_kv_cache`, index cache, SWA cache, and compressor circular buffer by layer; it should also include padding from the layer-outermost layout. Comparing only a single tensor's element count with the report's 890 B/token will miss much of this NPU memory.

| Cache class | Durable? | Optimization target |
| --- | --- | --- |
| Source-owned compressed global KV | Yes, across prefix reuse | Page precision, sharing, transfer. |
| Indexer keys | Yes while searchable | Scan bandwidth and separate quantization. |
| Open compressor group | Until emitted | Correct continuation at a prefix boundary. |
| 128-token SWA | Bounded local history | Replay versus retain. |

**Validation boundary:** No GPU or NPU run was performed. Capture actual tensor shape, stride, dtype, scale bytes, and prefix-hit restoration for all four classes before claiming an end-to-end bytes/token figure.
