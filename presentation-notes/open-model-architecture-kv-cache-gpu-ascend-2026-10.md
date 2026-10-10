# Open-weight model architecture and KV-cache deployment brief

Research snapshot: **10 October 2026**. Audience: serving and accelerator engineers preparing a presentation and a KV-cache optimization plan. Here, **NPU means Huawei Ascend with vLLM-Ascend**; other NPU stacks need separate validation. “Open” means downloadable weights, not necessarily an OSI-approved license. Check each checkpoint's license before deployment.

## Scope and model selection

| Family requested | Checkpoint analyzed | Why this one |
| --- | --- | --- |
| GLM-5.3 | [`zai-org/GLM-5.3`](https://huggingface.co/zai-org/GLM-5.3) | The named released open-weight model. |
| DeepSeek-V4.1-Flash | [`deepseek-ai/DeepSeek-V4.1-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | The named released checkpoint. |
| MiniMax-M3 | [`MiniMaxAI/MiniMax-M3`](https://huggingface.co/MiniMaxAI/MiniMax-M3) | The named released multimodal checkpoint. |
| Qwen3.8 | [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) | The newest Qwen3.8 architecture release listed by the [Qwen organization](https://huggingface.co/Qwen); distinct from the larger Qwen3.8-2.4T-A95B and smaller 27B models. |
| Kimi-K3 | [`moonshotai/Kimi-K3`](https://huggingface.co/moonshotai/Kimi-K3) | The named released checkpoint. |

The table compares **text-decoder cache**. Image/video encoders, prompt embedding tables, model weights, MoE communication buffers, and speculative draft models need separate memory budgets.

## 1. GLM-5.3

### Architecture and cache production

The [released config](https://huggingface.co/zai-org/GLM-5.3/blob/main/config.json) specifies 78 decoder layers, 64 query heads, 512-dimensional MLA KV latent, 64-dimensional positional key, a sparse indexer with 32 heads of width 128 and top-k 2,048, and an indexer pattern of `full` owners followed by `shared` consumers. The first three MLP layers are dense; subsequent layers use 256 routed experts with eight selected plus one shared expert. Z.ai says GLM-5.3 uses the [same base architecture as GLM-5.2](https://z.ai/blog/glm-5.3), which makes the repo's [GLM-5.2 GPU trace](../docs/frameworks/serving/vllm/glm-5.2-inference-path.md) useful for the cache mechanism. Its pinned code is evidence for the shared mechanism, not proof that every GLM-5.3 deployment uses identical kernels.

For token `t`, each layer projects its hidden state into a query and compressed KV latent, creates the positional key field, and writes the resulting **main MLA record** to a paged slot. An index-owner layer also projects an index key and writes it into a **separate paged indexer cache**; the index query scores historical index keys and selects positions. Shared layers reuse the owner's index/selection according to the checkpoint pattern, but still run their own attention and MoE computation. The selected positions are translated through the block table before sparse MLA gathers main-cache records. The indexer selection is a per-step buffer; the index keys and MLA records are historical state. [GPU trace](../docs/frameworks/serving/vllm/glm-5.2-inference-path.md), [config](https://huggingface.co/zai-org/GLM-5.3/blob/main/config.json).

### GPU storage state

| Stored object | Logical owner and lifetime | GPU location / layout evidence |
| --- | --- | --- |
| Main MLA latent + positional key | One token record per main-attention layer; survives future decode and reusable prefixes. | GPU HBM paged cache, logically `[block, token, 512 + 64]`; the sparse-MLA backend chooses physical packing and dtype. vLLM's GLM-family path writes by slot mapping. [GPU trace](../docs/frameworks/serving/vllm/glm-5.2-inference-path.md) |
| Indexer K | Index-owner groups; survives as searchable history. | Separate GPU HBM paged cache, commonly quantized independently of the main cache. [GPU trace](../docs/frameworks/serving/vllm/glm-5.2-inference-path.md) |
| Top-k token positions | Current query/step, shared where configured. | Transient `int32` selection buffer and sparse-gather workspace; it does not replace historical main KV. [GPU trace](../docs/frameworks/serving/vllm/glm-5.2-inference-path.md) |

At BF16 the **main payload alone** is `78 × (512 + 64) × 2 = 87.75 KiB/token`; indexer keys, scales, page padding and TP behavior add to this. A sparse top-k of 2,048 bounds the main attention *read set*, not the retained history.

### Ascend NPU storage state

The [GLM-5.3 Ascend guide](https://docs.vllm.ai/projects/ascend/en/v0.27.1rc1/tutorials/models/GLM5.3.html) is experimental and describes a W8A8C8 deployment. In the ordinary colocated path, main sparse-attention cache and indexer cache reside in NPU HBM as paged buffers; the separate Ascend [SFA C8 and LI C8 switches](https://docs.vllm.ai/projects/ascend/en/main/user_guide/configuration/additional_config.html) show that main KV and indexer K have distinct precision/layout contracts. The NPU attention operator consumes selected positions after the runtime maps them to physical cache slots. Ascend's [cache-layout design](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/Design_Documents/non_contiguous_kv_cache.html) permits page-aligned strided views, so the GPU's byte stride is not a portable layout specification.

An optional, separately documented [sparse Decode offload design](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/Design_Documents/layerwise_and_sparse_kv_cache_offloading.html) keeps **full main KV in pinned host memory**, while NPU HBM holds the full indexer history and per-layer hot top-k main-KV buffers. It writes new KV to host, selects positions on NPU, transfers only missing selected rows, then remaps them to hot slots. This is an optimization configuration, not the default placement of the GLM-5.3 guide; its published examples use GLM-5.2 and require separate GLM-5.3 validation.

**Optimization question:** Is HBM capacity dominated by main latent pages, or is decode dominated by indexer scans and sparse gathers? Measure both caches independently before choosing C8, offload or index reuse.

## 2. DeepSeek-V4.1-Flash

### Architecture and cache production

The [config](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/config.json) has 40 layers: 20 causal-encoder layers followed by 20 decoder layers, with 128-token sliding-window attention (SWA), compression ratios of two in encoder global-cache source groups and one in decoder groups, four global KV source layers (`2, 8, 14, 20`), sparse top-k 512, and eight index-source layers. The model also has MoE, Engram memory and DSpark, but those are separate from the durable attention cache. The [technical report note](../docs/training/model-families/deepseek/deepseek-v4.1/index.md) explains CSA2's Full/Reindex/Reuse modes and the causal encoder-decoder (CED) projection.

During prefill, the causal encoder processes the prompt and its boundary hidden state supplies projections for upper-layer global KV; the decoder need not rerun its full global path across the entire prefix. At a **KV source** layer, a compressor accumulates source token KV/score rows and emits one compressed global record per group. Consumers read that source cache instead of allocating a separate long-history cache. Full/Reindex indexers produce or reuse index keys and top-k positions. Every layer also computes its own local SWA K/V; these remain bounded to a short window and can be rebuilt by bounded replay after a prefix hit. [vLLM attention implementation](https://docs.vllm.ai/en/latest/api/vllm/models/deepseek_v41/attention/), [compressor implementation](https://docs.vllm.ai/en/latest/api/vllm/models/deepseek_v41/compressor/).

### GPU storage state

| Stored object | Production and persistence | GPU location / record |
| --- | --- | --- |
| Compressed global KV | Compressor at four source layers; consumer layers point to owner cache. Long-lived prefix state. | GPU HBM paged MLA cache. vLLM documents **288-byte NVFP4 compressed records** (`256` E2M1 data bytes + `32` E4M3 scales), and **528-byte MXFP8 records** (`512` data bytes + `16` scales), depending on cache dtype/backend. These are implementation formats, not simultaneous allocations. [GPU packing operator](https://docs.vllm.ai/en/latest/api/vllm/models/deepseek_v41/common/ops/fused_compress_quant_cache/) |
| Compressor open group | KV/score rows not yet emitted as one compressed record. | Small per-request ring/state cache, separate from completed global pages. [Compressor implementation](https://docs.vllm.ai/en/latest/api/vllm/models/deepseek_v41/compressor/) |
| Indexer K and selected positions | Index-source history is retained; top-k output is per step or reused by a CSA2 group. | Separate indexer state and temporary selection buffers. [vLLM attention implementation](https://docs.vllm.ai/en/latest/api/vllm/models/deepseek_v41/attention/) |
| SWA K/V | Each layer's recent 128-token local path. | Separate bounded SWA cache; can be regenerated by replay depending on cache policy. [vLLM attention implementation](https://docs.vllm.ai/en/latest/api/vllm/models/deepseek_v41/attention/) |

The report's **890 B/token global-cache** figure includes sharing/compression assumptions and is a model-level claim; it is not equal to the byte count of one physical compressed record and excludes live SWA, indexer and allocator overhead. [Report note](../docs/training/model-families/deepseek/deepseek-v4.1/index.md).

### Ascend NPU storage state

The [Ascend V4.1 guide](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/DeepSeek-V4.1-Flash.html) documents A2/A3 W8A8 deployments on a matching main-branch stack and currently requires model runner V1. The logical objects remain **compressed global KV by source group, indexer state, local SWA state and compressor state**. In a colocated run they occupy NPU-managed cache pools, with physical page padding/precision determined by the selected Ascend kernels. The guide does not publish a universal byte-for-byte NPU record equivalent to GPU's 288/528-byte formats; use runtime tensor descriptors before quoting a physical NPU layout. In its P/D recipe, Prefill owns prefix caching and Decode may recompute unavailable state rather than assuming the full prefix cache is locally resident. [Ascend guide](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/DeepSeek-V4.1-Flash.html).

**Optimization question:** Which records truly need long lifetime? Separate global compressed pages from disposable SWA and compressor rings; preserve shared-cache ownership during offload or transfer.

## 3. MiniMax-M3

### Architecture and cache production

The [M3 config](https://huggingface.co/MiniMaxAI/MiniMax-M3/blob/main/config.json) has 60 text layers, 64 query heads, four KV heads of width 128, 128 routed experts with four active plus one shared expert, and a vision tower. The first three layers are dense; the following 57 use MiniMax Sparse Attention (MSA). At each layer the hidden state produces **ordinary GQA K and V**. In sparse layers a second index-Q/index-K branch scores 128-token blocks, chooses up to 16 content blocks plus required initial/local blocks, and passes selected block IDs to the main attention kernel. The selection does **not** evict unselected K/V: future queries can select those blocks. [M3 card](https://huggingface.co/MiniMaxAI/MiniMax-M3), [GPU model implementation](https://docs.vllm.ai/en/latest/api/vllm/models/minimax_m3/nvidia/model/).

### GPU storage state

| Stored object | Production and persistence | GPU location / record |
| --- | --- | --- |
| Main K/V | QKV projection at every layer; full history retained. | vLLM sparse kernel documents one 128-token page as `(num_blocks, num_kv_heads, 128, 2 × head_dim)`, with K and V packed in the last dimension. Page size is forced to equal sparse block size. [Sparse kernel](https://docs.vllm.ai/en/latest/api/vllm/models/minimax_m3/common/ops/sparse_attn/) |
| Indexer K | Separate projection in each sparse layer; full index-key history retained for future block scoring. | Paged side cache, documented as `(num_blocks, 128, index_head_dim)` with one shared key vector per token; the config has `index_head_dim=128`. [Index-top-k kernel](https://docs.vllm.ai/en/latest/api/vllm/models/minimax_m3/amd/ops/index_topk/), [GPU model](https://docs.vllm.ai/en/latest/api/vllm/models/minimax_m3/nvidia/model/) |
| Selected blocks | Indexer computes current query's block IDs. | Temporary top-k buffer; sparse attention gathers corresponding K/V pages. [Sparse backend](https://docs.vllm.ai/en/latest/api/vllm/models/minimax_m3/common/sparse_attention/) |

At BF16, the main K/V is **120 KiB/token**. If all 57 sparse layers retain one BF16 128-dimensional index key per token, that side cache adds **14.25 KiB/token**, before any metadata or alignment. FP8 or another supported KV dtype changes the stored bytes; the GPU [backend feature table](https://docs.vllm.ai/en/latest/design/attention_backends/) lists the supported cache types. Thus MSA's top-16 read budget is not a 16-block memory footprint.

### Ascend NPU storage state

The [MiniMax-M3 Ascend guide](https://docs.vllm.ai/projects/ascend/en/main/tutorials/models/MiniMax-M3.html) documents colocated serving, P/D disaggregation and a KV pool. In the normal NPU path, **main GQA K/V pages and indexer-side cache** are long-lived device state; selected block IDs and gather buffers are transient. Ascend may expose the main K and V as separate logical views with strides/padding instead of the GPU kernel's packed last dimension, as described by the [Ascend cache-layout design](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/Design_Documents/non_contiguous_kv_cache.html). The guide's Mooncake connector can transfer cache between prefill and decode nodes, and its pool can preserve a reusable prefix outside local NPU HBM. [Ascend M3 guide](https://docs.vllm.ai/projects/ascend/en/main/tutorials/models/MiniMax-M3.html).

**Optimization question:** When capacity is limiting, compress or tier *both* the main K/V and index-key history. Sparse reads alone do not reduce the amount retained.

## 4. Qwen3.8-Flash-Next

### Architecture and cache production

This [checkpoint](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/config.json) has 48 text layers in 12 repeats of **three Gated DeltaNet (GDN) layers followed by one Qwen Sparse Attention (QSA) layer**. The 36 GDN layers update recurrent matrix/short-convolution state instead of appending a full token-wise KV record. Each of the 12 QSA layers projects normal GQA K/V (24 Q heads, two KV heads, width 256) and a separate sparse index branch. The indexer compresses groups of four key positions and has a 2,048-token selection budget. PLE n-gram embedding lookup, gated residual connections, MoE and the vision encoder are additional architecture components, not replacements for attention cache. [Qwen model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), [config](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/config.json).

### GPU storage state

| Stored object | Production and persistence | GPU location / record |
| --- | --- | --- |
| GDN recurrent/conv state | Updated by each token in 36 layers; a prefix snapshot is needed for reuse. | Fixed-size per-sequence state, managed by the hybrid cache path rather than 36 token-wise K/V histories. [Qwen/Ascend model description](https://docs.vllm.ai/projects/ascend/en/main/tutorials/models/Qwen3.8-Flash-Next.html) |
| Main QSA K/V | Produced by QKV projection at each of 12 QSA layers and written by slot mapping. | GPU HBM paged `FullAttentionSpec` with two KV heads and head size 256. [QSA implementation](https://docs.vllm.ai/en/latest/api/vllm/models/qwen4_exp/nvidia/qsa/) |
| Raw and compressed QSA index keys | Raw key for the current open four-token group, then one compressed key per completed group. | Two distinct side caches: fixed circular raw-key buffer per request and paged compressed-key owner. The main KV and side caches share the cache-layout planner, but have different logical lifetimes. [QSA cache implementation](https://docs.vllm.ai/en/latest/api/vllm/models/qwen4_exp/common/qsa_cache/) |
| Sparse selection | Indexer chooses current relevant tokens/blocks. | Temporary packed `int32` top-k buffer; its last column is a valid-count field. [QSA implementation](https://docs.vllm.ai/en/latest/api/vllm/models/qwen4_exp/nvidia/qsa/) |

Main QSA K/V alone is **24 KiB/token** in BF16. This omits the compressed index-key history, fixed raw-key rings, and GDN/PLE state checkpoints. A naive `12 × K/V` estimate therefore understates actual hybrid-cache allocation.

### Ascend NPU storage state

The [Ascend Qwen3.8-Flash-Next guide](https://docs.vllm.ai/projects/ascend/en/main/tutorials/models/Qwen3.8-Flash-Next.html) marks support **experimental** and validates Atlas A3 W8A8; it explicitly says A2 and Ascend 950 were not yet supported in that guide. The NPU stores the same **three classes of live state**: QSA K/V pages, QSA index side caches, and GDN/PLE recurrent-state checkpoints. Hybrid page alignment can create non-contiguous K/V and state views, with a physical block stride larger than the logical payload. [Ascend layout design](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/Design_Documents/non_contiguous_kv_cache.html). Prefix caching is experimental and requires `--mamba-cache-mode align` so GDN/PLE state checkpoints line up with prefix pages. [Ascend Qwen guide](https://docs.vllm.ai/projects/ascend/en/main/tutorials/models/Qwen3.8-Flash-Next.html).

**Optimization question:** Does the large-context cost come from the 12 QSA histories, the compressed index keys, or per-request GDN checkpoints? A cache policy must handle all three and restore coherent state at a prefix boundary.

## 5. Kimi-K3

### Architecture and cache production

The [K3 config](https://huggingface.co/moonshotai/Kimi-K3/blob/main/config.json) identifies 93 text layers: 69 KDA recurrent layers and 24 global MLA layers, mostly in a 3:1 rhythm with an extra final MLA layer. KDA applies a gated delta-rule update to a recurrent state and keeps short convolution state. MLA projects a **512-dimensional compressed KV latent plus a 64-dimensional positional-key field**; the runtime can absorb the KV up-projection into decode weights, so it need not materialize full per-head historical K/V. The model also has 896 routed experts, 16 active plus two shared, and a vision path. [K3 report note](../docs/training/model-families/kimi/kimi-k3/index.md), [GPU MLA implementation](https://docs.vllm.ai/en/latest/api/vllm/models/kimi_k3/nvidia/mla/).

### GPU storage state

| Stored object | Production and persistence | GPU location / record |
| --- | --- | --- |
| KDA recurrent and convolution state | Updated at each KDA layer, with a snapshot at a reusable prefix boundary. | Hybrid per-sequence state; not one K/V page per historical token. The [vLLM K3 code map](../docs/frameworks/serving/vllm/vllm-kimi-k3-code-reading.md) traces KDA metadata and state handling. |
| MLA latent + positional field | Projection at each MLA layer for every token, inserted into paged cache. | GPU HBM paged record with logical width `512 + 64 = 576`; prefill has a fused cache-insert path and decode appends to the same cache. vLLM also documents a **656-byte `fp8_ds_mla` record** for a particular self-scaled backend format. [GPU MLA implementation](https://docs.vllm.ai/en/latest/api/vllm/models/kimi_k3/nvidia/mla/) |
| Attention workspace | Current query and gather/up-projection intermediates. | Transient. MLA decode projects the query into latent space and projects the result back, instead of caching expanded per-head values. [GPU MLA implementation](https://docs.vllm.ai/en/latest/api/vllm/models/kimi_k3/nvidia/mla/) |

The BF16 MLA token payload is **27 KiB/token** (`24 × 576 × 2`), or **3.375 GiB at 128 Ki tokens**, before KDA states and physical padding. The prior 24 KiB figure omitted the 64-dimensional positional-key field; this revision corrects it. The config's `mla_use_nope` concerns position encoding behavior and does not remove the runtime's 64-dimensional cached field. [GPU MLA implementation](https://docs.vllm.ai/en/latest/api/vllm/models/kimi_k3/nvidia/mla/).

### Ascend NPU storage state

The [Kimi-K3 Ascend guide](https://docs.vllm.ai/projects/ascend/en/main/tutorials/models/Kimi-K3.html) covers A3 W4A8 serving, prefix caching and a matching DSpark draft. The NPU must retain **MLA token pages plus KDA recurrent/conv state** for active requests; prefix reuse requires a matching KDA-state checkpoint as well as MLA history. Ascend's hybrid cache design can place these in separate logical views of a page-aligned physical pool, with state-specific strides and offsets. [Ascend non-contiguous cache design](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/Design_Documents/non_contiguous_kv_cache.html). The guide does not specify one fixed byte layout for every Kimi NPU kernel/precision, so physical bytes and tensor strides should be captured from the exact A3 build. Generic [KVPP](https://docs.vllm.ai/projects/ascend/en/main/developer_guide/Design_Documents/kvpp.html) currently targets non-hybrid MLA/SFA and should not be assumed to handle the KDA+MLA combination.

**Optimization question:** How much memory is MLA history versus per-request KDA state and snapshots? Offload, precision and prefix policies should treat these as separate objects.

## Presentation summary

1. The five models do **not** have the same cache: token-wise GQA, latent MLA, shared compressed KV, and recurrent state coexist.
2. Sparsity usually reduces **attention reads and compute**, but does not automatically remove historical KV storage. MiniMax-M3 is the clearest example.
3. A nominal 1M-token context is a model limit, **not** a guarantee that one 1M-token request fits accelerator memory. Precision, tensor parallelism, allocator padding, indexer state, batching, and weights change the deployment result.
4. GPU and Ascend use similar logical block tables, but their **physical tensor layouts and supported kernels differ**. Cache transfer must be described in terms of logical records and backend-specific packing.
5. The first optimization target should be measured **bytes per live token and bytes read per decode token**, separated by cache class and layer, followed by prefix reuse and tiering experiments.

## Cross-model appendix: architecture at a glance

| Model | Decoder and attention | Experts and other architecture | Context and modality | Cache class |
| --- | --- | --- | --- | --- |
| GLM-5.3 | 78 layers; 64 query heads; MLA latent rank 512 plus 64 RoPE dimensions; DeepSeek-style sparse attention with indexer top-k 2,048. | 256 routed experts, 8 selected, 1 shared; first 3 layers dense; one next-token prediction layer. | 1,048,576 positions; text. | Per-layer latent KV plus sparse indexer keys/metadata. [Config](https://huggingface.co/zai-org/GLM-5.3/blob/main/config.json) |
| DeepSeek-V4.1-Flash | 40 layers arranged as 20-layer causal encoder and 20-layer decoder; compressed sparse attention 2 (CSA2), 128-token local sliding window, top-k 512 global entries. | 384 routed experts, 6 selected, 1 shared; Engram n-gram memory and three DSpark prediction layers. | 1,048,576 positions; text and images. | Shared compressed global KV, indexer K and selections, plus short-lived local SWA KV. [Config](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/config.json), [technical note](../docs/training/model-families/deepseek/deepseek-v4.1/index.md) |
| MiniMax-M3 | 60 text layers; 64 query heads / 4 KV heads, 128-dimensional heads; sparse attention enabled after the first 3 layers; top 16 blocks of 128 tokens with initial/local blocks. | 128 routed experts, 4 selected, 1 shared; first 3 layers dense; vision tower and multimodal input. | 1,048,576 positions; text, images, video. | Conventional per-layer GQA K and V plus sparse index/selection buffers. [Config](https://huggingface.co/MiniMaxAI/MiniMax-M3/blob/main/config.json), [model card](https://huggingface.co/MiniMaxAI/MiniMax-M3) |
| Qwen3.8-Flash-Next | 48 layers: 36 Gated DeltaNet linear-attention layers and 12 global/sparse-attention layers; global layers use 24 query / 2 KV heads, 256-dimensional heads and a 2,048-token indexer budget. | 512 routed experts, 10 selected plus shared expert; 4-stream hyper-connections, n-gram embedding lookup, vision encoder, one MTP layer. | 262,144 positions; text and images. | Fixed-size recurrent state in 36 layers; token-wise GQA KV and indexer state in 12 layers. [Config](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/config.json), [model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) |
| Kimi-K3 | 93 main layers: 69 Kimi Delta Attention (KDA) recurrent layers and 24 global MLA layers; 512-dimensional latent plus 64-dimensional positional field. | 896 routed experts, 16 selected and 2 shared; Stable LatentMoE; native vision. | 1,048,576 positions; text and images. | Fixed-size KDA recurrent/conv state plus token-wise MLA latent/positional records in 24 layers. [Config](https://huggingface.co/moonshotai/Kimi-K3/blob/main/config.json), [GPU MLA code](https://docs.vllm.ai/en/latest/api/vllm/models/kimi_k3/nvidia/mla/) |

The architecture numbers come from the linked **checkpoint configs**. For Qwen, the config's `model_type` is `qwen4_exp` even though the released product is named Qwen3.8-Flash-Next. MiniMax's “top 16 blocks” is a **read budget**, not a 2,048-token cache-retention policy. GLM's `num_key_value_heads: 64` does not mean 64 conventional KV vectors are stored per token: the MLA latent path determines the compressed payload. [GLM config](https://huggingface.co/zai-org/GLM-5.3/blob/main/config.json), [MiniMax config](https://huggingface.co/MiniMaxAI/MiniMax-M3/blob/main/config.json).

For sizing **weights as well as cache**, the reported model scales are: GLM-5.3 inherits the roughly **744B total / 40B active** GLM-5 base; DeepSeek-V4.1-Flash has a **552B backbone plus 196B Engram parameters**, activating **8B in prefill / 16B in decode**; MiniMax-M3 reports **428B / 23B active**; Qwen3.8-Flash-Next reports **125B / 6B active** in the text trunk plus **51B n-gram embeddings** and about **4B MTP parameters**; Kimi-K3 reports **2.8T / 104B active**. These totals use different accounting conventions, especially for external embedding tables and prediction heads, and should not be used as a KV-memory comparison. [Z.ai model lineage](https://z.ai/blog/glm-5.3), [GLM-5 scale](https://z.ai/blog/glm-5), [DeepSeek V4.1 report note](../docs/training/model-families/deepseek/deepseek-v4.1/index.md), [MiniMax card](https://huggingface.co/MiniMaxAI/MiniMax-M3), [Qwen card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), [Kimi report note](../docs/training/model-families/kimi/kimi-k3/index.md).

## Logical cache layout and memory floor

For ordinary GQA, one sequence token produces K and V of shape `[KV heads, head dimension]` **per attention layer**. A simple BF16 payload estimate is `2 × layers × KV heads × head dimension × 2 bytes`. MLA instead stores a compressed latent (and, where applicable, a positional key), while recurrent layers store state that is roughly independent of sequence length. These are **logical payload** estimates, before allocator pages, alignment, TP replication/sharding, indexer state, quantization metadata, graph buffers, prefix-sharing refcounts, or the model weights.

| Model | Durable token-dependent state to budget | BF16 payload floor per token | Approx. floor at 128 Ki tokens | Important excluded state |
| --- | --- | ---: | ---: | --- |
| GLM-5.3 | 78 × (512-dimensional latent + 64-dimensional RoPE key) | 87.75 KiB | 10.97 GiB | DSA indexer key cache, block tables, selected-token workspace; backend-specific latent packing. |
| DeepSeek-V4.1-Flash | Cross-layer shared, compressed global KV in model-defined FP4 storage | **890 B/token reported** | 0.109 GiB | FP4 scales/indexer metadata, active SWA buffers, host-side replay state, allocation overhead. |
| MiniMax-M3 | 60 × 2 × 4 × 128 BF16 GQA scalars | 120 KiB | 15.00 GiB | Sparse index features, selected block staging, vision cache. |
| Qwen3.8-Flash-Next | 12 × 2 × 2 × 256 BF16 GQA scalars | 24 KiB | 3.00 GiB | 36 GDN recurrent/conv states, sparse indexer, n-gram lookup state. |
| Kimi-K3 | 24 × (512 latent + 64 positional) BF16 scalars | 27 KiB | 3.375 GiB | 69 KDA recurrent/conv states, kernel scratch and backend-specific MLA packing. |

The GLM, MiniMax, Qwen, and Kimi figures are **calculations from linked configs and runtime cache-width evidence**, not measured peak HBM. `128 Ki tokens = 131,072`. At 1,048,576 tokens the same simple payload would be about **87.75, 120, 24, and 27 GiB**, respectively, per sequence before the excluded state. DeepSeek's 890 B/token is a **report claim** for global KV, not a measurement of total live GPU/NPU memory. The existing [DeepSeek V4.1 note](../docs/training/model-families/deepseek/deepseek-v4.1/index.md) details the FP4 format and 128-token SWA replay.

### Model-by-model state lifetime

| Model | Persistent across a prefix hit | Local or regenerable state | Critical optimization consequence |
| --- | --- | --- | --- |
| GLM-5.3 | MLA latent/RoPE entries and indexer history required for future sparse search. | Per-step top-k lists and scratch. | Quantize or tier latent and indexer caches separately; benchmark index selection and gather traffic. |
| DeepSeek-V4.1-Flash | Shared global KV and required indexer material; FP4 main KV. | 128-token SWA can be kept short term or rebuilt by bounded replay. | Split long-lived global and short-lived local cache policies; preserve CSA2 layer sharing during transfer. |
| MiniMax-M3 | Full token-wise GQA K/V history, even when only selected blocks are read per decode step. | Top-block selections and gather staging. | Sparse decode alone may improve bandwidth/compute but leave capacity limited; add cache precision or tiering if memory is the bottleneck. |
| Qwen3.8-Flash-Next | GQA history in 12 global layers and checkpoints of GDN/PLE state required for prefix reuse. | Incremental GDN state between checkpoints and sparse selection scratch. | A block-only KV allocator understates memory; checkpoint and restore recurrent state coherently with token pages. |
| Kimi-K3 | MLA latent/positional history in 24 global layers and KDA state at the reusable prefix boundary. | KDA update/conv workspace and current-token intermediates. | Prefix sharing must include KDA state snapshots; optimize token KV and recurrent state as different objects. |

## GPU vs Ascend NPU physical layout

The model config specifies **what information is cached**, while the serving backend decides tensor shape, page size, strides, and precision. The following is a deployment map, not a claim that all five checkpoints share one kernel layout.

| Concern | NVIDIA GPU / vLLM reference | Huawei Ascend / vLLM-Ascend reference | Deployment implication |
| --- | --- | --- | --- |
| Request mapping | Scheduler block table maps logical token blocks to physical blocks. The classic paged-attention kernel documents K `[blocks, kv_heads, head_dim/x, block_tokens, x]` and V `[blocks, kv_heads, head_dim, block_tokens]`. [vLLM design](https://docs.vllm.ai/en/latest/design/paged_attention/) | Same logical block IDs can address **strided** views with physical alignment padding; a documented attention view is `[block, kv_head, token, head_dim]`. [Ascend layout design](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/Design_Documents/non_contiguous_kv_cache.html) | Do not copy a raw GPU cache tensor and assume an NPU kernel can consume its strides. Serialize logical blocks or use a connector that handles layout conversion. |
| GQA (MiniMax, Qwen global layers) | Paginated K/V with backend-dependent in-block order. | Independent K/V views may have padding and distinct offsets in a shared allocation. [Ascend layout design](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/Design_Documents/non_contiguous_kv_cache.html) | Measure bytes allocated per block, not just tensor element counts. |
| MLA / compressed global KV (GLM, DeepSeek, Kimi) | Backend-specific latent, positional, indexer and sparse-gather buffers; plain GQA shape is inapplicable. | Ascend sparse/MLA kernels and quantized cache paths have model-specific contracts. [DeepSeek V4.1 guide](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/DeepSeek-V4.1-Flash.html) | Inventory each tensor by layer/group, dtype, shape, stride and lifetime before choosing offload/quantization. |
| Recurrent state (Qwen, Kimi) | Hybrid cache manager must account for state snapshots and token pages. | Ascend's non-contiguous design describes separate attention and Mamba-style state groups; Qwen prefix caching uses aligned GDN/PLE checkpoints. [Ascend layout](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/Design_Documents/non_contiguous_kv_cache.html), [Qwen guide](https://docs.vllm.ai/projects/ascend/en/main/tutorials/models/Qwen3.8-Flash-Next.html) | Test prefix-hit correctness and restored state, not just ordinary KV-block reuse. |
| Offload / tiering | GPU HBM → host memory or remote KV store, subject to connector and model support. | Ascend describes layerwise prefill offload and sparse decode offload with full main KV on host and hot indexer/top-k buffers on NPU; its design currently excludes hybrid cache layouts. [Ascend offload design](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/Design_Documents/layerwise_and_sparse_kv_cache_offloading.html) | Treat this as a candidate mainly for non-hybrid sparse models; validate support before applying it to Qwen/Kimi. |

An exact “GPU versus NPU byte layout” requires a **specific framework commit, attention backend, device generation, dtype and launch configuration**. The model does not prescribe a single universal physical layout. A useful trace records, for each cache tensor: `model/layer/group → semantic field → logical shape → physical shape/stride → dtype and scale format → residency tier → bytes allocated → bytes read per decode token`.

## Ascend deployment evidence and limits

| Model | Published Ascend route as of this snapshot | What to validate before a production claim |
| --- | --- | --- |
| GLM-5.3 | [GLM-5.3 experimental guide](https://docs.vllm.ai/projects/ascend/en/v0.27.1rc1/tutorials/models/GLM5.3.html). | Pin an image/commit that actually contains this architecture, then test C8/indexer kernels and cache precision on the target A3/A5 generation. |
| DeepSeek-V4.1-Flash | [Model guide](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/DeepSeek-V4.1-Flash.html) describes A2/A3 W8A8 and a main-branch requirement; V1 runner is required there. | Confirm exact checkpoint, FP4 cache behavior, SWA replay and prefix hit accuracy; do not assume a generic FP4 weight path implies FP4 KV support. |
| MiniMax-M3 | [Model guide](https://docs.vllm.ai/projects/ascend/en/main/tutorials/models/MiniMax-M3.html) includes colocated, P/D disaggregation and KV-pool recipes. | Measure full persistent GQA bytes, sparse indexer footprint, block-gather bandwidth, and A3/A5 kernel availability. |
| Qwen3.8-Flash-Next | [Experimental model guide](https://docs.vllm.ai/projects/ascend/en/main/tutorials/models/Qwen3.8-Flash-Next.html) documents `--mamba-cache-mode align` for prefix caching. | Measure GDN/PLE snapshot memory and correctness; do not treat experimental prefix caching as an unqualified capacity gain. |
| Kimi-K3 | [Model guide](https://docs.vllm.ai/projects/ascend/en/main/tutorials/models/Kimi-K3.html) covers A3 W4A8 text/multimodal serving and prefix cache. | Confirm MLA and KDA cache formats, snapshot semantics, and exact release/device compatibility. |

The Ascend [support matrix](https://docs.vllm.ai/projects/ascend/en/main/user_guide/support_matrix/supported_models.html) and model guides are versioned independently of checkpoint releases. Validate the particular image, CANN version, NPU generation, and checkpoint combination before quoting throughput or accuracy. The [KVPP design](https://docs.vllm.ai/projects/ascend/en/main/developer_guide/Design_Documents/kvpp.html) presently targets non-hybrid MLA/SFA eager execution, so it should not be presented as a ready-made solution for KDA/GDN hybrid models.

## Optimization work plan

1. **Collect a tensor inventory** on one pinned GPU stack and one pinned Ascend stack for each model: cache names, group/layer count, actual allocated bytes, stride, dtype, scale metadata and TP behavior. Compare the result with the payload floors above.
2. **Measure workload shapes**: 8K, 128K and near-limit prompts; 1 versus many concurrent requests; prefill and decode separately; shared-prefix hit and miss; text-only and multimodal. Report TTFT, inter-token latency, tokens/s, peak and steady HBM, host memory, cache hit rate, and transferred bytes.
3. **Choose the lever by bottleneck**: capacity → precision, page packing, prefix sharing, host tiering; decode bandwidth → sparse gather/indexer and locality; prefill compute → chunking or model-specific encoder/cache reuse; prefix reuse → coherent recurrent-state snapshots.
4. **Prototype per cache class**: GQA pages for MiniMax; MLA/indexer split for GLM; FP4 global versus SWA replay for DeepSeek; GDN/KDA state checkpoints plus periodic global KV for Qwen/Kimi. Keep baseline output/accuracy tests for each change.
5. **Compare useful capacity, not nominal bit rate**: active sequences × context at a fixed latency and quality target, including allocator padding, scales, temporary buffers, TP replication and transfer overhead. A low-bit format is a win only if the end-to-end kernel and memory hierarchy exploit it.

### Slide-ready takeaway

> **KV-cache optimization is now a heterogeneous state-placement problem.** MiniMax is dominated by retained token-wise GQA KV; GLM by latent plus sparse-index state; DeepSeek by shared compressed global KV and disposable local SWA; Qwen and Kimi by a combination of token pages and recurrent snapshots. GPU and Ascend can implement the same logical model with different physical pages and kernel contracts, so the first engineering artifact should be a measured per-tensor cache map on each target stack.

## Source trail and confidence

- **High confidence:** checkpoint-config layer counts, heads, ranks, sparse budgets and context limits from the five linked official Hugging Face configs; published backend layout contracts from [vLLM](https://docs.vllm.ai/en/latest/design/paged_attention/) and [vLLM-Ascend](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/Design_Documents/non_contiguous_kv_cache.html).
- **Medium confidence:** the calculated BF16 payload floors. They use model-level shapes and deliberately exclude kernel allocations, physical-page padding, scale metadata and parallel placement.
- **Report claim, not independently benchmarked here:** DeepSeek V4.1's 890 B/token global cache and its compression/replay behavior, summarized in the repository's [source-backed report note](../docs/training/model-families/deepseek/deepseek-v4.1/index.md). Kimi's hybrid behavior is also covered by the [K3 technical note](../docs/training/model-families/kimi/kimi-k3/index.md).
- **Not measured in this brief:** live GPU/NPU peak memory, per-model hardware performance, output quality after extra cache quantization, and exact physical strides for every backend. These require the pinned-stack inventory in the work plan.
