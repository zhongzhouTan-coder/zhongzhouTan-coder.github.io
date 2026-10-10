---
title: "MiniMax-M3 cache path on vLLM GPU and Ascend"
summary: "Code-backed analysis of MiniMax-M3 full GQA history, sparse block indexing, and NPU cache differences."
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

# MiniMax-M3 cache path on vLLM GPU and Ascend

**Pinned code:** [vLLM](https://github.com/vllm-project/vllm/tree/02aa10539f15c8c73aa7fd4a5ee0f691cbf88dcb) `02aa10539f15c8c73aa7fd4a5ee0f691cbf88dcb`; [vLLM Ascend](https://github.com/vllm-project/vllm-ascend/tree/51b0242c0d3be1d3bd59e763a7d78b8553cd45e1) `51b0242c0d3be1d3bd59e763a7d78b8553cd45e1`. Both checkouts were clean. This is static code analysis; no model or accelerator was run.

[Model cache study](index.md) · [vLLM](../vllm/index.md) · [vLLM Ascend](../vllm-ascend/index.md)

## Architecture and payload

The [presentation brief](../../../../presentation-notes/open-model-architecture-kv-cache-gpu-ascend-2026-10.md)'s `MiniMaxAI/MiniMax-M3` has 60 text layers, four KV heads of width 128, and sparse attention after the first three layers. Every layer projects conventional grouped-query attention (GQA) K and V. At BF16 the main payload is **60 × 2 × 4 × 128 × 2 = 120 KiB/token**, before sparse indexer keys, page padding, vision state, and tensor-parallel replication. The 16 selected content blocks are a per-query *read budget*, not a retained-cache cap.

## Architecture blueprint

The [released MiniMax-M3 config](https://huggingface.co/MiniMaxAI/MiniMax-M3/blob/main/config.json) contains a 60-layer text trunk with hidden width 6,144, 64 query heads, four KV heads, and head width 128. Layers 0–2 use full attention and dense feed-forward networks. Layers 3–59 use MiniMax Sparse Attention (MSA) and a routed expert network. Q/K use per-head normalization and partial rotary position encoding over 64 of the 128 head dimensions; the decoder uses Gemma-style RMS normalization. This is a conventional GQA K/V history with an *additional* sparse index branch, unlike GLM's compressed MLA history.

| Component | Checkpoint value | Runtime consequence |
| --- | --- | --- |
| Main attention | 64 Q / 4 KV heads, dimension 128 | Four K vectors and four V vectors retained per token per layer. |
| MSA index | Dimension 128, four index query heads, 128-token blocks | Sparse layer scores blocks before reading main K/V. |
| Selection | Top 16 content blocks plus required initial/local blocks | Read bandwidth is bounded; retained history is not. |
| Feed-forward | 128 routed experts, four selected, one shared; first three layers dense | Expert weights and dispatch buffers are separate from KV pages. |
| Multimodal/draft | 32-layer vision tower, patch merging, one next-token prediction layer | Image/video tokens affect prefill length; the vision tower is not decoder KV history. |

The GPU <a class="code-link" href="../../../../external-repos/vllm-02aa10539f15/vllm/models/minimax_m3/nvidia/model.py#L678" data-code-repo="vllm-02aa10539f15" data-code-path="vllm/models/minimax_m3/nvidia/model.py" data-code-line="678"><code>MiniMaxM3DecoderLayer</code></a> independently chooses sparse versus full attention and MoE versus dense MLP from two layer masks. It normalizes the input, runs attention, applies post-attention normalization, then runs <a class="code-link" href="../../../../external-repos/vllm-02aa10539f15/vllm/models/minimax_m3/nvidia/model.py#L195" data-code-repo="vllm-02aa10539f15" data-code-path="vllm/models/minimax_m3/nvidia/model.py" data-code-line="195"><code>MiniMaxM3MoE</code></a> on sparse layers. The NPU <a class="code-link" href="../../../../external-repos/vllm-ascend-51b0242c0d3b/vllm_ascend/models/minimax_m3/minimax_m3.py#L866" data-code-repo="vllm-ascend-51b0242c0d3b" data-code-path="vllm_ascend/models/minimax_m3/minimax_m3.py" data-code-line="866"><code>MiniMaxM3DecoderLayer</code></a> makes the same architectural choices from its config but instantiates Ascend attention and expert classes. This correspondence is stronger than assuming a GPU kernel layout carries over: both model paths implement the same layer recipe while their cache specs and kernels differ.

### A block-selection example

Suppose the current query is at token 4,096. The index branch scores 128-token historical blocks and selects up to 16 content blocks, while preserving the mandatory initial/local blocks. The sparse backend maps these block IDs through the request block table and gathers the corresponding main K/V pages. It returns an attended vector to the decoder layer, which applies normalization, routed experts, and residual handling. At token 4,097 the indexer can select a different set; therefore all historical K/V and index keys needed for future search remain allocated. The optimization opportunity is lower *decode read traffic*, not automatic context-capacity reduction.

The BF16 120 KiB/token main-cache estimate is a logical lower bound. In 57 sparse layers an additional 128-dimensional index key per token would add 14.25 KiB/token at BF16 if one such vector is retained per layer. That estimate is conditional on the checkpoint layout and excludes scales, alignment, first/local block metadata, and per-rank sharding.

## GPU production, lookup, and return

1. <a class="code-link" href="../../../../external-repos/vllm-02aa10539f15/vllm/models/minimax_m3/nvidia/model.py#L392" data-code-repo="vllm-02aa10539f15" data-code-path="vllm/models/minimax_m3/nvidia/model.py" data-code-line="392"><code>MiniMaxM3SparseAttention</code></a> fuses ordinary QKV and index Q/K projections in a sparse layer. The dense first layers use ordinary attention.
2. <a class="code-link" href="../../../../external-repos/vllm-02aa10539f15/vllm/models/minimax_m3/nvidia/model.py#L562" data-code-repo="vllm-02aa10539f15" data-code-path="vllm/models/minimax_m3/nvidia/model.py" data-code-line="562"><code>main KV cache spec</code></a> declares the main paged K/V storage. <a class="code-link" href="../../../../external-repos/vllm-02aa10539f15/vllm/models/minimax_m3/common/indexer.py#L102" data-code-repo="vllm-02aa10539f15" data-code-path="vllm/models/minimax_m3/common/indexer.py" data-code-line="102"><code>MiniMaxM3IndexerCache</code></a> declares a separate index-key side cache. Both histories persist so later queries may select old blocks.
3. The indexer scores block summaries and produces selected block IDs. <a class="code-link" href="../../../../external-repos/vllm-02aa10539f15/vllm/models/minimax_m3/nvidia/sparse_attention_msa.py#L185" data-code-repo="vllm-02aa10539f15" data-code-path="vllm/models/minimax_m3/nvidia/sparse_attention_msa.py" data-code-line="185"><code>MiniMaxM3SparseMSAImpl</code></a> uses the selection to read main K/V pages and return the attended hidden state. The decoder's output projection and MoE then continue; no eviction follows solely from not selecting a page this step.

## Ascend mapping

<a class="code-link" href="../../../../external-repos/vllm-ascend-51b0242c0d3b/vllm_ascend/models/__init__.py#L42" data-code-repo="vllm-ascend-51b0242c0d3b" data-code-path="vllm_ascend/models/__init__.py" data-code-line="42"><code>MiniMaxM3 registration</code></a> selects the NPU model. <a class="code-link" href="../../../../external-repos/vllm-ascend-51b0242c0d3b/vllm_ascend/models/minimax_m3/minimax_m3.py#L198" data-code-repo="vllm-ascend-51b0242c0d3b" data-code-path="vllm_ascend/models/minimax_m3/minimax_m3.py" data-code-line="198"><code>MiniMaxM3SparseAttention</code></a> owns its main attention cache and <a class="code-link" href="../../../../external-repos/vllm-ascend-51b0242c0d3b/vllm_ascend/models/minimax_m3/msa_m3.py#L202" data-code-repo="vllm-ascend-51b0242c0d3b" data-code-path="vllm_ascend/models/minimax_m3/msa_m3.py" data-code-line="202"><code>AscendMiniMaxM3IndexerCache</code></a> owns the indexer cache. <a class="code-link" href="../../../../external-repos/vllm-ascend-51b0242c0d3b/vllm_ascend/models/minimax_m3/msa_m3.py#L930" data-code-repo="vllm-ascend-51b0242c0d3b" data-code-path="vllm_ascend/models/minimax_m3/msa_m3.py" data-code-line="930"><code>AscendMiniMaxM3SparseImpl</code></a> maps selected history to the Ascend sparse kernel. The same logical K/V and index histories therefore exist on both platforms, but the NPU's page packing, dtype, and strides must be measured from the selected backend. A GPU tensor's packed K/V last dimension is not a portable NPU serialization format.

The NPU main owner returns `FullAttentionSpec`; its side owner returns `AscendSFAIndexerCacheSpec`. Thus a cache inventory should report two page groups, not a single MiniMax tensor. The code exposes each spec's block size, head count, dtype, and quantization mode, but allocator padding and active physical stride still require live tensor descriptors.

### Prefill and decode on Ascend

<a class="code-link" href="../../../../external-repos/vllm-ascend-51b0242c0d3b/vllm_ascend/models/minimax_m3/msa_m3.py#L951" data-code-repo="vllm-ascend-51b0242c0d3b" data-code-path="vllm_ascend/models/minimax_m3/msa_m3.py" data-code-line="951"><code>AscendMiniMaxM3SparseImpl.forward</code></a> branches on actual prefill and decode token counts. Decode passes selected block IDs, the block table, sequence lengths, and an optional FP8 dequantization scale to `minimax_m3_sparse_attn_decode`; prefill calls the corresponding sparse attention path with cumulative query lengths and context lengths. The output tensor then returns to the NPU decoder, which applies post-attention normalization and its dense or routed expert block. This is a complete attention handoff: index results become physical page reads, then the attended hidden state returns to the layer.

For performance experiments, separate three effects. Index scoring grows with searchable history and may dominate before the sparse read. Selected-page locality controls how efficiently the NPU kernel gathers main K/V. Finally, the main and index caches both occupy device memory even when only 16 content blocks are read. A useful measurement reports indexer time, sparse attention time, allocated bytes per cache group, and selected-block reuse across adjacent decode tokens.

| Object | Retention | Cost or failure mode |
| --- | --- | --- |
| Main K/V | All retained positions in 60 layers | Capacity remains linear in context despite sparse reads. |
| Indexer keys | Searchable positions in sparse layers | Extra long-lived cache; omitting it understates memory. |
| Block IDs and gather workspace | Current step | Bandwidth/locality cost, small persistent capacity. |

**Validation boundary:** Code was inspected statically. Verify the checkpoint's exact sparse-layer mask and backend selected on the target GPU/NPU. Measure main and indexer allocations independently, then compare sparse-gather latency against HBM pressure.
