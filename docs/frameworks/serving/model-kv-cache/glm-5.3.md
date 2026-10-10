---
title: "GLM-5.3 cache path on vLLM GPU and Ascend"
summary: "Code-backed analysis of GLM-5.3 sparse MLA cache ownership, index selection, and GPU versus Ascend limits."
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

# GLM-5.3 cache path on vLLM GPU and Ascend

**Pinned code:** [vLLM](https://github.com/vllm-project/vllm/tree/02aa10539f15c8c73aa7fd4a5ee0f691cbf88dcb) `02aa10539f15c8c73aa7fd4a5ee0f691cbf88dcb`; [vLLM Ascend](https://github.com/vllm-project/vllm-ascend/tree/51b0242c0d3be1d3bd59e763a7d78b8553cd45e1) `51b0242c0d3be1d3bd59e763a7d78b8553cd45e1`. Both checkouts were clean. This is static code analysis; no model or accelerator was run.

[Model cache study](index.md) · [vLLM](../vllm/index.md) · [vLLM Ascend](../vllm-ascend/index.md)

## Model identity and architecture

The presentation brief selects `zai-org/GLM-5.3`, a text decoder with 78 layers, multi-head latent attention (MLA), and sparse history selection. Its reported latent plus positional-key width is 512 + 64; 78 × 576 × 2 bytes gives a **87.75 KiB/token BF16 main-cache payload floor**. This calculation excludes indexer history, page alignment, scales, block tables, parallel placement, and MoE weights. These checkpoint figures come from the [presentation brief](../../../../presentation-notes/open-model-architecture-kv-cache-gpu-ascend-2026-10.md); the code checkouts do not contain the checkpoint config.

The selected checkpoint belongs to the `glm_moe_dsa` / DeepSeek-V3.2 family path. <a class="code-link" href="../../../../external-repos/vllm-02aa10539f15/vllm/models/deepseek_v32/__init__.py#L18" data-code-repo="vllm-02aa10539f15" data-code-path="vllm/models/deepseek_v32/__init__.py" data-code-line="18"><code>GlmMoeDsa alias</code></a> identifies the GPU alias and <a class="code-link" href="../../../../external-repos/vllm-02aa10539f15/vllm/models/deepseek_v32/nvidia/model.py#L169" data-code-repo="vllm-02aa10539f15" data-code-path="vllm/models/deepseek_v32/nvidia/model.py" data-code-line="169"><code>DeepseekV32Model</code></a> constructs its decoder. The similarly named `glm5next` package in these revisions is for **GLM-5.3-Flash**, which has recurrent layers and kpool compression; its cache model must not be applied to `zai-org/GLM-5.3`.

## Architecture blueprint

The [released GLM-5.3 configuration](https://huggingface.co/zai-org/GLM-5.3/blob/main/config.json) names `GlmMoeDsaForCausalLM` and `glm_moe_dsa`. Its 78-layer text decoder has hidden width 6,144 and 64 attention query heads. A query first passes through a rank-2,048 query bottleneck, then separates 192 non-positional and 64 rotary dimensions per head. Historical K/V is stored as a rank-512 latent plus a 64-dimensional positional key; the full 256-dimensional value vector per head is reconstructed for computation rather than retained as a conventional per-head value cache. This is why `num_key_value_heads: 64` must not be read as a GQA cache-width formula.

| Subsystem | Checkpoint recipe | Why it matters during serving |
| --- | --- | --- |
| Decoder/MLA | 78 layers; 64 query heads; latent 512 + positional key 64 | The long-lived main history is latent, not expanded per-head K/V. |
| Sparse index | 32 index query heads of width 128; 2,048 selected positions | Every decode query scans index history but gathers only selected MLA records. |
| Index reuse | `index_topk_freq: 4`, skip offset 3, plus `full`/`shared` indexer types | Index owners refresh the selection; following layers may consume the shared result. |
| Feed-forward | First three dense; later layers route to eight of 256 experts plus one shared expert | Weight/communication cost is distinct from attention-cache capacity. |
| Context and draft | Up to 1,048,576 positions; one next-token prediction layer | A draft step can reuse the target index selection but adds its own work. |

The GPU <a class="code-link" href="../../../../external-repos/vllm-02aa10539f15/vllm/models/deepseek_v32/nvidia/model.py#L57" data-code-repo="vllm-02aa10539f15" data-code-path="vllm/models/deepseek_v32/nvidia/model.py" data-code-line="57"><code>DeepseekV32DecoderLayer</code></a> chooses dense MLP versus routed MoE from `first_k_dense_replace` and `moe_layer_freq`, applies input normalization, runs attention, then the feed-forward block and residual return. Its <a class="code-link" href="../../../../external-repos/vllm-02aa10539f15/vllm/models/deepseek_v32/attention.py#L121" data-code-repo="vllm-02aa10539f15" data-code-path="vllm/models/deepseek_v32/attention.py" data-code-line="121"><code>DeepseekV32Attention</code></a> constructs MLA and decides whether a layer owns an indexer from frequency/pattern configuration. <a class="code-link" href="../../../../external-repos/vllm-02aa10539f15/vllm/models/deepseek_v32/attention.py#L50" data-code-repo="vllm-02aa10539f15" data-code-path="vllm/models/deepseek_v32/attention.py" data-code-line="50"><code>DeepseekV32Indexer</code></a> projects and retains index keys. The checkpoint's explicit `indexer_types` list is descriptive; this pinned GPU constructor reads `index_topk_pattern` or the frequency/offset fields. The visible `full`/`shared` sequence and those fields agree for the selected config, but a custom checkpoint should be checked against the actual constructor rather than assuming the list is consumed directly.

### Two-layer worked example

Take a decode token at an index-owner layer. Its normalized hidden state creates a query latent and a new main MLA record. The index branch writes an index key, scores eligible historical keys, and publishes up to 2,048 logical token positions. Sparse MLA maps those positions to paged main-cache slots, computes the attended vector, and the layer's routed MoE transforms it. At the next shared-index layer, that layer still creates its **own** MLA record and runs its own attention and experts, but can reuse the selected positions. No step deletes the unselected old records: a future query could choose them.

The [GLM-5.2 GPU trace](../vllm/glm-5.2-inference-path.md) supplies the broader V1 request and return path. This page focuses on the GLM-5.3 checkpoint's architecture and cache ownership; its kernel choice remains dependent on CUDA capability, cache dtype, block size, and launch flags.

## One decode token: GPU

1. The scheduler supplies token rows and paged slots to the model. <a class="code-link" href="../../../../external-repos/vllm-02aa10539f15/vllm/models/deepseek_v32/nvidia/model.py#L169" data-code-repo="vllm-02aa10539f15" data-code-path="vllm/models/deepseek_v32/nvidia/model.py" data-code-line="169"><code>DeepseekV32Model</code></a> builds the decoder layers and a model-wide index-group builder.
2. An index-owner layer writes the token's index key to its separate paged index cache and selects historical positions. <a class="code-link" href="../../../../external-repos/vllm-02aa10539f15/vllm/models/deepseek_v32/nvidia/model.py#L190" data-code-repo="vllm-02aa10539f15" data-code-path="vllm/models/deepseek_v32/nvidia/model.py" data-code-line="190"><code>shared top-k buffer</code></a> allocates the transient `int32` top-k buffer shared with sparse MLA and the draft proposer. Reusing a top-k buffer is different from sharing every layer's long-lived index keys.
3. MLA writes a compressed latent plus positional key for each token to the main cache. Sparse decode maps the selected logical positions through block tables, reads main-cache rows, produces attention output, and then the decoder executes its MoE path. The top-k list bounds the *read set*; it does not evict unselected MLA rows.

## Ascend path and physical memory

<a class="code-link" href="../../../../external-repos/vllm-ascend-51b0242c0d3b/vllm_ascend/models/__init__.py#L74" data-code-repo="vllm-ascend-51b0242c0d3b" data-code-path="vllm_ascend/models/__init__.py" data-code-line="74"><code>Ascend GLM registration</code></a> replaces the GLM family shell with <a class="code-link" href="../../../../external-repos/vllm-ascend-51b0242c0d3b/vllm_ascend/models/deepseek_mtp.py#L74" data-code-repo="vllm-ascend-51b0242c0d3b" data-code-path="vllm_ascend/models/deepseek_mtp.py" data-code-line="74"><code>AscendGlmMoeDsaForCausalLM</code></a>. The logical split remains main MLA pages, indexer history, and current-step positions. Ascend cache precision and page stride are backend configuration decisions; a 576-element logical record is not a physical NPU byte stride. The earlier [Ascend GLM-5.2 trace](../vllm-ascend/glm-5.2-inference-path.md) explains the sparse attention handoff and C8 switches, but this new revision alone does not prove that the selected GLM-5.3 checkpoint uses a particular C8 format or offload policy.

### Ascend code handoff

The plugin's GLM-specific shell is thin: it inherits the upstream family model and mainly adapts weight names and expert-layer accounting. Attention specialization happens in backend components. <a class="code-link" href="../../../../external-repos/vllm-ascend-51b0242c0d3b/vllm_ascend/attention/indexer.py#L97" data-code-repo="vllm-ascend-51b0242c0d3b" data-code-path="vllm_ascend/attention/indexer.py" data-code-line="97"><code>AscendSFAIndexerBackend</code></a> builds and executes the paged sparse-index path, while <a class="code-link" href="../../../../external-repos/vllm-ascend-51b0242c0d3b/vllm_ascend/attention/sfa_v1.py#L792" data-code-repo="vllm-ascend-51b0242c0d3b" data-code-path="vllm_ascend/attention/sfa_v1.py" data-code-line="792"><code>AscendSFAImpl</code></a> consumes selected positions and MLA cache metadata for sparse attention. This split is important when profiling: a slow decode token can be dominated by index scoring, selected-row gathering, or expert communication even when the main MLA payload is relatively compact.

The NPU implementation therefore needs two measurements for a capacity decision: allocated bytes for main MLA pages and for the indexer cache. For a bandwidth decision, measure index scan bytes, selected main-cache read bytes, and page-table/locality overhead. C8 main-cache and C8 indexer switches are independent in the [existing Ascend analysis](../vllm-ascend/glm-5.1-c8-950.md); this static GLM-5.3 reading does not establish that either switch is enabled for the selected deployment. Optional sparse offload must likewise be treated as a separate configuration rather than a property of the checkpoint.

| State | Lifetime | Capacity implication |
| --- | --- | --- |
| MLA latent and positional field | Entire retained prefix, by attention layer | Dominant main-cache payload floor; page/scale overhead adds bytes. |
| Indexer key history | Entire searchable prefix at owner layers | Budget separately from main MLA. |
| Top-k positions | One query or speculative step | Workspace, not replacement for either historical cache. |

**Validation boundary:** Confirm the checkpoint's `architectures`, index-owner pattern, backend, KV dtype, and NPU tensor strides in a live deployment. Compare per-cache allocated bytes and indexer scan time before choosing quantization or offload.
