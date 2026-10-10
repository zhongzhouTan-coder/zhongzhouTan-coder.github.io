---
title: "Kimi-K3 cache path on vLLM GPU and Ascend"
summary: "Code-backed analysis of Kimi-K3 recurrent KDA state, paged MLA records, and Ascend model overrides."
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

# Kimi-K3 cache path on vLLM GPU and Ascend

**Pinned code:** [vLLM](https://github.com/vllm-project/vllm/tree/02aa10539f15c8c73aa7fd4a5ee0f691cbf88dcb) `02aa10539f15c8c73aa7fd4a5ee0f691cbf88dcb`; [vLLM Ascend](https://github.com/vllm-project/vllm-ascend/tree/51b0242c0d3be1d3bd59e763a7d78b8553cd45e1) `51b0242c0d3be1d3bd59e763a7d78b8553cd45e1`. Both checkouts were clean. This is static code analysis; no model or accelerator was run.

[Model cache study](index.md) · [vLLM](../vllm/index.md) · [vLLM Ascend](../vllm-ascend/index.md)

## Architecture and payload

The [presentation brief](../../../../presentation-notes/open-model-architecture-kv-cache-gpu-ascend-2026-10.md)'s `moonshotai/Kimi-K3` has 93 text layers: 69 Kimi Delta Attention (KDA) recurrent layers and 24 global MLA layers. KDA keeps sequence state and a short convolution window; MLA appends a compressed latent and positional key. At BF16 the token-dependent MLA payload is **24 × (512 + 64) × 2 = 27 KiB/token**. This excludes 69 layers of KDA state, snapshots, page overhead, scales, vision buffers, and MoE weights.

## Architecture blueprint

The [released Kimi-K3 config](https://huggingface.co/moonshotai/Kimi-K3/blob/main/config.json) wraps a multimodal model around a `kimi_linear` text trunk. Its 93 text layers are arranged mostly as three Kimi Delta Attention (KDA) layers followed by one global MLA layer, with the final MLA at layer 93. The config lists 69 KDA and 24 full-attention layers explicitly. Hidden width is 7,168. KDA has 96 heads and a short-convolution kernel of width four; MLA uses 96 query heads, a rank-512 KV latent and a 64-dimensional positional field. Kimi's no-position-encoding (NoPE) MLA route and output gate are explicit config choices, not generic MLA assumptions.

| Subsystem | Checkpoint recipe | State or compute consequence |
| --- | --- | --- |
| KDA | 69 layers, full-rank gate, convolution width four | Recurrent state and short-convolution state per sequence. |
| MLA | 24 layers, rank-512 KV plus 64 positional dimensions | Token-linear paged history only in these layers. |
| AttnRes | Residual block size 12 | Cross-layer residual accumulation; activation state differs from KV history. |
| Experts | First layer dense; then 16 of 896 routed experts plus two shared | Distributed weight and dispatch cost, not 896 active experts per token. |
| Multimodal/draft | Native vision wrapper; main config has zero next-token prediction layers | Image preprocessing and optional separate draft systems need distinct budgets. |

The GPU <a class="code-link" href="../../../../external-repos/vllm-02aa10539f15/vllm/models/kimi_k3/nvidia/model.py#L857" data-code-repo="vllm-02aa10539f15" data-code-path="vllm/models/kimi_k3/nvidia/model.py" data-code-line="857"><code>KimiDecoderLayer</code></a> reads `is_kda_layer` to instantiate the private full-rank KDA path or MLA, then chooses dense MLP versus <a class="code-link" href="../../../../external-repos/vllm-02aa10539f15/vllm/models/kimi_k3/nvidia/model.py#L540" data-code-repo="vllm-02aa10539f15" data-code-path="vllm/models/kimi_k3/nvidia/model.py" data-code-line="540"><code>KimiMoE</code></a> by layer index. It also branches into AttnRes logic when the checkpoint requests 12-layer residual blocks. On Ascend, <a class="code-link" href="../../../../external-repos/vllm-ascend-51b0242c0d3b/vllm_ascend/models/kimi_k3.py#L499" data-code-repo="vllm-ascend-51b0242c0d3b" data-code-path="vllm_ascend/models/kimi_k3.py" data-code-line="499"><code>AscendKimiDecoderLayer</code></a> preserves that KDA/MLA and dense/MoE decision, but substitutes Ascend KDA, MLA, and <a class="code-link" href="../../../../external-repos/vllm-ascend-51b0242c0d3b/vllm_ascend/models/kimi_k3.py#L259" data-code-repo="vllm-ascend-51b0242c0d3b" data-code-path="vllm_ascend/models/kimi_k3.py" data-code-line="259"><code>AscendKimiMoE</code></a> classes. The model architecture is shared; the kernels, precision contracts, and cache packing are not implied to be identical.

### Worked four-layer segment

At layers 1–3, a new token successively changes three KDA recurrent states and their short-convolution windows. There are no three new full-history K/V pages. At layer 4, the token creates a 576-element logical MLA record and appends it to a paged history. Attention reads prior MLA records, projects the result back to the hidden width, and passes it through the expert block. The next segment repeats this pattern. A reusable prefix must capture all KDA state snapshots at the same token boundary as the 24 MLA histories; otherwise the next token resumes from inconsistent contexts. AttnRes additionally changes how residual hidden streams are combined across blocks, but it does not turn them into per-token attention cache.

The [Kimi-K3 report note](../../../training/model-families/kimi/kimi-k3/index.md) covers training-side architecture and scale. Here, the architecture is connected to the exact GPU and NPU serving layer constructors and cache objects.

## GPU state transitions

1. <a class="code-link" href="../../../../external-repos/vllm-02aa10539f15/vllm/models/kimi_k3/nvidia/model.py#L1581" data-code-repo="vllm-02aa10539f15" data-code-path="vllm/models/kimi_k3/nvidia/model.py" data-code-line="1581"><code>KimiLinearForCausalLM</code></a> builds the hybrid decoder. <a class="code-link" href="../../../../external-repos/vllm-02aa10539f15/vllm/models/kimi_k3/nvidia/kda.py#L544" data-code-repo="vllm-02aa10539f15" data-code-path="vllm/models/kimi_k3/nvidia/kda.py" data-code-line="544"><code>KimiK3DeltaAttention</code></a> updates a recurrent matrix and convolution state rather than appending per-token K/V at every KDA layer. <a class="code-link" href="../../../../external-repos/vllm-02aa10539f15/vllm/models/kimi_k3/nvidia/kda.py#L829" data-code-repo="vllm-02aa10539f15" data-code-path="vllm/models/kimi_k3/nvidia/kda.py" data-code-line="829"><code>KDA MambaSpec</code></a> exposes that state to the hybrid cache planner; prefix reuse needs the state at the same token boundary as MLA pages.
2. An MLA layer <a class="code-link" href="../../../../external-repos/vllm-02aa10539f15/vllm/models/kimi_k3/nvidia/mla.py#L123" data-code-repo="vllm-02aa10539f15" data-code-path="vllm/models/kimi_k3/nvidia/mla.py" data-code-line="123"><code>MultiHeadLatentAttention</code></a> projects compressed KV and positional components. <a class="code-link" href="../../../../external-repos/vllm-02aa10539f15/vllm/models/kimi_k3/nvidia/mla.py#L417" data-code-repo="vllm-02aa10539f15" data-code-path="vllm/models/kimi_k3/nvidia/mla.py" data-code-line="417"><code>MLA cache spec</code></a> declares paged MLA storage; its `fp8_ds_mla` branch has a 656-byte backend record. That format is conditional and should not be mixed with the BF16 logical payload estimate.
3. Decode writes the new MLA record, reads historical latent rows, projects the attended result back to the decoder hidden width, and continues through the expert block. A KDA state update and an MLA page append must advance together when resuming a prefix.

## Ascend implementation

<a class="code-link" href="../../../../external-repos/vllm-ascend-51b0242c0d3b/vllm_ascend/models/__init__.py#L9" data-code-repo="vllm-ascend-51b0242c0d3b" data-code-path="vllm_ascend/models/__init__.py" data-code-line="9"><code>Kimi model registration</code></a> registers the text and multimodal model names, including a compatibility alias. <a class="code-link" href="../../../../external-repos/vllm-ascend-51b0242c0d3b/vllm_ascend/models/kimi_k3.py#L1107" data-code-repo="vllm-ascend-51b0242c0d3b" data-code-path="vllm_ascend/models/kimi_k3.py" data-code-line="1107"><code>AscendKimiLinearForCausalLM</code></a> specializes the upstream shell. <a class="code-link" href="../../../../external-repos/vllm-ascend-51b0242c0d3b/vllm_ascend/models/kimi_k3.py#L389" data-code-repo="vllm-ascend-51b0242c0d3b" data-code-path="vllm_ascend/models/kimi_k3.py" data-code-line="389"><code>AscendKimiMLAAttention</code></a> extends the upstream MLA class for DSpark RoPE metadata; this class alone does not specify the physical NPU cache layout. Logical KDA state plus MLA pages remain the two cache classes, while NPU layout and precision depend on the selected backend and flags. The [existing Kimi code map](../vllm/vllm-kimi-k3-code-reading.md) covers broader MoE, multimodal, and speculative paths; this page isolates cache ownership.

### NPU layer construction and state handoff

<a class="code-link" href="../../../../external-repos/vllm-ascend-51b0242c0d3b/vllm_ascend/models/kimi_k3.py#L499" data-code-repo="vllm-ascend-51b0242c0d3b" data-code-path="vllm_ascend/models/kimi_k3.py" data-code-line="499"><code>AscendKimiDecoderLayer</code></a> makes the KDA-versus-MLA decision per layer and chooses Ascend MoE or dense MLP in the same constructor. <a class="code-link" href="../../../../external-repos/vllm-ascend-51b0242c0d3b/vllm_ascend/models/kimi_k3.py#L757" data-code-repo="vllm-ascend-51b0242c0d3b" data-code-path="vllm_ascend/models/kimi_k3.py" data-code-line="757"><code>AscendKimiLinearModel</code></a> builds the complete layer stack, prepares embedding and final normalization ownership for pipeline-parallel ranks, and carries auxiliary hidden-state transport used by draft paths. The MLA subclass itself only extends upstream behavior for DSpark RoPE metadata; the actual NPU attention backend and page layout must be inspected at the selected launch configuration. KDA uses a separate recurrent-state path, so ordinary paged-KV transfer alone does not prove a valid prefix hit.

A correct cross-device cache inventory should report 24 MLA layer pages and 69 KDA state groups, then add state snapshots at reusable prefix boundaries. It should also distinguish AttnRes hidden-state accumulators, MoE dispatch buffers, and multimodal image features from attention cache. To validate an offload or prefill/decode transfer, compare next-token logits after a resumed prefix with a full uninterrupted run and inspect both the MLA block mapping and KDA state checksum at the handoff boundary.

| State | Lifetime | Deployment concern |
| --- | --- | --- |
| MLA latent plus positional records | Full retained prefix in 24 layers | Token-linear HBM and transfer bytes. |
| KDA matrix and convolution state | Per active sequence, checkpointed for reuse | Correct restoration at the page boundary. |
| MLA/KDA workspaces | Current step | Peak memory, not durable prefix capacity. |

**Validation boundary:** This is a static trace only. Test KDA state restoration after prefix hits, tensor-parallel placement, MLA cache dtype, and measured physical NPU/GPU bytes before sizing an offload tier.
