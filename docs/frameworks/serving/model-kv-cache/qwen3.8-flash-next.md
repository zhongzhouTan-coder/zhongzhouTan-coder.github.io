---
title: "Qwen3.8-Flash-Next cache path on vLLM GPU and Ascend"
summary: "GPU code analysis of Qwen3.8 hybrid GDN, QSA, and index caches with an explicit Ascend code-support boundary."
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

# Qwen3.8-Flash-Next cache path on vLLM GPU and Ascend

**Pinned code:** [vLLM](https://github.com/vllm-project/vllm/tree/02aa10539f15c8c73aa7fd4a5ee0f691cbf88dcb) `02aa10539f15c8c73aa7fd4a5ee0f691cbf88dcb`; [vLLM Ascend](https://github.com/vllm-project/vllm-ascend/tree/51b0242c0d3be1d3bd59e763a7d78b8553cd45e1) `51b0242c0d3be1d3bd59e763a7d78b8553cd45e1`. Both checkouts were clean. This is static code analysis; no model or accelerator was run.

[Model cache study](index.md) · [vLLM](../vllm/index.md) · [vLLM Ascend](../vllm-ascend/index.md)

## Architecture and cache classes

The [presentation brief](../../../../presentation-notes/open-model-architecture-kv-cache-gpu-ascend-2026-10.md)'s `Qwen/Qwen3.8-Flash-Next` uses `qwen4_exp` as its architecture type. It reports 48 text layers: 36 gated DeltaNet (GDN) recurrent layers and 12 Qwen Sparse Attention (QSA) layers. The 12 QSA layers each have two KV heads of width 256; **12 × 2 × 2 × 256 × 2 = 24 KiB/token** is the BF16 *main QSA K/V only*. Recurrent snapshots and index caches are extra. The vision tower, MoE weights, and prediction head are separate budgets.

## Architecture blueprint

The [released Qwen3.8-Flash-Next config](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/config.json) identifies a multimodal `qwen4_exp` checkpoint with a 48-layer, 2,560-wide text trunk. `layer_types` repeats three `linear_attention` layers followed by one `full_attention` layer 12 times. In this model the full-attention layers use Qwen Sparse Attention (QSA), because the config supplies indexer heads. The 36 Gated DeltaNet (GDN) layers maintain recurrent matrix and short-convolution state rather than one K/V pair per historical token. The 12 QSA layers keep conventional paged GQA K/V plus separate raw and compressed index keys.

| Component | Checkpoint value | Serving implication |
| --- | --- | --- |
| Hybrid attention | 36 GDN and 12 QSA layers | Prefix reuse needs both recurrent snapshots and token pages. |
| QSA main path | 24 query heads, two KV heads, 256-dimensional heads | Main K/V grows with sequence length only in QSA layers. |
| QSA index | Four index query heads, one index KV head, 128-dimensional keys, compression ratio four, 2,048-token budget | Raw partial groups and compressed key history have separate cache lifetimes. |
| GDN state | 16 key heads, 48 value heads, 128-dimensional state components, convolution width four | Fixed-size per-sequence state plus prefix snapshots; not token-wise pages. |
| Residual and experts | Four hyper-connection streams; 10 of 512 routed experts plus shared expert | Stream activations and MoE buffers are outside persistent attention cache. |
| PLE, vision, MTP | PLE at layer 2, n-gram size three; 27-layer vision tower; one MTP layer | Input processing and draft work add separate memory and correctness boundaries. |

At load time <a class="code-link" href="../../../../external-repos/vllm-02aa10539f15/vllm/models/qwen4_exp/nvidia/model.py#L366" data-code-repo="vllm-02aa10539f15" data-code-path="vllm/models/qwen4_exp/nvidia/model.py" data-code-line="366"><code>Qwen4ExpModel</code></a> builds the 48-layer recipe from `layer_types`. Each <a class="code-link" href="../../../../external-repos/vllm-02aa10539f15/vllm/models/qwen4_exp/nvidia/model.py#L175" data-code-repo="vllm-02aa10539f15" data-code-path="vllm/models/qwen4_exp/nvidia/model.py" data-code-line="175"><code>Qwen4ExpDecoderLayer</code></a> chooses GDN or QSA, optionally applies <a class="code-link" href="../../../../external-repos/vllm-02aa10539f15/vllm/models/qwen4_exp/nvidia/ple_layer.py#L64" data-code-repo="vllm-02aa10539f15" data-code-path="vllm/models/qwen4_exp/nvidia/ple_layer.py" data-code-line="64"><code>Qwen4ExpPLELayer</code></a>, then mixes four residual streams around attention and MoE. That order matters: PLE changes the hidden stream before attention, while GDN/QSA decide what persistent state is updated. <a class="code-link" href="../../../../external-repos/vllm-02aa10539f15/vllm/models/qwen4_exp/nvidia/indexer_qsa.py#L90" data-code-repo="vllm-02aa10539f15" data-code-path="vllm/models/qwen4_exp/nvidia/indexer_qsa.py" data-code-line="90"><code>QSAIndexer</code></a> owns the two QSA side caches, quantizes compressed keys as BF16 or FP8 according to indexer configuration, and emits a packed sparse selection whose last integer is a valid-count field rather than a token address.

### Prefix-hit example

After 1,024 prompt tokens, a reusable prefix contains QSA main pages and compressed index-key pages for its 12 sparse layers. It also needs the GDN state at that boundary and any open raw-key group needed to complete a four-token compression group. Resuming only the main QSA blocks would leave the recurrent state and index history inconsistent. On the next token, GDN layers update state; QSA layers append K/V and index keys, select historical positions, gather main pages, and return the updated hidden stream to the next layer. This is why a 24 KiB/token main-KV floor is not a total hybrid-cache budget.

The [Ascend model guide](https://docs.vllm.ai/projects/ascend/en/main/tutorials/models/Qwen3.8-Flash-Next.html) describes experimental W8A8 serving on Atlas A3 and aligned prefix caching for release `0.26.0rc`. The pinned Ascend checkout here does not include that release's dedicated model path. The guide is evidence of a versioned deployment claim; this page's code-backed NPU claims stop at what the pinned checkout establishes.

## GPU token trace

1. <a class="code-link" href="../../../../external-repos/vllm-02aa10539f15/vllm/models/qwen4_exp/nvidia/model.py#L175" data-code-repo="vllm-02aa10539f15" data-code-path="vllm/models/qwen4_exp/nvidia/model.py" data-code-line="175"><code>Qwen4ExpDecoderLayer</code></a> chooses a GDN or QSA branch per layer. GDN updates a fixed-size recurrent/convolution state; a reusable prefix needs a coherent state snapshot. <a class="code-link" href="../../../../external-repos/vllm-02aa10539f15/vllm/models/qwen4_exp/nvidia/model_state.py#L17" data-code-repo="vllm-02aa10539f15" data-code-path="vllm/models/qwen4_exp/nvidia/model_state.py" data-code-line="17"><code>Qwen4ExpModelState</code></a> connects the model to the hybrid state manager.
2. A QSA layer's <a class="code-link" href="../../../../external-repos/vllm-02aa10539f15/vllm/models/qwen4_exp/nvidia/qsa.py#L333" data-code-repo="vllm-02aa10539f15" data-code-path="vllm/models/qwen4_exp/nvidia/qsa.py" data-code-line="333"><code>Qwen4ExpQSAAttention</code></a> projects main Q/K/V and an index branch. <a class="code-link" href="../../../../external-repos/vllm-02aa10539f15/vllm/models/qwen4_exp/nvidia/qsa.py#L541" data-code-repo="vllm-02aa10539f15" data-code-path="vllm/models/qwen4_exp/nvidia/qsa.py" data-code-line="541"><code>QSA main cache spec</code></a> declares paged main K/V. The constructor currently requires BF16 model activations, BF16 or FP8-e4m3 main KV, and excludes context parallelism for QSA; these are code-version limits, not general architecture rules.
3. <a class="code-link" href="../../../../external-repos/vllm-02aa10539f15/vllm/models/qwen4_exp/common/qsa_cache.py#L811" data-code-repo="vllm-02aa10539f15" data-code-path="vllm/models/qwen4_exp/common/qsa_cache.py" data-code-line="811"><code>QSAKeyStateCache</code></a> holds the open raw-key group and <a class="code-link" href="../../../../external-repos/vllm-02aa10539f15/vllm/models/qwen4_exp/common/qsa_cache.py#L865" data-code-repo="vllm-02aa10539f15" data-code-path="vllm/models/qwen4_exp/common/qsa_cache.py" data-code-line="865"><code>QSACompressedKeyCache</code></a> holds compressed historical index keys. The indexer selects sparse positions into a transient buffer, then QSA gathers retained main K/V and returns an attention output. A top-k selection does not shrink the main history.

## Ascend evidence boundary

The presentation brief cites an experimental Ascend model guide and aligned prefix-state mode. In this pinned Ascend checkout, <a class="code-link" href="../../../../external-repos/vllm-ascend-51b0242c0d3b/vllm_ascend/models/__init__.py#L1" data-code-repo="vllm-ascend-51b0242c0d3b" data-code-path="vllm_ascend/models/__init__.py" data-code-line="1"><code>Ascend model registration</code></a> contains **no `Qwen4Exp` registration**, and a scoped search found no `qwen4_exp` model package. Consequently this revision does not establish a runnable NPU implementation for the selected checkpoint. A platform-specific Qwen3.5 or Qwen3.8-27B path cannot be substituted for this hybrid QSA model. Treat NPU cache layout, aligned GDN/PLE snapshots, and supported devices as **guide claims requiring a matching release checkout and live validation**.

### Versioned NPU validation path

The [Ascend guide](https://docs.vllm.ai/projects/ascend/en/main/tutorials/models/Qwen3.8-Flash-Next.html) says the checkpoint first appeared in its `0.26.0rc` release and documents Atlas A3 W8A8 text and multimodal runs. That statement and the pinned main-branch checkout answer different questions: the guide is a deployment recipe for its validated image; the checkout inspected here is a static source snapshot that lacks a dedicated `qwen4_exp` registration/package. An upstream generic registry path or release-specific patch could still exist outside this inspected scope. Therefore the NPU column below remains an evidence gap rather than a claim of universal incompatibility.

To close it, pin the exact guide image and matching vLLM/vLLM-Ascend commits, inspect the model registry, QSA kernel registration, GDN state cache, and PLE state handling, then run a prefix-hit test at a four-token index compression boundary. Record NPU cache tensor shape, stride, dtype, allocation bytes, and output equality for hit versus full replay. Only then can the GPU QSA 24 KiB/token floor be compared with an Ascend physical footprint.

| State | GPU evidence | Ascend status here |
| --- | --- | --- |
| QSA main K/V pages | Direct cache spec | No checkpoint-specific implementation found. |
| Raw and compressed QSA index keys | Separate owners | Physical layout unverified. |
| GDN/PLE state checkpoints | Hybrid model state | Prefix restoration unverified. |

**Next validation:** Pin the exact Ascend release advertised by the model guide, confirm registry and model code for `Qwen4Exp`, then record NPU tensor descriptors and a prefix-hit correctness trace.
