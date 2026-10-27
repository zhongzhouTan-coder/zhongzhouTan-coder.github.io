---
title: "DeepSeek-V4 Inference on Ascend: The DSA Serving Stack in vllm-ascend"
summary: "How vllm-ascend runs DeepSeek-V4 on Ascend NPUs: mHC, hybrid c4/c128 compression, DSA attention, heterogeneous caches, and MTP/DSpark drafting, including the A3 0731 and 950/A5 branch differences."
layout: default
redirect_from:
  - /frameworks/vllm-ascend/deepseek-v4-inference/
confidence: high
code_links: strict
code_evidence: strict
sources:
  - raw/frameworks/serving/vllm-ascend/vllm-ascend-codebase--github-32a59d4e349c.md
  - derived/repo-analysis/frameworks/serving/vllm-ascend/vllm-ascend/32a59d4e349c12c32cdbc1916436c16e39939afc/important-files.md
  - raw/frameworks/serving/vllm-ascend/vllm-ascend-codebase--github-51b0242c0d3b.md
  - derived/repo-analysis/frameworks/serving/vllm-ascend/vllm-ascend/51b0242c0d3be1d3bd59e763a7d78b8553cd45e1/important-files.md
  - derived/repo-analysis/frameworks/serving/vllm-ascend/vllm-ascend/51b0242c0d3be1d3bd59e763a7d78b8553cd45e1/deepseek-v4-flash-0731-runtime-flow.md
  - raw/frameworks/serving/vllm-ascend/vllm-ascend-codebase--github-7702ccd7d8de.md
  - derived/repo-analysis/frameworks/serving/vllm-ascend/vllm-ascend/7702ccd7d8dea6b4dabdacb0118adb522dedbec7/deepseek-v4-flash-0731-deployment-evidence.md
  - raw/frameworks/serving/vllm-ascend/vllm-ascend-codebase--github-8405ff749d05.md
  - derived/repo-analysis/frameworks/serving/vllm-ascend/vllm-ascend/8405ff749d05aee320923d7d1dd9ad24c9246d4c/important-files.md
  - raw/frameworks/serving/vllm-ascend/vllm-ascend-codebase--github-8405ff749d05.md
  - derived/repo-analysis/frameworks/serving/vllm-ascend/vllm-ascend/8405ff749d05aee320923d7d1dd9ad24c9246d4c/important-files.md
updated: 2026-10-27
---

# DeepSeek-V4 Inference on Ascend: The DSA Serving Stack in vllm-ascend

**Repository:** [vllm-project/vllm-ascend](https://github.com/vllm-project/vllm-ascend) @ `32a59d4e349c12c32cdbc1916436c16e39939afc` (main, clean, inspected 2026-08-05)

**Supplementary evidence:** Section 4's C4/C128 compression clarification uses vLLM-Ascend `8405ff749d05aee320923d7d1dd9ad24c9246d4c` (main, clean, inspected 2026-10-27). Section 14's DSpark execution path uses `51b0242c0d3be1d3bd59e763a7d78b8553cd45e1` (inspected 2026-10-10), and its deployment/support statement uses `7702ccd7d8dea6b4dabdacb0118adb522dedbec7` (inspected 2026-08-27). These are static readings; no A3 or 950 runtime was executed.

**Related pages:** [vLLM Ascend Hub](index.md), [vLLM-Ascend Architecture](architecture.md), [DeepSeek-V4 Lightning Indexer C8 Quantization](deepseek-v4-lightning-indexer-c8.md), [DeepSeek-V4 Attention Code Reading](../vllm/deepseek/v4-attention-code-reading.md), [DeepSeek-V4: Million-Token Context](../../../training/model-families/deepseek/deepseek-v4/index.md), [DeepSeek-V3.2 Sparse Attention](../../../algorithms/model-architectures/deepseek-v3-2/index.md)

## TL;DR

**What:** vllm-ascend replaces upstream vLLM's DeepSeek-V4 model and attention backend with Ascend-specific implementations — `AscendDeepseekV4ForCausalLM` plus the `AscendDSA` (DeepSeek Sparse Attention) backend — so a hybrid c4/c128 compressed-attention model with mHC [hyper-connections](../../../terms/hyper-connections.md) and MTP runs entirely through NPU custom operators.

**How:** A model override wires the MLA-style query/KV prologue, per-layer compressor, and [Lightning Indexer](../../../terms/lightning-indexer.md) into the DSA backend; the backend selects device-specific sparse-attention and cache paths, while mHC wraps the decoder and MTP or DSpark supplies the configured draft path.

**The number:** One serving stack replaces three upstream layers at once — model class, attention backend, and [KV cache](../../../terms/kv-cache.md) specs — and its heterogeneous cache holds five KV types whose per-type page sizes and dtypes are chosen per device family (910B/A2/A3 vs A5).

## The Big Picture

[Mermaid source](assets/dsv4-inference-runtime.mmd)

```mermaid
flowchart TD
    subgraph MODEL["AscendDeepseekV4ForCausalLM (deepseek_v4.py)"]
        A["VocabParallelEmbedding<br/>embed_tokens"] --> B["DeepseekV4Model.forward"]
        B --> C["hc_head: mHC merge (npu_hc_pre/post per layer)"]
        C --> D["Decoder layers (interleaved c4 / c128 / swa)"]
        D --> E["RMSNorm + lm_head + LogitsProcessor"]
    end

    subgraph LAYER["DeepseekV2DecoderLayer (per layer)"]
        F["hc_pre (npu_hc_pre_v2)"] --> G["input_layernorm"]
        G --> H["DeepseekV4Attention.forward"]
        H --> I["hc_post (npu_hc_post)"]
        I --> J["post_attention_layernorm"]
        J --> K["DeepseekV4MoE.forward (FusedMoE)"]
        K --> L["hc_post"]
    end

    H --> ATTN["AscendDeepseekSparseAttention<br/>(ops/dsa.py) -> torch.ops.vllm.dsa_forward"]
    ATTN --> IMPL["AscendDSAImpl.forward (dsa_v1.py)"]
    IMPL --> PREFILL["_forward_prefill"]
    IMPL --> DECODE["_forward_decode"]
    PREFILL --> MLA["_mla_prolog_multistream: wq_a/wkv/wq_b CV split<br/>q + kv + rope + SWA scatter"]
    DECODE --> MLA
    MLA --> COMP["torch.ops._C_ascend.compressor<br/>inline compress + state update"]
    COMP --> INDEXER["indexer_select_qli / npu_vllm_quant_lightning_indexer<br/>(c4 layers only)"]
    INDEXER --> SPARSE["npu_sparse_attn_sharedkv<br/>SWA ori_kv + cmp_kv + topk indices"]
    SPARSE --> OPROJ["_forward_o_proj: inv-RoPE + wo_a + wo_b"]

    subgraph KV["Heterogeneous KV cache (per layer group)"]
        SWA_C["AscendDeepseekV4SWACache<br/>sliding-window tokens"]
        STATE_C["AscendCompressorStateCache<br/>fp32 partial compression states"]
        MLA_C["AscendMLAAttentionSpec main cache<br/>compressed KV entries"]
        IDX_C["AscendDeepseekV4IndexerCache<br/>8-bit indexer keys + scales"]
    end
    MLA --> SWA_C
    COMP --> STATE_C
    MLA --> MLA_C
    INDEXER --> IDX_C
```

*Synthesized runtime flow, not a source figure. ① The registered model runs a standard embed → decoder-layer → mHC-head → LM-head chain. ② Each decoder layer brackets attention and MoE with mHC `hc_pre`/`hc_post` custom ops. ③ Attention delegates to the DSA backend through a custom `dsa_forward` op, which splits prefill and decode. ④ Per step the MLA prologue builds q/kv, the compressor folds tokens into compressed entries, the indexer picks top-k blocks (c4 layers), and `npu_sparse_attn_sharedkv` scores SWA + compressed KV. ⑤ Five KV cache types live in per-layer groups with device-specific page sizes.*

*Branch note:* The figure shows the base DSA implementation. With `enable_dsa_cp`, the same attention layer dispatches through the context-parallel implementation; the documented A3 0731 branch is described in Section 14.

## Why This Exists

DeepSeek-V4 is a hybrid model: some layers compress KV by 4× and attend sparsely through a Lightning Indexer ("c4"/CSA layers), others compress by 128× and attend densely ("c128"/HCA layers), and every layer sits inside Manifold-Constrained Hyper-Connections (mHC) that multiply the residual stream by `hc_mult` before and after attention and FFN. At a 1M-token context, the SWA cache still holds the raw recent tokens, the compressor must fold every 4th/128th token into a state, and the indexer must score thousands of compressed blocks per query.

Upstream vLLM serves this on NVIDIA GPUs with FlashMLA/FlashInfer/Triton kernels and a CUDA-centric KV coordinator. On an Ascend NPU, none of that maps directly: the kernels are different (`npu_sparse_attn_sharedkv`), the KV paging layout is different (PA_ND vs PA_BSND, device-specific block sizes), the multi-stream overlap model is different (NPU streams + ACL graph capture), and even mHC needs custom fused operators. Without a dedicated port, DeepSeek-V4 simply could not run on Ascend hardware.

The vllm-ascend port solves this with three coordinated replacements: a **model override** (`AscendDeepseekV4ForCausalLM`), a **DSA attention backend** (`AscendDSABackend`), and **Ascend KV cache specs** (`AscendMLAAttentionSpec` / `AscendSlidingWindowMLASpec`) that teach the vLLM KV coordinator how to allocate and page the five cache types.

## The Landscape

[Mermaid source](assets/dsv4-inference-landscape.mmd)

```mermaid
flowchart TD
    A["DeepSeek-V2 MLA (low-rank KV latent)<br/>vllm/vllm-ascend AscendMLABackend"] --> B["DeepSeek-V3 MLA serving"]
    B --> C["DeepSeek-V3.2 DSA sparse attention (lightning indexer over tokens)<br/>AscendSFABackend + npu_lightning_indexer*"]
    C --> D["DeepSeek-V4 hybrid CSA/HCA (compression + sparsity)<br/>AscendDSABackend + npu_sparse_attn_sharedkv"]
    A --> E["FlashAttention-family IO-aware kernels (GPU)"]
    E --> F["Cross-platform vLLM deepseek_v4 (FlashMLA/FlashInfer/Triton)"]
    F --> D
    B --> G["MiniMax MSA index branch"]
    G --> C
    C --> H["SFA C8 indexer cache (vllm-ascend, V3.2-style)"]
    D --> I["DSA Lightning Indexer C8 (vllm-ascend, V4) + AICPU metadata pre-op"]
```

The Ascend DSA stack descends from DeepSeek's MLA line and its own earlier SFA backend. The V3.2 SFA backend already used a lightning indexer over *tokens*; V4's DSA backend keeps the indexer concept but applies it to *compressed blocks* inside a new `npu_sparse_attn_sharedkv` operator, with the indexer's cache running at 8-bit "C8" precision (see the [C8 page](deepseek-v4-lightning-indexer-c8.md)).

## The Core Idea

The Ascend port does not translate kernels one-for-one; it re-architects the DeepSeek-V4 serving path around what the NPU does well: fused custom operators, heterogeneous paged KV groups, and NPU stream overlap — all captured into ACL graphs for steady-state decode. The model declares its five KV caches through Ascend specs; the DSA backend builds per-layer metadata, runs the MLA prologue, compressor, indexer, and sparse attention as a chain of `torch.ops._C_ascend` / `torch.ops.vllm` custom ops; and mHC + MTP wrap everything with `npu_hc_pre_v2` / `npu_hc_post` and a shared residual-stream buffer.

## Symbol Map

| Symbol | Human name | Meaning |
|---|---|---|
| `c4` / `c128` | main-KV compression ratio | Both compress main KV: ratio 4 for C4 and ratio 128 for C128. C4 additionally owns an indexer for sparse selection; C128 reads its compressed KV without that C4 top-k indexer. |
| `hc_mult` | mHC stream multiplier | Residual stream expanded to `hc_mult × hidden_size` per layer |
| `wq_a`/`wq_b`/`wkv` | MLA projections | Query down-projection, query up-projection, KV projection |
| `q_norm`/`kv_norm` | MLA norms | RMSNorm over latent query / KV |
| `swa_cache` | sliding-window cache | Raw recent tokens (high-fidelity window), separate from the C4/C128 compressed main-KV cache |
| `state_cache` | compressor state cache | fp32 partial compression states, KV + gate score |
| `compress_kv_cache` | compressed KV cache | Main MLA cache of compressed entries |
| `indexer_k_cache`/`indexer_scale_cache` | indexer caches | 8-bit indexer keys + per-token-head dequant scales |
| `sas_metadata` / `qli_metadata` | operator metadata | Core-partition info for sparse attention / quantized lightning indexer |
| `topk_indices_buffer` | IndexCache buffer | Shared top-k cache so later c4 layers can reuse an earlier layer's selection |

## Deep Dive

### 1. Model registration and the override class

**What it does:** vllm-ascend registers `AscendDeepseekV4ForCausalLM` (and `DeepSeekV4MTP`) into vLLM's `ModelRegistry`, replacing the upstream DeepSeek-V4 model for NPU runs.

**Why it matters:** The model class is the entry point for everything below; swapping it is how Ascend-specific layers, ops, and KV specs enter the pipeline without forking upstream vLLM.

**How it works:** <a class="code-link" href="../../../../external-repos/vllm-ascend/vllm_ascend/models/__init__.py#L4" data-code-repo="vllm-ascend-32a59d4e349c" data-code-path="vllm_ascend/models/__init__.py" data-code-line="4" data-code-end-line="18"><code>register_model</code></a> maps `DeepseekV4ForCausalLM` to `vllm_ascend.models.deepseek_v4:AscendDeepseekV4ForCausalLM` and `DeepSeekV4MTPModel` to the MTP class. <a class="code-link" href="../../../../external-repos/vllm-ascend/vllm_ascend/models/deepseek_v4.py#L1217" data-code-repo="vllm-ascend-32a59d4e349c" data-code-path="vllm_ascend/models/deepseek_v4.py" data-code-line="1217" data-code-end-line="1290"><code>AscendDeepseekV4ForCausalLM</code></a> inherits `SupportsPP`, `DeepseekV2MixtureOfExperts`, `SupportsLoRA`, and `SupportsEagle3`; it builds `DeepseekV4Model` + `lm_head` + `LogitsProcessor`, and collects MoE metadata in `set_moe_parameters`.

**The intuition:** The override is the "package manifest" — it decides which layers, MoE runner, and attention backend the model will use.

**A concrete example:** When vLLM loads `DeepseekV4ForCausalLM` on an NPU worker, `register_model` routes it to the Ascend class, whose `DeepseekV4Model` builds `DeepseekV2DecoderLayer`s (with mHC + DSA attention) instead of upstream's layers.

**Remember:** The whole Ascend DeepSeek-V4 path hangs off one registry swap.

### 2. The model forward: mHC from the first layer to the head

**What it does:** `DeepseekV4Model.forward` runs embedding, expands to `hc_mult` streams, executes the decoder layers, and merges back through `hc_head` before the LM head.

**Why it matters:** mHC (Manifold-Constrained Hyper-Connections) is how DeepSeek-V4 keeps signal stable through its deepest layers; the Ascend port implements it as custom NPU ops instead of plain tensor math.

**How it works:** On the first PP rank, hidden states are unsqueezed and repeated to `(b, s, hc_mult, h)` (<a class="code-link" href="../../../../external-repos/vllm-ascend/vllm_ascend/models/deepseek_v4.py#L1103" data-code-repo="vllm-ascend-32a59d4e349c" data-code-path="vllm_ascend/models/deepseek_v4.py" data-code-line="1103" data-code-end-line="1160"><code>DeepseekV4Model.forward</code></a>). Each layer returns `(hidden_states, residual)`. After the layers, the pre-`hc_head` residual stream is stashed into a stable-address `_mtp_hidden_buffer` for the MTP draft (all-gathering first when FlashComm [sequence parallelism](../../../terms/sequence-parallelism.md) is enabled), then `hc_head` (<a class="code-link" href="../../../../external-repos/vllm-ascend/vllm_ascend/models/deepseek_v4.py#L1094" data-code-repo="vllm-ascend-32a59d4e349c" data-code-path="vllm_ascend/models/deepseek_v4.py" data-code-line="1094" data-code-end-line="1101"><code>hc_head</code></a>) computes a learned sigmoid-weighted mix of the streams, followed by `RMSNorm`.

**The intuition:** mHC turns one residual stream into several learnable streams per layer; `hc_head` is the final learned weighted average before the LM head.

**A concrete example:** A `hc_mult=3` model keeps `(b, s, 3, h)` alive through every layer — each decoder layer's `hc_pre`/`hc_post` combine the 3 streams with learned mixing weights, and only at the head are they folded back into one `h`-dimensional vector.

**Remember:** `hc_pre_v2`/`hc_post` run at every attention and FFN boundary; `hc_head` is the single merge point before logits.

### 3. The decoder layer: mHC + DSA attention + MoE

**What it does:** `DeepseekV2DecoderLayer.forward` is the per-layer sandwich: `hc_pre` → input norm → attention → `hc_post` → post norm → [MoE](../../../terms/mixture-of-experts.md) → `hc_post`.

**Why it matters:** This is where the model's three big features (mHC, hybrid attention, MoE) physically meet in one forward pass.

**How it works:** <a class="code-link" href="../../../../external-repos/vllm-ascend/vllm_ascend/models/deepseek_v4.py#L972" data-code-repo="vllm-ascend-32a59d4e349c" data-code-path="vllm_ascend/models/deepseek_v4.py" data-code-line="972" data-code-end-line="982"><code>hc_pre</code></a> calls `torch.ops._C_ascend.npu_hc_pre_v2(x, hc_fn, hc_scale, hc_base, ...)` (registered in <a class="code-link" href="../../../../external-repos/vllm-ascend/csrc/torch_binding.cpp#L2519" data-code-repo="vllm-ascend-32a59d4e349c" data-code-path="csrc/torch_binding.cpp" data-code-line="2519" data-code-end-line="2525"><code>npu_hc_pre_v2</code> registration</a>); <a class="code-link" href="../../../../external-repos/vllm-ascend/vllm_ascend/models/deepseek_v4.py#L984" data-code-repo="vllm-ascend-32a59d4e349c" data-code-path="vllm_ascend/models/deepseek_v4.py" data-code-line="984" data-code-end-line="1005"><code>DeepseekV2DecoderLayer.forward</code></a> runs attention and MoE between the two `hc_post` calls (<a class="code-link" href="../../../../external-repos/vllm-ascend/vllm_ascend/models/deepseek_v4.py#L978" data-code-repo="vllm-ascend-32a59d4e349c" data-code-path="vllm_ascend/models/deepseek_v4.py" data-code-line="978" data-code-end-line="982"><code>hc_post</code></a> → `npu_hc_post`, registration at <a class="code-link" href="../../../../external-repos/vllm-ascend/csrc/torch_binding.cpp#L2500" data-code-repo="vllm-ascend-32a59d4e349c" data-code-path="csrc/torch_binding.cpp" data-code-line="2500" data-code-end-line="2507"><code>npu_hc_post</code> registration</a>). Attention is the `DeepseekV4Attention` wrapper (Section 4); the FFN is the `DeepseekV4MoE` (Section 5).

**The intuition:** Every layer is a symmetric mHC sandwich; the mHC ops are what make deep stacks trainable at this scale.

**A concrete example:** Following one token through a layer: `hc_pre` mixes 3 streams → attention returns one stream → `hc_post` merges it back into 3 streams → MoE → `hc_post` again, keeping the manifold constraint throughout.

**Remember:** mHC wraps *both* attention and FFN inside each layer.

### 4. Attention construction: per-layer compressors, indexers, and rope

**What it does:** `DeepseekV4Attention.__init__` builds the MLA prologue modules, a per-layer `Compressor` (and `Indexer` for c4 layers), the SWA cache, and a `ComplexExpRotaryEmbedding`, all bundled into `AscendDeepseekSparseAttention`.

**Why it matters:** The per-layer `compress_ratio` (c4 vs c128 vs plain SWA) decides which modules and caches a layer gets; getting this right is what makes hybrid serving correct.

**How it works:** <a class="code-link" href="../../../../external-repos/vllm-ascend/vllm_ascend/models/deepseek_v4.py#L711" data-code-repo="vllm-ascend-32a59d4e349c" data-code-path="vllm_ascend/models/deepseek_v4.py" data-code-line="711" data-code-end-line="904"><code>DeepseekV4Attention.__init__</code></a> computes the layer index and ratio via <a class="code-link" href="../../../../external-repos/vllm-ascend/vllm_ascend/utils.py#L80" data-code-repo="vllm-ascend-32a59d4e349c" data-code-path="vllm_ascend/utils.py" data-code-line="80" data-code-end-line="109"><code>extract_dsv4_layer_index</code>/<code>get_dsv4_compress_ratio</code></a>. Compressor-bearing layers get a `Compressor` (with its fp32 `state_cache`); c4 layers additionally get an `Indexer` (with its 8-bit cache). The layer's rope uses `ComplexExpRotaryEmbedding` (<a class="code-link" href="../../../../external-repos/vllm-ascend/vllm_ascend/ops/rope_dsv4.py#L164" data-code-repo="vllm-ascend-32a59d4e349c" data-code-path="vllm_ascend/ops/rope_dsv4.py" data-code-line="164" data-code-end-line="190"><code>ComplexExpRotaryEmbedding</code></a>) with a `c{ratio}` rope group and per-layer `rope_theta`. The `IndexCache` `skip_topk` flag (reusing a previous c4 layer's top-k, per arXiv:2603.12201) is decided here too. All modules are handed to <a class="code-link" href="../../../../external-repos/vllm-ascend/vllm_ascend/ops/dsa.py#L41" data-code-repo="vllm-ascend-32a59d4e349c" data-code-path="vllm_ascend/ops/dsa.py" data-code-line="41" data-code-end-line="59"><code>DSAModules</code></a> and then to `AscendDeepseekSparseAttention`.

**Main-KV compression versus indexer sparsity:** `c4` and `c128` are the compressor ratios for the layer's **main compressed-KV cache**, not names for indexer-only caches. Any layer with `compress_ratio > 1` gets a `Compressor` (<a class="code-link" href="../../../../external-repos/vllm-ascend-8405ff749d05/vllm_ascend/models/deepseek_v4/model.py#L570" data-code-repo="vllm-ascend-8405ff749d05" data-code-path="vllm_ascend/models/deepseek_v4/model.py" data-code-line="570"><code>ratio selection</code></a>, <a class="code-link" href="../../../../external-repos/vllm-ascend-8405ff749d05/vllm_ascend/models/deepseek_v4/model.py#L616" data-code-repo="vllm-ascend-8405ff749d05" data-code-path="vllm_ascend/models/deepseek_v4/model.py" data-code-line="616"><code>compressor construction</code></a>); the compressor passes that ratio to its NPU op as `cmp_ratio` (<a class="code-link" href="../../../../external-repos/vllm-ascend-8405ff749d05/vllm_ascend/models/deepseek_v4/compressor.py#L226" data-code-repo="vllm-ascend-8405ff749d05" data-code-path="vllm_ascend/models/deepseek_v4/compressor.py" data-code-line="226"><code>compressor ratio argument</code></a>). Thus **C4 compresses main KV at ratio 4, and C128 compresses main KV at ratio 128**. C4 additionally owns an Indexer (<a class="code-link" href="../../../../external-repos/vllm-ascend-8405ff749d05/vllm_ascend/models/deepseek_v4/model.py#L627" data-code-repo="vllm-ascend-8405ff749d05" data-code-path="vllm_ascend/models/deepseek_v4/model.py" data-code-line="627"><code>C4-only indexer construction</code></a>) whose separate key/scale cache produces top-k indices over those C4 compressed main-KV entries. At runtime, the C4 branch writes compressed output to `compress_kv_cache` through the indexer's overlap plan (<a class="code-link" href="../../../../external-repos/vllm-ascend-8405ff749d05/vllm_ascend/attention/dsa_v1.py#L2652" data-code-repo="vllm-ascend-8405ff749d05" data-code-path="vllm_ascend/attention/dsa_v1.py" data-code-line="2652"><code>C4 main-KV write</code></a>); C128 uses the same compressor and writes the same main-cache role without an indexer (<a class="code-link" href="../../../../external-repos/vllm-ascend-8405ff749d05/vllm_ascend/attention/dsa_v1.py#L2687" data-code-repo="vllm-ascend-8405ff749d05" data-code-path="vllm_ascend/attention/dsa_v1.py" data-code-line="2687"><code>C128 main-KV write</code></a>). Attention receives `compress_kv_cache` as `cmp_kv` for either compressed ratio (<a class="code-link" href="../../../../external-repos/vllm-ascend-8405ff749d05/vllm_ascend/attention/dsa_v1.py#L2834" data-code-repo="vllm-ascend-8405ff749d05" data-code-path="vllm_ascend/attention/dsa_v1.py" data-code-line="2834"><code>compressed main-KV read</code></a>); only C4 adds `cmp_sparse_indices` (<a class="code-link" href="../../../../external-repos/vllm-ascend-8405ff749d05/vllm_ascend/attention/dsa_v1.py#L2838" data-code-repo="vllm-ascend-8405ff749d05" data-code-path="vllm_ascend/attention/dsa_v1.py" data-code-line="2838"><code>C4 top-k selection</code></a>). The sliding-window cache remains a separate recent-token cache; a ratio-1/SWA-only layer has no compressed main-KV cache or C4 indexer.

**The intuition:** The layer type (c4/c128/swa) is a config-driven decision that determines the whole module + cache footprint of that layer.

**A concrete example:** A `compress_ratios=[4,128,...]` config makes layer 0 a c4 layer with `Compressor(4)` + `Indexer` + SWA cache, while a `compress_ratio=1` layer gets only the SWA cache and no compressor.

**Remember:** C4 = 4-ratio compressed main KV + indexer; C128 = 128-ratio compressed main KV without the C4 indexer; ratio 1 = sliding-window attention without compressed main KV.

### 5. MoE forward: routed experts with shared experts and mHC scaling

**What it does:** `DeepseekV4MoE.forward` routes tokens through grouped top-k experts and applies the `routed_scaling_factor`, with optional sequence-parallel chunking.

**Why it matters:** DeepSeek-V4 is a 1.6T/49B-active MoE; the FFN dominates FLOPs, and the Ascend port runs it through the patched `FusedMoE`/`AscendMoERunner`.

**How it works:** <a class="code-link" href="../../../../external-repos/vllm-ascend/vllm_ascend/models/deepseek_v4.py#L354" data-code-repo="vllm-ascend-32a59d4e349c" data-code-path="vllm_ascend/models/deepseek_v4.py" data-code-line="354" data-code-end-line="460"><code>DeepseekV4MoE.__init__</code></a> builds a `ReplicatedLinear` gate (with hash routing on early layers) and a `FusedMoE` with grouped top-k, EPLB redundancy, and shared experts. <a class="code-link" href="../../../../external-repos/vllm-ascend/vllm_ascend/models/deepseek_v4.py#L462" data-code-repo="vllm-ascend-32a59d4e349c" data-code-path="vllm_ascend/models/deepseek_v4.py" data-code-line="462" data-code-end-line="529"><code>DeepseekV4MoE.forward</code></a> computes router logits (or lets the internal router run), executes the experts, then applies `muls_add_triton` to fold in shared-expert output and `routed_scaling_factor`. Sequence-parallel mode chunks before and all-gathers after.

**The intuition:** The MoE is the compute-heavy half of each layer; the shared experts + scaling are folded into the routed output with a Triton fused add.

**A concrete example:** With `n_shared_experts=1`, a token routed to 8 of 256 experts gets expert output scaled by `routed_scaling_factor` and added to the shared MLP output via `muls_add_triton`.

**Remember:** Grouped top-k, EPLB redundant experts, and shared-expert fusion are all configured at construction, executed in one forward.

### 6. The DSA attention operator: one custom op for the whole attention pass

**What it does:** `AscendDeepseekSparseAttention.forward` wraps the entire attention + output projection in a single `torch.ops.vllm.dsa_forward` custom op that dispatches to the DSA backend implementation.

**Why it matters:** Running the whole attention pass inside one custom op is what makes it capturable into an ACL graph (a CUDA-graph replacement on NPU).

**How it works:** <a class="code-link" href="../../../../external-repos/vllm-ascend/vllm_ascend/ops/dsa.py#L61" data-code-repo="vllm-ascend-32a59d4e349c" data-code-path="vllm_ascend/ops/dsa.py" data-code-line="61" data-code-end-line="176"><code>AscendDeepseekSparseAttention</code></a> builds a `DSAAttention` layer that instantiates the backend impl; <a class="code-link" href="../../../../external-repos/vllm-ascend/vllm_ascend/ops/dsa.py#L157" data-code-repo="vllm-ascend-32a59d4e349c" data-code-path="vllm_ascend/ops/dsa.py" data-code-line="157" data-code-end-line="177"><code>forward</code></a> calls `torch.ops.vllm.dsa_forward(hidden_states, need_gather_q_kv, output, self.prefix)`. The registered <a class="code-link" href="../../../../external-repos/vllm-ascend/vllm_ascend/ops/dsa.py#L178" data-code-repo="vllm-ascend-32a59d4e349c" data-code-path="vllm_ascend/ops/dsa.py" data-code-line="178" data-code-end-line="212"><code>dsa_forward</code></a> looks up the layer's impl from the forward context, filters metadata for this layer, and calls `AscendDSAImpl.forward`. The profiling run is special-cased (Section 8).

**The intuition:** One op = one ACL-graph node = one capturable, replayable attention pass.

**A concrete example:** During ACL graph capture, the profiler calls `dsa_forward` once per layer with no metadata; the impl zero-fills output (or runs the OTP collectives) so the graph records every op that later replay needs.

**Remember:** `dsa_forward` is registered with `dispatch_key="PrivateUse1"` and a fake impl, so torch.compile / graph capture sees a stable op boundary.

### 7. The DSA backend: metadata build + forward orchestration

**What it does:** `AscendDSABackend` provides the metadata builder and impl factory; `AscendDSAImpl.forward` orchestrates the per-layer prefill/decode execution.

**Why it matters:** This is the runtime heart — it turns scheduler output into per-layer metadata and drives all five caches and three operator chains.

**How it works:** <a class="code-link" href="../../../../external-repos/vllm-ascend/vllm_ascend/attention/dsa_v1.py#L191" data-code-repo="vllm-ascend-32a59d4e349c" data-code-path="vllm_ascend/attention/dsa_v1.py" data-code-line="191" data-code-end-line="233"><code>AscendDSABackend</code></a> returns `AscendDSAMetadataBuilder` (or the DSA-CP variant) and `AscendDSAImpl` (or `AscendDSACPImpl`). The builder's <a class="code-link" href="../../../../external-repos/vllm-ascend/vllm_ascend/attention/dsa_v1.py#L621" data-code-repo="vllm-ascend-32a59d4e349c" data-code-path="vllm_ascend/attention/dsa_v1.py" data-code-line="621" data-code-end-line="720"><code>build</code></a> splits decodes/prefills, computes cos/sin via <a class="code-link" href="../../../../external-repos/vllm-ascend/vllm_ascend/ops/rope_dsv4.py#L83" data-code-repo="vllm-ascend-32a59d4e349c" data-code-path="vllm_ascend/ops/rope_dsv4.py" data-code-line="83" data-code-end-line="139"><code>get_cos_and_sin_dsa</code></a>, formats slot mappings, and builds prefill + decode metadata. <a class="code-link" href="../../../../external-repos/vllm-ascend/vllm_ascend/attention/dsa_v1.py#L1761" data-code-repo="vllm-ascend-32a59d4e349c" data-code-path="vllm_ascend/attention/dsa_v1.py" data-code-line="1761" data-code-end-line="1835"><code>AscendDSAImpl.forward</code></a> splits hidden states into decode/prefill segments, runs `_forward_prefill`/`_forward_decode`, applies inverse partial RoPE to the output, and runs `_forward_o_proj`.

**The intuition:** The metadata builder turns ragged scheduler output into the fixed tensors the NPU kernels need; the impl then runs the same operator chain for whichever phase is present.

**A concrete example:** One step schedules 2 decodes + 1 prefill of 100 tokens: the builder emits `num_decode_tokens=2`, `num_prefill_tokens=100`, separate metadata, and the impl calls `_forward_prefill` on the prefill slice and `_forward_decode` on the decode slice, merging outputs into one `o_proj_input`.

**Remember:** Prefill and decode share the operator chain but get separate metadata and separate code paths.

### 8. The prefill path: MLA prologue → C4/C128 main-KV compression → optional C4 indexer → sparse attention

**What it does:** `_forward_prefill` builds q and recent-window KV, compresses main KV for both C4 and C128 layers, runs index selection only for C4, then scores the recent and compressed KV tiers.

**Why it matters:** Both compressed attention types need their compressed main-KV history for long-context reads. The C4 indexer is an extra selector over the C4-compressed records; it is not the compression step that creates the C4 main-KV cache.

**How it works:** <a class="code-link" href="../../../../external-repos/vllm-ascend/vllm_ascend/attention/dsa_v1.py#L1933" data-code-repo="vllm-ascend-32a59d4e349c" data-code-path="vllm_ascend/attention/dsa_v1.py" data-code-line="1933" data-code-end-line="2244"><code>_forward_prefill</code></a> unpacks the five caches via <a class="code-link" href="../../../../external-repos/vllm-ascend/vllm_ascend/device/device_op.py#L807" data-code-repo="vllm-ascend-32a59d4e349c" data-code-path="vllm_ascend/device/device_op.py" data-code-line="807" data-code-end-line="841"><code>unpack_dsa_forward_kv_cache</code></a>. The MLA prologue runs either the multi-stream `_mla_prolog_multistream` (<a class="code-link" href="../../../../external-repos/vllm-ascend/vllm_ascend/attention/dsa_v1.py#L1836" data-code-repo="vllm-ascend-32a59d4e349c" data-code-path="vllm_ascend/attention/dsa_v1.py" data-code-line="1836" data-code-end-line="1932"><code>_mla_prolog_multistream</code></a> — CV-split q/kv on two NPU streams) or the serial path. For every compressed layer, the compressor writes main-KV records to `compress_kv_cache`: C4 wires this write through the indexer's overlap plan (<a class="code-link" href="../../../../external-repos/vllm-ascend-8405ff749d05/vllm_ascend/attention/dsa_v1.py#L2652" data-code-repo="vllm-ascend-8405ff749d05" data-code-path="vllm_ascend/attention/dsa_v1.py" data-code-line="2652"><code>C4 main-KV write</code></a>), while the C128 path writes through the same compressor directly (<a class="code-link" href="../../../../external-repos/vllm-ascend-8405ff749d05/vllm_ascend/attention/dsa_v1.py#L2687" data-code-repo="vllm-ascend-8405ff749d05" data-code-path="vllm_ascend/attention/dsa_v1.py" data-code-line="2687"><code>C128 main-KV write</code></a>). C4 additionally selects top-k records; the attention call receives `cmp_kv` for both ratios but `cmp_sparse_indices` only for C4 (<a class="code-link" href="../../../../external-repos/vllm-ascend-8405ff749d05/vllm_ascend/attention/dsa_v1.py#L2834" data-code-repo="vllm-ascend-8405ff749d05" data-code-path="vllm_ascend/attention/dsa_v1.py" data-code-line="2834"><code>compressed KV read</code></a>, <a class="code-link" href="../../../../external-repos/vllm-ascend-8405ff749d05/vllm_ascend/attention/dsa_v1.py#L2838" data-code-repo="vllm-ascend-8405ff749d05" data-code-path="vllm_ascend/attention/dsa_v1.py" data-code-line="2838"><code>C4-only sparse indices</code></a>). Thus sparse selection applies to C4 compressed KV; C128 still uses compressed main KV, read densely within that compressed tier. The A3 DSA-CP variant follows the same distinction: its compressor runs for `compress_ratio > 1` (<a class="code-link" href="../../../../external-repos/vllm-ascend-8405ff749d05/vllm_ascend/attention/context_parallel/dsa_cp.py#L1927" data-code-repo="vllm-ascend-8405ff749d05" data-code-path="vllm_ascend/attention/context_parallel/dsa_cp.py" data-code-line="1927"><code>DSA-CP compressor</code></a>) and scatters those output records into the compressed main cache (<a class="code-link" href="../../../../external-repos/vllm-ascend-8405ff749d05/vllm_ascend/attention/context_parallel/dsa_cp.py#L1950" data-code-repo="vllm-ascend-8405ff749d05" data-code-path="vllm_ascend/attention/context_parallel/dsa_cp.py" data-code-line="1950"><code>DSA-CP main-KV write</code></a>); its C4 call adds sparse indices while its C128 call still supplies `cmp_kv` (<a class="code-link" href="../../../../external-repos/vllm-ascend-8405ff749d05/vllm_ascend/attention/context_parallel/dsa_cp.py#L2002" data-code-repo="vllm-ascend-8405ff749d05" data-code-path="vllm_ascend/attention/context_parallel/dsa_cp.py" data-code-line="2002"><code>DSA-CP C4 read</code></a>, <a class="code-link" href="../../../../external-repos/vllm-ascend-8405ff749d05/vllm_ascend/attention/context_parallel/dsa_cp.py#L2022" data-code-repo="vllm-ascend-8405ff749d05" data-code-path="vllm_ascend/attention/context_parallel/dsa_cp.py" data-code-line="2022"><code>DSA-CP C128 read</code></a>).

On the pinned non-A5 branch, the shared operator registration is <a class="code-link" href="../../../../external-repos/vllm-ascend/csrc/torch_binding.cpp#L2418" data-code-repo="vllm-ascend-32a59d4e349c" data-code-path="csrc/torch_binding.cpp" data-code-line="2418" data-code-end-line="2444"><code>npu_sparse_attn_sharedkv</code></a>; the A5 branches choose different sparse-attention operators as described in Section 14.

**The intuition:** Prefill = build every cache + compute attention once; the compressor is the step that collapses 4 (or 128) tokens into one KV entry.

**A concrete example:** In a C4 layer, input KV is compressed into the main C4 cache and the separate indexer chooses which compressed records sparse attention reads, alongside recent SWA KV. In a C128 layer, input KV is compressed at the C128 ratio into its main cache; sparse attention reads that compressed tier without C4 top-k indices, alongside SWA KV.

**Remember:** The compressor output is scattered into the compressed cache with a block-offset slot mapping; a zero-row output skips scatter (A5 guard).

### 9. The decode path: one token at a time through the same chain

**What it does:** `_forward_decode` runs the same MLA → compressor → indexer → sparse-attention chain for 1-token queries.

**Why it matters:** Decode is memory-bound and must be fast; it is also the path ACL graphs capture for steady-state token generation.

**How it works:** <a class="code-link" href="../../../../external-repos/vllm-ascend/vllm_ascend/attention/dsa_v1.py#L2245" data-code-repo="vllm-ascend-32a59d4e349c" data-code-path="vllm_ascend/attention/dsa_v1.py" data-code-line="2245" data-code-end-line="2558"><code>_forward_decode</code></a> reuses the phase-specific metadata: one query token per request and one KV update per request. The compressor updates its partial state and emits compressed main-KV records for either C4 or C128; the C4 indexer additionally refreshes the side cache and reselects top-k over C4 records. C128 has no such indexer step.

**The intuition:** Decode is prefill's "one new token" specialization — the same chain, but the compressor only updates state and the indexer re-ranks.

**A concrete example:** After prefill, a decode step updates the layer's compressor state and (when a complete compressed output is produced) its main compressed-KV cache. A C4 layer also re-picks top-k from its compressed records; a C128 layer reads its C128-compressed main KV without that C4 selector.

**Remember:** Decode and prefill share helper code but diverge on metadata; decode is what the ACL graph replays every step.

### 10. Output projection: inverse RoPE, grouped o_proj, and TP variants

**What it does:** `_forward_o_proj` converts attention output back to hidden space via inverse partial RoPE, `wo_a`, and `wo_b`, with three TP strategies.

**Why it matters:** The output projection is where the attention result re-enters the residual stream; its TP handling (A5 quantized path, OTP [all-to-all](../../../terms/all-to-all.md), olora TP) differs per hardware and config.

**How it works:** <a class="code-link" href="../../../../external-repos/vllm-ascend/vllm_ascend/attention/dsa_v1.py#L1652" data-code-repo="vllm-ascend-32a59d4e349c" data-code-path="vllm_ascend/attention/dsa_v1.py" data-code-line="1652" data-code-end-line="1760"><code>_forward_o_proj</code></a> first applies inverse partial rotary (`inplace_partial_rotary_mul` with `-sin`), then reshapes into groups. On A5 it uses an [FP8](../../../terms/fp8.md) MX-quantized batch [matmul](../../../terms/gemm.md) path; with `oproj_tp_enable()` it does a static-buffer `all_to_all_single` + batch matmul + `reduce_scatter_tensor`; with `olora_tp_enable()` it uses `wo_a`/`wo_b`; otherwise a plain `npu_transpose_batchmatmul` + `wo_b`.

**The intuition:** The output projection is a grouped low-rank matmul (`wo_a` then `wo_b`); on multi-rank setups it becomes an all-to-all across output-tensor-parallel groups.

**A concrete example:** With output TP=2 and 16 local groups, each rank sends half its groups to the other rank (`all_to_all`), computes `wo_a` on the received groups, then `reduce_scatter`s the result back — all through address-stable static buffers so the ACL graph stays replay-safe.

**Remember:** OTP uses static send/recv buffers on purpose — per-call allocations would desync the HCCL operator during graph replay.

### 11. Heterogeneous KV cache: five types, per-layer groups, device-specific paging

**What it does:** Each DeepSeek-V4 layer declares its KV caches through Ascend specs; the patched KV coordinator groups them and allocates device-specific page sizes.

**Why it matters:** The five cache types have wildly different sizes and dtypes; correct paging is what makes 1M-token contexts fit in NPU memory.

**How it works:** The caches are `AscendDeepseekV4SWACache` (<a class="code-link" href="../../../../external-repos/vllm-ascend/vllm_ascend/models/deepseek_v4.py#L179" data-code-repo="vllm-ascend-32a59d4e349c" data-code-path="vllm_ascend/models/deepseek_v4.py" data-code-line="179" data-code-end-line="212"><code>AscendDeepseekV4SWACache</code></a>), `AscendCompressorStateCache` (<a class="code-link" href="../../../../external-repos/vllm-ascend/vllm_ascend/models/deepseek_v4.py#L110" data-code-repo="vllm-ascend-32a59d4e349c" data-code-path="vllm_ascend/models/deepseek_v4.py" data-code-line="110" data-code-end-line="142"><code>AscendCompressorStateCache</code></a>), the main compressed MLA cache (`AscendMLAAttentionSpec`, <a class="code-link" href="../../../../external-repos/vllm-ascend/vllm_ascend/core/kv_cache_interface.py#L19" data-code-repo="vllm-ascend-32a59d4e349c" data-code-path="vllm_ascend/core/kv_cache_interface.py" data-code-line="19" data-code-end-line="88"><code>AscendMLAAttentionSpec</code></a>), and the indexer caches (<a class="code-link" href="../../../../external-repos/vllm-ascend/vllm_ascend/models/deepseek_v4.py#L143" data-code-repo="vllm-ascend-32a59d4e349c" data-code-path="vllm_ascend/models/deepseek_v4.py" data-code-line="143" data-code-end-line="177"><code>AscendDeepseekV4IndexerCache</code></a>). Block sizes come from `DSV4_BLOCK_SIZES` (<a class="code-link" href="../../../../external-repos/vllm-ascend/vllm_ascend/models/layer/attention/layer.py#L28" data-code-repo="vllm-ascend-32a59d4e349c" data-code-path="vllm_ascend/models/layer/attention/layer.py" data-code-line="28" data-code-end-line="50"><code>get_dsv4_block_sizes</code></a>): `[mla, swa, c4-state, c128-state]` plus padded page sizes, with a separate A5 table. The <a class="code-link" href="../../../../external-repos/vllm-ascend/vllm_ascend/patch/platform/patch_kv_cache_coordinator.py#L68" data-code-repo="vllm-ascend-32a59d4e349c" data-code-path="vllm_ascend/patch/platform/patch_kv_cache_coordinator.py" data-code-line="68" data-code-end-line="80"><code>AscendHybridKVCacheCoordinator</code></a> patch handles the hybrid multi-group case.

**The intuition:** DeepSeek-V4 doesn't have one KV cache — it has five, each sized and paged for a different granularity (raw window, partial states, compressed entries, indexer keys, indexer scales).

**A concrete example:** On a 910B device with `block_size=128`, the table gives `[128, 128, 8, 32]` — SWA and MLA use 128-token pages, c4 compressor state uses 8-token pages, c128 state uses 32-token pages — and page-padding sizes `[16640, 131072]`.

**Remember:** Cache dtype also differs by device: on A5 the SWA/compressed caches are FP8 e4m3fn; on 910B/A2/A3 they are BF16 (indexer cache is INT8).

### 12. MTP: the draft model reuses the same layer machinery

**What it does:** `DeepSeekV4MTP` is the multi-token-prediction draft model used for speculative decoding, reusing `DeepseekV2DecoderLayer` and mHC.

**Why it matters:** MTP is what lets DeepSeek-V4 speculate tokens cheaply; the Ascend port must keep the draft and target models' residual streams in sync.

**How it works:** <a class="code-link" href="../../../../external-repos/vllm-ascend/vllm_ascend/models/deepseek_v4_mtp.py#L201" data-code-repo="vllm-ascend-32a59d4e349c" data-code-path="vllm_ascend/models/deepseek_v4_mtp.py" data-code-line="201" data-code-end-line="250"><code>DeepSeekV4MTP</code></a> wraps `DeepSeekMultiTokenPredictor`, whose layers combine the current token's embedding with the target model's `_mtp_hidden_buffer` (the pre-`hc_head` residual stream stashed by the target forward). Each MTP layer is a `DeepSeekMultiTokenPredictorLayer` (<a class="code-link" href="../../../../external-repos/vllm-ascend/vllm_ascend/models/deepseek_v4_mtp.py#L56" data-code-repo="vllm-ascend-32a59d4e349c" data-code-path="vllm_ascend/models/deepseek_v4_mtp.py" data-code-line="56" data-code-end-line="138"><code>DeepSeekMultiTokenPredictorLayer</code></a>) that norm-merges `e_proj(embed)` with `h_proj(previous_hidden_states)` and runs one `DeepseekV2DecoderLayer`; logits come from a shared head.

**The intuition:** MTP is "one more decoder layer" that conditions on the target model's final residual stream instead of only on the previous token.

**A concrete example:** When `num_nextn_predict_layers=1`, each draft step runs one MTP layer: embedding → `e_proj` + target-stream `h_proj` → a full decoder layer (mHC + DSA attention + MoE) → shared-head logits.

**Remember:** The target model must keep `_mtp_hidden_buffer` fresh (all-gathering under FlashComm) or the draft's acceptance rate collapses.

### 13. Hardware divergence: A5 vs 910B/A2/A3

**What it does:** The same code paths branch on `get_ascend_device_type()` to pick dtypes, layouts, and fused ops per device family.

**Why it matters:** A5 (Ascend 950) supports FP8 caches and fused scatter paths the older devices do not; the port keeps one codebase with device-dispatch.

**How it works:** A5 selects FP8 e4m3fn SWA/compressed/indexer caches with FP32 scales, the fused `indexer_compress_epilog_v2` path, a one-dimensional slot mapping, and an FP8 quantized `o_proj`; non-A5 uses BF16 caches (INT8 indexer), the `npu_scatter_nd_update_v2`-based scatter, 2-D `[block_idx, offset]` slot mappings, and a Triton q-RMS. `AscendConfig` gates the SFA/LI C8 switches (see the [C8 page](deepseek-v4-lightning-indexer-c8.md) for the SFA side).

**The intuition:** A5 gets the newer FP8/fused path; the older devices get the proven INT8/BF16 path — same model, different backend knobs.

**A concrete example:** `AscendDeepseekV4IndexerCache.get_kv_cache_spec` sets `dtype=float8_e4m3fn` and `cache_dtype="float8_e4m3fn"` only on A5; elsewhere the indexer key cache stays INT8 with FP16 scales.

**Remember:** Device family is read at construction (cache dtype) and at runtime (op selection) — the branches are consistent but duplicated in several files.

### 14. DeepSeek-V4-Flash-0731: DSpark on A3 and the 950/A5 evidence boundary

In this repository's device abstraction, an `ascend950` SoC resolves to internal device type A5 (<a class="code-link" href="../../../../external-repos/vllm-ascend-7702ccd7d8de/vllm_ascend/device/hardware.py#L52" data-code-repo="vllm-ascend-7702ccd7d8de" data-code-path="vllm_ascend/device/hardware.py" data-code-line="52"><code>device_type_from_soc_version</code></a>).

**Selected workload and boundary.** Follow one scheduled target-verification batch for the `DeepSeek-V4-Flash-0731-w8a8` checkpoint from <a class="code-link" href="../../../../external-repos/vllm-ascend-51b0242c0d3b/vllm_ascend/worker/model_runner_v1.py#L2228" data-code-repo="vllm-ascend-51b0242c0d3b" data-code-path="vllm_ascend/worker/model_runner_v1.py" data-code-line="2228"><code>NPUModelRunner.execute_model</code></a> to the worker's `ModelRunnerOutput`. This is a static trace to the engine/scheduler boundary, not an HTTP-response trace or a device run.

**Checkpoint and A3 configuration.** The pinned model guide identifies the July 31 weights as DSpark and states A2/A3 support from vLLM-Ascend v0.25.0 onward (<a class="code-link" href="../../../../external-repos/vllm-ascend-7702ccd7d8de/docs/source/tutorials/models/DeepSeek-V4-Flash.md#L28" data-code-repo="vllm-ascend-7702ccd7d8de" data-code-path="docs/source/tutorials/models/DeepSeek-V4-Flash.md" data-code-line="28"><code>0731 checkpoint</code></a>, <a class="code-link" href="../../../../external-repos/vllm-ascend-7702ccd7d8de/docs/source/tutorials/models/DeepSeek-V4-Flash.md#L42" data-code-repo="vllm-ascend-7702ccd7d8de" data-code-path="docs/source/tutorials/models/DeepSeek-V4-Flash.md" data-code-line="42"><code>A2/A3 DSpark support</code></a>). Its A3 recipe uses `--block-size 32`, TP=4, DP=4, and expert parallel (<a class="code-link" href="../../../../external-repos/vllm-ascend-7702ccd7d8de/docs/source/tutorials/models/DeepSeek-V4-Flash.md#L281" data-code-repo="vllm-ascend-7702ccd7d8de" data-code-path="docs/source/tutorials/models/DeepSeek-V4-Flash.md" data-code-line="281"><code>A3 launch command</code></a>), with seven DSpark speculative tokens (<a class="code-link" href="../../../../external-repos/vllm-ascend-7702ccd7d8de/docs/source/tutorials/models/DeepSeek-V4-Flash.md#L298" data-code-repo="vllm-ascend-7702ccd7d8de" data-code-path="docs/source/tutorials/models/DeepSeek-V4-Flash.md" data-code-line="298"><code>DSpark config</code></a>) and DSA context parallel enabled (<a class="code-link" href="../../../../external-repos/vllm-ascend-7702ccd7d8de/docs/source/tutorials/models/DeepSeek-V4-Flash.md#L306" data-code-repo="vllm-ascend-7702ccd7d8de" data-code-path="docs/source/tutorials/models/DeepSeek-V4-Flash.md" data-code-line="306"><code>enable_dsa_cp</code></a>). This is different from the older `DeepSeek-V4-Flash-w8a8-mtp` recipe.

**Target and draft models.** The pinned registry maps `DeepseekV4ForCausalLM` to the Ascend target model (<a class="code-link" href="../../../../external-repos/vllm-ascend-51b0242c0d3b/vllm_ascend/models/__init__.py#L32" data-code-repo="vllm-ascend-51b0242c0d3b" data-code-path="vllm_ascend/models/__init__.py" data-code-line="32"><code>target registration</code></a>) and `DSparkDraftModel` separately (<a class="code-link" href="../../../../external-repos/vllm-ascend-51b0242c0d3b/vllm_ascend/models/__init__.py#L52" data-code-repo="vllm-ascend-51b0242c0d3b" data-code-path="vllm_ascend/models/__init__.py" data-code-line="52"><code>DSpark registration</code></a>). Per-layer C4/C128 roles come from the model configuration (<a class="code-link" href="../../../../external-repos/vllm-ascend-51b0242c0d3b/vllm_ascend/models/deepseek_v4/model.py#L548" data-code-repo="vllm-ascend-51b0242c0d3b" data-code-path="vllm_ascend/models/deepseek_v4/model.py" data-code-line="548"><code>compression-ratio selection</code></a>); only C4 layers add top-k indexing.

**A3 runtime branch.** Because the recipe sets `enable_dsa_cp`, <a class="code-link" href="../../../../external-repos/vllm-ascend-51b0242c0d3b/vllm_ascend/attention/dsa_v1.py#L262" data-code-repo="vllm-ascend-51b0242c0d3b" data-code-path="vllm_ascend/attention/dsa_v1.py" data-code-line="262"><code>AscendDSABackend.get_impl_cls</code></a> selects the context-parallel implementation rather than the ordinary single-rank `AscendDSAImpl`. <a class="code-link" href="../../../../external-repos/vllm-ascend-51b0242c0d3b/vllm_ascend/attention/context_parallel/dsa_cp.py#L1801" data-code-repo="vllm-ascend-51b0242c0d3b" data-code-path="vllm_ascend/attention/context_parallel/dsa_cp.py" data-code-line="1801"><code>AscendDSACPImpl._forward</code></a> updates the sequence KV caches and computes attention for local query rows. C4 layers update the indexer cache, select compressed-record IDs (<a class="code-link" href="../../../../external-repos/vllm-ascend-51b0242c0d3b/vllm_ascend/attention/context_parallel/dsa_cp.py#L2116" data-code-repo="vllm-ascend-51b0242c0d3b" data-code-path="vllm_ascend/attention/context_parallel/dsa_cp.py" data-code-line="2116"><code>indexer top-k</code></a>), then pass those IDs as `cmp_sparse_indices` with the compressed KV and its block table (<a class="code-link" href="../../../../external-repos/vllm-ascend-51b0242c0d3b/vllm_ascend/attention/context_parallel/dsa_cp.py#L2000" data-code-repo="vllm-ascend-51b0242c0d3b" data-code-path="vllm_ascend/attention/context_parallel/dsa_cp.py" data-code-line="2000"><code>compressed-index handoff</code></a>). They address C4 cache records, not output token IDs.

| State or choice | A3 path in the documented recipe | 950/A5 code path |
|---|---|---|
| Recent SWA KV | Per-layer sliding-window state; the existing device mapping uses BF16 SWA/compressed KV on A2/A3. | FP8 e4m3 cache path unless BF16 is explicitly selected; the cache spec includes the packed scale width when FP8 is used (<a class="code-link" href="../../../../external-repos/vllm-ascend-51b0242c0d3b/vllm_ascend/models/deepseek_v4/model.py#L129" data-code-repo="vllm-ascend-51b0242c0d3b" data-code-path="vllm_ascend/models/deepseek_v4/model.py" data-code-line="129"><code>SWA cache spec</code></a>). |
| Compressor state and compressed KV | C4/C128 compressor state is FP32 (<a class="code-link" href="../../../../external-repos/vllm-ascend-51b0242c0d3b/vllm_ascend/models/deepseek_v4/compressor.py#L154" data-code-repo="vllm-ascend-51b0242c0d3b" data-code-path="vllm_ascend/models/deepseek_v4/compressor.py" data-code-line="154"><code>FP32 state dtype</code></a>); the non-A5 plan uses <a class="code-link" href="../../../../external-repos/vllm-ascend-51b0242c0d3b/vllm_ascend/attention/dsa_attn_kv_plan.py#L228" data-code-repo="vllm-ascend-51b0242c0d3b" data-code-path="vllm_ascend/attention/dsa_attn_kv_plan.py" data-code-line="228"><code>npu_sparse_attn_sharedkv</code></a>. | The non-BF16 A5 plan writes through <a class="code-link" href="../../../../external-repos/vllm-ascend-51b0242c0d3b/vllm_ascend/attention/dsa_attn_kv_plan.py#L195" data-code-repo="vllm-ascend-51b0242c0d3b" data-code-path="vllm_ascend/attention/dsa_attn_kv_plan.py" data-code-line="195"><code>kv_compress_epilog</code></a> and calls <a class="code-link" href="../../../../external-repos/vllm-ascend-51b0242c0d3b/vllm_ascend/attention/dsa_attn_kv_plan.py#L258" data-code-repo="vllm-ascend-51b0242c0d3b" data-code-path="vllm_ascend/attention/dsa_attn_kv_plan.py" data-code-line="258"><code>npu_kv_quant_sparse_attn_sharedkv</code></a>; explicit BF16 selects <a class="code-link" href="../../../../external-repos/vllm-ascend-51b0242c0d3b/vllm_ascend/attention/dsa_attn_kv_plan.py#L245" data-code-repo="vllm-ascend-51b0242c0d3b" data-code-path="vllm_ascend/attention/dsa_attn_kv_plan.py" data-code-line="245"><code>sparse_flash_mla</code></a>. |
| C4 indexer keys/scales | INT8 keys with FP16 scales. | FP8 e4m3 keys with FP32 scales (<a class="code-link" href="../../../../external-repos/vllm-ascend-51b0242c0d3b/vllm_ascend/models/deepseek_v4/indexer.py#L98" data-code-repo="vllm-ascend-51b0242c0d3b" data-code-path="vllm_ascend/models/deepseek_v4/indexer.py" data-code-line="98"><code>indexer key dtype</code></a>, <a class="code-link" href="../../../../external-repos/vllm-ascend-51b0242c0d3b/vllm_ascend/models/deepseek_v4/indexer.py#L155" data-code-repo="vllm-ascend-51b0242c0d3b" data-code-path="vllm_ascend/models/deepseek_v4/indexer.py" data-code-line="155"><code>scale dtype</code></a>). |
| Sparse-attention and slot plan | <a class="code-link" href="../../../../external-repos/vllm-ascend-51b0242c0d3b/vllm_ascend/attention/dsa_attn_kv_plan.py#L228" data-code-repo="vllm-ascend-51b0242c0d3b" data-code-path="vllm_ascend/attention/dsa_attn_kv_plan.py" data-code-line="228"><code>npu_sparse_attn_sharedkv</code></a>; flat slot `s` is split into `(s // block_size, s % block_size)`. | Non-BF16 uses <a class="code-link" href="../../../../external-repos/vllm-ascend-51b0242c0d3b/vllm_ascend/attention/dsa_attn_kv_plan.py#L258" data-code-repo="vllm-ascend-51b0242c0d3b" data-code-path="vllm_ascend/attention/dsa_attn_kv_plan.py" data-code-line="258"><code>npu_kv_quant_sparse_attn_sharedkv</code></a> with flat physical slots; explicit BF16 uses <a class="code-link" href="../../../../external-repos/vllm-ascend-51b0242c0d3b/vllm_ascend/attention/dsa_attn_kv_plan.py#L245" data-code-repo="vllm-ascend-51b0242c0d3b" data-code-path="vllm_ascend/attention/dsa_attn_kv_plan.py" data-code-line="245"><code>sparse_flash_mla</code></a> (<a class="code-link" href="../../../../external-repos/vllm-ascend-51b0242c0d3b/vllm_ascend/attention/dsa_attn_kv_plan.py#L168" data-code-repo="vllm-ascend-51b0242c0d3b" data-code-path="vllm_ascend/attention/dsa_attn_kv_plan.py" data-code-line="168"><code>slot-coordinate conversion</code></a>). |

The hardware/dtype branch is chosen by <a class="code-link" href="../../../../external-repos/vllm-ascend-51b0242c0d3b/vllm_ascend/attention/dsa_attn_kv_plan.py#L206" data-code-repo="vllm-ascend-51b0242c0d3b" data-code-path="vllm_ascend/attention/dsa_attn_kv_plan.py" data-code-line="206"><code>get_dsa_attn_kv_plan</code></a>. In the block-offset case, `block_size` is cache slots per page: the flat slot number is divided into page index and in-page offset. Top-k indices instead live in the C4 compressed-record space (`cmp_ratio=4`) and are passed alongside a distinct compressed-cache block table. Do not treat either coordinate system as a sampled token.

The repo's 950 support row marks the DeepSeek-V4-Flash family supported, notes native mixed MXFP8/MXFP4 weights, and marks W8A8 in the feature row (<a class="code-link" href="../../../../external-repos/vllm-ascend-7702ccd7d8de/docs/source/user_guide/support_matrix/supported_models.md#L22" data-code-repo="vllm-ascend-7702ccd7d8de" data-code-path="docs/source/user_guide/support_matrix/supported_models.md" data-code-line="22"><code>950 family support</code></a>). Its A2/A3 row is marked experimental (<a class="code-link" href="../../../../external-repos/vllm-ascend-7702ccd7d8de/docs/source/user_guide/support_matrix/supported_models.md#L31" data-code-repo="vllm-ascend-7702ccd7d8de" data-code-path="docs/source/user_guide/support_matrix/supported_models.md" data-code-line="31"><code>A2/A3 status</code></a>). That is family-level support: the 0731 guide gives A2/A3 recipes, not a 950 DSpark recipe. The repo's DSpark acceptance test uses a separate W4A8 test checkpoint and includes a DSA-CP case (<a class="code-link" href="../../../../external-repos/vllm-ascend-7702ccd7d8de/tests/e2e/pull_request/four_card/spec_decode/test_dspark_deepseekv4.py#L33" data-code-repo="vllm-ascend-7702ccd7d8de" data-code-path="tests/e2e/pull_request/four_card/spec_decode/test_dspark_deepseekv4.py" data-code-line="33"><code>test checkpoint</code></a>, <a class="code-link" href="../../../../external-repos/vllm-ascend-7702ccd7d8de/tests/e2e/pull_request/four_card/spec_decode/test_dspark_deepseekv4.py#L61" data-code-repo="vllm-ascend-7702ccd7d8de" data-code-path="tests/e2e/pull_request/four_card/spec_decode/test_dspark_deepseekv4.py" data-code-line="61"><code>DSA-CP case</code></a>); it does not validate the exact 0731 W8A8 checkpoint on 950.

**Draft and return.** The target forward emits configured auxiliary hidden states (<a class="code-link" href="../../../../external-repos/vllm-ascend-51b0242c0d3b/vllm_ascend/models/deepseek_v4/model.py#L961" data-code-repo="vllm-ascend-51b0242c0d3b" data-code-path="vllm_ascend/models/deepseek_v4/model.py" data-code-line="961"><code>auxiliary-state producer</code></a>); the DSpark branch concatenates them for the proposer (<a class="code-link" href="../../../../external-repos/vllm-ascend-51b0242c0d3b/vllm_ascend/worker/model_runner_v1.py#L2068" data-code-repo="vllm-ascend-51b0242c0d3b" data-code-path="vllm_ascend/worker/model_runner_v1.py" data-code-line="2068"><code>target-to-draft handoff</code></a>), and the draft model projects/normalizes them (<a class="code-link" href="../../../../external-repos/vllm-ascend-51b0242c0d3b/vllm_ascend/models/deepseek_v4/dspark.py#L174" data-code-repo="vllm-ascend-51b0242c0d3b" data-code-path="vllm_ascend/models/deepseek_v4/dspark.py" data-code-line="174"><code>DSpark hidden-state combine</code></a>). Target logits plus speculative metadata are passed to rejection sampling (<a class="code-link" href="../../../../external-repos/vllm-ascend-51b0242c0d3b/vllm_ascend/worker/model_runner_v1.py#L2972" data-code-repo="vllm-ascend-51b0242c0d3b" data-code-path="vllm_ascend/worker/model_runner_v1.py" data-code-line="2972"><code>target verification sampler</code></a>); valid sampled IDs leave the worker in `ModelRunnerOutput` (<a class="code-link" href="../../../../external-repos/vllm-ascend-51b0242c0d3b/vllm_ascend/worker/model_runner_v1.py#L2925" data-code-repo="vllm-ascend-51b0242c0d3b" data-code-path="vllm_ascend/worker/model_runner_v1.py" data-code-line="2925"><code>worker return</code></a>). This route differs from MTP's `_mtp_hidden_buffer` path.

**MTP distinction.** The worker substitutes the target's `_mtp_hidden_buffer` only for `method == "mtp"` (<a class="code-link" href="../../../../external-repos/vllm-ascend-51b0242c0d3b/vllm_ascend/worker/model_runner_v1.py#L2058" data-code-repo="vllm-ascend-51b0242c0d3b" data-code-path="vllm_ascend/worker/model_runner_v1.py" data-code-line="2058"><code>MTP-only hidden-state override</code></a>); the DSpark branch above instead uses configured auxiliary hidden states.

**Optional A5 projection branch.** If DSA-CP full-`o_proj` gathering is enabled, the code additionally gates it on 950/A5 (<a class="code-link" href="../../../../external-repos/vllm-ascend-51b0242c0d3b/vllm_ascend/attention/context_parallel/dsa_cp.py#L1455" data-code-repo="vllm-ascend-51b0242c0d3b" data-code-path="vllm_ascend/attention/context_parallel/dsa_cp.py" data-code-line="1455"><code>A5 hardware gate</code></a>); it is selected only for prefill, not decode (<a class="code-link" href="../../../../external-repos/vllm-ascend-51b0242c0d3b/vllm_ascend/attention/context_parallel/dsa_cp.py#L1721" data-code-repo="vllm-ascend-51b0242c0d3b" data-code-path="vllm_ascend/attention/context_parallel/dsa_cp.py" data-code-line="1721"><code>prefill-only branch</code></a>).

## Putting It Together

① A request is scheduled; the metadata builder computes positions, slot mappings, and prefill/decode metadata. ② The model runner invokes `AscendDeepseekV4ForCausalLM.forward`: embed → `DeepseekV4Model.forward` expands to `hc_mult` streams → each decoder layer runs `hc_pre` → attention → `hc_post` → MoE → `hc_post`. ③ The attention wrapper dispatches through the base DSA or configured DSA-CP/PCP implementation; the documented A3 0731 recipe selects DSA-CP. ④ The selected device plan writes SWA/compressed KV, runs C4 top-k where applicable, then reads the cache groups. ⑤ The attention output projection returns to the residual stream; after all layers, the final norm and LM head produce target logits. ⑥ MTP or DSpark drafts according to the selected method, and the target sampler returns accepted token IDs. Graph behavior depends on the speculative method and launch configuration.

## What This Buys You

### The headline claim

vllm-ascend serves DeepSeek-V4's hybrid c4/c128 compressed attention and mHC on Ascend NPUs through one model override, one DSA backend, and heterogeneous KV groups, with MTP and DSpark as distinct draft paths.

### How we know: static code reading

This is a static reading of revision `32a59d4e349c`; there are no runtime numbers in the checkout. The evidence table in the supporting note lists every claim's file, symbol, and line.

### The mechanism behind the numbers

The design buys two things structurally: **graph capture** (attention and o_proj run inside `dsa_forward`, so ACL graph replay covers the whole steady-state decode) and **heterogeneous paging** (each cache type gets its own block size, so compressed state and indexer caches don't waste the large pages needed by SWA/MLA).

### ⚠️ How to read these numbers

Do not infer throughput or latency from this page. The A5 FP8 path, OTP collectives, and `indexer_compress_epilog_v2` fusion are read from device-dispatch code, not executed. Treat all performance claims as unverified until run on real hardware.

## Where It Breaks

| Failure mode | When it happens | Impact |
|---|---|---|
| IndexCache stale top-k | `use_index_cache` with `skip_topk` layers reading a buffer not refreshed by the owning c4 layer (e.g. MTP draft) | Wrong sparse selection; the code excludes `.mtp.` prefixes and asserts the buffer exists |
| A5-only paths unexercised | FP8 o_proj, `indexer_compress_epilog_v2`, FP8 cache dtypes on 910B/A2/A3 | Those branches are device-dispatched and not run on non-A5 |
| FlashComm + MTP mismatch | Sequence parallelism enabled but `_mtp_hidden_buffer` not all-gathered | NaN values and low acceptance rate (the code all-gathers and pads explicitly) |
| OTP static buffer under-capacity | `num_tokens > exchange_num_tokens` (static `potential_max_tokens` too small) | Raised ValueError before graph replay |
| Compressor zero-row scatter | `compressed_kv.shape[0] == 0` | Scatter must be skipped; A5 scatter dereferences `x.view()` on empty tensors |
| `quant_mode` beyond 0 | Any `query_quant_mode`/`key_quant_mode` other than per-token-head (0) | Not implemented in the operator |
| 310P devices | MLA/SFA/DSA models on 310P | Selection collapses to dense `AscendAttentionBackend310`; sparse/compressed models unsupported |

## One Thing to Remember

**DeepSeek-V4 inference on Ascend is one model override plus one DSA backend that drive five KV cache types and a chain of NPU custom operators** — mHC wraps the decoder, `compressor` folds KV, the Lightning Indexer selects compressed records, and device dispatch chooses the sparse-attention path. MTP and DSpark are distinct draft routes; the 0731 A3 recipe also enables DSA context parallel.

## Code Reading Path

1. **Model entry:** <a class="code-link" href="../../../../external-repos/vllm-ascend/vllm_ascend/models/__init__.py#L4" data-code-repo="vllm-ascend-32a59d4e349c" data-code-path="vllm_ascend/models/__init__.py" data-code-line="4" data-code-end-line="18"><code>register_model</code></a>; then `AscendDeepseekV4ForCausalLM` and `DeepseekV4Model.forward` (Sections 1–2).
2. **Layer + mHC:** `DeepseekV2DecoderLayer.forward`, `hc_pre`/`hc_post`, and the `npu_hc_pre_v2`/`npu_hc_post` registrations in <a class="code-link" href="../../../../external-repos/vllm-ascend/csrc/torch_binding.cpp#L2500" data-code-repo="vllm-ascend-32a59d4e349c" data-code-path="csrc/torch_binding.cpp" data-code-line="2500" data-code-end-line="2525"><code>torch_binding.cpp</code></a> (Section 3).
3. **Attention construction:** `DeepseekV4Attention.__init__` and `AscendDeepseekSparseAttention` (Sections 4, 6).
4. **MoE:** `DeepseekV4MoE.__init__`/`forward` (Section 5).
5. **Backend runtime:** `AscendDSABackend` selects the base, DSA-CP, or PCP builder/implementation; the documented A3 0731 recipe uses DSA-CP (Sections 7–9 and 14).
6. **Output projection + KV:** `_forward_o_proj`, `AscendMLAAttentionSpec`, `get_dsv4_block_sizes`, `AscendHybridKVCacheCoordinator` (Sections 10–11).
7. **Drafting:** the MTP baseline uses `DeepSeekV4MTP` and `DeepSeekMultiTokenPredictorLayer` (Section 12); the 0731 DSpark target-state handoff is in Section 14.
8. **C8 indexer depth:** see [DeepSeek-V4 Lightning Indexer C8 Quantization](deepseek-v4-lightning-indexer-c8.md).
