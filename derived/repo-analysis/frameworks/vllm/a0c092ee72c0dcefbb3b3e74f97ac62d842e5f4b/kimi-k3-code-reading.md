---
kind: repository-analysis
repository_id: github:vllm-project/vllm@a0c092ee72c0dcefbb3b3e74f97ac62d842e5f4b
commit: a0c092ee72c0dcefbb3b3e74f97ac62d842e5f4b
source_record: raw/frameworks/vllm-codebase--github-a0c092ee72c0.md
generated: 2026-07-29
---

# vLLM Kimi K3 Code Reading Notes

## Required Code Evidence

| Docs page | Finding | File | Symbol | Start | End |
|---|---|---|---|---:|---:|
| `docs/frameworks/vllm/vllm-kimi-k3-code-reading.md` | platform-dispatch | `vllm/models/kimi_k3/__init__.py` | `kimi_k3` platform dispatch | 10 | — |
| `docs/frameworks/vllm/vllm-kimi-k3-code-reading.md` | model-entry | `vllm/models/kimi_k3/nvidia/model.py` | `KimiK3ForConditionalGeneration` | 1441 | — |
| `docs/frameworks/vllm/vllm-kimi-k3-code-reading.md` | text-model | `vllm/models/kimi_k3/nvidia/model.py` | `KimiLinearForCausalLM` | 1311 | — |
| `docs/frameworks/vllm/vllm-kimi-k3-code-reading.md` | decoder-selection | `vllm/models/kimi_k3/nvidia/model.py` | `KimiDecoderLayer` | 689 | — |
| `docs/frameworks/vllm/vllm-kimi-k3-code-reading.md` | mla-layer | `vllm/models/kimi_k3/nvidia/mla.py` | `MultiHeadLatentAttention` | 102 | — |
| `docs/frameworks/vllm/vllm-kimi-k3-code-reading.md` | moe-router | `vllm/models/kimi_k3/nvidia/model.py` | `KimiMoE` | 381 | — |
| `docs/frameworks/vllm/vllm-kimi-k3-code-reading.md` | latent-moe-runner | `vllm/model_executor/layers/fused_moe/runner/latent_moe_runner.py` | `LatentMoERunner` | 22 | — |
| `docs/frameworks/vllm/vllm-kimi-k3-code-reading.md` | latent-moe-tail | `vllm/models/kimi_k3/nvidia/ops/latent_moe_tail.py` | `KimiK3LatentMoETailOp` | 40 | — |
| `docs/frameworks/vllm/vllm-kimi-k3-code-reading.md` | mtp-draft | `vllm/models/kimi_k3/nvidia/mtp.py` | `KimiK3MTP` | 202 | — |
| `docs/frameworks/vllm/vllm-kimi-k3-code-reading.md` | renderer | `vllm/renderers/kimi_k3.py` | `KimiK3Renderer` | 143 | — |

## Runtime Flow Evidence

1. Registration and request surface — `platform-dispatch`, `renderer`.
2. Model construction — `model-entry`, `text-model`.
3. Hybrid attention selection — `decoder-selection`, `mla-layer`.
4. Routed MoE forward — `moe-router`, `latent-moe-runner`, `latent-moe-tail`.
5. Speculative decoding — `mtp-draft`.

## Evidence Map

### Registration and request surface

- `vllm/config/model.py` adds `kimi_k3` as a tokenizer mode and sets it
  automatically for `KimiK3ForConditionalGeneration`.
- `vllm/tokenizers/registry.py` maps `kimi_k3` to the cached Hugging Face
  tokenizer path.
- `vllm/renderers/registry.py` maps `kimi_k3` to `KimiK3Renderer`, and
  `vllm/renderers/kimi_k3.py` applies K3's Python XTML chat-template encoding,
  thinking-effort normalization, image media defaults, and tool-result
  reordering.
- `vllm/parser/parser_manager.py`, `vllm/parser/kimi_k3.py`,
  `vllm/reasoning/kimi_k3_reasoning_parser.py`, and
  `vllm/tool_parsers/kimi_k3_tool_parser.py` connect K3's XTML reasoning and
  tool-call parser.
- `vllm/tool_parsers/structural_tag_registry.py` registers the `kimi_k3`
  structural tag grammar for constrained tool-call output.

### Config, processor, and multimodal model

- `vllm/transformers_utils/configs/kimi_k3.py` defines `KimiK3Config` and
  `KimiK3VisionConfig`. `KimiK3Config` owns a `KimiLinearConfig` text config
  and a Kimi K3 vision config, then forces the vision projector output size to
  match text hidden size.
- `vllm/transformers_utils/processors/kimi_k3.py` adapts bare image inputs into
  the media dictionaries expected by the K3 vision processor and leaves
  resolution-aware placeholder expansion to the vLLM model side.
- `vllm/models/kimi_k3/__init__.py` dispatches to NVIDIA or AMD
  implementations based on platform.
- `vllm/models/kimi_k3/nvidia/model.py` defines
  `KimiK3ForConditionalGeneration`, registering it with the multimodal
  registry. The class builds MoonViT/Kimi-K2.5 vision, a Kimi-K2.5 multimodal
  projector, and an inner `KimiLinearForCausalLM` text model.

### Text model, attention, and residual path

- `KimiLinearModel` builds embeddings, decoder layers, final norm, optional
  attention-residual state, and optional sequence-parallel sharding.
- `KimiDecoderLayer` chooses `KimiK3DeltaAttention` for KDA layers and
  `MultiHeadLatentAttention` for NoPE MLA layers.
- `vllm/models/kimi_k3/nvidia/kda.py` defines `KimiK3DeltaAttention` with
  full-rank gate support, FlashKDA prefill selection, fused decode support,
  speculative decode splitting, and KDA-specific metadata.
- `vllm/models/kimi_k3/nvidia/mla.py` defines a self-contained MLA layer that
  owns its backend, KV cache spec, fused prefill/decode cache-insert kernels,
  optional output gate overlap, and `kv_b_proj` absorption into decode-time
  `W_UK_T` and `W_UV`.
- `vllm/models/kimi_k3/nvidia/ops/attn_res.py` wraps
  `torch.ops._C.kimi_k3_attn_res` and Triton fallback kernels for K3's
  attention-residual accumulation and normalization.

### MoE path

- `KimiMoE` builds a fp32 router gate, correction bias, optional shared
  experts, latent routed expert down/up projections, and either
  `KimiK3MegaMoEExperts` or `FusedMoE`.
- `KimiK3MegaMoEExperts` adapts the DeepSeek V4 MegaMoE expert path to Kimi K3
  by transforming FP8/FP4 expert weights for DeepGEMM, caching symmetric
  buffers, mapping EPLB logical experts to physical experts, and calling
  `deep_gemm.fp8_fp4_mega_moe(...)`.
- Non-MegaMoE mode uses upstream `FusedMoE` with `LatentMoERunner` when latent
  MoE is active. `KimiMoE.forward()` overlaps the router gate with routed
  latent down-projection when possible, then sends latent states into routed
  experts and original hidden states into shared experts.
- `vllm/model_executor/layers/fused_moe/runner/latent_moe_runner.py` defines
  `LatentMoERunner`. Its fused path concatenates routed latent and shared
  output for one all-reduce, normalizes/up-projects the latent output locally,
  overlaps shared-output all-reduce with up-projection where possible, and has
  an optional K3 tail-fusion path.
- `vllm/models/kimi_k3/nvidia/ops/latent_moe_tail.py` defines the optional
  CuTe DSL K3 latent-MoE tail fusion. It is constrained to CUDA SM100,
  bfloat16, TP 8/16, hidden size 7168, latent size 3584, and up to 16 tokens.

### MTP and speculative decoding

- `vllm/models/kimi_k3/nvidia/mtp.py` defines `KimiK3MTP`, an inference-only
  multi-token-prediction draft model. Each MTP layer fuses current token
  embedding and previous hidden states, runs a Kimi decoder layer with
  attention residual disabled, and shares the LM head.
- `vllm/models/kimi_k3/nvidia/dspark_mla.py` defines a dense MLA DSpark draft
  model path.

## Reproduction Commands

```bash
git -C external-repos/vllm rev-parse HEAD
find external-repos/vllm/vllm/models/kimi_k3 -maxdepth 4 -type f | sort
rg -n "KimiK3ForConditionalGeneration|KimiLinearForCausalLM|KimiK3MTP|LatentMoERunner|KimiK3MegaMoEExperts|kimi_k3" external-repos/vllm/vllm --glob '*.py'
rg -n "fused_kimi_k3|kimi_k3_attn_res|VLLM_ENABLE_K3_LATENT_MOE_TAIL_FUSION" external-repos/vllm/vllm external-repos/vllm/csrc --glob '*.py' --glob '*.cu' --glob '*.h'
```

## Limitations

- Static code reading only; no Kimi K3 checkpoint was loaded and no CUDA/ROCm
  kernel path was executed.
- The reading focused on NVIDIA paths first because the NVIDIA implementation
  includes the richest K3-specific MoE, MLA, KDA, low-latency GEMM, and
  tail-fusion code. AMD entrypoints were identified but not traced in equal
  depth.
