---
kind: web-extraction
source_url: "https://recipes.vllm.ai/XiaomiMiMo/MiMo-V2.6-Flash-RL"
final_url: "https://recipes.vllm.ai/XiaomiMiMo/MiMo-V2.6-Flash-RL"
canonical_url: "https://recipes.vllm.ai/XiaomiMiMo/MiMo-V2.6-Flash-RL"
title: "MiMo-V2.6-Flash-RL on vLLM — Serve command for H200, GB300 NVL4, MI300X, MI325X, MI355X"
author: ""
published_at: "2026-09-22"
captured_at: "2026-10-10T08:03:13.408Z"
content_sha256: cae25103d982084304d2ddcb5e404bffa0e913272c1a8aa2589e3178aa63d739
renderer: http
extractor: "defuddle@0.13.0 + turndown@7.2.4"
---

[![vLLM](https://docs.vllm.ai/en/latest/assets/logos/vllm-logo-text-light.png)![vLLM](https://docs.vllm.ai/en/latest/assets/logos/vllm-logo-text-dark.png)/Recipes](https://recipes.vllm.ai/)

[Browse](https://recipes.vllm.ai/browse)[Docs](https://docs.vllm.ai/)[GitHub](https://github.com/vllm-project/recipes)

Providers

[![Image](https://recipes.vllm.ai/providers/arcee-ai.png)Arcee AI](https://recipes.vllm.ai/arcee-ai)

[![Image](https://recipes.vllm.ai/providers/baidu.png)Ernie (Baidu)](https://recipes.vllm.ai/baidu)

[![Image](https://recipes.vllm.ai/providers/bharatgenai.jpeg)BharatGen](https://recipes.vllm.ai/bharatgenai)

[![Image](https://recipes.vllm.ai/providers/bosonai.png)Boson AI](https://recipes.vllm.ai/bosonai)

[![Image](https://recipes.vllm.ai/providers/ByteDance-Seed.png)Seed (ByteDance)](https://recipes.vllm.ai/ByteDance-Seed)

[![Image](https://recipes.vllm.ai/providers/deepseek-ai.png)DeepSeek](https://recipes.vllm.ai/deepseek-ai)

[![Image](https://recipes.vllm.ai/providers/dots-studio.png)Dots](https://recipes.vllm.ai/dots-studio)

[![Image](https://recipes.vllm.ai/providers/fishaudio.png)Fish Audio](https://recipes.vllm.ai/fishaudio)

[![Image](https://recipes.vllm.ai/providers/Google.png)Google](https://recipes.vllm.ai/Google)

[

i

ibm-granite](https://recipes.vllm.ai/ibm-granite)

[![Image](https://recipes.vllm.ai/providers/IFM.png)IFM](https://recipes.vllm.ai/IFM)

[![Image](https://recipes.vllm.ai/providers/inclusionAI.jpeg)inclusionAI](https://recipes.vllm.ai/inclusionAI)

[![Image](https://recipes.vllm.ai/providers/IndexTeam.png)IndexTeam](https://recipes.vllm.ai/IndexTeam)

[![Image](https://recipes.vllm.ai/providers/internlm.png)InternLM](https://recipes.vllm.ai/internlm)

[![Image](https://recipes.vllm.ai/providers/JetBrains.png)JetBrains](https://recipes.vllm.ai/JetBrains)

[![Image](https://recipes.vllm.ai/providers/jinaai.png)Jina AI](https://recipes.vllm.ai/jinaai)

[![Image](https://recipes.vllm.ai/providers/Lightricks.webp)Lightricks](https://recipes.vllm.ai/Lightricks)

[![Image](https://recipes.vllm.ai/providers/LiquidAI.png)Liquid AI](https://recipes.vllm.ai/LiquidAI)

[![Image](https://recipes.vllm.ai/providers/meituan-longcat.png)LongCat (Meituan)](https://recipes.vllm.ai/meituan-longcat)

[![Image](https://recipes.vllm.ai/providers/meta-llama.png)Meta](https://recipes.vllm.ai/meta-llama)

[![Image](https://recipes.vllm.ai/providers/meta-models.png)Muse (Meta)](https://recipes.vllm.ai/meta-models)

[![Image](https://recipes.vllm.ai/providers/microsoft.png)Microsoft](https://recipes.vllm.ai/microsoft)

[![Image](https://recipes.vllm.ai/providers/mindlab-research.png)MindLab Research](https://recipes.vllm.ai/mindlab-research)

[![Image](https://recipes.vllm.ai/providers/MiniMaxAI.jpeg)MiniMax](https://recipes.vllm.ai/MiniMaxAI)

[![Image](https://recipes.vllm.ai/providers/mistralai.png)Mistral AI](https://recipes.vllm.ai/mistralai)

[![Image](https://recipes.vllm.ai/providers/moonshotai.jpeg)Moonshot AI](https://recipes.vllm.ai/moonshotai)

[![Image](https://recipes.vllm.ai/providers/nvidia.png)NVIDIA](https://recipes.vllm.ai/nvidia)

[![Image](https://recipes.vllm.ai/providers/openai.png)OpenAI](https://recipes.vllm.ai/openai)

[![Image](https://recipes.vllm.ai/providers/openbmb.png)MiniCPM (OpenBMB)](https://recipes.vllm.ai/openbmb)

[![Image](https://recipes.vllm.ai/providers/OpenGVLab.jpeg)InternVL (OpenGVLab)](https://recipes.vllm.ai/OpenGVLab)

[![Image](https://recipes.vllm.ai/providers/OpenMOSS-Team.png)OpenMOSS](https://recipes.vllm.ai/OpenMOSS-Team)

[![Image](https://recipes.vllm.ai/providers/PaddlePaddle.png)PaddlePaddle](https://recipes.vllm.ai/PaddlePaddle)

[![Image](https://recipes.vllm.ai/providers/pfnet.png)Preferred Networks](https://recipes.vllm.ai/pfnet)

[![Image](https://recipes.vllm.ai/providers/poolside.png)Poolside](https://recipes.vllm.ai/poolside)

[![Image](https://recipes.vllm.ai/providers/Qwen.png)Qwen](https://recipes.vllm.ai/Qwen)

[![Image](https://recipes.vllm.ai/providers/stabilityai.png)Stability AI](https://recipes.vllm.ai/stabilityai)

[![Image](https://recipes.vllm.ai/providers/stepfun-ai.png)StepFun](https://recipes.vllm.ai/stepfun-ai)

[![Image](https://recipes.vllm.ai/providers/tencent.png)Hunyuan (Tencent)](https://recipes.vllm.ai/tencent)

[![Image](https://recipes.vllm.ai/providers/thinkingmachines.png)Thinking Machines Lab](https://recipes.vllm.ai/thinkingmachines)

[![Image](https://recipes.vllm.ai/providers/Wan-AI.png)Wan (Alibaba)](https://recipes.vllm.ai/Wan-AI)

[![Image](https://recipes.vllm.ai/providers/XiaomiMiMo.jpeg)MiMo (Xiaomi)](https://recipes.vllm.ai/XiaomiMiMo)

[MiMo-V2.6-Flash-RL](https://recipes.vllm.ai/XiaomiMiMo/MiMo-V2.6-Flash-RL)[MiMo-V2.6-Pro-RL](https://recipes.vllm.ai/XiaomiMiMo/MiMo-V2.6-Pro-RL)[MiMo-V2.5](https://recipes.vllm.ai/XiaomiMiMo/MiMo-V2.5)[MiMo-V2.5-Pro](https://recipes.vllm.ai/XiaomiMiMo/MiMo-V2.5-Pro)[MiMo-V2-Flash](https://recipes.vllm.ai/XiaomiMiMo/MiMo-V2-Flash)

[![Image](https://recipes.vllm.ai/providers/zai-org.png)GLM (Z-AI)](https://recipes.vllm.ai/zai-org)

![MiMo (Xiaomi)](https://recipes.vllm.ai/providers/XiaomiMiMo.jpeg)

XiaomiMiMo/MiMo-V2.6-Flash-RL
=============================

Xiaomi's efficiency-balanced omnimodal MoE (309B total / 15B active) with hybrid attention, FP8-compute/mxfp4-stored weights, 1M context, and a DFlash speculative decoder — the default MOPD checkpoint fixes the RL release's tool-call repetition

[![Image](https://recipes.vllm.ai/platform-logos/huggingface.svg)View on HuggingFace](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL)

moe309B / 15B1,048,576 ctx[vLLM 0.31.0+](https://vllm.ai/#quick-start "Install vLLM")textmultimodal

Guide[](#recipe-guide "Copy link to this section")

Overview[](#overview "Copy link to this section")
-------------------------------------------------

This recipe serves the MiMo-V2.6-Flash family from two checkpoints on the Variant row:

-   **MOPD** (default) — [XiaomiMiMo/MiMo-V2.6-Flash-MOPD](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-MOPD), the MOPD2-distilled upgrade and the recommended starting point for new deployments. It fixes the RL release's most visible agentic failure mode — tool-call repetition, where the model keeps re-issuing the same or highly similar tool calls while making no progress (the blog measures the Flash line as the more affected of the two) — with benchmark quality unchanged, folded into a regular MOPD2 pass via a short specialized-teacher run ([diagnosis blog](https://mimo.xiaomi.com/blog/mimo-v2-6-tool-call-repetition)).
-   **RL (original)** — [XiaomiMiMo/MiMo-V2.6-Flash-RL](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL), the 2026-09-22 RL release, kept for reproducing earlier results. It can get stuck re-issuing the same tool calls in agentic workloads; prefer **MOPD** unless you are reproducing a specific earlier run.

MOPD2 (Technical Report §5.6) is multi-teacher on-policy distillation: domain-specialized mixRL and SFT teachers supervise the student on three on-policy streams (standard MOPD, teacher-prefix OPD, and SFT-prefix OPD).

The architecture is shared by both checkpoints: a sparse MoE model with 309B total parameters and 15B active per token — 48 layers (1 dense + 47 MoE) with 256 routed experts (top-8), hybrid attention (sliding-window 128 and global attention), and a 5-layer DFlash-style MTP drafter that predicts 7 tokens per pass. The model is natively omnimodal (text, image, video, audio via a 681M-param MiMo ViT and audio encoders) and supports up to 1M tokens of context.

Weights are stored as mxfp4 and computed as FP8 (block-wise e4m3, 128x128), so the checkpoint is only 173 GB on disk. Loading this mixed storage format requires vLLM with the MiMo V2 mxfp4/bf16-router support ([vllm-project/vllm#57784](https://github.com/vllm-project/vllm/pull/57784)), first shipped in the v0.31.0 stable release.

Prerequisites[](#prerequisites "Copy link to this section")
-----------------------------------------------------------

-   Hardware: 4x H200 (TP4), 4x MI355X/MI325X/MI300X (TP4), or equivalent aggregate VRAM (>= 208 GB)

### Pull the vLLM docker image[](#pull-the-vllm-docker-image "Copy link to this section")

vLLM 0.31.0 is the first stable release that can load the mxfp4-stored weights ([vllm-project/vllm#57784](https://github.com/vllm-project/vllm/pull/57784)); earlier stable releases cannot:

```bash
docker pull vllm/vllm-openai:v0.31.0
```

MI300X / MI325X (gfx942) additionally need the MiMo MXFP4 routing gate ([vllm-project/vllm#58262](https://github.com/vllm-project/vllm/pull/58262)), which merged after the v0.31.0 branch cut — on those GPUs use a nightly wheel built after 2026-09-30 (`uv pip install -U vllm --extra-index-url https://wheels.vllm.ai/nightly/cu130`) or the ROCm nightly image.

This release also registers the `mimo` tool-call and reasoning parsers with the Rust frontend ([vllm-project/vllm#57933](https://github.com/vllm-project/vllm/pull/57933)).

The Hugging Face model cards point at `vllm/vllm-openai:mimov25-cu129`, the V2.5-era image; that tag predates the mxfp4-stored-weight support linked above, so use the v0.31.0 image recommended above.

Launch command[](#launch-command "Copy link to this section")
-------------------------------------------------------------

Single-node TP4 (H200). Selecting the **RL (original)** variant in the builder swaps only the checkpoint id (`XiaomiMiMo/MiMo-V2.6-Flash-RL`); everything else is identical:

```bash
vllm serve XiaomiMiMo/MiMo-V2.6-Flash-MOPD \
  --tensor-parallel-size 4 \
  --trust-remote-code \
  --gpu-memory-utilization 0.95 \
  --max-model-len auto \
  --reasoning-parser mimo \
  --tool-call-parser mimo \
  --enable-auto-tool-choice \
  --generation-config vllm
```

### AMD (MI355X, gfx950)[](#amd-mi355x-gfx950 "Copy link to this section")

```bash
VLLM_ROCM_USE_AITER=1 \
VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4 \
vllm serve XiaomiMiMo/MiMo-V2.6-Flash-MOPD \
  --tensor-parallel-size 4 \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --max-model-len auto \
  --reasoning-parser mimo \
  --tool-call-parser mimo \
  --enable-auto-tool-choice \
  --generation-config vllm
```

MI355X also runs TP2 just fine.

### AMD (MI300X / MI325X, gfx942)[](#amd-mi300x--mi325x-gfx942 "Copy link to this section")

gfx942 has no native MXFP4 MoE path: without an explicit backend vLLM's oracle falls back to OCP emulation — correct, but about 6.5x lower output throughput. Select AITER's Triton W4A16 kernel instead with `--moe-backend aiter_triton_mxfp4_bf16`; supported since [vllm-project/vllm#58262](https://github.com/vllm-project/vllm/pull/58262) (merged 2026-09-30, after the v0.31.0 branch cut — not in any stable release yet, so this flag needs a nightly wheel; it also needs aiter >= 0.1.20, which the ROCm nightly image ships). Verified end-to-end on MI300X in that PR — see it for the gsm8k and `vllm bench serve` numbers against the emulation baseline:

```bash
VLLM_ROCM_USE_AITER=1 \
VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4 \
vllm serve XiaomiMiMo/MiMo-V2.6-Flash-MOPD \
  --tensor-parallel-size 2 \
  --trust-remote-code \
  --max-model-len auto \
  --reasoning-parser mimo \
  --tool-call-parser mimo \
  --enable-auto-tool-choice \
  --moe-backend aiter_triton_mxfp4_bf16 \
  --generation-config vllm
```

With DFlash speculative decoding (drafter ships inside the checkpoint). vLLM does not resolve a `<repo>/dflash` Hub subfolder as the draft model — `model` must be a concrete local path to the `dflash/` directory inside the downloaded snapshot. Resolve it per checkpoint (run `hf download <checkpoint>` first if it is not cached yet):

```bash
# MOPD variant
DFLASH_DIR=$(ls -d ~/.cache/huggingface/hub/models--XiaomiMiMo--MiMo-V2.6-Flash-MOPD/snapshots/*/dflash)
# RL (original) variant
# DFLASH_DIR=$(ls -d ~/.cache/huggingface/hub/models--XiaomiMiMo--MiMo-V2.6-Flash-RL/snapshots/*/dflash)
echo "$DFLASH_DIR"
```

```bash
vllm serve XiaomiMiMo/MiMo-V2.6-Flash-MOPD \
  --tensor-parallel-size 4 \
  --trust-remote-code \
  --gpu-memory-utilization 0.95 \
  --max-model-len auto \
  --speculative-config "{\"method\":\"dflash\",\"model\":\"$DFLASH_DIR\",\"num_speculative_tokens\":7}" \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --reasoning-parser mimo \
  --tool-call-parser mimo \
  --enable-auto-tool-choice \
  --generation-config vllm
```

The drafter inherits the target's tensor-parallel size by default. To pin it explicitly, add `"draft_tensor_parallel_size": 4` to the speculative config (useful if you want the draft on fewer GPUs than the target, or to silence mis-detection on asymmetric topologies).

Client Usage[](#client-usage "Copy link to this section")
---------------------------------------------------------

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "XiaomiMiMo/MiMo-V2.6-Flash-MOPD",
    "messages": [{"role": "user", "content": "Hello MiMo!"}],
    "temperature": 1.0,
    "top_p": 0.95,
    "chat_template_kwargs": {"enable_thinking": true}
  }'
```

Recommended sampling: `temperature=1.0`, `top_p=0.95`. Set `"enable_thinking": false` (or omit the kwargs) to disable thinking mode.

References[](#references "Copy link to this section")
-----------------------------------------------------

-   [MiMo-V2.6-Flash-MOPD on Hugging Face](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-MOPD)
-   [MiMo-V2.6-Flash-RL on Hugging Face](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL)
-   [MiMo-V2.6 technical report](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL/blob/main/MiMo_V2_6_technical_report.pdf) (§5.6 covers MOPD2)
-   [Diagnosing and Mitigating Tool-Call Repetition in MiMo-V2.6](https://mimo.xiaomi.com/blog/mimo-v2-6-tool-call-repetition)
-   [MiMo-V2.6 blog post](https://mimo.xiaomi.com/mimo-v2-6)
-   [vLLM PR #57784 — mxfp4 MoE / bf16 router / DFlash for MiMo V2](https://github.com/vllm-project/vllm/pull/57784)
-   [vLLM PR #57933 — MiMo tool-call/reasoning parsers for the Rust frontend](https://github.com/vllm-project/vllm/pull/57933)
-   [vLLM PR #58262 — MiMo-V2.6 MXFP4 on gfx942 (`--moe-backend aiter_triton_mxfp4_bf16`)](https://github.com/vllm-project/vllm/pull/58262)
-   [SGLang MiMo cookbook](https://docs.sglang.io/cookbook/autoregressive/Xiaomi/MiMo-V2.5)
-   [vLLM ROCm GPU install docs](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/?device=rocm)

Updated 2026-10-07 [Edit recipe](https://github.com/vllm-project/recipes/edit/main/models/XiaomiMiMo/MiMo-V2.6-Flash-RL.yaml) [Report issue](https://github.com/vllm-project/recipes/issues)Related: [MiMo-V2.6-Pro-RL](https://recipes.vllm.ai/XiaomiMiMo/MiMo-V2.6-Pro-RL) · [MiMo-V2-Flash](https://recipes.vllm.ai/XiaomiMiMo/MiMo-V2-Flash)

[GitHub](https://github.com/vllm-project/recipes)[Request a recipe](https://github.com/vllm-project/recipes/issues)[Documentation](https://docs.vllm.ai/)[Supported Models & Hardware](https://vllm.ai/#compatibility)[Install vLLM](https://vllm.ai/#quick-start)[JSON API](https://recipes.vllm.ai/models.json)
