---
title: "MiMo-V2.6-Pro-MOPD: vLLM Inference and Deployment"
summary: "Model-specific vLLM serving configuration, hardware reference shapes, weight-format requirements, and optional DFlash decoding for Xiaomi's MiMo-V2.6-Pro-MOPD."
layout: default
confidence: medium
sources:
  - raw/frameworks/serving/vllm/mimo-v2-6-pro-mopd-model-card--web-2026-10-10-33e8a9c2cb40.html
  - raw/frameworks/serving/vllm/mimo-v2-6-pro-mopd-model-card--web-2026-10-10-33e8a9c2cb40.metadata.json
  - derived/web-markdown/frameworks/serving/vllm/mimo-v2-6-pro-mopd-model-card--web-2026-10-10-33e8a9c2cb40.md
  - raw/frameworks/serving/vllm/mimo-v2-6-pro-rl-vllm-recipe--web-2026-10-10-6d7d2094d16b.html
  - raw/frameworks/serving/vllm/mimo-v2-6-pro-rl-vllm-recipe--web-2026-10-10-6d7d2094d16b.metadata.json
  - derived/web-markdown/frameworks/serving/vllm/mimo-v2-6-pro-rl-vllm-recipe--web-2026-10-10-6d7d2094d16b.md
updated: 2026-10-10
---

# MiMo-V2.6-Pro-MOPD: vLLM Inference and Deployment

**Model:** [XiaomiMiMo/MiMo-V2.6-Pro-MOPD](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-MOPD)  
**vLLM recipe:** [MiMo-V2.6-Pro-RL on vLLM](https://recipes.vllm.ai/XiaomiMiMo/MiMo-V2.6-Pro-RL) — its current default variant and launch command target MOPD.  
**Related pages:** [vLLM](index.md), [vLLM data-parallel deployment](data-parallel-deployment/index.md), [Mixture of Experts](../../../terms/mixture-of-experts.md), and [KV Cache](../../../terms/kv-cache.md).

The model card and vLLM recipe were captured on 2026-10-10. The recipe reports
an update on 2026-10-07. The configurations below are source-backed reference
setups; no model load, GPU generation, or performance test was run for this
page.

## TL;DR

**What:** MiMo-V2.6-Pro-MOPD is the MOPD2-distilled Pro checkpoint; Xiaomi
positions it as the new-use variant that mitigates repeated tool calls.

**How:** The vLLM recipe uses vLLM 0.31.0+, a single-node H200 TP8 baseline,
the `mimo` reasoning/tool parsers, and the `vllm` generation config.

**The deployment constraint:** The recipe estimates 566 GB of checkpoint
weights and requires at least 680 GB of aggregate GPU memory; 42B active
parameters does not mean only 42B parameters must be loaded.

## What Is Being Served

![MiMo-V2.6 architecture with visual and audio encoders, a hybrid attention MoE backbone, and an MTP block](assets/mimo-v2-6-pro-mopd-architecture.png)

*Source: [Xiaomi's MiMo-V2.6-Pro-MOPD model card](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-MOPD/blob/main/assets/architecture.png). ① Vision and audio encoders feed the multimodal backbone. ② The backbone combines sliding-window and global attention with sparse experts. ③ The model includes a multi-token prediction block; serving it with DFlash requires an additional vLLM speculative config.*

The model card describes MOPD as an upgrade to MiMo-V2.6-Pro-RL. The vLLM
recipe presents MOPD as its default variant and recommends it for new
deployments; keep the RL checkpoint when reproducing earlier runs.

| Serving-relevant property | Released configuration | Deployment implication |
|---|---|---|
| Sparse core | 1.02T total / 42B active parameters; 70 layers, 384 routed experts, top-8 | Active compute is not the same as resident weight memory. |
| Attention and context | 60 sliding-window layers and 10 global-attention layers; window 128; maximum context 1,048,576 tokens | Keep the recipe's `--max-model-len auto` as the starting point; a 1M-token ceiling is not a promise of 1M-token concurrency. |
| Weight and compute format | MXFP4-stored weights; FP8 block-wise e4m3 compute (128 × 128); 566 GB checkpoint | Use a vLLM release that supports the MiMo V2 MXFP4/BF16-router path. |
| Modalities | Text, image, video, and audio; 681M-parameter vision encoder plus audio encoders | The text smoke test below does not validate image, video, or audio inputs. |
| Draft model | Five-layer DFlash-style MTP drafter; predicts up to seven tokens per pass | The checkpoint includes a drafter, but the recipe's explicit DFlash launch adds a separate `--speculative-config`. |

The long-context limit also increases [KV-cache](../../../terms/kv-cache.md)
pressure as requests grow. The reference command does not force the advertised
1,048,576-token maximum.

## Runtime Version and Hardware

The current vLLM recipe states that **vLLM 0.31.0 is the first stable release
that can load these MXFP4-stored weights**; earlier stable releases cannot.
Its pinned Docker reference is:

```bash
docker pull vllm/vllm-openai:v0.31.0
```

The model card's general `pip install vllm` snippet does not pin a version. It
also links the V2.5-era `vllm/vllm-openai:mimov25-cu129` image. The newer vLLM
recipe says that image predates the required MXFP4 support, so use the v0.31.0
image or a later validated vLLM build instead.

| Platform | Recipe reference shape | Notes |
|---|---|---|
| NVIDIA H200 | One node, 8 GPUs, TP8 | The launch command below is the recipe's baseline. |
| AMD MI355X | 4 GPUs, TP4 | Use a ROCm-compatible vLLM environment and the separate settings below. |
| Other hardware | At least 680 GB aggregate VRAM, per the recipe | Aggregate capacity alone does not establish kernel support or equivalent performance. |

The captured recipe title also names GB300 NVL4, but its explicit launch
examples provide H200 TP8 and MI355X TP4 settings. Do not infer a GB300
parallelism shape from the title alone.

## H200 TP8 Launch

This is the model-specific single-node command in the vLLM recipe:

```bash
vllm serve XiaomiMiMo/MiMo-V2.6-Pro-MOPD \
  --tensor-parallel-size 8 \
  --trust-remote-code \
  --gpu-memory-utilization 0.95 \
  --max-model-len auto \
  --reasoning-parser mimo \
  --tool-call-parser mimo \
  --enable-auto-tool-choice \
  --generation-config vllm
```

| Setting | Purpose |
|---|---|
| `--tensor-parallel-size 8` | Matches the recipe's single-node H200 topology. |
| `--trust-remote-code` | Enables the model repository's custom model code. Pin and review the model revision when reproducibility or code provenance matters. |
| `--gpu-memory-utilization 0.95` | Uses the recipe's 95% vLLM memory target; reduce it when sharing GPUs, then re-check load and KV-cache capacity. |
| `--max-model-len auto` | Preserves the recipe's memory-aware reference setting instead of forcing the full advertised context. |
| `--reasoning-parser mimo` | Selects the MiMo reasoning parser. |
| `--tool-call-parser mimo --enable-auto-tool-choice` | Enables MiMo tool-call parsing and automatic tool selection for tool-using clients. |
| `--generation-config vllm` | Uses the generation-config mode selected by the model-specific recipe. |

The reference command does not set `--dtype` or add a generic quantization
override. Keep the checkpoint's MXFP4 loading behavior tied to the supported
vLLM version rather than copying unverified dtype flags from unrelated recipes.

## OpenAI-Compatible Smoke Test

The recipe's client example uses the OpenAI chat-completions endpoint:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "XiaomiMiMo/MiMo-V2.6-Pro-MOPD",
    "messages": [{"role": "user", "content": "Hello MiMo!"}],
    "temperature": 1.0,
    "top_p": 0.95,
    "chat_template_kwargs": {"enable_thinking": true}
  }'
```

The recipe recommends `temperature=1.0` and `top_p=0.95`. Set
`"enable_thinking": false` or omit that template keyword to disable thinking.
For tool requests, send the OpenAI `tools` payload; the server-side MiMo parser
and auto-choice flags are already enabled above.

## AMD MI355X TP4 Variant

The vLLM recipe's MI355X configuration uses ROCm-specific environment settings
and FP8 KV cache:

```bash
VLLM_ROCM_USE_AITER=1 \
VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4 \
vllm serve XiaomiMiMo/MiMo-V2.6-Pro-MOPD \
  --tensor-parallel-size 4 \
  --trust-remote-code \
  --gpu-memory-utilization 0.93 \
  --kv-cache-dtype fp8 \
  --max-model-len auto \
  --reasoning-parser mimo \
  --tool-call-parser mimo \
  --enable-auto-tool-choice \
  --generation-config vllm
```

The recipe says `VLLM_ROCM_USE_AITER=1` is required to use the AITER MXFP4 MoE
and FP8 block-scaled GEMM kernels. It describes TP4 as the MI355X default and
says vLLM repacks the checkpoint's QKV shards during loading. Treat these as
MI355X-specific instructions, not generic AMD settings.

## Optional DFlash Speculative Decoding

The checkpoint includes a DFlash drafter, but vLLM needs its **local
`dflash/` directory** as the speculative model path. First download the
checkpoint, then set `DFLASH_DIR` to the absolute path of that directory in the
resolved local snapshot:

```bash
DFLASH_DIR=/absolute/path/to/local-snapshot/dflash

vllm serve XiaomiMiMo/MiMo-V2.6-Pro-MOPD \
  --tensor-parallel-size 8 \
  --trust-remote-code \
  --gpu-memory-utilization 0.95 \
  --max-model-len auto \
  --speculative-config "{\"method\":\"dflash\",\"model\":\"${DFLASH_DIR}\",\"num_speculative_tokens\":7}" \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --reasoning-parser mimo \
  --tool-call-parser mimo \
  --enable-auto-tool-choice \
  --generation-config vllm
```

Do not pass the Hub repository ID or a textual `<repo>/dflash` subfolder as
the speculative model path; the recipe says vLLM needs a concrete local path.
The drafter inherits the target tensor-parallel size by default. Set
`draft_tensor_parallel_size` explicitly only when using a different draft
topology or when the topology needs disambiguation.

## Deployment Checks

| Failure condition | Why it matters | Response |
|---|---|---|
| Stable vLLM older than 0.31.0 or the V2.5-era `mimov25-cu129` image | The recipe says these predate the MXFP4 loading support used by this checkpoint. | Start from vLLM 0.31.0 or a later build verified against MiMo-V2.6. |
| Forcing the full 1M context without checking cache capacity | Long requests consume more KV-cache memory and can reduce concurrent capacity or prevent a request from fitting. | Keep `auto` initially; measure the target prompt/output lengths and concurrency before setting an explicit cap. |
| DFlash configured with a Hub ID instead of the local `dflash/` directory | The recipe says vLLM does not resolve the Hub subfolder as a draft model path. | Download the checkpoint and pass the concrete local directory. |
| Treating a text-only success as multimodal validation | The model card lists image, video, and audio support, but the smoke test above sends text only. | Exercise each required modality and the production chat template before rollout. |

**Remember:** this is a 566 GB MXFP4 checkpoint with a 680 GB aggregate-VRAM
reference floor, not a 42B-weight model. Pin a compatible vLLM build first,
start with the documented TP topology and automatic context setting, then opt
into DFlash only after resolving its local drafter path.

## Go Deeper

- **Model card:** [XiaomiMiMo/MiMo-V2.6-Pro-MOPD](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-MOPD)
- **vLLM deployment recipe:** [MiMo-V2.6-Pro-RL on vLLM](https://recipes.vllm.ai/XiaomiMiMo/MiMo-V2.6-Pro-RL) — currently serves MOPD as its default variant.
- **Scale-out:** [vLLM Data-Parallel Deployment](data-parallel-deployment/index.md)
- **Memory background:** [KV Cache](../../../terms/kv-cache.md) and [Mixture of Experts](../../../terms/mixture-of-experts.md)
