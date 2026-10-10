---
kind: web-extraction
source_url: "https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-MOPD"
final_url: "https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-MOPD"
canonical_url: "https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-MOPD"
title: "Hugging Face"
author: ""
published_at: ""
captured_at: "2026-10-10T07:47:34.267Z"
content_sha256: 33e8a9c2cb4023142e2412ee95f120f7e6a41e5cc0801773a3a9038760a424cb
renderer: http
extractor: "defuddle@0.13.0 + turndown@7.2.4"
---

[![Hugging Face's logo](https://huggingface.co/front/assets/huggingface_logo-noborder.svg) Hugging Face](https://huggingface.co/)

-   [Models](https://huggingface.co/models)
-   [Datasets](https://huggingface.co/datasets)
-   [Spaces](https://huggingface.co/spaces)
-   [Buckets new](https://huggingface.co/storage)
-   [Docs](https://huggingface.co/docs)
-   [Enterprise](https://huggingface.co/enterprise)
-   [Pricing](https://huggingface.co/pricing)
-   -   Website

        -   [Tasks](https://huggingface.co/tasks)
        -   [HuggingChat](https://huggingface.co/chat)
        -   [Collections](https://huggingface.co/collections)
        -   [Languages](https://huggingface.co/languages)
        -   [Organizations](https://huggingface.co/organizations)
    -   Community

        -   [Blog](https://huggingface.co/blog/zh)
        -   [Posts](https://huggingface.co/posts)
        -   [Daily Papers](https://huggingface.co/papers)
        -   [Hardware](https://huggingface.co/hardware)
        -   [Learn](https://huggingface.co/learn)
        -   [Discord](https://huggingface.co/join/discord)
        -   [Forum](https://discuss.huggingface.co/)
        -   [GitHub](https://github.com/huggingface)
    -   Solutions

        -   [Team & Enterprise](https://huggingface.co/enterprise)
        -   [Hugging Face PRO](https://huggingface.co/pro)
        -   [Enterprise Support](https://huggingface.co/support)
        -   [Inference Providers](https://huggingface.co/inference/models)
        -   [Inference Endpoints](https://huggingface.co/inference-endpoints)
        -   [Storage Buckets](https://huggingface.co/storage)

-   * * *

-   [Log In](https://huggingface.co/login)
-   [Sign Up](https://huggingface.co/join)

[

![Image](https://cdn-avatars.huggingface.co/v1/production/uploads/680cb7d1233834890a64acee/5w_4aLfF-7MAyaIPOV498.jpeg)

](https://huggingface.co/XiaomiMiMo)

[XiaomiMiMo](https://huggingface.co/XiaomiMiMo)

/

[MiMo-V2.6-Pro-MOPD](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-MOPD)

Like 29

Follow

![Image](https://cdn-avatars.huggingface.co/v1/production/uploads/680cb7d1233834890a64acee/5w_4aLfF-7MAyaIPOV498.jpeg) Xiaomi MiMo 5.01k

================================================================================================================================================================================================================================================================================================================================================================================================================================================================

[

Text Generation

](https://huggingface.co/models?pipeline_tag=text-generation)[

Transformers

](https://huggingface.co/models?library=transformers)[

Safetensors

](https://huggingface.co/models?library=safetensors)[

English

](https://huggingface.co/models?language=en)[

Chinese

](https://huggingface.co/models?language=zh)[

mimo\_v2

](https://huggingface.co/models?other=mimo_v2)[

multimodal

](https://huggingface.co/models?other=multimodal)[

vision-language

](https://huggingface.co/models?other=vision-language)[

audio

](https://huggingface.co/models?other=audio)[

agent

](https://huggingface.co/models?other=agent)[

video-understanding

](https://huggingface.co/models?other=video-understanding)[

long-context

](https://huggingface.co/models?other=long-context)[

conversational

](https://huggingface.co/models?other=conversational)[

custom\_code

](https://huggingface.co/models?other=custom_code)[

8-bit precision

](https://huggingface.co/models?other=8-bit)[

fp8

](https://huggingface.co/models?other=fp8)

License: mit

[Model card](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-MOPD) [Files Files and versions

xet

](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-MOPD/tree/main)[Community

2

](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-MOPD/discussions)

Deploy

Copy to bucket new

Use this model

### Instructions to use XiaomiMiMo/MiMo-V2.6-Pro-MOPD with libraries, inference providers, notebooks, and local apps. Follow these links to get started.

-   Libraries
-   [Transformers](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-MOPD?library=transformers)

    How to use XiaomiMiMo/MiMo-V2.6-Pro-MOPD with Transformers:

    \# Use a pipeline as a high-level helper
    from transformers import pipeline

    pipe = pipeline("text-generation", model="XiaomiMiMo/MiMo-V2.6-Pro-MOPD", trust\_remote\_code=True)
    messages = \[
        {"role": "user", "content": "Who are you?"},
    \]
    pipe(messages)

    \# pip install -U transformers accelerate
    # Load model directly
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from\_pretrained("XiaomiMiMo/MiMo-V2.6-Pro-MOPD", trust\_remote\_code=True, device\_map="auto")

-   Notebooks
-   [Google Colab](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-MOPD/colab)
-   [Kaggle](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-MOPD/kaggle)
-   Local Apps [Settings](https://huggingface.co/settings/local-apps "Set up your favorite local applications")
-   [vLLM](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-MOPD?local-app=vllm)

    How to use XiaomiMiMo/MiMo-V2.6-Pro-MOPD with vLLM:

    ##### Install from pip and serve model

    \# Install vLLM from pip:
    pip install vllm
    # Start the vLLM server:
    vllm serve "XiaomiMiMo/MiMo-V2.6-Pro-MOPD"
    # Call the server using curl (OpenAI-compatible API):
    curl -X POST "http://localhost:8000/v1/chat/completions" \\
    	-H "Content-Type: application/json" \\
    	--data '{
    		"model": "XiaomiMiMo/MiMo-V2.6-Pro-MOPD",
    		"messages": \[
    			{
    				"role": "user",
    				"content": "What is the capital of France?"
    			}
    		\]
    	}'

    ##### Use Docker

    docker model run hf.co/XiaomiMiMo/MiMo-V2.6-Pro-MOPD

-   [SGLang](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-MOPD?local-app=sglang)

    How to use XiaomiMiMo/MiMo-V2.6-Pro-MOPD with SGLang:

    ##### Install from pip and serve model

    \# Install SGLang from pip:
    pip install sglang
    # Start the SGLang server:
    python3 -m sglang.launch\_server \\
        --model-path "XiaomiMiMo/MiMo-V2.6-Pro-MOPD" \\
        --host 0.0.0.0 \\
        --port 30000
    # Call the server using curl (OpenAI-compatible API):
    curl -X POST "http://localhost:30000/v1/chat/completions" \\
    	-H "Content-Type: application/json" \\
    	--data '{
    		"model": "XiaomiMiMo/MiMo-V2.6-Pro-MOPD",
    		"messages": \[
    			{
    				"role": "user",
    				"content": "What is the capital of France?"
    			}
    		\]
    	}'

    ##### Use Docker images

    docker run --gpus all \\
        --shm-size 32g \\
        -p 30000:30000 \\
        -v ~/.cache/huggingface:/root/.cache/huggingface \\
        --env "HF\_TOKEN=<secret>" \\
        --ipc=host \\
        lmsysorg/sglang:latest \\
        python3 -m sglang.launch\_server \\
            --model-path "XiaomiMiMo/MiMo-V2.6-Pro-MOPD" \\
            --host 0.0.0.0 \\
            --port 30000
    # Call the server using curl (OpenAI-compatible API):
    curl -X POST "http://localhost:30000/v1/chat/completions" \\
    	-H "Content-Type: application/json" \\
    	--data '{
    		"model": "XiaomiMiMo/MiMo-V2.6-Pro-MOPD",
    		"messages": \[
    			{
    				"role": "user",
    				"content": "What is the capital of France?"
    			}
    		\]
    	}'

-   [Docker Model Runner](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-MOPD?local-app=docker-model-runner)

    How to use XiaomiMiMo/MiMo-V2.6-Pro-MOPD with Docker Model Runner:

    docker model run hf.co/XiaomiMiMo/MiMo-V2.6-Pro-MOPD

-   [Browse Quantizations](https://huggingface.co/models?other=base_model:quantized:XiaomiMiMo/MiMo-V2.6-Pro-MOPD) to use this model in llama.cpp, Ollama, LM Studio, or any compatible app.

-   [MiMo-V2.6-Pro-MOPD](#mimo-v26-pro-mopd "MiMo-V2.6-Pro-MOPD")
    -   [1\. Introduction](#1-introduction "1. Introduction")
        -   [How MOPD2 works](#how-mopd2-works "How MOPD2 works")
        -   [Tool-call repetition](#tool-call-repetition "Tool-call repetition")
    -   [Model Summary](#model-summary "Model Summary")

    -   [2\. Downloads](#2-downloads "2. Downloads")

    -   [3\. Model Architecture](#3-model-architecture "3. Model Architecture")
        -   [LLM Backbone](#llm-backbone "LLM Backbone")
        -   [Vision Encoder (MiMo ViT)](#vision-encoder-mimo-vit "Vision Encoder (MiMo ViT)")
        -   [Audio Encoders](#audio-encoders "Audio Encoders")
        -   [Speculative Decoder](#speculative-decoder "Speculative Decoder")
    -   [4\. Deployment](#4-deployment "4. Deployment")
        -   [SGLang](#sglang "SGLang")
        -   [vLLM](#vllm "vLLM")
    -   [Citation](#citation "Citation")

    -   [Contact](#contact "Contact")

 ![Xiaomi-MiMo](https://github.com/XiaomiMiMo/MiMo/raw/main/figures/Xiaomi_MiMo.png?raw=true)

| [🤗 HuggingFace](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-MOPD)  | [📰 Blog](https://mimo.xiaomi.com/mimo-v2-6)  | [🎨 Xiaomi MiMo API Platform](https://platform.xiaomimimo.com/)  | [🗨Xiaomi MiMo Studio](https://aistudio.xiaomimimo.com/)  | [💻 Xiaomi MiMo Desktop](https://mimo.xiaomimimo.com/desktop/)  |

**Community**  
[WeChat Group](https://huggingface.co/XiaomiMiMo/MiMo-V2.5-Pro/blob/main/assets/wechat.jpg)  |  [Discord](https://discord.gg/kKC2kNnQEX)  |  [Telegram](https://t.me/+3T-I0pekOVIyNDBl)  |  [Reddit](https://www.reddit.com/r/XiaomiMiMo_Official/)

[](#mimo-v26-pro-mopd)MiMo-V2.6-Pro-MOPD
========================================

[**Technical Report**](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL/blob/main/MiMo_V2_6_technical_report.pdf)

> **This is the MOPD upgrade of the [MiMo-V2.6-Pro-RL](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL) checkpoint.**
> 
> -   **MOPD2** (👉 [**Technical Report §5.6**](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL/blob/main/MiMo_V2_6_technical_report.pdf))  
>     Fuses several domain-specialized teachers into one model, extending to domains where reliable training-time verification is hard, such as long-horizon game development, scientific research and embodied intelligence.
> -   **Diagnosing and Mitigating Tool-Call Repetition in MiMo-V2.6** (👉 [**Technical Blog**](https://mimo.xiaomi.com/blog/mimo-v2-6-tool-call-repetition))  
>     An easy-to-overlook failure mode in which the model keeps issuing the same or highly similar tool calls, appearing busy while making no progress. Nothing fails outright, so it tends to go unnoticed. The MOPD stage handles it efficiently, with a short specialized-teacher run that converges quickly.

[](#1-introduction)1\. Introduction
-----------------------------------

### [](#how-mopd2-works)How MOPD2 works

MOPD2 distills several domain-specialized teachers into the student on-policy. The teachers fall into two families: **mixRL teachers**, trained on verifiable tasks, and **SFT teachers**, trained on synthetic demonstrations for open-domain tasks where a reliable reward is hard to design. Three streams contribute to a single update:

-   **Standard MOPD**: mixRL teachers supervise full autonomous rollouts.
-   **Teacher-Prefix OPD**: prefixes come from teacher rollouts. A trajectory with *k* assistant turns yields *k* history prefixes, one per turn. The model generates a single new turn from each, and the teacher scores it against the same history.
-   **SFT-Prefix OPD**: prefixes come from SFT demonstrations. The demonstration supplies the history, and the model writes its own continuation.

Method details are in [Technical Report §5.6](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL/blob/main/MiMo_V2_6_technical_report.pdf).

### [](#tool-call-repetition)Tool-call repetition

Following the release of MiMo-V2.6, tool-call repetition emerged as one of the most noticeable issues in agentic settings: the model would sometimes issue the same or highly similar tool calls repeatedly, consuming time and context without making progress. This checkpoint mitigates it.

[![Tool-call repetition rate on MiMo-V2.6-Pro before and after MOPD](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-MOPD/resolve/main/assets/mimo-repetition-rate-pro-en.png)](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-MOPD/blob/main/assets/mimo-repetition-rate-pro-en.png)

*Figure: response-level repetition rate on MiMo-V2.6-Pro, RL-stage versus this checkpoint, across context lengths and agent harnesses.*

The [technical blog](https://mimo.xiaomi.com/blog/mimo-v2-6-tool-call-repetition) has the full diagnosis. The fix is lightweight to train: a short specialized-teacher run that folds into the normal MOPD pass.

[](#model-summary)Model Summary
-------------------------------

-   **Architecture**: Sparse MoE (Mixture of Experts), 1.02T total / 42B activated parameters
-   **Context Length**: 1M tokens
-   **Modalities**: Text, Image, Video, Audio
-   **Vision Encoder**: 681M-param MiMo ViT (28 layers: 24 SWA + 4 Full)
-   **Audio Encoder**: 308M AudioTokenizer + 127M audio patch encoder
-   **Multi-Token Prediction (MTP)**: 5-layer speculative decoder

[![Figure 1: MiMo-V2.6 architecture — omni encoders, hybrid SWA backbone, and MTP blocks](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-MOPD/resolve/main/assets/architecture.png)](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-MOPD/blob/main/assets/architecture.png)

*Figure 1. MiMo-V2.6 architecture.*

[](#2-downloads)2\. Downloads
-----------------------------

Model

Download

**MiMo-V2.6-Pro-RL**

[🤗 HuggingFace](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL) · [🤖 ModelScope](https://www.modelscope.cn/models/XiaomiMiMo/MiMo-V2.6-Pro-RL)

**MiMo-V2.6-Flash-RL**

[🤗 HuggingFace](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL) · [🤖 ModelScope](https://www.modelscope.cn/models/XiaomiMiMo/MiMo-V2.6-Flash-RL)

**MiMo-V2.6-Pro-MOPD**

[🤗 HuggingFace](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-MOPD) · [🤖 ModelScope](https://www.modelscope.cn/models/XiaomiMiMo/MiMo-V2.6-Pro-MOPD)

**MiMo-V2.6-Flash-MOPD**

[🤗 HuggingFace](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-MOPD) · [🤖 ModelScope](https://www.modelscope.cn/models/XiaomiMiMo/MiMo-V2.6-Flash-MOPD)

[](#3-model-architecture)3\. Model Architecture
-----------------------------------------------

### [](#llm-backbone)LLM Backbone

Component

MiMo-V2.6-Pro-MOPD

Layers (Total / SWA / GA)

70 / 60 / 10

Hidden Size

6144

SWA Heads (Q/KV)

128 / 8

GA Heads (Q/KV)

128 / 8

Head Dimensions (QK / V)

192 / 128

Sliding Window Size

128

Routed Experts (Total / Activated)

384 / 8

Max Context Length

1M

MTP / Speculative Decoder

5 SWA layers, window 1024

The first Transformer block uses global attention with a dense FFN. Remaining blocks interleave local SWA and GA; both use sparse MoE FFNs without shared experts.

### [](#vision-encoder-mimo-vit)Vision Encoder (MiMo ViT)

Configuration

Value

Layers (Total / SWA / GA)

28 / 24 / 4

Hidden Size

1280

Attention Heads (Q / KV)

32 / 8

Head Dimension

64

Patch Size (T × H × W)

2 × 16 × 16

Sliding Window (Left / Right)

64 / 64

Spatial Merge Size

2 × 2

Parameters

681M

### [](#audio-encoders)Audio Encoders

AudioTokenizer encoder: 24 layers (12 SWA / 12 GA), hidden 1024, 20 RVQ codebooks, 308M parameters. Audio patch encoder: 6 layers, 127M parameters; four frames per patch (25 Hz → 6.25 Hz).

### [](#speculative-decoder)Speculative Decoder

5-layer SWA MTP drafter (DFlash-style). Predicts 7 subsequent tokens per forward pass for parallel verification.

[](#4-deployment)4\. Deployment
-------------------------------

For best performance, follow the [SGLang MiMo cookbook](https://docs.sglang.io/cookbook/autoregressive/Xiaomi/MiMo-V2.5). Docker image: `lmsysorg/sglang:latest`.

### [](#sglang)SGLang

```bash
sglang serve \
  --trust-remote-code \
  --model-path XiaomiMiMo/MiMo-V2.6-Pro-MOPD \
  --tp 16 \
  --dp 2 \
  --enable-dp-attention \
  --mm-enable-dp-encoder \
  --ep 16 \
  --moe-a2a-backend deepep \
  --moe-dense-tp-size 1 \
  --mem-fraction-static 0.7 \
  --max-running-requests 128 \
  --chunked-prefill-size 32768 \
  --page-size 64 \
  --swa-full-tokens-ratio 0.3 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --enable-multi-layer-eagle \
  --reasoning-parser mimo \
  --tool-call-parser mimo \
  --host 0.0.0.0 \
  --port 30000 \
  --nnodes 2 \
  --node-rank <node-rank> \
  --dist-init-addr <node0-ip>:20000
```

### [](#vllm)vLLM

Follow the [vLLM MiMo-V2.5 recipe](https://recipes.vllm.ai/XiaomiMiMo/MiMo-V2.5). Pre-built image: `docker pull vllm/vllm-openai:mimov25-cu129`.

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

Recommended sampling: `temperature=1.0`, `top_p=0.95`.

Also available in AI Studio, MiMo Code, Xiaomi MiMo Desktop, Xiaomi MiMo Open Platform API, and OpenRouter.

[](#citation)Citation
---------------------

```bibtex
@misc{mimo2026v26promopd,
  title={MiMo-V2.6-Pro-MOPD},
  author={{Xiaomi MiMo Team}},
  year={2026},
  howpublished={\url{https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-MOPD}},
}
```

[](#contact)Contact
-------------------

For questions or feedback, reach us at [mimo@xiaomi.com](mailto:mimo@xiaomi.com) or join our community:

-   [WeChat Group](https://work.weixin.qq.com/apph5/external_room/join/group_mng?plg_id=c417f99bd9014b5dd894daa8bfe19790&)
-   [Discord](https://discord.gg/WX2R2uNp)
-   [Telegram](https://t.me/+3T-I0pekOVIyNDBl)
-   [Reddit](https://www.reddit.com/r/XiaomiMiMo_Official/)

Downloads last month

4,004

Safetensors[](https://huggingface.co/docs/safetensors)

Model size

1T params

Tensor type

BF16

F32

F8\_E4M3

U8

Chat template

Files info

Inference Providers [NEW](https://huggingface.co/docs/inference-providers)

[

Text Generation

](https://huggingface.co/tasks/text-generation "Learn more about text-generation")

This model isn't deployed by any Inference Provider. [🙋 5 Ask for provider support](https://huggingface.co/spaces/huggingface/InferenceSupport/discussions/12653)

Model tree for XiaomiMiMo/MiMo-V2.6-Pro-MOPD[](https://huggingface.co/docs/hub/model-cards#specifying-a-base-model)
-------------------------------------------------------------------------------------------------------------------

Finetunes

[1 model](https://huggingface.co/models?other=base_model:finetune:XiaomiMiMo/MiMo-V2.6-Pro-MOPD)

Quantizations

[](https://huggingface.co/models?apps=llama.cpp&other=base_model:quantized:XiaomiMiMo/MiMo-V2.6-Pro-MOPD "Use with llama.cpp")[](https://huggingface.co/models?apps=lmstudio&other=base_model:quantized:XiaomiMiMo/MiMo-V2.6-Pro-MOPD "Use with LM Studio")[](https://huggingface.co/models?apps=jan&other=base_model:quantized:XiaomiMiMo/MiMo-V2.6-Pro-MOPD "Use with Jan")[](https://huggingface.co/models?apps=ollama&other=base_model:quantized:XiaomiMiMo/MiMo-V2.6-Pro-MOPD "Use with Ollama")

[7 models](https://huggingface.co/models?other=base_model:quantized:XiaomiMiMo/MiMo-V2.6-Pro-MOPD)

Spaces using XiaomiMiMo/MiMo-V2.6-Pro-MOPD 2
--------------------------------------------

[

📈

tardellirs/model-pulse

](https://huggingface.co/spaces/tardellirs/model-pulse)[

⚡

fastgpu/vram-and-gpu-cost-calculator

](https://huggingface.co/spaces/fastgpu/vram-and-gpu-cost-calculator)

Collection including XiaomiMiMo/MiMo-V2.6-Pro-MOPD
--------------------------------------------------

[

#### MiMo-V2.6

Collection

5 items • Updated 13 days ago • 128

](https://huggingface.co/collections/XiaomiMiMo/mimo-v26)

System theme

Company

[TOS](https://huggingface.co/terms-of-service) [Privacy](https://huggingface.co/privacy) [About](https://huggingface.co/huggingface) [Careers](https://apply.workable.com/huggingface/)[](https://huggingface.co/)

Website

[Models](https://huggingface.co/models) [Datasets](https://huggingface.co/datasets) [Spaces](https://huggingface.co/spaces) [Pricing](https://huggingface.co/pricing) [Docs](https://huggingface.co/docs)
