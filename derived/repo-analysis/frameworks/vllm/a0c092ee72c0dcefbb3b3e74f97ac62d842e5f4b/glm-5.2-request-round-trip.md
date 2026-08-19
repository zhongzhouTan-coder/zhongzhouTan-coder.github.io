---
kind: repository-analysis
repository_id: github:vllm-project/vllm@a0c092ee72c0dcefbb3b3e74f97ac62d842e5f4b
commit: a0c092ee72c0dcefbb3b3e74f97ac62d842e5f4b
source_record: raw/frameworks/vllm-codebase--github-a0c092ee72c0.md
generated: 2026-08-18
---

# GLM-5.2 Request Round-Trip Evidence

Consuming page: `docs/frameworks/vllm-ascend/glm-5.2-inference-path.md`

This note records the upstream vLLM request, scheduler, model-shell, and return
path paired with the pinned vllm-ascend revision used by the consuming page.

## Required Code Evidence

| Docs page | Finding | File | Symbol | Start | End |
|---|---|---|---|---:|---:|
| `docs/frameworks/vllm-ascend/glm-5.2-inference-path.md` | api-admission | `vllm/entrypoints/openai/chat_completion/serving.py` | `OpenAIServingChat._create_chat_completion` | 255 | 407 |
| `docs/frameworks/vllm-ascend/glm-5.2-inference-path.md` | async-request-stream | `vllm/v1/engine/async_llm.py` | `AsyncLLM.generate` | 527 | 599 |
| `docs/frameworks/vllm-ascend/glm-5.2-inference-path.md` | engine-step | `vllm/v1/engine/core.py` | `EngineCore.step` | 584 | 614 |
| `docs/frameworks/vllm-ascend/glm-5.2-inference-path.md` | scheduler-budget | `vllm/v1/core/sched/scheduler.py` | `Scheduler.schedule` | 427 | 570 |
| `docs/frameworks/vllm-ascend/glm-5.2-inference-path.md` | model-registration | `vllm/model_executor/models/registry.py` | `GlmMoeDsaForCausalLM` registry entry | 117 | — |
| `docs/frameworks/vllm-ascend/glm-5.2-inference-path.md` | causal-lm-construction | `vllm/model_executor/models/deepseek_v2.py` | `DeepseekV2ForCausalLM.__init__` | 1801 | 1863 |
| `docs/frameworks/vllm-ascend/glm-5.2-inference-path.md` | forward-and-logits | `vllm/model_executor/models/deepseek_v2.py` | `DeepseekV2ForCausalLM.forward` and `compute_logits` | 1894 | 1911 |
| `docs/frameworks/vllm-ascend/glm-5.2-inference-path.md` | glm-alias | `vllm/model_executor/models/deepseek_v2.py` | `GlmMoeDsaForCausalLM` | 1938 | 1939 |
| `docs/frameworks/vllm-ascend/glm-5.2-inference-path.md` | decoder-stack | `vllm/model_executor/models/deepseek_v2.py` | `DeepseekV2Model.forward` | 1431 | 1519 |
| `docs/frameworks/vllm-ascend/glm-5.2-inference-path.md` | decoder-layer | `vllm/model_executor/models/deepseek_v2.py` | `DeepseekV2DecoderLayer.forward` | 1290 | 1361 |
| `docs/frameworks/vllm-ascend/glm-5.2-inference-path.md` | output-processing | `vllm/v1/engine/output_processor.py` | `OutputProcessor.process_outputs` | 589 | 711 |
| `docs/frameworks/vllm-ascend/glm-5.2-inference-path.md` | response-stream | `vllm/entrypoints/openai/chat_completion/serving.py` | `chat_completion_stream_generator` delta serialization | 573 | 753 |

## Verification Boundary

Static reading only. The upstream checkout is clean and pinned, but no HTTP,
model, scheduler, or device execution was run. The paired vllm-ascend source
record identifies this upstream revision as its model-substrate reference.
