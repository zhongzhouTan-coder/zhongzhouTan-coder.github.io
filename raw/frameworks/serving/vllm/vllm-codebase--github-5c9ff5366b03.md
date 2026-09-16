---
kind: repository-source
provider: github
clone_url: https://github.com/vllm-project/vllm
repository_url: https://github.com/vllm-project/vllm
local_checkout: external-repos/vllm-5c9ff5366b03/
commit: 5c9ff5366b039a69b344773bdfead8466ed9a097
ref: detached
inspected: 2026-08-18
checkout_state: clean
---

# vllm Codebase Source Record

## Reading Scope

- GLM-5.2 OpenAI request round trip through V1 scheduling, GPU model runner, sparse MLA indexer/attention, MoE, sampling, and response emission

## Important Entry Files

- `vllm/entrypoints/openai/chat_completion/serving.py` — OpenAI chat admission and streaming response assembly
- `vllm/v1/engine/async_llm.py` — Frontend request admission and asynchronous output path
- `vllm/v1/engine/core.py` — Engine scheduling and execution loop
- `vllm/v1/core/sched/scheduler.py` — Token-budget scheduling and KV-cache allocation
- `vllm/v1/worker/gpu_model_runner.py` — GPU batch preparation, model execution, and sampling
- `vllm/model_executor/models/registry.py` — GLM architecture registration
- `vllm/model_executor/models/deepseek_v2.py` — GLM model shell, sparse MLA layers, and MoE construction
- `vllm/v1/attention/backends/mla/indexer.py` — Sparse indexer interface and backend execution
- `vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py` — FlashInfer sparse MLA backend
- `vllm/v1/attention/backends/mla/flashattn_mla_sparse.py` — FlashAttention sparse MLA backend
- `vllm/platforms/cuda.py` — CUDA backend selection
- `vllm/v1/engine/output_processor.py` — Token detokenization and request completion

## Limitations

- Static code reading only; runtime behavior was not executed.
