---
kind: repository-source
provider: github
clone_url: https://github.com/vllm-project/vllm
repository_url: https://github.com/vllm-project/vllm
local_checkout: external-repos/vllm-fce64bc5c2c9/
commit: fce64bc5c2c939a4521bfb9b462c8c74c1c13893
ref: detached
inspected: 2026-10-20
checkout_state: clean
---

# vllm Codebase Source Record

## Reading Scope

- Qwen3.8-Flash-Next Qwen4Exp NVIDIA registration, model/state path, sparse-attention and embedding behavior, and GPU batch-to-sampled-token boundary; static reading only

## Important Entry Files

- `vllm/model_executor/models/registry.py` — Maps the checkpoint architecture name to Qwen4Exp
- `vllm/model_executor/models/config.py` — Applies Qwen4Exp config validation and runtime gates
- `vllm/models/qwen4_exp/nvidia/model.py` — Builds and runs the NVIDIA Qwen4Exp model
- `vllm/models/qwen4_exp/nvidia/model_state.py` — Owns request-scoped recurrent and attention state
- `vllm/models/qwen4_exp/nvidia/qsa.py` — Implements the NVIDIA Qwen sparse-attention path
- `vllm/models/qwen4_exp/nvidia/indexer_qsa.py` — Selects context blocks for QSA
- `vllm/models/qwen4_exp/nvidia/ngram_embedding.py` — Implements the NVIDIA n-gram embedding path
- `vllm/models/qwen4_exp/nvidia/hyperconnection.py` — Handles the model's gated residual stream
- `vllm/v1/worker/gpu_model_runner.py` — Consumes scheduled GPU batches and returns sampled token IDs
- `vllm/v1/kv_cache_interface.py` — Defines KV and recurrent cache specifications

## Limitations

- Static code reading only; runtime behavior was not executed.
