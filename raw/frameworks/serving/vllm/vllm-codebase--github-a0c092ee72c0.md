---
kind: repository-source
provider: github
clone_url: https://github.com/vllm-project/vllm.git
repository_url: https://github.com/vllm-project/vllm
local_checkout: external-repos/vllm/
commit: a0c092ee72c0dcefbb3b3e74f97ac62d842e5f4b
ref: main
inspected: 2026-07-29
checkout_state: clean
---

# vllm Codebase Source Record

## Reading Scope

- Kimi K3 model, attention, MoE, parser, processor, and speculative decoding implementation

## Important Entry Files

- `vllm/models/kimi_k3/__init__.py` — Hardware-isolated Kimi K3 model entrypoint
- `vllm/models/kimi_k3/nvidia/model.py` — NVIDIA Kimi K3 multimodal model, decoder, attention, MoE, and weight-loading path
- `vllm/models/kimi_k3/nvidia/kda.py` — Kimi K3 Delta Attention implementation
- `vllm/models/kimi_k3/nvidia/mla.py` — Kimi K3 Multi-head Latent Attention implementation
- `vllm/model_executor/layers/fused_moe/runner/latent_moe_runner.py` — Generic latent-MoE runner used by Kimi K3 routed experts
- `vllm/models/kimi_k3/nvidia/ops/latent_moe_tail.py` — Optional Kimi K3 latent-MoE tail fusion
- `vllm/models/kimi_k3/nvidia/mtp.py` — Kimi K3 MTP draft model
- `vllm/parser/kimi_k3.py` — Kimi K3 XTML parser composition

## Limitations

- Static code reading only; runtime behavior was not executed.
