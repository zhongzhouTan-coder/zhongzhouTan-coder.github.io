---
kind: repository-analysis
repository_id: github:vllm-project/vllm@a0c092ee72c0dcefbb3b3e74f97ac62d842e5f4b
commit: a0c092ee72c0dcefbb3b3e74f97ac62d842e5f4b
source_record: raw/frameworks/vllm-codebase--github-a0c092ee72c0.md
generated: 2026-07-29
---

# vllm Codebase Important Files

## Evidence Map

- `vllm/models/kimi_k3/__init__.py` — Hardware-isolated Kimi K3 model entrypoint
- `vllm/models/kimi_k3/nvidia/model.py` — NVIDIA Kimi K3 multimodal model, decoder, attention, MoE, and weight-loading path
- `vllm/models/kimi_k3/nvidia/kda.py` — Kimi K3 Delta Attention implementation
- `vllm/models/kimi_k3/nvidia/mla.py` — Kimi K3 Multi-head Latent Attention implementation
- `vllm/model_executor/layers/fused_moe/runner/latent_moe_runner.py` — Generic latent-MoE runner used by Kimi K3 routed experts
- `vllm/models/kimi_k3/nvidia/ops/latent_moe_tail.py` — Optional Kimi K3 latent-MoE tail fusion
- `vllm/models/kimi_k3/nvidia/mtp.py` — Kimi K3 MTP draft model
- `vllm/parser/kimi_k3.py` — Kimi K3 XTML parser composition

## Reproduction Commands

Record exact read-only search or counting commands here when the docs make
quantitative codebase claims.
