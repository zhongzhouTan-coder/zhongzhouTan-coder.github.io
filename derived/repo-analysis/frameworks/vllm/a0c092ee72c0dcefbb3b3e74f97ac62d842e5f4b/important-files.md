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

## Qwen3.5 MTP and target verification extension

Consuming page: `docs/frameworks/vllm-ascend/qwen3.5-mtp.md`

| Docs page | Finding | File | Symbol | Start | End |
|---|---|---|---|---:|---:|
| `docs/frameworks/vllm-ascend/qwen3.5-mtp.md` | mtp-model | `vllm/model_executor/models/qwen3_5_mtp.py` | `Qwen3_5MultiTokenPredictor` | 64 | 189 |
| `docs/frameworks/vllm-ascend/qwen3.5-mtp.md` | mtp-head | `vllm/model_executor/models/qwen3_5_mtp.py` | `Qwen3_5MTP.compute_logits` | 212 | 299 |
| `docs/frameworks/vllm-ascend/qwen3.5-mtp.md` | spec-metadata | `vllm/v1/spec_decode/metadata.py` | `SpecDecodeMetadata` | 8 | 31 |
| `docs/frameworks/vllm-ascend/qwen3.5-mtp.md` | sampler-entry | `vllm/v1/worker/gpu_model_runner.py` | `GPUModelRunner._sample` | 3692 | 3719 |
| `docs/frameworks/vllm-ascend/qwen3.5-mtp.md` | rejection-forward | `vllm/v1/sample/rejection_sampler.py` | `RejectionSampler.forward` | 38 | 181 |
| `docs/frameworks/vllm-ascend/qwen3.5-mtp.md` | greedy-verify | `vllm/v1/sample/rejection_sampler.py` | `rejection_greedy_sample_kernel` | 715 | 769 |
| `docs/frameworks/vllm-ascend/qwen3.5-mtp.md` | random-verify | `vllm/v1/sample/rejection_sampler.py` | `rejection_random_sample_kernel` | 774 | 845 |

## Reproduction Commands

Record exact read-only search or counting commands here when the docs make
quantitative codebase claims.
