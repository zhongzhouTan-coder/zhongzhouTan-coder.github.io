---
kind: repository-analysis
repository_id: github:vllm-project/vllm@a0c092ee72c0dcefbb3b3e74f97ac62d842e5f4b
commit: a0c092ee72c0dcefbb3b3e74f97ac62d842e5f4b
source_record: raw/frameworks/vllm-codebase--github-a0c092ee72c0.md
generated: 2026-09-08
---

# GLM-5.1 Expert-Parallel Path Evidence

Consuming page: `docs/frameworks/vllm/glm-5.1-expert-parallel/index.md`

## Reader Contract

- **Audience:** vLLM users and developers who understand a transformer MoE
  layer but want to follow expert parallelism through the upstream runner.
- **Question:** how does one GLM-5.1 routed token move from global router IDs
  to a local expert shard and back to the decoder?
- **Mental model:** GLM-5.1 uses a shared DeepSeek-V2-family model shell; EP
  makes expert ownership rank-local, while prepare/finalize owns movement and
  the expert kernel owns local MLP computation.
- **Load time:** registry resolution, EP group construction, parallel sizing,
  expert maps, weight allocation, and all-to-all backend selection.
- **Runtime:** router selection, optional activation quantization, dispatch,
  local expert GEMMs, top-k weighting, combine, and sequence-shape restoration.
- **Limits:** static reading only; no GLM-5.1 checkpoint, multi-rank process
  group, CUDA device, specialized EP library, or performance run was available.

## Representation Plan

| Reader question | Evidence | Representation | Teaching job |
|---|---|---|---|
| Which component owns each part of the routed token? | GLM model, FusedMoE runner, prepare/finalize, all2all manager | Editable Mermaid flow plus conceptual prose | Separate router, transport, local compute, and return ownership. |
| What changes when EP replaces MoE TP? | FusedMoEParallelConfig and EP group construction | Parallelism table | Make the flattened EP size and whole-expert ownership explicit. |
| What state changes during one token's trip? | Expert map, dispatch, local kernel, combine | Numbered trace and state table | Show global IDs, local IDs, rank-local batches, and source-token output order. |
| Why can two EP deployments behave differently? | Backend selector and communicator table | Backend comparison table | Separate the stable contract from transport-specific implementations. |

## Required Code Evidence

| Docs page | Finding | File | Symbol | Start | End |
|---|---|---|---|---:|---:|
| `docs/frameworks/vllm/glm-5.1-expert-parallel/index.md` | glm-support | `docs/models/supported_models.md` | GLM-5, GLM-5.1, GLM-5.2 supported-model row | 383 | — |
| `docs/frameworks/vllm/glm-5.1-expert-parallel/index.md` | model-registry | `vllm/model_executor/models/registry.py` | `GlmMoeDsaForCausalLM` registry entry | 117 | — |
| `docs/frameworks/vllm/glm-5.1-expert-parallel/index.md` | glm-alias | `vllm/model_executor/models/deepseek_v2.py` | `GlmMoeDsaForCausalLM` | 1938 | — |
| `docs/frameworks/vllm/glm-5.1-expert-parallel/index.md` | moe-model | `vllm/model_executor/models/deepseek_v2.py` | `DeepseekV2MoE` construction and forward | 279 | — |
| `docs/frameworks/vllm/glm-5.1-expert-parallel/index.md` | moe-fused-factory | `vllm/model_executor/models/deepseek_v2.py` | `FusedMoE` construction | 364 | — |
| `docs/frameworks/vllm/glm-5.1-expert-parallel/index.md` | decoder-layer | `vllm/model_executor/models/deepseek_v2.py` | `DeepseekV2DecoderLayer.forward` | 1290 | — |
| `docs/frameworks/vllm/glm-5.1-expert-parallel/index.md` | ep-group | `vllm/distributed/parallel_state.py` | EP group construction | 1919 | — |
| `docs/frameworks/vllm/glm-5.1-expert-parallel/index.md` | parallel-sizing | `vllm/model_executor/layers/fused_moe/config.py` | `FusedMoEParallelConfig.make` | 1124 | — |
| `docs/frameworks/vllm/glm-5.1-expert-parallel/index.md` | expert-map | `vllm/model_executor/layers/fused_moe/expert_map_manager.py` | `determine_expert_map` | 22 | — |
| `docs/frameworks/vllm/glm-5.1-expert-parallel/index.md` | ep-configuration | `vllm/config/parallel.py` | `enable_expert_parallel` and sequence/all-to-all properties | 165 | — |
| `docs/frameworks/vllm/glm-5.1-expert-parallel/index.md` | fused-moe-factory | `vllm/model_executor/layers/fused_moe/layer.py` | `FusedMoE` factory | 100 | — |
| `docs/frameworks/vllm/glm-5.1-expert-parallel/index.md` | runner-routing | `vllm/model_executor/layers/fused_moe/runner/moe_runner.py` | `_apply_quant_method` | 577 | — |
| `docs/frameworks/vllm/glm-5.1-expert-parallel/index.md` | runner-forward | `vllm/model_executor/layers/fused_moe/runner/moe_runner.py` | `_forward_impl` dispatch/compute/combine | 837 | — |
| `docs/frameworks/vllm/glm-5.1-expert-parallel/index.md` | modular-prepare | `vllm/model_executor/layers/fused_moe/modular_kernel.py` | `FusedMoEKernelModularImpl._prepare` | 1189 | — |
| `docs/frameworks/vllm/glm-5.1-expert-parallel/index.md` | modular-apply | `vllm/model_executor/layers/fused_moe/modular_kernel.py` | `FusedMoEKernelModularImpl.apply` | 1430 | — |
| `docs/frameworks/vllm/glm-5.1-expert-parallel/index.md` | routed-experts | `vllm/model_executor/layers/fused_moe/routed_experts.py` | `RoutedExperts.forward_modular` | 1181 | — |
| `docs/frameworks/vllm/glm-5.1-expert-parallel/index.md` | naive-prepare-finalize | `vllm/model_executor/layers/fused_moe/prepare_finalize/naive_dp_ep.py` | `MoEPrepareAndFinalizeNaiveDPEPModular` | 71 | — |
| `docs/frameworks/vllm/glm-5.1-expert-parallel/index.md` | default-dispatch | `vllm/distributed/device_communicators/all2all.py` | `AgRsAll2AllManager.dispatch` | 101 | — |
| `docs/frameworks/vllm/glm-5.1-expert-parallel/index.md` | default-combine | `vllm/distributed/device_communicators/all2all.py` | `AgRsAll2AllManager.combine` | 138 | — |
| `docs/frameworks/vllm/glm-5.1-expert-parallel/index.md` | backend-selector | `vllm/model_executor/layers/fused_moe/all2all_utils.py` | `maybe_make_prepare_finalize` | 118 | — |
| `docs/frameworks/vllm/glm-5.1-expert-parallel/index.md` | communicator-selector | `vllm/distributed/device_communicators/cuda_communicator.py` | all-to-all manager selection | 140 | — |
| `docs/frameworks/vllm/glm-5.1-expert-parallel/index.md` | parallel-test | `tests/kernels/moe/test_moe_layer.py` | parallel MoE test configuration | 1842 | — |

## Static Findings

- The supported-model table associates GLM-5.1 with `GlmMoeDsaForCausalLM`,
  and the registry maps that name to the shared `deepseek_v2` module. The GLM
  class is a zero-override subclass of `DeepseekV2ForCausalLM`.
- `FusedMoEParallelConfig.make()` treats EP as whole-expert ownership. When EP
  is enabled, the MoE-local tensor-parallel size becomes 1 and the EP size is
  the flattened active TP x PCP x DP size.
- `ExpertMapManager` creates a global-to-local expert map. Linear placement is
  the default; round-robin placement requires compatible grouped experts and a
  backend with the needed routing tables.
- Modular FusedMoE separates router selection, prepare/dispatch, local expert
  compute, and finalize/combine. The default all-gather/reduce-scatter manager
  gathers activation and route metadata and reduce-scatters the weighted output.
- The same prepare/finalize contract can be replaced by DeepEP, DeepEP v2,
  Mori, NIXL EP, or FlashInfer NVLink backends. Their token formats,
  dependencies, graph constraints, and output-reduction ownership differ.

## Runtime Flow Evidence

1. `DeepseekV2MoE.forward` produces router logits, optionally chunks
   sequence-parallel rows, invokes the FusedMoE runner, and gathers the result
   back when sequence parallelism is active.
2. `MoERunner._forward_impl` enters the sequence-parallel context, performs
   dispatch, calls routing/expert computation, and performs combine.
3. `FusedMoEKernelModularImpl._prepare` delegates quantization and dispatch;
   its `apply` method invokes local expert compute and `_finalize`.
4. The naive EP prepare/finalize implementation calls `get_ep_group().dispatch`
   and `get_ep_group().combine`; `AgRsAll2AllManager` implements those as
   all-gather and reduce-scatter operations.
5. The decoder layer receives the returned hidden rows and continues the
   residual stream.

## Verification Boundary

The inspected checkout is pinned at
`a0c092ee72c0dcefbb3b3e74f97ac62d842e5f4b`. Its only working-tree change is an
unrelated debug edit in `vllm/model_executor/layers/linear.py`; none of the
paths listed in the evidence table is modified. This note records static code
evidence only. The focused parallel MoE tests were identified but not run, and
no GLM-5.1 checkpoint, CUDA device, multi-rank collective, or specialized EP
library was available for runtime verification.
