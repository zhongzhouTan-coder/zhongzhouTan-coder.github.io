---
kind: repository-analysis
repository_id: github:vllm-project/vllm-ascend@51b0242c0d3be1d3bd59e763a7d78b8553cd45e1
commit: 51b0242c0d3be1d3bd59e763a7d78b8553cd45e1
source_record: raw/frameworks/serving/vllm-ascend/vllm-ascend-codebase--github-51b0242c0d3b.md
generated: 2026-10-10
---

# vllm-ascend Codebase Important Files

## Reader Contract

- **Audience:** serving and accelerator engineers mapping the same five checkpoints to Ascend NPUs.
- **Question:** which model overrides and cache owners are established by this pinned Ascend revision?
- **Mental model:** a plugin may replace a model shell, cache owner, or attention backend while preserving the checkpoint's logical state; the physical NPU page contract is backend-specific.
- **Offline/load time:** registry entries select overrides and configuration creates distinct cache specifications.
- **Runtime:** NPU kernels read indexed pages or recurrent state and return attended hidden states to the decoder.
- **Limits:** this is static reading. Qwen4Exp has no dedicated registration or package in this checkout; live support and NPU tensor layout must be validated against a matching release.

## Representation Plan

| Reader question | Evidence | Representation | Teaching job |
|---|---|---|---|
| Which model path is actually registered? | `vllm_ascend/models/__init__.py` | Per-model evidence rows | Separate supported overrides from guide-level claims. |
| Which cache classes need separate measurement? | Model cache owners and attention implementations | Per-model state tables | Guide NPU allocation inventory. |
| What is known only statically? | Pinned source and absent hardware run | Verification limits | Avoid claiming measured stride or throughput. |

## Evidence Map

- `vllm_ascend/models/deepseek_mtp.py` — GLM-5.3 family registration
- `vllm_ascend/models/deepseek_v41/model.py` — DeepSeek-V4.1 cache path
- `vllm_ascend/models/minimax_m3/minimax_m3.py` — MiniMax-M3 sparse attention
- `vllm_ascend/models/kimi_k3.py` — Kimi-K3 NPU override

## Reproduction Commands

The negative Qwen4Exp finding is scoped to the pinned model registry and model
tree. Recheck it with `rg -n 'Qwen4Exp|qwen4_exp' external-repos/vllm-ascend-51b0242c0d3b/vllm_ascend/models`
and `rg --files external-repos/vllm-ascend-51b0242c0d3b/vllm_ascend/models | rg 'qwen4_exp|qwen4exp'`.

## Scope and verification

This pinned revision records the five-checkpoint code reading. The findings above are from static source reading; no model, GPU, or Ascend NPU was executed. Checkpoint dimensions require confirmation against released deployment configs.

## Model-specific findings

### GLM-5.3

- `vllm_ascend/models/__init__.py:74` registers a GLM sparse-MLA family override in `deepseek_mtp.py:74`. This is the relevant family path for the selected checkpoint if its architecture resolves to `GlmMoeDsaForCausalLM`.
- `vllm_ascend/models/glm5next/` is a separate GLM-5.3-Flash package with KDA and kpool. It must not be used to infer the selected GLM-5.3 checkpoint's cache payload.

### DeepSeek-V4.1-Flash

- `vllm_ascend/models/__init__.py:38` registers the V4.1 model. `deepseek_v41/model.py:491` coordinates shared attention state; `:565` holds local SWA state and `:675` implements the attention owner.
- The Ascend implementation has NPU-specific compressor/indexer modules in `vllm_ascend/models/deepseek_v41/`; GPU packed-record byte counts cannot be transferred without inspecting NPU tensor descriptors.

### MiniMax-M3

- `vllm_ascend/models/__init__.py:42` registers the MiniMax model. `minimax_m3/minimax_m3.py:198` owns main sparse attention and `minimax_m3/msa_m3.py:202` owns indexer state. `msa_m3.py:930` executes the sparse read path.
- A selected block ID is an address for a current read, not a retained-cache deletion. Main and side-cache capacity remain separate concerns.

### Qwen3.8-Flash-Next

- Scoped search command: `rg -n 'qwen4_exp|Qwen4Exp|Qwen3.8-Flash' external-repos/vllm-ascend-51b0242c0d3b/vllm_ascend/models/__init__.py external-repos/vllm-ascend-51b0242c0d3b/vllm_ascend`. It returned no matches.
- The absence of a dedicated model package or registration in this revision does not prove that no other release supports the model, or that upstream fallback cannot execute any subset. It means this pinned checkout does not establish checkpoint-specific NPU behavior.

### Kimi-K3

- `vllm_ascend/models/__init__.py:9` registers Kimi text and multimodal names. `kimi_k3.py:1107` specializes the upstream shell, while `:389` extends the upstream MLA class specifically for DSpark RoPE metadata. The class does not itself define physical cache packing.
- The static override shows a distinct NPU execution path, but device generation, dtype, graph mode, cache stride, and prefix-hit correctness require a live stack.
