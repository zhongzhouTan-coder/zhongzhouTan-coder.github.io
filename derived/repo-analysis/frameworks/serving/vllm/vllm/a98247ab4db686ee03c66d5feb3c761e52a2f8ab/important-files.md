---
kind: repository-analysis
repository_id: github:vllm-project/vllm@a98247ab4db686ee03c66d5feb3c761e52a2f8ab
commit: a98247ab4db686ee03c66d5feb3c761e52a2f8ab
source_record: raw/frameworks/serving/vllm/vllm-codebase--github-a98247ab4db6.md
generated: 2026-10-26
---

# upstream vLLM: Qwen3.8 A3 Code Evidence

## Reader Contract

- **Audience:** inference engineers familiar with vLLM scheduling and paged caches.
- **Question:** how does one A3 text decode row reach its sampled token, and which component owns each state?
- **Mental model:** the runner supplies row and cache addresses, the hybrid decoder updates GDN or QSA state, and the runner samples the resulting hidden row.
- **Load time:** the wrapper, quantization methods, cache specs, and backend are selected.
- **Runtime:** prepared rows update cache owners, sparse attention returns a vector, and sampling returns request token IDs.
- **Boundary:** static source reading at the commit above; no NPU execution or exact physical memory measurement.

## Representation Plan

| Reader question | Evidence | Representation | Teaching job |
| --- | --- | --- | --- |
| Which actor holds each persistent state? | Cache specs and slab owner code | State table | Separate index keys from main K/V and recurrent state. |
| How does one row travel to output? | Runner, model, indexer, GBSA, sampler | Sequence and numbered trace | Make writes, reads, and return values explicit. |
| Which numbers are logical versus allocated? | Model config, cache specs, planner | Worked calculation and table | Avoid equating token records with NPU pool reservation. |

## Evidence Map

### Required Code Evidence

| Docs page | Finding | File | Symbol or transition | Start | End | Flow position |
| --- | --- | --- | --- | --- | --- | --- |
| docs/frameworks/serving/vllm-ascend/qwen3.8-flash-next-a3-inference-path.md | a3-001 | vllm/models/qwen4_exp/amd/model.py | decoder branch | 209 | 235 | Mental model |
| docs/frameworks/serving/vllm-ascend/qwen3.8-flash-next-a3-inference-path.md | a3-002 | vllm/models/qwen4_exp/common/qsa_cache.py | side-cache metadata builder | 599 | 725 | Resolution and state map |
| docs/frameworks/serving/vllm-ascend/qwen3.8-flash-next-a3-inference-path.md | a3-003 | vllm/models/qwen4_exp/common/qsa_cache.py | raw ring specification | 844 | 862 | Resolution and state map |
| docs/frameworks/serving/vllm-ascend/qwen3.8-flash-next-a3-inference-path.md | a3-004 | vllm/models/qwen4_exp/common/qsa_cache.py | compressed-key specification | 865 | 876 | Resolution and state map |
| docs/frameworks/serving/vllm-ascend/qwen3.8-flash-next-a3-inference-path.md | a3-005 | vllm/models/qwen4_exp/amd/qsa.py | main QSA specification | 323 | 331 | Resolution and state map |
| docs/frameworks/serving/vllm-ascend/qwen3.8-flash-next-a3-inference-path.md | a3-006 | vllm/models/qwen4_exp/common/qsa_cache.py | compressed-slot mapping | 161 | 189 | Resolution and state map |
| docs/frameworks/serving/vllm-ascend/qwen3.8-flash-next-a3-inference-path.md | a3-007 | vllm/models/qwen4_exp/amd/model.py | upstream conditional-generation model | 843 | 906 | Load-time path and backend contract |
| docs/frameworks/serving/vllm-ascend/qwen3.8-flash-next-a3-inference-path.md | a3-008 | vllm/models/qwen4_exp/amd/model.py | decoder constructor | 175 | 256 | Load-time path and backend contract |
| docs/frameworks/serving/vllm-ascend/qwen3.8-flash-next-a3-inference-path.md | a3-009 | vllm/models/qwen4_exp/amd/model.py | model loop | 378 | 404 | Load-time path and backend contract |
| docs/frameworks/serving/vllm-ascend/qwen3.8-flash-next-a3-inference-path.md | a3-010 | vllm/models/qwen4_exp/amd/qsa.py | requires BF16 model and cache storage, causal attention, and no context parallelism | 185 | 220 | Load-time path and backend contract |
| docs/frameworks/serving/vllm-ascend/qwen3.8-flash-next-a3-inference-path.md | a3-011 | vllm/models/qwen4_exp/amd/qsa.py | QSA owner | 185 | 221 | Cache bytes for one target token |
| docs/frameworks/serving/vllm-ascend/qwen3.8-flash-next-a3-inference-path.md | a3-012 | vllm/models/qwen4_exp/amd/indexer_qsa.py | indexer construction | 108 | 151 | Cache bytes for one target token |
| docs/frameworks/serving/vllm-ascend/qwen3.8-flash-next-a3-inference-path.md | a3-013 | vllm/models/qwen4_exp/common/qsa_cache.py | compressed-key spec | 865 | 878 | Cache bytes for one target token |
| docs/frameworks/serving/vllm-ascend/qwen3.8-flash-next-a3-inference-path.md | a3-014 | vllm/models/qwen4_exp/amd/model.py | upstream backbone | 456 | 494 | 3. The decoder chooses GDN or QSA |
| docs/frameworks/serving/vllm-ascend/qwen3.8-flash-next-a3-inference-path.md | a3-015 | vllm/models/qwen4_exp/amd/model.py | pending residual state is combined before PLE adds to the streams | 269 | 286 | 3. The decoder chooses GDN or QSA |
| docs/frameworks/serving/vllm-ascend/qwen3.8-flash-next-a3-inference-path.md | a3-016 | vllm/models/qwen4_exp/amd/model.py | GDN recurrent attention or QSA self-attention | 288 | 304 | 3. The decoder chooses GDN or QSA |
| docs/frameworks/serving/vllm-ascend/qwen3.8-flash-next-a3-inference-path.md | a3-017 | vllm/models/qwen4_exp/amd/model.py | routed MoE block | 306 | 311 | 3. The decoder chooses GDN or QSA |
| docs/frameworks/serving/vllm-ascend/qwen3.8-flash-next-a3-inference-path.md | a3-018 | vllm/models/qwen4_exp/amd/qsa.py | output gate and row-parallel output projection | 407 | 471 | 6. GBSA reads main K/V and returns a vector |
| docs/frameworks/serving/vllm-ascend/qwen3.8-flash-next-a3-inference-path.md | a3-019 | vllm/models/qwen4_exp/amd/model.py | final hyper-connection mixer | 519 | 541 | 7. Hidden state becomes logits |
| docs/frameworks/serving/vllm-ascend/qwen3.8-flash-next-a3-inference-path.md | a3-020 | vllm/models/qwen4_exp/amd/model.py | causal wrapper | 642 | 656 | 7. Hidden state becomes logits |
| docs/frameworks/serving/vllm-ascend/qwen3.8-flash-next-a3-inference-path.md | a3-021 | vllm/models/qwen4_exp/amd/model.py | compute_logits() | 784 | 788 | 7. Hidden state becomes logits |
| docs/frameworks/serving/vllm-ascend/qwen3.8-flash-next-a3-inference-path.md | a3-022 | vllm/models/qwen4_exp/common/qsa_cache.py | capacity rule | 844 | 862 | Branches and limits |
| docs/frameworks/serving/vllm-ascend/qwen3.8-flash-next-a3-inference-path.md | a3-023 | vllm/models/qwen4_exp/amd/model.py | construction | 843 | 906 | Branches and limits |
| docs/frameworks/serving/vllm-ascend/qwen3.8-flash-next-a3-inference-path.md | a3-024 | vllm/models/qwen4_exp/common/qsa_cache.py | QSA metadata builder | 689 | 724 | 1. The runner accepts a scheduled row |

## Reproduction Commands

- `git -C external-repos/vllm-qwen38-upstream rev-parse HEAD`
- `rg -n '^class |^def |^    def ' external-repos/vllm-qwen38-upstream/vllm/models/qwen4_exp`

## Scope and Verification

- 24 distinct revision-aware code anchors are used by the consuming page.
- Source inspection only. PR-reported A3 results predate the latest rebase; this checkout was not executed on an NPU.
