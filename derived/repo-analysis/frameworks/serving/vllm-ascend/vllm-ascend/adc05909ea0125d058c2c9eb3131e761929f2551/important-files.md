---
kind: repository-analysis
repository_id: github:shaopeng-666/vllm-ascend@adc05909ea0125d058c2c9eb3131e761929f2551
commit: adc05909ea0125d058c2c9eb3131e761929f2551
source_record: raw/frameworks/serving/vllm-ascend/vllm-ascend-codebase--github-adc05909ea01.md
generated: 2026-10-26
---

# vLLM Ascend PR: Qwen3.8 A3 Code Evidence

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
| docs/frameworks/serving/vllm-ascend/qwen3.8-flash-next-a3-inference-path.md | a3-001 | vllm_ascend/models/qwen4_exp/lightning_indexer.py | Lightning adapter | 97 | 108 | Resolution and state map |
| docs/frameworks/serving/vllm-ascend/qwen3.8-flash-next-a3-inference-path.md | a3-002 | vllm_ascend/models/qwen4_exp/gbsa.py | rejects group IDs outside the visible completed groups | 176 | 194 | Resolution and state map |
| docs/frameworks/serving/vllm-ascend/qwen3.8-flash-next-a3-inference-path.md | a3-003 | vllm_ascend/models/__init__.py | registers the causal, multimodal, and MTP architecture names | 4 | 16 | Load-time path and backend contract |
| docs/frameworks/serving/vllm-ascend/qwen3.8-flash-next-a3-inference-path.md | a3-004 | vllm_ascend/models/qwen4_exp/model.py | model wrapper | 20 | 75 | Load-time path and backend contract |
| docs/frameworks/serving/vllm-ascend/qwen3.8-flash-next-a3-inference-path.md | a3-005 | vllm_ascend/quantization/configs/modelslim_config.py | quantization method selector | 735 | 842 | Load-time path and backend contract |
| docs/frameworks/serving/vllm-ascend/qwen3.8-flash-next-a3-inference-path.md | a3-006 | vllm_ascend/quantization/configs/modelslim_config.py | maps the MTP layer's runtime index to checkpoint-local layer zero | 665 | 673 | Load-time path and backend contract |
| docs/frameworks/serving/vllm-ascend/qwen3.8-flash-next-a3-inference-path.md | a3-007 | vllm_ascend/models/qwen4_exp/qsa.py | QSA adapter | 482 | 507 | Load-time path and backend contract |
| docs/frameworks/serving/vllm-ascend/qwen3.8-flash-next-a3-inference-path.md | a3-008 | vllm_ascend/models/qwen4_exp/qsa.py | QSA backend | 448 | 479 | Load-time path and backend contract |
| docs/frameworks/serving/vllm-ascend/qwen3.8-flash-next-a3-inference-path.md | a3-009 | vllm_ascend/worker/v1/model_runner.py | runner's six-region binding | 6170 | 6209 | Load-time path and backend contract |
| docs/frameworks/serving/vllm-ascend/qwen3.8-flash-next-a3-inference-path.md | a3-010 | vllm_ascend/core/six_region_kv_cache_layout.py | six-region planner | 96 | 114 | How the PR allocates physical pages |
| docs/frameworks/serving/vllm-ascend/qwen3.8-flash-next-a3-inference-path.md | a3-011 | vllm_ascend/core/six_region_kv_cache_layout.py | requires one main, one raw, and one compressed owner for each QSA source | 186 | 210 | How the PR allocates physical pages |
| docs/frameworks/serving/vllm-ascend/qwen3.8-flash-next-a3-inference-path.md | a3-012 | vllm_ascend/core/six_region_kv_cache_layout.py | region construction | 267 | 327 | How the PR allocates physical pages |
| docs/frameworks/serving/vllm-ascend/qwen3.8-flash-next-a3-inference-path.md | a3-013 | vllm_ascend/worker/v1/model_runner.py | runner | 5613 | 5652 | How the PR allocates physical pages |
| docs/frameworks/serving/vllm-ascend/qwen3.8-flash-next-a3-inference-path.md | a3-014 | vllm_ascend/worker/v1/model_runner.py | main K/V views | 6170 | 6209 | How the PR allocates physical pages |
| docs/frameworks/serving/vllm-ascend/qwen3.8-flash-next-a3-inference-path.md | a3-015 | vllm_ascend/worker/v1/model_runner.py | side or recurrent-state views | 6211 | 6287 | How the PR allocates physical pages |
| docs/frameworks/serving/vllm-ascend/qwen3.8-flash-next-a3-inference-path.md | a3-016 | vllm_ascend/worker/v1/model_runner.py | execute_model() | 2416 | 2487 | 1. The runner accepts a scheduled row |
| docs/frameworks/serving/vllm-ascend/qwen3.8-flash-next-a3-inference-path.md | a3-017 | vllm_ascend/worker/v1/model_runner.py | _prepare_inputs() | 2522 | 2540 | 1. The runner accepts a scheduled row |
| docs/frameworks/serving/vllm-ascend/qwen3.8-flash-next-a3-inference-path.md | a3-018 | vllm_ascend/models/qwen4_exp/input_adapter.py | input adapter is constructed only when ple_layer_ids exists | 177 | 192 | 2. The runner prepares PLE and attention metadata |
| docs/frameworks/serving/vllm-ascend/qwen3.8-flash-next-a3-inference-path.md | a3-019 | vllm_ascend/models/qwen4_exp/input_adapter.py | prepare() | 118 | 174 | 2. The runner prepares PLE and attention metadata |
| docs/frameworks/serving/vllm-ascend/qwen3.8-flash-next-a3-inference-path.md | a3-020 | vllm_ascend/worker/v1/model_runner.py | runner adds both to model_kwargs | 3655 | 3681 | 2. The runner prepares PLE and attention metadata |
| docs/frameworks/serving/vllm-ascend/qwen3.8-flash-next-a3-inference-path.md | a3-021 | vllm_ascend/worker/v1/model_runner.py | forward context | 2692 | 2708 | 2. The runner prepares PLE and attention metadata |
| docs/frameworks/serving/vllm-ascend/qwen3.8-flash-next-a3-inference-path.md | a3-022 | vllm_ascend/models/qwen4_exp/qsa.py | Ascend QSA owner | 578 | 646 | 4. QSA creates and stores its index state |
| docs/frameworks/serving/vllm-ascend/qwen3.8-flash-next-a3-inference-path.md | a3-023 | vllm_ascend/models/qwen4_exp/qsa.py | indexer projection | 292 | 309 | 4. QSA creates and stores its index state |
| docs/frameworks/serving/vllm-ascend/qwen3.8-flash-next-a3-inference-path.md | a3-024 | vllm_ascend/models/qwen4_exp/qsa.py | compression transaction | 311 | 351 | 4. QSA creates and stores its index state |
| docs/frameworks/serving/vllm-ascend/qwen3.8-flash-next-a3-inference-path.md | a3-025 | vllm_ascend/models/qwen4_exp/qsa.py | indexer call | 353 | 370 | 5. Lightning Indexer selects groups |
| docs/frameworks/serving/vllm-ascend/qwen3.8-flash-next-a3-inference-path.md | a3-026 | vllm_ascend/models/qwen4_exp/lightning_indexer.py | CANN Lightning Indexer adapter | 64 | 108 | 5. Lightning Indexer selects groups |
| docs/frameworks/serving/vllm-ascend/qwen3.8-flash-next-a3-inference-path.md | a3-027 | vllm_ascend/models/qwen4_exp/lightning_indexer.py | npu_lightning_indexer | 146 | 170 | 5. Lightning Indexer selects groups |
| docs/frameworks/serving/vllm-ascend/qwen3.8-flash-next-a3-inference-path.md | a3-028 | vllm_ascend/models/qwen4_exp/qsa.py | writes K and V to the main cache before attention reads it | 619 | 646 | 6. GBSA reads main K/V and returns a vector |
| docs/frameworks/serving/vllm-ascend/qwen3.8-flash-next-a3-inference-path.md | a3-029 | vllm_ascend/models/qwen4_exp/qsa.py | attention implementation | 393 | 445 | 6. GBSA reads main K/V and returns a vector |
| docs/frameworks/serving/vllm-ascend/qwen3.8-flash-next-a3-inference-path.md | a3-030 | vllm_ascend/models/qwen4_exp/gbsa.py | GBSA adapter | 124 | 153 | 6. GBSA reads main K/V and returns a vector |
| docs/frameworks/serving/vllm-ascend/qwen3.8-flash-next-a3-inference-path.md | a3-031 | vllm_ascend/models/qwen4_exp/gbsa.py | filters invalid or future group IDs, computes per-row sparse counts | 176 | 194 | 6. GBSA reads main K/V and returns a vector |
| docs/frameworks/serving/vllm-ascend/qwen3.8-flash-next-a3-inference-path.md | a3-032 | vllm_ascend/models/qwen4_exp/gbsa.py | generic_block_sparse_attention | 198 | 234 | 6. GBSA reads main K/V and returns a vector |
| docs/frameworks/serving/vllm-ascend/qwen3.8-flash-next-a3-inference-path.md | a3-033 | vllm_ascend/worker/v1/model_runner.py | selects the logits rows | 2791 | 2816 | 7. Hidden state becomes logits |
| docs/frameworks/serving/vllm-ascend/qwen3.8-flash-next-a3-inference-path.md | a3-034 | vllm_ascend/worker/v1/model_runner.py | execute_model() returns None | 2839 | 2859 | 7. Hidden state becomes logits |
| docs/frameworks/serving/vllm-ascend/qwen3.8-flash-next-a3-inference-path.md | a3-035 | vllm_ascend/worker/v1/model_runner.py | sample_tokens() | 2945 | 2994 | 8. Sampling returns per-request token IDs |
| docs/frameworks/serving/vllm-ascend/qwen3.8-flash-next-a3-inference-path.md | a3-036 | vllm_ascend/worker/v1/model_runner.py | _sample() | 3135 | 3157 | 8. Sampling returns per-request token IDs |
| docs/frameworks/serving/vllm-ascend/qwen3.8-flash-next-a3-inference-path.md | a3-037 | vllm_ascend/worker/v1/model_runner.py | Bookkeeping | 3213 | 3237 | 8. Sampling returns per-request token IDs |
| docs/frameworks/serving/vllm-ascend/qwen3.8-flash-next-a3-inference-path.md | a3-038 | vllm_ascend/worker/v1/model_runner.py | constructs and returns ModelRunnerOutput | 3090 | 3117 | 8. Sampling returns per-request token IDs |
| docs/frameworks/serving/vllm-ascend/qwen3.8-flash-next-a3-inference-path.md | a3-039 | vllm_ascend/models/__init__.py | registers an Ascend MTP wrapper | 12 | 16 | Branches and limits |
| docs/frameworks/serving/vllm-ascend/qwen3.8-flash-next-a3-inference-path.md | a3-040 | vllm_ascend/models/qwen4_exp/mtp.py | predictor | 27 | 85 | Branches and limits |
| docs/frameworks/serving/vllm-ascend/qwen3.8-flash-next-a3-inference-path.md | a3-041 | vllm_ascend/worker/v1/model_runner.py | proposal handoff | 3005 | 3030 | Branches and limits |
| docs/frameworks/serving/vllm-ascend/qwen3.8-flash-next-a3-inference-path.md | a3-042 | vllm_ascend/worker/v1/model_runner.py | _sample() | 3135 | 3157 | Branches and limits |
| docs/frameworks/serving/vllm-ascend/qwen3.8-flash-next-a3-inference-path.md | a3-043 | vllm_ascend/models/qwen4_exp/input_adapter.py | PLE adapter | 138 | 174 | Branches and limits |
| docs/frameworks/serving/vllm-ascend/qwen3.8-flash-next-a3-inference-path.md | a3-044 | vllm_ascend/models/qwen4_exp/qsa.py | fused norm and M-RoPE path | 536 | 573 | Branches and limits |
| docs/frameworks/serving/vllm-ascend/qwen3.8-flash-next-a3-inference-path.md | a3-045 | vllm_ascend/worker/v1/model_runner.py | Mamba state copy | 2755 | 2786 | Branches and limits |

## Reproduction Commands

- `git -C external-repos/vllm-ascend-qwen38-pr rev-parse HEAD`
- `rg -n '^class |^def |^    def ' external-repos/vllm-ascend-qwen38-pr/vllm_ascend/models/qwen4_exp`

## Scope and Verification

- 45 distinct revision-aware code anchors are used by the consuming page.
- Source inspection only. PR-reported A3 results predate the latest rebase; this checkout was not executed on an NPU.
