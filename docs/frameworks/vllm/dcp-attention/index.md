---
title: "vLLM DCP Attention: From Local LSE to Exact Global Output"
summary: "A code-level derivation of how vLLM shards KV context across DCP ranks, rescales local attention outputs with gathered LSE values, and preserves the no-DCP attention result."
layout: default
confidence: medium
code_links: strict
code_evidence: strict
sources:
  - raw/frameworks/vllm-codebase--github-a0c092ee72c0.md
  - derived/repo-analysis/frameworks/vllm/a0c092ee72c0dcefbb3b3e74f97ac62d842e5f4b/important-files.md
  - derived/repo-analysis/frameworks/vllm/a0c092ee72c0dcefbb3b3e74f97ac62d842e5f4b/context-parallelism.md
updated: 2026-08-14
---

# vLLM DCP Attention: From Local LSE to Exact Global Output

**Repository:** [vllm-project/vllm](https://github.com/vllm-project/vllm)
**Inspected commit:** `a0c092ee72c0dcefbb3b3e74f97ac62d842e5f4b`
**Checkout state:** clean, static reading on 2026-08-11

**Related pages:** [vLLM DCP and PCP overview](../vllm-context-parallelism.md),
[Context Parallelism](../../../terms/context-parallelism.md),
[All-Gather](../../../terms/all-gather.md),
[All-Reduce](../../../terms/all-reduce.md),
[All-to-All](../../../terms/all-to-all.md)

## TL;DR

**What:** DCP splits the persistent KV context across ranks while each rank uses the same query to compute attention over its local KV shard.

**How:** vLLM all-gathers per-rank log-sum-exp values, computes the global LSE, scales each local output by `exp(local_lse - global_lse)`, and then reduces the corrected outputs.

**The number:** The result is exactly the same as ordinary attention over the union of all KV shards; `AG+RS` and `AG+AR` differ only in where the final heads are stored.

## The Big Picture

```text
DCP rank i:
    replicated query Q + local KV shard (K_i, V_i)
        -> local attention
        -> local output o_i [B, H, D] and local LSE L_i [B, H]

DCP group:
    all-gather [L_0, L_1, ..., L_(N-1)]
        -> stable global LSE L
        -> local correction o_i * exp(L_i - L)
        -> AG+RS, AG+AR, or A2A reduction
        -> exact global attention output
```

*Code-path summary. The kernel computes one rank's corrected contribution; the following collective performs the cross-rank sum.*

## Why This Exists

Without DCP, one rank must read every key and value in a long request's [KV cache](../../../terms/kv-cache.md). That makes decode attention increasingly expensive and forces every rank to store the same persistent context. DCP lets rank `i` read only its KV shard, but local attention normalizes against only that shard. Simply summing the local outputs would therefore be wrong because every rank used a different softmax denominator.

The missing information is small: each rank needs to contribute its local output and its local log-sum-exp statistic. The LSE tells the group how much weight that local output should receive in the global softmax. This is why DCP communicates normalization metadata instead of exchanging the full KV context.

## The Core Idea

DCP decomposes attention by KV ownership, not by query ownership. Each rank computes a locally normalized output, then vLLM converts it into the rank's correctly weighted contribution to the global softmax. The collective at the end only sums those contributions, so the algebra is identical to non-DCP attention even though the intermediate computation is distributed.

## Symbol Map

The derivation applies independently to every batch row `b` and head `h`. `N` is the DCP group size, `B` is the number of query rows, `H` is the head count, and `D` is the head dimension.

| Symbol | Human name | Scope / shape | Plain meaning |
|---|---|---|---|
| $q_{b,h}$ | query vector | per batch row and head | The query replicated to every DCP rank for decode. |
| $\mathcal{K}_i$ | local KV shard | rank `i` | The keys and values owned by DCP rank `i`. |
| $o_i$ | local attention output | `[B, H, D]` | Output after softmax over only $\mathcal{K}_i$. |
| $L_i$ | local LSE | `[B, H]` | Local log-sum-exp normalizer from rank `i`. |
| $L$ | global LSE | `[B, H]` | Log-sum-exp over all DCP shards. |
| $\alpha_i$ | merge weight | `[B, H]` | $\exp(L_i - L)$, the global weight of rank `i`'s local output. |
| $O$ | logical global output | `[B, H, D]` | Exact attention output before the final head distribution. |

## Deep Dive

### 1. The attention path starts with local KV ownership

**What it does:** The MLA decode path selects a DCP merge backend after local attention returns an output and LSE. <a class="code-link" href="../../../../external-repos/vllm/vllm/model_executor/layers/attention/mla_attention.py#L896" data-code-repo="vllm-a0c092ee72c0" data-code-path="vllm/model_executor/layers/attention/mla_attention.py" data-code-line="896" data-code-end-line="921"><code>MLAImpl.forward_impl</code></a> chooses A2A when configured, `AG+AR` when PCP needs replicated heads, and `AG+RS` otherwise.

**Why it matters:** The same local attention result can be finished with different communication layouts without changing the mathematical attention result.

**How it works:** For a decode row, every DCP rank receives the same query `Q`, but rank `i` reads only `(K_i, V_i)`. It returns:

```text
cp_attn_out_i: [B, H, D]
cp_attn_lse_i: [B, H]
```

The query is replicated because it is the thing that must attend over every shard. The persistent KV context is what DCP partitions.

**The intuition:** Every rank asks the same question against a different part of the memory.

**A concrete example:** With DCP4, one decode query runs four local attentions: the query is the same, while each rank sees one interleaved quarter of the request's KV history.

**Remember:** DCP splits KV ownership; it does not split a one-token decode query.

### 2. `_cp_lse_common` gathers only normalization metadata

**What it does:** `_cp_lse_common` collects one `[B, H]` LSE tensor from every DCP rank and reshapes the result to `[N, B, H]` before correction. <a class="code-link" href="../../../../external-repos/vllm/vllm/v1/attention/ops/common.py#L182" data-code-repo="vllm-a0c092ee72c0" data-code-path="vllm/v1/attention/ops/common.py" data-code-line="182" data-code-end-line="210"><code>_cp_lse_common</code></a> returns immediately when the group size is one.

**Why it matters:** Sending the full KV context would defeat DCP. The LSE is enough to reconstruct the global softmax denominator after each rank has computed its local output.

**How it works:** The helper makes the LSE contiguous, performs `all_gather(..., dim=0)`, and views the gathered rows as:

```text
rank 0: [B, H]  -> lses[0]
rank 1: [B, H]  -> lses[1]
...
rank N-1        -> lses[N-1]

lses: [N, B, H]
```

Each `(b, h)` position now contains `[L_0, L_1, ..., L_(N-1)]`. The local output remains local until the correction kernel and final reduction.

**The intuition:** Exchange the one statistic needed to compare local softmax scales, not the data that was already processed.

**A concrete example:** For DCP2, every rank receives two LSE values for each query row and head, but it does not receive the other rank's KV tensor.

**Remember:** LSE all-gather is the first cross-rank step; it does not yet produce the final output.

### 3. The Triton kernel computes a stable global LSE

**What it does:** `_correct_attn_cp_out_kernel` processes one `(batch_idx, head_idx)` pair per Triton program and computes the global LSE from all ranks' local LSE values. <a class="code-link" href="../../../../external-repos/vllm/vllm/v1/attention/ops/common.py#L10" data-code-repo="vllm-a0c092ee72c0" data-code-path="vllm/v1/attention/ops/common.py" data-code-line="10"><code>_correct_attn_cp_out_kernel</code></a> then scales that rank's output vector in place.

**Why it matters:** Directly evaluating `log(sum(exp(L_i)))` can overflow when LSE values are large. The max-subtraction form keeps the exponentials bounded.

**How it works:** For one `(b, h)`, the kernel loads the vector of rank LSEs using the provided strides. It treats NaN and positive infinity as invalid by replacing them with negative infinity, then computes:

$$
m = \max_i L_i
$$

$$
L = m + \log \sum_i \exp(L_i - m)
$$

The code uses `exp`/`log` for natural-log LSE and `exp2`/`log2` when `IS_BASE_E` is false. The final `L` is written to `vlse_ptr[b, h]`. If every value is invalid, the max is temporarily set to zero so the reduction remains defined and produces a negative-infinity LSE rather than a NaN.

**The intuition:** Subtract the largest scale to calculate safely, then add it back because it was factored out of the logarithm.

**A concrete example:** If `L_0=100` and `L_1=99`, the kernel computes `100 + log(1 + exp(-1))`, not just `log(1 + exp(-1))`.

**Remember:** The max is a numerical trick; adding it back preserves the original LSE scale.

### 4. The kernel converts local output into a global contribution

**What it does:** After computing `L`, the kernel selects the current rank's `L_i`, calculates `exp(L_i - L)`, and multiplies every one of the `D` output features by that factor. <a class="code-link" href="../../../../external-repos/vllm/vllm/v1/attention/ops/common.py#L10" data-code-repo="vllm-a0c092ee72c0" data-code-path="vllm/v1/attention/ops/common.py" data-code-line="10"><code>_correct_attn_cp_out_kernel</code></a> receives `lse_idx=cp_rank` for this selection, while <a class="code-link" href="../../../../external-repos/vllm/vllm/v1/attention/ops/common.py#L111" data-code-repo="vllm-a0c092ee72c0" data-code-path="vllm/v1/attention/ops/common.py" data-code-line="111" data-code-end-line="178"><code>correct_attn_out</code></a> launches the grid `(B, H, 1)` and passes the current rank index.

**Why it matters:** A locally normalized output cannot be added directly to another locally normalized output. The factor restores the rank's share of the global softmax denominator.

**How it works:** If rank `i` computed:

$$
o_i = \frac{\sum_{j \in \mathcal{K}_i} \exp(s_{i,j}) v_j}{\exp(L_i)}
$$

then the kernel writes:

$$
\widetilde{o}_i = o_i \exp(L_i - L)
$$

Because `new_output_ptr` and `outputs_ptr` are both passed as `out` by `correct_attn_out`, this correction is in place. The kernel returns the corrected local contribution and the global `[B, H]` LSE buffer; it does not perform the cross-rank sum.

**The intuition:** Reweight each local answer by how much probability mass its shard contributes globally.

**A concrete example:** If rank 0 owns a shard with twice the softmax mass of rank 1, its corrected output receives twice the merge weight, regardless of the two local outputs' raw magnitudes.

**Remember:** The kernel produces a contribution, not the final global attention output.

### 5. `cp_lse_ag_out_rs` sums and distributes head slices

**What it does:** `cp_lse_ag_out_rs` calls the common LSE correction and then reduce-scatters the corrected output along the head dimension. <a class="code-link" href="../../../../external-repos/vllm/vllm/v1/attention/ops/common.py#L213" data-code-repo="vllm-a0c092ee72c0" data-code-path="vllm/v1/attention/ops/common.py" data-code-line="213" data-code-end-line="236"><code>cp_lse_ag_out_rs</code></a> implements this default DCP path.

**Why it matters:** Every rank contributes to every head's global attention, but after the sum each rank only needs a slice of the heads for the following tensor-parallel computation.

**How it works:** The input to `reduce_scatter` is `[B, H, D]` on every rank after local correction. The collective sums rank contributions and returns:

```text
per-rank output: [B, H/N, D]
```

If `return_lse=True`, the global LSE is sliced to the same head range as the returned output. The mathematical output is still the same `O`; only its physical ownership changes.

**The intuition:** Sum the complete answer, then hand each rank only the heads it is responsible for storing.

**A concrete example:** With DCP4 and 32 heads, each rank receives the exact result for eight heads after the reduce-scatter.

**Remember:** `AG+RS` combines contributions and head-shards the result.

### 6. `cp_lse_ag_out_ar` sums and replicates the full output

**What it does:** `cp_lse_ag_out_ar` uses the same LSE gather and correction, then all-reduces the corrected output. <a class="code-link" href="../../../../external-repos/vllm/vllm/v1/attention/ops/common.py#L238" data-code-repo="vllm-a0c092ee72c0" data-code-path="vllm/v1/attention/ops/common.py" data-code-line="238" data-code-end-line="261"><code>cp_lse_ag_out_ar</code></a> returns the complete reduced tensor on every rank.

**Why it matters:** PCP can require replicated heads after DCP attention, so a head-sharded reduce-scatter would not satisfy the next stage's shape contract.

**How it works:** Each rank contributes $\widetilde{o}_i$; `all_reduce` sums those contributions and leaves:

```text
per-rank output: [B, H, D]
```

The following MLA path can then use the full head layout on every participant.

**The intuition:** Same exact sum, different delivery: every rank gets the whole answer.

**A concrete example:** In a PCP plus DCP configuration, `AG+AR` keeps the DCP result replicated so PCP's later head restoration does not need a missing head slice.

**Remember:** `AG+AR` changes output distribution, not attention mathematics.

### 7. A2A uses the same algebra with a different transport

**What it does:** The optional `dcp_a2a_lse_reduce` path packs output and fp32 LSE payloads, exchanges them with one all-to-all, and performs an exact LSE-weighted reduction. <a class="code-link" href="../../../../external-repos/vllm/vllm/v1/attention/ops/dcp_alltoall.py#L392" data-code-repo="vllm-a0c092ee72c0" data-code-path="vllm/v1/attention/ops/dcp_alltoall.py" data-code-line="392" data-code-end-line="460"><code>dcp_a2a_lse_reduce</code></a> enforces that the head count is divisible by the DCP world size and returns a head-scattered result.

**Why it matters:** A2A can change communication volume and packing behavior without changing the required exact merge.

**How it works:** The function computes `H_per_rank = H / N`, packs each rank's output plus LSE, runs `all_to_all_single`, and unpacks/reduces the received values. Unlike the common AG path, it performs the LSE-weighted combination inside its A2A reducer rather than calling the common Triton correction helper.

**The intuition:** Transport the partial answers and their normalization metadata together, then apply the same global-softmax identity after exchange.

**A concrete example:** If `H=32` and `DCP=4`, A2A exchanges four eight-head partitions and returns `[B, 8, D]` per rank.

**Remember:** A2A is a transport variant, not an approximation variant.

## The Exactness Proof

Without DCP, ordinary attention over the full KV set $\mathcal{K}$ is:

$$
O = \frac{\sum_{j \in \mathcal{K}} \exp(s_j) v_j}{\sum_{j \in \mathcal{K}} \exp(s_j)}.
$$

DCP partitions the keys and values into disjoint shards $\mathcal{K}_i$. Define:

$$
Z_i = \sum_{j \in \mathcal{K}_i} \exp(s_j),
\qquad
N_i = \sum_{j \in \mathcal{K}_i} \exp(s_j) v_j.
$$

Each rank computes $o_i = N_i / Z_i$ and $L_i = \log Z_i$. The global LSE satisfies:

$$
\exp(L) = \sum_i \exp(L_i) = \sum_i Z_i.
$$

The correction kernel computes:

$$
\widetilde{o}_i = o_i \exp(L_i - L)
= \frac{N_i}{Z_i}\frac{Z_i}{\sum_k Z_k}
= \frac{N_i}{\sum_k Z_k}.
$$

The final reduction therefore gives:

$$
\sum_i \widetilde{o}_i
= \frac{\sum_i N_i}{\sum_i Z_i}
= \frac{\sum_{j \in \mathcal{K}} \exp(s_j) v_j}{\sum_{j \in \mathcal{K}} \exp(s_j)}
= O.
$$

This is why DCP matches the no-DCP attention flow: it distributes the numerator and denominator separately, then recombines them exactly.

## Putting It Together

For one decode step with DCP enabled:

1. The model computes local attention on every rank with the same query and rank-local KV shard.
2. Each rank returns `cp_attn_out [B, H, D]` and `cp_attn_lse [B, H]`.
3. `_cp_lse_common` all-gathers LSE into `[N, B, H]`.
4. `correct_attn_out` launches `_correct_attn_cp_out_kernel` for each `(b, h)` pair.
5. The kernel computes stable global LSE and scales the current rank's output by `exp(L_i - L)`.
6. `cp_lse_ag_out_rs` reduce-scatters the corrected sum, or `cp_lse_ag_out_ar` all-reduces it; configured A2A uses its packed all-to-all reducer.
7. MLA continues with the chosen physical head layout and the output projection.

For `DCP=1`, `_cp_lse_common` returns the local output immediately, so there is no LSE collective or correction kernel in the path.

## What This Buys You

### The headline claim

DCP reduces the KV context each rank must own and process while preserving the exact attention result through a small normalization exchange plus a final output reduction.

### How we know: code-path evidence

| Evidence | What it establishes | Boundary |
|---|---|---|
| `MLAImpl.forward_impl` | Backend selection among AG+RS, AG+AR, and A2A | Static reading of the pinned revision |
| `_correct_attn_cp_out_kernel` | Stable global LSE and local-output correction | Triton kernel behavior was not run on GPU here |
| `cp_lse_ag_out_rs` / `cp_lse_ag_out_ar` | Final reduction and output distribution | Does not establish interconnect performance |
| `dcp_a2a_lse_reduce` | Packed A2A exact merge path | Hardware-specific transport remains unverified |

### The mechanism behind the result

DCP saves local KV work and KV capacity, but it adds an LSE exchange and an output collective. The choice between reduce-scatter, all-reduce, and A2A is a shape and communication contract, not a different attention definition.

### How to read these claims

The algebra proves equivalence under the implementation's assumptions: KV shards are correctly partitioned, local LSE values describe the same scores as the non-DCP computation, and every required collective is executed by the full DCP group. This page makes no throughput claim; no multi-GPU serving benchmark was run here.

## Where It Breaks

| Failure mode | When it happens | Impact |
|---|---|---|
| Missing collective participant | One rank skips the LSE gather or final reduction | The distributed step can hang or produce no valid result. |
| Invalid A2A head shape | `H` is not divisible by DCP world size | A2A raises before packing the buffers. |
| Invalid or empty LSE values | Local attention emits NaN or unusable positive infinity | The kernel masks invalid values; an all-invalid row remains negative infinity and needs the caller's normal masking semantics. |
| Wrong KV ownership metadata | Block tables or local sequence lengths disagree with DCP interleaving | A rank can attend to the wrong physical KV locations. |
| Unsupported cache/backend combination | The configured cache type or attention backend is not DCP-aware | Startup or backend selection rejects the configuration. |

## One Thing to Remember

**DCP makes local attention incomplete but makes the missing information recoverable:** each rank computes a local numerator and denominator, the LSE exchange reconstructs the global denominator, and `AG+RS`, `AG+AR`, or A2A sums the correctly rescaled contributions to recover ordinary attention exactly.

## Verification Boundary and Limits

This page is a static reading of the clean vLLM checkout at commit `a0c092ee72c0dcefbb3b3e74f97ac62d842e5f4b`, inspected on 2026-08-11. The repository includes CPU reference coverage for A2A and focused DCP localization tests, but this workspace did not run a CUDA multi-GPU serving test. The proof describes exact arithmetic; floating-point reduction order and backend implementation details can introduce ordinary numerical differences without changing the algorithmic contract.

## Go Deeper

- **Understand the overview:** [vLLM DCP and PCP](../vllm-context-parallelism.md)
- **Read the communication primitives:** [All-Gather](../../../terms/all-gather.md), [All-Reduce](../../../terms/all-reduce.md), and [All-to-All](../../../terms/all-to-all.md)
- **Inspect the source revision:** [vllm-project/vllm at the pinned commit](https://github.com/vllm-project/vllm/tree/a0c092ee72c0dcefbb3b3e74f97ac62d842e5f4b)
- **Reproduce:** A CUDA-capable distributed environment is required for an end-to-end DCP run; it was not available for this static reading.
