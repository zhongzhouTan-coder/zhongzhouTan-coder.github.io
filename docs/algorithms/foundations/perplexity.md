---
title: "Perplexity (PPL): From Next-Token Loss to Reliable Evaluation"
summary: "Explains perplexity as exponentiated average next-token negative log likelihood, with a worked example, fixed-length evaluation protocol, and comparison caveats."
layout: default
confidence: medium
sources:
  - raw/algorithms/perplexity-pytorch-calculation--web-2026-08-22-1786b54fa0c7.html
  - raw/algorithms/perplexity-pytorch-calculation--web-2026-08-22-1786b54fa0c7.metadata.json
  - derived/web-markdown/algorithms/perplexity-pytorch-calculation--web-2026-08-22-1786b54fa0c7.md
  - raw/algorithms/perplexity-fixed-length-models--web-2026-08-22-32174cdba6b9.html
  - raw/algorithms/perplexity-fixed-length-models--web-2026-08-22-32174cdba6b9.metadata.json
  - derived/web-markdown/algorithms/perplexity-fixed-length-models--web-2026-08-22-32174cdba6b9.md
updated: 2026-08-22
---

# Perplexity (PPL): From Next-Token Loss to Reliable Evaluation

**Requested source:** [关于困惑度 PPL 的知乎文章](https://zhuanlan.zhihu.com/p/686808564)

**Accessible evidence:** [PyTorch perplexity explanation](https://geek-docs.com/pytorch/pytorch-questions/35_pytorch_calculate_perplexity_in_pytorch.html) · [Hugging Face fixed-length perplexity guide](https://huggingface.co/docs/transformers/perplexity)

**Related pages:** [The Transformer](transformer.md) · [The Softmax Function](softmax.md) · [GPT-1](../../training/foundation-models/gpt-1.md) · [GPT-2](../../training/foundation-models/gpt-2.md) · [Perplexity glossary term](../../terms/perplexity.md)

> **Warning:** Direct HTTP capture of the requested Zhihu page returned HTTP 403, and the browser-assisted attempt returned an access-check JSON response. The local evidence is therefore a concise, explicitly marked excerpt of the linked PyTorch page plus the official Hugging Face guide, not a complete transcription of the Zhihu article.

## TL;DR

**What:** [Perplexity](../../terms/perplexity.md) (PPL) measures how well an autoregressive language model assigns probability to the actual next tokens in an evaluation corpus.

**How:** Compute the average token-level negative log likelihood, then exponentiate it: `PPL = exp(average NLL)` when the loss uses natural logarithms.

**The number:** A lower PPL means better next-token fit only when the model, tokenizer, corpus, context protocol, and loss definition are held comparable; it does not directly mean better reasoning, truthfulness, or helpfulness.

## The Big Picture

The whole metric is one pipeline:

`logits → token probabilities → target-token NLL → average over valid tokens → exponentiate → PPL`

The important detail is that PPL is not a second training objective. It is a human-readable re-expression of the same average next-token loss used by a causal language model.

## Why This Exists

A causal language model predicts a probability distribution over the vocabulary at every position. For a target sequence, its probability is a product of many conditional probabilities:

$$
p(x_{1:N})=\prod_{t=1}^{N}p_\theta(x_t\mid x_{<t}).
$$

That product is awkward to compare: it rapidly becomes tiny as the sequence gets longer. Taking the negative logarithm turns multiplication into addition, and averaging over tokens removes the direct dependence on sequence length. The resulting loss is mathematically useful but not very intuitive, so PPL exponentiates it back into an “effective number of choices.”

## The Core Idea

**PPL is the geometric mean of the inverse probabilities that the model assigned to the observed tokens.** If the model gives the correct next token high probability, its contribution to NLL is small and PPL falls. If the model is surprised repeatedly, the inverse probabilities grow and PPL rises.

The “number of choices” interpretation is an effective-uniform-choice analogy, not a claim that the model literally considers exactly that many equally likely tokens at every position.

## Symbol Map

The notation is small enough to keep explicit:

| Symbol | Meaning | Scope |
|---|---|---|
| $x_t$ | The target token at position `t` | One token in the evaluation sequence |
| $x_{<t}$ | All preceding tokens visible to the causal model | Context for position `t` |
| $p_\theta(x_t\mid x_{<t})$ | Probability assigned to the observed target token | One prediction position |
| $N$ | Number of valid target tokens | Sequence or corpus after masking |
| `NLL` | Negative log likelihood, $-\log p$ | Token-level loss or its average |
| `PPL` | Exponentiated average NLL | One scalar evaluation metric |

With natural logarithms, PPL uses `exp`. With base-2 cross-entropy measured in bits, the equivalent expression is `2^(cross entropy)`.

## From Next-Token Probabilities to PPL

### 1. Predict a distribution, then select the observed token

At position `t`, the model emits logits for the whole vocabulary. Softmax turns them into probabilities, and evaluation selects the probability assigned to the actual target token. The metric does not ask whether the target was the top-1 prediction; it uses the full probability assigned to that target.

### 2. Convert probability into surprise

For a target-token probability `p_t`, define its negative log likelihood as

$$
\ell_t=-\log p_t.
$$

Assigning probability `1` gives zero loss. Assigning a small probability gives a large penalty, which is why one very surprising token can materially raise the sequence PPL.

### 3. Average over valid tokens and exponentiate

The corpus-level metric is

$$
\operatorname{PPL}(x_{1:N})
=\exp\left(\frac{1}{N}\sum_{t=1}^{N}-\log p_\theta(x_t\mid x_{<t})\right).
$$

For one-hot next-token labels, the average causal cross-entropy is the average NLL, so PPL is simply the exponential of that evaluation loss.

**The intuition:** NLL adds evidence token by token; PPL converts the average back to the scale of an equivalent uniform choice set.

**A concrete example:** Suppose the model assigns the observed tokens probabilities `0.5`, `0.25`, and `0.125`.

| Target token | Assigned probability | NLL with natural log |
|---:|---:|---:|
| `x₁` | 0.500 | 0.693 |
| `x₂` | 0.250 | 1.386 |
| `x₃` | 0.125 | 2.079 |
| **Average** | — | **1.386** |

The PPL is `exp(1.386) ≈ 4`. The model behaves, in the effective-choice sense, like a uniform four-way choice on average.

**Remember:** PPL is built from the probability of the correct target token, not from accuracy alone.

## Evaluating a Fixed-Length Model Correctly

The formula assumes that each target token is conditioned on the preceding sequence. A model with a finite context window cannot literally see an arbitrarily long prefix, so the evaluation protocol becomes part of the metric.

| Evaluation protocol | What it does | Consequence |
|---|---|---|
| Disjoint chunks | Splits a long corpus into non-overlapping windows | Fast, but early tokens in each chunk see little context and PPL is usually worse |
| Sliding window | Reuses overlapping context and scores only newly exposed targets | Closer to the full autoregressive decomposition, but costs more forward passes |
| Strided sliding window | Moves the window by a stride larger than one | Practical speed/quality compromise; smaller strides usually give more context |

When a target window overlaps the previous window, tokens used only as context must be masked from the loss. Otherwise the same token is counted both as context and as a scored target, inflating the effective token count or evaluating the wrong positions.

The Hugging Face guide reports GPT-2 PPL of `19.44` with stride `1024` and `16.44` with stride `512`. These are not contradictory model-quality claims: they demonstrate that a smaller stride gives target tokens more preceding context and changes the reported score.

## Putting It Together

For one long evaluation sequence, the runtime/evaluation trace is:

1. **Tokenize:** convert the corpus with the model's tokenizer; record the exact token count and tokenizer version.
2. **Window:** choose a maximum context length and a stride; slice a window from the token sequence.
3. **Shift targets:** use the next token as the label for each causal position, respecting the model's internal label shift.
4. **Mask context:** mark overlapping context tokens, padding, and any ignored labels so they do not enter the loss.
5. **Accumulate NLL:** multiply each batch's mean loss by its number of valid loss tokens, then sum both NLL and token count.
6. **Finish:** divide total NLL by total valid tokens and compute `exp(total_NLL / total_tokens)`.

The token-weighted accumulation matters: averaging per-window PPL values would give short and long windows equal weight, which is not the same as computing one corpus-level average NLL.

## What This Buys You

### The headline claim

PPL is a compact intrinsic metric for comparing how well causal language models predict a specified token distribution under a specified evaluation protocol.

### How we know: metric behavior

| Question | PPL answers | PPL does not answer |
|---|---|---|
| What is being measured? | Probability assigned to observed next tokens | Whether an answer is factually correct |
| What is a lower score? | Less average predictive surprise on this corpus | Automatically better conversation or reasoning |
| Why is it useful? | Fast, reference-free comparison for the same tokenized corpus | A universal replacement for task-specific or human evaluation |
| What changes the number? | Model fit, tokenizer, corpus, context length, stride, and masking | Only model “intelligence” in isolation |

### The mechanism behind the numbers

PPL exposes distributional fit. A model can reduce PPL by becoming better at the regularities of the evaluation corpus, even if that corpus is narrow, repetitive, noisy, or unrelated to the task users care about. This is why PPL is especially useful for pretraining and language-modeling regressions, while downstream benchmarks and human checks remain necessary for generation quality.

### ⚠️ How to read these numbers

Never compare a PPL number without also recording the tokenizer, tokenization normalization, evaluation corpus, context window, stride, ignored-label policy, and log base. A lower number from a different protocol may simply reflect easier tokenization or more available context.

## Where It Breaks

| Failure mode | When it happens | Impact |
|---|---|---|
| Tokenizer mismatch | Two models split the same text into different token sequences | Raw PPL values are not directly comparable because the unit being averaged changed |
| Corpus mismatch | Models are evaluated on different domains or languages | A score can reflect domain familiarity rather than general model quality |
| Context-protocol mismatch | One evaluation uses disjoint chunks and another uses an overlapping stride | The model receives different amounts of history, so the numbers measure different tasks |
| Masked-model misuse | PPL is applied to a model such as BERT without a causal next-token factorization | The conditional probability being measured is not the same autoregressive quantity |
| Accuracy-only interpretation | A model is judged from whether it selected the top token | Probability calibration and useful uncertainty are discarded |
| Generation-quality overclaim | A low-PPL model is assumed to be factual, helpful, or aligned | Fluency on the test distribution is mistaken for task success or truth |
| Per-window averaging | PPL is averaged across windows instead of accumulating token-weighted NLL | Short windows receive too much influence and the corpus score is biased |

## One Thing to Remember

**Perplexity is exponentiated average next-token surprise:** it tells you how well a causal model fits a particular tokenized corpus under a particular context protocol, and its “effective number of choices” intuition is useful only when those conditions stay fixed.

## Go Deeper

- **Read:** [Requested Zhihu article](https://zhuanlan.zhihu.com/p/686808564) and the [linked PyTorch explanation](https://geek-docs.com/pytorch/pytorch-questions/35_pytorch_calculate_perplexity_in_pytorch.html).
- **Implement correctly:** [Hugging Face’s fixed-length perplexity guide](https://huggingface.co/docs/transformers/perplexity), especially the strided sliding-window example.
- **Understand the pipeline:** [The Transformer](transformer.md) and [The Softmax Function](softmax.md).
- **See the metric in a model paper:** [GPT-1](../../training/foundation-models/gpt-1.md) and [GPT-2](../../training/foundation-models/gpt-2.md).
