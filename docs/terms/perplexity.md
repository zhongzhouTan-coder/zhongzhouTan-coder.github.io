---
title: "Perplexity"
summary: "The exponential of a causal language model's average next-token negative log likelihood on a specified tokenized corpus."
tooltip: "Perplexity (PPL) converts average next-token negative log likelihood into an effective-choice scale: lower is better for the same tokenizer, corpus, and context protocol. It measures predictive fit, not truthfulness or general usefulness."
layout: default
confidence: high
category: algorithms
sources:
  - raw/algorithms/perplexity-pytorch-calculation--web-2026-08-22-1786b54fa0c7.html
  - raw/algorithms/perplexity-pytorch-calculation--web-2026-08-22-1786b54fa0c7.metadata.json
  - derived/web-markdown/algorithms/perplexity-pytorch-calculation--web-2026-08-22-1786b54fa0c7.md
  - raw/algorithms/perplexity-fixed-length-models--web-2026-08-22-32174cdba6b9.html
  - raw/algorithms/perplexity-fixed-length-models--web-2026-08-22-32174cdba6b9.metadata.json
  - derived/web-markdown/algorithms/perplexity-fixed-length-models--web-2026-08-22-32174cdba6b9.md
aliases:
  - PPL
  - 困惑度
mention_lint: canonical
appears_in:
  - docs/algorithms/foundations/perplexity.md
  - docs/training/foundation-models/gpt-1.md
  - docs/training/foundation-models/gpt-2.md
updated: 2026-08-22
---

# Perplexity

**Perplexity** is the exponential of a causal language model's average next-token negative log likelihood on a specified tokenized corpus.

## Why It Exists

Sequence probabilities multiply and quickly become tiny as text gets longer. Negative log likelihood makes the evidence additive, averaging removes the direct effect of sequence length, and exponentiation returns the result to an effective-choice scale that is easier to read.

## How It Works

For target tokens $x_1,\ldots,x_N$:

$$
\operatorname{PPL}=\exp\left(-\frac{1}{N}\sum_{t=1}^{N}\log p_\theta(x_t\mid x_{<t})\right).
$$

Lower PPL means the model assigned higher probability to the observed tokens, but only under the same tokenizer, evaluation corpus, context window, stride, masking policy, and log base. The common “number of choices” explanation is an effective uniform-choice analogy, not a literal count of candidates.

## Tradeoffs

PPL is fast and reference-free, which makes it useful for language-modeling regressions and pretraining comparisons. It does not measure factuality, instruction following, reasoning, or human preference, and PPL values across different tokenizers or corpora are not directly comparable.

## Common Confusions

- **PPL vs. accuracy:** Accuracy checks only whether the top prediction is correct; PPL rewards assigning calibrated probability to the observed token even when it is not top-1.
- **PPL vs. generation quality:** A fluent continuation can have low PPL while being false or unhelpful.
- **PPL vs. cross-entropy:** For one-hot causal labels, PPL is the exponential of the average cross-entropy/NLL.
- **PPL vs. token count:** A PPL of 20 does not mean the vocabulary has 20 tokens; it is an average effective branching factor under the evaluation distribution.

## Where It Appears

- [Perplexity (PPL): From Next-Token Loss to Reliable Evaluation](../algorithms/foundations/perplexity.md) — Derives the metric, works through a three-token example, and explains fixed-length sliding-window evaluation.
- [GPT-1](../training/foundation-models/gpt-1.md) — Reports language-modeling perplexity on BooksCorpus during the original pretraining-plus-finetuning study.
- [GPT-2](../training/foundation-models/gpt-2.md) — Uses held-out perplexity to show that the scaled model continues improving on WebText.

## Related Terms

- [The Softmax Function](../algorithms/foundations/softmax.md) — Converts logits into the token probabilities used by NLL.
- [Layer Normalization](layer-normalization.md) — A separate Transformer operation; normalization is not what PPL measures.
