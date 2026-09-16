---
kind: web-extraction
source_url: "https://huggingface.co/docs/transformers/perplexity"
final_url: "https://huggingface.co/docs/transformers/perplexity"
canonical_url: "https://huggingface.co/docs/transformers/perplexity"
title: "Perplexity of fixed-length models"
author: "Hugging Face Transformers documentation"
published_at: ""
captured_at: "2026-08-22T03:35:00.000Z"
content_sha256: 32174cdba6b96df1695c9686968196466e5a8ec1b077c2c18281c8958cc1c516
renderer: manual-evidence-excerpt
extractor: "browser-readable web retrieval, locally preserved excerpt"
---

# Perplexity of fixed-length models

For an autoregressive language model, perplexity is the exponentiated average negative log likelihood of the target token sequence:

$$
\operatorname{PPL}(X)=\exp\left(-\frac{1}{T}\sum_{t=1}^{T}\log p_\theta(x_t\mid x_{<t})\right).
$$

The source warns that tokenization directly changes PPL, and that the metric is intended for causal/autoregressive models rather than masked language models such as BERT. For fixed-length models, disjoint chunks waste context; a sliding or strided sliding window gives target tokens more preceding context. Context-only labels must be masked from the loss. The guide reports GPT-2 PPL of 19.44 with stride 1024 and 16.44 with stride 512.
