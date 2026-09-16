---
kind: web-extraction
source_url: "https://geek-docs.com/pytorch/pytorch-questions/35_pytorch_calculate_perplexity_in_pytorch.html"
final_url: "https://geek-docs.com/pytorch/pytorch-questions/35_pytorch_calculate_perplexity_in_pytorch.html"
canonical_url: "https://geek-docs.com/pytorch/pytorch-questions/35_pytorch_calculate_perplexity_in_pytorch.html"
title: "Pytorch 中计算 perplexity 的方法"
author: ""
published_at: ""
captured_at: "2026-08-22T03:35:00.000Z"
content_sha256: 1786b54fa0c7ccd3984b938f575aceb5f991f976e5ece79a42abdc3ed24b5c20
renderer: manual-evidence-excerpt
extractor: "browser-readable web retrieval, locally preserved excerpt"
---

# Pytorch 中计算 perplexity 的方法

The source describes perplexity (困惑度) as a measure of how well a language model predicts sentences in a test set. It recommends preparing a test set, running the model to obtain target-token log likelihoods, and exponentiating the average negative log likelihood.

The source's short formula writes `perplexity = exp(平均对数似然)`. Because log probabilities are non-positive, a mathematically correct implementation must use the average **negative** log likelihood, or make the negative sign explicit.
