---
title: "Terms Glossary"
summary: "Alphabetical glossary of technical terms used across the knowledge base, with concise definitions and cross-links to papers and insight pages."
layout: default
confidence: high
sources:
  - AGENTS.md
  - .github/instructions/docs-terms.instructions.md
updated: 2026-07-27
---

# Terms Glossary

Quick-lookup definitions for technical concepts that appear across multiple papers in this knowledge base. Each term page is a self-contained definition page with backlinks to the papers that use it.

## Training

- [Microbatch](microbatch.md) — A small chunk of a training batch used to enable pipeline parallelism; the unit of work injected into a pipeline schedule.
- [Scatter/Gather](scatter-gather.md) — A cross-node communication optimization in pipeline parallelism that avoids sending redundant activation tensors over slow inter-node links.
- [All-Gather](all-gather.md) — An NCCL collective that gathers shards across ranks into a full replicated tensor; the split/all-gather pattern reduces cross-node traffic at pipeline boundaries.
- [All-Reduce](all-reduce.md) — An NCCL collective that sums tensors element-wise across ranks; powers gradient synchronization and tensor-parallel output aggregation.
- [Sequence Parallelism](sequence-parallelism.md) — A distributed training strategy that splits the input sequence along the length dimension across GPUs, forming a fourth parallelism dimension orthogonal to data, pipeline, and tensor parallelism.

## Algorithms

_No terms yet. Terms are auto-created when agents detect a concept reused across multiple paper insight pages._

## Hardware

_No terms yet._

## Frameworks

_No terms yet._

## Benchmarks

_No terms yet._
