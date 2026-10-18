---
name: analyze-model-inference-path
description: Trace a model's serving or inference implementation from a request or scheduled batch through model layers, attention backends, caches, sampling, and returned output. Use for repository-backed inference code analysis or reader-facing code-path docs, not paper-only model architecture summaries.
---

# Analyze Model Inference Path

Produce a code-evidenced path that lets a reader answer: what enters the model,
which state changes, which implementation branch runs, and what reaches the
caller? Select one concrete workload and device/backend branch for the first
walkthrough, then explain important alternatives.

## Establish the Boundary and Evidence

- Follow the user's requested outcome. A one-off explanation does not itself
  authorize new wiki artifacts. For maintained repository-backed pages here,
  read `AGENTS.md` and `.github/instructions/repo-reading.instructions.md` for
  pinned revisions, read-only checkouts, derived evidence, and `code-link`
  requirements.
- State the entry and return boundaries, workload (for example prefill or one
  ordinary decode token), hardware, relevant configuration, and what was only
  inferred from code. Do not present a selected static branch as a measured GPU
  run or as the default for every device.
- Separate load-time decisions (model registry, layer construction, backend
  selector, cache specs) from runtime operations (batch preparation, cache
  writes, index selection, attention read, sampling, return).

## Follow State Down and Back

Trace the same request, token row, or tensor through the actual call sites:

1. Worker/request entry, scheduled rows, positions, slots, and attention
   metadata.
2. Model wrapper and decoder/other model components that the implementation
   actually invokes.
3. Layer input projections and the state each layer or source layer writes.
4. The central attention/backend call, its cache tensors and indices, and the
   first output buffer it produces.
5. Layer output, final hidden state, logits, sampler, postprocessing, and the
   value returned to the chosen boundary.

For every non-obvious handoff, link the smallest useful pinned code span on
both sides when the producer and consumer differ. Quote a short code snapshot
only when a condition, write, kernel call, or return is easier to understand
with its actual arguments. Explain what to notice beside the excerpt.

## Resolve Backend and Cache Contracts

- Compare automatic and explicitly configured backend choices. Record device
  tests, eligibility gates, fallback versus initialization failure, and the
  kernel call for the chosen branch. Do not infer behavior from a backend's name
  alone.
- Map cache ownership separately from physical layout: local attention KV,
  shared/compressed KV, index or retrieval keys, and any buffer of selected
  positions. An index buffer is not attention output or a sampled token.
- For each relevant backend, follow dtype resolution and cache specs to the
  actual record format, tensor shape, page/slot mapping, and decode read.
  Distinguish bytes per stored record from total bytes per input token; account
  for compression ratio, sharing, and page padding before making memory claims.
  The same logical dtype label may select different physical records on
  different GPU families.
- Keep optional roles conditional. A layer that consumes shared state need not
  own its cache, and two source roles may coincide in one layer. Confirm the
  allocation and write predicates before drawing ownership arrows.

## Teach the Path

Start with a code-free mental model, then a compact input/state/output trace
with direct code links. Use a small table for hardware/backend/cache mappings
and a numbered trace for runtime changes. Add a figure only when it clarifies
ownership, branching, or handoffs; place it before its detailed trace and
explain how to read it immediately beside it. Label a paper architecture image
as design intent and a synthesized flow as a pinned-code explanation. Preserve
local editable diagram sources and a readable render as required by `AGENTS.md`.

The [DeepSeek V4.1 GPU inference path](../../../docs/frameworks/serving/deepseek/v4.1-gpu-inference-path/index.md)
is an example of the method, not a required page template or a source of facts
for another model. In particular, its SM100/SM120 cache records and layer-role
names are model-specific.

Before finishing, check that every selected path reaches the stated return
boundary, every cache write has a corresponding read or justified lifetime,
conditional branches are marked, and static claims are distinguished from
runtime verification. Run `./scripts/lint-docs.sh` after docs changes.
