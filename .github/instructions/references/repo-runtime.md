# Repository Evidence: Mechanisms and Round Trips

## Design the Page for Its Reader

### Explain Mechanisms, Not Just Paths

A call path alone does not make a professional reader understand a serving
system. For every performance, memory, or correctness mechanism the page
touches, answer these questions explicitly:

- **Motivation before mechanics.** State the cost or correctness problem the
  mechanism removes — in units the reader cares about (bytes read per decode
  token, extra tokens recomputed per prefix hit, added latency per layer) —
  before describing how it works. If removing the mechanism would silently
  corrupt results rather than merely slow them down, say so first.
- **Coordinate spaces and units.** For every index, offset, or length, name
  its unit (token position, compressed or pooled state, cache slot, block id,
  byte offset) and link the code that converts between spaces. A reader who
  cannot tell which space an index lives in cannot follow the trace. When a
  buffer's indices differ in resolution from the cache they address (for
  example, compressed-group indices addressing a paged KV cache), give the
  concrete mapping: direction, scaling factor, and the line where it happens.
- **State hierarchies as hierarchies.** When a system keeps several caches or
  state stores, compare them on scope (local vs global), lifetime, precision
  or dtype, record width, and who reads each — one compact table or paragraph
  covering every store. Do not let the reader reconstruct the hierarchy from
  scattered mentions.
- **Overlap and asynchrony need a reason.** When work moves to auxiliary
  streams, background threads, or deferred sub-batches, explain what it hides
  under and which invariant the synchronization before consumption protects.
- **Cross-component mechanisms need both ends.** For scheduler-to-worker and
  producer-to-consumer handoffs (a flag set at schedule time, consumed by a
  kernel), link the assignment site, the transport (message field or shared
  buffer), and the consumption site. State which requests or steps the
  mechanism applies to and which it excludes (prefill vs decode, dummy or
  capture batches, idle data-parallel ranks).
- **Hierarchical or two-level mechanisms.** When selection, routing, or
  filtering happens in stages, describe every stage's granularity and gate
  configuration, or state explicitly why a stage is out of scope.

### Trace Request Round Trips

When runtime request handling is part of the page's central question, include a
code-evidenced **request round trip**. Follow one concrete request, token,
tensor, cache block, or message from its boundary entry down to the deepest
meaningful operation and back to the externally visible result. Do not present
an entry-to-backend call chain as an end-to-end flow when the return path is
missing.

Cover the applicable stages:

1. Boundary entry, parsing, or admission.
2. Routing, scheduling, or coordination.
3. Worker, model, storage, or backend handoff.
4. Core state transition or computation.
5. First material result and its propagation back to the caller.
6. Result transformation, aggregation, or response assembly.
7. Response emission plus cleanup, release, or important failure return.

For each non-trivial step, name the actor, input state, operation, output or
return value, and a revision-aware code link to the exact file, symbol, and
smallest useful line range. A component name or unlinked symbol is orientation,
not code evidence. Evidence only at the deepest callee is also insufficient
when a caller transforms, aggregates, serializes, or conditionally discards the
returned value; link that caller-side behavior separately. Omit trivial helper
returns that do not change state, ownership, control flow, or representation.
Separate load-time ownership decisions from runtime writes and reads. At the
central computation, inspect the actual callee or kernel invocation as well as
its selector or wrapper. When a shared buffer matters, link both its producer
and consumer; distinguish intermediate state such as selected indices from the
attention result and externally returned token.

Use a Mermaid sequence diagram when calls and returns cross three or more
components, processes, services, or devices. Show both call and return arrows,
mark asynchronous handoffs and important branches, and keep the editable
`.mmd` source beside the page. The diagram is the orientation layer, not a
substitute for evidence: follow it with a numbered trace or compact table whose
meaningful steps link to the pinned implementation and map to finding IDs in
the derived evidence note.
Draw optional operations as optional. If the page embeds the diagram, keep its
Mermaid block identical to the saved `.mmd` source. Parse or render edited
Mermaid with an available tool; `./scripts/lint-docs.sh` does not check Mermaid
syntax. Sequence-message text must not contain unescaped statement separators
such as `;`. If no parser is available, keep the syntax simple and report that
rendering was not verified.

Use portable Markdown callouts labeled `Evidence` and `Inference` when the
reader could otherwise confuse direct code behavior with repository-level
synthesis; do not hide required reasoning inside callouts.
