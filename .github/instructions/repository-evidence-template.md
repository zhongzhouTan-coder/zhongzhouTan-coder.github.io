# Repository Evidence Map Template

Copy the sections below into `important-files.md` or the purpose-specific
derived analysis note that directly supports a repository-backed docs page.
Replace every example value; do not keep placeholder rows.

Set both `code_links: strict` and `code_evidence: strict` in the consuming docs
page's front matter. The latter makes a missing evidence table a lint error.

## Required Code Evidence

Use exactly these six column names. The code-link checker treats each row as a
contract: the declared docs page must contain a revision-aware code link with
the same repository-relative file, start line, and optional end line.

| Docs page | Finding | File | Symbol | Start | End |
|---|---|---|---|---:|---:|
| `docs/frameworks/example/runtime.md` | request-entry | `src/server.py` | `Server.handle_request` | 120 | 168 |
| `docs/frameworks/example/runtime.md` | request-dispatch | `src/scheduler.py` | `Scheduler.dispatch` | 74 | 109 |
| `docs/frameworks/example/runtime.md` | backend-result | `src/worker.py` | `Worker.execute` | 201 | 248 |
| `docs/frameworks/example/runtime.md` | response-assembly | `src/server.py` | `Server.build_response` | 170 | 194 |

Column meanings:

- **Docs page:** repository-relative path of the consuming page.
- **Finding:** short stable identifier for the implementation claim.
- **File:** repository-relative source file, never a checkout-relative path.
- **Symbol:** class, function, method, constant, or most relevant entry point.
- **Start:** smallest useful source start line; always required.
- **End:** end of a short complete implementation range, or an em dash when a
  single-line target is sufficient.

## Request Round-Trip Evidence

When runtime request handling is in scope, enumerate the complete round trip in
prose before drafting. Follow the same concrete request or stateful object on
the descent and return. Every non-trivial step must map to at least one row in
**Required Code Evidence**.

1. Entry or admission — finding ID.
2. Coordination or dispatch — finding ID.
3. Worker, model, storage, or backend handoff — finding ID.
4. Core state transition or computation — finding ID.
5. First material result returned to its caller — finding ID.
6. Result transformation, aggregation, or response assembly — finding ID.
7. Response emission and cleanup, failure, or release — finding ID.

Remove irrelevant steps instead of inventing evidence. Add steps when the real
runtime path is longer. Do not treat a component name, unlinked symbol, or
diagram node as evidence. Add separate findings when caller-side code changes
the returned value, representation, ownership, or control flow.

Present the evidence-backed round trip in the consuming page with a numbered
trace or a compact table. Link the **Code evidence** cell with the repository's
revision-aware `code-link` anchor; do not leave it as plain text.

| Step | Direction | Actor | Code evidence | Input state | Action | Output or return |
|---:|---|---|---|---|---|---|
| 1 | Request down | API boundary | `request-entry` link | Serialized request | Parse and validate | Internal request |
| 2 | Dispatch down | Scheduler | `request-dispatch` link | Internal request | Select worker | Execution plan |
| 3 | Execute down | Worker | `backend-result` link | Execution plan | Run core operation | Raw result |
| 4 | Return up | API boundary | `response-assembly` link | Raw result | Transform and serialize | External response |

Use a sequence diagram as an orientation layer when component boundaries or
asynchronous behavior make the calls hard to follow. The diagram must show
important return arrows, but it does not replace the linked trace or the
**Required Code Evidence** rows.

## Link Completion

- [ ] Every Required Code Evidence row has a matching code link.
- [ ] Every non-trivial round-trip step names at least one declared finding.
- [ ] The trace follows the same request or stateful object down and back.
- [ ] Return transformation, response assembly, and cleanup are covered when
      they exist in scope.
- [ ] The first meaningful occurrence of every major implementation symbol is
      linked.
- [ ] Symbol maps link important implementation types and operations.
- [ ] Repeated mentions and generic variables remain ordinary inline code.
- [ ] `./scripts/lint-docs.sh` passes.
