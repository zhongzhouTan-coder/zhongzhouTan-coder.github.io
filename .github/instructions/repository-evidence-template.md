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
| `docs/frameworks/example/runtime.md` | cache-allocation | `src/cache.py` | `Cache.allocate` | 74 | — |

Column meanings:

- **Docs page:** repository-relative path of the consuming page.
- **Finding:** short stable identifier for the implementation claim.
- **File:** repository-relative source file, never a checkout-relative path.
- **Symbol:** class, function, method, constant, or most relevant entry point.
- **Start:** smallest useful source start line; always required.
- **End:** end of a short complete implementation range, or an em dash when a
  single-line target is sufficient.

## Runtime Flow Evidence

Before drafting, enumerate the end-to-end path in prose. Every step must map to
at least one row in **Required Code Evidence**.

1. Entry or admission — finding ID.
2. Coordination or dispatch — finding ID.
3. Core state transition or computation — finding ID.
4. Materialization or backend handoff — finding ID.
5. Cleanup, failure, or release — finding ID.

Remove irrelevant steps instead of inventing evidence. Add steps when the real
runtime path is longer.

## Link Completion

- [ ] Every Required Code Evidence row has a matching code link.
- [ ] Every runtime-flow step names at least one declared finding.
- [ ] The first meaningful occurrence of every major implementation symbol is
      linked.
- [ ] Symbol maps link important implementation types and operations.
- [ ] Repeated mentions and generic variables remain ordinary inline code.
- [ ] `./scripts/lint-docs.sh` passes.
