---
description: "Use when ingesting, inspecting, or documenting a local code repository checkout. Enforces immutable commit-pinned evidence and readable repository-backed docs."
applyTo: "external-repos/**/*, raw/**/*.md, derived/repo-analysis/**/*, docs/**/*.md, sources.json"
---

# Repository Reading Rules

Use this instruction when the evidence is a code repository rather than a
paper, PDF, benchmark report, or web source. General wiki navigation, logs,
front matter, and confidence rules remain in `AGENTS.md` and their dedicated
instruction files.

## Core Invariants

- Choose **reuse**, **defer**, **new revision**, or **new repository** before
  reading code.
- A repository source record is immutable and pinned to one full commit SHA.
  Never rewrite old raw or derived evidence to point at another revision.
- Keep third-party worktrees beneath ignored `external-repos/` and treat them as
  read-only. Do not edit, format, switch, pull, or vendor their files.
- Use `scripts/repositories/worktree.py` for refresh and materialization; do not
  manage evidence worktrees with ad hoc destructive Git commands.
- Keep factual, file-referenced evidence under `derived/repo-analysis/`; reserve
  teaching, comparison, and interpretation for `docs/`.
- Distinguish static code inference from behavior verified by tests or runtime
  execution. Surface that boundary in every consuming page.

## Choose and Prepare the Revision

For an existing repository, start from the newest relevant registry entry and
compare only the requested scope:

```bash
./scripts/run-in-workspace.sh python scripts/repositories/worktree.py sync \
  <repo-slug>-<pinned-short-sha> \
  --path path/to/subsystem \
  --sparse path/to/subsystem
```

Repeat `--path` for every relevant file or directory. Omit it only when the
whole repository is genuinely in scope.

- `decision: reuse`: the scoped implementation is unchanged; reuse its raw
  record, derived evidence, and pinned checkout.
- `decision: defer`: relevant implementation changed, but the latest evidence
  snapshot is less than 14 days old. Keep using the pinned revision and do not
  scaffold or document the upstream candidate yet.
- `decision: new revision`: relevant implementation changed; initialize the
  revision-specific worktree created by the command.
- **new repository**: use only when no registry or source entry exists.

Do not create a new evidence revision for unrelated upstream changes, and do
not use `git pull` for freshness. The default promotion interval limits an
evidence chain to one new immutable revision every 14 days. Use
`--min-revision-interval-days N` to tune that interval, or
`--force-new-revision` only when a release, security fix, regression, or
explicit user request makes an immediate refresh necessary. A deferred result
is a successful freshness check, not permission to inspect the unpinned tip.

In a fresh workspace, check and materialize only the required registered
revisions with
`./scripts/bootstrap-external-repos.sh --status` and
`./scripts/bootstrap-external-repos.sh <repo-key> ...`.

Before writing, confirm the checkout is clean and capture its normalized
origin, provider, repository URL, full SHA, ref, and inspected date. A dirty
checkout is not reproducible from its SHA alone; accept it only when explicitly
required, record the dirty state, and use low confidence on consuming pages.

## Register Immutable Evidence

Use the scaffolder rather than hand-writing repository metadata:

```bash
./scripts/run-in-workspace.sh python scripts/repositories/init_source.py \
  external-repos/<repo> \
  --category <category> \
  --topic-path <topic/project> \
  --docs-path docs/<category>/<page>.md \
  --scope "Subsystem or question inspected" \
  --important-file "path/to/file.py::Why it matters"
```

The canonical artifacts are:

```text
raw/{category}/{topic-path}/{repo-slug}-codebase--{provider}-{short-sha}.md
derived/repo-analysis/{category}/{topic-path}/{repo-slug}/{full-sha}/
```

The scaffolder maintains the matching `sources.json` entry and
`docs/_data/code_repositories.json` registry record. Repository manifest entries
use `docs_paths` because one revision may support multiple pages. Every revision
requires `important-files.md`; add purpose-specific notes such as
`runtime-flow.md` or `module-map.md` only when they improve retrieval.

Keep derived notes factual. Record the exact command behind quantitative
codebase claims. Do not edit old revision evidence after it has been superseded;
create a new revision instead.

## Build the Evidence Map First

Before prose drafting, copy the sections from
[`repository-evidence-template.md`](repository-evidence-template.md) into
`important-files.md` or the directly supporting purpose-specific note.

For every important finding, record:

- consuming docs page;
- stable finding ID;
- repository-relative file and symbol;
- smallest useful start and optional end line;
- its position and direction in the end-to-end runtime flow, when applicable.

The evidence table is the handoff between code reading and prose drafting. Add
rows while investigating, not as a retrospective inventory. The code-link
checker requires every declared row to have a matching link in its consuming
page.

## Design the Page for Its Reader

Choose the page type explicitly:

- A **code-reading map** teaches navigation, runtime flow, modules, extension
  points, and failure surfaces.
- A **synthesis page** teaches a mechanism or comparison using evidence from
  one or more repositories.

Before drafting either type, record a small reader contract in the working
notes:

- intended audience and assumed prerequisites;
- the question the page answers;
- a one-sentence, code-free mental model;
- which behavior occurs offline, at load time, and at runtime;
- hardware, configuration, fallback, and verification limitations.

When several paths exist, select one supported branch for the first walkthrough.
State the configuration, hardware, workload, and entry/return boundaries that
select it; do not imply that an uninspected checkpoint or unexecuted backend
was observed. Explain materially different branches after the main path.

Draft concepts before implementation names. Expand important acronyms on first
use, explain why each stage exists, and introduce concrete files and symbols
only after the reader has a system-level map. For beginner-facing pages, keep
the initial path to these answers short:

1. What is the mechanism and why does it exist?
2. What data or state changes?
3. When does each change happen?
4. Where do the supported platforms diverge?
5. What was inferred statically versus verified at runtime?

Plan each rich element around a reader question and a teaching job. Use prose
for one idea, a table for exact mappings or platform differences, and a worked
trace for runtime state changes. When a flow diagram materially improves
hierarchy, branching, concurrency, or ownership, save its editable source
locally and link it from the page (`.mmd` for Mermaid or `.drawio` for Draw.io).
Prefer one reader question per visual; keep a symbol-heavy implementation map
separate from the first conceptual visual. Omit any diagram that only restates
nearby prose.
Put a compact input/state/output path before a long sequence diagram or module
inventory. A paper's architecture figure can explain design intent, but it is
not evidence that the repository implements the same call path.

Place a code-path figure immediately before the worked trace it prepares the
reader to follow. State the selected workload and branch first. Beside the
figure, give a short reading guide in visual order: entry, what each lane or
symbol represents, the important state handoff, and the result that leaves the
figure. Explain conditional roles and distinguish intermediate state from the
user-visible result. Do not make readers cross a long table to find this guide.
Use the same names in the figure and trace, and number the figure stages when
that helps readers match them to trace steps. Caption a source figure as design
intent or a synthesized figure as an explanation of the pinned code branch;
name any unverified configuration or runtime assumptions. Keep revision-aware
code links in the adjacent trace, including both producer and consumer of
shared state; the figure itself is orientation, not code evidence.

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

## Draft Repository-Backed Docs

Every new or materially updated repository-backed page must:

- cite every supporting raw revision and derived note in front matter;
- state each inspected full commit SHA near the top;
- set both `code_links: strict` and `code_evidence: strict`;
- appear in each supporting manifest entry's `docs_paths`;
- use concrete paths and symbols only when they help the reader navigate;
- state static-reading, runtime-validation, dependency, and hardware limits;
- follow confidence rules from `docs-front-matter.instructions.md`.

Keep generated files, vendored dependencies, lockfiles, and large fixtures
low-priority unless central to the question. Put detailed symbol indexes and
recommended code-reading paths after the conceptual explanation when the page
targets beginners.

## Link Directly to Inspected Code

Insert revision-aware links while drafting. Link the first meaningful
occurrence of important files and symbols, every numbered runtime-flow step,
and non-obvious implementation evidence. Repeated mentions and generic language
constructs may remain ordinary inline code.

Use a short verbatim code excerpt when the actual condition, state write, core
call, or return helps the reader stay on the page. Pair it with a precise code
link and a sentence explaining what to notice. Prefer decisive operations over
incidental setup; keep excerpts contiguous, mark omissions, and compare their
lines against the pinned checkout. When an evidence row declares an end line,
the matching anchor must include `data-code-end-line`.

Use this structure:

```html
<a class="code-link"
   href="../../../external-repos/vllm/vllm/v1/core/sched/scheduler.py#L248"
   data-code-repo="vllm-a0c092ee72c0"
   data-code-path="vllm/v1/core/sched/scheduler.py"
   data-code-line="248"
   data-code-end-line="312"><code>Scheduler.schedule()</code></a>
```

The `href` is relative from the docs page to the registered checkout. The
Jekyll layout converts its `data-code-*` metadata to a provider URL pinned to
the full revision. Always provide a start line; use an end line only for a
short, complete range. Never use an ordinary Markdown link such as
`[source](../../../external-repos/repo/file.py)`: it bypasses revision metadata,
breaks when another agent has not materialized that checkout, and cannot be
rewritten to the pinned provider URL. The code-link checker rejects this form.
Never commit a machine-specific absolute path.

## Complete the Workflow

1. Confirm the chosen mode and exact pinned revision.
2. Inspect only the declared scope and finish the evidence map.
3. Draft the page from the reader contract, then attach code evidence.
4. Update `sources.json`, navigation, related pages, and the chronological log
   only where repository rules require them.
5. Run `./scripts/lint-docs.sh`.

The normal lint is portable across agent workspaces: it checks repository
registry metadata and code-link structure but does not require every ignored
checkout to be materialized. Do not run markdownlint with a repository-wide
glob, because upstream Markdown under `external-repos/` is outside this wiki's
lint scope. When the checkout used for the current insight is available, add
the checkout-dependent verification separately:

```bash
./scripts/run-in-workspace.sh python scripts/checks/code_links.py --local
```

Completion requires agreement among raw metadata, derived metadata, the
manifest, and the code-repository registry; complete evidence coverage for
strict pages; an explicit verification boundary; and no unreported lint
failure.
