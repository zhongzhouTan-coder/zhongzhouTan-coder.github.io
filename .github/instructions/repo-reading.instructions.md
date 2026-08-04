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
- Use `scripts/kb-repo-worktree.py` for refresh and materialization; do not
  manage evidence worktrees with ad hoc destructive Git commands.
- Keep factual, file-referenced evidence under `derived/repo-analysis/`; reserve
  teaching, comparison, and interpretation for `docs/`.
- Distinguish static code inference from behavior verified by tests or runtime
  execution. Surface that boundary in every consuming page.

## Choose and Prepare the Revision

For an existing repository, start from the newest relevant registry entry and
compare only the requested scope:

```bash
./scripts/run-in-workspace.sh python scripts/kb-repo-worktree.py sync \
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
./scripts/run-in-workspace.sh python scripts/kb-init-repo-source.py \
  external-repos/<repo> \
  --category <category> \
  --docs-path docs/<category>/<page>.md \
  --scope "Subsystem or question inspected" \
  --important-file "path/to/file.py::Why it matters"
```

The canonical artifacts are:

```text
raw/{category}/{repo-slug}-codebase--{provider}-{short-sha}.md
derived/repo-analysis/{category}/{repo-slug}/{full-sha}/
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
- its position in the end-to-end runtime flow, when applicable.

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

Draft concepts before implementation names. Expand important acronyms on first
use, explain why each stage exists, and introduce concrete files and symbols
only after the reader has a system-level map. For beginner-facing pages, keep
the initial path to these answers short:

1. What is the mechanism and why does it exist?
2. What data or state changes?
3. When does each change happen?
4. Where do the supported platforms diverge?
5. What was inferred statically versus verified at runtime?

Use a table or prose when sufficient. When a flow diagram materially improves
the mechanism explanation, save its editable Mermaid source as a local `.mmd`
asset and link it from the page. Prefer one reader question per visual; keep a
symbol-heavy implementation map separate from the first conceptual visual.

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
short, complete range. Never commit a machine-specific absolute path.

## Complete the Workflow

1. Confirm the chosen mode and exact pinned revision.
2. Inspect only the declared scope and finish the evidence map.
3. Draft the page from the reader contract, then attach code evidence.
4. Update `sources.json`, navigation, related pages, and the chronological log
   only where repository rules require them.
5. Run `./scripts/lint-docs.sh`.

Completion requires agreement among raw metadata, derived metadata, the
manifest, and the code-repository registry; complete evidence coverage for
strict pages; an explicit verification boundary; and no unreported lint
failure.
