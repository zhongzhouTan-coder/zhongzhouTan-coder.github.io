---
description: "Use for repository-backed evidence and docs. Route to revision, evidence, runtime, code-link, and completion rules."
applyTo: "external-repos/**/*, raw/**/*codebase--*.md, derived/repo-analysis/**/*, docs/**/*.md, sources.json"
---

# Repository Reading Instructions

Apply these rules when a code repository is the evidence for the current task.
A path match on `docs/**/*.md` or `sources.json` alone does not make a page
repository-backed. General wiki and front matter rules remain in `AGENTS.md`
and [docs-front-matter.instructions.md](docs-front-matter.instructions.md).

Read the relevant references before working:

- [Revision and registration](references/repo-revisions.md): immutable commits,
  sync decisions, source scaffolding.
- [Evidence map and reader contract](references/repo-reader.md): finding IDs,
  audience, scope, conceptual explanation.
- [Mechanisms and request round trips](references/repo-runtime.md): state,
  handoffs, runtime traces, diagrams.
- [Drafting and code links](references/repo-code-links.md): page metadata,
  revision-aware anchors, excerpts.
- [Completion](references/repo-completion.md): integration and verification.
- [Evidence map template](repository-evidence-template.md): required rows
  and round-trip evidence.

For a new repository-backed page, read every reference above. For a scoped
change, read the relevant references plus completion. Keep old evidence pinned
to its recorded commit; use the revision workflow before inspecting upstream.
