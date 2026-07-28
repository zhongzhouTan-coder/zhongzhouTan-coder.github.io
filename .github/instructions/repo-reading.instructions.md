---
description: "Use when ingesting, inspecting, or documenting a local code repository checkout. Enforces commit-pinned source records, ignored checkout handling, generated repo-analysis notes, and reusable code-reading docs pages."
applyTo: "external-repos/**/*, raw/**/*.md, derived/repo-analysis/**/*, docs/**/*.md, sources.json"
---

# Repository Reading Rules

Use this instruction when a source is a code repository rather than a paper, PDF, benchmark report, or web reference.

## Local Checkout Rules

- Local third-party checkouts live under `external-repos/`.
- `external-repos/` is ignored by git and must not be committed.
- Agents may inspect files under `external-repos/` with read-only tools such as `rg`, `sed`, `find`, package metadata readers, and test discovery.
- Do not edit, format, delete, or vendor files inside `external-repos/`.
- Always pin the exact commit SHA before writing docs. Branch names and tags are useful context but are not stable enough by themselves.
- If the checkout has local uncommitted changes, record that limitation in the raw source record and lower confidence unless the user explicitly says those changes are the source of truth.

## Canonical Source Record

Do not put the full repository under `raw/`. Instead, create a compact source record:

```text
raw/{category}/{repo-slug}-codebase--github.md
```

The source record should include:

- repository URL
- local checkout path under `external-repos/`
- commit SHA
- branch or tag, if relevant
- inspected date
- reading scope
- important upstream docs or entry files inspected
- known limitations, such as missing dependencies, generated files skipped, or local modifications

Use `sources.json` entries like:

```json
{
  "id": "github:owner/repo@commit-sha",
  "title": "Repo Name Codebase",
  "slug": "repo-name-codebase",
  "category": "frameworks",
  "kind": "repository",
  "raw_paths": ["raw/frameworks/repo-name-codebase--github.md"],
  "derived_path": "derived/repo-analysis/frameworks/repo-name/",
  "docs_path": "docs/frameworks/repo-name-code-reading.md",
  "status": "ingested"
}
```

## Derived Repo Analysis

Generated repository notes belong under:

```text
derived/repo-analysis/{category}/{repo-slug}/
```

Prefer small, purpose-specific notes:

- `inventory.md` - top-level directory map and package metadata.
- `entrypoints.md` - CLIs, servers, APIs, package exports, and configuration loading.
- `runtime-flow.md` - request, job, or command path through the core modules.
- `module-map.md` - major modules, responsibilities, and important ownership boundaries.
- `build-test-notes.md` - discovered build, test, and development commands.
- `important-files.md` - curated file list with why each file matters.

These files are derived from the pinned checkout and may be regenerated. Keep them factual and file-referenced; reserve synthesis and teaching for the docs page.

## Code-Reading Docs Page

Code-reading pages should optimize for navigation and runtime understanding, not paper-style contribution analysis. Use this structure unless the existing target page has a stronger established pattern:

```markdown
# Repo Name: Code Reading Map

**Repository:** [owner/repo]
**Commit:** [full or short SHA]
**Local checkout:** `external-repos/...`

**Related pages:** [Internal links]

## TL;DR
## What This Repo Is For
## How To Navigate It
## The Big Picture
## Main Runtime Flow
## Important Modules
## Extension Points
## Build, Test, and Run Surface
## Where It Breaks
## Reading Path
## Go Deeper
```

Rules:

- Cite the canonical raw source record in front matter.
- Cite generated repo-analysis notes in front matter when they directly support the page.
- State the inspected commit near the top of the body.
- Prefer relative links to existing internal pages and term pages.
- Use Mermaid diagrams for request flows, execution pipelines, and module relationships.
- Keep file references concrete. Mention paths and symbols when they are central to navigation.
- Treat generated files, vendored dependencies, lockfiles, and large fixtures as low-priority unless they are central to the repo's behavior.

## Related Page Maintenance

Before writing, read `docs/logs/index.md` and inspect likely related pages in the target category. Update related pages only when the repository reading changes an existing comparison, clarifies an implementation detail, or deserves a cross-link.

## Confidence

- Use `confidence: high` only when the page is directly backed by a clean pinned commit and the relevant files were inspected.
- Use `confidence: medium` when the page includes architectural synthesis across many files.
- Use `confidence: low` when the checkout has local modifications, missing generated artifacts, partial dependency context, or the analysis could not inspect the runtime path end to end.
