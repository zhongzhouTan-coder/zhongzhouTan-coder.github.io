---
description: "Use when ingesting, inspecting, or documenting a local code repository checkout. Enforces immutable commit-pinned source revisions, ignored checkout handling, generated evidence notes, and reusable repository-backed docs."
applyTo: "external-repos/**/*, raw/**/*.md, derived/repo-analysis/**/*, docs/**/*.md, sources.json"
---

# Repository Reading Rules

Use this instruction when a source is a code repository rather than a paper,
PDF, benchmark report, or web reference.

## Choose the Workflow Mode

Select one mode before reading code:

- **Reuse an existing revision** when its pinned commit still supports the
  requested docs change. Do not refresh evidence merely because checkout
  `HEAD` has moved.
- **Add a new revision** when the implementation evidence must be refreshed.
  Create a new immutable source record and analysis directory.
- **Add a new repository** when no source entry exists for the checkout.

Repository source records are immutable. Never rewrite an old record or its
derived revision to point at a different commit.

## Local Checkout Rules

- Local third-party checkouts live under `external-repos/`.
- `external-repos/` must be ignored by git and must not be committed.
- Inspect checkouts with read-only tools such as `rg`, `sed`, `find`, package
  metadata readers, and test discovery.
- Do not edit, format, delete, switch, pull, or vendor files inside
  `external-repos/` as part of docs work.
- Capture the repository URL, exact 40-character commit SHA, branch or tag,
  inspected date, and checkout state before writing docs.
- A dirty checkout is not reproducible from its commit SHA alone. Prefer a
  clean checkout. If dirty state is explicitly accepted, record it and use
  `confidence: low` on every consuming docs page.

Use these read-only checks:

```bash
git check-ignore external-repos/<repo>
git -C external-repos/<repo> remote get-url origin
git -C external-repos/<repo> rev-parse HEAD
git -C external-repos/<repo> branch --show-current
git -C external-repos/<repo> status --porcelain
```

## Immutable Source Revision

Each repository commit gets a distinct canonical record:

```text
raw/{category}/{repo-slug}-codebase--github-{short-sha}.md
```

`short-sha` is the first 12 characters of the full commit. The manifest keeps
`repo_slug` separate from the source `slug` so the meanings do not drift:

- `repo_slug`: stable repository name, such as `vllm`.
- `slug`: source title slug, normally `{repo-slug}-codebase`.

Use this machine-readable source-record template:

```markdown
---
kind: repository-source
repository_url: https://github.com/owner/repo
local_checkout: external-repos/repo/
commit: 0123456789abcdef0123456789abcdef01234567
ref: main
inspected: 2026-07-28
checkout_state: clean
---

# Repo Codebase Source Record

## Reading Scope

- Subsystem or question inspected.

## Important Entry Files

- `path/to/file.py` — why it matters.

## Limitations

- Static reading only; runtime behavior was not executed.
```

The corresponding `sources.json` entry is:

```json
{
  "id": "github:owner/repo@0123456789abcdef0123456789abcdef01234567",
  "title": "Repo Codebase",
  "slug": "repo-codebase",
  "repo_slug": "repo",
  "revision": "0123456789abcdef0123456789abcdef01234567",
  "category": "frameworks",
  "kind": "repository",
  "raw_paths": [
    "raw/frameworks/repo-codebase--github-0123456789ab.md"
  ],
  "derived_path": "derived/repo-analysis/frameworks/repo/0123456789abcdef0123456789abcdef01234567/",
  "docs_paths": [
    "docs/frameworks/repo-code-reading.md"
  ],
  "status": "ingested"
}
```

Repository entries use `docs_paths` because one revision may support multiple
pages, including pages outside the source record's canonical category.

## Derived Revision Evidence

Generated notes belong to the exact revision:

```text
derived/repo-analysis/{category}/{repo-slug}/{full-sha}/
```

Every revision requires `important-files.md` with this front matter:

```yaml
---
kind: repository-analysis
repository_id: github:owner/repo@0123456789abcdef0123456789abcdef01234567
commit: 0123456789abcdef0123456789abcdef01234567
source_record: raw/frameworks/repo-codebase--github-0123456789ab.md
generated: 2026-07-28
---
```

Add purpose-specific notes when they improve retrieval:

- `inventory.md` — top-level directory map and package metadata.
- `entrypoints.md` — CLIs, servers, APIs, exports, and configuration loading.
- `runtime-flow.md` — request, job, or command path through core modules.
- `module-map.md` — major modules, responsibilities, and ownership boundaries.
- `build-test-notes.md` — discovered build, test, and development commands.

Keep derived notes factual and file-referenced. Record the exact search or
counting command behind every quantitative codebase claim. Reserve teaching,
comparisons, and interpretation for the docs page.

## Repository-Backed Docs

First decide the page type:

- A **code-reading map** teaches navigation, runtime flow, modules, extension
  points, and failure surfaces.
- A **synthesis page** teaches a cross-cutting concept using examples from one
  or more repositories.

Code-reading maps should normally use:

```markdown
# Repo Name: Code Reading Map

**Repository:** [owner/repo]
**Commit:** [full SHA]

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

Synthesis pages may use a free-form structure. Both page types must:

- list every raw repository revision and directly supporting derived note in
  front matter;
- state every inspected full commit SHA near the top of the body;
- use concrete file paths and symbol names;
- add every consuming page to the manifest entry's `docs_paths`;
- keep generated files, vendored dependencies, lockfiles, and large fixtures
  low-priority unless central to the question.

When a flow diagram materially helps, save its Mermaid source as a local
`.mmd` asset and link it from the page. An inline rendered copy may accompany
the linked source.

## Related Page Maintenance

Before writing, read `docs/logs/index.md` and inspect likely related pages.
Update related pages only when the reading corrects a claim, changes a
comparison, or deserves a retrieval-oriented cross-link.

Update `docs/logs/index.md` when navigation changes. Append to
`docs/logs/log.md` when a source revision or reusable docs page is added or
materially refreshed. Do not add log noise for wording-only corrections.

## Confidence

Apply the most conservative matching rule:

- Use `confidence: high` for narrow, directly verified facts from a clean
  pinned revision when the relevant path was inspected end to end.
- Use `confidence: medium` for architectural synthesis, broad comparisons, or
  partially traced runtime flows, even when the revision is clean.
- Use `confidence: low` for dirty checkouts, missing generated artifacts,
  missing dependency context, or unverified runtime behavior.

## Canonical Workflow

1. Choose `reuse`, `new revision`, or `new repository`.
2. Validate checkout location and ignored status; capture remote, SHA, ref, and
   dirty state.
3. Reuse the exact existing revision or scaffold a new immutable revision:

   ```bash
   python3 scripts/kb-init-repo-source.py external-repos/<repo> \
     --category <category> \
     --docs-path docs/<category>/<page>.md \
     --scope "Subsystem or question inspected" \
     --important-file "path/to/file.py::Why it matters"
   ```

4. Inspect only the declared scope and write factual revision evidence.
5. Create or update all consuming docs pages and their `sources` front matter.
6. Update navigation and the chronological log when required.
7. Run:

   ```bash
   python3 scripts/kb-normalize-source-name.py
   python3 scripts/kb-check-integrity.py
   ./scripts/lint-docs.sh
   npx markdownlint-cli2
   ```

## Completion Checklist

- [ ] Workflow mode was chosen explicitly.
- [ ] Checkout is beneath ignored `external-repos/`.
- [ ] Remote, full SHA, ref, inspected date, and checkout state were captured.
- [ ] Exact immutable raw revision exists or was reused.
- [ ] Manifest revision, raw metadata, and derived metadata agree.
- [ ] `important-files.md` exists under the full-SHA revision directory.
- [ ] Commands supporting quantitative claims are recorded in derived notes.
- [ ] Every consuming page appears in `docs_paths`.
- [ ] Every consuming page cites the raw record and a derived revision file.
- [ ] Every consuming page states the full SHA near the top.
- [ ] Confidence follows the most conservative applicable rule.
- [ ] Navigation and log updates were made only when warranted.
- [ ] Source normalization, integrity checks, and docs lint all pass.
