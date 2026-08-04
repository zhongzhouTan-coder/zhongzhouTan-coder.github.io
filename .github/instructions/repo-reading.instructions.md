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
  requested docs change. Check the latest upstream commit first, but do not
  refresh evidence merely because unrelated code changed.
- **Add a new revision** when the implementation evidence must be refreshed.
  Create a new immutable source record and analysis directory.
- **Add a new repository** when no source entry exists for the checkout.

Repository source records are immutable. Never rewrite an old record or its
derived revision to point at a different commit.

## Local Checkout Rules

- Local third-party checkouts live under `external-repos/`.
- `external-repos/` must be ignored by git and must not be committed.
- Store shared bare object databases under
  `external-repos/.cache/{provider}/{owner}/{repo}.git`. Materialized revision
  worktrees share this object database instead of cloning full history again.
- Treat a registered `local_checkout` as a stable materialization path. It may
  be absent after an inactive worktree is retired; its immutable raw record,
  derived evidence, manifest entry, registry metadata, and protected Git ref
  remain canonical.
- Inspect checkouts with read-only tools such as `rg`, `sed`, `find`, package
  metadata readers, and test discovery.
- Do not edit, format, switch, pull, or vendor files inside a materialized
  worktree as part of docs work. Use `scripts/kb-repo-worktree.py` for shared
  cache refresh, revision materialization, and explicit retirement; do not
  manage those worktrees with ad hoc destructive commands.
- Capture the origin clone URL, normalized provider and repository web URL,
  exact 40-character commit SHA, branch or tag, inspected date, and checkout
  state before writing docs.
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

## Latest-First Refresh Without Revision Sprawl

For an existing repository, identify the newest registry key that supports the
question and run a scoped freshness check before inspecting code:

```bash
./scripts/run-in-workspace.sh python scripts/kb-repo-worktree.py sync \
  <repo-slug>-<pinned-short-sha> \
  --path path/to/subsystem \
  --sparse path/to/subsystem
```

The command fetches upstream into a partial bare cache, protects the pinned
commit under `refs/kb/revisions/`, and compares the pinned revision with the
upstream default branch. Its decision controls the workflow:

- `decision: reuse` means the inspected paths are unchanged. Continue with the
  existing raw record and derived evidence even if upstream has newer commits.
- `decision: new revision` means relevant files changed. The command creates a
  detached shared-object worktree at a revision-specific path; pass that path
  to `scripts/kb-init-repo-source.py` before writing new evidence.

Repeat `--path` for every directory or file in scope. Omit it only when the
whole repository is genuinely relevant. Pass `--remote-ref` when the upstream
default branch is neither `main` nor `master`, or cannot be resolved from
`origin/HEAD`.

Do not use `git pull` for freshness. Pulling mutates a checkout that may already
back immutable evidence. Fetching the shared cache separates discovery of new
upstream commits from the decision to register a new evidence revision.

Retire a clean inactive shared worktree without deleting its evidence:

```bash
./scripts/run-in-workspace.sh python scripts/kb-repo-worktree.py retire \
  <repo-slug>-<short-sha>
```

Restore it later at the same registered path:

```bash
./scripts/run-in-workspace.sh python scripts/kb-repo-worktree.py materialize \
  <repo-slug>-<short-sha>
```

In a fresh agent or checkout environment, inspect or hydrate the complete
registered repository workspace with:

```bash
./scripts/bootstrap-external-repos.sh --status
./scripts/bootstrap-external-repos.sh <repo-key> [<repo-key> ...]
./scripts/bootstrap-external-repos.sh  # all registered revisions
```

Treat `docs/_data/code_repositories.json` as the portable lock table. Do not
create a second checkout manifest: the registry already records the stable
local path, canonical repository URL, provider, and exact revision used by
code links. Hydration is deliberately separate from `bootstrap-workspace.sh`
because most docs tasks do not need every large third-party codebase.

Retirement is optional and must be intentional. The helper refuses dirty,
wrong-revision, or independently cloned directories, and protects the commit
before removing a shared worktree. `check-code-links.py --local` requires all
referenced worktrees to be materialized; the normal docs lint validates remote
pinned links without requiring inactive local worktrees.

## Immutable Source Revision

Each repository commit gets a distinct canonical record:

```text
raw/{category}/{repo-slug}-codebase--{provider}-{short-sha}.md
```

`short-sha` is the first 12 characters of the full commit. The manifest keeps
`repo_slug` separate from the source `slug` so the meanings do not drift:

- `repo_slug`: stable repository name, such as `vllm`.
- `slug`: source title slug, normally `{repo-slug}-codebase`.

Use this machine-readable source-record template:

```markdown
---
kind: repository-source
provider: gitcode
clone_url: git@gitcode.com:owner/repo.git
repository_url: https://gitcode.com/owner/repo
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
  "id": "gitcode:owner/repo@0123456789abcdef0123456789abcdef01234567",
  "title": "Repo Codebase",
  "slug": "repo-codebase",
  "repo_slug": "repo",
  "revision": "0123456789abcdef0123456789abcdef01234567",
  "category": "frameworks",
  "kind": "repository",
  "provider": "gitcode",
  "repository_url": "https://gitcode.com/owner/repo",
  "raw_paths": [
    "raw/frameworks/repo-codebase--gitcode-0123456789ab.md"
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
repository_id: gitcode:owner/repo@0123456789abcdef0123456789abcdef01234567
commit: 0123456789abcdef0123456789abcdef01234567
source_record: raw/frameworks/repo-codebase--gitcode-0123456789ab.md
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

Use revision-aware code links as the default drafting process, not as a
cleanup pass after the analysis is written:

1. Register the pinned checkout before drafting the page.
2. While collecting evidence, record the repository-relative file path,
   symbol, and smallest useful line range for each important finding.
3. Insert the code link at the first meaningful occurrence of that file or
   symbol as the finding is written.
4. Before completion, enable `code_links: strict` and run the repository code
   link checker through `./scripts/lint-docs.sh`.

New repository-backed pages must use `code_links: strict` from their first
draft. When materially updating an older repository-backed page, migrate its
concrete file references and enable strict mode as part of the update. On
strict pages, every concrete repository filename in inline code must use a
validated `code-link` anchor. Rewrite bare extensions, generated output
patterns, and hypothetical filenames as prose; they are not navigable
repository evidence.

### Link Directly to Inspected Code

Make important file paths and symbols navigable with a revision-aware code
anchor instead of leaving readers to search for inline-code paths manually:

```html
<a class="code-link"
   href="../../../external-repos/vllm/vllm/v1/core/sched/scheduler.py#L248"
   data-code-repo="vllm-a0c092ee72c0"
   data-code-path="vllm/v1/core/sched/scheduler.py"
   data-code-line="248"
   data-code-end-line="312"><code>Scheduler.schedule()</code></a>
```

`scripts/kb-init-repo-source.py` registers each immutable revision in
`docs/_data/code_repositories.json` from the checkout's normalized `origin`.
The registry key must identify the revision, and its provider, web URL, and
full `revision` must agree with the raw source record and `sources.json` entry.
Keep `local_checkout` beneath `external-repos/`.

The `href` must be a relative path from the Markdown page to the registered
checkout, so a Markdown preview opens the local dependency. The Jekyll layout
uses the `data-code-*` metadata and registered provider to replace that
destination with a GitHub or GitCode `blob` URL pinned to the full commit.
Local Jekyll serving follows the web behavior and also renders the remote URL.
Never put a machine-specific absolute workspace path in a docs page or
committed config.

Use code links for:

- the first meaningful occurrence of every important file or symbol;
- each step in a runtime flow or recommended reading path;
- exact implementation evidence behind a non-obvious architectural claim.

Treat repository-specific class, function, method, constant, command, and
configuration names as link candidates when a reader would reasonably search
for their implementation. Link the symbol label to the defining or most
relevant use site. Generic language keywords, API concepts, generated names,
and repeated mentions do not need links. If a directory or module name matters
but has no single source line, link its most useful entry-point file and name
that entry point in the label rather than creating a non-navigable path.

Always provide a start line. Add `end_line` when a short range contains the
complete implementation being discussed. Keep ordinary backticks for repeated
mentions that do not benefit from another link.

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

1. For an existing repository, run the scoped latest-first refresh. Use its
   result to choose `reuse` or `new revision`; choose `new repository` only
   when no source entry exists.
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
- [ ] Latest upstream was checked against the relevant paths, or an offline or
      explicitly pinned limitation was recorded.
- [ ] Checkout is beneath ignored `external-repos/`.
- [ ] Remote, full SHA, ref, inspected date, and checkout state were captured.
- [ ] Exact immutable raw revision exists or was reused.
- [ ] Manifest revision, raw metadata, and derived metadata agree.
- [ ] `important-files.md` exists under the full-SHA revision directory.
- [ ] Commands supporting quantitative claims are recorded in derived notes.
- [ ] Every consuming page appears in `docs_paths`.
- [ ] Every consuming page cites the raw record and a derived revision file.
- [ ] Every consuming page states the full SHA near the top.
- [ ] Important files, symbols, and runtime-flow steps use revision-aware code links.
- [ ] Confidence follows the most conservative applicable rule.
- [ ] Navigation and log updates were made only when warranted.
- [ ] Source normalization, integrity checks, and docs lint all pass.
