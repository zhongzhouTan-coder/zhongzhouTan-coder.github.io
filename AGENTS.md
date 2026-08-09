
# Personal Docs Routing

This repository is a markdown knowledge base.

## Workspace Environment

- Before running repository tools in a fresh checkout or agent environment, run
  `./scripts/bootstrap-workspace.sh` once. It installs the locked Python, Node.js,
  and Ruby dependencies under this workspace instead of relying on mutable
  agent-global packages.
- Run Python, Node.js, npm, Bundler, and Jekyll commands through
  `./scripts/run-in-workspace.sh <command> [args...]`. Repository shell entry
  points such as `./scripts/lint-docs.sh` and `./scripts/serve-local.sh` already
  prefer the workspace environment.
- For repository code reading in a fresh environment, run
  `./scripts/bootstrap-external-repos.sh --status`, then materialize the needed
  registry keys with `./scripts/bootstrap-external-repos.sh <repo-key> ...`.
  Running it without keys materializes every pinned revision. Keep this step
  separate from the normal dependency bootstrap because external repositories
  can be large.
- If a workspace dependency is missing or stale, rerun the bootstrap script. Do
  not install packages into the agent's global environment as a workaround.
- Run documentation lint through `./scripts/lint-docs.sh` (or run
  `npx markdownlint-cli2` with no extra file glob). Do not pass repository-wide
  globs such as `**/*.md`: `external-repos/` contains third-party Markdown that
  is read-only evidence, not part of this knowledge base's lint scope.

## Source Locations

- `raw/` stores original source files. Never modify them.
- `derived/pdf-markdown/` stores generated Markdown extracted from PDFs and other document formats. These files are derived from `raw/` sources and may be regenerated.
- `derived/web-markdown/` stores generated Markdown extracted from immutable HTML snapshots under `raw/`. These files may be regenerated from the matching snapshot.
- `external-repos/` stores local third-party repository checkouts used for code reading. This directory is ignored by git. Agents may inspect files there but must not edit them. In docs, never link a checkout file with ordinary Markdown; use the revision-aware `code-link` anchor required by `.github/instructions/repo-reading.instructions.md` so links remain valid when another agent has not materialized the checkout.
- `derived/repo-analysis/` stores generated code-reading notes extracted from local repository checkouts. These files are derived from pinned repository commits and may be regenerated.
- `docs/` stores AI-maintained markdown knowledge pages by topic.
- `docs/logs/index.md` is the wiki index.
- `docs/logs/log.md` is the chronological change log.

## Instruction Files

- For docs page front matter and topic placement, read `.github/instructions/docs-front-matter.instructions.md`.
- For paper-insight page body structure, read `.github/instructions/docs-content-structure.instructions.md`.
- For code repository reading pages and local checkout handling, read `.github/instructions/repo-reading.instructions.md`.
- For term glossary pages, read `.github/instructions/docs-terms.instructions.md`.
- For source categories, raw/derived naming, and manifest maintenance, read `.github/instructions/source-organization.instructions.md`.
- For web capture, rendering, provenance, and synthesis, read `.github/instructions/web-source.instructions.md`.
- For `docs/logs/index.md` and `docs/logs/log.md`, read `.github/instructions/logs-maintenance.instructions.md`.

## Intent-First Workflow Routing

- Follow the user's requested outcome before classifying the input type. The
  presence of a paper, repository URL or checkout, web page, Markdown file, or
  other source does not by itself authorize ingesting it into the knowledge
  base.
- Select source ingest only when the user explicitly asks to add, ingest,
  archive, capture, or integrate the source into this knowledge base, or when
  they clearly ask for a durable knowledge page backed by that source.
- When a source is supplied only as context for another task, use it only for
  that task. Do not copy it into `raw/`, create derived artifacts, update
  `sources.json`, synthesize a docs page, or change the wiki index or log unless
  the user also requests knowledge-base integration.
- Route ordinary repository work, such as coding, review, planning, debugging,
  transformation, or one-off analysis, to the instructions for that task. It
  does not become a wiki workflow merely because the input is a paper,
  repository, web page, or Markdown file.
- Route knowledge-base questions and maintenance requests to the relevant wiki
  query or maintenance workflow without treating them as new-source ingest.
- If the requested outcome is genuinely ambiguous and ingestion would create
  durable repository changes, prefer the narrower non-ingest interpretation.
  Ask for clarification only when the task cannot be completed safely under
  that interpretation.

## Ingest Workflow Triggers

The triggers below apply only after the request has passed the intent gate and
has been classified as source ingest.

- When a new source is added for knowledge-base ingestion: read the source from `raw/`, create or update the relevant `docs/` page, update `docs/logs/index.md`, and append to `docs/logs/log.md`.
- When a new PDF source is added: choose its canonical category from `kb-categories.json`, normalize its filename with the policy in `.github/instructions/source-organization.instructions.md`, add or update `sources.json`, convert it to Markdown with `./scripts/run-in-workspace.sh python scripts/ingestion/mineru_to_markdown.py --mode precise`, then read the generated Markdown to write the insight page. Save generated Markdown under the matching category subdirectory below `derived/pdf-markdown/`, for example `./scripts/run-in-workspace.sh python scripts/ingestion/mineru_to_markdown.py --mode precise raw/algorithms/example--arxiv-0000.00000v1.pdf --output-dir derived/pdf-markdown/algorithms`. Treat the generated Markdown as the primary source for docs synthesis. Keep the original PDF content unchanged, and cite the canonical `raw/` PDF path in docs front matter.
- When a new web page source is added: capture it with `scripts/ingestion/web_to_markdown.mjs`, preserve the immutable HTML and metadata under `raw/`, use the generated `derived/web-markdown/` file for synthesis, then cite all three artifacts from the docs page before changing the manifest status from `captured` to `ingested`.
- For any new or existing repository-backed page, follow the canonical
  `reuse` / `new revision` / `new repository` workflow in
  `.github/instructions/repo-reading.instructions.md`. For an existing
  repository, refresh the shared cache and compare the latest upstream commit
  against the pinned revision with `scripts/repositories/worktree.py sync`, scoped
  to the subsystem being inspected. Reuse the recorded revision when that
  scope is unchanged. When relevant implementation changed, honor the sync
  command's revision interval: defer the refresh while the latest evidence
  snapshot is too recent, and create a new immutable revision only when it is
  eligible. Override the interval only for an explicitly urgent or
  release-specific inspection. Never pull into a checkout that already
  supports pinned evidence.
- When adding a new paper or docs page, inspect existing related pages from `docs/logs/index.md` and the target topic folder. Update those pages when the new material changes the landscape, supersedes an older claim, adds a needed comparison, or should be linked as related context.
- Do not rewrite related pages just because a new page exists. Keep related-page edits scoped to factual corrections, changed confidence, cross-links, short comparison notes, or synthesis that improves retrieval.
- If MinerU API conversion is unavailable, rate-limited, blocked by SSL/network issues, or clearly incomplete, record that limitation in the resulting docs page and use a fallback extractor only after checking for generated Markdown.
- When answering knowledge-base questions: read `docs/logs/index.md` first, inspect the relevant `docs/` pages, synthesize from the docs, and save valuable reusable answers as docs pages.
- Prefer updating existing pages over creating duplicates.
- Use internal links when related pages already exist.
- If a docs page references sibling assets such as images or Draw.io files, prefer a folder-backed page at `topic/index.md`.
- **When creating or updating a paper insight page:** identify key technical terms that appear across multiple papers (e.g., "microbatch", "pipeline bubble", "KV cache"). For each term, check `docs/terms/{term-slug}.md`. If missing, create a term page following `.github/instructions/docs-terms.instructions.md`. If present, add the new paper to the term's `appears_in` list and "Where It Appears" section. Link the first meaningful in-content occurrence of each term to its term page using ordinary Markdown links. Update `docs/terms/index.md` when adding a new term.

## Docs Organization

- **Keep categories organized with subcategory folders.** Inside each topic category (`docs/frameworks/`, `docs/algorithms/`, `docs/benchmarks/`, `docs/training/`, `docs/hardware/`), group related pages into per-project or per-theme subfolders (for example `docs/frameworks/vllm/`, `docs/algorithms/flashattention/`, `docs/benchmarks/agent-eval/`). Give each subcategory a hub page at `<subfolder>/index.md` that lists its pages, and group links under matching subheadings in the category `index.md`.
- **Keep pages linked to avoid orphans.** Every docs page must be reachable from a category index, a subcategory hub, or `docs/logs/index.md`; the lint reports unreferenced pages as orphans.
- **Do not move a page across categories casually.** `kb-check-integrity.py` requires each `sources.json` `docs_path` to stay under `docs/{category}/`, and the page's raw/derived sources live under `raw/{category}/` and `derived/pdf-markdown/{category}/`. Relocating a page to another category therefore requires moving its raw and derived sources and updating the manifest `category`. Prefer reorganizing within the page's existing category.
- **When moving or renaming any docs page:** use `git mv` to preserve history, recompute relative links in the moved page and in every page that links to it, update `sources.json` (`docs_path` or `docs_paths`), update the category index and subcategory hub, update `docs/logs/index.md`, append `docs/logs/log.md`, then run `./scripts/lint-docs.sh`.

## Diagram Conventions

- **Original source figures are the first choice.** Before creating any visual,
  inspect the paper's extracted figures and the captured web page's images. For
  every section except **The Landscape**, use a suitable original paper/web
  figure whenever one exists; do not replace it with Mermaid, Draw.io, or an
  AI-generated visual merely for stylistic consistency.
- **Preserve selected source figures locally.** Copy or download each selected
  original figure into the consuming page's `assets/` directory, use a
  descriptive filename, embed the local relative path, and add a caption that
  identifies and links to the source. Never rely on a remote image URL as the
  page's only copy, and never modify files under `raw/`.
- **The Landscape always uses Mermaid.** Represent the evolutionary tree of
  prior work, siblings, and the current method as a locally saved `.mmd` file.
  This is the standard exception to the original-figure-first rule.
- **Synthesized visuals are fallback-only outside The Landscape.** If no usable
  original source figure exists, prefer a table or prose. Create a Mermaid,
  Draw.io, Excalidraw, or AI-generated explanatory visual only when the user
  explicitly requests one or when the mechanism cannot be understood clearly
  without it; label the caption as a synthesized explanation rather than a
  source figure.
- **Always save editable diagrams locally** in the docs folder (for example,
  `docs/topic/assets/`). Use `.mmd` for the Landscape and any approved simple
  fallback flow, `.drawio` for approved complex architecture diagrams, and
  `.excalidraw` for approved hand-drawn explainers.
- **Avoid opening diagrams in the browser.** Generate diagram content and save
  it directly to a local file. Do not use tools that open a browser tab.
- **Link all editable diagram files** from the docs page so they are
  discoverable and reusable.

Run `./scripts/lint-docs.sh` after docs or logs changes.
