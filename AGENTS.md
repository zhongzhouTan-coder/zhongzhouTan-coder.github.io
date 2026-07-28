
# Personal Docs Routing

This repository is a markdown knowledge base.

## Source Locations

- `raw/` stores original source files. Never modify them.
- `derived/pdf-markdown/` stores generated Markdown extracted from PDFs and other document formats. These files are derived from `raw/` sources and may be regenerated.
- `derived/web-markdown/` stores generated Markdown extracted from immutable HTML snapshots under `raw/`. These files may be regenerated from the matching snapshot.
- `external-repos/` stores local third-party repository checkouts used for code reading. This directory is ignored by git. Agents may inspect files there but must not edit them.
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

## Workflow Triggers

- When a new source is added: read the source from `raw/`, create or update the relevant `docs/` page, update `docs/logs/index.md`, and append to `docs/logs/log.md`.
- When a new PDF source is added: choose its canonical category from `kb-categories.json`, normalize its filename with the policy in `.github/instructions/source-organization.instructions.md`, add or update `sources.json`, convert it to Markdown with `scripts/mineru-agent-to-markdown.py --mode precise`, then read the generated Markdown to write the insight page. Save generated Markdown under the matching category subdirectory below `derived/pdf-markdown/`, for example `scripts/mineru-agent-to-markdown.py --mode precise raw/algorithms/example--arxiv-0000.00000v1.pdf --output-dir derived/pdf-markdown/algorithms`. Treat the generated Markdown as the primary source for docs synthesis. Keep the original PDF content unchanged, and cite the canonical `raw/` PDF path in docs front matter.
- When a new web page source is added: capture it with `scripts/web-source-to-markdown.mjs`, preserve the immutable HTML and metadata under `raw/`, use the generated `derived/web-markdown/` file for synthesis, then cite all three artifacts from the docs page before changing the manifest status from `captured` to `ingested`.
- For any new or existing repository-backed page, follow the canonical
  `reuse` / `new revision` / `new repository` workflow in
  `.github/instructions/repo-reading.instructions.md`. Do not automatically
  refresh evidence to checkout `HEAD`; reuse the recorded revision unless the
  implementation evidence is intentionally being updated.
- When adding a new paper or docs page, inspect existing related pages from `docs/logs/index.md` and the target topic folder. Update those pages when the new material changes the landscape, supersedes an older claim, adds a needed comparison, or should be linked as related context.
- Do not rewrite related pages just because a new page exists. Keep related-page edits scoped to factual corrections, changed confidence, cross-links, short comparison notes, or synthesis that improves retrieval.
- If MinerU API conversion is unavailable, rate-limited, blocked by SSL/network issues, or clearly incomplete, record that limitation in the resulting docs page and use a fallback extractor only after checking for generated Markdown.
- When answering knowledge-base questions: read `docs/logs/index.md` first, inspect the relevant `docs/` pages, synthesize from the docs, and save valuable reusable answers as docs pages.
- Prefer updating existing pages over creating duplicates.
- Use internal links when related pages already exist.
- If a docs page references sibling assets such as images or Draw.io files, prefer a folder-backed page at `topic/index.md`.
- **When creating or updating a paper insight page:** identify key technical terms that appear across multiple papers (e.g., "microbatch", "pipeline bubble", "KV cache"). For each term, check `docs/terms/{term-slug}.md`. If missing, create a term page following `.github/instructions/docs-terms.instructions.md`. If present, add the new paper to the term's `appears_in` list and "Where It Appears" section. Link the first meaningful in-content occurrence of each term to its term page using ordinary Markdown links. Update `docs/terms/index.md` when adding a new term.

## Diagram Conventions

- **Always save diagrams locally** in the docs folder (e.g., `docs/topic/assets/`) — never only open them in a browser. Use `.drawio` for complex architecture diagrams, `.mmd` for Mermaid flowcharts, and `.excalidraw` for hand-drawn explainers.
- **Avoid opening diagrams in the browser.** Generate the diagram content and save it directly to a local file via `create_file`. Do not use tools that open a browser tab (e.g., `open_drawio_*`); use the MCP tools only to obtain the XML/JSON content, then write it to disk.
- **Mermaid for simple flows.** Use Mermaid (`.mmd`) for straightforward flowcharts, decision trees, and sequence diagrams — keep Draw.io for diagrams that need swimlanes, rich shapes, containers, custom positioning, or industry-specific icons.
- **Link all diagram files** from the docs page so they are discoverable and editable.

Run `./scripts/lint-docs.sh` after docs or logs changes.
