
# Personal Docs Routing

This repository is a markdown knowledge base.

## Source Locations

- `raw/` stores original source files. Never modify them.
- `derived/pdf-markdown/` stores generated Markdown extracted from PDFs and other document formats. These files are derived from `raw/` sources and may be regenerated.
- `docs/` stores AI-maintained markdown knowledge pages by topic.
- `docs/logs/index.md` is the wiki index.
- `docs/logs/log.md` is the chronological change log.

## Instruction Files

- For docs page front matter and topic placement, read `.github/instructions/docs-front-matter.instructions.md`.
- For paper-insight page body structure, read `.github/instructions/docs-content-structure.instructions.md`.
- For `docs/logs/index.md` and `docs/logs/log.md`, read `.github/instructions/logs-maintenance.instructions.md`.

## Workflow Triggers

- When a new source is added: read the source from `raw/`, create or update the relevant `docs/` page, update `docs/logs/index.md`, and append to `docs/logs/log.md`.
- When a new PDF source is added: convert it to Markdown first with `scripts/mineru-agent-to-markdown.py --mode precise`, then read the generated Markdown to write the insight page. Save generated Markdown under a category subdirectory below `derived/pdf-markdown/` instead of the top level; choose the category from the source topic or target docs topic, for example `scripts/mineru-agent-to-markdown.py --mode precise raw/example.pdf --output-dir derived/pdf-markdown/ai`. Treat the generated Markdown as the primary source for docs synthesis. Keep the original PDF in `raw/` unchanged, and cite the original `raw/` PDF path in docs front matter.
- If MinerU API conversion is unavailable, rate-limited, blocked by SSL/network issues, or clearly incomplete, record that limitation in the resulting docs page and use a fallback extractor only after checking for generated Markdown.
- When answering knowledge-base questions: read `docs/logs/index.md` first, inspect the relevant `docs/` pages, synthesize from the docs, and save valuable reusable answers as docs pages.
- Prefer updating existing pages over creating duplicates.
- Use internal links when related pages already exist.
- If a docs page references sibling assets such as images or Draw.io files, prefer a folder-backed page at `topic/index.md`.

## Diagram Conventions

- **Always save diagrams locally** in the docs folder (e.g., `docs/topic/assets/`) — never only open them in a browser. Use `.drawio` for complex architecture diagrams, `.mmd` for Mermaid flowcharts, and `.excalidraw` for hand-drawn explainers.
- **Avoid opening diagrams in the browser.** Generate the diagram content and save it directly to a local file via `create_file`. Do not use tools that open a browser tab (e.g., `open_drawio_*`); use the MCP tools only to obtain the XML/JSON content, then write it to disk.
- **Mermaid for simple flows.** Use Mermaid (`.mmd`) for straightforward flowcharts, decision trees, and sequence diagrams — keep Draw.io for diagrams that need swimlanes, rich shapes, containers, custom positioning, or industry-specific icons.
- **Link all diagram files** from the docs page so they are discoverable and editable.

Run `./scripts/lint-docs.sh` after docs or logs changes.
