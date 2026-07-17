
# Personal Docs Routing

This repository is a markdown knowledge base.

## Source Locations

- `raw/` stores original source files. Never modify them.
- `docs/` stores AI-maintained markdown knowledge pages by topic.
- `docs/logs/index.md` is the wiki index.
- `docs/logs/log.md` is the chronological change log.

## Instruction Files

- For docs page front matter and topic placement, read `.github/instructions/docs-front-matter.instructions.md`.
- For paper-insight page body structure, read `.github/instructions/docs-content-structure.instructions.md`.
- For `docs/logs/index.md` and `docs/logs/log.md`, read `.github/instructions/logs-maintenance.instructions.md`.

## Workflow Triggers

- When a new source is added: read the source from `raw/`, create or update the relevant `docs/` page, update `docs/logs/index.md`, and append to `docs/logs/log.md`.
- When answering knowledge-base questions: read `docs/logs/index.md` first, inspect the relevant `docs/` pages, synthesize from the docs, and save valuable reusable answers as docs pages.
- Prefer updating existing pages over creating duplicates.
- Use internal links when related pages already exist.
- If a docs page references sibling assets such as images or Draw.io files, prefer a folder-backed page at `topic/index.md`.

Run `./scripts/lint-docs.sh` after docs or logs changes.
