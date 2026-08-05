---
description: "Use when adding, renaming, classifying, or validating source files under raw/, generated PDF or web markdown, or generated code-reading notes. Maintains canonical categories, readable source filenames, and sources.json manifest entries."
applyTo: "raw/**/*, derived/pdf-markdown/**/*, derived/web-markdown/**/*, derived/repo-analysis/**/*, sources.json, kb-categories.json"
---

# Source Organization Rules

Use this instruction whenever source files or generated extraction files are added, moved, or renamed.

## Canonical Categories

Use the category keys in `kb-categories.json` as the source of truth:

- `algorithms` — inference algorithms, attention variants, decoding methods, kernels, scheduling, and mathematical procedures.
- `benchmarks` — benchmark construction, domains, task generation, metrics, evaluation harnesses, and empirical benchmark results.
- `frameworks` — LLM serving systems, programming frameworks, orchestration systems, and runtime architecture.
- `hardware` — numerics, quantization, accelerator features, precision formats, and hardware-aware implementation details.
- `training` — pretraining, fine-tuning, RL, scaling, transfer learning, model-parallel training, and generalization.
- `codex` — Codex product notes, agent workflows, skills, hooks, and local automation behavior.

Do not create new top-level `raw/`, `derived/pdf-markdown/`, `derived/repo-analysis/`, or `docs/` categories unless the existing categories would make the material hard to retrieve.

## Naming Policy

PDF source names must be deterministic and readable:

```text
raw/{category}/{short-title-slug}--{source-id}.pdf
```

Markdown source names should use:

```text
raw/{category}/{short-title-slug}.md
```

Repository source records should use:

```text
raw/{category}/{repo-slug}-codebase--{provider}-{short-sha}.md
```

Generated Markdown should use:

```text
derived/pdf-markdown/{category}/{short-title-slug}.md
```

When a generated extraction needs sibling assets, use a folder-backed primary Markdown file:

```text
derived/pdf-markdown/{category}/{short-title-slug}/{short-title-slug}.md
```

Generated repository analysis should use:

```text
derived/repo-analysis/{category}/{repo-slug}/{full-sha}/
```

Immutable web captures should use:

```text
raw/{category}/{short-title-slug}--web-{capture-date}-{short-sha256}.html
raw/{category}/{short-title-slug}--web-{capture-date}-{short-sha256}.metadata.json
derived/web-markdown/{category}/{short-title-slug}--web-{capture-date}-{short-sha256}.md
derived/web-markdown/{category}/{short-title-slug}--web-{capture-date}-{short-sha256}.assets/
```

Rules:

- Use lowercase ASCII slugs.
- Use hyphens between words.
- Keep arXiv versions when the filename already has one.
- Use a short stable source suffix such as `arxiv-2205.14135v2`, `paper`, `web`, `github`, or a publisher label.
- Do not use bare arXiv IDs, spaces, pipes, or punctuation-heavy filenames.
- Do not store full repository checkouts under `raw/`; store them under ignored `external-repos/` and keep only the pinned source record under `raw/`.
- Treat each repository commit as an immutable source revision. Use the first
  12 commit characters in the raw filename and the full 40-character commit
  in the derived directory. Derive `{provider}` from the checkout's `origin`;
  supported values are `github` and `gitcode`.
- Treat each web capture as an immutable source revision. Hash the rendered
  HTML with SHA-256, use the first 12 characters in filenames and the source
  ID, and store the full hash in `sources.json` and the metadata sidecar.
- Keep extracted web sidecar assets in the matching `.assets/` directory.
  Derived Markdown must list those paths in front matter and link each asset
  with a relative Markdown image reference.

## Manifest Maintenance

Every ingested source should have an entry in `sources.json` with:

- `id`
- `title`
- `slug`
- `category`
- `kind`
- `raw_paths`
- `docs_path` for non-repository sources, or `docs_paths` for repository revisions
- `status`

Add `derived_path` when a generated Markdown extraction exists.
For `kind: "repository"`, require `repo_slug`, `provider`, `repository_url`,
`revision`, `docs_paths`, and a `derived_path` ending in
`derived/repo-analysis/{category}/{repo-slug}/{full-sha}/`.
For `kind: "web"`, require `source_url`, `final_url`, `captured_at`, a
64-character SHA-256 `revision`, two immutable `raw_paths`, and a
`derived_path` below `derived/web-markdown/{category}/`. A captured source may
omit `docs_path`; an ingested source must name and be cited by its docs page.

`docs_path` (and each `docs_paths` entry) must stay under `docs/{category}/`.
`kb-check-integrity.py` enforces this prefix, and the page's raw and derived
sources live under the same category. Reorganize within the category instead of
relocating a page across categories unless the raw and derived sources are moved
together and the manifest `category` is updated.

Run these checks after source or docs changes:

```bash
python3 scripts/checks/source_names.py
python3 scripts/checks/repository_integrity.py
./scripts/lint-docs.sh
```

## Source Movement

Renaming or moving files under `raw/` is allowed only as repository organization work. Do not alter the content of original raw sources. When a raw path changes, update:

- `sources.json`
- docs page front matter
- references in `docs/logs/log.md`
- any related `derived/pdf-markdown/` paths
- any related `derived/web-markdown/` paths
- any related `derived/repo-analysis/` paths

Docs page moves under `docs/` follow the same "update everything that
references the old path" rule. When moving or renaming a docs page:

- Use `git mv` to preserve history.
- Recompute relative links in the moved page and every page linking to it
  (broken links and orphan pages fail `scripts/lint-docs.sh`).
- Update `sources.json` `docs_path` / `docs_paths`, the category `index.md`, and
  the subcategory hub page.
- Update `docs/logs/index.md` and append `docs/logs/log.md`.
- Keep the page under its `sources.json` category prefix (see Manifest
  Maintenance).
