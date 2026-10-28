# Source Organization: Manifest and Moves

## Manifest Maintenance

Every ingested source should have an entry in `sources.json` with:

- `id`
- `title`
- `slug`
- `category`
- `topic_path`
- `kind`
- `raw_paths`
- `docs_path` for non-repository sources, or `docs_paths` for repository revisions
- `status`

Add `derived_path` when a generated Markdown extraction exists.
For `kind: "repository"`, require `repo_slug`, `provider`, `repository_url`,
`revision`, `docs_paths`, and a `derived_path` ending in
`derived/repo-analysis/{category}/{topic-path}/{repo-slug}/{full-sha}/`.
For `kind: "web"`, require `source_url`, `final_url`, `captured_at`, a
64-character SHA-256 `revision`, two immutable `raw_paths`, and a
`derived_path` below `derived/web-markdown/{category}/`. A captured source may
omit `docs_path`; an ingested source must name and be cited by its docs page.

`docs_path` for a non-repository source must stay under
`docs/{category}/{topic-path}/`. Repository `docs_paths` may span topic paths
within the same category because one pinned revision can support several
subsystems.
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
