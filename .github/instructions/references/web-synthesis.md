# Web Source: Synthesis and Validation

## Synthesis

Treat the generated file under `derived/web-markdown/` as the primary readable
source. Before writing, read `docs/logs/index.md` and inspect related topic
pages.

When the resulting docs page is ready:

1. Cite the raw HTML, raw metadata, and derived Markdown in its `sources` front
   matter.
2. Set the manifest entry's `docs_path`.
3. Change its status from `captured` to `ingested`.
4. Update `docs/logs/index.md` and append to `docs/logs/log.md`.

## Renderer and Evidence Notes

- Prefer the HTTP renderer for stable, public, server-rendered pages.
- Use Chromium for JavaScript-rendered pages and record any missing,
  authenticated, lazy-loaded, or interactive content as a limitation.
- The extractor normalizes common lazy-loaded image attributes and preserves
  meaningful inline SVG diagrams as local files next to the derived Markdown.
  Ordinary remote raster images remain remote in the extraction and are not
  immutable. During synthesis, inspect those images before creating any new
  visual. Download every selected original figure into the final docs page's
  `assets/` directory, embed the local copy, and cite the captured web page in
  its caption. Original source figures are preferred over newly synthesized
  visuals everywhere except The Landscape, which uses a locally saved Mermaid
  diagram.
- Do not treat a browser-rendered snapshot as proof of content that was hidden
  behind an inaccessible interaction.
- Response metadata excludes cookie and proxy-authentication headers so session
  credentials are not written to the repository.
- Do not pass signed URLs, access tokens, or other secrets in the source URL;
  the URL is intentionally recorded in raw metadata and `sources.json`.
- Do not use `--allow-private-network` for untrusted URLs.

## Validation

Run:

```bash
python3 scripts/checks/source_names.py
python3 scripts/checks/repository_integrity.py
./scripts/lint-docs.sh
```
