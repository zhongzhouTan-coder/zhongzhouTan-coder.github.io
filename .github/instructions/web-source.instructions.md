---
description: "Use when capturing an HTML page as an immutable local source and extracting it to Markdown with scripts/web-source-to-markdown.mjs."
applyTo: "raw/**/*.html, raw/**/*.metadata.json, derived/web-markdown/**/*, scripts/web-source-to-markdown.mjs, sources.json"
---

# Web Source Ingestion Rules

Use this instruction for HTML pages. Send URLs that resolve to PDFs through the
PDF workflow and use the repository-reading workflow when the source of truth is
a code repository.

## Capture

Run:

```bash
npm run ingest:web -- \
  --url "https://example.com/article" \
  --category frameworks \
  --slug example-article
```

The command defaults to an HTTP capture and falls back to the installed
Chromium browser when the extracted content is suspiciously short. Use
`--renderer chromium` for pages that require client-side rendering.
Chromium honors `WEB_INGEST_PROXY`, or the standard `HTTPS_PROXY` /
`HTTP_PROXY` and `NO_PROXY` environment variables when the explicit variable
is absent.

The command creates one immutable source revision:

```text
raw/{category}/{slug}--web-{capture-date}-{short-sha256}.html
raw/{category}/{slug}--web-{capture-date}-{short-sha256}.metadata.json
derived/web-markdown/{category}/{slug}--web-{capture-date}-{short-sha256}.md
```

It also adds a `kind: "web"` entry with `status: "captured"` to `sources.json`.
Never edit or replace the raw HTML or metadata. A changed page produces a new
revision with a new content hash.

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
- The extractor keeps links and images by default, but remote assets are not
  immutable. Save essential figures locally under the docs page's assets
  directory.
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
python3 scripts/kb-normalize-source-name.py
python3 scripts/kb-check-integrity.py
./scripts/lint-docs.sh
```
