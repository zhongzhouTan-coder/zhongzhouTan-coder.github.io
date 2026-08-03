# zhongzhouTan-coder.github.io

I want to build a long-term personal blog, and all the content will be written by github copilot and hosted on github pages. I want to test that we can change our work style and do knowledge precipitation automatically.

## Web source capture

Install the pinned dependencies and capture a page with:

```bash
npm ci
npm run ingest:web -- \
  --url "https://example.com/article" \
  --category frameworks \
  --slug example-article
```

The command saves immutable HTML and metadata under `raw/`, writes readable
Markdown under `derived/web-markdown/`, preserves inline SVG diagrams as local
sidecar assets, and registers the captured revision in `sources.json`. See
[the web source instructions](.github/instructions/web-source.instructions.md)
for renderer choices and the `captured` to `ingested` workflow.

## Wiki maintenance utilities

Read `docs/logs/index.md` first when exploring the knowledge base. Use the
dependency-free BM25 search only when the index and internal links do not expose
enough evidence:

```bash
npm run wiki:search -- "paged attention" --limit 5
npm run wiki:search -- "pipeline bubble" --category training --json
```

Inspect the Markdown link graph to find hubs, orphan content pages, sinks, and
disconnected topic clusters:

```bash
npm run wiki:graph -- --top 10
npm run wiki:graph -- --json
```

The source naming and knowledge-base integrity checks also support structured
output for agents and CI:

```bash
python3 scripts/kb-normalize-source-name.py --json
python3 scripts/kb-check-integrity.py --json
```
