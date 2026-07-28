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
Markdown under `derived/web-markdown/`, and registers the captured revision in
`sources.json`. See
[the web source instructions](.github/instructions/web-source.instructions.md)
for renderer choices and the `captured` to `ingested` workflow.
