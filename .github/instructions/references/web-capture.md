# Web Source: Capture

## Capture

Run:

```bash
npm run ingest:web -- \
  --url "https://example.com/article" \
  --category frameworks \
  --topic-path serving/example \
  --slug example-article
```

The command defaults to an HTTP capture and falls back to the installed
Chromium browser when the extracted content is suspiciously short or the HTTP
response is an access-check interstitial. Chromium waits up to 10 seconds for
automatic JavaScript checks to clear; adjust this with `--challenge-wait-ms`.
Use `--renderer chromium` for pages that require client-side rendering.
Chromium honors `WEB_INGEST_PROXY`, or the standard `HTTPS_PROXY` /
`HTTP_PROXY` and `NO_PROXY` environment variables when the explicit variable
is absent. For a local agent, install the locked Playwright Chromium once into
the ignored workspace directory:

```bash
./scripts/bootstrap-workspace.sh
./scripts/run-in-workspace.sh npm run install:web-browser
```

The workspace wrapper sets `PLAYWRIGHT_BROWSERS_PATH` to
`.workspace/playwright`, and the capture script prefers that managed browser
over system browsers and Snap Chromium. When the local Linux system is missing
Chromium runtime libraries, run the following from a trusted host shell with
the required privileges:

```bash
./scripts/run-in-workspace.sh npx playwright-core install-deps chromium
```

The local-agent sandbox must also permit outbound access to the source URL, and
any configured proxy must be reachable from inside that sandbox.

The capture command detects common verification and CAPTCHA pages and refuses
to save them as source content. It does not solve CAPTCHAs or bypass
authentication and access controls. When a permitted public page still needs a
manual check, complete it in a normal browser, export the resulting HTML, and
ingest that file with `--input-html`.

The command creates one immutable source revision:

```text
raw/{category}/{topic-path}/{slug}--web-{capture-date}-{short-sha256}.html
raw/{category}/{topic-path}/{slug}--web-{capture-date}-{short-sha256}.metadata.json
derived/web-markdown/{category}/{topic-path}/{slug}--web-{capture-date}-{short-sha256}.md
derived/web-markdown/{category}/{topic-path}/{slug}--web-{capture-date}-{short-sha256}.assets/
```

It also adds a `kind: "web"` entry with `status: "captured"` to `sources.json`.
Never edit or replace the raw HTML or metadata. A changed page produces a new
revision with a new content hash. The `.assets/` directory is created only when
the extractor needs local sidecar files, such as meaningful inline SVG
diagrams.

To rebuild derived Markdown and sidecar assets after improving the extractor,
reuse the exact raw snapshot recorded in `sources.json`:

```bash
npm run ingest:web -- \
  --url "https://example.com/article" \
  --category frameworks \
  --topic-path serving/example \
  --slug example-article \
  --input-html raw/frameworks/serving/example/example-article--web-YYYY-MM-DD-SHA.html \
  --captured-at "YYYY-MM-DDTHH:MM:SS.sssZ" \
  --regenerate-derived
```

This mode verifies the raw HTML hash and changes only generated files. It does
not fetch the live page or modify the immutable raw bundle or manifest.
