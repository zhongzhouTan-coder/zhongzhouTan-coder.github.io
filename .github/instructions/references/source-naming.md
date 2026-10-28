# Source Organization: Categories and Names

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
raw/{category}/{topic-path}/{short-title-slug}--{source-id}.pdf
```

Markdown source names should use:

```text
raw/{category}/{topic-path}/{short-title-slug}.md
```

Repository source records should use:

```text
raw/{category}/{topic-path}/{repo-slug}-codebase--{provider}-{short-sha}.md
```

Generated Markdown should use:

```text
derived/pdf-markdown/{category}/{topic-path}/{short-title-slug}.md
```

When a generated extraction needs sibling assets, use a folder-backed primary Markdown file:

```text
derived/pdf-markdown/{category}/{topic-path}/{short-title-slug}/{short-title-slug}.md
```

Generated repository analysis should use:

```text
derived/repo-analysis/{category}/{topic-path}/{repo-slug}/{full-sha}/
```

Immutable web captures should use:

```text
raw/{category}/{topic-path}/{short-title-slug}--web-{capture-date}-{short-sha256}.html
raw/{category}/{topic-path}/{short-title-slug}--web-{capture-date}-{short-sha256}.metadata.json
derived/web-markdown/{category}/{topic-path}/{short-title-slug}--web-{capture-date}-{short-sha256}.md
derived/web-markdown/{category}/{topic-path}/{short-title-slug}--web-{capture-date}-{short-sha256}.assets/
```

Rules:

- Select a `topic_path` whose first component is listed in the category's
  `topic_roots` in `kb-categories.json`. Use additional components for stable
  project or method families, not for source type or publication date.
- Mirror the same `topic_path` across raw and derived evidence. Keep a
  repository revision under its repository's stable home even when it supports
  docs pages in several topics.
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
