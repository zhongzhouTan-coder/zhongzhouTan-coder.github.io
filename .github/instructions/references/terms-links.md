# Glossary Terms: Links and Validation

## Hover Preview Format

Related paper pages may render term links with hover previews, similar to a compact wiki tooltip. The visible links belong on the first meaningful term occurrences in the Markdown body, not in page description/front matter and not in a detached glossary line unless the page genuinely needs a manual term list. The preview text should come from the term page `tooltip` field, falling back to `summary`, then to a short extract from the rendered term page body when those fields are missing.

Use this style when linking terms from paper pages:

```markdown
The interleaved schedule keeps [microbatches](../../terms/microbatch.md) in flight.
```

Use ordinary Markdown link text when the surrounding sentence needs
pluralization, lowercase text, or an acronym. The rendered site enhances links
to `docs/terms/{slug}.md` with hover previews by reading `docs/terms.json`.
That build-generated endpoint also contains exhaustive consumer backlinks,
leaving `appears_in` free to remain curated. Do not duplicate tooltip text
inside paper pages; keep reusable descriptions on the term page.

## Term Link Validation

Run the glossary consistency check after adding or changing terms or their
consumer pages:

```bash
./scripts/run-in-workspace.sh python scripts/checks/term_links.py --fix
./scripts/run-in-workspace.sh python scripts/checks/term_links.py
./scripts/run-in-workspace.sh python scripts/checks/term_links.py \
  --review-mentions docs/path/to/changed-page.md
```

The default command performs structural validation only. It checks term
metadata, the glossary index, curated `appears_in` entries, "Where It Appears,"
consumer link targets, and reviewed exclusions without searching the whole wiki
for new link opportunities. `--fix` only adds missing "Where It Appears" links
for pages already selected in `appears_in`; it never promotes every consumer
into the curated set.

Use `--review-mentions` during authoring. With no paths it reviews all consumer
docs; with file or directory arguments it reviews only that scope. Add
`--strict-mentions` to make its findings errors. Existing Markdown link labels
are excluded because a navigation link already resolves the reader's need.
Plain-text mention decisions remain agent work because they require semantic
judgment.

The checker treats each term page as the source of truth for definition,
aliases, mention policy, and curated appearances; do not maintain a separate
hand-written exhaustive registry. Front matter, headings, code, image captions,
existing Markdown links, term pages, and log pages are excluded from mention
discovery. Glossary-to-glossary navigation is also excluded from `appears_in`
validation; keep those links under "Related Terms" instead.

When a detected occurrence is genuinely not a glossary reference, suppress
that occurrence with a reviewed HTML comment using the term page's filename
slug and a required reason:

```markdown
KV Cache is the name of this benchmark fixture.
<!-- termlint-ignore: kv-cache -- Fixture name, not the attention concept. -->
```

The comment may instead appear at the end of the prose line. A comment-only
directive applies only to the immediately preceding line. Suppressions are
occurrence-local: the checker continues looking for later unsuppressed mentions
on the same page. Unknown slugs, missing reasons, misplaced comments, comments
for already-linked terms, and comments whose target no longer contains a
detectable name are lint errors. Do not use suppressions to avoid linking a
meaningful first occurrence.

## Slug Convention

- Use kebab-case: `pipeline-parallelism.md`, `scatter-gather.md`, `kv-cache.md`.
- Prefer the shortest unambiguous name.
- If a term has a widely-used acronym or shorthand, use that as the canonical slug (e.g., `gqa.md` for Grouped-Query Attention) and list the full name in `aliases`.
