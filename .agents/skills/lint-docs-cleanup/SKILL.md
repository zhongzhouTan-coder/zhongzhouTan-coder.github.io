---
name: lint-docs-cleanup
description: Use when linting docs, fixing safe documentation issues, reviewing broken or unused links, and identifying low-quality or orphan pages for cleanup in this repository.
---

# Lint Docs Cleanup

Run the repository docs lint and cleanup workflow for the requested scope.

Primary lint commands:

```bash
# Full lint: custom checks + markdownlint
./scripts/lint-docs.sh

# Markdownlint only (configured docs, instructions, and source records)
npx markdownlint-cli2
```

Do not add a repository-wide glob such as `**/*.md`. The configuration excludes
`external-repos/` because materialized third-party checkouts are immutable
evidence, not files maintained or linted by this wiki.

Workflow:

1. Read the relevant files under `docs/`, `logs/`, and `raw/` before changing anything.
2. Run `./scripts/lint-docs.sh` to identify broken links, orphan docs pages, missing front matter fields, stub pages, placeholder text, and markdownlint violations.
3. Run `npx markdownlint-cli2` separately for the configured Markdown scope,
   including docs, repository-reading instructions, repository source records,
   and derived repository notes. Use the configuration's globs as-is so an
   agent with materialized checkouts does not import upstream lint findings.
4. Fix safe issues directly when the correct repair is clear.
5. When fixing markdownlint violations, follow the rules in `.markdownlint-cli2.jsonc`:
   - Fenced code blocks must have a language tag (MD040)
   - No bare URLs — wrap in `<>` or `[]()` (MD034)
   - No emphasis-as-headings — use `##` headings instead (MD036)
   - Ordered lists must use sequential numbering 1/2/3 (MD029)
   - Blank lines around headings (MD022) and fenced code blocks (MD031)
   - Use backtick fenced code blocks, not indented (MD046, MD048)
6. Update `logs/index.md` or `logs/log.md` if the cleanup changes repository navigation or ingest history.
7. Treat deletion conservatively.

Safe fixes include:

- repairing clearly broken relative links
- adding missing required front matter fields using the repository docs schema
- removing placeholder text when the correct final wording is already supported by source material
- updating `logs/index.md` when a valid docs page exists but is missing from the index
- adding missing code fence language tags (` ```text ` as fallback)
- wrapping bare URLs in angle brackets
- replacing bold-text-as-headings with proper `##` headings

Deletion rules:

- Do not delete `raw/` files.
- Do not delete a docs page only because it is low quality.
- Delete a docs page only when it is clearly orphaned, redundant, or an empty stub, and the reason is supported by the current repository state.
- Before deleting any docs page, verify that its useful content is either preserved elsewhere or has no supported value.
- If deletion is uncertain, leave the file in place and report it as a cleanup candidate instead.

Output format:

- list the lint findings you addressed
- list the files you changed
- state any deletion candidates you did not remove and why
- state whether `logs/index.md` or `logs/log.md` needed updates
