---
name: "Lint Docs Cleanup"
description: "Use when linting docs, fixing safe documentation issues, reviewing broken or unused links, and identifying low-quality or orphan pages for cleanup in this repository."
argument-hint: "Describe the docs area or source/topic to lint and clean up."
agent: "Docs Ingest Agent"
tools: [read, edit, search, execute]
model: "GPT-5.4 (copilot)"
---

Run the repository docs lint and cleanup workflow for the requested scope.

Primary lint command:

```bash
./scripts/lint-docs.sh
```

Workflow:

1. Read the relevant files under `docs/`, `logs/`, and `raw/` before changing anything.
2. Run `./scripts/lint-docs.sh` to identify broken links, orphan docs pages, missing front matter fields, stub pages, and placeholder text.
3. Fix safe issues directly when the correct repair is clear.
4. Update `logs/index.md` or `logs/log.md` if the cleanup changes repository navigation or ingest history.
5. Treat deletion conservatively.

Safe fixes include:

- repairing clearly broken relative links
- adding missing required front matter fields using the existing docs schema
- removing placeholder text when the correct final wording is already supported by source material
- updating `logs/index.md` when a valid docs page exists but is missing from the index

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
