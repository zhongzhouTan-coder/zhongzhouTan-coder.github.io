---
description: "Use when creating or updating logs/index.md or logs/log.md. Enforces the repository's wiki index structure, chronological log entry format, and concise markdown conventions for ingest tracking."
applyTo: "logs/index.md, logs/log.md"
---

# Logs Maintenance Rules

Use this instruction when updating the knowledge-base index or the chronological change log.

## logs/index.md

- Keep the top-level title as `# Wiki Index`.
- Maintain category sections as markdown headings.
- Use concise bullet lists under each category.
- Add one bullet per docs page or topic entry.
- Each bullet should link to the relevant page in `docs/` using a relative markdown link.
- Prefer short link labels that match the page title or topic name.
- Keep entries grouped by the most relevant category instead of duplicating the same page under many categories.
- If a page clearly belongs in more than one place, choose the strongest primary category and mention cross-links inside the docs page itself.

## logs/log.md

- Keep the top-level title as `# Wiki Log`.
- Append new entries in chronological order with the newest entry at the end of the file.
- Start each entry with a `## YYYY-MM-DD` heading.
- Under each date heading, use flat bullet lists for individual updates.
- Each bullet should briefly state what changed, which source or topic it came from, and which docs page or confidence level was updated.
- Keep log entries factual and concise.
- If multiple related changes happen on the same date, group them under the same date heading instead of repeating the heading.

## Style Rules

- Use markdown only.
- Use relative links for references to files inside the repository.
- Keep wording compact and scannable.
- Do not add raw HTML, tables, or decorative formatting.
- Preserve existing useful entries; update them only when correcting mistakes or consolidating duplicate information.

## Examples

Example `logs/index.md` entry:

```md
- [Topic Name](../docs/algorithms/topic-name.md)
```

Example `logs/log.md` entry:

```md
## 2026-05-28

- Ingested `raw/example.md` into [Example Topic](../docs/benchmarks/example-topic.md) and classified it as medium confidence.
```
