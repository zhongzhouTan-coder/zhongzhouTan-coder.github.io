
# Personal docs Rules

## Directory structure

- raw/ stores original source files. Never modify them.
- docs/ stores AI-maintained markdown knowledge pages.
- logs/index.md stores the docs page index.
- logs/log.md stores chronological change logs.

## Writing rules

- Use markdown only.
- Keep each page focused on one topic.
- Add internal links whenever related pages already exist.
- Update existing pages instead of creating duplicates.
- If new information conflicts with old information, note the contradiction explicitly.
- We can draw some images with mermaid syntax, and prefer images when they clarify complex relationships or processes better than text alone.

## Ingest workflow

When a new source is added:

1. Read the source from raw/
2. Create or update relevant pages in docs/
3. Update logs/index.md
4. Append a new entry to logs/log.md

## Query workflow

When answering questions:

1. Read logs/index.md first
2. Find relevant docs pages
3. Synthesize the answer from docs/
4. If the answer is valuable, save it as a new page in docs/
