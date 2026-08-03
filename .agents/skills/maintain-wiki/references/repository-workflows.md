# Repository Wiki Workflows

Use this reference after selecting an operation in `SKILL.md`.

## Instruction Routing

Read each selected instruction file completely before editing its governed artifacts.

| Scope | Required instruction |
|---|---|
| Any docs page | `.github/instructions/docs-front-matter.instructions.md` |
| Paper insight body | `.github/instructions/docs-content-structure.instructions.md` |
| Technical term page or paper ingest | `.github/instructions/docs-terms.instructions.md` |
| Source naming, category, derived path, or manifest | `.github/instructions/source-organization.instructions.md` |
| Web capture or web-derived synthesis | `.github/instructions/web-source.instructions.md` |
| Repository-backed evidence | `.github/instructions/repo-reading.instructions.md` |
| Wiki index or chronological log | `.github/instructions/logs-maintenance.instructions.md` |

Treat `AGENTS.md` as authoritative when this reference and the repository diverge.

## Source Ingest

### Preflight

1. Inspect `git status` and preserve unrelated user changes.
2. Read `docs/logs/index.md` first.
3. Search the target category and likely subcategory for an existing page or hub.
4. Inspect `sources.json` for the source, revision, slug, raw paths, derived path, status, and docs mapping.
5. Inspect related pages before choosing a primary docs page.

### Route by Evidence Type

#### PDF or document

Apply `.agents/skills/mineru-doc-ingest/SKILL.md` and the PDF-triggered repository rules. Choose the canonical category from `kb-categories.json`, maintain `sources.json`, and use a complete `derived/pdf-markdown/` extraction as the primary readable source. Check for an existing extraction before invoking MinerU.

#### Web page

Apply `.github/instructions/web-source.instructions.md`. Capture an immutable HTML and metadata revision with the repository script, synthesize from `derived/web-markdown/`, cite the raw HTML, metadata, and derived Markdown as required, and change the manifest status to `ingested` only after its docs page is ready.

#### Repository

Apply `.github/instructions/repo-reading.instructions.md`. Choose deliberately among `reuse`, `new revision`, and `new repository`. Never refresh to checkout `HEAD` merely because it exists. Read `external-repos/` without editing it, store the pinned source record under `raw/`, and keep generated notes under the recorded revision in `derived/repo-analysis/`.

#### Markdown or other directly readable source

Require the canonical evidence to live under `raw/`. If the requested source is outside `raw/`, identify the appropriate category and canonical path; copy or move it only when the request clearly authorizes adding it. Read it directly, add or update its manifest entry, and cite its canonical raw path.

### Synthesize

1. Extract the source's scope, strongest claims, evidence, assumptions, limitations, and relationship to existing work.
2. Separate source claims from repository-level inference.
3. Select the best existing page to update, or create one with a distinct retrieval purpose.
4. Follow the paper insight structure only for paper insight pages; adapt structure for reference, framework, comparison, or learning pages.
5. Inspect original source figures before creating visuals. Follow the repository's diagram rules and keep selected assets local to the consuming page.
6. Identify cross-paper technical terms. Create or update term pages and link the first meaningful occurrence as required.
7. Update related pages only when the new evidence changes a claim, confidence, comparison, limitation, landscape, or useful retrieval link.
8. Handle contradictions with the policy in `SKILL.md`; avoid treating incomparable experiments as disagreement.

### Integrate

1. Ensure the primary page is reachable from its category index or subcategory hub.
2. Update `docs/logs/index.md` with one strong primary placement.
3. Update `sources.json` paths and status consistently.
4. Append a compact factual entry to `docs/logs/log.md`.
5. Run the source normalization and integrity checks required by the source-organization rules, then run `./scripts/lint-docs.sh`.

## Index-First Query

1. Read `docs/logs/index.md` before searching broadly.
2. Select and read the smallest useful set of topic, hub, term, comparison, and source-backed pages.
3. Follow relevant internal links. When the index and followed links do not surface enough evidence, run `python3 scripts/kb-search.py "<query>" --json`; use `--category <name>` to narrow a broad result set. Fall back to `rg` for exact strings or structural searches.
4. Distinguish what the wiki supports, what sources disagree about, and what the wiki does not yet know.
5. Answer directly with links to the supporting docs pages and precise source paths when useful.
6. File the answer back only when it meets the durable-value threshold in `SKILL.md`. Repository instructions already authorize saving valuable reusable answers; do not add an unnecessary confirmation pause for a routine, in-scope docs update.
7. If the answer exposes an evidence gap, recommend the specific source type or topic needed instead of inventing content.

## Semantic Maintenance

Review meaning and retrieval structure beyond what mechanical lint can prove:

- claims contradicted or narrowed by newer sources;
- claims that appear stale because their evidence is time-sensitive;
- pages whose confidence no longer matches their evidence;
- important concepts repeated across papers but missing a term page;
- plain-text term occurrences that should link to an existing glossary page;
- related pages that lack a useful comparison or cross-link;
- topic folders missing a hub or category index placement;
- index entries that do not reflect the best current retrieval path;
- manifest-to-docs provenance drift;
- synthesized visuals that replaced available original figures without justification.

Run `python3 scripts/kb-graph.py --json` when hubs, sinks, or disconnected topic clusters would help target the review. Treat graph findings as prioritization signals, not automatic defects: a leaf reference page may legitimately have no outbound link, while an orphaned content page usually needs navigation.

When the user asks for a review or audit, report findings without broad edits. When the user asks to improve or fix the wiki, apply safe, evidence-backed repairs and list uncertain items separately.

## Mechanical Health Check

Apply `.agents/skills/lint-docs-cleanup/SKILL.md`.

Run the repository checks appropriate to the touched scope:

```bash
python3 scripts/kb-normalize-source-name.py
python3 scripts/kb-check-integrity.py
./scripts/lint-docs.sh
```

Do not assume every command is read-only: inspect its documented behavior and current diff before accepting generated changes. Fix clear failures within scope. Report uncertain deletions, category moves, source renames, and semantic conflicts instead of guessing.

## Failure Handling

- Preserve `raw/` contents even when extraction fails.
- Reuse a complete existing derived extraction before introducing a fallback.
- Record conversion, rendering, missing-asset, or repository-evidence limitations where they affect interpretation or confidence.
- Leave manifest status short of `ingested` when required provenance or the docs page is incomplete.
- Never hide lint or integrity failures; state the failing command and the remaining issue.
