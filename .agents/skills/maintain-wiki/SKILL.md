---
name: maintain-wiki
description: Maintain and compound this repository's Markdown knowledge base. Use when ingesting PDF, web, Markdown, or repository sources; answering knowledge-base questions that may produce reusable docs; reconciling contradictions or superseded claims; improving cross-links, topic hubs, the wiki index, or the chronological log; or performing a semantic wiki-health review. Coordinate the repository's source-specific workflows, manifests, terms, and lint checks instead of introducing a separate vault schema.
---

# Maintain Wiki

Treat the wiki as a compiled, source-backed knowledge layer: preserve immutable evidence, integrate each useful source once, and make later retrieval better through focused synthesis and cross-links.

## Preserve the Repository Model

- Treat `raw/` as immutable source material. Do not edit source contents.
- Treat `derived/` as regenerable evidence and `docs/` as the curated knowledge layer.
- Treat `sources.json` as the source-to-docs provenance map.
- Treat `docs/logs/index.md` as the retrieval entry point and `docs/logs/log.md` as the append-only maintenance history.
- Inspect `docs/logs/index.md` before ingesting, answering, or creating a page.
- Prefer improving an existing page over creating a parallel summary.
- Make evidence-shaped edits. Do not force an arbitrary number of related-page changes.
- Keep `external-repos/` read-only and reuse pinned repository revisions unless the user intentionally requests newer evidence.

## Select the Operation

Read [repository-workflows.md](references/repository-workflows.md) for the selected operation and its required instruction files.

| Request | Operation |
|---|---|
| Add a paper, article, local document, web page, or repository | Source ingest |
| Ask what the knowledge base says, compare methods, or synthesize a topic | Index-first query |
| Resolve inconsistent claims, improve a topic hub, or repair missing context | Semantic maintenance |
| Find or fix broken links, orphans, front matter, or Markdown issues | Mechanical health check |

For PDF or document extraction, also follow `.agents/skills/mineru-doc-ingest/SKILL.md`. For mechanical lint and safe cleanup, also follow `.agents/skills/lint-docs-cleanup/SKILL.md`. Keep this skill responsible for cross-source synthesis, provenance, navigation, and the final repository-wide completion pass.

## Run the Core Workflow

1. Read `AGENTS.md`, then read only the operation-specific instruction files listed in the reference.
2. Inspect `git status`, `docs/logs/index.md`, the relevant category index and subcategory hub, `sources.json`, and likely related pages.
3. Classify the evidence path before writing: PDF/document, immutable web capture, pinned repository revision, or directly readable source under `raw/`.
4. Read the canonical evidence. Prefer the matching complete derived extraction for synthesis when repository rules designate it as primary.
5. Identify the smallest coherent change set:
   - one existing or new primary docs page;
   - related pages whose claims, confidence, comparison, or retrieval path genuinely change;
   - term pages required by the paper-ingest rules;
   - category/subcategory navigation, the global wiki index, manifest, and log entries required by the operation.
6. Apply the edits without requiring confirmation for routine, reversible maintenance. Pause for category migrations, deletions, ambiguous duplicate sources, or changes that would broaden the requested scope materially.
7. Run the validation commands required by the touched artifacts. Always run `./scripts/lint-docs.sh` after docs or log changes.
8. Report the evidence used, pages changed, contradictions or limitations found, and validation result.

## Handle Contradictions

- Distinguish a direct contradiction from different assumptions, workloads, dates, metrics, or levels of abstraction.
- Cite both supporting sources near the disputed claim.
- Add a concise caution or comparison note to every page whose reader could otherwise be misled.
- Lower `confidence` only when the page's overall evidentiary strength changed; do not lower it merely because two sources study different regimes.
- Preserve the historical claim when useful and explain what newer evidence supersedes or narrows.
- Record consequential contradiction work in `docs/logs/log.md`.

## Compound Valuable Queries

Answer from the maintained docs first. Save the result back into `docs/` when it provides durable value, such as a multi-source comparison, a reusable mechanism explanation, a resolved contradiction, or a synthesis that materially improves retrieval.

When filing an answer back:

- update the best existing page when possible;
- create a page only when it has a distinct retrieval purpose;
- cite the motivating `raw/` sources in front matter;
- add relative links to supporting and neighboring pages;
- update the appropriate hub, `docs/logs/index.md`, and `docs/logs/log.md`;
- run the normal integrity and docs checks.

Do not create a page for a trivial answer, unsupported speculation, or material already easy to retrieve from one existing page.

## Completion Standard

Finish with a connected, provenance-preserving wiki state: no invented source paths, no uncited new claims, no avoidable duplicate page, no orphaned new page or asset, and no unreported validation failure.
