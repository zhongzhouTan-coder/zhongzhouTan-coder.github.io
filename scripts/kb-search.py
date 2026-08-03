#!/usr/bin/env python3
"""Search curated knowledge-base pages with a small BM25 index."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(__file__).resolve().parents[1]
LATIN_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_'-]*", re.IGNORECASE)
CJK_RUN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "with",
}


def parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    """Return simple scalar front matter and the Markdown body."""
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---\n", 4)
    if end < 0:
        return {}, text
    metadata: dict[str, str] = {}
    for line in text[4:end].splitlines():
        key, separator, value = line.partition(":")
        if separator and not line.startswith((" ", "\t")):
            metadata[key.strip()] = value.strip().strip('"\'')
    return metadata, text[end + 5 :]


def tokenize(text: str) -> list[str]:
    """Tokenize English-like words and CJK runs without dependencies."""
    tokens = [
        token.lower()
        for token in LATIN_TOKEN_RE.findall(text)
        if token.lower() not in STOPWORDS and len(token) > 1
    ]
    for run in CJK_RUN_RE.findall(text):
        tokens.extend(run)
        tokens.extend(run[index : index + 2] for index in range(len(run) - 1))
    return tokens


def iter_docs(root: Path, category: str | None) -> list[dict[str, Any]]:
    docs_root = root / "docs"
    if not docs_root.is_dir():
        raise ValueError(f"docs directory not found: {docs_root}")
    search_root = docs_root / category if category else docs_root
    if not search_root.is_dir():
        raise ValueError(f"docs category not found: {search_root}")

    documents: list[dict[str, Any]] = []
    for path in sorted(search_root.rglob("*.md")):
        relative = path.relative_to(root).as_posix()
        if "_site" in path.parts or relative == "docs/logs/log.md":
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        metadata, body = parse_front_matter(text)
        title = metadata.get("title") or path.stem.replace("-", " ").title()
        summary = metadata.get("summary", "")
        category_name = path.relative_to(docs_root).parts[0]
        weighted_text = " ".join(
            [title] * 4
            + [summary] * 2
            + [path.stem.replace("-", " ")]
            + [body]
        )
        tokens = tokenize(weighted_text)
        documents.append(
            {
                "path": relative,
                "title": title,
                "summary": summary,
                "category": category_name,
                "body": body,
                "tokens": tokens,
                "term_frequency": Counter(tokens),
                "length": len(tokens),
            }
        )
    return documents


def bm25_scores(
    documents: list[dict[str, Any]], query_tokens: list[str]
) -> list[tuple[int, float]]:
    """Return document indexes and BM25 relevance scores."""
    if not documents:
        return []
    average_length = sum(document["length"] for document in documents) / len(
        documents
    )
    average_length = average_length or 1.0
    document_frequency = Counter(
        token for document in documents for token in set(document["tokens"])
    )
    scores: list[tuple[int, float]] = []
    for index, document in enumerate(documents):
        score = 0.0
        for token in query_tokens:
            frequency = document["term_frequency"].get(token, 0)
            if not frequency:
                continue
            containing = document_frequency[token]
            inverse_frequency = math.log(
                1 + (len(documents) - containing + 0.5) / (containing + 0.5)
            )
            normalization = frequency + 1.5 * (
                0.25 + 0.75 * document["length"] / average_length
            )
            score += inverse_frequency * frequency * 2.5 / normalization
        if score:
            scores.append((index, score))
    return sorted(scores, key=lambda item: (-item[1], documents[item[0]]["path"]))


def make_snippet(body: str, query_tokens: list[str], width: int = 240) -> str:
    compact = re.sub(r"\s+", " ", body).strip()
    lower = compact.lower()
    positions = [lower.find(token) for token in query_tokens]
    positions = [position for position in positions if position >= 0]
    start = max(0, (min(positions) if positions else 0) - width // 4)
    end = min(len(compact), start + width)
    snippet = compact[start:end]
    return ("…" if start else "") + snippet + ("…" if end < len(compact) else "")


def search(
    root: Path, query: str, *, category: str | None, limit: int
) -> dict[str, Any]:
    if limit < 1:
        raise ValueError("limit must be at least 1")
    query_tokens = tokenize(query)
    if not query_tokens:
        raise ValueError("query has no searchable terms")
    documents = iter_docs(root, category)
    hits = []
    for index, score in bm25_scores(documents, query_tokens)[:limit]:
        document = documents[index]
        hits.append(
            {
                "path": document["path"],
                "title": document["title"],
                "summary": document["summary"],
                "category": document["category"],
                "score": round(score, 4),
                "snippet": make_snippet(document["body"], query_tokens),
            }
        )
    return {
        "query": query,
        "category": category,
        "document_count": len(documents),
        "hit_count": len(hits),
        "hits": hits,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="BM25 fallback search over curated docs pages."
    )
    parser.add_argument("query", help="Search terms.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--category", help="Limit search to docs/<category>.")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    try:
        result = search(
            args.root.resolve(),
            args.query,
            category=args.category,
            limit=args.limit,
        )
    except ValueError as error:
        if args.json:
            print(json.dumps({"ok": False, "error": str(error)}, indent=2))
        else:
            print(f"kb search: {error}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps({"ok": True, **result}, indent=2, ensure_ascii=False))
        return 0
    if not result["hits"]:
        print(f"No matches for: {args.query}")
        return 0
    print(f"Knowledge-base search: {args.query} ({result['hit_count']} hits)")
    for hit in result["hits"]:
        print(f"\n[{hit['score']:.4f}] {hit['title']} — {hit['path']}")
        if hit["summary"]:
            print(f"  {hit['summary']}")
        print(f"  {hit['snippet']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
