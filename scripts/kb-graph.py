#!/usr/bin/env python3
"""Analyze the repository's ordinary Markdown link graph."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit


DEFAULT_ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_LINK_RE = re.compile(
    r"(?<!!)\[[^\]]+\]\(\s*(?:<([^>]+)>|([^\s)]+))"
)
REMOTE_SCHEMES = {"http", "https", "mailto", "tel", "data"}


def docs_pages(root: Path) -> list[Path]:
    docs_root = root / "docs"
    if not docs_root.is_dir():
        raise ValueError(f"docs directory not found: {docs_root}")
    return [
        path
        for path in sorted(docs_root.rglob("*.md"))
        if "_site" not in path.parts
        and path.relative_to(root).as_posix() != "docs/logs/log.md"
    ]


def link_targets(text: str) -> list[str]:
    return [match.group(1) or match.group(2) for match in MARKDOWN_LINK_RE.finditer(text)]


def resolve_docs_link(root: Path, source: Path, raw_target: str) -> Path | None:
    target = unquote(raw_target.strip())
    if not target or target.startswith("#"):
        return None
    parsed = urlsplit(target)
    if parsed.scheme.lower() in REMOTE_SCHEMES or parsed.netloc:
        return None
    path_text = parsed.path
    if not path_text:
        return None
    candidate = root / path_text.lstrip("/") if path_text.startswith("/") else source.parent / path_text
    candidate = candidate.resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    if candidate.is_dir():
        candidate = candidate / "index.md"
    elif not candidate.suffix and candidate.with_suffix(".md").is_file():
        candidate = candidate.with_suffix(".md")
    if candidate.is_file() and candidate.suffix.lower() == ".md":
        try:
            candidate.relative_to((root / "docs").resolve())
        except ValueError:
            return None
        return candidate
    return None


def connected_components(
    nodes: set[str], outbound: dict[str, set[str]], inbound: dict[str, set[str]]
) -> list[set[str]]:
    components: list[set[str]] = []
    seen: set[str] = set()
    for node in sorted(nodes):
        if node in seen:
            continue
        component: set[str] = set()
        pending = [node]
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            component.add(current)
            neighbors = outbound.get(current, set()) | inbound.get(current, set())
            pending.extend(sorted(neighbors - seen, reverse=True))
        components.append(component)
    return sorted(components, key=lambda component: (-len(component), sorted(component)))


def analyze(root: Path, top: int) -> dict[str, Any]:
    if top < 1:
        raise ValueError("top must be at least 1")
    pages = docs_pages(root)
    path_to_key = {path.resolve(): path.relative_to(root).as_posix() for path in pages}
    nodes = set(path_to_key.values())
    outbound: dict[str, set[str]] = defaultdict(set)
    inbound: dict[str, set[str]] = defaultdict(set)

    for path in pages:
        source_key = path_to_key[path.resolve()]
        text = path.read_text(encoding="utf-8", errors="replace")
        for raw_target in link_targets(text):
            target = resolve_docs_link(root, path, raw_target)
            target_key = path_to_key.get(target.resolve()) if target else None
            if target_key and target_key != source_key:
                outbound[source_key].add(target_key)
                inbound[target_key].add(source_key)

    navigation = {node for node in nodes if Path(node).name == "index.md"}
    content_nodes = nodes - navigation
    components = connected_components(nodes, outbound, inbound)
    top_outbound = sorted(nodes, key=lambda node: (-len(outbound[node]), node))[:top]
    top_inbound = sorted(nodes, key=lambda node: (-len(inbound[node]), node))[:top]
    orphans = sorted(node for node in content_nodes if not inbound[node])
    sinks = sorted(node for node in content_nodes if not outbound[node])

    return {
        "total_pages": len(nodes),
        "content_pages": len(content_nodes),
        "navigation_pages": len(navigation),
        "total_edges": sum(len(targets) for targets in outbound.values()),
        "component_count": len(components),
        "components": [
            {"size": len(component), "sample": sorted(component)[:5]}
            for component in components[:10]
        ],
        "orphans": orphans,
        "sinks": sinks,
        "top_outbound_hubs": [
            {"page": node, "outbound": len(outbound[node])} for node in top_outbound
        ],
        "top_inbound_hubs": [
            {"page": node, "inbound": len(inbound[node])} for node in top_inbound
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze hubs, orphans, sinks, and components in docs/."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        result = analyze(args.root.resolve(), args.top)
    except ValueError as error:
        if args.json:
            print(json.dumps({"ok": False, "error": str(error)}, indent=2))
        else:
            print(f"kb graph: {error}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps({"ok": True, **result}, indent=2))
        return 0
    print(
        "Knowledge-base graph: "
        f"{result['total_pages']} pages, {result['total_edges']} links, "
        f"{result['component_count']} components"
    )
    print(f"Orphan content pages: {len(result['orphans'])}")
    for page in result["orphans"][:10]:
        print(f"  - {page}")
    print(f"Content pages with no outbound docs links: {len(result['sinks'])}")
    for page in result["sinks"][:10]:
        print(f"  - {page}")
    print("Top outbound hubs:")
    for hub in result["top_outbound_hubs"]:
        print(f"  - {hub['page']} ({hub['outbound']})")
    print("Top inbound hubs:")
    for hub in result["top_inbound_hubs"]:
        print(f"  - {hub['page']} ({hub['inbound']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
