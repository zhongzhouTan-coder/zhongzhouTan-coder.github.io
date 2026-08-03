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
MARKDOWN_LINK_RE = re.compile(r"(?<!!)\[[^\]]+\]\(\s*(?:<([^>]+)>|([^\s)]+))")
REMOTE_SCHEMES = {"http", "https", "mailto", "tel", "data"}
FRONT_MATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
TITLE_IN_FRONT_MATTER_RE = re.compile(r"^title:\s*['\"]?(.+?)['\"]?\s*$", re.MULTILINE)
H1_RE = re.compile(r"^#\s+(.+)$", re.MULTILINE)
CATEGORY_RE = re.compile(r"^docs/([^/]+)/")


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


def _extract_title(path: Path) -> str:
    """Extract a human-readable title from a Markdown file.

    Prefers the ``title`` field in Jekyll front matter; falls back to the
    first H1 heading; finally falls back to the filename stem.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    fm_match = FRONT_MATTER_RE.match(text)
    if fm_match:
        fm = fm_match.group(1)
        title_match = TITLE_IN_FRONT_MATTER_RE.search(fm)
        if title_match:
            return title_match.group(1).strip()
    h1_match = H1_RE.search(text)
    if h1_match:
        return h1_match.group(1).strip()
    return path.stem


def _category_from_path(key: str) -> str:
    m = CATEGORY_RE.match(key)
    return m.group(1) if m else "other"


def _url_from_key(key: str) -> str:
    """Convert a repository-relative docs key to a published site URL."""
    key = key.removeprefix("docs/")
    if key.endswith("/index.md"):
        url = "/" + key[: -len("index.md")]
    elif key.endswith(".md"):
        url = "/" + key[:-3] + "/"
    else:
        url = "/" + key
    return url


def link_targets(text: str) -> list[str]:
    return [
        match.group(1) or match.group(2) for match in MARKDOWN_LINK_RE.finditer(text)
    ]


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
    candidate = (
        root / path_text.lstrip("/")
        if path_text.startswith("/")
        else source.parent / path_text
    )
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
    return sorted(
        components, key=lambda component: (-len(component), sorted(component))
    )


def dump_graph(root: Path) -> dict[str, Any]:
    """Build nodes and edges for a force-directed graph visualization.

    Returns a dict with ``nodes`` (list of {id, label, category, isIndex, url,
    degree}) and ``edges`` (list of {source, target}).
    """
    pages = docs_pages(root)
    path_to_key = {path.resolve(): path.relative_to(root).as_posix() for path in pages}
    nodes_map: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, str]] = []
    outbound: dict[str, set[str]] = defaultdict(set)
    inbound: dict[str, set[str]] = defaultdict(set)

    for path in pages:
        source_key = path_to_key[path.resolve()]
        if source_key not in nodes_map:
            nodes_map[source_key] = {
                "id": source_key,
                "label": _extract_title(path),
                "category": _category_from_path(source_key),
                "isIndex": Path(source_key).name == "index.md",
                "url": _url_from_key(source_key),
            }
        text = path.read_text(encoding="utf-8", errors="replace")
        for raw_target in link_targets(text):
            target = resolve_docs_link(root, path, raw_target)
            target_key = path_to_key.get(target.resolve()) if target else None
            if target_key and target_key != source_key:
                if target_key not in nodes_map:
                    target_path = root / target_key
                    nodes_map[target_key] = {
                        "id": target_key,
                        "label": _extract_title(target_path)
                        if target_path.is_file()
                        else target_key,
                        "category": _category_from_path(target_key),
                        "isIndex": Path(target_key).name == "index.md",
                        "url": _url_from_key(target_key),
                    }
                outbound[source_key].add(target_key)
                inbound[target_key].add(source_key)

    # Deduplicate edges (undirected for visualization)
    seen_edges: set[tuple[str, str]] = set()
    for src in sorted(outbound):
        for tgt in sorted(outbound[src]):
            pair = tuple(sorted([src, tgt]))
            if pair not in seen_edges:
                seen_edges.add(pair)
                edges.append({"source": src, "target": tgt})

    # Compute degree for sizing nodes
    for node_id, node in nodes_map.items():
        node["degree"] = len(outbound.get(node_id, set())) + len(
            inbound.get(node_id, set())
        )

    return {
        "nodes": sorted(nodes_map.values(), key=lambda n: n["id"]),
        "edges": sorted(edges, key=lambda edge: (edge["source"], edge["target"])),
    }


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
    parser.add_argument(
        "--dump-graph",
        action="store_true",
        help="Output nodes and edges JSON for graph visualization.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write graph JSON to this file (default: stdout). Only used with --dump-graph.",
    )
    args = parser.parse_args()

    if args.dump_graph:
        try:
            graph_data = dump_graph(args.root.resolve())
        except ValueError as error:
            print(f"kb graph: {error}", file=sys.stderr)
            return 2
        json_text = json.dumps(graph_data, indent=2, ensure_ascii=False)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json_text, encoding="utf-8")
            print(
                f"Graph data written to {args.output} ({len(graph_data['nodes'])} nodes, {len(graph_data['edges'])} edges)"
            )
        else:
            print(json_text)
        return 0

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
