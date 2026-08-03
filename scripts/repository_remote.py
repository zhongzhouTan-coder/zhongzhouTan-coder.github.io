"""Normalize supported Git repository remotes for source provenance."""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlsplit


PROVIDERS_BY_HOST = {
    "github.com": "github",
    "gitcode.com": "gitcode",
}
HOSTS_BY_PROVIDER = {provider: host for host, provider in PROVIDERS_BY_HOST.items()}
PATH_PART_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
SCP_REMOTE_RE = re.compile(
    r"^(?:[^@/:]+@)?(?P<host>[^/:]+):(?P<path>[^?#]+)$"
)


@dataclass(frozen=True)
class RepositoryRemote:
    provider: str
    host: str
    repository_path: str
    clone_url: str
    repository_url: str

    def source_id(self, revision: str) -> str:
        return f"{self.provider}:{self.repository_path}@{revision}"


def parse_repository_remote(remote: str) -> RepositoryRemote:
    """Return provider and canonical web metadata for one Git remote URL."""
    value = remote.strip()
    scp_match = SCP_REMOTE_RE.fullmatch(value)
    if scp_match and "://" not in value:
        host = scp_match.group("host").lower()
        repository_path = scp_match.group("path")
    else:
        parsed = urlsplit(value)
        host = (parsed.hostname or "").lower()
        repository_path = parsed.path

    provider = PROVIDERS_BY_HOST.get(host)
    if not provider:
        supported = ", ".join(sorted(PROVIDERS_BY_HOST))
        raise ValueError(f"unsupported repository host {host or value!r}; expected {supported}")

    repository_path = repository_path.strip("/")
    if repository_path.endswith(".git"):
        repository_path = repository_path[:-4]
    parts = repository_path.split("/")
    if len(parts) < 2 or any(not PATH_PART_RE.fullmatch(part) for part in parts):
        raise ValueError(f"invalid repository path in origin: {remote}")

    canonical_url = f"https://{host}/{repository_path}"
    return RepositoryRemote(
        provider=provider,
        host=host,
        repository_path=repository_path,
        clone_url=value,
        repository_url=canonical_url,
    )


def expected_repository_url(provider: str, repository_path: str) -> str:
    host = HOSTS_BY_PROVIDER.get(provider)
    if not host:
        raise ValueError(f"unsupported repository provider: {provider}")
    return f"https://{host}/{repository_path}"
