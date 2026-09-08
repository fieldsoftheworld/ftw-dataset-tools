"""Remote source inputs: fetch a fiboa file by URL, cache it, and record provenance."""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from ftw_dataset_tools import __version__

_CHUNK = 1 << 20
_TIMEOUT_SECONDS = 60


class SourceFetchError(OSError):
    """Raised when a remote source could not be downloaded."""


def is_url(value: str) -> bool:
    """True for http(s) URLs; everything else is treated as a local path.

    Plain ``http://`` is recognised here so it gets a clear error instead of
    being mistaken for a local path; only ``https://`` is actually fetchable
    (see :func:`fetch_source`).
    """
    return value.startswith(("http://", "https://"))


def is_https_url(value: str) -> bool:
    """True only for ``https://`` URLs, the only remote scheme ftwd will fetch."""
    return value.startswith("https://")


def cache_filename(url: str) -> str:
    """Content-addressed cache name: sha256(url)[:16] plus the URL basename."""
    basename = Path(urlsplit(url).path).name or "source.parquet"
    return f"{hashlib.sha256(url.encode()).hexdigest()[:16]}-{basename}"


@dataclass(frozen=True)
class SourceRecord:
    """Where the input came from and what bytes were used."""

    href: str
    local_path: Path
    sha256: str
    size: int
    fetched_at: str | None

    def to_dict(self, via: str | None = None) -> dict[str, Any]:
        return {
            "href": self.href,
            "via": via,
            "sha256": self.sha256,
            "size": self.size,
            "fetched_at": self.fetched_at,
        }


def _hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def describe_local_source(path: Path | str) -> SourceRecord:
    """Provenance for a local input file.

    ``href`` is the bare filename: the published provenance identifies the file
    without recording the build machine's directory layout. ``local_path`` keeps
    the resolved path for use inside this process.
    """
    local = Path(path).resolve()
    sha256, size = _hash_file(local)
    return SourceRecord(
        href=local.name, local_path=local, sha256=sha256, size=size, fetched_at=None
    )


def _raise_on_size_mismatch(url: str, content_length: str, size: int) -> None:
    """Check a downloaded body against the declared Content-Length.

    Raises:
        SourceFetchError: If the header is not a number, or the body was short.
    """
    try:
        declared = int(content_length)
    except (TypeError, ValueError) as err:
        raise SourceFetchError(
            f"could not fetch {url}: server sent a non-numeric Content-Length "
            f"header ({content_length!r}); the download cannot be verified."
        ) from err
    if declared != size:
        raise SourceFetchError(f"{url}: expected {declared} bytes, got {size}")


def fetch_source(
    url: str,
    cache_dir: Path | str,
    *,
    refresh: bool = False,
    opener=urllib.request.urlopen,
) -> SourceRecord:
    """Download ``url`` into ``cache_dir`` once and describe it.

    A cached copy is reused unless ``refresh`` is set. The download streams to a
    ``.part`` file and is renamed into place only when complete, so a failed
    fetch leaves nothing behind. ``fetched_at`` is set only when bytes were
    actually fetched in this call.
    """
    if not is_https_url(url):
        raise SourceFetchError(
            f"refusing to fetch {url}: source URLs must use https:// so the recorded "
            "checksum attests to bytes that could not be tampered with in transit."
        )

    cache = Path(cache_dir).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / cache_filename(url)

    if target.exists() and not refresh:
        sha256, size = _hash_file(target)
        return SourceRecord(href=url, local_path=target, sha256=sha256, size=size, fetched_at=None)

    partial = target.with_name(f"{target.name}.{os.getpid()}.part")
    digest = hashlib.sha256()
    size = 0
    headers = {"User-Agent": f"ftw-dataset-tools/{__version__}"}
    request = urllib.request.Request(url, headers=headers)
    try:
        with opener(request, timeout=_TIMEOUT_SECONDS) as response, partial.open("wb") as out:
            for chunk in iter(lambda: response.read(_CHUNK), b""):
                digest.update(chunk)
                size += len(chunk)
                out.write(chunk)
            response_headers = getattr(response, "headers", None)
            expected = (
                response_headers.get("Content-Length") if response_headers is not None else None
            )
        if expected is not None:
            _raise_on_size_mismatch(url, expected, size)
        partial.replace(target)
    except SourceFetchError:
        partial.unlink(missing_ok=True)
        raise
    except urllib.error.HTTPError as err:
        partial.unlink(missing_ok=True)
        raise SourceFetchError(f"could not fetch {url}: HTTP {err.code}: {err.reason}") from err
    except (OSError, http.client.HTTPException) as err:
        # HTTPException (e.g. IncompleteRead when a connection drops mid-body)
        # is not an OSError, so it needs catching explicitly.
        partial.unlink(missing_ok=True)
        raise SourceFetchError(f"could not fetch {url}: {err}") from err
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    fetched_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    return SourceRecord(
        href=url, local_path=target, sha256=digest.hexdigest(), size=size, fetched_at=fetched_at
    )


def installed_git_commit(dist_name: str = "ftw-dataset-tools") -> str | None:
    """The git commit ftwd was installed from (PEP 610), or None."""
    try:
        dist = metadata.distribution(dist_name)
    except metadata.PackageNotFoundError:
        return None
    raw = dist.read_text("direct_url.json")
    if not raw:
        return None
    try:
        return json.loads(raw).get("vcs_info", {}).get("commit_id") or None
    except (ValueError, AttributeError):
        return None
