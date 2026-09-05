"""Tests for remote source fetching and provenance."""

import hashlib
import io
import json
from pathlib import Path


class TestIsUrl:
    def test_http_and_https(self) -> None:
        from ftw_dataset_tools.api.source import is_url

        assert is_url("https://data.source.coop/x/y.parquet")
        assert is_url("http://example.org/y.parquet")
        assert not is_url("/tmp/y.parquet")
        assert not is_url("s3://bucket/y.parquet")


class TestCacheFilename:
    def test_prefix_and_basename(self) -> None:
        from ftw_dataset_tools.api.source import cache_filename

        url = "https://data.source.coop/ftw/harmonized-field-data/lu/latest/lu.parquet"
        name = cache_filename(url)
        assert name.endswith("-lu.parquet")
        assert len(name.split("-")[0]) == 16
        assert cache_filename("https://example.org/").endswith("-source.parquet")


class _FakeStream(io.BytesIO):
    """A fake urlopen response: a byte stream plus a headers mapping."""

    def __init__(self, payload: bytes, *, content_length: int | None = None) -> None:
        super().__init__(payload)
        length = len(payload) if content_length is None else content_length
        self.headers = {"Content-Length": str(length)}


def _fake_opener(payload: bytes, *, content_length: int | None = None):
    calls = {"n": 0}

    def opener(request, timeout=None):
        assert timeout == 60
        assert request.get_header("User-agent", "").startswith("ftw-dataset-tools/")
        calls["n"] += 1
        return _FakeStream(payload, content_length=content_length)

    return opener, calls


class TestFetchSource:
    def test_downloads_hashes_and_caches(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.source import fetch_source

        payload = b"PAR1fake"
        opener, calls = _fake_opener(payload)
        url = "https://example.org/data/lu.parquet"

        rec = fetch_source(url, tmp_path, opener=opener)

        assert rec.href == url
        assert rec.local_path.exists() and rec.local_path.read_bytes() == payload
        assert rec.local_path.parent == tmp_path
        assert rec.sha256 == hashlib.sha256(payload).hexdigest()
        assert rec.size == len(payload)
        assert rec.fetched_at is not None and rec.fetched_at.endswith("Z")
        assert calls["n"] == 1
        assert not list(tmp_path.glob("*.part"))

    def test_cache_hit_skips_network(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.source import fetch_source

        opener, calls = _fake_opener(b"PAR1fake")
        url = "https://example.org/data/lu.parquet"
        first = fetch_source(url, tmp_path, opener=opener)
        second = fetch_source(url, tmp_path, opener=opener)

        assert calls["n"] == 1
        assert second.local_path == first.local_path
        assert second.sha256 == first.sha256
        assert second.fetched_at is None  # served from cache, no fetch happened now

    def test_refresh_redownloads(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.source import fetch_source

        opener, calls = _fake_opener(b"PAR1fake")
        url = "https://example.org/data/lu.parquet"
        fetch_source(url, tmp_path, opener=opener)
        fetch_source(url, tmp_path, refresh=True, opener=opener)

        assert calls["n"] == 2

    def test_failed_download_leaves_no_file(self, tmp_path: Path) -> None:
        import pytest

        from ftw_dataset_tools.api.source import SourceFetchError, fetch_source

        def broken(request, timeout=None):  # noqa: ARG001
            raise OSError("connection reset")

        with pytest.raises(SourceFetchError):
            fetch_source("https://example.org/data/lu.parquet", tmp_path, opener=broken)

        assert list(tmp_path.iterdir()) == []

    def test_truncated_body_raises_and_leaves_no_part(self, tmp_path: Path) -> None:
        import pytest

        from ftw_dataset_tools.api.source import SourceFetchError, fetch_source

        payload = b"PAR1fake"
        opener, _calls = _fake_opener(payload, content_length=len(payload) + 100)
        url = "https://example.org/data/lu.parquet"

        with pytest.raises(SourceFetchError, match="expected"):
            fetch_source(url, tmp_path, opener=opener)

        assert list(tmp_path.iterdir()) == []
        assert not list(tmp_path.glob("*.part"))

    def test_http_error_surfaces_as_source_fetch_error(self, tmp_path: Path) -> None:
        import urllib.error

        import pytest

        from ftw_dataset_tools.api.source import SourceFetchError, fetch_source

        def forbidden(request, timeout=None):  # noqa: ARG001
            raise urllib.error.HTTPError(
                "https://example.org/data/lu.parquet", 403, "Forbidden", {}, None
            )

        with pytest.raises(SourceFetchError, match="403"):
            fetch_source("https://example.org/data/lu.parquet", tmp_path, opener=forbidden)

        assert list(tmp_path.iterdir()) == []


class TestDescribeLocalSource:
    def test_hashes_local_file(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.source import describe_local_source

        p = tmp_path / "f.parquet"
        p.write_bytes(b"abc")
        rec = describe_local_source(p)

        assert rec.href == str(p.resolve())
        assert rec.local_path == p.resolve()
        assert rec.sha256 == hashlib.sha256(b"abc").hexdigest()
        assert rec.size == 3
        assert rec.fetched_at is None


class TestSourceRecordToDict:
    def test_shape(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.source import SourceRecord

        rec = SourceRecord(
            "https://x/y.parquet", tmp_path / "y.parquet", "ab" * 32, 10, "2026-09-04T00:00:00Z"
        )
        d = rec.to_dict(via="https://x/collection.json")

        assert d == {
            "href": "https://x/y.parquet",
            "via": "https://x/collection.json",
            "sha256": "ab" * 32,
            "size": 10,
            "fetched_at": "2026-09-04T00:00:00Z",
        }
        assert json.dumps(d)


class TestInstalledGitCommit:
    def test_reads_vcs_info(self, monkeypatch) -> None:
        from importlib import metadata

        from ftw_dataset_tools.api import source

        class FakeDist:
            def read_text(self, name):
                assert name == "direct_url.json"
                return json.dumps(
                    {"url": "https://github.com/x/y", "vcs_info": {"commit_id": "a" * 40}}
                )

        monkeypatch.setattr(metadata, "distribution", lambda name: FakeDist())  # noqa: ARG005
        assert source.installed_git_commit() == "a" * 40

    def test_none_without_vcs_info(self, monkeypatch) -> None:
        from importlib import metadata

        from ftw_dataset_tools.api import source

        class FakeDist:
            def read_text(self, name):  # noqa: ARG002
                return json.dumps({"url": "file:///x", "dir_info": {"editable": True}})

        monkeypatch.setattr(metadata, "distribution", lambda name: FakeDist())  # noqa: ARG005
        assert source.installed_git_commit() is None

    def test_none_when_not_installed(self, monkeypatch) -> None:
        from importlib import metadata

        from ftw_dataset_tools.api import source

        def missing(name):
            raise metadata.PackageNotFoundError(name)

        monkeypatch.setattr(metadata, "distribution", missing)
        assert source.installed_git_commit() is None
