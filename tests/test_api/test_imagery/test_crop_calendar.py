"""Tests for crop calendar file downloading."""

from __future__ import annotations

from pathlib import Path

import pytest
import requests

from ftw_dataset_tools.api.imagery.crop_calendar import (
    CROP_CALENDAR_FILES,
    download_crop_calendar_files,
    get_crop_calendar_cache_dir,
)


class _FakeResponse:
    """Minimal stand-in for requests.Response used by the streaming download."""

    def __init__(self, content: bytes, status_code: int = 200) -> None:
        self._content = content
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def iter_content(self, **_kwargs: object) -> list[bytes]:
        return [self._content]


@pytest.fixture(autouse=True)
def _isolated_cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the crop calendar cache at a per-test temp directory."""
    monkeypatch.setenv("FTW_CACHE_DIR", str(tmp_path))
    return get_crop_calendar_cache_dir()


class TestDownloadCropCalendarFiles:
    """Tests for download_crop_calendar_files."""

    def test_downloads_all_files_via_requests(
        self, monkeypatch: pytest.MonkeyPatch, _isolated_cache_dir: Path
    ) -> None:
        """Test each configured file is fetched with requests.get and written to disk."""
        requested_urls: list[str] = []

        def fake_get(url: str, **_kwargs: object) -> _FakeResponse:
            requested_urls.append(url)
            return _FakeResponse(b"fake-tiff-bytes")

        monkeypatch.setattr(requests, "get", fake_get)

        download_crop_calendar_files()

        assert len(requested_urls) == len(CROP_CALENDAR_FILES)
        for filename in CROP_CALENDAR_FILES:
            file_path = _isolated_cache_dir / filename
            assert file_path.exists()
            assert file_path.read_bytes() == b"fake-tiff-bytes"

    def test_skips_existing_files_by_default(
        self, monkeypatch: pytest.MonkeyPatch, _isolated_cache_dir: Path
    ) -> None:
        """Test files already on disk are not re-downloaded unless force=True."""
        _isolated_cache_dir.mkdir(parents=True, exist_ok=True)
        for filename in CROP_CALENDAR_FILES:
            (_isolated_cache_dir / filename).write_bytes(b"already-here")

        call_count = 0

        def fake_get(url: str, **_kwargs: object) -> _FakeResponse:
            nonlocal call_count
            call_count += 1
            return _FakeResponse(b"fresh-bytes")

        monkeypatch.setattr(requests, "get", fake_get)

        download_crop_calendar_files()

        assert call_count == 0
        for filename in CROP_CALENDAR_FILES:
            assert (_isolated_cache_dir / filename).read_bytes() == b"already-here"

    def test_force_redownloads_existing_files(
        self, monkeypatch: pytest.MonkeyPatch, _isolated_cache_dir: Path
    ) -> None:
        """Test force=True re-fetches files even when they already exist."""
        _isolated_cache_dir.mkdir(parents=True, exist_ok=True)
        for filename in CROP_CALENDAR_FILES:
            (_isolated_cache_dir / filename).write_bytes(b"stale-bytes")

        def fake_get(url: str, **_kwargs: object) -> _FakeResponse:
            return _FakeResponse(b"fresh-bytes")

        monkeypatch.setattr(requests, "get", fake_get)

        download_crop_calendar_files(force=True)

        for filename in CROP_CALENDAR_FILES:
            assert (_isolated_cache_dir / filename).read_bytes() == b"fresh-bytes"

    def test_http_error_propagates(
        self, monkeypatch: pytest.MonkeyPatch, _isolated_cache_dir: Path
    ) -> None:
        """Test a failed download raises instead of silently leaving a partial file."""

        def fake_get(url: str, **_kwargs: object) -> _FakeResponse:
            return _FakeResponse(b"", status_code=500)

        monkeypatch.setattr(requests, "get", fake_get)

        with pytest.raises(requests.HTTPError):
            download_crop_calendar_files()
