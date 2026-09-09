"""Tests for the crop calendar cache, which several selection threads share.

Selection runs on a thread pool, so the first-time download can be entered from
every worker at once. These tests drive that case directly: the download is
replaced with a slow, chunked writer, so any implementation that wrote into the
final filename (or unlinked it mid-write) would leave a short file behind for a
reader to pick up.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from ftw_dataset_tools.api.imagery import crop_calendar
from ftw_dataset_tools.api.imagery.settings import CROP_CALENDAR_FILES

if TYPE_CHECKING:
    from pathlib import Path

# Long enough that a torn write is unmistakable when compared against it.
FILE_BODY = b"crop-calendar-payload-" * 512


@pytest.fixture
def cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the crop calendar cache at a temp directory."""
    monkeypatch.setenv("FTW_CACHE_DIR", str(tmp_path))
    return tmp_path / "crop_calendar"


def slow_urlretrieve(url: str, filename: str) -> None:  # noqa: ARG001
    """Write the payload in chunks, with a gap wide enough to observe."""
    from pathlib import Path

    with Path(filename).open("wb") as handle:
        for start in range(0, len(FILE_BODY), 4096):
            handle.write(FILE_BODY[start : start + 4096])
            handle.flush()
            time.sleep(0.002)


class TestGetCropCalendarCacheDir:
    """The cache location follows FTW_CACHE_DIR."""

    def test_uses_env_var(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FTW_CACHE_DIR", str(tmp_path))

        assert crop_calendar.get_crop_calendar_cache_dir() == tmp_path / "crop_calendar"

    def test_defaults_to_home_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("FTW_CACHE_DIR", raising=False)

        result = crop_calendar.get_crop_calendar_cache_dir()

        assert result.parts[-3:] == (".cache", "ftw-tools", "crop_calendar")


class TestDownloadCropCalendarFiles:
    """Downloads land atomically and are not repeated needlessly."""

    def test_downloads_every_file(self, cache_dir: Path) -> None:
        with patch.object(crop_calendar.urllib.request, "urlretrieve", slow_urlretrieve):
            crop_calendar.download_crop_calendar_files()

        for filename in CROP_CALENDAR_FILES:
            assert (cache_dir / filename).read_bytes() == FILE_BODY

    def test_skips_files_already_present(self, cache_dir: Path) -> None:
        cache_dir.mkdir(parents=True)
        for filename in CROP_CALENDAR_FILES:
            (cache_dir / filename).write_bytes(FILE_BODY)

        calls: list[str] = []

        def record(url: str, filename: str) -> None:
            calls.append(url)
            slow_urlretrieve(url, filename)

        with patch.object(crop_calendar.urllib.request, "urlretrieve", record):
            crop_calendar.download_crop_calendar_files()

        assert calls == []

    def test_leaves_no_partial_files_behind(self, cache_dir: Path) -> None:
        with patch.object(crop_calendar.urllib.request, "urlretrieve", slow_urlretrieve):
            crop_calendar.download_crop_calendar_files()

        assert sorted(p.name for p in cache_dir.iterdir()) == sorted(CROP_CALENDAR_FILES)

    def test_failed_download_cleans_up_and_leaves_no_target(self, cache_dir: Path) -> None:
        def boom(url: str, filename: str) -> None:  # noqa: ARG001
            from pathlib import Path as _Path

            _Path(filename).write_bytes(b"half")
            raise OSError("connection reset")

        with (
            patch.object(crop_calendar.urllib.request, "urlretrieve", boom),
            pytest.raises(OSError, match="connection reset"),
        ):
            crop_calendar.download_crop_calendar_files()

        assert list(cache_dir.iterdir()) == []


class TestConcurrentCropCalendarDownload:
    """The severity-1 case: every worker reaches the cold cache at once."""

    def test_concurrent_callers_leave_one_complete_file(self, cache_dir: Path) -> None:
        """Eight threads race the download; observers only ever see whole files.

        Each thread ensures the cache, then reads back every file. A reader that
        caught a download mid-flight would see a short body and fail the
        comparison - which is exactly what a garbage-but-openable raster does to
        a real run, turning every chip into a "skipped" reason.
        """
        observed: list[bytes] = []
        errors: list[Exception] = []
        start = threading.Barrier(8)

        def worker() -> None:
            try:
                start.wait(timeout=10)
                directory = crop_calendar.ensure_crop_calendar_exists()
                for filename in CROP_CALENDAR_FILES:
                    observed.append((directory / filename).read_bytes())
            except Exception as err:  # recorded so the assert reports it
                errors.append(err)

        with patch.object(crop_calendar.urllib.request, "urlretrieve", slow_urlretrieve):
            threads = [threading.Thread(target=worker) for _ in range(8)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=30)

        assert errors == []
        assert len(observed) == 8 * len(CROP_CALENDAR_FILES)
        assert all(body == FILE_BODY for body in observed)

        # Nothing partial survives in the user's cache directory.
        assert sorted(p.name for p in cache_dir.iterdir()) == sorted(CROP_CALENDAR_FILES)

    def test_concurrent_callers_download_each_file_once(self, cache_dir: Path) -> None:
        """The lock plus the re-check under it means no duplicate fetches."""
        urls: list[str] = []
        urls_lock = threading.Lock()
        start = threading.Barrier(6)

        def counting(url: str, filename: str) -> None:
            with urls_lock:
                urls.append(url)
            slow_urlretrieve(url, filename)

        def worker() -> None:
            start.wait(timeout=10)
            crop_calendar.ensure_crop_calendar_exists()

        with patch.object(crop_calendar.urllib.request, "urlretrieve", counting):
            threads = [threading.Thread(target=worker) for _ in range(6)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=30)

        assert len(urls) == len(CROP_CALENDAR_FILES)
        assert sorted(p.name for p in cache_dir.iterdir()) == sorted(CROP_CALENDAR_FILES)


class TestEnsureCropCalendarExists:
    """A warm cache is a no-op."""

    def test_no_download_when_files_present(self, cache_dir: Path) -> None:
        cache_dir.mkdir(parents=True)
        for filename in CROP_CALENDAR_FILES:
            (cache_dir / filename).write_bytes(FILE_BODY)

        with patch.object(crop_calendar, "download_crop_calendar_files") as download:
            result = crop_calendar.ensure_crop_calendar_exists()

        download.assert_not_called()
        assert result == cache_dir

    def test_downloads_when_a_file_is_missing(self, cache_dir: Path) -> None:
        cache_dir.mkdir(parents=True)
        (cache_dir / CROP_CALENDAR_FILES[0]).write_bytes(FILE_BODY)

        with patch.object(crop_calendar.urllib.request, "urlretrieve", slow_urlretrieve):
            crop_calendar.ensure_crop_calendar_exists()

        for filename in CROP_CALENDAR_FILES:
            assert (cache_dir / filename).read_bytes() == FILE_BODY
