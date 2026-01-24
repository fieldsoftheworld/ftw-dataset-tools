"""Crop calendar lookup for determining planting and harvest dates."""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import rasterio

if TYPE_CHECKING:
    from collections.abc import Callable

from ftw_dataset_tools.api.imagery.settings import (
    CROP_CAL_SUMMER_END,
    CROP_CAL_SUMMER_START,
    CROP_CALENDAR_BASE_URL,
    CROP_CALENDAR_FILES,
)

__all__ = [
    "CropCalendarDates",
    "ensure_crop_calendar_exists",
    "get_crop_calendar_cache_dir",
    "get_crop_calendar_dates",
    "harvest_day_to_datetime",
]


@dataclass
class CropCalendarDates:
    """Crop calendar dates for a location."""

    planting_day: int  # Day of year (1-365)
    harvest_day: int  # Day of year (1-365)

    def to_datetime(self, year: int) -> tuple[datetime, datetime]:
        """
        Convert day-of-year values to datetime objects for a given year.

        Handles southern hemisphere where harvest may be in the following year.

        Args:
            year: The calendar year for the planting date

        Returns:
            Tuple of (planting_datetime, harvest_datetime)
        """
        planting_dt = harvest_day_to_datetime(self.planting_day, year)

        # Handle southern hemisphere: if harvest day < planting day,
        # harvest is in the following year
        harvest_year = year + 1 if self.harvest_day < self.planting_day else year
        harvest_dt = harvest_day_to_datetime(self.harvest_day, harvest_year)

        return (planting_dt, harvest_dt)


def get_crop_calendar_cache_dir() -> Path:
    """
    Get the cache directory for crop calendar files.

    Uses FTW_CACHE_DIR environment variable if set,
    otherwise defaults to ~/.cache/ftw-tools/crop_calendar/

    Returns:
        Path to cache directory
    """
    cache_base = os.environ.get("FTW_CACHE_DIR")
    cache_base_path = Path(cache_base) if cache_base else Path.home() / ".cache" / "ftw-tools"
    return cache_base_path / "crop_calendar"


def ensure_crop_calendar_exists(
    on_progress: Callable[[str], None] | None = None,
) -> Path:
    """
    Ensure crop calendar files exist, downloading if necessary.

    Args:
        on_progress: Optional callback for progress messages

    Returns:
        Path to cache directory containing crop calendar files
    """
    cache_dir = get_crop_calendar_cache_dir()

    all_files_exist = cache_dir.exists() and all(
        (cache_dir / filename).exists() for filename in CROP_CALENDAR_FILES
    )

    if not all_files_exist:
        if on_progress:
            on_progress("Downloading crop calendar files (first-time setup)...")
        download_crop_calendar_files(on_progress=on_progress)

    return cache_dir


def download_crop_calendar_files(
    force: bool = False,
    on_progress: Callable[[str], None] | None = None,
) -> None:
    """
    Download all crop calendar files.

    Args:
        force: If True, re-download even if files exist
        on_progress: Optional callback for progress messages
    """
    cache_dir = get_crop_calendar_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)

    for filename in CROP_CALENDAR_FILES:
        file_path = cache_dir / filename

        if file_path.exists() and not force:
            continue

        url = CROP_CALENDAR_BASE_URL + filename
        if on_progress:
            on_progress(f"Downloading {filename}...")

        if file_path.exists():
            file_path.unlink()

        # Download file
        urllib.request.urlretrieve(url, str(file_path))

    if on_progress:
        on_progress(f"Crop calendar files cached at {cache_dir}")


def harvest_day_to_datetime(harvest_day: int, year: int) -> datetime:
    """
    Convert a day-of-year integer to a datetime object.

    Args:
        harvest_day: Day of the year (1-365/366)
        year: The year

    Returns:
        datetime object for that day
    """
    return datetime.strptime(f"{year}-{harvest_day}", "%Y-%j")


def _sample_raster_at_center(
    raster_path: Path,
    bbox: tuple[float, float, float, float],
) -> int:
    """
    Sample a raster at the center of a bounding box.

    For small bboxes (smaller than pixel size), from_bounds returns empty arrays.
    This function samples the single pixel at the bbox center instead.

    Args:
        raster_path: Path to the raster file
        bbox: Bounding box (minx, miny, maxx, maxy) in EPSG:4326

    Returns:
        Integer value at the bbox center

    Raises:
        ValueError: If the center is outside raster bounds or has nodata
    """
    minx, miny, maxx, maxy = bbox
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2

    with rasterio.open(raster_path) as src:
        # Check if center is within raster bounds
        if not (
            src.bounds.left <= cx <= src.bounds.right and src.bounds.bottom <= cy <= src.bounds.top
        ):
            raise ValueError(f"Bbox center ({cx:.4f}, {cy:.4f}) is outside raster bounds")

        # Get pixel coordinates
        row, col = src.index(cx, cy)

        # Use windowed read to only fetch the single pixel needed
        from rasterio.windows import Window

        window = Window(col, row, 1, 1)
        data = src.read(1, window=window)
        value = data[0, 0]

        # Check for nodata
        nodata = src.nodata or 0
        if value == nodata or value <= 0:
            raise ValueError(f"No crop calendar data found for bbox {bbox}")

        return int(value)


def get_crop_calendar_dates(
    bbox: tuple[float, float, float, float],
    on_progress: Callable[[str], None] | None = None,
) -> CropCalendarDates:
    """
    Get crop calendar dates for a bounding box.

    Samples the crop calendar at the bbox center. This handles small bboxes
    that are smaller than the crop calendar pixel size (~50km).

    Currently uses summer crop calendar only.

    Args:
        bbox: Bounding box (minx, miny, maxx, maxy) in EPSG:4326
        on_progress: Optional callback for progress messages

    Returns:
        CropCalendarDates with planting and harvest day-of-year values

    Raises:
        ValueError: If no valid crop calendar data found for the region
    """
    cache_dir = ensure_crop_calendar_exists(on_progress=on_progress)

    start_raster_path = cache_dir / CROP_CAL_SUMMER_START
    end_raster_path = cache_dir / CROP_CAL_SUMMER_END

    planting_day = _sample_raster_at_center(start_raster_path, bbox)
    harvest_day = _sample_raster_at_center(end_raster_path, bbox)

    return CropCalendarDates(
        planting_day=planting_day,
        harvest_day=harvest_day,
    )
