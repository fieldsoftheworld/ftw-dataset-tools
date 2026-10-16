"""Quarterly mosaic selection: Q1-Q4 of one year for each chip."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ftw_dataset_tools.api.imagery.mosaic_search import find_containing_tile, year_available
from ftw_dataset_tools.api.imagery.nodata_analysis import calculate_nodata_percentage
from ftw_dataset_tools.api.imagery.settings import DEFAULT_NODATA_MAX, MOSAIC_MAX_YEAR_TRIES
from ftw_dataset_tools.api.imagery.slots import MOSAIC_SLOTS

if TYPE_CHECKING:
    from collections.abc import Callable

    import pystac

__all__ = [
    "MosaicSelectionResult",
    "MosaicYearError",
    "SelectedMosaic",
    "check_mosaic_year",
    "fallback_years",
    "select_mosaics_for_chip",
]

# How far from the requested year the fallback looks before giving up.
_MAX_YEAR_DISTANCE = 10


class MosaicYearError(ValueError):
    """The requested mosaic year has no complete quarterly index."""

    def __init__(self, year: int) -> None:
        super().__init__(
            f"Mosaics for {year} are not available (its quarterly index is incomplete). "
            "Pick another --year."
        )
        self.year = year


@dataclass
class SelectedMosaic:
    """The mosaic tile chosen for one quarter of a chip."""

    item: pystac.Item
    quarter: str
    year: int


@dataclass
class MosaicSelectionResult:
    """Result of mosaic selection for a chip."""

    chip_id: str
    bbox: tuple[float, float, float, float]
    requested_year: int
    imagery_year: int | None = None
    quarters: dict[str, SelectedMosaic] = field(default_factory=dict)
    years_tried: list[int] = field(default_factory=list)
    skipped_reason: str | None = None
    candidates_checked: int = 0

    @property
    def success(self) -> bool:
        return self.imagery_year is not None


def check_mosaic_year(year: int) -> None:
    """Raise MosaicYearError unless ``year`` has all four quarterly indexes."""
    if not year_available(year):
        raise MosaicYearError(year)


def fallback_years(
    year: int,
    is_available: Callable[[int], bool] | None = None,
    max_tries: int = MOSAIC_MAX_YEAR_TRIES,
) -> list[int]:
    """``year`` then the nearest years (Y-1, Y+1, Y-2, ...); unavailable years don't count."""
    is_available = is_available or year_available
    candidates = [year]
    for distance in range(1, _MAX_YEAR_DISTANCE + 1):
        candidates += [year - distance, year + distance]
    years: list[int] = []
    for candidate in candidates:
        if len(years) == max_tries:
            break
        if is_available(candidate):
            years.append(candidate)
    return years


def select_mosaics_for_chip(
    chip_id: str,
    bbox: tuple[float, float, float, float],
    year: int,
    nodata_max: float = DEFAULT_NODATA_MAX,
    on_progress: Callable[[str], None] | None = None,
) -> MosaicSelectionResult:
    """Pick Q1-Q4 mosaics of one year for a chip, falling back to nearby years.

    A year is used only if every quarter has a tile wholly containing the chip and
    the chip's nodata is within ``nodata_max`` (same check as scene selection).
    """

    def log(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    result = MosaicSelectionResult(chip_id=chip_id, bbox=bbox, requested_year=year)
    reason = "No mosaic year available"
    for candidate in fallback_years(year):
        result.years_tried.append(candidate)
        quarters, reason = _select_year(bbox, candidate, nodata_max, log, result)
        if quarters is not None:
            result.imagery_year = candidate
            result.quarters = quarters
            log(f"Selected mosaics for {candidate}")
            return result
        log(f"  {reason}")

    tried = ", ".join(str(y) for y in result.years_tried)
    result.skipped_reason = f"No complete mosaic year (tried {tried}): {reason}"
    return result


def _select_year(
    bbox: tuple[float, float, float, float],
    year: int,
    nodata_max: float,
    log: Callable[[str], None],
    result: MosaicSelectionResult,
) -> tuple[dict[str, SelectedMosaic] | None, str]:
    quarters: dict[str, SelectedMosaic] = {}
    for number, slot in enumerate(MOSAIC_SLOTS, start=1):
        tile = find_containing_tile(bbox, year, number)
        if tile is None:
            return None, "no single mosaic tile covers the chip"
        result.candidates_checked += 1
        nodata = calculate_nodata_percentage(tile.assets["nir"].href, bbox)
        if nodata > nodata_max:
            return None, f"{year} Q{number}: {nodata:.1f}% nodata in chip window"
        log(f"  {year} Q{number}: {tile.id}")
        quarters[slot] = SelectedMosaic(item=tile, quarter=slot, year=year)
    return quarters, ""
