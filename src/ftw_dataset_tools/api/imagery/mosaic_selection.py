"""Quarterly mosaic selection: Q1-Q4 of one year for each chip."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ftw_dataset_tools.api.chip_grid import utm_crs_for_cell
from ftw_dataset_tools.api.imagery.mosaic_search import containing_tiles, year_available
from ftw_dataset_tools.api.imagery.nodata_analysis import calculate_nodata_percentage
from ftw_dataset_tools.api.imagery.settings import DEFAULT_NODATA_MAX
from ftw_dataset_tools.api.imagery.slots import MOSAIC_SLOTS
from ftw_dataset_tools.api.logging_config import get_logger

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pystac

__all__ = [
    "MosaicSelectionResult",
    "MosaicYearError",
    "SelectedMosaic",
    "check_mosaic_year",
    "select_mosaics_for_chip",
]

logger = get_logger(__name__)


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
    skipped_reason: str | None = None
    candidates_checked: int = 0

    @property
    def success(self) -> bool:
        return self.imagery_year is not None


def check_mosaic_year(year: int) -> None:
    """Raise MosaicYearError unless ``year`` has all four quarterly indexes."""
    if not year_available(year):
        raise MosaicYearError(year)


def select_mosaics_for_chip(
    chip_id: str,
    bbox: tuple[float, float, float, float],
    year: int,
    nodata_max: float = DEFAULT_NODATA_MAX,
    footprint: Mapping[str, Any] | None = None,
) -> MosaicSelectionResult:
    """Pick Q1-Q4 mosaics of ``year`` for a chip, or none.

    Only the chip's own year is used, never a nearby one: a dataset's imagery must
    show the same year as its labels. A year is used only if every quarter has a tile
    wholly containing the chip with the chip's nodata within ``nodata_max`` (same
    check as scene selection). Both are measured on the chip's GeoJSON ``footprint``
    when given, else on ``bbox``.
    """
    result = MosaicSelectionResult(chip_id=chip_id, bbox=bbox, requested_year=year)
    if not year_available(year):
        result.skipped_reason = f"No mosaics for {year}"
        return result
    crs = utm_crs_for_cell(chip_id, bbox)
    quarters, reason = _select_year(bbox, year, nodata_max, result, footprint, crs)
    if quarters is None:
        result.skipped_reason = f"No complete {year} mosaics: {reason}"
        return result
    result.imagery_year = year
    result.quarters = quarters
    logger.info(f"Selected mosaics for {year}")
    return result


def _select_year(
    bbox: tuple[float, float, float, float],
    year: int,
    nodata_max: float,
    result: MosaicSelectionResult,
    footprint: Mapping[str, Any] | None = None,
    crs: str | None = None,
) -> tuple[dict[str, SelectedMosaic] | None, str]:
    quarters: dict[str, SelectedMosaic] = {}
    for number, slot in enumerate(MOSAIC_SLOTS, start=1):
        tiles = containing_tiles(bbox, year, number, footprint, crs)
        if not tiles:
            return None, "no single mosaic tile covers the chip"
        tile, reason = _least_nodata_tile(tiles, bbox, nodata_max, result, footprint)
        if tile is None:
            return None, f"Q{number} {reason}"
        logger.info(f"  {year} Q{number}: {tile.id}")
        quarters[slot] = SelectedMosaic(item=tile, quarter=slot, year=year)
    return quarters, ""


def _least_nodata_tile(
    tiles: list[pystac.Item],
    bbox: tuple[float, float, float, float],
    nodata_max: float,
    result: MosaicSelectionResult,
    footprint: Mapping[str, Any] | None,
) -> tuple[pystac.Item | None, str]:
    """The tile of ``tiles`` with the least nodata over the chip, if within ``nodata_max``.

    A neighbouring zone's tile can hold the chip with full data where the chip's own
    zone's tile has a gap, or can't be read, so every containing tile is read until
    one is clean. On equal nodata the earlier, better-ranked tile wins. If no tile is
    within ``nodata_max`` and one could not be read, its error is raised: the chip
    failed, which a rerun retries, rather than having no mosaics.

    Raises:
        Exception: The last read error, when no tile was within ``nodata_max``.
    """
    best, best_nodata, nodata_seen, error = None, float("inf"), [], None
    for tile in tiles:
        result.candidates_checked += 1
        try:
            nodata = calculate_nodata_percentage(tile.assets["nir"].href, bbox, footprint)
        except Exception as e:
            logger.warning(f"{tile.id}: chip nodata check failed ({e})")
            error = e
            continue
        nodata_seen.append(f"{tile.properties.get('ftw:mosaic_tile', tile.id)} {nodata:.2f}%")
        if nodata < best_nodata:
            best, best_nodata = tile, nodata
        if nodata == 0:
            break
    if best is not None and best_nodata <= nodata_max:
        return best, ""
    if error is not None:
        raise error
    return None, f"nodata in chip window ({', '.join(nodata_seen)})"
