"""Pixel-level cloud and nodata analysis from the Sentinel-2 SCL band."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import rasterio

from ftw_dataset_tools.api.imagery.nodata_analysis import chip_nodata_pct, chip_window

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["SCLChipStats", "read_scl_chip_stats"]


class SCLChipStats(NamedTuple):
    """What one SCL window says about a chip: how much is missing, how much is cloud."""

    nodata_pct: float
    cloud_pct: float


def read_scl_chip_stats(
    scl_href: str,
    bbox: tuple[float, float, float, float],
    footprint: Mapping[str, Any] | None = None,
) -> SCLChipStats:
    """Nodata and cloud percentages for a chip, from a single SCL read.

    SCL carries both facts: class 0 is no-data, and classes 3/8/9/10 are the
    cloud classes. Selection needs both for every candidate scene, so reading
    them together halves the range requests a candidate costs - the nodata
    answer used to come from a separate read of the 10 m NIR band.

    ``cloud_pct`` is the share of *valid* pixels that are cloud; ``nodata_pct`` is the share of the
    whole chip that is missing, counting any part of it that lies beyond the
    scene's raster as missing too. Pass the chip's GeoJSON polygon as
    ``footprint`` whenever it is known (see :func:`chip_window`).
    """
    with rasterio.open(scl_href) as src:
        window, outside = chip_window(src, bbox, footprint)
        if window is None:
            # Nothing of the chip was imaged.
            return SCLChipStats(nodata_pct=100.0, cloud_pct=0.0)
        data = src.read(1, window=window)

    if data.size == 0:
        return SCLChipStats(nodata_pct=100.0, cloud_pct=0.0)

    valid_mask = data > 0
    valid_count = int(np.count_nonzero(valid_mask))
    nodata_pct = chip_nodata_pct(outside, (data.size - valid_count) / data.size)
    if valid_count == 0:
        return SCLChipStats(nodata_pct=nodata_pct, cloud_pct=0.0)

    cloud_mask = (data == 3) | (data == 8) | (data == 9) | (data == 10)
    cloud_pct = int(np.count_nonzero(cloud_mask & valid_mask)) / valid_count * 100.0
    return SCLChipStats(nodata_pct=nodata_pct, cloud_pct=cloud_pct)
