"""Nodata/blackfill detection for Sentinel-2 imagery.

Measures a chip's nodata on the COG's full-resolution pixels.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import rasterio
import shapely
from rasterio.crs import CRS
from rasterio.warp import transform_bounds
from rasterio.windows import Window, from_bounds
from shapely.geometry import box, shape

from ftw_dataset_tools.api.chip_grid import reprojector

if TYPE_CHECKING:
    from collections.abc import Mapping

    from shapely.geometry.base import BaseGeometry

# Sentinel-2 nodata value
SENTINEL2_NODATA = 0

_WGS84 = CRS.from_epsg(4326)
# Vertex spacing for a lon/lat footprint before reprojection (~100 m), so its
# edges follow the curve they make on a UTM grid.
_DENSIFY_DEGREES = 1e-3
# Overhang below this share of the chip (0.01%: a 0.2 m strip along a 2 km edge)
# is float noise from reprojection, not a chip off the raster's edge.
_OVERHANG_TOLERANCE = 1e-4


def calculate_nodata_percentage(
    cog_url: str,
    bbox: tuple[float, float, float, float],
    footprint: Mapping[str, Any] | None = None,
) -> float:
    """Calculate percentage of nodata pixels in a COG within bbox.

    Reads full-resolution pixels: a chip window is small, and an overview would
    average away the few-pixel holes a quarterly mosaic leaves where every scene was
    cloudy, letting a chip with gaps pass a zero-nodata check.

    Args:
        cog_url: URL to COG file
        bbox: Bounding box (minx, miny, maxx, maxy) in EPSG:4326
        footprint: The chip's GeoJSON polygon in EPSG:4326, measured instead of bbox

    Returns:
        Percentage of nodata pixels (0-100)
    """
    with rasterio.open(cog_url) as src:
        return _calculate_nodata_for_source(src, bbox, footprint)


def _calculate_nodata_for_source(
    src: rasterio.DatasetReader,
    bbox: tuple[float, float, float, float],
    footprint: Mapping[str, Any] | None = None,
) -> float:
    """Calculate nodata percentage for an open rasterio source.

    Args:
        src: Open rasterio dataset
        bbox: Bounding box in EPSG:4326
        footprint: The chip's GeoJSON polygon in EPSG:4326, measured instead of bbox

    Returns:
        Percentage of nodata pixels (0-100)
    """
    window, outside = chip_window(src, bbox, footprint)
    if window is None:
        return 100.0  # The chip lies wholly outside the scene

    # Read first band at window
    data = src.read(1, window=window)
    if data.size == 0:
        return 100.0

    nodata = src.nodata if src.nodata is not None else SENTINEL2_NODATA
    return chip_nodata_pct(outside, float(np.sum(data == nodata)) / data.size)


def chip_window(
    src: rasterio.DatasetReader,
    bbox: tuple[float, float, float, float],
    footprint: Mapping[str, Any] | None = None,
) -> tuple[Window | None, float]:
    """The window to read for a chip, and the share (0-1) of the chip beyond the raster.

    A chip hanging off the granule edge has only its covered part to read; measured
    on that part alone it would look clean while its image comes out partly empty,
    so the part beyond the raster is returned to be counted as nodata.

    Pass the chip's polygon as ``footprint`` whenever it is known. A lon/lat bbox
    taken back onto a UTM scene grid is an envelope of an envelope: away from the
    zone's central meridian it is up to ~90 m wider per side than the chip, which
    would push a chip that sits just inside the granule's edge over it.

    Args:
        src: Open rasterio dataset
        bbox: The chip's bounding box in EPSG:4326, used when there is no footprint
        footprint: The chip's GeoJSON polygon in EPSG:4326

    Returns:
        The in-raster window (None when the chip misses the raster entirely) and
        the share of the chip's area that lies beyond the raster.
    """
    chip = chip_in_crs(src.crs, bbox, footprint)
    inside = chip.intersection(box(*src.bounds))
    if chip.area <= 0 or inside.is_empty or inside.area <= 0:
        return None, 1.0
    outside = 1.0 - inside.area / chip.area
    # Reprojection leaves a chip that sits exactly on the raster edge a hair over it.
    if outside < _OVERHANG_TOLERANCE:
        outside = 0.0
    return from_bounds(*inside.bounds, src.transform), outside


def chip_nodata_pct(outside: float, nodata_share: float) -> float:
    """Percentage (0-100) of a chip that is nodata: the part beyond the raster plus
    the nodata share of the part inside it."""
    return 100.0 * (outside + (1.0 - outside) * nodata_share)


def chip_in_crs(
    crs: CRS,
    bbox: tuple[float, float, float, float],
    footprint: Mapping[str, Any] | None,
) -> BaseGeometry:
    """The chip as a polygon in ``crs``: its footprint when known, else its bbox envelope."""
    if footprint is None:
        if crs == _WGS84:
            return box(*bbox)
        return box(*transform_bounds(_WGS84, crs, *bbox))
    chip = shapely.segmentize(shape(footprint), _DENSIFY_DEGREES)
    if crs == _WGS84:
        return chip
    return shapely.transform(chip, reprojector(_WGS84, crs))
