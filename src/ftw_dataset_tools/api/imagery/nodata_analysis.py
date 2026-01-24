"""Nodata/blackfill detection for Sentinel-2 imagery.

Provides efficient nodata percentage calculation using COG overviews.
"""

from __future__ import annotations

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.warp import transform_bounds
from rasterio.windows import Window, from_bounds

# Sentinel-2 nodata value
SENTINEL2_NODATA = 0


def get_nodata_from_metadata(item) -> float | None:
    """Get nodata percentage from STAC metadata if available.

    Sentinel-2 items may have s2:nodata_pixel_percentage in their properties.

    Args:
        item: PySTAC Item

    Returns:
        Nodata percentage (0-100) if available, None otherwise
    """
    return item.properties.get("s2:nodata_pixel_percentage")


def calculate_nodata_percentage(
    cog_url: str,
    bbox: tuple[float, float, float, float],
) -> float:
    """Calculate percentage of nodata pixels in a COG within bbox.

    Uses COG overviews for efficient reading (16x less data at overview level 2).

    Args:
        cog_url: URL to COG file
        bbox: Bounding box (minx, miny, maxx, maxy) in EPSG:4326

    Returns:
        Percentage of nodata pixels (0-100)
    """
    # Use overview level 2 for efficient reading (16x less data)
    try:
        with rasterio.open(cog_url, overview_level=2) as src:
            nodata_pct = _calculate_nodata_for_source(src, bbox)
    except rasterio.errors.RasterioIOError:
        # If overview level 2 doesn't exist, try without specifying level
        with rasterio.open(cog_url) as src:
            nodata_pct = _calculate_nodata_for_source(src, bbox)

    return nodata_pct


def _calculate_nodata_for_source(
    src: rasterio.DatasetReader,
    bbox: tuple[float, float, float, float],
) -> float:
    """Calculate nodata percentage for an open rasterio source.

    Args:
        src: Open rasterio dataset
        bbox: Bounding box in EPSG:4326

    Returns:
        Percentage of nodata pixels (0-100)
    """
    # Transform bbox to source CRS if needed
    if src.crs != CRS.from_epsg(4326):
        src_bbox = transform_bounds(CRS.from_epsg(4326), src.crs, *bbox)
    else:
        src_bbox = bbox

    window = from_bounds(*src_bbox, src.transform)

    # Clamp window to dataset bounds
    dataset_window = Window(0, 0, src.width, src.height)
    window = window.intersection(dataset_window)

    if window.width <= 0 or window.height <= 0:
        return 100.0  # No valid data in bbox

    # Read first band at window
    data = src.read(1, window=window)

    nodata = src.nodata if src.nodata is not None else SENTINEL2_NODATA
    nodata_count = np.sum(data == nodata)
    total_pixels = data.size

    if total_pixels == 0:
        return 100.0

    return (nodata_count / total_pixels) * 100
