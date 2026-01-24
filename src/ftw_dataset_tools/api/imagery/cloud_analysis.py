"""Pixel-level cloud cover analysis using SCL or cloud probability bands."""

from __future__ import annotations

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

__all__ = ["calculate_pixel_cloud_cover"]


def calculate_pixel_cloud_cover(
    cloud_href: str,
    bbox: tuple[float, float, float, float],
    cloud_type: str = "scl",
    cloud_threshold: int = 30,
) -> float:
    """Calculate cloud cover percentage for a bounding box.

    Reads the Scene Classification Layer (SCL) or cloud probability COG
    and calculates the percentage of cloudy pixels within the bounding box.

    Args:
        cloud_href: URL or path to cloud mask COG (SCL or cloud probability)
        bbox: Bounding box (minx, miny, maxx, maxy) in EPSG:4326
        cloud_type: "scl" for Scene Classification Layer or "probability" for cloud probability
        cloud_threshold: For probability type, minimum value (0-100) to consider as cloud

    Returns:
        Cloud cover percentage (0-100)

    Note:
        SCL classes considered as cloud:
        - 3: Cloud shadow
        - 8: Cloud medium probability
        - 9: Cloud high probability
        - 10: Thin cirrus
    """
    minx, miny, maxx, maxy = bbox

    with rasterio.open(cloud_href) as src:
        # Transform bbox to source CRS if needed
        if src.crs != CRS.from_epsg(4326):
            src_bbox = transform_bounds(CRS.from_epsg(4326), src.crs, minx, miny, maxx, maxy)
        else:
            src_bbox = bbox

        # Read windowed data
        window = from_bounds(*src_bbox, src.transform)
        data = src.read(1, window=window)

        if data.size == 0:
            return 0.0

        if cloud_type == "scl":
            # SCL classes:
            # 0: No data, 1: Saturated/defective, 2: Dark area, 3: Cloud shadow
            # 4: Vegetation, 5: Bare soil, 6: Water, 7: Unclassified
            # 8: Cloud medium probability, 9: Cloud high probability
            # 10: Thin cirrus, 11: Snow/ice
            # Cloud classes are typically 3 (shadow), 8, 9, 10
            cloud_mask = (data == 3) | (data == 8) | (data == 9) | (data == 10)
            valid_mask = data > 0  # Exclude no-data
        else:
            # Cloud probability: values 0-100 (or 0-255 scaled)
            if data.max() > 100:
                # Assume 0-255 scale
                data = data / 255.0 * 100
            cloud_mask = data >= cloud_threshold
            valid_mask = np.ones_like(data, dtype=bool)

        valid_count = np.sum(valid_mask)
        if valid_count == 0:
            return 0.0

        cloud_count = np.sum(cloud_mask & valid_mask)
        cloud_pct = (cloud_count / valid_count) * 100

        return float(cloud_pct)
