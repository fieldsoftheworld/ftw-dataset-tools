"""Pixel-level cloud cover analysis using SCL, UDM2 or cloud probability bands."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.errors import WindowError
from rasterio.features import geometry_mask
from rasterio.warp import transform_bounds
from rasterio.windows import Window, from_bounds
from shapely.geometry import box
from shapely.ops import transform as shapely_transform

__all__ = [
    "SCLChipStats",
    "calculate_pixel_cloud_cover",
    "read_scl_chip_stats",
    "read_udm2_chip_stats",
]


class SCLChipStats(NamedTuple):
    """What one SCL or UDM2 window says about a chip: how much is missing, how much is cloud."""

    nodata_pct: float
    cloud_pct: float


def read_scl_chip_stats(scl_href: str, bbox: tuple[float, float, float, float]) -> SCLChipStats:
    """Nodata and cloud percentages for a chip, from a single SCL read.

    SCL carries both facts: class 0 is no-data, and classes 3/8/9/10 are the
    cloud classes. Selection needs both for every candidate scene, so reading
    them together halves the range requests a candidate costs - the nodata
    answer used to come from a separate read of the 10 m NIR band.

    ``cloud_pct`` is the share of *valid* pixels that are cloud, matching
    :func:`calculate_pixel_cloud_cover`; ``nodata_pct`` is the share of the
    whole window that is missing.
    """
    with rasterio.open(scl_href) as src:
        if src.crs != CRS.from_epsg(4326):
            src_bbox = transform_bounds(CRS.from_epsg(4326), src.crs, *bbox)
        else:
            src_bbox = bbox

        requested = from_bounds(*src_bbox, src.transform)
        try:
            window = requested.intersection(Window(0, 0, src.width, src.height))
        except WindowError:
            # rasterio raises rather than returning an empty window when the
            # chip falls outside the scene: nothing of it was imaged.
            return SCLChipStats(nodata_pct=100.0, cloud_pct=0.0)
        if window.width <= 0 or window.height <= 0:
            return SCLChipStats(nodata_pct=100.0, cloud_pct=0.0)

        data = src.read(1, window=window)

    if data.size == 0:
        return SCLChipStats(nodata_pct=100.0, cloud_pct=0.0)

    valid_mask = data > 0
    valid_count = int(np.count_nonzero(valid_mask))
    nodata_pct = (data.size - valid_count) / data.size * 100.0
    if valid_count == 0:
        return SCLChipStats(nodata_pct=nodata_pct, cloud_pct=0.0)

    cloud_mask = (data == 3) | (data == 8) | (data == 9) | (data == 10)
    cloud_pct = int(np.count_nonzero(cloud_mask & valid_mask)) / valid_count * 100.0
    return SCLChipStats(nodata_pct=float(nodata_pct), cloud_pct=float(cloud_pct))


def read_udm2_chip_stats(udm2_path: str, bbox: tuple[float, float, float, float]) -> SCLChipStats:
    """Nodata and not-clear percentages for a chip, from a PlanetScope UDM2 file.

    UDM2 band 1 is the clear map (1 = clear) and band 8 is the legacy unusable-data
    mask, whose bit 0 marks blackfill: pixels outside the imaged area. As with SCL,
    ``cloud_pct`` is the share of *valid* pixels that are not clear (cloud, shadow,
    haze or snow) and ``nodata_pct`` the share of the chip that is blackfill or
    falls outside the file.

    Pixels are counted inside the chip polygon reprojected to the file's CRS, not
    its bounding box: delivered files are clipped close to the chips, and a
    reprojected box would reach past the file's edge and read as missing data.
    """
    with rasterio.open(udm2_path) as src:
        to_src = Transformer.from_crs(CRS.from_epsg(4326), src.crs, always_xy=True)
        chip = shapely_transform(to_src.transform, box(*bbox))
        window = from_bounds(*chip.bounds, src.transform).round_offsets().round_lengths()
        if window.width <= 0 or window.height <= 0:
            return SCLChipStats(nodata_pct=100.0, cloud_pct=0.0)
        # Outside the file reads as blackfill, so it counts as missing.
        clear = src.read(1, window=window, boundless=True, fill_value=0)
        unusable = src.read(8, window=window, boundless=True, fill_value=1)
        inside = geometry_mask(
            [chip], out_shape=clear.shape, transform=src.window_transform(window), invert=True
        )

    chip_px = int(np.count_nonzero(inside))
    if chip_px == 0:
        return SCLChipStats(nodata_pct=100.0, cloud_pct=0.0)
    valid = inside & ((unusable & 1) == 0)
    valid_count = int(np.count_nonzero(valid))
    nodata_pct = (chip_px - valid_count) / chip_px * 100.0
    if valid_count == 0:
        return SCLChipStats(nodata_pct=float(nodata_pct), cloud_pct=0.0)
    not_clear = int(np.count_nonzero((clear != 1) & valid))
    return SCLChipStats(nodata_pct=float(nodata_pct), cloud_pct=not_clear / valid_count * 100.0)


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
