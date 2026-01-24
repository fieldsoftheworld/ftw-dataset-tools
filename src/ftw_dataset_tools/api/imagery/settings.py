"""Constants and configuration for the imagery pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass

# GDAL HTTP optimization for remote COG access
os.environ.update(
    {
        "GDAL_HTTP_MULTIPLEX": "YES",
        "GDAL_HTTP_VERSION": "2",
        "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
        "GDAL_HTTP_MAX_RETRY": "3",
        "GDAL_HTTP_TIMEOUT": "30",
        "VSI_CACHE": "TRUE",
        "VSI_CACHE_SIZE": "50000000",  # 50MB cache
    }
)

# STAC configuration - EarthSearch is the only supported host
STAC_URL = "https://earth-search.aws.element84.com/v1"

# Sentinel-2 collection identifiers
S2_COLLECTIONS = {
    "old-baseline": "sentinel-2-l2a",
    "c1": "sentinel-2-c1-l2a",
}

# Default bands of interest
BANDS_OF_INTEREST = ["red", "green", "blue", "nir"]

# Cloud probability band (for pixel-level cloud filtering)
CLOUD_PROBABILITY_BAND = "cloud_probability"

# Crop Calendar Configuration
CROP_CALENDAR_BASE_URL = "https://data.source.coop/ftw/ftw-inference-input/global-crop-calendar/"

# Crop calendar files (summer crop only for now)
CROP_CAL_SUMMER_START = "sc-sos-3x3-v2-cog.tiff"
CROP_CAL_SUMMER_END = "sc-eos-3x3-v2-cog.tiff"
CROP_CAL_WINTER_START = "wc-sos-3x3-v2-cog.tiff"
CROP_CAL_WINTER_END = "wc-eos-3x3-v2-cog.tiff"

CROP_CALENDAR_FILES = [
    CROP_CAL_SUMMER_START,
    CROP_CAL_SUMMER_END,
    CROP_CAL_WINTER_START,
    CROP_CAL_WINTER_END,
]

# Default parameter values
DEFAULT_CLOUD_COVER_SCENE = 75  # Internal scene-level filter for STAC query
DEFAULT_CLOUD_COVER_CHIP = 2  # Maximum chip-level cloud cover percentage
DEFAULT_NODATA_MAX = 0  # Maximum nodata percentage (0 = reject any nodata)
DEFAULT_BUFFER_DAYS = 14  # Days to search around crop calendar dates
DEFAULT_NUM_BUFFER_EXPANSIONS = 3  # Number of times to expand buffer for cloudy chips
DEFAULT_BUFFER_EXPANSION_SIZE = 14  # Days to add on each buffer expansion

# Hybrid cloud filtering thresholds
# Skip pixel check if scene cloud cover is below this (too clear to matter)
PIXEL_CHECK_SKIP_THRESHOLD = 0.1
# Don't bother with pixel check if scene cloud cover is above this
PIXEL_CHECK_MAX_SCENE_THRESHOLD = 50.0


@dataclass
class STACHostConfig:
    """Configuration for EarthSearch STAC host."""

    name: str
    url: str
    collection: str
    bands: list[str]


def get_stac_host_config(s2_collection: str = "c1") -> STACHostConfig:
    """
    Get STAC host configuration (EarthSearch).

    Args:
        s2_collection: Sentinel-2 collection identifier ("c1" or "old-baseline")

    Returns:
        STACHostConfig with URL, collection, and band names
    """
    collection = S2_COLLECTIONS.get(s2_collection, S2_COLLECTIONS["c1"])
    return STACHostConfig(
        name="earthsearch",
        url=STAC_URL,
        collection=collection,
        bands=BANDS_OF_INTEREST,
    )
