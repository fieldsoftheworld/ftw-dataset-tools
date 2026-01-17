"""Constants and configuration for the imagery pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# STAC Host URLs
EARTHSEARCH_URL = "https://earth-search.aws.element84.com/v1"
MSPC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

# Sentinel-2 collection identifiers
S2_COLLECTIONS = {
    "earthsearch": {
        "old-baseline": "sentinel-2-l2a",
        "c1": "sentinel-2-c1-l2a",
    },
    "mspc": {
        "default": "sentinel-2-l2a",
    },
}

# STAC host configuration
STAC_HOSTS = {
    "earthsearch": {
        "url": EARTHSEARCH_URL,
        "default_collection": "sentinel-2-c1-l2a",
        "bands": ["red", "green", "blue", "nir"],
    },
    "mspc": {
        "url": MSPC_URL,
        "default_collection": "sentinel-2-l2a",
        "bands": ["B04", "B03", "B02", "B08"],  # MSPC uses band codes
    },
}

# Default bands of interest (EarthSearch naming)
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
DEFAULT_CLOUD_COVER_SCENE = 10  # Maximum scene-level cloud cover percentage
DEFAULT_CLOUD_COVER_PIXEL = 0  # Maximum pixel-level cloud cover percentage
DEFAULT_BUFFER_DAYS = 14  # Days to search around crop calendar dates
DEFAULT_STAC_HOST = "earthsearch"

# Hybrid cloud filtering thresholds
# Skip pixel check if scene cloud cover is below this (too clear to matter)
PIXEL_CHECK_SKIP_THRESHOLD = 0.1
# Don't bother with pixel check if scene cloud cover is above this
PIXEL_CHECK_MAX_SCENE_THRESHOLD = 50.0


@dataclass
class STACHostConfig:
    """Configuration for a STAC host."""

    name: str
    url: str
    collection: str
    bands: list[str]


def get_stac_host_config(
    host: Literal["earthsearch", "mspc"] = "earthsearch",
    s2_collection: str = "c1",
) -> STACHostConfig:
    """
    Get STAC host configuration.

    Args:
        host: STAC host name ("earthsearch" or "mspc")
        s2_collection: Sentinel-2 collection identifier (only used for earthsearch)

    Returns:
        STACHostConfig with URL, collection, and band names
    """
    if host == "earthsearch":
        collection = S2_COLLECTIONS["earthsearch"].get(
            s2_collection, S2_COLLECTIONS["earthsearch"]["c1"]
        )
        return STACHostConfig(
            name="earthsearch",
            url=EARTHSEARCH_URL,
            collection=collection,
            bands=STAC_HOSTS["earthsearch"]["bands"],
        )
    elif host == "mspc":
        return STACHostConfig(
            name="mspc",
            url=MSPC_URL,
            collection=S2_COLLECTIONS["mspc"]["default"],
            bands=STAC_HOSTS["mspc"]["bands"],
        )
    else:
        raise ValueError(f"Unknown STAC host: {host}. Use 'earthsearch' or 'mspc'.")
