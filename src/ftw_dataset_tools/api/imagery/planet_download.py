"""Download and clip Planet imagery using the Planet SDK.

This module provides asset activation and download functionality for PlanetScope imagery,
handling the async activation workflow with polling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds as transform_from_bounds
from rasterio.warp import Resampling, transform_bounds
from rasterio.windows import from_bounds

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from ftw_dataset_tools.api.imagery.planet_client import PlanetClient

__all__ = [
    "PlanetDownloadResult",
    "activate_asset",
    "download_and_clip_planet_scene",
    "wait_for_activation",
]


# Default asset type for PSScene (4-band analytic)
DEFAULT_ASSET_TYPE = "ortho_analytic_4b"

# Valid asset types for PSScene
VALID_ASSET_TYPES = [
    "ortho_analytic_4b",
    "ortho_analytic_4b_sr",
    "ortho_analytic_8b",
    "ortho_analytic_8b_sr",
    "ortho_visual",
    "ortho_udm2",
]


@dataclass
class PlanetDownloadResult:
    """Result of downloading and clipping a Planet scene."""

    output_path: Path
    scene_id: str
    season: Literal["planting", "harvest"]
    bands: list[str]
    width: int
    height: int
    crs: str
    success: bool = True
    error: str | None = None


def activate_asset(
    client: PlanetClient,
    item_id: str,
    asset_type: str = DEFAULT_ASSET_TYPE,
    item_type: str = "PSScene",
) -> dict:
    """Start asset activation (non-blocking).

    Args:
        client: Authenticated Planet client
        item_id: Planet scene ID
        asset_type: Asset type to activate (default: ortho_analytic_4b)
        item_type: Item type (default: PSScene)

    Returns:
        Asset status dict with activation state
    """
    pl = client.get_sdk_client()

    # Get asset info
    asset = pl.data.get_asset(item_type, item_id, asset_type)

    # Activate if not already active
    if asset.get("status") != "active":
        pl.data.activate_asset(asset)

    return asset


def wait_for_activation(
    client: PlanetClient,
    item_id: str,
    asset_type: str = DEFAULT_ASSET_TYPE,
    item_type: str = "PSScene",
    timeout: int = 3600,
    on_progress: Callable[[str], None] | None = None,
) -> dict:
    """Wait for asset activation with polling.

    Args:
        client: Authenticated Planet client
        item_id: Planet scene ID
        asset_type: Asset type to wait for
        item_type: Item type
        timeout: Maximum wait time in seconds (default: 1 hour)
        on_progress: Optional callback for progress messages

    Returns:
        Asset status dict when active

    Raises:
        TimeoutError: If asset doesn't activate within timeout
    """
    pl = client.get_sdk_client()

    def log(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    # Get current asset state
    asset = pl.data.get_asset(item_type, item_id, asset_type)

    # If already active, return immediately
    if asset.get("status") == "active":
        log(f"Asset {item_id}/{asset_type} already active")
        return asset

    # Start activation if needed
    if asset.get("status") != "activating":
        pl.data.activate_asset(asset)
        log(f"Started activation for {item_id}/{asset_type}")

    # Wait with callback
    def status_callback(asset_status: dict) -> None:
        status = asset_status.get("status", "unknown")
        log(f"Waiting for {item_id}/{asset_type}: {status}")

    try:
        asset = pl.data.wait_asset(
            asset,
            callback=status_callback if on_progress else None,
            timeout=timeout,
        )
        log(f"Asset {item_id}/{asset_type} is now active")
        return asset
    except Exception as e:
        raise TimeoutError(f"Asset activation timed out after {timeout}s: {e}") from e


def download_and_clip_planet_scene(
    client: PlanetClient,
    item_id: str,
    bbox: tuple[float, float, float, float],
    output_path: Path,
    asset_type: str = DEFAULT_ASSET_TYPE,
    item_type: str = "PSScene",
    bands: list[str] | None = None,
    resolution: float = 3.0,
    season: Literal["planting", "harvest"] = "planting",
    on_progress: Callable[[str], None] | None = None,
) -> PlanetDownloadResult:
    """Download asset, clip to bbox, and return result.

    Args:
        client: Authenticated Planet client
        item_id: Planet scene ID
        bbox: Bounding box (minx, miny, maxx, maxy) in EPSG:4326
        output_path: Path for output GeoTIFF
        asset_type: Asset type to download
        item_type: Item type
        bands: Band indices to extract (default: all)
        resolution: Target resolution in meters (default: 3.0 for PlanetScope)
        season: Season identifier for result
        on_progress: Optional callback for progress messages

    Returns:
        PlanetDownloadResult with output information
    """

    def log(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    log(f"Downloading {item_id} asset {asset_type}")

    # Ensure asset is active
    try:
        asset = wait_for_activation(
            client=client,
            item_id=item_id,
            asset_type=asset_type,
            item_type=item_type,
            on_progress=on_progress,
        )
    except TimeoutError as e:
        return PlanetDownloadResult(
            output_path=output_path,
            scene_id=item_id,
            season=season,
            bands=bands or [],
            width=0,
            height=0,
            crs="",
            success=False,
            error=str(e),
        )

    # Get download URL from asset
    download_url = asset.get("location")
    if not download_url:
        return PlanetDownloadResult(
            output_path=output_path,
            scene_id=item_id,
            season=season,
            bands=bands or [],
            width=0,
            height=0,
            crs="",
            success=False,
            error="No download location in activated asset",
        )

    log(f"Reading from {download_url[:50]}...")

    minx, miny, maxx, maxy = bbox

    # Calculate target dimensions based on resolution
    lat_center = (miny + maxy) / 2
    meters_per_degree_lon = 111320 * np.cos(np.radians(lat_center))
    meters_per_degree_lat = 111320

    width_meters = (maxx - minx) * meters_per_degree_lon
    height_meters = (maxy - miny) * meters_per_degree_lat

    target_width = max(1, int(width_meters / resolution))
    target_height = max(1, int(height_meters / resolution))

    log(f"Target dimensions: {target_width}x{target_height} pixels")

    try:
        with rasterio.open(download_url) as src:
            source_crs = src.crs
            band_count = src.count

            # Transform bbox to source CRS if needed
            if source_crs != CRS.from_epsg(4326):
                src_bbox = transform_bounds(CRS.from_epsg(4326), source_crs, minx, miny, maxx, maxy)
            else:
                src_bbox = bbox

            # Determine which bands to read
            if bands:
                # Map band names to indices (1-indexed)
                band_mapping = {
                    "blue": 1,
                    "green": 2,
                    "red": 3,
                    "nir": 4,
                    "coastal_blue": 1,  # 8-band
                    "green_i": 4,  # 8-band
                    "yellow": 5,  # 8-band
                    "red_edge": 6,  # 8-band
                }
                band_indices = [band_mapping.get(b, i + 1) for i, b in enumerate(bands)]
                band_indices = [i for i in band_indices if i <= band_count]
            else:
                # Read all bands
                band_indices = list(range(1, band_count + 1))
                bands = [f"band_{i}" for i in band_indices]

            log(f"Reading {len(band_indices)} bands...")

            # Read windowed data
            window = from_bounds(*src_bbox, src.transform)
            data = src.read(
                band_indices,
                window=window,
                out_shape=(len(band_indices), target_height, target_width),
                resampling=Resampling.bilinear,
            )

    except Exception as e:
        return PlanetDownloadResult(
            output_path=output_path,
            scene_id=item_id,
            season=season,
            bands=bands or [],
            width=0,
            height=0,
            crs="",
            success=False,
            error=f"Failed to read source imagery: {e}",
        )

    # Calculate output transform
    out_transform = transform_from_bounds(*src_bbox, target_width, target_height)

    # Write output as COG
    output_path.parent.mkdir(parents=True, exist_ok=True)

    profile = {
        "driver": "COG",
        "dtype": data.dtype,
        "width": target_width,
        "height": target_height,
        "count": len(band_indices),
        "crs": source_crs,
        "transform": out_transform,
        "compress": "deflate",
    }

    log(f"Writing to {output_path}...")

    try:
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data)
            # Add band descriptions
            for i, band_name in enumerate(bands, start=1):
                dst.set_band_description(i, band_name)
    except Exception as e:
        return PlanetDownloadResult(
            output_path=output_path,
            scene_id=item_id,
            season=season,
            bands=bands,
            width=0,
            height=0,
            crs="",
            success=False,
            error=f"Failed to write output: {e}",
        )

    log(f"Successfully wrote {output_path}")

    return PlanetDownloadResult(
        output_path=output_path,
        scene_id=item_id,
        season=season,
        bands=bands,
        width=target_width,
        height=target_height,
        crs=str(source_crs),
        success=True,
    )
