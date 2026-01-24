"""Download and clip satellite imagery from STAC items."""

from __future__ import annotations

import contextlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import pystac
import rasterio
from rasterio.crs import CRS
from rasterio.warp import Resampling, calculate_default_transform, reproject
from rasterio.windows import from_bounds

from ftw_dataset_tools.api.imagery.settings import BANDS_OF_INTEREST
from ftw_dataset_tools.api.imagery.thumbnails import (
    ThumbnailError,
    generate_overlay_thumbnail,
    generate_thumbnail,
    has_rgb_bands,
)
from ftw_dataset_tools.api.stac_items import update_parent_item

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from ftw_dataset_tools.api.imagery.scene_selection import SelectedScene

__all__ = [
    "DownloadResult",
    "download_and_clip_scene",
    "process_downloaded_scene",
]


@dataclass
class DownloadResult:
    """Result of downloading and clipping a scene."""

    output_path: Path
    scene_id: str
    season: Literal["planting", "harvest"]
    bands: list[str]
    width: int
    height: int
    crs: str
    success: bool = True
    error: str | None = None


def _get_band_hrefs(
    item: pystac.Item,
    bands: list[str],
) -> dict[str, str]:
    """
    Get hrefs for specified bands from a STAC item.

    Args:
        item: STAC item
        bands: List of band names to get

    Returns:
        Dict mapping band name to href
    """
    hrefs = {}
    for band in bands:
        asset = item.assets.get(band)
        if asset:
            hrefs[band] = asset.href
    return hrefs


def _read_band_windowed(
    href: str,
    bbox: tuple[float, float, float, float],
    target_crs: CRS,
    target_width: int,
    target_height: int,
) -> np.ndarray:
    """
    Read a band from a COG, clipping to bbox and reprojecting if needed.

    Args:
        href: URL or path to the COG
        bbox: Target bounding box (minx, miny, maxx, maxy) in target_crs
        target_crs: Target CRS
        target_width: Target width in pixels
        target_height: Target height in pixels

    Returns:
        numpy array with band data
    """
    minx, miny, maxx, maxy = bbox

    with rasterio.open(href) as src:
        # Check if we need to reproject
        if src.crs != target_crs:
            # Calculate transform for target
            transform, width, height = calculate_default_transform(
                src.crs,
                target_crs,
                target_width,
                target_height,
                *bbox,
            )

            # Read and reproject
            data = np.empty((height, width), dtype=src.dtypes[0])
            reproject(
                source=rasterio.band(src, 1),
                destination=data,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=target_crs,
                resampling=Resampling.bilinear,
            )
            return data
        else:
            # Read windowed data directly
            window = from_bounds(minx, miny, maxx, maxy, src.transform)
            data = src.read(1, window=window, out_shape=(target_height, target_width))
            return data


def download_and_clip_scene(
    scene: SelectedScene,
    bbox: tuple[float, float, float, float],
    output_path: Path,
    bands: list[str] | None = None,
    resolution: float = 10.0,
    on_progress: Callable[[str], None] | None = None,
) -> DownloadResult:
    """
    Download and clip a scene to the specified bounding box.

    Args:
        scene: Selected scene with STAC item
        bbox: Bounding box (minx, miny, maxx, maxy) in EPSG:4326
        output_path: Path for output GeoTIFF
        bands: Bands to download (default: red, green, blue, nir)
        resolution: Target resolution in meters (default: 10.0)
        on_progress: Optional callback for progress messages

    Returns:
        DownloadResult with output information
    """
    if bands is None:
        bands = BANDS_OF_INTEREST.copy()

    def log(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    log(f"Downloading {scene.id} bands: {bands}")

    # Get band hrefs
    band_hrefs = _get_band_hrefs(scene.item, bands)
    if not band_hrefs:
        return DownloadResult(
            output_path=output_path,
            scene_id=scene.id,
            season=scene.season,
            bands=bands,
            width=0,
            height=0,
            crs="",
            success=False,
            error=f"No matching band assets found in scene. Available: {list(scene.item.assets.keys())}",
        )

    found_bands = list(band_hrefs.keys())
    log(f"Found {len(found_bands)} bands: {found_bands}")

    minx, miny, maxx, maxy = bbox

    # Calculate target dimensions based on resolution
    # Approximate degrees to meters conversion at the equator
    # For more accuracy, this should consider latitude
    lat_center = (miny + maxy) / 2
    meters_per_degree_lon = 111320 * np.cos(np.radians(lat_center))
    meters_per_degree_lat = 111320

    width_meters = (maxx - minx) * meters_per_degree_lon
    height_meters = (maxy - miny) * meters_per_degree_lat

    target_width = max(1, int(width_meters / resolution))
    target_height = max(1, int(height_meters / resolution))

    log(f"Target dimensions: {target_width}x{target_height} pixels")

    # Use UTM CRS for the output (based on center of bbox)
    # For simplicity, we'll use the CRS of the first band
    first_href = next(iter(band_hrefs.values()))

    try:
        with rasterio.open(first_href) as src:
            source_crs = src.crs
    except Exception as e:
        return DownloadResult(
            output_path=output_path,
            scene_id=scene.id,
            season=scene.season,
            bands=bands,
            width=0,
            height=0,
            crs="",
            success=False,
            error=f"Failed to open source imagery: {e}",
        )

    # Transform bbox to source CRS once (used by all bands)
    if source_crs != CRS.from_epsg(4326):
        from rasterio.warp import transform_bounds

        src_bbox = transform_bounds(CRS.from_epsg(4326), source_crs, minx, miny, maxx, maxy)
    else:
        src_bbox = bbox

    def read_single_band(band_name: str, href: str) -> tuple[str, np.ndarray]:
        """Read a single band from a COG."""
        with rasterio.open(href) as src:
            window = from_bounds(*src_bbox, src.transform)
            data = src.read(
                1,
                window=window,
                out_shape=(target_height, target_width),
                resampling=Resampling.bilinear,
            )
            return band_name, data

    # Read bands in parallel for faster network throughput
    log(f"Reading {len(found_bands)} bands in parallel...")
    band_results: dict[str, np.ndarray] = {}
    failed_band = None
    failed_error = None

    with ThreadPoolExecutor(max_workers=min(4, len(found_bands))) as executor:
        futures = {
            executor.submit(read_single_band, band_name, href): band_name
            for band_name, href in band_hrefs.items()
        }

        for future in as_completed(futures):
            band_name = futures[future]
            try:
                name, data = future.result()
                band_results[name] = data
            except Exception as e:
                failed_band = band_name
                failed_error = str(e)
                # Cancel remaining futures
                for f in futures:
                    f.cancel()
                break

    if failed_band is not None:
        return DownloadResult(
            output_path=output_path,
            scene_id=scene.id,
            season=scene.season,
            bands=bands,
            width=0,
            height=0,
            crs="",
            success=False,
            error=f"Failed to read band {failed_band}: {failed_error}",
        )

    # Stack bands in the original order
    band_data = [band_results[band_name] for band_name in found_bands]

    # Stack bands
    stacked = np.stack(band_data, axis=0)
    log(f"Stacked shape: {stacked.shape}")

    # Calculate transform for output (reuse src_bbox computed earlier)
    from rasterio.transform import from_bounds as transform_from_bounds

    out_transform = transform_from_bounds(*src_bbox, target_width, target_height)

    # Write output as COG
    output_path.parent.mkdir(parents=True, exist_ok=True)

    profile = {
        "driver": "COG",
        "dtype": stacked.dtype,
        "width": target_width,
        "height": target_height,
        "count": len(found_bands),
        "crs": source_crs,
        "transform": out_transform,
        "compress": "deflate",
    }

    log(f"Writing to {output_path}...")

    try:
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(stacked)
            # Add band descriptions
            for i, band_name in enumerate(found_bands, start=1):
                dst.set_band_description(i, band_name)
    except Exception as e:
        return DownloadResult(
            output_path=output_path,
            scene_id=scene.id,
            season=scene.season,
            bands=bands,
            width=0,
            height=0,
            crs="",
            success=False,
            error=f"Failed to write output: {e}",
        )

    log(f"Successfully wrote {output_path}")

    return DownloadResult(
        output_path=output_path,
        scene_id=scene.id,
        season=scene.season,
        bands=found_bands,
        width=target_width,
        height=target_height,
        crs=str(source_crs),
        success=True,
    )


@dataclass
class ProcessedSceneResult:
    """Result of processing a downloaded scene."""

    thumbnail_path: Path | None = None
    overlay_path: Path | None = None
    is_overlay: bool = False


def process_downloaded_scene(
    item: pystac.Item,
    item_path: Path,
    output_path: Path,
    output_filename: str,
    band_list: list[str],
    season: Literal["planting", "harvest"],
    base_id: str,
    generate_thumbnails: bool = True,
) -> ProcessedSceneResult:
    """Process a downloaded scene: update STAC assets, generate thumbnails, update parent.

    This function handles all post-download processing that should be identical
    between `download-images` command and `create-dataset --download-images`.

    Args:
        item: Child STAC item to update
        item_path: Path to the child item JSON file
        output_path: Path to the downloaded image file
        output_filename: Filename of the downloaded image
        band_list: List of bands in the downloaded image
        season: Season identifier ("planting" or "harvest")
        base_id: Base chip ID (without season suffix)
        generate_thumbnails: Whether to generate thumbnails

    Returns:
        ProcessedSceneResult with paths to generated files
    """
    result = ProcessedSceneResult()

    # Replace remote band assets with single local "image" asset
    for band in band_list:
        item.assets.pop(band, None)
    item.assets["image"] = pystac.Asset(
        href=f"./{output_filename}",
        media_type="image/tiff; application=geotiff; profile=cloud-optimized",
        title=f"Clipped {len(band_list)}-band image ({','.join(band_list)})",
        roles=["data"],
    )

    # Generate thumbnail if RGB bands available
    if generate_thumbnails and has_rgb_bands(band_list):
        try:
            thumbnail_filename = output_filename.replace(".tif", ".jpg")
            thumbnail_path = output_path.parent / thumbnail_filename
            generate_thumbnail(output_path, thumbnail_path)
            item.assets["thumbnail"] = pystac.Asset(
                href=f"./{thumbnail_filename}",
                media_type=pystac.MediaType.JPEG,
                title="JPEG preview",
                roles=["thumbnail"],
            )
            result.thumbnail_path = thumbnail_path
        except ThumbnailError:
            pass

    # Save the child item
    item.save_object(str(item_path))

    # Update parent chip item with asset reference
    parent_item_path = item_path.parent / f"{base_id}.json"
    if parent_item_path.exists():
        parent_item = pystac.Item.from_file(str(parent_item_path))

        # Generate overlay thumbnail for planting season if mask exists
        thumb_for_parent = None
        is_overlay = False

        if result.thumbnail_path and season == "planting":
            # Look for semantic 3-class mask
            mask_path = item_path.parent / f"{base_id}_semantic_3_class.tif"
            if mask_path.exists():
                try:
                    overlay_filename = f"{base_id}_overlay.jpg"
                    overlay_path = item_path.parent / overlay_filename
                    generate_overlay_thumbnail(result.thumbnail_path, mask_path, overlay_path)
                    thumb_for_parent = overlay_filename
                    is_overlay = True
                    result.overlay_path = overlay_path
                    result.is_overlay = True
                except ThumbnailError:
                    # Fallback to plain thumbnail
                    thumb_for_parent = result.thumbnail_path.name
            else:
                thumb_for_parent = result.thumbnail_path.name

        # Suppress errors - child item was saved successfully
        with contextlib.suppress(Exception):
            update_parent_item(
                parent_item=parent_item,
                parent_path=parent_item_path,
                season=season,
                output_filename=output_filename,
                band_list=band_list,
                thumbnail_filename=thumb_for_parent,
                is_overlay=is_overlay,
            )

    return result
