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
from rasterio.transform import from_bounds as transform_from_bounds
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling

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

    from affine import Affine

    from ftw_dataset_tools.api.imagery.scene_selection import SelectedScene

__all__ = [
    "DownloadResult",
    "download_and_clip_scene",
    "find_reference_mask_for_output",
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


def find_reference_mask_for_output(output_path: Path) -> Path | None:
    """Find a co-located mask raster to use as reference grid for imagery.

    Prefers semantic 3-class masks, then semantic 2-class, then instance masks.

    Args:
        output_path: Target imagery output path (e.g. ``chip_001_2024_planting_image_s2.tif``)

    Returns:
        Path to reference mask if available, otherwise ``None``.
    """
    stem = output_path.stem

    if stem.endswith("_planting_image_s2"):
        base_id = stem.removesuffix("_planting_image_s2")
    elif stem.endswith("_harvest_image_s2"):
        base_id = stem.removesuffix("_harvest_image_s2")
    else:
        return None

    candidates = [
        output_path.parent / f"{base_id}_semantic_3_class.tif",
        output_path.parent / f"{base_id}_semantic_2_class.tif",
        output_path.parent / f"{base_id}_instance.tif",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def compute_target_grid(
    bbox: tuple[float, float, float, float],
    output_path: Path,
    resolution: float,
    reference_raster: Path | None,
    on_progress: Callable[[str], None] | None = None,
) -> tuple[tuple[CRS, Affine, int, int, Path | None] | None, str | None]:
    """Compute output grid from reference mask or bbox+resolution fallback."""

    def log(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    minx, miny, maxx, maxy = bbox
    reference_grid = reference_raster or find_reference_mask_for_output(output_path)

    if reference_grid is not None:
        try:
            with rasterio.open(reference_grid) as reference_dataset:
                target_crs = reference_dataset.crs
                target_transform = reference_dataset.transform
                target_width = reference_dataset.width
                target_height = reference_dataset.height
                if target_crs is None:
                    raise ValueError("Reference raster has no CRS")

            log(
                f"Grid: source=reference_mask path={reference_grid.name} "
                f"crs={target_crs} width={target_width} height={target_height}"
            )
            log(f"Grid: using reference transform={target_transform}")
            log(
                f"Grid: requested resolution={resolution}m ignored because "
                "reference mask defines output grid"
            )
            log(
                f"Using reference mask grid: {reference_grid.name} "
                f"({target_width}x{target_height}, {target_crs})"
            )

            return (
                (target_crs, target_transform, target_width, target_height, reference_grid),
                None,
            )
        except Exception as error:
            return None, f"Failed to use reference raster {reference_grid}: {error}"

    if resolution <= 0:
        return (
            None,
            "Invalid resolution for fallback grid construction: "
            f"{resolution}. Resolution must be > 0 when no reference mask grid is found.",
        )

    lat_center = (miny + maxy) / 2
    meters_per_degree_lon = 111320 * np.cos(np.radians(lat_center))
    meters_per_degree_lat = 111320

    width_meters = (maxx - minx) * meters_per_degree_lon
    height_meters = (maxy - miny) * meters_per_degree_lat

    target_width = max(1, int(width_meters / resolution))
    target_height = max(1, int(height_meters / resolution))
    target_crs = CRS.from_epsg(4326)
    target_transform = transform_from_bounds(minx, miny, maxx, maxy, target_width, target_height)

    log(
        f"Grid: source=fallback_bbox_resolution bbox=({minx:.6f}, {miny:.6f}, "
        f"{maxx:.6f}, {maxy:.6f}) resolution_m={resolution}"
    )
    log(
        f"Grid: computed crs={target_crs} width={target_width} "
        f"height={target_height} transform={target_transform}"
    )
    log(f"No reference mask found, using EPSG:4326 fallback grid: {target_width}x{target_height}")

    return (target_crs, target_transform, target_width, target_height, None), None


def read_single_band(
    band_name: str,
    href: str,
    target_crs: CRS,
    target_transform: Affine,
    target_width: int,
    target_height: int,
) -> tuple[str, np.ndarray]:
    """Read one band reprojected to the target grid."""
    resampling = (
        Resampling.nearest if band_name in {"scl", "cloud", "snow"} else Resampling.bilinear
    )

    with (
        rasterio.open(href) as source_dataset,
        WarpedVRT(
            source_dataset,
            crs=target_crs,
            transform=target_transform,
            width=target_width,
            height=target_height,
            resampling=resampling,
        ) as warped_dataset,
    ):
        data = warped_dataset.read(1)

    return band_name, data


def write_cog(
    output_path: Path,
    stacked: np.ndarray,
    found_bands: list[str],
    profile: dict,
) -> str | None:
    """Write stacked imagery as a COG and set band descriptions."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with rasterio.open(output_path, "w", **profile) as destination_dataset:
            destination_dataset.write(stacked)
            for band_index, band_name in enumerate(found_bands, start=1):
                destination_dataset.set_band_description(band_index, band_name)
    except Exception as error:
        return f"Failed to write output: {error}"

    return None


def validate_alignment(output_path: Path, reference_grid: Path | None) -> str | None:
    """Validate output alignment against reference mask; cleanup output on failure."""
    if reference_grid is None:
        return None

    try:
        with (
            rasterio.open(output_path) as output_dataset,
            rasterio.open(reference_grid) as reference_dataset,
        ):
            if (
                output_dataset.crs != reference_dataset.crs
                or output_dataset.transform != reference_dataset.transform
                or output_dataset.width != reference_dataset.width
                or output_dataset.height != reference_dataset.height
            ):
                output_path.unlink(missing_ok=True)
                return f"Output image does not match reference mask grid ({reference_grid.name})"
    except Exception as error:
        output_path.unlink(missing_ok=True)
        return f"Failed to validate output alignment: {error}"

    return None


def download_and_clip_scene(
    scene: SelectedScene,
    bbox: tuple[float, float, float, float],
    output_path: Path,
    bands: list[str] | None = None,
    resolution: float = 10.0,
    reference_raster: Path | None = None,
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
        reference_raster: Optional mask raster path used as exact output grid
            reference (CRS, transform, width, height). If not provided, the
            function auto-detects a co-located mask from ``output_path``.
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

    target_grid, target_grid_error = compute_target_grid(
        bbox=bbox,
        output_path=output_path,
        resolution=resolution,
        reference_raster=reference_raster,
        on_progress=on_progress,
    )

    if target_grid_error is not None or target_grid is None:
        return DownloadResult(
            output_path=output_path,
            scene_id=scene.id,
            season=scene.season,
            bands=bands,
            width=0,
            height=0,
            crs="",
            success=False,
            error=target_grid_error,
        )

    target_crs, target_transform, target_width, target_height, reference_grid = target_grid

    log(f"Target dimensions: {target_width}x{target_height} pixels")

    # Read bands in parallel for faster network throughput
    log(f"Reading {len(found_bands)} bands in parallel...")
    band_results: dict[str, np.ndarray] = {}
    failed_band = None
    failed_error = None

    with ThreadPoolExecutor(max_workers=min(4, len(found_bands))) as executor:
        futures = {
            executor.submit(
                read_single_band,
                band_name,
                href,
                target_crs,
                target_transform,
                target_width,
                target_height,
            ): band_name
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

    profile = {
        "driver": "COG",
        "dtype": stacked.dtype,
        "width": target_width,
        "height": target_height,
        "count": len(found_bands),
        "crs": target_crs,
        "transform": target_transform,
        "compress": "deflate",
    }

    log(f"Writing to {output_path}...")

    write_error = write_cog(output_path, stacked, found_bands, profile)
    if write_error is not None:
        return DownloadResult(
            output_path=output_path,
            scene_id=scene.id,
            season=scene.season,
            bands=bands,
            width=0,
            height=0,
            crs="",
            success=False,
            error=write_error,
        )

    log(f"Successfully wrote {output_path}")

    alignment_error = validate_alignment(output_path, reference_grid)
    if alignment_error is not None:
        return DownloadResult(
            output_path=output_path,
            scene_id=scene.id,
            season=scene.season,
            bands=bands,
            width=0,
            height=0,
            crs="",
            success=False,
            error=alignment_error,
        )

    return DownloadResult(
        output_path=output_path,
        scene_id=scene.id,
        season=scene.season,
        bands=found_bands,
        width=target_width,
        height=target_height,
        crs=str(target_crs),
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
