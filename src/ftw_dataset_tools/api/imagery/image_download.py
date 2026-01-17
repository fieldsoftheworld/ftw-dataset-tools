"""Download and clip satellite imagery from STAC items."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.warp import Resampling, calculate_default_transform, reproject
from rasterio.windows import from_bounds

from ftw_dataset_tools.api.imagery.settings import BANDS_OF_INTEREST

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    import pystac

    from ftw_dataset_tools.api.imagery.scene_selection import SelectedScene

__all__ = [
    "DownloadResult",
    "download_and_clip_scene",
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

    # Read each band
    band_data = []
    for band_name in found_bands:
        href = band_hrefs[band_name]
        log(f"Reading {band_name}...")

        try:
            # For COGs, we can read windowed data directly
            with rasterio.open(href) as src:
                # Transform bbox to source CRS if needed
                if source_crs != CRS.from_epsg(4326):
                    from rasterio.warp import transform_bounds

                    src_bbox = transform_bounds(
                        CRS.from_epsg(4326), source_crs, minx, miny, maxx, maxy
                    )
                else:
                    src_bbox = bbox

                window = from_bounds(*src_bbox, src.transform)
                data = src.read(
                    1,
                    window=window,
                    out_shape=(target_height, target_width),
                    resampling=Resampling.bilinear,
                )
                band_data.append(data)
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
                error=f"Failed to read band {band_name}: {e}",
            )

    # Stack bands
    stacked = np.stack(band_data, axis=0)
    log(f"Stacked shape: {stacked.shape}")

    # Calculate transform for output
    if source_crs != CRS.from_epsg(4326):
        from rasterio.warp import transform_bounds

        out_bbox = transform_bounds(CRS.from_epsg(4326), source_crs, minx, miny, maxx, maxy)
    else:
        out_bbox = bbox

    from rasterio.transform import from_bounds as transform_from_bounds

    out_transform = transform_from_bounds(*out_bbox, target_width, target_height)

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
