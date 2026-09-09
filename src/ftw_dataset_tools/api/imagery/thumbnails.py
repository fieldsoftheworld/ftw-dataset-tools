"""JPEG thumbnail generation for satellite imagery."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from PIL import Image
from rasterio.enums import Resampling
from rasterio.errors import RasterioIOError
from rasterio.vrt import WarpedVRT


class ThumbnailError(Exception):
    """Error generating thumbnail."""


# Default colors for semantic 3-class mask overlay
DEFAULT_MASK_COLORS: dict[int, tuple[int, int, int]] = {
    1: (0, 200, 0),  # Green for field interiors
    2: (255, 165, 0),  # Orange for boundaries
}


# Bands that can be used for RGB thumbnail (in priority order)
RGB_BAND_SETS = [
    ("red", "green", "blue"),
    ("nir", "red", "green"),  # False color composite
]


def has_rgb_bands(band_list: list[str]) -> bool:
    """Check if band list contains bands suitable for RGB thumbnail.

    Args:
        band_list: List of band names

    Returns:
        True if RGB thumbnail can be generated
    """
    band_set = set(band_list)
    return any(all(b in band_set for b in rgb_bands) for rgb_bands in RGB_BAND_SETS)


def generate_thumbnail(
    tif_path: str | Path,
    output_path: str | Path,
    quality: int = 85,
) -> Path:
    """Generate JPEG thumbnail from a multi-band GeoTIFF.

    Args:
        tif_path: Path to GeoTIFF (must have at least 3 bands for RGB)
        output_path: Output path for JPEG
        quality: JPEG quality (1-100)

    Returns:
        Path to generated thumbnail

    Raises:
        ThumbnailError: If thumbnail generation fails
    """
    tif_path = Path(tif_path)
    output_path = Path(output_path)

    if not tif_path.exists():
        raise ThumbnailError(f"Input file does not exist: {tif_path}")

    try:
        with rasterio.open(tif_path) as src:
            if src.count < 3:
                raise ThumbnailError(f"Need at least 3 bands for RGB thumbnail, got {src.count}")

            # Read at native dimensions so the preview matches the model-input TIF.
            data = src.read(
                indexes=[1, 2, 3],
                masked=True,
            )

            # Handle nodata
            if np.ma.is_masked(data):
                data = data.filled(fill_value=0)

        # Normalize for display (percentile stretch)
        data = _normalize_for_display(data)

        # Convert to PIL image
        rgb_array = np.transpose(data, (1, 2, 0)).astype(np.uint8)
        img = Image.fromarray(rgb_array, mode="RGB")
        img.save(output_path, "JPEG", quality=quality, optimize=True)

    except RasterioIOError as e:
        raise ThumbnailError(f"Failed to read {tif_path}: {e}") from e
    except OSError as e:
        # Clean up partial output
        if output_path.exists():
            output_path.unlink()
        raise ThumbnailError(f"Failed to write thumbnail: {e}") from e

    return output_path


def _normalize_for_display(
    data: np.ndarray,
    percentile_clip: tuple[float, float] = (2, 98),
) -> np.ndarray:
    """Normalize array to 0-255 using per-band percentile stretching.

    Processing each band separately because percentiles are computed
    independently for proper color balance.
    """
    result = np.zeros_like(data, dtype=np.float32)
    for i in range(data.shape[0]):
        band = data[i].astype(np.float32)
        valid = band[band > 0]
        if len(valid) > 0:
            p_low, p_high = np.percentile(valid, percentile_clip)
            if p_high > p_low:
                band = np.clip((band - p_low) / (p_high - p_low) * 255, 0, 255)
        result[i] = band
    return result.astype(np.uint8)


def generate_scene_thumbnail(
    visual_href: str,
    reference_raster: str | Path,
    output_path: str | Path,
    max_size: int = 512,
    quality: int = 85,
) -> Path:
    """Write a JPEG preview of one chip read straight from a remote true-colour COG.

    The clipped-image path writes a 4-band GeoTIFF per chip and previews that. A
    dataset that keeps the scene COG remote has no such file, so read the chip's
    window out of the scene instead: a windowed, downsampled ``WarpedVRT`` read is a
    handful of range requests against the COG's overviews, not a whole-scene fetch.

    ``reference_raster`` is a raster already on the chip's own grid - one of its mask
    COGs - so the preview lands in exactly the same CRS, extent and aspect as the
    masks. That is what lets ``generate_overlay_thumbnail`` composite a mask over the
    result without any registration of its own.

    Args:
        visual_href: URL of the scene's true-colour (8-bit RGB) COG.
        reference_raster: A raster on the chip grid, used for CRS/extent/aspect.
        output_path: Output path for the JPEG.
        max_size: Longest edge of the preview in pixels.
        quality: JPEG quality (1-100).

    Returns:
        Path to the generated thumbnail.

    Raises:
        ThumbnailError: If the scene or reference cannot be read, or the write fails.
    """
    reference_raster = Path(reference_raster)
    output_path = Path(output_path)

    if not reference_raster.exists():
        raise ThumbnailError(f"Reference raster does not exist: {reference_raster}")

    try:
        with rasterio.open(reference_raster) as reference:
            target_crs = reference.crs
            width, height = reference.width, reference.height
            base_transform = reference.transform

        scale = max(width, height) / max_size
        if scale < 1:
            scale = 1.0
        out_width = max(1, round(width / scale))
        out_height = max(1, round(height / scale))
        out_transform = base_transform * Affine.scale(scale, scale)

        with (
            rasterio.open(visual_href) as scene,
            WarpedVRT(
                scene,
                crs=target_crs,
                transform=out_transform,
                width=out_width,
                height=out_height,
                resampling=Resampling.bilinear,
            ) as warped,
        ):
            if warped.count < 3:
                raise ThumbnailError(
                    f"Need at least 3 bands for an RGB preview, got {warped.count}"
                )
            data = warped.read(indexes=[1, 2, 3], masked=True)

        if np.ma.is_masked(data):
            data = data.filled(fill_value=0)

        # The visual asset is already display-stretched 8-bit, but a single chip is a
        # tiny crop of a whole scene and often occupies a narrow slice of that range.
        # Stretch it the same way the clipped path does, so the two look alike.
        data = _normalize_for_display(data)

        rgb_array = np.transpose(data, (1, 2, 0)).astype(np.uint8)
        Image.fromarray(rgb_array, mode="RGB").save(
            output_path, "JPEG", quality=quality, optimize=True
        )
    except RasterioIOError as err:
        raise ThumbnailError(f"Failed to read {visual_href}: {err}") from err
    except OSError as err:
        if output_path.exists():
            output_path.unlink()
        raise ThumbnailError(f"Failed to write thumbnail: {err}") from err

    return output_path


def generate_overlay_thumbnail(
    base_image_path: str | Path,
    mask_path: str | Path,
    output_path: str | Path,
    mask_colors: dict[int, tuple[int, int, int]] | None = None,
    opacity: float = 0.6,
    quality: int = 85,
) -> Path:
    """Generate thumbnail with semantic mask overlay.

    Composites a colorized mask onto a base JPEG image at the specified opacity.
    Class 0 (background) is rendered as transparent; other classes get colors
    from mask_colors.

    Args:
        base_image_path: Path to base JPEG thumbnail
        mask_path: Path to semantic mask GeoTIFF
        output_path: Output path for overlay JPEG
        mask_colors: Class value to RGB color mapping. Defaults to green for
            class 1 (field interiors) and orange for class 2 (boundaries).
        opacity: Opacity of mask overlay (0.0-1.0)
        quality: JPEG quality (1-100)

    Returns:
        Path to generated overlay thumbnail

    Raises:
        ThumbnailError: If overlay generation fails
    """
    base_image_path = Path(base_image_path)
    mask_path = Path(mask_path)
    output_path = Path(output_path)

    if not base_image_path.exists():
        raise ThumbnailError(f"Base image does not exist: {base_image_path}")
    if not mask_path.exists():
        raise ThumbnailError(f"Mask file does not exist: {mask_path}")

    if mask_colors is None:
        mask_colors = DEFAULT_MASK_COLORS

    try:
        # Load base JPEG and convert to RGBA
        base_img = Image.open(base_image_path).convert("RGBA")
        thumb_width, thumb_height = base_img.size

        # Read mask and resample to thumbnail size
        with rasterio.open(mask_path) as src:
            mask_data = src.read(
                1,
                out_shape=(thumb_height, thumb_width),
                resampling=Resampling.nearest,  # Preserve class values
            )

        # Create RGBA overlay from mask
        overlay_rgba = np.zeros((thumb_height, thumb_width, 4), dtype=np.uint8)
        alpha_value = int(opacity * 255)

        for class_val, color in mask_colors.items():
            mask_pixels = mask_data == class_val
            overlay_rgba[mask_pixels, 0] = color[0]  # R
            overlay_rgba[mask_pixels, 1] = color[1]  # G
            overlay_rgba[mask_pixels, 2] = color[2]  # B
            overlay_rgba[mask_pixels, 3] = alpha_value  # A

        # Alpha composite overlay onto base
        overlay_img = Image.fromarray(overlay_rgba, mode="RGBA")
        composite = Image.alpha_composite(base_img, overlay_img)

        # Convert to RGB and save as JPEG
        composite_rgb = composite.convert("RGB")
        composite_rgb.save(output_path, "JPEG", quality=quality, optimize=True)

    except RasterioIOError as e:
        raise ThumbnailError(f"Failed to read mask {mask_path}: {e}") from e
    except OSError as e:
        if output_path.exists():
            output_path.unlink()
        raise ThumbnailError(f"Failed to write overlay thumbnail: {e}") from e

    return output_path
