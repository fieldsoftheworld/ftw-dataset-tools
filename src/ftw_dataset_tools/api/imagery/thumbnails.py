"""JPEG thumbnail generation for satellite imagery."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.errors import RasterioIOError


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
    max_size: int = 512,
    quality: int = 85,
) -> Path:
    """Generate JPEG thumbnail from a multi-band GeoTIFF.

    Uses rasterio's out_shape to leverage COG overviews automatically.

    Args:
        tif_path: Path to GeoTIFF (must have at least 3 bands for RGB)
        output_path: Output path for JPEG
        max_size: Maximum dimension in pixels
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

            # Calculate output dimensions maintaining aspect ratio
            scale = min(max_size / src.width, max_size / src.height)
            out_width = max(1, int(src.width * scale))
            out_height = max(1, int(src.height * scale))

            # Read RGB bands at thumbnail size (uses overviews automatically)
            data = src.read(
                indexes=[1, 2, 3],
                out_shape=(3, out_height, out_width),
                resampling=Resampling.bilinear,
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
