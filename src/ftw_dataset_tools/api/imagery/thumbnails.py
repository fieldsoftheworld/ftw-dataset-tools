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
