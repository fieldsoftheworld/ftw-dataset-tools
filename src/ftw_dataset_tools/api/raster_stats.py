"""Per-band raster statistics: compute, embed as GDAL tags, read back.

Portolan requires every COG band to carry embedded minimum, maximum, mean and
standard deviation, plus valid percent when the band has a nodata value. GDAL
stores these as ``STATISTICS_*`` band metadata; writing them on the open dataset
keeps them inside the COG instead of an ``.aux.xml`` sidecar.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import rasterio

if TYPE_CHECKING:
    from pathlib import Path

    from rasterio.io import DatasetWriter

TAG_MIN = "STATISTICS_MINIMUM"
TAG_MAX = "STATISTICS_MAXIMUM"
TAG_MEAN = "STATISTICS_MEAN"
TAG_STDDEV = "STATISTICS_STDDEV"
TAG_VALID_PERCENT = "STATISTICS_VALID_PERCENT"


@dataclass(frozen=True)
class BandStats:
    """Exact statistics for one raster band."""

    minimum: float
    maximum: float
    mean: float
    stddev: float
    valid_percent: float | None


def compute_band_stats(data: np.ndarray, nodata: float | int | None = None) -> BandStats:
    """Compute exact statistics over a 2-D band array, excluding nodata pixels.

    ``valid_percent`` is None when no nodata value is defined.
    """
    values = np.asarray(data)
    if nodata is None:
        valid = values.reshape(-1)
        valid_percent: float | None = None
    else:
        # Handle NaN nodata: NaN != NaN is True, so use isnan for NaN values
        if isinstance(nodata, float) and np.isnan(nodata):
            mask = ~np.isnan(values)
        else:
            mask = values != nodata
        valid = values[mask]
        valid_percent = float(100.0 * valid.size / values.size) if values.size else 0.0

    if valid.size == 0:
        return BandStats(0.0, 0.0, 0.0, 0.0, valid_percent)

    as_float = valid.astype(np.float64)
    return BandStats(
        minimum=float(as_float.min()),
        maximum=float(as_float.max()),
        mean=float(as_float.mean()),
        stddev=float(as_float.std()),
        valid_percent=valid_percent,
    )


def embed_band_stats(dataset: DatasetWriter, band_index: int, stats: BandStats) -> None:
    """Write statistics as band tags on an open writable dataset."""
    tags = {
        TAG_MIN: repr(stats.minimum),
        TAG_MAX: repr(stats.maximum),
        TAG_MEAN: repr(stats.mean),
        TAG_STDDEV: repr(stats.stddev),
    }
    if stats.valid_percent is not None:
        tags[TAG_VALID_PERCENT] = repr(stats.valid_percent)
    dataset.update_tags(band_index, **tags)


def band_stats_from_tags(tags: dict[str, str]) -> BandStats | None:
    """Parse embedded ``STATISTICS_*`` tags, or None if a required tag is missing."""
    try:
        return BandStats(
            minimum=float(tags[TAG_MIN]),
            maximum=float(tags[TAG_MAX]),
            mean=float(tags[TAG_MEAN]),
            stddev=float(tags[TAG_STDDEV]),
            valid_percent=float(tags[TAG_VALID_PERCENT]) if TAG_VALID_PERCENT in tags else None,
        )
    except KeyError:
        return None


def read_band_stats(path: Path | str, band_index: int) -> BandStats | None:
    """Read embedded statistics for a band, or None if any required tag is missing."""
    with rasterio.open(path) as src:
        return band_stats_from_tags(src.tags(band_index))
