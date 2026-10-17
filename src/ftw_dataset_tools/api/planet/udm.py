"""UDM2/2.1 patch statistics; zeros are valid class values, not nodata."""

from pathlib import Path

import numpy as np
import rasterio

from ftw_dataset_tools.api.planet.raster import aoi_mask

CLASS_NAMES = ("clear", "snow", "shadow", "light_haze", "heavy_haze", "cloud")


def quality(path: Path, geometry: dict, min_clear: float = 0.95) -> dict:
    """Report fractions over the AOI, including blackfill in the denominator.

    UDM2.1 retains the eight-band layout; band 5 is always zero. Band 8 is
    bit-encoded: bit 0 is blackfill, while any nonzero bit is unusable.
    Confidence is averaged over observed pixels only.
    """
    if not 0 <= min_clear <= 1:
        raise ValueError("min_clear must be in [0, 1]")
    with rasterio.open(path) as source:
        if source.count != 8 or source.dtypes[0] != "uint8":
            raise ValueError("Expected an eight-band uint8 UDM2 raster")
        data = source.read()
        inside = aoi_mask(geometry, source.crs, source.transform, source.height, source.width)

    total = int(inside.sum())
    if not total:
        raise ValueError("AOI contains no pixel centers")
    observed = inside & ((data[7] & 1) == 0)
    result = {
        name: float(((data[index] > 0) & observed).sum() / total)
        for index, name in enumerate(CLASS_NAMES)
    }
    result.update(
        pixels=total,
        observed_fraction=float(observed.sum() / total),
        blackfill=float((inside & ((data[7] & 1) != 0)).sum() / total),
        unusable=float((inside & (data[7] != 0)).sum() / total),
        confidence=float(np.mean(data[6][observed])) if observed.any() else None,
    )
    result["usable"] = result["clear"] >= min_clear and result["unusable"] <= 1 - min_clear
    return result
