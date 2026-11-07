"""Native-grid raster crops and atomic, provenance-tagged GeoTIFF output."""

import math
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.warp import transform_geom
from rasterio.windows import Window, from_bounds
from shapely.geometry import shape

COG_OPTIONS = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_USE_HEAD": "NO",
    "GDAL_HTTP_MULTIPLEX": "YES",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "GDAL_HTTP_CONNECTTIMEOUT": 20,
    "GDAL_HTTP_TIMEOUT": 120,
    "GDAL_HTTP_MAX_RETRY": 3,
    "GDAL_HTTP_RETRY_DELAY": 2,
    "GDAL_CACHEMAX": 64 * 1024 * 1024,
}


def validate_raster(path: Path) -> None:
    """Read every block to reject empty, truncated, or ungeoreferenced downloads."""
    with rasterio.open(path) as source:
        if source.crs is None or source.width < 1 or source.height < 1:
            raise ValueError("Raster has no valid grid")
        for _, window in source.block_windows(1):
            source.read(window=window)


def output_matches(path: Path, signature: str) -> bool:
    """Resume only complete readable outputs from the requested inputs."""
    try:
        with rasterio.open(path) as source:
            if source.tags().get("ftwd_signature") != signature:
                return False
        validate_raster(path)
    except (OSError, ValueError, rasterio.errors.RasterioError):
        return False
    return True


def crop_window(source: rasterio.io.DatasetReader, geometry: dict) -> Window:
    """Cover the complete AOI bbox on the source grid, including scene-edge padding."""
    projected = transform_geom("EPSG:4326", source.crs, geometry)
    window = from_bounds(*shape(projected).bounds, transform=source.transform)
    left, top = math.floor(window.col_off + 1e-7), math.floor(window.row_off + 1e-7)
    right, bottom = (
        math.ceil(window.col_off + window.width - 1e-7),
        math.ceil(window.row_off + window.height - 1e-7),
    )
    if (right - left) * (bottom - top) > 16_777_216:
        raise ValueError("AOI exceeds 16 million pixels; split it before downloading")
    return Window(left, top, right - left, bottom - top)


def aoi_mask(geometry: dict, crs: object, transform: object, height: int, width: int) -> np.ndarray:
    """Pixel-center polygon mask; holes and nonrectangular AOIs are respected."""
    projected = transform_geom("EPSG:4326", crs, geometry)
    return geometry_mask([projected], (height, width), transform, invert=True)


def write_raster(
    path: Path,
    data: np.ndarray,
    profile: dict,
    signature: str,
    *,
    mask: np.ndarray | None = None,
    tags: dict | None = None,
    descriptions: tuple[str, ...] | None = None,
) -> None:
    """Publish a complete raster and internal validity mask with an atomic rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix=".tif", delete=False) as stream:
        temporary = Path(stream.name)
    settings = dict(
        profile,
        driver="GTiff",
        count=data.shape[0],
        height=data.shape[1],
        width=data.shape[2],
        dtype=data.dtype,
        compress="deflate",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )
    try:
        with (
            rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True),
            rasterio.open(temporary, "w", **settings) as target,
        ):
            target.write(data)
            target.update_tags(ftwd_signature=signature, **(tags or {}))
            if mask is not None:
                target.write_mask(mask.astype("uint8") * 255)
            if descriptions is not None:
                target.descriptions = descriptions
        validate_raster(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
