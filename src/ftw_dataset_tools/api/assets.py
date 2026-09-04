"""Decorate STAC assets with file, raster and classification extension fields.

Call these after the asset has been added to its Item or Collection so pystac
registers the extension schema URIs on the owner.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import TYPE_CHECKING

import rasterio
from pystac.extensions.classification import (
    Classification,
    ClassificationExtension,
)
from pystac.extensions.file import FileExtension
from pystac.extensions.raster import DataType, RasterBand, RasterExtension, Statistics

from ftw_dataset_tools.api.raster_stats import band_stats_from_tags

if TYPE_CHECKING:
    import pystac

# Multihash prefix for sha2-256: code 0x12, length 0x20 (32 bytes).
_MULTIHASH_SHA256_PREFIX = "1220"

# Matches the metres-per-degree factor used in masks._grid_raster_geometry to
# convert a metre resolution into degrees for geographic-CRS grids.
METRES_PER_DEGREE = 111000.0

# (value, name, description) per classified mask kind. The background value is
# substituted at call time ONLY for the semantic kinds: presence-only masks use
# background=3 there, while the DECODE layers fold presence-only into 0.
MASK_CLASSES: dict[str, list[tuple[int, str, str]]] = {
    "semantic_2class": [
        (0, "background", "Not a field"),
        (1, "field", "Field polygon interior"),
    ],
    "semantic_3class": [
        (0, "background", "Not a field"),
        (1, "field", "Field polygon interior"),
        (2, "boundary", "Field boundary line"),
    ],
    "decode_boundary": [
        (0, "background", "Not a field boundary"),
        (1, "boundary", "DECODE field boundary ring (inner one-pixel ring of each field)"),
    ],
}

# Mask kinds that are continuous or id-valued: described, never classified.
MASK_DESCRIPTIONS: dict[str, str] = {
    "instance": (
        "Instance mask: the background value (0, or 3 for presence-only labels) marks "
        "non-field pixels; other values are per-field instance ids"
    ),
    "decode_distance": (
        "DECODE normalized Euclidean distance to the nearest field boundary in [0, 1]; "
        "multiply by the decode_distance_max_px dataset tag to recover pixels"
    ),
}

_BACKGROUND_SUBSTITUTED_KINDS = frozenset({"semantic_2class", "semantic_3class"})


def multihash_sha256(path: Path) -> str:
    """Return the multihash-encoded sha2-256 of a file as a hex string."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return _MULTIHASH_SHA256_PREFIX + digest.hexdigest()


def add_file_info(asset: pystac.Asset, path: Path, *, checksum: bool = False) -> None:
    """Set ``file:size`` and, when requested, ``file:checksum`` on an asset."""
    path = Path(path)
    ext = FileExtension.ext(asset, add_if_missing=True)
    ext.size = path.stat().st_size
    if checksum:
        ext.checksum = multihash_sha256(path)


def add_table_columns(asset: pystac.Asset, path: Path, geometry_column: str | None = None) -> None:
    """Set ``table:columns`` (name, type) and ``table:row_count`` from a parquet file.

    Types are DuckDB's, lowercased. The geometry column is typed ``geometry``
    (DuckDB reports the WKB blob). Descriptions are added by the docs stage.
    """
    # Imported here: api.geo pulls geopandas and geoparquet-io into every assets import otherwise.
    import duckdb
    from pystac.extensions.table import Column, TableExtension

    from ftw_dataset_tools.api.geo import detect_geometry_column

    path = Path(path)
    geometry_column = geometry_column or detect_geometry_column(path)
    con = duckdb.connect(":memory:")
    try:
        rows = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{path}')").fetchall()
        row_count = con.execute(f"SELECT count(*) FROM read_parquet('{path}')").fetchone()[0]
    finally:
        con.close()

    columns = []
    for name, dtype, *_ in rows:
        column_type = "geometry" if name == geometry_column else str(dtype).lower()
        columns.append(Column({"name": name, "type": column_type}))

    table = TableExtension.ext(asset, add_if_missing=True)
    table.columns = columns
    table.row_count = int(row_count)


def _spatial_resolution(transform: rasterio.Affine, crs: rasterio.crs.CRS | None) -> float:
    """Return the pixel size in metres, per the raster extension's ``spatial_resolution``.

    The transform's pixel size is in CRS units, which is degrees for a geographic
    CRS. Convert those to metres using the same factor as
    ``masks._grid_raster_geometry`` uses to go the other way.
    """
    pixel_size = abs(transform.a)
    if crs is not None and crs.is_geographic:
        pixel_size *= METRES_PER_DEGREE
    return round(pixel_size, 6)


def _band_nodata(value: float | int | None) -> float | int | str | None:
    """Return a JSON-safe nodata value, using the raster extension's ``"nan"`` string form."""
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    return value


def add_raster_bands(asset: pystac.Asset, path: Path) -> None:
    """Set ``raster:bands`` from the file's data types, nodata and embedded stats."""
    path = Path(path)
    with rasterio.open(path) as src:
        dtypes = list(src.dtypes)
        nodatas = list(src.nodatavals)
        resolution = _spatial_resolution(src.transform, src.crs)
        descriptions = list(src.descriptions)
        tags = [src.tags(i) for i in range(1, src.count + 1)]

    bands: list[RasterBand] = []
    for index, dtype in enumerate(dtypes, start=1):
        stats = band_stats_from_tags(tags[index - 1])
        statistics = None
        if stats is not None:
            statistics = Statistics.create(
                minimum=stats.minimum,
                maximum=stats.maximum,
                mean=stats.mean,
                stddev=stats.stddev,
                valid_percent=stats.valid_percent,
            )
        band = RasterBand.create(
            nodata=_band_nodata(nodatas[index - 1]),
            data_type=DataType(dtype),
            spatial_resolution=resolution,
            statistics=statistics,
        )
        description = descriptions[index - 1]
        if description:
            band.properties["description"] = description
        bands.append(band)

    RasterExtension.ext(asset, add_if_missing=True).bands = bands


def add_mask_classification(asset: pystac.Asset, mask_kind: str, background_value: int = 0) -> None:
    """Describe the classes of a mask asset on its first raster band.

    ``mask_kind`` is a key of ``MASK_CLASSES`` or ``MASK_DESCRIPTIONS``.
    Requires ``add_raster_bands`` to have run first.
    """
    raster = RasterExtension.ext(asset)
    bands = raster.bands or []
    if not bands:
        raise ValueError("add_raster_bands must be called before add_mask_classification")
    band = bands[0]

    if mask_kind in MASK_DESCRIPTIONS:
        band.properties["description"] = MASK_DESCRIPTIONS[mask_kind]
        raster.bands = bands
        return

    try:
        spec = MASK_CLASSES[mask_kind]
    except KeyError as err:
        raise ValueError(f"Unknown mask kind: {mask_kind}") from err

    substitute = mask_kind in _BACKGROUND_SUBSTITUTED_KINDS
    classes = [
        Classification.create(
            value=background_value if (substitute and name == "background") else value,
            name=name,
            description=description,
        )
        for value, name, description in spec
    ]
    ClassificationExtension.ext(band, add_if_missing=True).classes = classes
    owner = asset.owner
    if owner is not None:
        ClassificationExtension.add_to(owner)
    raster.bands = bands
