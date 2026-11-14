"""Per-field embedding statistics from AlphaEarth Foundations and OlmoEarth v1.3.

For every field polygon, the annual 10 m embeddings of the dataset's year are read
from both products and summarised per dimension: the mean and the standard deviation
over the field's pixels. The result is a plain Parquet table with one row per field,
in the fields file's order and keyed by its id column, so it joins back onto the
fields without carrying their geometry again.

Products, both 64 dimensions, int8-quantised on a 10 m UTM grid:

- **AlphaEarth Foundations** (Google and Google DeepMind), the Satellite Embedding
  dataset's annual COGs, read from the copy on Source Cooperative (tge-labs/aef).
  CC BY 4.0. Its vectors are unit length.
- **OlmoEarth v1.3** (Ai2), the Zarr store on Source Cooperative
  (olmoearth/olmoearth-v1-3-embeddings). Only the first 64 of its 128 dimensions are
  read: the model was trained so they form a complete embedding on their own. Its
  vectors are not normalised. OlmoEarth Artifact License: anything derived from it
  carries the license's use restrictions (Section 2) downstream.

Both are dequantised before any arithmetic, as their providers specify:
``value = sign(q) * (|q| / 127.5) ** 2``, with ``q = -128`` meaning no data.

Pixels: a field takes the 10 m pixels whose centre falls inside it. A field too small
or thin to hold a pixel centre takes every pixel it touches instead, and says so in
``pixel_rule``. Pixels a product has no data for are left out of that product's
statistics only, and ``<product>_pixels`` counts the ones used. Overlapping fields
each count the pixels they share.

Each field is read in the UTM zone of its centre, in the northern-hemisphere CRS of
that zone (``EPSG:326NN``, negative northings south of the equator), which is how
OlmoEarth stores every zone. Both products sit on the same 10 m grid in that CRS, so
nothing is resampled.
"""

from __future__ import annotations

import json
import math
import tempfile
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import shapely
from pyproj import CRS, Transformer
from pyproj.exceptions import CRSError
from rasterio.enums import MergeAlg
from rasterio.features import geometry_mask, rasterize
from rasterio.transform import from_origin

from ftw_dataset_tools.api.chips_io import parquet_columns
from ftw_dataset_tools.api.geo import (
    detect_geometry_column,
    ensure_spatial_loaded,
    sql_path,
)
from ftw_dataset_tools.api.logging_config import get_logger
from ftw_dataset_tools.api.source import SourceFetchError

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping

    import zarr

logger = get_logger(__name__)

BANDS = 64
NODATA = -128
PIXEL_M = 10.0
#: Fields are grouped into work units by the tile their centre falls in. 20,480 m is
#: one OlmoEarth shard, so a unit's reads stay within a few shards.
BLOCK_M = 20_480.0
#: Pixels per side of the windows a work unit is read and rasterised in; bounds memory.
TILE_PX = 1024
DEFAULT_WORKERS = 8
#: Fields wider or taller than this (in their UTM zone) are not on a usable grid:
#: polygons crossing the antimeridian, or projected far outside their zone.
MAX_FIELD_M = 200_000.0
#: Attempts per remote read before the run stops, and the first backoff in seconds.
READ_ATTEMPTS = 4
READ_BACKOFF_S = 2.0
#: Finished fields are saved in parts of about this many rows, so a stopped run resumes.
PART_ROWS = 20_000
#: Key of the JSON description in the output Parquet's key-value metadata.
METADATA_KEY = "ftw:field_embeddings"

SOURCE_COOP_BUCKET = "us-west-2.opendata.source.coop"
SOURCE_COOP_REGION = "us-west-2"
AEF_INDEX = f"s3://{SOURCE_COOP_BUCKET}/tge-labs/aef/v1/annual/aef_index.parquet"
OLMOEARTH_PREFIX = "olmoearth/olmoearth-v1-3-embeddings/embeddings.zarr"

#: Bumped whenever the statistics change, so older outputs and saved parts are not reused.
ALGORITHM_VERSION = 1
#: Memory DuckDB may use before spilling to the run's temp folder.
DUCKDB_MEMORY_LIMIT = "4GB"

#: Statistics per product and dimension.
STATISTICS = ("mean", "std")
PIXEL_RULE_CENTRE = "centre"
PIXEL_RULE_TOUCHED = "touched"


@dataclass(frozen=True)
class Product:
    """An embedding product: where it comes from and how to credit it."""

    key: str
    title: str
    url: str
    license: str
    attribution: str
    notes: str


AEF = Product(
    key="aef",
    title="AlphaEarth Foundations Satellite Embedding, annual v1",
    url="https://source.coop/tge-labs/aef",
    license="CC-BY-4.0",
    attribution=(
        "AlphaEarth Foundations Satellite Embedding dataset by Google and Google DeepMind, "
        "CC-BY-4.0 (source of the aef_* columns)"
    ),
    notes="64 dimensions; each pixel's vector has unit length.",
)
OLMOEARTH = Product(
    key="olmoearth",
    title="OlmoEarth v1.3 embeddings, first 64 of 128 dimensions",
    url="https://source.coop/olmoearth/olmoearth-v1-3-embeddings",
    license="https://data.source.coop/olmoearth/olmoearth-v1-3-embeddings/LICENSE.txt",
    attribution=(
        "OlmoEarth v1.3 embeddings by Ai2 (Allen Institute for AI), OlmoEarth Artifact "
        "License, including its Section 2 use restrictions (source of the olmoearth_* columns)"
    ),
    notes=(
        "First 64 of 128 dimensions, which form a complete embedding on their own. "
        "Vectors are not normalised. Ai2 asks that 64- and 128-dimension vectors are "
        "not compared with each other."
    ),
)
PRODUCTS: dict[str, Product] = {p.key: p for p in (AEF, OLMOEARTH)}

DEQUANTIZATION = "value = sign(q) * (|q| / 127.5) ** 2; q = -128 is no data"


class FieldEmbeddingsError(ValueError):
    """The fields cannot be given embedding statistics (no id column, no year, bad CRS)."""

    def __init__(self, path: Path, message: str) -> None:
        super().__init__(message)
        self.path = path


def _dequantize_table() -> np.ndarray:
    q = np.arange(-128, 128, dtype=np.float32)
    values = np.sign(q) * (np.abs(q) / 127.5) ** 2
    values[0] = np.nan
    return values.astype(np.float32)


#: Dequantised value for each int8 code, indexed by ``code + 128``; NaN for no data.
DEQUANT = _dequantize_table()


def dequantize(codes: np.ndarray) -> np.ndarray:
    """Float32 embeddings from int8 codes, NaN where there is no data."""
    return DEQUANT[codes.astype(np.int16) + 128]


# ---- sources -------------------------------------------------------------------


@dataclass(frozen=True)
class Window:
    """A window on the 10 m grid of ``EPSG:326<zone>``: top-left corner and size."""

    zone: int
    x0: float
    y1: float
    width: int
    height: int

    @property
    def x1(self) -> float:
        return self.x0 + self.width * PIXEL_M

    @property
    def y0(self) -> float:
        return self.y1 - self.height * PIXEL_M


class EmbeddingSource(Protocol):
    """One product for one year: int8 codes for a window, NODATA outside its coverage."""

    def read(self, window: Window) -> np.ndarray:
        """Codes shaped (BANDS, height, width), north up."""
        ...


def _empty(window: Window) -> np.ndarray:
    return np.full((BANDS, window.height, window.width), NODATA, dtype=np.int8)


class GridError(ValueError):
    """A product's grid does not line up with the 10 m grid the fields are read on."""


def _grid_index(value: float, what: str) -> int:
    """A grid offset in pixels, refusing a source whose grid is not on whole pixels."""
    index = value / PIXEL_M
    if abs(index - round(index)) > 1e-6:
        raise GridError(f"{what} is not on the 10 m grid ({value})")
    return round(index)


class OlmoEarthSource:
    """OlmoEarth v1.3 from its Zarr store: one group per UTM zone, ``utm01``..``utm60``."""

    def __init__(self, year: int, root: zarr.Group) -> None:
        self.year = year
        self._root = root
        self._zones: dict[int, tuple[zarr.Array, int, tuple[float, ...]] | None] = {}

    @classmethod
    def open(cls, year: int) -> OlmoEarthSource:
        import zarr
        from obstore.store import S3Store
        from zarr.storage import ObjectStore

        store = S3Store(
            SOURCE_COOP_BUCKET,
            prefix=OLMOEARTH_PREFIX,
            region=SOURCE_COOP_REGION,
            skip_signature=True,
        )
        try:
            root = zarr.open_group(ObjectStore(store, read_only=True), mode="r")
        except Exception as err:
            raise SourceFetchError(f"Could not open the OlmoEarth store: {err}") from err
        return cls(year, root)

    def years(self) -> list[int]:
        """The years the store holds, read from one zone (all zones share them)."""
        group = self._root[next(iter(sorted(self._root.group_keys())))]
        return [int(t) for t in group["time"][:]]

    def _zone(self, zone: int) -> tuple[zarr.Array, int, tuple[float, ...]] | None:
        if zone not in self._zones:
            name = f"utm{zone:02d}"
            if name not in self._root:
                self._zones[zone] = None
            else:
                group = self._root[name]
                times = [int(t) for t in group["time"][:]]
                transform = tuple(group.attrs["spatial:transform"])
                self._zones[zone] = (
                    (group["embeddings"], times.index(self.year), transform)
                    if self.year in times
                    else None
                )
        return self._zones[zone]

    def read(self, window: Window) -> np.ndarray:
        out = _empty(window)
        entry = self._zone(window.zone)
        if entry is None:
            return out
        array, t, (a, _, c, _, e, f) = entry
        if a != PIXEL_M or e != -PIXEL_M:
            raise GridError(f"OlmoEarth utm{window.zone:02d} is not a 10 m north-up grid")
        col0 = _grid_index(window.x0 - c, "OlmoEarth window x")
        row0 = _grid_index(f - window.y1, "OlmoEarth window y")
        _, _, n_rows, n_cols = array.shape
        _paste(out, col0, row0, n_cols, n_rows, lambda r, c_: array[t, :BANDS, r, c_])
        return out


def _paste(
    out: np.ndarray,
    col0: int,
    row0: int,
    n_cols: int,
    n_rows: int,
    fetch: Callable[[slice, slice], np.ndarray],
) -> None:
    """Fill ``out`` (a window at col0/row0 of a top-down array) from the part inside it."""
    _, height, width = out.shape
    r0, r1 = max(row0, 0), min(row0 + height, n_rows)
    c0, c1 = max(col0, 0), min(col0 + width, n_cols)
    if r0 >= r1 or c0 >= c1:
        return
    out[:, r0 - row0 : r1 - row0, c0 - col0 : c1 - col0] = fetch(slice(r0, r1), slice(c0, c1))


@dataclass(frozen=True)
class _AefTile:
    path: str
    zone: int
    south: bool
    west: float
    south_edge: float
    east: float
    north: float


class AefSource:
    """AlphaEarth Foundations annual COGs, located through the dataset's index.

    The COGs are stored south-up (positive y pixel size, origin at the south edge), and
    southern zones use ``EPSG:327NN``, 10,000 km north of ``EPSG:326NN``.
    """

    def __init__(self, year: int, tiles: list[_AefTile], opener: Callable | None = None) -> None:
        self.year = year
        self._tiles: dict[int, list[_AefTile]] = {}
        for tile in tiles:
            self._tiles.setdefault(tile.zone, []).append(tile)
        self._opener = opener

    @classmethod
    def open(cls, year: int, index: str = AEF_INDEX) -> AefSource:
        con = duckdb.connect(":memory:")
        try:
            if index.startswith("s3://"):
                from ftw_dataset_tools.api.geo import configure_source_coop_s3

                configure_source_coop_s3(con)
            rows = con.execute(
                f"""SELECT path, crs, utm_west, utm_south, utm_east, utm_north
                    FROM read_parquet('{sql_path(index)}') WHERE year = ?""",
                [year],
            ).fetchall()
        except (duckdb.IOException, duckdb.HTTPException) as err:
            raise SourceFetchError(f"Could not read the AlphaEarth index ({index}): {err}") from err
        finally:
            con.close()
        return cls(year, [_aef_tile(*row) for row in rows])

    def _open(self, path: str) -> Any:
        """A rasterio dataset for one COG (the Source Cooperative bucket is public)."""
        if self._opener is not None:
            return self._opener(path)
        import rasterio

        return rasterio.open(path)

    def read(self, window: Window) -> np.ndarray:
        out = _empty(window)
        offset = 10_000_000.0
        for tile in self._tiles.get(window.zone, []):
            shift = offset if tile.south else 0.0
            # Tile bounds in EPSG:326NN.
            if (
                tile.east <= window.x0
                or tile.west >= window.x1
                or tile.north - shift <= window.y0
                or tile.south_edge - shift >= window.y1
            ):
                continue
            self._paste_tile(out, window, tile.path, shift)
        return out

    def _paste_tile(self, out: np.ndarray, window: Window, path: str, shift: float) -> None:
        from rasterio.windows import Window as RioWindow

        with _public_s3_env(), self._open(path) as src:
            t = src.transform
            if t.a != PIXEL_M or abs(t.e) != PIXEL_M:
                raise GridError(f"{path} is not a 10 m grid")
            col0 = _grid_index(window.x0 - t.c, "AlphaEarth window x")
            south_up = t.e > 0
            if south_up:
                # Row 0 is the southern edge; turn it into a top-down row offset.
                bottom = _grid_index(window.y0 + shift - t.f, "AlphaEarth window y")
                row0 = src.height - (bottom + window.height)
            else:
                row0 = _grid_index(t.f - (window.y1 + shift), "AlphaEarth window y")

            def fetch(rows: slice, cols: slice) -> np.ndarray:
                r0, r1 = rows.start, rows.stop
                if south_up:
                    r0, r1 = src.height - r1, src.height - r0
                data = src.read(window=RioWindow(cols.start, r0, cols.stop - cols.start, r1 - r0))
                return data[:, ::-1, :] if south_up else data

            region = np.full_like(out, NODATA)
            _paste(region, col0, row0, src.width, src.height, fetch)
        # Tiles do not overlap, but never let a tile's no-data edge erase another's data.
        fill = (out[0] == NODATA) & (region[0] != NODATA)
        out[:, fill] = region[:, fill]


def _public_s3_env() -> Any:
    """GDAL settings for anonymous reads of the Source Cooperative bucket.

    The bucket name has dots, so virtual-hosted URLs fail the TLS check; path-style
    addressing avoids that. rasterio keeps these per thread.
    """
    import rasterio

    return rasterio.Env(
        AWS_NO_SIGN_REQUEST="YES",
        AWS_REGION=SOURCE_COOP_REGION,
        AWS_VIRTUAL_HOSTING="FALSE",
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    )


def _aef_tile(
    path: str, crs: str, west: float, south: float, east: float, north: float
) -> _AefTile:
    code = int(crs.split(":")[1])
    return _AefTile(path, code % 100, code // 100 == 327, west, south, east, north)


def _with_retries(read: Callable[[], Any], what: str) -> Any:
    """Run a remote read, retrying with backoff; SourceFetchError once it keeps failing.

    Any exception counts (rasterio, zarr and obstore each raise their own network errors,
    and a truncated response can surface as a ValueError) except GridError, a mismatch
    no retry can fix.
    """
    import time

    for attempt in range(READ_ATTEMPTS):
        try:
            return read()
        except (GridError, TypeError, AttributeError, KeyError, IndexError, NameError):
            raise  # a mismatch or a bug: retrying cannot help
        except Exception as err:
            if attempt == READ_ATTEMPTS - 1:
                raise SourceFetchError(
                    f"Could not read {what} after {READ_ATTEMPTS} attempts: {err}"
                ) from err
            logger.debug(f"Retrying {what} after: {err}")
            time.sleep(READ_BACKOFF_S * 2**attempt)
    raise AssertionError("unreachable")


def open_sources(year: int, products: tuple[str, ...] = tuple(PRODUCTS)) -> dict[str, object]:
    """The default remote source for each product that has ``year``; warns for the rest."""
    openers: dict[str, Callable[[int], object]] = {
        AEF.key: AefSource.open,
        OLMOEARTH.key: OlmoEarthSource.open,
    }
    sources: dict[str, object] = {}
    for key in products:
        source = _with_retries(lambda o=openers[key]: o(year), PRODUCTS[key].title)
        if _with_retries(lambda s=source: _has_year(s, year), f"{PRODUCTS[key].title} years"):
            sources[key] = source
        else:
            logger.warning(f"{PRODUCTS[key].title} has no embeddings for {year}; left empty")
    return sources


def _has_year(source: object, year: int) -> bool:
    if isinstance(source, AefSource):
        return bool(source._tiles)
    if isinstance(source, OlmoEarthSource):
        return year in source.years()
    return True


# ---- statistics ------------------------------------------------------------------


@dataclass
class _Sums:
    """Running per-field pixel counts and sums for one product."""

    n: np.ndarray
    s: np.ndarray
    ss: np.ndarray

    @classmethod
    def zeros(cls, k: int) -> _Sums:
        return cls(np.zeros(k, np.int64), np.zeros((k, BANDS)), np.zeros((k, BANDS)))

    def add(self, fields: np.ndarray, codes: np.ndarray) -> None:
        """Add pixels: ``fields`` (P,) local field index per pixel, ``codes`` (BANDS, P) int8.

        One band at a time, summed in float64 by bincount: memory stays at a few
        vectors of P, and float64 squares keep a constant field's std at 0.
        """
        if fields.size == 0:
            return
        k = self.n.size
        self.n += np.bincount(fields, minlength=k)
        for band in range(BANDS):
            v = dequantize(codes[band]).astype(np.float64)
            self.s[:, band] += np.bincount(fields, weights=v, minlength=k)
            self.ss[:, band] += np.bincount(fields, weights=v * v, minlength=k)


@dataclass
class _Accumulator:
    """Per-field sums for the centre rule and, for fields it misses, every touched pixel."""

    k: int
    products: tuple[str, ...]
    centre_px: np.ndarray = field(init=False)
    touched_px: np.ndarray = field(init=False)
    centre: dict[str, _Sums] = field(init=False)
    touched: dict[str, _Sums] = field(init=False)

    def __post_init__(self) -> None:
        self.centre_px = np.zeros(self.k, np.int64)
        self.touched_px = np.zeros(self.k, np.int64)
        self.centre = {p: _Sums.zeros(self.k) for p in self.products}
        self.touched = {p: _Sums.zeros(self.k) for p in self.products}

    def add(
        self, fields: np.ndarray, pixels: np.ndarray, codes: dict[str, np.ndarray], touched: bool
    ) -> None:
        """Count pixels (flat indices into the window) for fields (one index per pixel)."""
        np.add.at(self.touched_px if touched else self.centre_px, fields, 1)
        sums = self.touched if touched else self.centre
        for product, data in codes.items():
            flat = data.reshape(BANDS, -1)[:, pixels]
            # Both products mark no data in every band at once; requiring all bands keeps
            # one stray code from turning a field's mean into NaN.
            valid = (flat != NODATA).all(axis=0)
            sums[product].add(fields[valid], flat[:, valid])


def _tiles(bounds: tuple[float, float, float, float], zone: int, tile_px: int) -> Iterator[Window]:
    """Windows of at most tile_px pixels covering bounds, aligned to multiples of the tile."""
    xmin, ymin, xmax, ymax = bounds
    size = tile_px * PIXEL_M
    x0 = math.floor(xmin / PIXEL_M) * PIXEL_M
    y1 = math.ceil(ymax / PIXEL_M) * PIXEL_M
    x_end = math.ceil(xmax / PIXEL_M) * PIXEL_M
    y_end = math.floor(ymin / PIXEL_M) * PIXEL_M
    if x_end == x0:
        x_end += PIXEL_M
    if y_end == y1:
        y_end -= PIXEL_M
    tx = x0
    while tx < x_end:
        tx_next = min((math.floor(tx / size) + 1) * size, x_end)
        ty = y1
        while ty > y_end:
            ty_next = max((math.ceil(ty / size) - 1) * size, y_end)
            yield Window(
                zone, tx, ty, round((tx_next - tx) / PIXEL_M), round((ty - ty_next) / PIXEL_M)
            )
            ty = ty_next
        tx = tx_next


def _window_fields(geoms: np.ndarray, boxes: np.ndarray, window: Window) -> np.ndarray:
    """Indices of the fields whose bounding box meets the window."""
    return np.flatnonzero(
        (boxes[:, 0] < window.x1)
        & (boxes[:, 2] > window.x0)
        & (boxes[:, 1] < window.y1)
        & (boxes[:, 3] > window.y0)
        & ~shapely.is_missing(geoms)
        & ~shapely.is_empty(geoms)
    )


def _field_pixels(geom: object, window: Window, *, all_touched: bool) -> np.ndarray:
    """Flat window indices of the pixels a field takes, rasterised over its own bbox only."""
    xmin, ymin, xmax, ymax = shapely.bounds(geom)
    c0 = max(math.floor((xmin - window.x0) / PIXEL_M), 0)
    c1 = min(math.ceil((xmax - window.x0) / PIXEL_M), window.width)
    r0 = max(math.floor((window.y1 - ymax) / PIXEL_M), 0)
    r1 = min(math.ceil((window.y1 - ymin) / PIXEL_M), window.height)
    if c0 >= c1 or r0 >= r1:
        return np.empty(0, np.int64)
    inside = ~geometry_mask(
        [geom],
        out_shape=(r1 - r0, c1 - c0),
        transform=from_origin(window.x0 + c0 * PIXEL_M, window.y1 - r0 * PIXEL_M, PIXEL_M, PIXEL_M),
        all_touched=all_touched,
    )
    rows, cols = np.nonzero(inside)
    return (rows + r0) * window.width + (cols + c0)


def _add_window(
    acc: _Accumulator,
    geoms: np.ndarray,
    idx: np.ndarray,
    window: Window,
    codes: dict[str, np.ndarray],
) -> None:
    """Add one window's pixels to the fields in idx (indices into geoms)."""
    shape = (window.height, window.width)
    transform = from_origin(window.x0, window.y1, PIXEL_M, PIXEL_M)
    labels = rasterize(
        [(g, i + 1) for i, g in enumerate(geoms[idx])],
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype="int32",
    ).ravel()
    cover = rasterize(
        [(g, 1) for g in geoms[idx]],
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype="uint16",
        merge_alg=MergeAlg.add,
    ).ravel()
    shared = cover > 1
    single = np.flatnonzero((labels > 0) & ~shared)
    owners = labels[single] - 1
    acc.add(idx[owners], single, codes, touched=False)
    here = np.bincount(owners, minlength=idx.size)
    if shared.any():
        here += _add_shared(acc, geoms, idx, shared, window, codes)
    # A field with no pixel centre in this window may still touch some of its pixels.
    # Its touched pixels are kept apart and used only if it holds no centre anywhere.
    for j in np.flatnonzero(here == 0):
        pixels = _field_pixels(geoms[idx[j]], window, all_touched=True)
        acc.add(np.full(pixels.size, idx[j]), pixels, codes, touched=True)


def _add_shared(
    acc: _Accumulator,
    geoms: np.ndarray,
    idx: np.ndarray,
    shared: np.ndarray,
    window: Window,
    codes: dict[str, np.ndarray],
) -> np.ndarray:
    """Pixels whose centre more than one field holds: each of them counts the pixel.

    Returns the shared pixels each field in idx took.
    """
    pixels = np.flatnonzero(shared)
    rows, cols = np.divmod(pixels, window.width)
    centres = shapely.points(window.x0 + (cols + 0.5) * PIXEL_M, window.y1 - (rows + 0.5) * PIXEL_M)
    _, hit = shapely.STRtree(geoms[idx]).query(centres, predicate="intersects")
    taken = np.zeros(idx.size, np.int64)
    for j in np.unique(hit):
        mine = _field_pixels(geoms[idx[j]], window, all_touched=False)
        mine = mine[shared[mine]]
        acc.add(np.full(mine.size, idx[j]), mine, codes, touched=False)
        taken[j] = mine.size
    return taken


# ---- work units ------------------------------------------------------------------


@dataclass
class _Task:
    """Fields whose centre falls in one block of one UTM zone."""

    zone: int
    rids: np.ndarray
    wkb: list[bytes]


def _transform(
    transformer: Transformer, x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Transform coordinate arrays of any length.

    pyproj 3.7 takes a one-element array down its scalar path, which numpy deprecates,
    so a lone coordinate is doubled and the copy dropped.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.size == 1:
        tx, ty = transformer.transform(np.repeat(x, 2), np.repeat(y, 2))
        return np.asarray(tx)[:1], np.asarray(ty)[:1]
    tx, ty = transformer.transform(x, y)
    return np.asarray(tx), np.asarray(ty)


def _zone_transformer(crs: CRS, zone: int) -> Transformer:
    return Transformer.from_crs(crs, CRS.from_epsg(32600 + zone), always_xy=True)


def _to_zone(geoms: np.ndarray, crs: CRS, zone: int) -> np.ndarray:
    transformer = _zone_transformer(crs, zone)

    def project(coords: np.ndarray) -> np.ndarray:
        x, y = _transform(transformer, coords[:, 0], coords[:, 1])
        return np.column_stack([x, y])

    return shapely.transform(geoms, project)


def _task_stats(
    task: _Task, crs: CRS, sources: Mapping[str, EmbeddingSource], tile_px: int
) -> pa.RecordBatch:
    """Statistics for one work unit's fields."""
    source_geoms = shapely.from_wkb(task.wkb)
    geoms = _to_zone(source_geoms, crs, task.zone)
    boxes = shapely.bounds(geoms)
    acc = _Accumulator(len(geoms), tuple(sources))
    usable = _usable(boxes)
    if crs.is_geographic:
        # Written across the antimeridian, a polygon spans most of the globe in
        # longitude and projects to a small but meaningless shape.
        lon = shapely.bounds(source_geoms)[:, [0, 2]]
        usable &= (lon[:, 1] - lon[:, 0]) <= 180
    geoms = np.where(usable, geoms, None)
    if usable.any():
        b = boxes[usable]
        bounds = (b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max())
        for tile in _tiles(bounds, task.zone, tile_px):
            idx = _window_fields(geoms, boxes, tile)
            if idx.size == 0:
                continue
            window = _clip(tile, boxes[idx])
            codes = {
                key: _with_retries(lambda s=source, w=window: s.read(w), f"{key} {window}")
                for key, source in sources.items()
            }
            _add_window(acc, geoms, idx, window, codes)
    return _batch(task.rids, acc)


def _usable(boxes: np.ndarray) -> np.ndarray:
    """Fields with finite bounds of a plausible size in their zone."""
    finite = np.isfinite(boxes).all(axis=1)
    size = np.where(finite[:, None], boxes[:, 2:] - boxes[:, :2], np.inf)
    return finite & (size < MAX_FIELD_M).all(axis=1)


def _clip(tile: Window, boxes: np.ndarray) -> Window:
    """The part of a tile the given field bboxes reach, snapped outward to the grid.

    Reads and rasterising cover only that, not the whole tile: a field spilling a few
    metres into a neighbouring tile then costs a sliver, not a tile.
    """
    x0 = max(tile.x0, math.floor(boxes[:, 0].min() / PIXEL_M) * PIXEL_M)
    x1 = min(tile.x1, math.ceil(boxes[:, 2].max() / PIXEL_M) * PIXEL_M)
    y0 = max(tile.y0, math.floor(boxes[:, 1].min() / PIXEL_M) * PIXEL_M)
    y1 = min(tile.y1, math.ceil(boxes[:, 3].max() / PIXEL_M) * PIXEL_M)
    width = max(round((x1 - x0) / PIXEL_M), 1)
    height = max(round((y1 - y0) / PIXEL_M), 1)
    return Window(tile.zone, x0, y1, width, height)


def _stats(sums: _Sums, pick: np.ndarray, other: _Sums) -> tuple[np.ndarray, ...]:
    """Pixel count, mean and std per field, from ``sums`` where pick else ``other``."""
    n = np.where(pick, sums.n, other.n)
    s = np.where(pick[:, None], sums.s, other.s)
    ss = np.where(pick[:, None], sums.ss, other.ss)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s / n[:, None]
        std = np.sqrt(np.maximum(ss / n[:, None] - mean * mean, 0.0))
    return n, mean.astype(np.float32), std.astype(np.float32)


def _vectors(values: np.ndarray, missing: np.ndarray) -> pa.Array:
    """64-value lists, null where missing.

    Null entries have zero length: Parquet writers refuse null list entries that still
    hold values (pyarrow does for fixed-size lists with a null between two values).
    """
    present = ~missing
    lengths = np.where(present, BANDS, 0)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int32)
    flat = pa.array(values[present].reshape(-1), type=pa.float32())
    return pa.ListArray.from_arrays(pa.array(offsets), flat, mask=pa.array(missing))


SCHEMA = pa.schema(
    [
        ("rid", pa.int64()),
        ("pixels", pa.int32()),
        ("pixel_rule", pa.string()),
        *[
            column
            for key in PRODUCTS
            for column in (
                (f"{key}_pixels", pa.int32()),
                *((f"{key}_{stat}", pa.list_(pa.float32())) for stat in STATISTICS),
            )
        ],
    ]
)


def _batch(rids: np.ndarray, acc: _Accumulator) -> pa.RecordBatch:
    centre = acc.centre_px > 0
    pixels = np.where(centre, acc.centre_px, acc.touched_px)
    rule = np.where(centre, PIXEL_RULE_CENTRE, PIXEL_RULE_TOUCHED)
    columns: dict[str, pa.Array] = {
        "rid": pa.array(rids, pa.int64()),
        "pixels": pa.array(pixels, pa.int32()),
        "pixel_rule": pa.array(rule.tolist(), pa.string(), mask=pixels == 0),
    }
    k = rids.size
    for key in PRODUCTS:
        if key not in acc.centre:
            columns[f"{key}_pixels"] = pa.nulls(k, pa.int32())
            for stat in STATISTICS:
                columns[f"{key}_{stat}"] = pa.nulls(k, SCHEMA.field(f"{key}_{stat}").type)
            continue
        n, mean, std = _stats(acc.centre[key], centre, acc.touched[key])
        columns[f"{key}_pixels"] = pa.array(n, pa.int32())
        columns[f"{key}_mean"] = _vectors(mean, n == 0)
        columns[f"{key}_std"] = _vectors(std, n == 0)
    return pa.RecordBatch.from_pydict(columns, schema=SCHEMA)


# ---- planning --------------------------------------------------------------------


def utm_zone(lon: np.ndarray) -> np.ndarray:
    """The UTM zone number of each longitude (1..60), as OlmoEarth assigns them."""
    return np.clip(np.floor((np.asarray(lon) + 180.0) / 6.0).astype(np.int64) + 1, 1, 60)


def _plan(con: duckdb.DuckDBPyConnection, fields_sql: str, geom_col: str, crs: CRS) -> int:
    """Table ``_keys`` (rid, zone, bx, by): each field's zone and block; returns its size."""
    rows = con.execute(f"""
        SELECT file_row_number AS rid, ST_X(c) AS x, ST_Y(c) AS y FROM (
            SELECT file_row_number,
                   -- A zero-area polygon's centroid is empty (or NULL); use its bbox centre.
                   CASE WHEN ST_Centroid("{geom_col}") IS NULL
                             OR ST_IsEmpty(ST_Centroid("{geom_col}"))
                        THEN ST_Point((ST_XMin("{geom_col}") + ST_XMax("{geom_col}")) / 2,
                                      (ST_YMin("{geom_col}") + ST_YMax("{geom_col}")) / 2)
                        ELSE ST_Centroid("{geom_col}") END AS c
            FROM {fields_sql}
            WHERE "{geom_col}" IS NOT NULL AND NOT ST_IsEmpty("{geom_col}"))""").fetchnumpy()
    rid = np.asarray(rows["rid"])
    x = np.ma.filled(np.ma.asarray(rows["x"], float), np.nan)
    y = np.ma.filled(np.ma.asarray(rows["y"], float), np.nan)
    to_lonlat = Transformer.from_crs(crs, CRS.from_epsg(4326), always_xy=True)
    lon, lat = _transform(to_lonlat, x, y)
    finite = np.isfinite(lon) & np.isfinite(lat)
    rid, lon, lat = rid[finite], lon[finite], lat[finite]
    zone = utm_zone(lon)
    bx = np.zeros(rid.size, np.int64)
    by = np.zeros(rid.size, np.int64)
    for z in np.unique(zone):
        sel = zone == z
        to_zone = Transformer.from_crs(4326, 32600 + int(z), always_xy=True)
        ux, uy = _transform(to_zone, lon[sel], lat[sel])
        bx[sel] = np.floor(ux / BLOCK_M)
        by[sel] = np.floor(uy / BLOCK_M)
    keys = pa.table({"rid": rid, "zone": zone, "bx": bx, "by": by})
    con.register("_keys_arrow", keys)
    con.execute("CREATE OR REPLACE TEMP TABLE _keys AS SELECT * FROM _keys_arrow")
    con.unregister("_keys_arrow")
    return rid.size


def _tasks(
    con: duckdb.DuckDBPyConnection, fields_sql: str, geom_col: str, batch_rows: int = 50_000
) -> Iterator[_Task]:
    """Fields grouped by (zone, block), streamed in key order."""
    reader = con.execute(f"""
        SELECT k.zone, k.bx, k.by, k.rid, ST_AsWKB(f."{geom_col}") AS wkb
        FROM {fields_sql} f JOIN _keys k ON k.rid = f.file_row_number
        ORDER BY k.zone, k.bx, k.by, k.rid""").fetch_record_batch(batch_rows)
    key: tuple[int, int, int] | None = None
    rids: list[int] = []
    wkb: list[bytes] = []
    for batch in reader:
        cols = batch.to_pydict()
        for zone, bx, by, rid, geom in zip(
            cols["zone"], cols["bx"], cols["by"], cols["rid"], cols["wkb"], strict=True
        ):
            if (zone, bx, by) != key:
                if rids:
                    yield _Task(key[0], np.array(rids, np.int64), wkb)
                key, rids, wkb = (zone, bx, by), [], []
            rids.append(rid)
            wkb.append(geom)
    if rids and key is not None:
        yield _Task(key[0], np.array(rids, np.int64), wkb)


# ---- entry point -----------------------------------------------------------------


@dataclass(frozen=True)
class FieldEmbeddingsResult:
    """Outcome of :func:`add_field_embeddings`."""

    total_fields: int
    by_centre: int
    by_touch: int
    products: tuple[str, ...]
    year: int

    @property
    def without_pixels(self) -> int:
        return self.total_fields - self.by_centre - self.by_touch


def field_embeddings_summary(result: FieldEmbeddingsResult | None) -> str:
    """One line for a run summary; None when this run wrote none (step off, or kept)."""
    if result is None:
        return "Field embeddings: none written this run"
    products = ", ".join(result.products) or "no product has this year"
    line = (
        f"Field embeddings ({products}, {result.year}): "
        f"{result.by_centre + result.by_touch:,} of {result.total_fields:,} fields"
    )
    if result.by_touch:
        line += f", {result.by_touch:,} from touched pixels because they hold no pixel centre"
    if result.without_pixels:
        line += f", {result.without_pixels:,} without pixels (no geometry, or not on a UTM grid)"
    return line


def describe(
    year: int,
    id_col: str,
    products: tuple[str, ...],
    fingerprint: str | None = None,
    requested: tuple[str, ...] | None = None,
) -> dict:
    """The JSON stored under ``METADATA_KEY``: what the table holds and where it came from.

    ``products`` are the ones with embeddings for the year; ``requested`` the ones
    looked for (all of them unless sources were given), so a year a product lacks is
    not mistaken for an incomplete file.
    """
    return {
        "algorithm_version": ALGORITHM_VERSION,
        "year": year,
        "id_column": id_col,
        "fields_fingerprint": fingerprint,
        "requested_products": sorted(requested if requested is not None else products),
        "statistics": list(STATISTICS),
        "std": "population standard deviation (ddof = 0) over the field's pixels",
        "dimensions": BANDS,
        "pixel_rule": (
            "pixels whose centre is inside the field; fields holding no pixel centre use "
            "every pixel they touch (pixel_rule = 'touched'); pixels with no data are left "
            "out per product (<product>_pixels)"
        ),
        "dequantization": DEQUANTIZATION,
        "products": {
            key: {
                "title": product.title,
                "url": product.url,
                "license": product.license,
                "attribution": product.attribution,
                "notes": product.notes,
                "included": key in products,
            }
            for key, product in PRODUCTS.items()
        },
    }


#: Output columns besides the id; the id column may not share a name with one.
OUTPUT_COLUMNS = tuple(name for name in SCHEMA.names if name != "rid")


def _check_fields(fields_path: Path, id_col: str) -> tuple[str, CRS]:
    """The geometry column and CRS, after checking the id column is usable."""
    from ftw_dataset_tools.api.geo import crs_is_explicit_null, geoparquet_crs

    if '"' in id_col or id_col in OUTPUT_COLUMNS or id_col == "rid":
        raise FieldEmbeddingsError(fields_path, f"'{id_col}' cannot be the id column")
    geom_col = detect_geometry_column(fields_path)
    if geom_col is None:
        raise FieldEmbeddingsError(fields_path, f"{fields_path.name} is not GeoParquet")
    if id_col not in parquet_columns(fields_path):
        raise FieldEmbeddingsError(fields_path, f"{fields_path.name} has no '{id_col}' column")
    if crs_is_explicit_null(fields_path, geom_col):
        raise FieldEmbeddingsError(
            fields_path, f"{fields_path.name} declares no CRS (crs: null); set one first"
        )
    try:
        crs = geoparquet_crs(fields_path, geom_col)
    except CRSError as err:
        raise FieldEmbeddingsError(fields_path, f"{fields_path.name}: unusable CRS") from err
    return geom_col, crs


def fields_fingerprint(fields_file: Path | str, id_col: str = "id") -> str:
    """A digest of the fields' CRS and every field's row number, id and geometry.

    Stable across library versions (BLAKE2 per row, combined by XOR so row order in
    the scan does not matter); changes when any field, id or the CRS does.
    """
    import hashlib

    fields_path = Path(fields_file).resolve()
    geom_col, crs = _check_fields(fields_path, id_col)
    combined = 0
    rows = 0
    con = duckdb.connect(":memory:")
    con.execute("SET enable_progress_bar = false")
    con.execute(f"SET memory_limit = '{DUCKDB_MEMORY_LIMIT}'")
    ensure_spatial_loaded(con)
    try:
        reader = con.execute(f"""
            SELECT file_row_number, "{id_col}"::VARCHAR AS id, ST_AsWKB("{geom_col}") AS wkb
            FROM read_parquet('{sql_path(fields_path)}', file_row_number = true)
            """).fetch_record_batch(100_000)
        for batch in reader:
            cols = batch.to_pydict()
            for rid, fid, wkb in zip(cols["file_row_number"], cols["id"], cols["wkb"], strict=True):
                h = hashlib.blake2b(f"{rid}\0{fid}\0".encode(), digest_size=16)
                h.update(wkb or b"")
                combined ^= int.from_bytes(h.digest(), "big")
                rows += 1
    finally:
        con.close()
    crs_digest = hashlib.blake2b(crs.to_json().encode(), digest_size=8).hexdigest()
    return f"{rows}:{crs_digest}:{combined:032x}"


def is_current(path: Path | str, fields_file: Path | str, *, year: int, id_col: str = "id") -> bool:
    """Whether ``path`` holds every product's statistics for these fields and year."""
    path = Path(path)
    if not path.exists():
        return False
    description = read_description(path) or {}
    return (
        description.get("algorithm_version") == ALGORITHM_VERSION
        and description.get("year") == year
        and description.get("id_column") == id_col
        and set(description.get("requested_products", ())) == set(PRODUCTS)
        and description.get("fields_fingerprint") == fields_fingerprint(fields_file, id_col)
    )


def add_field_embeddings(
    fields_file: Path | str,
    output_file: Path | str,
    *,
    year: int,
    id_col: str = "id",
    workers: int = DEFAULT_WORKERS,
    sources: Mapping[str, EmbeddingSource] | None = None,
    tile_px: int = TILE_PX,
    on_progress: Callable[[int, int], None] | None = None,
) -> FieldEmbeddingsResult:
    """Write per-field embedding statistics for a fields GeoParquet.

    Finished fields are saved under ``<output_file>.partial/`` as the run goes; a run
    stopped part way (a network failure, a killed process) resumes from there when
    started again on the same fields, year and products. The folder is removed once
    the output is written.

    Args:
        fields_file: Field polygons (GeoParquet, any CRS).
        output_file: Parquet to write: the id column, then pixel counts and statistics.
        year: Embedding year.
        id_col: Field id column, copied to the output as the join key.
        workers: Work units read and summarised at once.
        sources: Product key -> source; defaults to the remote products for ``year``.
        tile_px: Window size in pixels (memory per worker grows with its square).
        on_progress: Called with (fields done, total fields).
    """
    fields_path = Path(fields_file).resolve()
    output_path = Path(output_file).resolve()
    geom_col, crs = _check_fields(fields_path, id_col)
    requested = tuple(PRODUCTS) if sources is None else tuple(sources)
    if sources is None:
        sources = open_sources(year)
    fingerprint = fields_fingerprint(fields_path, id_col)
    description = describe(year, id_col, tuple(sources), fingerprint, requested)
    fields_sql = f"read_parquet('{sql_path(fields_path)}', file_row_number = true)"

    with _Parts.open(output_path, description) as parts:
        con = duckdb.connect(":memory:")
        con.execute("SET enable_progress_bar = false")
        con.execute(f"SET memory_limit = '{DUCKDB_MEMORY_LIMIT}'")
        con.execute(f"SET temp_directory = '{sql_path(parts.root / 'duckdb')}'")
        ensure_spatial_loaded(con)
        try:
            _warn_on_ids(con, fields_sql, id_col)
            total = _plan(con, fields_sql, geom_col, crs)
            resumed = parts.skip_done(con)
            if resumed:
                logger.info(f"Resuming: {resumed:,} of {total:,} fields already done")
            if sources:
                _compute(
                    con,
                    fields_sql,
                    geom_col,
                    crs,
                    sources,
                    parts,
                    workers,
                    tile_px,
                    total,
                    resumed,
                    on_progress,
                )
            else:
                logger.warning(f"No embedding product has {year}; writing empty statistics")
            parts.finish()
            parts.check(con, total if sources else 0)
            _write_output(con, fields_sql, id_col, parts, output_path, description)
            result = _result(con, output_path, year, tuple(sources))
        finally:
            con.close()
        parts.remove()
    logger.info(field_embeddings_summary(result))
    return result


@dataclass
class _Parts:
    """Finished fields of a run, saved as Parquet parts beside the output.

    A lock file next to the output keeps a second run on the same output from adopting
    or deleting this run's parts.
    """

    root: Path
    lock: Any
    count: int = 0

    @classmethod
    def open(cls, output_path: Path, description: dict) -> _Parts:
        """The run's parts folder: kept when it belongs to the same run, else started over."""
        import fcntl
        import shutil

        lock = _lock_path(output_path).open("w")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as err:
            lock.close()
            raise FieldEmbeddingsError(
                output_path, f"another run is writing {output_path.name}"
            ) from err
        root = output_path.with_name(output_path.name + ".partial")
        marker = root / "run.json"
        try:
            same = json.loads(marker.read_text()) == description
        except (OSError, ValueError):
            same = False
        if root.exists() and not same:
            shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)
        tmp = marker.with_suffix(".tmp")
        tmp.write_text(json.dumps(description))
        tmp.replace(marker)
        return cls(root, lock, count=len(list(root.glob("part-*.parquet"))))

    def __enter__(self) -> _Parts:
        return self

    def __exit__(self, *_exc: object) -> None:
        # The lock file stays: unlinking it would let a waiting run lock a stale inode.
        self.lock.close()

    @property
    def glob(self) -> str:
        return sql_path(self.root / "part-*.parquet")

    def skip_done(self, con: duckdb.DuckDBPyConnection) -> int:
        """Drop the fields earlier parts hold from ``_keys``; returns how many."""
        if self.count == 0:
            return 0
        con.execute(f"DELETE FROM _keys WHERE rid IN (SELECT rid FROM read_parquet('{self.glob}'))")
        return con.execute(f"SELECT count(*) FROM read_parquet('{self.glob}')").fetchone()[0]

    def write(self, batches: list[pa.RecordBatch]) -> None:
        """Save finished batches as the next part, atomically."""
        path = self.root / f"part-{self.count:06d}.parquet"
        tmp = path.with_suffix(".tmp")
        pq.write_table(pa.Table.from_batches(batches, schema=SCHEMA), tmp)
        tmp.replace(path)
        self.count += 1

    def finish(self) -> None:
        """At least one part, so the output's join has a schema to read."""
        if self.count == 0:
            self.write([])

    def check(self, con: duckdb.DuckDBPyConnection, expected: int) -> None:
        """Every planned field exactly once, before anything is published."""
        rows, distinct = con.execute(
            f"SELECT count(*), count(DISTINCT rid) FROM read_parquet('{self.glob}')"
        ).fetchone()
        if rows != distinct or rows != expected:
            raise FieldEmbeddingsError(
                self.root,
                f"saved parts hold {rows:,} rows for {distinct:,} fields, expected {expected:,}; "
                f"delete {self.root.name} and run again",
            )

    def remove(self) -> None:
        import shutil

        shutil.rmtree(self.root, ignore_errors=True)


def _lock_path(output_path: Path) -> Path:
    """Where a run on ``output_path`` keeps its lock: the temp dir, not the dataset folder."""
    import hashlib

    digest = hashlib.blake2b(str(output_path).encode(), digest_size=8).hexdigest()
    return Path(tempfile.gettempdir()) / f"ftwd-field-embeddings-{digest}.lock"


def _warn_on_ids(con: duckdb.DuckDBPyConnection, fields_sql: str, id_col: str) -> None:
    """Duplicate or missing ids make the output ambiguous to join; say so."""
    rows, distinct, nulls = con.execute(f"""
        SELECT count(*), count(DISTINCT "{id_col}"), count(*) - count("{id_col}")
        FROM {fields_sql}""").fetchone()
    if nulls or distinct < rows - nulls:
        logger.warning(
            f"'{id_col}' is not a unique key ({rows - nulls - distinct:,} duplicates, "
            f"{nulls:,} missing); rows stay in the fields' order, but joins on it will not"
        )


def _compute(
    con: duckdb.DuckDBPyConnection,
    fields_sql: str,
    geom_col: str,
    crs: CRS,
    sources: Mapping[str, EmbeddingSource],
    parts: _Parts,
    workers: int,
    tile_px: int,
    total: int,
    done: int,
    on_progress: Callable[[int, int], None] | None,
) -> None:
    """Run every work unit on a thread pool, saving results in parts as they finish.

    On a failure, the results already finished are saved before it is raised, and the
    queued work units are cancelled rather than waited for.
    """
    buffer: list[pa.RecordBatch] = []

    def drain(futures: set[Future]) -> None:
        nonlocal done
        error: BaseException | None = None
        for future in futures:
            try:
                batch = future.result()
            except BaseException as err:  # keep collecting the others first
                error = error or err
                continue
            buffer.append(batch)
            done += batch.num_rows
            if on_progress is not None:
                on_progress(done, total)
        if buffer and (error is not None or sum(b.num_rows for b in buffer) >= PART_ROWS):
            parts.write(buffer)
            buffer.clear()
        if error is not None:
            raise error

    pool = ThreadPoolExecutor(workers)
    try:
        pending: set[Future] = set()
        for task in _tasks(con, fields_sql, geom_col):
            pending.add(pool.submit(_task_stats, task, crs, sources, tile_px))
            if len(pending) >= 2 * workers:
                finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                drain(finished)
        drain(pending)
        if buffer:
            parts.write(buffer)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def _write_output(
    con: duckdb.DuckDBPyConnection,
    fields_sql: str,
    id_col: str,
    parts: _Parts,
    output_path: Path,
    description: dict,
) -> None:
    """The statistics joined onto the field ids, in the fields file's order."""
    meta = json.dumps(description).replace("'", "''")
    tmp_out = output_path.with_name(output_path.name + ".tmp")
    con.execute(f"""
        COPY (
            SELECT f."{id_col}", s.* EXCLUDE (rid)
            FROM {fields_sql} f LEFT JOIN read_parquet('{parts.glob}') s
              ON s.rid = f.file_row_number
            ORDER BY f.file_row_number
        ) TO '{sql_path(tmp_out)}'
        (FORMAT parquet, COMPRESSION zstd, KV_METADATA {{'{METADATA_KEY}': '{meta}'}})""")
    tmp_out.replace(output_path)


def _result(
    con: duckdb.DuckDBPyConnection, output_path: Path, year: int, products: tuple[str, ...]
) -> FieldEmbeddingsResult:
    total, centre, touched = con.execute(f"""
        SELECT count(*), count(*) FILTER (WHERE pixel_rule = '{PIXEL_RULE_CENTRE}'),
               count(*) FILTER (WHERE pixel_rule = '{PIXEL_RULE_TOUCHED}')
        FROM read_parquet('{sql_path(output_path)}')""").fetchone()
    return FieldEmbeddingsResult(total, centre, touched, products, year)


def read_description(path: Path | str) -> dict | None:
    """The ``METADATA_KEY`` description of a field embeddings file, or None."""
    metadata = pq.read_metadata(path).metadata or {}
    raw = metadata.get(METADATA_KEY.encode())
    return json.loads(raw) if raw else None
