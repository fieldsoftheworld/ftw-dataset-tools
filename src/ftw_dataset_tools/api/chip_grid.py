"""Chip sizes, FTW chip ids, and cutting the FTW grid to a custom chip size.

A chip id is ``ftw-<square><easting><northing>``, the chip's south-west corner in its
MGRS 100 km square, at the coarsest precision that fits the chip size: 2 digits per
axis for whole km (``ftw-36NXF6658``), up to 5 for any whole metre.
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import duckdb
    import pandas as pd

__all__ = [
    "GridCutError",
    "InvalidChipSizeError",
    "chip_size_label",
    "chip_size_m",
    "cut_grid",
    "expand_bounds",
    "id_digits",
    "parse_chip_ids",
    "utm_crs_for_cell",
]

SQUARE_M = 100_000
# Group 1 is the square (e.g. 36NXF), group 2 the offset digits.
CHIP_ID_PATTERN = r"^ftw-(\d{1,2}[A-Z]{3})((?:\d\d){2,5})$"
# Reprojected edges land sub-millimetre off the lattice; snapping stops slivers leaking.
_SNAP_M = 0.01
# Edge vertex spacing, as in the published grid, so big chips follow the UTM lines.
_EDGE_STEP_M = 1000


class InvalidChipSizeError(ValueError):
    """A chip size that is not a whole number of metres between 1 m and 100 km."""

    def __init__(self, km_size: object, message: str) -> None:
        super().__init__(message)
        self.km_size = km_size


class GridCutError(ValueError):
    """The grid cannot be cut to a new chip size because its ids are not FTW chip ids."""

    def __init__(self, bad_ids: list[str]) -> None:
        super().__init__(
            "Cannot cut the grid to a custom chip size: its cells need FTW chip ids "
            f"(ftw-<MGRS square><easting><northing>), but found {bad_ids[:5]}"
        )
        self.bad_ids = bad_ids


def chip_size_m(km_size: float) -> int:
    """Chip edge length in whole metres, validating ``km_size`` (given in kilometres).

    Raises:
        InvalidChipSizeError: If the size is not a whole number of metres between
            1 m and 100 km (one MGRS square).
    """
    if isinstance(km_size, bool) or not isinstance(km_size, int | float):
        raise InvalidChipSizeError(km_size, f"km_size must be a number (got {km_size!r})")
    if not math.isfinite(km_size) or km_size <= 0:
        raise InvalidChipSizeError(km_size, f"km_size must be positive (got {km_size!r})")
    if km_size > SQUARE_M / 1000:
        raise InvalidChipSizeError(
            km_size,
            f"km_size is in kilometres and cannot exceed 100 (one MGRS square); got "
            f"{km_size:g}. For {km_size:g} m chips pass {km_size / 1000:g}.",
        )
    metres = round(km_size * 1000)
    if metres < 1 or abs(km_size * 1000 - metres) > 1e-6:
        raise InvalidChipSizeError(
            km_size,
            f"km_size must be a whole number of metres (got {km_size:g} km = {km_size * 1000:g} m)",
        )
    return metres


def id_digits(size_m: int) -> int:
    """Digits per axis in the ids of chips ``size_m`` metres wide (2 for whole km)."""
    return next((d for d in (2, 3, 4) if size_m % 10 ** (5 - d) == 0), 5)


def chip_size_label(km_size: float) -> str:
    """Human-readable chip size, e.g. '0.1 km (100 m)'."""
    return f"{km_size:g} km ({chip_size_m(km_size):,} m)"


def parse_chip_ids(chip_ids: pd.Series) -> pd.DataFrame:
    """Split FTW chip ids into their MGRS square and south-west corner offsets in metres.

    Returns a frame indexed like ``chip_ids`` with columns ``square``, ``east_m``,
    ``north_m`` and ``unit_m`` (the precision of the id's digits: 1000 for 2-digit
    ids). Ids of any precision (2 to 5 digits per axis) parse, so 2 km and 100 m chip
    ids yield offsets on the same metre scale.

    Raises:
        ValueError: If any id does not match ``ftw-<square><easting><northing>``.
    """
    import pandas as pd

    parts = chip_ids.astype(str).str.extract(CHIP_ID_PATTERN)
    bad = parts[0].isna()
    if bad.any():
        raise ValueError(
            "Invalid chip ID format: unable to extract numeric easting/northing. Expected "
            f"ftw-<zone><band><square><easting><northing>; found {chip_ids[bad].tolist()[:5]}"
        )
    digits = parts[1]
    half = digits.str.len() // 2
    unit = (10 ** (5 - half)).astype("int64")
    east = pd.Series(0, index=chip_ids.index, dtype="int64")
    north = pd.Series(0, index=chip_ids.index, dtype="int64")
    for h in half.unique():
        rows = half == h
        east[rows] = digits[rows].str[:h].astype("int64")
        north[rows] = digits[rows].str[h:].astype("int64")
    return pd.DataFrame(
        {"square": parts[0], "east_m": east * unit, "north_m": north * unit, "unit_m": unit}
    )


def expand_bounds(
    bounds: tuple[float, float, float, float], km_size: float
) -> tuple[float, float, float, float]:
    """Grow lon/lat bounds by one chip on every side.

    A chip touching ``bounds`` extends up to one chip beyond them, and cutting it
    needs every source cell under it, not only those inside ``bounds``.
    """
    xmin, ymin, xmax, ymax = bounds
    margin = chip_size_m(km_size) / 111_320
    cos_lat = max(math.cos(math.radians(min(max(abs(ymin), abs(ymax)) + margin, 90))), 0.01)
    return (
        max(xmin - margin / cos_lat, -180.0),
        max(ymin - margin, -90.0),
        min(xmax + margin / cos_lat, 180.0),
        min(ymax + margin, 90.0),
    )


def cut_grid(
    conn: duckdb.DuckDBPyConnection,
    table: str,
    km_size: float,
    keep_bounds: tuple[float, float, float, float] | None = None,
    geom_col: str = "geometry",
) -> int:
    """Replace a lon/lat FTW grid table with cells of ``km_size`` km, in place.

    Source cells are cut in UTM along a lattice anchored at their MGRS 100 km square
    and the pieces of each new cell merged, so new cells are clipped at square, zone
    and band edges just like the published grid. The published grid's columns are kept.

    Args:
        conn: DuckDB connection with the spatial extension loaded
        table: Grid table in lon/lat with FTW chip ids in ``id``
        km_size: New chip edge length in kilometres
        keep_bounds: If set, keep only cells whose bbox overlaps these lon/lat bounds
        geom_col: Geometry column of ``table``

    Returns:
        Number of cells in the cut grid.

    Raises:
        InvalidChipSizeError: If ``km_size`` is not a whole number of metres up to 100 km
        GridCutError: If any id in ``table`` is not an FTW chip id
    """
    size_m = chip_size_m(km_size)
    bad_ids = [
        row[0]
        for row in conn.execute(
            f"SELECT id FROM \"{table}\" WHERE NOT regexp_matches(id, '{CHIP_ID_PATTERN}') LIMIT 5"
        ).fetchall()
    ]
    if bad_ids:
        raise GridCutError(bad_ids)

    geom_type = conn.execute(f'SELECT typeof("{geom_col}") FROM "{table}" LIMIT 1').fetchone()
    cast = f"::{geom_type[0]}" if geom_type else ""
    keep = ""
    if keep_bounds is not None:
        xmin, ymin, xmax, ymax = keep_bounds
        keep = (
            f"WHERE ST_XMin(geom) <= {xmax} AND ST_XMax(geom) >= {xmin} "
            f"AND ST_YMin(geom) <= {ymax} AND ST_YMax(geom) >= {ymin}"
        )
    query = _CUT_QUERY.format(
        table=table,
        geom_col=geom_col,
        size=size_m,
        square=SQUARE_M,
        last=-(-SQUARE_M // size_m) - 1,
        digits=id_digits(size_m),
        unit=10 ** (5 - id_digits(size_m)),
        snap=_SNAP_M,
        edge_steps=max(1, -(-size_m // _EDGE_STEP_M)),
        pattern=CHIP_ID_PATTERN,
        cast=cast,
        keep=keep,
    )
    conn.execute(f'CREATE TABLE "{table}_cut" AS {query}')
    conn.execute(f'DROP TABLE "{table}"')
    conn.execute(f'ALTER TABLE "{table}_cut" RENAME TO "{table}"')
    return conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]


# The square origin is the centroid less the id's offset, not the bounds: a few published
# cells carry stray slivers outside their square, which the clamped lattice drops.
_CUT_QUERY = """
WITH parsed AS (
    SELECT
        regexp_extract(id, '{pattern}', 1) AS square,
        regexp_extract(id, '{pattern}', 2) AS digits,
        length(digits) // 2 AS half,
        CAST(substr(digits, 1, half) AS BIGINT) * pow(10, 5 - half) AS east_m,
        CAST(substr(digits, half + 1, half) AS BIGINT) * pow(10, 5 - half) AS north_m,
        'EPSG:' || (
            CASE WHEN regexp_extract(square, '^\\d{{1,2}}([A-Z])', 1) >= 'N' THEN 32600 ELSE 32700 END
            + CAST(regexp_extract(square, '^(\\d{{1,2}})', 1) AS INTEGER)
        ) AS utm,
        ST_ReducePrecision(
            ST_Transform("{geom_col}"::GEOMETRY, 'OGC:CRS84', utm), {snap}
        ) AS g
    FROM "{table}"
),
framed AS (
    SELECT
        *,
        floor((ST_X(ST_Centroid(g)) - east_m + 0.5) / {square}) * {square} AS x0,
        floor((ST_Y(ST_Centroid(g)) - north_m + 0.5) / {square}) * {square} AS y0
    FROM parsed
    WHERE NOT ST_IsEmpty(g)
),
pieces AS (
    SELECT
        square, utm, ix, iy,
        ST_MakeEnvelope(
            x0 + ix * {size}, y0 + iy * {size},
            LEAST(x0 + (ix + 1) * {size}, x0 + {square}),
            LEAST(y0 + (iy + 1) * {size}, y0 + {square})
        ) AS box,
        ST_Intersection(box, g) AS piece
    FROM framed,
        UNNEST(generate_series(
            GREATEST(0, CAST(floor((ST_XMin(g) - x0) / {size}) AS BIGINT)),
            LEAST({last}, CAST(ceil((ST_XMax(g) - x0) / {size}) AS BIGINT) - 1)
        )) AS cols(ix),
        UNNEST(generate_series(
            GREATEST(0, CAST(floor((ST_YMin(g) - y0) / {size}) AS BIGINT)),
            LEAST({last}, CAST(ceil((ST_YMax(g) - y0) / {size}) AS BIGINT) - 1)
        )) AS rows_(iy)
),
cells AS (
    SELECT square, utm, ix, iy, ANY_VALUE(box) AS box, ST_Union_Agg(piece) AS g
    FROM pieces
    WHERE ST_Area(piece) > 0
    GROUP BY square, utm, ix, iy
),
projected AS (
    SELECT
        regexp_extract(square, '^(\\d{{1,2}}[A-Z])', 1) AS gzd,
        square
            || CAST(ix * {size} // 10000 AS VARCHAR)
            || CAST(iy * {size} // 10000 AS VARCHAR) AS mgrs_10km,
        'ftw-' || square
            || lpad(CAST(ix * {size} // {unit} AS VARCHAR), {digits}, '0')
            || lpad(CAST(iy * {size} // {unit} AS VARCHAR), {digits}, '0') AS id,
        ST_Transform(
            CASE WHEN ST_Area(g) >= ST_Area(box) * (1 - 1e-9) THEN ST_MakePolygon(ST_MakeLine(
                [ST_Point(ST_XMin(box) + i * (ST_XMax(box) - ST_XMin(box)) / {edge_steps},
                          ST_YMin(box)) FOR i IN range({edge_steps})]
                || [ST_Point(ST_XMax(box),
                             ST_YMin(box) + i * (ST_YMax(box) - ST_YMin(box)) / {edge_steps})
                    FOR i IN range({edge_steps})]
                || [ST_Point(ST_XMax(box) - i * (ST_XMax(box) - ST_XMin(box)) / {edge_steps},
                             ST_YMax(box)) FOR i IN range({edge_steps})]
                || [ST_Point(ST_XMin(box),
                             ST_YMax(box) - i * (ST_YMax(box) - ST_YMin(box)) / {edge_steps})
                    FOR i IN range({edge_steps} + 1)]
            )) ELSE g END,
            utm, 'OGC:CRS84'
        ) AS geom
    FROM cells
)
SELECT
    gzd,
    mgrs_10km,
    id,
    geom{cast} AS "{geom_col}",
    STRUCT_PACK(
        xmin := ST_XMin(geom), ymin := ST_YMin(geom),
        xmax := ST_XMax(geom), ymax := ST_YMax(geom)
    ) AS bbox
FROM projected
{keep}
ORDER BY id
"""


_FTW_ZONE = re.compile(r"^ftw-(?P<zone>\d{1,2})(?P<band>[C-HJ-NP-X])[A-HJ-NP-Z]{2}\d")


def utm_crs_for_cell(grid_id: str, bounds: tuple[float, float, float, float]) -> str:
    """The UTM CRS (``EPSG:326NN`` / ``EPSG:327NN``) a grid cell's native grid uses.

    FTW chips take the zone from their own MGRS square, so the grid always agrees
    with the chip id, also at zone seams and in the Norway/Svalbard exceptions.
    Other ids fall back to the zone of the cell's centre, given in lon/lat ``bounds``.
    """
    match = _FTW_ZONE.match(grid_id)
    if match:
        zone = int(match.group("zone"))
        north = match.group("band") >= "N"
    else:
        lon = (bounds[0] + bounds[2]) / 2
        lat = (bounds[1] + bounds[3]) / 2
        zone = min(int((lon + 180) // 6) + 1, 60)
        north = lat >= 0
    return f"EPSG:{32600 + zone if north else 32700 + zone}"
