"""Identify chips lying on the edge of a labelled cluster.

Source label collections are sampled as separate blocks, and labelling stops at a
block edge without following the chip grid. A chip straddling that edge is only
partly labelled: its unlabelled side rasterises to the background class, so real
fields are recorded as "not a field". Such chips have to go.

The labelled region is estimated from the chips that hold fields: a morphological
closing bridges gaps narrower than ``gap_chips``, enclosed holes (lakes, forests,
towns) are filled so they never register as edges, and a chip is a border chip
when growing it by one chip width escapes the region.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import duckdb

__all__ = [
    "DEFAULT_BORDER_GAP_CHIPS",
    "BorderChipResult",
    "find_border_chips",
]

# How wide an unlabelled gap must be, in chips, before it counts as a cluster edge.
DEFAULT_BORDER_GAP_CHIPS = 2


@dataclass
class BorderChipResult:
    """Chips on a cluster edge, and how many clusters were found."""

    border_rowids: list[int]
    cluster_count: int

    @property
    def border_count(self) -> int:
        return len(self.border_rowids)


def chip_width(conn: duckdb.DuckDBPyConnection, table: str, geom_col: str) -> float:
    """Width of a single chip, taken as the median extent of the chip geometries."""
    width = conn.execute(f"""
        SELECT median(ST_XMax("{geom_col}") - ST_XMin("{geom_col}"))
        FROM "{table}"
    """).fetchone()[0]
    if not width or width <= 0:
        raise ValueError(f"could not determine chip width from {table}.{geom_col}")
    return float(width)


def find_border_chips(
    conn: duckdb.DuckDBPyConnection,
    table: str,
    geom_col: str,
    coverage_col: str,
    *,
    gap_chips: int = DEFAULT_BORDER_GAP_CHIPS,
    rowid_col: str = "rowid",
) -> BorderChipResult:
    """Find chips on the edge of any labelled cluster.

    Args:
        conn: DuckDB connection with the spatial extension loaded
        table: Table holding one row per chip
        geom_col: Chip geometry column
        coverage_col: Field coverage column; chips above zero define the labelled region
        gap_chips: How wide an unlabelled gap must be, in chips, to count as a cluster edge
        rowid_col: Expression identifying a row, returned in ``border_rowids``

    Raises:
        ValueError: If ``gap_chips`` is negative or the chip width cannot be determined
    """
    if gap_chips < 0:
        raise ValueError(f"gap_chips must be 0 or greater, got {gap_chips}")

    occupied = conn.execute(
        f'SELECT COUNT(*) FROM "{table}" WHERE "{coverage_col}" > 0'
    ).fetchone()[0]
    if occupied == 0:
        return BorderChipResult(border_rowids=[], cluster_count=0)

    width = chip_width(conn, table, geom_col)
    buffer = gap_chips * width / 2

    # Closing (dilate then erode) bridges gaps narrower than 2*buffer; taking each
    # part's exterior ring then fills any hole enclosed by labelled chips.
    closing = f'ST_Union_Agg("{geom_col}")'
    if buffer > 0:
        closing = f"ST_Buffer(ST_Buffer({closing}, {buffer}), -{buffer})"

    conn.execute(f"""
        CREATE OR REPLACE TEMP TABLE _border_region AS
        WITH closed AS (
            SELECT {closing} AS geom
            FROM "{table}"
            WHERE "{coverage_col}" > 0
        ), parts AS (
            SELECT UNNEST(ST_Dump(geom)).geom AS geom FROM closed
        )
        SELECT ST_MakePolygon(ST_ExteriorRing(geom)) AS geom FROM parts
    """)

    cluster_count = conn.execute("SELECT COUNT(*) FROM _border_region").fetchone()[0]

    # Growing a chip by one chip width and requiring containment is an erosion by a
    # 3x3 neighbourhood: a chip is a border chip when a neighbour falls outside.
    # Chips with no fields are outside every cluster, so they fail this test too and are
    # dropped along with the edges, as the previous convex-hull rule did.
    rows = conn.execute(f"""
        SELECT c."{rowid_col}"
        FROM "{table}" c
        WHERE NOT EXISTS (
            SELECT 1 FROM _border_region r
            WHERE ST_Within(ST_Buffer(c."{geom_col}", {width}), r.geom)
        )
    """).fetchall()

    conn.execute("DROP TABLE IF EXISTS _border_region")

    return BorderChipResult(
        border_rowids=[row[0] for row in rows],
        cluster_count=cluster_count,
    )
