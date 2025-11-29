"""Grid utilities for fetching MGRS grids from cloud sources."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import duckdb

from ftw_dataset_tools.api.geo import detect_crs

if TYPE_CHECKING:
    from collections.abc import Callable

# Default MGRS grid source
DEFAULT_GRID_SOURCE = "s3://us-west-2.opendata.source.coop/tge-labs/mgrs/gzd_partition/*/*.parquet"


class CRSError(Exception):
    """Raised when input file has incorrect CRS."""

    def __init__(self, crs: str, input_file: str) -> None:
        self.crs = crs
        self.input_file = input_file
        super().__init__(
            f"Input file has CRS '{crs}', but must be EPSG:4326.\n"
            f"Please reproject first with:\n"
            f"  ftwd reproject {input_file} --target-crs EPSG:4326"
        )


@dataclass
class GetGridResult:
    """Result of a get-grid operation."""

    output_path: Path
    grid_count: int
    bounds: tuple[float, float, float, float]  # xmin, ymin, xmax, ymax


def get_grid(
    input_file: str | Path,
    output_file: str | Path | None = None,
    grid_source: str = DEFAULT_GRID_SOURCE,
    geom_col: str = "geometry",
    precise: bool = False,
    on_progress: Callable[[str], None] | None = None,
) -> GetGridResult:
    """
    Fetch MGRS grid cells that intersect the input file's geometries.

    Args:
        input_file: Path to input GeoParquet file (must be EPSG:4326)
        output_file: Path to output file. If None, creates <input>_grid.parquet
        grid_source: URL/path to the grid source (default: MGRS grid on Source Coop)
        geom_col: Name of the geometry column in the input file
        precise: If True, use geometry union for precise matching. If False (default),
                 use bounding box only (faster but may include extra grids).
        on_progress: Optional callback for progress messages

    Returns:
        GetGridResult with information about the operation

    Raises:
        FileNotFoundError: If input file doesn't exist
        CRSError: If input file is not in EPSG:4326
    """
    input_path = Path(input_file).resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    def log(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    # Check CRS using geo module
    crs_info = detect_crs(input_path, geom_col)
    crs_str = str(crs_info)

    if crs_info.authority_code is None or crs_info.authority_code.upper() != "EPSG:4326":
        raise CRSError(crs_str, str(input_file))

    log(f"CRS: {crs_str}")

    # Create DuckDB connection with required extensions
    conn = duckdb.connect(":memory:")
    conn.execute("INSTALL spatial; LOAD spatial;")
    conn.execute("INSTALL httpfs; LOAD httpfs;")

    # Set S3 region for Source Coop
    conn.execute("SET s3_region = 'us-west-2';")

    # Load input file
    log("Loading input file...")
    conn.execute(f"CREATE TABLE input_data AS SELECT * FROM '{input_path}'")
    feature_count = conn.execute("SELECT COUNT(*) FROM input_data").fetchone()[0]
    log(f"Loaded {feature_count:,} features")

    # Compute bounds for bbox filtering
    bounds_result = conn.execute(f"""
        SELECT
            MIN(ST_XMin("{geom_col}")) as xmin,
            MIN(ST_YMin("{geom_col}")) as ymin,
            MAX(ST_XMax("{geom_col}")) as xmax,
            MAX(ST_YMax("{geom_col}")) as ymax
        FROM input_data
    """).fetchone()
    xmin, ymin, xmax, ymax = bounds_result
    log(f"Bounds: [{xmin:.6f}, {ymin:.6f}, {xmax:.6f}, {ymax:.6f}]")

    # Fetch grids that intersect the bounding box
    log("Fetching grid cells by bounding box...")
    conn.execute(f"""
        CREATE TABLE grid_bbox AS
        SELECT *
        FROM '{grid_source}'
        WHERE "bbox".xmin <= {xmax}
          AND "bbox".xmax >= {xmin}
          AND "bbox".ymin <= {ymax}
          AND "bbox".ymax >= {ymin}
    """)

    bbox_count = conn.execute("SELECT COUNT(*) FROM grid_bbox").fetchone()[0]
    log(f"Found {bbox_count:,} grid cells in bounding box")

    if precise:
        # Compute union of all geometries for precise filtering
        log("Computing geometry union for precise matching...")
        conn.execute(f"""
            CREATE TABLE input_union AS
            SELECT ST_Union_Agg("{geom_col}") as union_geom FROM input_data
        """)

        # Get union geometry as WKT for reporting
        union_wkt = conn.execute("SELECT ST_AsText(union_geom) FROM input_union").fetchone()[0]
        log(f"Union geometry: {union_wkt}")

        # Filter locally using geometry intersection
        log("Filtering grids by geometry intersection...")
        conn.execute("""
            CREATE TABLE grid_result AS
            SELECT g.*
            FROM grid_bbox g, input_union u
            WHERE ST_Intersects(g.geom, u.union_geom)
        """)
    else:
        # Use bbox results directly
        conn.execute("CREATE TABLE grid_result AS SELECT * FROM grid_bbox")

    # Get count
    grid_count = conn.execute("SELECT COUNT(*) FROM grid_result").fetchone()[0]
    log(f"Found {grid_count:,} intersecting grid cells")

    # Determine output path
    if output_file:
        out_path = Path(output_file).resolve()
    else:
        out_path = input_path.parent / f"{input_path.stem}_grid.parquet"

    # Write output with zstd compression level 16
    log(f"Writing output to: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    conn.execute(f"""
        COPY grid_result TO '{out_path}'
        (FORMAT PARQUET, COMPRESSION 'zstd', COMPRESSION_LEVEL 16)
    """)

    conn.close()

    return GetGridResult(
        output_path=out_path,
        grid_count=grid_count,
        bounds=(xmin, ymin, xmax, ymax),
    )
