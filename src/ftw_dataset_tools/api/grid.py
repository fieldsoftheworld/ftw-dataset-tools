"""Grid utilities for fetching MGRS grids from cloud sources."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import duckdb

from ftw_dataset_tools.api.geo import detect_crs

if TYPE_CHECKING:
    from collections.abc import Callable

# Default MGRS grid source
DEFAULT_GRID_SOURCE = "s3://us-west-2.opendata.source.coop/tge-labs/mgrs/gzd_partition/*/*.parquet"


def get_grid_cache_dir() -> Path:
    """
    Get the cache directory for grid query results.

    Uses FTW_CACHE_DIR environment variable if set,
    otherwise defaults to ~/.cache/ftw-tools/grid/

    Returns:
        Path to cache directory
    """
    cache_base = os.environ.get("FTW_CACHE_DIR")
    cache_base_path = Path(cache_base) if cache_base else Path.home() / ".cache" / "ftw-tools"
    cache_dir = cache_base_path / "grid"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _get_cache_key(bounds: tuple[float, float, float, float], precise: bool) -> str:
    """Generate a cache key from bounds and precision setting."""
    # Round bounds to 4 decimal places (~11m precision) to allow some tolerance
    rounded = tuple(round(b, 4) for b in bounds)
    key_str = f"{rounded}_{precise}"
    return hashlib.md5(key_str.encode()).hexdigest()[:12]


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
    use_cache: bool = True,
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
        use_cache: If True (default), cache grid results by bounding box for faster
                   repeated queries in the same area.
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
    bounds = (xmin, ymin, xmax, ymax)
    log(f"Bounds: [{xmin:.6f}, {ymin:.6f}, {xmax:.6f}, {ymax:.6f}]")

    # Check cache for existing grid results
    cache_file = None
    if use_cache:
        cache_key = _get_cache_key(bounds, precise)
        cache_file = get_grid_cache_dir() / f"grid_{cache_key}.parquet"

        if cache_file.exists():
            log(f"Using cached grid from {cache_file}")
            conn.execute(f"CREATE TABLE grid_result AS SELECT * FROM '{cache_file}'")
            grid_count = conn.execute("SELECT COUNT(*) FROM grid_result").fetchone()[0]
            log(f"Loaded {grid_count:,} cached grid cells")

            # Skip to output writing
            if output_file:
                out_path = Path(output_file).resolve()
            else:
                out_path = input_path.parent / f"{input_path.stem}_grid.parquet"

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
                bounds=bounds,
            )

    # Fetch grids that intersect the bounding box from remote source
    log("Fetching grid cells from remote source...")
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

    # Save to cache if caching is enabled
    if use_cache and cache_file:
        log(f"Caching grid results to {cache_file}")
        conn.execute(f"""
            COPY grid_result TO '{cache_file}'
            (FORMAT PARQUET, COMPRESSION 'zstd', COMPRESSION_LEVEL 16)
        """)

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
        bounds=bounds,
    )
