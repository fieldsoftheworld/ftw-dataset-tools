"""Core API for calculating field coverage statistics on grid cells."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import duckdb

from ftw_dataset_tools.api.geo import (
    CRSMismatchError,
    detect_crs,
    detect_geometry_column,
    ensure_spatial_loaded,
    reproject,
    write_geoparquet,
)

if TYPE_CHECKING:
    from collections.abc import Callable

# Default FTW grid source on Source Cooperative
DEFAULT_FTW_GRID_SOURCE = (
    "s3://us-west-2.opendata.source.coop/ftw/ftw-grid/v0.1/partitioned/by_gzd/gzd=*/*.parquet"
)

# Re-export for convenience
__all__ = ["DEFAULT_FTW_GRID_SOURCE", "CRSMismatchError", "FieldStatsResult", "add_field_stats"]


@dataclass
class FieldStatsResult:
    """Result of field statistics calculation."""

    output_path: Path
    total_cells: int
    cells_with_coverage: int
    average_coverage: float
    max_coverage: float

    @property
    def coverage_percentage(self) -> float:
        """Percentage of cells that have field coverage."""
        if self.total_cells == 0:
            return 0.0
        return 100 * self.cells_with_coverage / self.total_cells


def detect_bbox_column(
    conn: duckdb.DuckDBPyConnection, file_path: str | Path, geom_col: str
) -> str | None:
    """
    Detect bbox column for a parquet file.

    First checks GeoParquet metadata, then falls back to schema inspection.

    Args:
        conn: DuckDB connection with spatial extension loaded
        file_path: Path to the parquet file
        geom_col: Name of the geometry column

    Returns:
        The bbox column name if found, None otherwise.
    """
    file_path = str(file_path)

    # First: Check GeoParquet metadata for covering.bbox
    try:
        result = conn.execute(
            "SELECT value FROM parquet_kv_metadata(?) WHERE key = 'geo'", [file_path]
        ).fetchone()

        if result:
            geo_meta = json.loads(result[0])
            columns = geo_meta.get("columns", {})
            geom_info = columns.get(geom_col, {})
            covering = geom_info.get("covering", {})
            bbox_info = covering.get("bbox", {})

            if bbox_info:
                # GeoParquet 1.1+ uses covering.bbox with column references
                xmin_col = (
                    bbox_info.get("xmin", [None])[0]
                    if isinstance(bbox_info.get("xmin"), list)
                    else None
                )
                if xmin_col:
                    # Extract the parent column name (e.g., "bbox" from ["bbox", "xmin"])
                    return xmin_col
    except Exception:
        pass

    # Fallback: Check schema for bbox-like STRUCT columns
    try:
        schema = conn.execute(f"DESCRIBE SELECT * FROM '{file_path}'").fetchall()

        # Look for common bbox column names
        bbox_candidates = ["bbox", f"{geom_col}_bbox", "geometry_bbox"]

        for row in schema:
            col_name = row[0]
            col_type = row[1]

            if (
                col_name in bbox_candidates
                and "STRUCT" in col_type.upper()
                and all(coord in col_type.lower() for coord in ["xmin", "ymin", "xmax", "ymax"])
            ):
                return col_name
    except Exception:
        pass

    return None


def _build_coverage_query(
    grid_geom_col: str,
    fields_geom_col: str,
    grid_bbox_col: str | None,
    fields_bbox_col: str | None,
    coverage_col: str,
) -> str:
    """Build the coverage calculation SQL query.

    Uses ST_MakeValid to repair any invalid geometries before spatial operations,
    preventing TopologyException errors from self-intersecting or otherwise
    invalid input geometries.
    """
    # Use ST_MakeValid to handle invalid geometries
    valid_grid_geom = f'ST_MakeValid(g."{grid_geom_col}")'
    valid_fields_geom = f'ST_MakeValid(f."{fields_geom_col}")'

    # Build JOIN condition
    if grid_bbox_col and fields_bbox_col:
        join_condition = f"""
            g."{grid_bbox_col}".xmin <= f."{fields_bbox_col}".xmax
            AND g."{grid_bbox_col}".xmax >= f."{fields_bbox_col}".xmin
            AND g."{grid_bbox_col}".ymin <= f."{fields_bbox_col}".ymax
            AND g."{grid_bbox_col}".ymax >= f."{fields_bbox_col}".ymin
            AND ST_Intersects({valid_grid_geom}, {valid_fields_geom})
        """
    else:
        join_condition = f"ST_Intersects({valid_grid_geom}, {valid_fields_geom})"

    return f"""
    WITH intersections AS (
        SELECT
            g.rowid as grid_rowid,
            ST_Intersection({valid_grid_geom}, {valid_fields_geom}) as intersect_geom
        FROM grid_table g
        JOIN fields_table f ON {join_condition}
    ),
    coverage AS (
        SELECT
            grid_rowid,
            ST_Union_Agg(intersect_geom) as total_coverage
        FROM intersections
        GROUP BY grid_rowid
    )
    SELECT
        g.*,
        COALESCE(
            ROUND(100.0 * ST_Area(c.total_coverage) / ST_Area({valid_grid_geom}), 2),
            0.0
        ) as "{coverage_col}"
    FROM grid_table g
    LEFT JOIN coverage c ON g.rowid = c.grid_rowid
    """


def add_field_stats(
    fields_file: str | Path,
    grid_file: str | Path | None = None,
    output_file: str | Path | None = None,
    grid_geom_col: str | None = None,
    fields_geom_col: str | None = None,
    grid_bbox_col: str | None = None,
    fields_bbox_col: str | None = None,
    coverage_col: str = "field_coverage_pct",
    min_coverage: float | None = None,
    reproject_to_4326: bool = False,
    drop_border_chips: bool = False,
    grid_source: str = DEFAULT_FTW_GRID_SOURCE,
    on_progress: Callable[[str], None] | None = None,
) -> FieldStatsResult:
    """
    Calculate field coverage percentage for each grid cell.

    This function computes what percentage of each grid cell is covered by
    field boundary polygons using DuckDB's spatial extension.

    If no grid file is provided, fetches grid cells from the FTW grid on
    Source Cooperative, filtered by the bounds of the fields file.

    Args:
        fields_file: Path to parquet file containing field boundary polygons
        grid_file: Path to parquet file containing grid geometries (e.g., MGRS cells).
            If None, fetches from grid_source filtered by fields file bounds.
        output_file: Output file path. If None, creates chips_<fields_basename>.parquet.
        grid_geom_col: Column name for grid geometry (auto-detected from GeoParquet
            metadata if None, falls back to "geometry")
        fields_geom_col: Column name for fields geometry (auto-detected from GeoParquet
            metadata if None, falls back to "geometry")
        grid_bbox_col: Column name for grid bbox (auto-detected if None)
        fields_bbox_col: Column name for fields bbox (auto-detected if None)
        coverage_col: Name for the new coverage column (default: "field_coverage_pct")
        min_coverage: If set, exclude grid cells with coverage below this percentage
            (e.g., 0.01 to exclude cells with 0% coverage)
        reproject_to_4326: If True, reproject both inputs to EPSG:4326 before processing
        drop_border_chips: If True, remove chips along the outer border of the dataset
            (where fields may have partial coverage)
        grid_source: URL/path to fetch grid from when grid_file is None
            (default: FTW grid on Source Coop)
        on_progress: Optional callback for progress messages

    Returns:
        FieldStatsResult with statistics about the calculation

    Raises:
        FileNotFoundError: If input files don't exist
        CRSMismatchError: If input files have different CRS and reproject_to_4326 is False
        duckdb.Error: If there are issues with the spatial queries
    """
    fields_path = Path(fields_file).resolve()

    if not fields_path.exists():
        raise FileNotFoundError(f"Fields file not found: {fields_path}")

    grid_path: Path | None = None
    if grid_file is not None:
        grid_path = Path(grid_file).resolve()
        if not grid_path.exists():
            raise FileNotFoundError(f"Grid file not found: {grid_path}")

    def log(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    # Auto-detect fields geometry column from GeoParquet metadata
    if fields_geom_col is None:
        fields_geom_col = detect_geometry_column(fields_path) or "geometry"
        log(f"Detected fields geometry column: {fields_geom_col}")

    # Track temp files for cleanup
    temp_files: list[Path] = []

    try:
        # Create DuckDB connection and load spatial extension
        conn = duckdb.connect(":memory:")
        ensure_spatial_loaded(conn)

        # Load fields table first (needed for bounds calculation if fetching grid from S3)
        log("Loading fields data...")
        conn.execute(f"CREATE TABLE fields_table AS SELECT * FROM '{fields_path}'")
        fields_count = conn.execute("SELECT COUNT(*) FROM fields_table").fetchone()[0]
        log(f"Loaded {fields_count:,} field polygons")

        # Handle grid loading - either from local file or S3
        if grid_path is not None:
            # Local grid file provided
            if grid_geom_col is None:
                grid_geom_col = detect_geometry_column(grid_path) or "geometry"
                log(f"Detected grid geometry column: {grid_geom_col}")

            # Check CRS compatibility
            grid_crs = detect_crs(grid_path, grid_geom_col)
            fields_crs = detect_crs(fields_path, fields_geom_col)

            log(f"Grid CRS: {grid_crs}")
            log(f"Fields CRS: {fields_crs}")

            if not grid_crs.is_equivalent_to(fields_crs):
                if reproject_to_4326:
                    log("CRS mismatch detected, reprojecting to EPSG:4326...")

                    # Reproject grid if needed
                    if grid_crs.authority_code != "EPSG:4326":
                        grid_temp = Path(tempfile.mktemp(suffix=".parquet"))
                        temp_files.append(grid_temp)
                        reproject(grid_path, grid_temp, "EPSG:4326", on_progress)
                        grid_path = grid_temp
                        log(f"Reprojected grid to: {grid_temp}")

                    # Reproject fields if needed - need to reload fields table
                    if fields_crs.authority_code != "EPSG:4326":
                        fields_temp = Path(tempfile.mktemp(suffix=".parquet"))
                        temp_files.append(fields_temp)
                        reproject(fields_path, fields_temp, "EPSG:4326", on_progress)
                        fields_path = fields_temp
                        log(f"Reprojected fields to: {fields_temp}")
                        # Reload fields table with reprojected data
                        conn.execute("DROP TABLE fields_table")
                        conn.execute(f"CREATE TABLE fields_table AS SELECT * FROM '{fields_path}'")
                else:
                    raise CRSMismatchError(
                        crs1=str(grid_crs),
                        crs2=str(fields_crs),
                        file1=str(grid_file),
                        file2=str(fields_file),
                    )

            log("Loading grid data...")
            conn.execute(f"CREATE TABLE grid_table AS SELECT * FROM '{grid_path}'")
        else:
            # Fetch grid from S3 based on fields bounds
            # First check that fields file is in EPSG:4326 (required for S3 grid)
            fields_crs = detect_crs(fields_path, fields_geom_col)
            if (
                fields_crs.authority_code is None
                or fields_crs.authority_code.upper() != "EPSG:4326"
            ):
                raise ValueError(
                    f"Fields file must be in EPSG:4326 when using remote grid, "
                    f"but has CRS '{fields_crs}'.\n"
                    f"Please reproject first with:\n"
                    f"  ftwd reproject {fields_file} --target-crs EPSG:4326"
                )

            log("Fetching grid from Source Coop...")

            # Load httpfs for S3 access
            conn.execute("INSTALL httpfs; LOAD httpfs;")
            conn.execute("SET s3_region = 'us-west-2';")

            # Compute bounds from fields geometry
            bounds_result = conn.execute(f"""
                SELECT
                    MIN(ST_XMin("{fields_geom_col}")) as xmin,
                    MIN(ST_YMin("{fields_geom_col}")) as ymin,
                    MAX(ST_XMax("{fields_geom_col}")) as xmax,
                    MAX(ST_YMax("{fields_geom_col}")) as ymax
                FROM fields_table
            """).fetchone()
            xmin, ymin, xmax, ymax = bounds_result
            log(f"Fields bounds: [{xmin:.6f}, {ymin:.6f}, {xmax:.6f}, {ymax:.6f}]")
            log(
                f"BBox Finder URL: https://bboxfinder.com/#{ymin:.6f},{xmin:.6f},{ymax:.6f},{xmax:.6f}"
            )

            # Fetch grid cells that intersect the bounding box
            log("Fetching grid cells by bounding box...")
            conn.execute(f"""
                CREATE TABLE grid_table AS
                SELECT *
                FROM '{grid_source}'
                WHERE bbox.xmin <= {xmax}
                  AND bbox.xmax >= {xmin}
                  AND bbox.ymin <= {ymax}
                  AND bbox.ymax >= {ymin}
            """)

            # Auto-detect grid geometry column from the fetched data
            if grid_geom_col is None:
                grid_geom_col = "geometry"
                log(f"Using grid geometry column: {grid_geom_col}")

        # Get grid count
        grid_count = conn.execute("SELECT COUNT(*) FROM grid_table").fetchone()[0]
        log(f"Loaded {grid_count:,} grid cells")

        # Filter out border chips if requested
        if drop_border_chips:
            log("Identifying border chips to remove...")

            # Strategy: Remove chips that are not completely within the convex hull of field polygons
            # This identifies chips on the boundary that may have partial field coverage

            # Create convex hull of all field geometries
            conn.execute(f"""
                CREATE TABLE fields_hull AS
                SELECT ST_ConvexHull(ST_Union_Agg({fields_geom_col})) as hull
                FROM fields_table
            """)

            # Identify chips that are NOT completely within the convex hull
            # ST_Within returns true only if the chip geometry is completely inside the hull
            conn.execute(f"""
                CREATE TABLE border_chips AS
                SELECT g.rowid
                FROM grid_table g, fields_hull h
                WHERE NOT ST_Within(g.{grid_geom_col}, h.hull)
            """)

            border_count = conn.execute("SELECT COUNT(*) FROM border_chips").fetchone()[0]

            if border_count > 0:
                # Remove border chips from grid_table
                conn.execute("""
                    DELETE FROM grid_table
                    WHERE rowid IN (SELECT rowid FROM border_chips)
                """)

                remaining_count = conn.execute("SELECT COUNT(*) FROM grid_table").fetchone()[0]
                log(
                    f"Removed {border_count:,} border chips (outside convex hull), {remaining_count:,} chips remaining"
                )
                grid_count = remaining_count
            else:
                log("No border chips found to remove")

            # Clean up temporary tables
            conn.execute("DROP TABLE fields_hull")

        # Auto-detect bbox columns if not specified
        detected_grid_bbox = grid_bbox_col
        detected_fields_bbox = fields_bbox_col

        if detected_grid_bbox is None:
            if grid_path is not None:
                detected_grid_bbox = detect_bbox_column(conn, str(grid_path), grid_geom_col)
            else:
                # For S3 source, we know the bbox column is "bbox"
                detected_grid_bbox = "bbox"
            if detected_grid_bbox:
                log(f"Detected grid bbox column: {detected_grid_bbox}")
            else:
                log("Warning: grid has no bbox column, spatial queries may be slower")

        if detected_fields_bbox is None:
            detected_fields_bbox = detect_bbox_column(conn, str(fields_path), fields_geom_col)
            if detected_fields_bbox:
                log(f"Detected fields bbox column: {detected_fields_bbox}")
            else:
                log(
                    f"Warning: {fields_path.name} has no bbox column, spatial queries may be slower"
                )

        # Report optimization status
        if detected_grid_bbox and detected_fields_bbox:
            log("Using bbox optimization for spatial joins")
        else:
            log("Bbox optimization disabled (missing bbox columns)")

        # Build and execute coverage query
        log("Calculating coverage...")
        query = _build_coverage_query(
            grid_geom_col=grid_geom_col,
            fields_geom_col=fields_geom_col,
            grid_bbox_col=detected_grid_bbox,
            fields_bbox_col=detected_fields_bbox,
            coverage_col=coverage_col,
        )

        conn.execute(f"CREATE TABLE result AS {query}")

        # Filter by min_coverage if specified
        if min_coverage is not None:
            before_count = conn.execute("SELECT COUNT(*) FROM result").fetchone()[0]
            conn.execute(f"""
                DELETE FROM result WHERE "{coverage_col}" < {min_coverage}
            """)
            after_count = conn.execute("SELECT COUNT(*) FROM result").fetchone()[0]
            removed = before_count - after_count
            log(f"Filtered out {removed:,} cells with coverage < {min_coverage}%")

        # Calculate summary statistics
        stats = conn.execute(f"""
            SELECT
                COUNT(*) as total,
                COUNT(*) FILTER (WHERE "{coverage_col}" > 0) as with_coverage,
                ROUND(AVG("{coverage_col}"), 2) as avg_coverage,
                ROUND(MAX("{coverage_col}"), 2) as max_coverage
            FROM result
        """).fetchone()

        total_grids, grids_with_coverage, avg_coverage, max_coverage = stats

        # Determine output path
        # Default: chips_<fields_basename>.parquet in same directory as fields file
        if output_file:
            out_path = Path(output_file).resolve()
        else:
            out_path = fields_path.parent / f"chips_{fields_path.stem}.parquet"

        # Write output with proper GeoParquet metadata
        log(f"Writing output to: {out_path}")
        write_geoparquet(out_path, conn=conn, query="SELECT * FROM result")

        conn.close()

        return FieldStatsResult(
            output_path=out_path,
            total_cells=total_grids,
            cells_with_coverage=grids_with_coverage,
            average_coverage=avg_coverage or 0.0,
            max_coverage=max_coverage or 0.0,
        )
    finally:
        # Clean up temp files from reprojection
        for temp_file in temp_files:
            if temp_file.exists():
                temp_file.unlink()
