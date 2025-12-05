"""Core API for extracting polygon boundaries as lines."""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import duckdb
from geoparquet_io.core.add_bbox_column import add_bbox_column

from ftw_dataset_tools.api.geo import detect_crs, detect_geometry_column

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass
class BoundaryResult:
    """Result of a single boundary extraction operation."""

    input_path: Path
    output_path: Path
    feature_count: int


@dataclass
class CreateBoundariesResult:
    """Result of boundary extraction operation."""

    files_processed: list[BoundaryResult]
    files_skipped: list[tuple[Path, str]]  # (path, reason)

    @property
    def total_processed(self) -> int:
        """Total number of files processed."""
        return len(self.files_processed)

    @property
    def total_skipped(self) -> int:
        """Total number of files skipped."""
        return len(self.files_skipped)

    @property
    def total_features(self) -> int:
        """Total features across all processed files."""
        return sum(r.feature_count for r in self.files_processed)


def _has_polygon_geometries(
    conn: duckdb.DuckDBPyConnection, file_path: Path, geom_col: str
) -> bool:
    """Check if a parquet file contains polygon geometries."""
    try:
        result = conn.execute(f"""
            SELECT ST_GeometryType("{geom_col}") as geom_type
            FROM '{file_path}'
            WHERE "{geom_col}" IS NOT NULL
            LIMIT 1
        """).fetchone()

        if result:
            geom_type = result[0].upper() if result[0] else ""
            return "POLYGON" in geom_type or "MULTIPOLYGON" in geom_type
    except Exception:
        pass
    return False


def _extract_boundaries(
    conn: duckdb.DuckDBPyConnection,
    input_path: Path,
    output_path: Path,
    geom_col: str,
) -> int:
    """Extract boundaries from polygons and write to output file."""
    # Create boundary lines using ST_Boundary
    conn.execute(f"""
        COPY (
            SELECT * EXCLUDE ("{geom_col}"),
                   ST_Boundary("{geom_col}") AS "{geom_col}"
            FROM '{input_path}'
        ) TO '{output_path}' (FORMAT PARQUET)
    """)

    # Get count
    count_result = conn.execute(f"SELECT COUNT(*) FROM '{output_path}'").fetchone()
    count = count_result[0] if count_result else 0

    # Add bbox column and proper GeoParquet metadata using geoparquet-io
    with Path(os.devnull).open("w") as devnull, contextlib.redirect_stdout(devnull):
        add_bbox_column(
            str(output_path),
            str(output_path),
            compression="ZSTD",
            compression_level=16,
        )

    return count


def create_boundaries(
    input_path: str | Path,
    output_dir: str | Path | None = None,
    output_prefix: str = "boundary_lines_",
    geom_col: str | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> CreateBoundariesResult:
    """
    Convert polygon geometries to boundary lines using ST_Boundary.

    Takes an input file or directory of parquet files and creates new parquet
    files with the polygon boundaries converted to lines.

    Args:
        input_path: Path to a parquet file or directory containing parquet files
        output_dir: Output directory. If None, writes to same directory as input files.
        output_prefix: Prefix for output filenames (default: "boundary_line_")
        geom_col: Geometry column name (auto-detected if None)
        on_progress: Optional callback for progress messages

    Returns:
        CreateBoundariesResult with information about processed and skipped files

    Raises:
        FileNotFoundError: If input path doesn't exist
        ValueError: If no valid parquet files found
    """
    input_path = Path(input_path).resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    def log(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    # Collect parquet files to process
    if input_path.is_file():
        parquet_files = [input_path]
    else:
        parquet_files = list(input_path.glob("**/*.parquet"))
        # Exclude files that already start with the output prefix
        parquet_files = [f for f in parquet_files if not f.name.startswith(output_prefix)]

    if not parquet_files:
        raise ValueError(f"No parquet files found in: {input_path}")

    log(f"Found {len(parquet_files)} parquet file(s) to check")

    # Create DuckDB connection
    conn = duckdb.connect(":memory:")
    conn.execute("INSTALL spatial; LOAD spatial;")

    processed: list[BoundaryResult] = []
    skipped: list[tuple[Path, str]] = []

    for parquet_file in parquet_files:
        log(f"Checking: {parquet_file.name}")

        # Detect geometry column
        file_geom_col = geom_col
        if file_geom_col is None:
            file_geom_col = detect_geometry_column(parquet_file) or "geometry"

        # Check CRS and warn if not lat/long
        crs_info = detect_crs(parquet_file, file_geom_col)
        if crs_info.authority_code and crs_info.authority_code.upper() != "EPSG:4326":
            log(f"  Warning: CRS is {crs_info}, not lat/long. Operations may not work as expected.")

        # Check if file has polygon geometries
        if not _has_polygon_geometries(conn, parquet_file, file_geom_col):
            skipped.append((parquet_file, "no polygon geometries found"))
            log("  Skipping: no polygon geometries")
            continue

        # Determine output path
        out_dir = Path(output_dir).resolve() if output_dir else parquet_file.parent

        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{output_prefix}{parquet_file.name}"

        log("  Converting to boundary lines...")

        try:
            count = _extract_boundaries(conn, parquet_file, out_path, file_geom_col)
            processed.append(
                BoundaryResult(
                    input_path=parquet_file,
                    output_path=out_path,
                    feature_count=count,
                )
            )
            log(f"  Wrote {count:,} features to: {out_path.name}")
        except Exception as e:
            skipped.append((parquet_file, str(e)))
            log(f"  Error: {e}")

    conn.close()

    return CreateBoundariesResult(
        files_processed=processed,
        files_skipped=skipped,
    )
