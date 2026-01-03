"""API for creating hierarchical FTW grids from 1km MGRS data."""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import duckdb
from geoparquet_io.core.add_bbox_column import add_bbox_column
from geoparquet_io.core.common import write_parquet_with_metadata

from ftw_dataset_tools.api.geo import get_bbox_column_name, has_bbox_column

if TYPE_CHECKING:
    from collections.abc import Callable

# Valid km sizes that divide 100 evenly
VALID_KM_SIZES = {1, 2, 4, 5, 10, 20, 25, 50, 100}

# Required columns (case-insensitive check)
REQUIRED_COLUMNS = {"gzd", "mgrs"}

__all__ = [
    "CreateFTWGridResult",
    "InvalidKmSizeError",
    "MultipleGZDError",
    "create_ftw_grid",
]


class InvalidKmSizeError(Exception):
    """Raised when km_size doesn't divide 100 evenly."""

    def __init__(self, km_size: int) -> None:
        self.km_size = km_size
        valid_sizes = ", ".join(str(s) for s in sorted(VALID_KM_SIZES))
        super().__init__(
            f"km_size must divide 100 evenly. Got {km_size}, valid values are: {valid_sizes}"
        )


class MultipleGZDError(Exception):
    """Raised when input file contains more than one GZD."""

    def __init__(self, gzd_count: int, gzds: list[str]) -> None:
        self.gzd_count = gzd_count
        self.gzds = gzds
        gzd_list = ", ".join(gzds[:10])
        if gzd_count > 10:
            gzd_list += f", ... ({gzd_count} total)"
        super().__init__(
            f"Input file must contain exactly one GZD, but found {gzd_count}: {gzd_list}"
        )


@dataclass
class CreateFTWGridResult:
    """Result of FTW grid creation."""

    output_path: Path
    gzd: str | None  # None for partitioned output
    km_size: int
    total_cells: int
    gzd_count: int
    skipped_files: list[str] | None = None  # Files skipped due to missing columns


def _normalize_columns(conn: duckdb.DuckDBPyConnection) -> dict[str, str]:
    """
    Get column name mapping for case-insensitive access.

    Returns a dict mapping lowercase names to actual column names.
    """
    columns = conn.execute("DESCRIBE mgrs_1km").fetchall()
    return {col[0].lower(): col[0] for col in columns}


def _check_required_columns(col_map: dict[str, str]) -> list[str]:
    """Check for required columns, return list of missing ones."""
    missing = []
    for required in REQUIRED_COLUMNS:
        if required not in col_map:
            missing.append(required.upper())
    return missing


def _get_column(col_map: dict[str, str], name: str) -> str:
    """Get actual column name from case-insensitive name."""
    return col_map.get(name.lower(), name)


def create_ftw_grid(
    input_path: str | Path,
    output_path: str | Path | None = None,
    km_size: int = 2,
    on_progress: Callable[[str], None] | None = None,
    on_file_progress: Callable[[int, int, str], None] | None = None,
) -> CreateFTWGridResult:
    """
    Create a hierarchical FTW grid from 1km MGRS GeoParquet file(s).

    Groups 1km MGRS cells into larger grid cells and unions their geometries.
    Never crosses GZD or kmSQ_ID boundaries when aggregating.

    Args:
        input_path: Path to either:
            - A single GeoParquet file (must contain exactly one GZD)
            - A folder containing partitioned parquet files
        output_path: Output path.
            - For file input: If None, creates ftw_<gzd>.parquet
            - For folder input: Required, must be a folder for hive-partitioned output
        km_size: Grid cell size in km (must divide 100 evenly).
            Valid values: 1, 2, 4, 5, 10, 20, 25, 50, 100
        on_progress: Optional callback for progress messages
        on_file_progress: Optional callback for file processing progress (current, total, file_name)

    Returns:
        CreateFTWGridResult with statistics about the operation

    Raises:
        FileNotFoundError: If input path doesn't exist
        InvalidKmSizeError: If km_size doesn't divide 100 evenly
        MultipleGZDError: If single file input contains more than one GZD
        ValueError: If folder input but no output_path specified
    """
    input_p = Path(input_path).resolve()

    if not input_p.exists():
        raise FileNotFoundError(f"Input path not found: {input_p}")

    if km_size not in VALID_KM_SIZES:
        raise InvalidKmSizeError(km_size)

    def log(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    # Determine if input is file or folder
    is_folder = input_p.is_dir()

    if is_folder:
        if output_path is None:
            raise ValueError("Output path is required when input is a folder")
        return _create_ftw_grid_partitioned(
            input_p, Path(output_path), km_size, log, on_file_progress
        )
    else:
        return _create_ftw_grid_single(input_p, output_path, km_size, log)


def _create_ftw_grid_single(
    input_path: Path,
    output_file: str | Path | None,
    km_size: int,
    log: Callable[[str], None],
) -> CreateFTWGridResult:
    """Create FTW grid from a single file with exactly one GZD."""
    log(f"Creating {km_size}x{km_size}km FTW grid from: {input_path.name}")

    # Create DuckDB connection with spatial extension
    conn = duckdb.connect(":memory:")
    conn.execute("INSTALL spatial; LOAD spatial;")

    # Load input file
    log("Loading 1km MGRS data...")
    conn.execute(f"CREATE TABLE mgrs_1km AS SELECT * FROM '{input_path}'")

    input_count = conn.execute("SELECT COUNT(*) FROM mgrs_1km").fetchone()[0]
    log(f"Loaded {input_count:,} 1km cells")

    # Get column mapping for case-insensitive access
    col_map = _normalize_columns(conn)

    # Check for required columns
    missing = _check_required_columns(col_map)
    if missing:
        conn.close()
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    # Get actual column names
    gzd_col = _get_column(col_map, "GZD")
    mgrs_col = _get_column(col_map, "MGRS")

    # Check for optional columns
    kmsq_col = _get_column(col_map, "kmSQ_ID") if "kmsq_id" in col_map else None
    mgrs_10km_col = _get_column(col_map, "MGRS_10km") if "mgrs_10km" in col_map else None
    easting_col = _get_column(col_map, "EASTING") if "easting" in col_map else None
    northing_col = _get_column(col_map, "NORTHING") if "northing" in col_map else None

    # Check for bbox column from geoparquet metadata
    bbox_col = get_bbox_column_name(input_path)

    # Check that input has exactly one GZD
    gzd_result = conn.execute(
        f'SELECT DISTINCT "{gzd_col}" FROM mgrs_1km ORDER BY "{gzd_col}"'
    ).fetchall()
    gzds = [row[0] for row in gzd_result]

    if len(gzds) != 1:
        conn.close()
        raise MultipleGZDError(len(gzds), gzds)

    gzd = gzds[0]
    log(f"GZD: {gzd}")

    # Build the query based on available columns
    result = _build_and_execute_query(
        conn,
        km_size,
        gzd_col,
        mgrs_col,
        kmsq_col,
        mgrs_10km_col,
        easting_col,
        northing_col,
        bbox_col,
        log,
    )

    total_cells, _kmsq_count = result

    # Determine output path
    if output_file:
        out_path = Path(output_file).resolve()
    else:
        out_path = input_path.parent / f"ftw_{gzd}.parquet"

    # Write output
    _write_output(conn, out_path, input_path, log)
    conn.close()

    return CreateFTWGridResult(
        output_path=out_path,
        gzd=gzd,
        km_size=km_size,
        total_cells=total_cells,
        gzd_count=1,
    )


def _create_ftw_grid_partitioned(
    input_folder: Path,
    output_folder: Path,
    km_size: int,
    log: Callable[[str], None],
    on_file_progress: Callable[[int, int, str], None] | None = None,
) -> CreateFTWGridResult:
    """Create FTW grid from partitioned parquet files, output as hive partitions."""
    log(f"Creating {km_size}x{km_size}km FTW grid from folder: {input_folder}")

    # Find all input parquet files
    input_files = sorted(input_folder.glob("**/*.parquet"))
    log(f"Found {len(input_files)} parquet files to process")

    if not input_files:
        raise ValueError(f"No parquet files found in {input_folder}")

    # Create output folder
    output_folder.mkdir(parents=True, exist_ok=True)

    total_cells = 0
    gzds_processed = []
    skipped_files = []

    # Process each file individually
    total_files = len(input_files)
    for i, input_file in enumerate(input_files):
        file_name = input_file.name

        try:
            # Process single file (suppress logging noise)
            def file_log(msg: str) -> None:
                pass  # Suppress individual file logs

            result = _create_ftw_grid_single_internal(input_file, km_size, file_log)

            if result is None:
                skipped_files.append(str(input_file))
                # Report progress even for skipped files
                if on_file_progress:
                    on_file_progress(i + 1, total_files, file_name)
                continue

            gzd, cells, conn = result

            # Write to hive partition folder
            gzd_folder = output_folder / f"gzd={gzd}"
            gzd_folder.mkdir(parents=True, exist_ok=True)
            out_file = gzd_folder / f"ftw_{gzd}.parquet"

            # Write output
            write_parquet_with_metadata(
                conn,
                "SELECT * FROM ftw_grid",
                str(out_file),
                compression="ZSTD",
                compression_level=16,
            )
            conn.close()

            # Add bbox if input doesn't have one (suppress stdout from library)
            if not has_bbox_column(input_file):
                with Path(os.devnull).open("w") as devnull, contextlib.redirect_stdout(devnull):
                    add_bbox_column(
                        str(out_file),
                        str(out_file),
                        compression="ZSTD",
                        compression_level=16,
                    )

            total_cells += cells
            gzds_processed.append(gzd)

            # Report progress after successful processing
            if on_file_progress:
                on_file_progress(i + 1, total_files, file_name)

        except (ValueError, MultipleGZDError) as e:
            log(f"Warning: Skipping {file_name}: {e}")
            skipped_files.append(str(input_file))
            # Report progress even for error files
            if on_file_progress:
                on_file_progress(i + 1, total_files, file_name)
            continue

    if not gzds_processed:
        raise ValueError("No files were successfully processed")

    log(f"Processed {len(gzds_processed)} GZDs, skipped {len(skipped_files)} files")

    return CreateFTWGridResult(
        output_path=output_folder,
        gzd=None,  # Multiple GZDs
        km_size=km_size,
        total_cells=total_cells,
        gzd_count=len(gzds_processed),
        skipped_files=skipped_files if skipped_files else None,
    )


def _create_ftw_grid_single_internal(
    input_path: Path,
    km_size: int,
    log: Callable[[str], None],
) -> tuple[str, int, duckdb.DuckDBPyConnection] | None:
    """
    Process a single file and return (gzd, cell_count, conn) without writing.

    Returns None if file should be skipped (missing required columns).
    The caller is responsible for closing the connection.
    """
    # Create DuckDB connection with spatial extension
    conn = duckdb.connect(":memory:")
    conn.execute("INSTALL spatial; LOAD spatial;")

    # Load input file
    conn.execute(f"CREATE TABLE mgrs_1km AS SELECT * FROM '{input_path}'")

    # Get column mapping for case-insensitive access
    col_map = _normalize_columns(conn)

    # Check for required columns
    missing = _check_required_columns(col_map)
    if missing:
        conn.close()
        return None

    # Get actual column names
    gzd_col = _get_column(col_map, "GZD")
    mgrs_col = _get_column(col_map, "MGRS")

    # Check for optional columns
    kmsq_col = _get_column(col_map, "kmSQ_ID") if "kmsq_id" in col_map else None
    mgrs_10km_col = _get_column(col_map, "MGRS_10km") if "mgrs_10km" in col_map else None
    easting_col = _get_column(col_map, "EASTING") if "easting" in col_map else None
    northing_col = _get_column(col_map, "NORTHING") if "northing" in col_map else None

    # Check for bbox column from geoparquet metadata
    bbox_col = get_bbox_column_name(input_path)

    # Check that input has exactly one GZD
    gzd_result = conn.execute(
        f'SELECT DISTINCT "{gzd_col}" FROM mgrs_1km ORDER BY "{gzd_col}"'
    ).fetchall()
    gzds = [row[0] for row in gzd_result]

    if len(gzds) != 1:
        conn.close()
        raise MultipleGZDError(len(gzds), gzds)

    gzd = gzds[0]
    log(f"GZD: {gzd}")

    # Build and execute query
    total_cells, _ = _build_and_execute_query(
        conn,
        km_size,
        gzd_col,
        mgrs_col,
        kmsq_col,
        mgrs_10km_col,
        easting_col,
        northing_col,
        bbox_col,
        log,
    )

    return gzd, total_cells, conn


def _build_and_execute_query(
    conn: duckdb.DuckDBPyConnection,
    km_size: int,
    gzd_col: str,
    mgrs_col: str,
    kmsq_col: str | None,
    mgrs_10km_col: str | None,
    easting_col: str | None,
    northing_col: str | None,
    bbox_col: str | None,
    log: Callable[[str], None],
) -> tuple[int, int]:
    """Build and execute the aggregation query. Returns (total_cells, kmsq_count)."""

    # Determine how to get kmsq_id - from column or parse from MGRS
    if kmsq_col:
        kmsq_expr = f'"{kmsq_col}"'
    else:
        # Parse from MGRS code - characters 3-4 (0-indexed) after the GZD
        # MGRS format: GZD (2-3 chars) + 100km square (2 chars) + coordinates
        kmsq_expr = f'SUBSTRING("{mgrs_col}", LENGTH("{gzd_col}") + 1, 2)'

    # Determine how to get MGRS_10km - from column or derive
    if mgrs_10km_col:
        mgrs_10km_expr = f'"{mgrs_10km_col}"'
    else:
        # Take first 7 characters of MGRS for 10km precision (GZD + 100km + 2 digits)
        mgrs_10km_expr = f'SUBSTRING("{mgrs_col}", 1, LENGTH("{gzd_col}") + 4)'

    # Determine easting/northing expressions
    if easting_col and northing_col:
        # Check format of EASTING
        sample_easting = conn.execute(f'SELECT "{easting_col}" FROM mgrs_1km LIMIT 1').fetchone()[0]
        log(f"Sample EASTING value: {sample_easting}")

        if "m" in str(sample_easting).lower():
            easting_expr = f"(CAST(REGEXP_REPLACE(\"{easting_col}\", '[^0-9]', '', 'g') AS BIGINT) // 1000) % 100"
            northing_expr = f"(CAST(REGEXP_REPLACE(\"{northing_col}\", '[^0-9]', '', 'g') AS BIGINT) // 1000) % 100"
            log("Detected full coordinate format (e.g., '706000mE'), extracting 1km indices")
        else:
            easting_expr = f'CAST("{easting_col}" AS INT)'
            northing_expr = f'CAST("{northing_col}" AS INT)'
            log("Detected simple index format")
    else:
        # Parse from MGRS code - last 4 digits are easting (2) + northing (2) for 1km
        easting_expr = f'CAST(SUBSTRING("{mgrs_col}", LENGTH("{gzd_col}") + 3, 2) AS INT)'
        northing_expr = f'CAST(SUBSTRING("{mgrs_col}", LENGTH("{gzd_col}") + 5, 2) AS INT)'
        log("Parsing easting/northing from MGRS code")

    # Build bbox expressions if input has bbox
    bbox_parsed = ""
    bbox_with_parents = ""
    bbox_select = ""
    if bbox_col:
        bbox_parsed = f',\n            "{bbox_col}"'
        bbox_with_parents = f',\n            "{bbox_col}"'
        bbox_select = f""",
        STRUCT_PACK(
            xmin := MIN("{bbox_col}".xmin),
            ymin := MIN("{bbox_col}".ymin),
            xmax := MAX("{bbox_col}".xmax),
            ymax := MAX("{bbox_col}".ymax)
        ) AS bbox"""

    # Build the aggregation query
    log(f"Aggregating into {km_size}x{km_size}km cells (this may take a while)...")

    query = f"""
    WITH parsed AS (
        SELECT
            "{gzd_col}" AS GZD,
            {kmsq_expr} AS kmSQ_ID,
            {mgrs_10km_expr} AS MGRS_10km,
            CAST({easting_expr} AS INT) AS e,
            CAST({northing_expr} AS INT) AS n,
            geometry{bbox_parsed}
        FROM mgrs_1km
    ),
    with_parents AS (
        SELECT
            GZD,
            kmSQ_ID,
            MGRS_10km,
            e,
            n,
            (e // {km_size}) * {km_size} AS parent_e,
            (n // {km_size}) * {km_size} AS parent_n,
            geometry{bbox_with_parents}
        FROM parsed
    )
    SELECT
        GZD AS gzd,
        ANY_VALUE(MGRS_10km) AS mgrs_10km,
        CONCAT(
            'ftw-', GZD, kmSQ_ID,
            LPAD(CAST(parent_e AS VARCHAR), 2, '0'),
            LPAD(CAST(parent_n AS VARCHAR), 2, '0')
        ) AS id,
        ST_Union_Agg(geometry) AS geometry{bbox_select}
    FROM with_parents
    GROUP BY
        GZD, kmSQ_ID, parent_e, parent_n
    ORDER BY
        GZD, kmSQ_ID, parent_e, parent_n
    """

    conn.execute(f"CREATE TABLE ftw_grid AS {query}")
    log("Aggregation complete")

    # Get statistics
    stats = conn.execute("""
        SELECT
            COUNT(*) as total_cells,
            COUNT(DISTINCT mgrs_10km) as kmsq_count
        FROM ftw_grid
    """).fetchone()

    total_cells, kmsq_count = stats
    log(f"Created {total_cells:,} {km_size}x{km_size}km cells across {kmsq_count} 10km squares")

    return total_cells, kmsq_count


def _write_output(
    conn: duckdb.DuckDBPyConnection,
    out_path: Path,
    input_path: Path,
    log: Callable[[str], None],
) -> None:
    """Write output as GeoParquet with bbox."""
    log(f"Writing output to: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if input has bbox - if so, we'll include it in output
    input_has_bbox = has_bbox_column(input_path)

    # Write initial file with proper metadata
    write_parquet_with_metadata(
        conn,
        "SELECT * FROM ftw_grid",
        str(out_path),
        compression="ZSTD",
        compression_level=16,
    )

    # Add bbox column if input didn't have one
    if not input_has_bbox:
        log("Adding bbox column...")
        with Path(os.devnull).open("w") as devnull, contextlib.redirect_stdout(devnull):
            add_bbox_column(
                str(out_path),
                str(out_path),
                compression="ZSTD",
                compression_level=16,
            )
