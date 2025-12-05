"""Geospatial utilities for CRS detection and reprojection."""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import duckdb
import geopandas as gpd
from geoparquet_io.core.add_bbox_column import add_bbox_column

if TYPE_CHECKING:
    from collections.abc import Callable


def detect_geometry_column(
    file_path: str | Path,
    conn: duckdb.DuckDBPyConnection | None = None,
) -> str | None:
    """
    Detect the primary geometry column from GeoParquet metadata.

    Args:
        file_path: Path to the GeoParquet file
        conn: Optional existing DuckDB connection

    Returns:
        The primary geometry column name, or None if not found
    """
    file_path = str(Path(file_path).resolve())
    close_conn = False

    if conn is None:
        conn = duckdb.connect(":memory:")
        close_conn = True

    try:
        result = conn.execute(
            "SELECT value FROM parquet_kv_metadata(?) WHERE key = 'geo'", [file_path]
        ).fetchone()

        if result:
            geo_meta = json.loads(result[0])
            return geo_meta.get("primary_column")
    except Exception:
        pass
    finally:
        if close_conn:
            conn.close()

    return None


def has_bbox_column(
    file_path: str | Path,
    conn: duckdb.DuckDBPyConnection | None = None,
) -> bool:
    """
    Check if file has a bbox column defined in GeoParquet metadata.

    Checks for bbox in the geometry column's covering metadata.

    Args:
        file_path: Path to the GeoParquet file
        conn: Optional existing DuckDB connection

    Returns:
        True if bbox column exists in metadata, False otherwise
    """
    return get_bbox_column_name(file_path, conn) is not None


def get_bbox_column_name(
    file_path: str | Path,
    conn: duckdb.DuckDBPyConnection | None = None,
) -> str | None:
    """
    Get the bbox column name from GeoParquet metadata.

    Args:
        file_path: Path to the GeoParquet file
        conn: Optional existing DuckDB connection

    Returns:
        The bbox column name, or None if not found
    """
    file_path = str(Path(file_path).resolve())
    close_conn = False

    if conn is None:
        conn = duckdb.connect(":memory:")
        close_conn = True

    try:
        result = conn.execute(
            "SELECT value FROM parquet_kv_metadata(?) WHERE key = 'geo'", [file_path]
        ).fetchone()

        if result:
            geo_meta = json.loads(result[0])
            primary_col = geo_meta.get("primary_column")
            if primary_col:
                columns = geo_meta.get("columns", {})
                col_meta = columns.get(primary_col, {})
                # Get bbox column name from covering.bbox.xmin[0]
                covering = col_meta.get("covering", {})
                bbox_info = covering.get("bbox", {})
                if bbox_info:
                    # The column name is the first element of any bbox field
                    xmin_info = bbox_info.get("xmin", [])
                    if xmin_info and len(xmin_info) > 0:
                        return xmin_info[0]
        return None
    except Exception:
        return None
    finally:
        if close_conn:
            conn.close()


class CRSMismatchError(Exception):
    """Raised when two datasets have different coordinate reference systems."""

    def __init__(self, crs1: str | None, crs2: str | None, file1: str, file2: str) -> None:
        self.crs1 = crs1
        self.crs2 = crs2
        self.file1 = file1
        self.file2 = file2
        super().__init__(
            f"CRS mismatch: {file1} has CRS '{crs1 or 'unknown'}' but "
            f"{file2} has CRS '{crs2 or 'unknown'}'. "
            "Use --reproject to reproject both to EPSG:4326."
        )


@dataclass
class CRSInfo:
    """Information about a dataset's coordinate reference system."""

    authority: str | None  # e.g., "EPSG"
    code: str | None  # e.g., "4326"
    wkt: str | None  # Full WKT representation
    projjson: dict | None  # PROJJSON representation

    @property
    def authority_code(self) -> str | None:
        """Return authority:code string if both are available."""
        if self.authority and self.code:
            return f"{self.authority}:{self.code}"
        return None

    def is_equivalent_to(self, other: CRSInfo) -> bool:
        """Check if this CRS is equivalent to another."""
        # If both have authority codes, compare those
        if self.authority_code and other.authority_code:
            return self.authority_code.upper() == other.authority_code.upper()

        # If both have WKT, compare (simplified - just check if identical)
        if self.wkt and other.wkt:
            return self.wkt == other.wkt

        # If we can't compare, assume they're different
        return False

    def __str__(self) -> str:
        if self.authority_code:
            return self.authority_code
        if self.wkt:
            return f"WKT({self.wkt[:50]}...)"
        return "unknown"


def detect_crs(
    file_path: str | Path,
    geom_col: str = "geometry",
    conn: duckdb.DuckDBPyConnection | None = None,
) -> CRSInfo:
    """
    Detect CRS from a GeoParquet file.

    Args:
        file_path: Path to the GeoParquet file
        geom_col: Name of the geometry column
        conn: Optional existing DuckDB connection

    Returns:
        CRSInfo with detected CRS information
    """
    file_path = str(Path(file_path).resolve())
    close_conn = False

    if conn is None:
        conn = duckdb.connect(":memory:")
        close_conn = True

    try:
        result = conn.execute(
            "SELECT value FROM parquet_kv_metadata(?) WHERE key = 'geo'", [file_path]
        ).fetchone()

        if not result:
            return CRSInfo(authority=None, code=None, wkt=None, projjson=None)

        geo_meta = json.loads(result[0])
        columns = geo_meta.get("columns", {})
        geom_info = columns.get(geom_col, {})
        crs = geom_info.get("crs")

        if crs is None:
            # GeoParquet 1.0 spec: missing CRS means WGS84
            return CRSInfo(authority="EPSG", code="4326", wkt=None, projjson=None)

        # Parse PROJJSON format
        if isinstance(crs, dict):
            authority = None
            code = None

            # Look for id field (contains authority and code)
            id_info = crs.get("id", {})
            if isinstance(id_info, dict):
                authority = id_info.get("authority")
                code = str(id_info.get("code")) if id_info.get("code") else None

            return CRSInfo(
                authority=authority,
                code=code,
                wkt=None,
                projjson=crs,
            )

        # Handle WKT string
        if isinstance(crs, str):
            # Try to extract EPSG code from WKT
            authority = None
            code = None
            if 'AUTHORITY["EPSG"' in crs:
                import re

                match = re.search(r'AUTHORITY\["EPSG",\s*"?(\d+)"?\]', crs)
                if match:
                    authority = "EPSG"
                    code = match.group(1)

            return CRSInfo(authority=authority, code=code, wkt=crs, projjson=None)

    except Exception:
        pass
    finally:
        if close_conn:
            conn.close()

    return CRSInfo(authority=None, code=None, wkt=None, projjson=None)


def validate_crs_match(
    file1: str | Path,
    file2: str | Path,
    geom_col1: str = "geometry",
    geom_col2: str = "geometry",
) -> tuple[CRSInfo, CRSInfo]:
    """
    Validate that two files have matching CRS.

    Args:
        file1: Path to first file
        file2: Path to second file
        geom_col1: Geometry column name in first file
        geom_col2: Geometry column name in second file

    Returns:
        Tuple of (crs1, crs2) if they match

    Raises:
        CRSMismatchError: If CRS don't match
    """
    crs1 = detect_crs(file1, geom_col1)
    crs2 = detect_crs(file2, geom_col2)

    if not crs1.is_equivalent_to(crs2):
        raise CRSMismatchError(
            crs1=str(crs1),
            crs2=str(crs2),
            file1=str(file1),
            file2=str(file2),
        )

    return crs1, crs2


@dataclass
class ReprojectResult:
    """Result of a reprojection operation."""

    output_path: Path
    source_crs: str
    target_crs: str
    feature_count: int


def reproject(
    input_file: str | Path,
    output_file: str | Path | None = None,
    target_crs: str = "EPSG:4326",
    on_progress: Callable[[str], None] | None = None,
) -> ReprojectResult:
    """
    Reproject a GeoParquet file to a different CRS using geopandas.

    Args:
        input_file: Path to input GeoParquet file
        output_file: Path to output file. If None, generates name from input.
        target_crs: Target CRS (default: EPSG:4326)
        on_progress: Optional callback for progress messages

    Returns:
        ReprojectResult with information about the operation

    Raises:
        FileNotFoundError: If input file doesn't exist
    """
    input_path = Path(input_file).resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    def log(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    # Read with geopandas
    log("Loading data...")
    gdf = gpd.read_parquet(input_path)
    count = len(gdf)
    log(f"Loaded {count:,} features")

    # Get source CRS
    source_crs = gdf.crs
    source_crs_str = str(source_crs) if source_crs else "unknown"
    log(f"Source CRS: {source_crs_str}")
    log(f"Target CRS: {target_crs}")

    # Reproject using geopandas
    log("Reprojecting...")
    gdf_reprojected = gdf.to_crs(target_crs)

    # Determine output path
    if output_file:
        out_path = Path(output_file).resolve()
    else:
        # Generate output name: input_4326.parquet
        target_suffix = target_crs.replace(":", "_").lower()
        out_path = input_path.parent / f"{input_path.stem}_{target_suffix}.parquet"

    # Write output
    log(f"Writing output to: {out_path}")

    if out_path == input_path:
        # Write to temp file first, then replace
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            gdf_reprojected.to_parquet(tmp_path)
            shutil.move(tmp_path, out_path)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        gdf_reprojected.to_parquet(out_path)

    # Add bbox column for efficient spatial queries
    log("Adding bbox column...")
    with Path(os.devnull).open("w") as devnull, contextlib.redirect_stdout(devnull):
        add_bbox_column(
            str(out_path),
            str(out_path),
            compression="ZSTD",
            compression_level=16,
        )

    return ReprojectResult(
        output_path=out_path,
        source_crs=source_crs_str,
        target_crs=target_crs,
        feature_count=count,
    )
