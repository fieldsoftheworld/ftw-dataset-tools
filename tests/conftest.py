"""Shared test fixtures for ftw-dataset-tools tests."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import duckdb

if TYPE_CHECKING:
    from pathlib import Path
import pyarrow.parquet as pq
import pytest


@pytest.fixture
def duckdb_conn():
    """Create a DuckDB connection with spatial extension loaded."""
    conn = duckdb.connect(":memory:")
    conn.execute("INSTALL spatial; LOAD spatial;")
    yield conn
    conn.close()


@pytest.fixture
def create_geoparquet(duckdb_conn, tmp_path):
    """Factory fixture to create GeoParquet files with CRS metadata for testing.

    Returns a function that creates a GeoParquet file with the specified
    geometry (in WKT format) and CRS (as EPSG code).

    The geometry coordinates should be in the coordinate system of the specified CRS.
    """

    def _create_geoparquet(
        filename: str,
        geometry_wkt: str,
        crs_epsg: int,
        geom_col: str = "geometry",
        extra_columns: dict | None = None,
    ) -> Path:
        """Create a GeoParquet file with CRS metadata.

        Args:
            filename: Name of the output file (will be created in tmp_path)
            geometry_wkt: WKT representation of the geometry
            crs_epsg: EPSG code for the CRS (e.g., 32610 for UTM Zone 10N)
            geom_col: Name of the geometry column (default: "geometry")
            extra_columns: Optional dict of column_name -> value to add

        Returns:
            Path to the created GeoParquet file
        """
        output_path = tmp_path / filename

        # Build SQL for extra columns
        extra_sql = ""
        if extra_columns:
            extra_sql = ", " + ", ".join(
                f"'{v}' AS {k}" if isinstance(v, str) else f"{v} AS {k}"
                for k, v in extra_columns.items()
            )

        # Create parquet with DuckDB (writes geo metadata but no CRS)
        query = f"""
            SELECT
                1 AS id{extra_sql},
                ST_GeomFromText('{geometry_wkt}') AS {geom_col}
        """
        duckdb_conn.execute(f"COPY ({query}) TO '{output_path}' (FORMAT PARQUET)")

        # Patch the geo metadata to add CRS info using pyarrow
        table = pq.read_table(output_path)
        metadata = table.schema.metadata or {}

        # Parse existing geo metadata
        geo_meta = json.loads(metadata.get(b"geo", b"{}"))

        # Add CRS in PROJJSON format (GeoParquet 1.0+ format)
        if geom_col in geo_meta.get("columns", {}):
            geo_meta["columns"][geom_col]["crs"] = {
                "$schema": "https://proj.org/schemas/v0.7/projjson.schema.json",
                "type": "GeographicCRS" if crs_epsg == 4326 else "ProjectedCRS",
                "id": {"authority": "EPSG", "code": crs_epsg},
            }

        # Update metadata
        new_metadata = {**metadata, b"geo": json.dumps(geo_meta).encode()}
        new_schema = table.schema.with_metadata(new_metadata)
        new_table = table.cast(new_schema)

        # Write back
        pq.write_table(new_table, output_path)

        return output_path

    return _create_geoparquet
