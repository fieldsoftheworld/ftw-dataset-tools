"""Shared test fixtures for ftw-dataset-tools tests."""

from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import Polygon, box


@pytest.fixture
def sample_polygon() -> Polygon:
    """Create a simple polygon for testing."""
    return box(0, 0, 1, 1)


@pytest.fixture
def sample_geoparquet_4326(tmp_path: Path) -> Path:
    """Create a sample GeoParquet file in EPSG:4326."""
    gdf = gpd.GeoDataFrame(
        {"id": [1, 2, 3], "name": ["a", "b", "c"]},
        geometry=[box(0, 0, 1, 1), box(1, 1, 2, 2), box(2, 2, 3, 3)],
        crs="EPSG:4326",
    )
    path = tmp_path / "sample_4326.parquet"
    gdf.to_parquet(path)
    return path


@pytest.fixture
def sample_geoparquet_3035(tmp_path: Path) -> Path:
    """Create a sample GeoParquet file in EPSG:3035 (Europe)."""
    # Coordinates in EPSG:3035 (LAEA Europe) - approximate Lithuania area
    gdf = gpd.GeoDataFrame(
        {"id": [1]},
        geometry=[box(5150000, 3540000, 5160000, 3550000)],
        crs="EPSG:3035",
    )
    path = tmp_path / "sample_3035.parquet"
    gdf.to_parquet(path)
    return path


@pytest.fixture
def sample_fields_geoparquet(tmp_path: Path) -> Path:
    """Create a sample fields GeoParquet with polygons in EPSG:4326."""
    gdf = gpd.GeoDataFrame(
        {"id": [1, 2, 3, 4]},
        geometry=[
            box(10.0, 50.0, 10.01, 50.01),
            box(10.02, 50.0, 10.03, 50.01),
            box(10.0, 50.02, 10.01, 50.03),
            box(10.02, 50.02, 10.03, 50.03),
        ],
        crs="EPSG:4326",
    )
    path = tmp_path / "fields.parquet"
    gdf.to_parquet(path)
    return path


@pytest.fixture
def sample_grid_geoparquet(tmp_path: Path) -> Path:
    """Create a sample grid GeoParquet with square cells in EPSG:4326."""
    gdf = gpd.GeoDataFrame(
        {"id": ["grid_001", "grid_002"]},
        geometry=[
            box(10.0, 50.0, 10.02, 50.02),
            box(10.02, 50.0, 10.04, 50.02),
        ],
        crs="EPSG:4326",
    )
    path = tmp_path / "grid.parquet"
    gdf.to_parquet(path)
    return path


@pytest.fixture
def sample_mgrs_1km_geoparquet(tmp_path: Path) -> Path:
    """Create a sample MGRS 1km GeoParquet with a single GZD.

    Creates 4 1km cells in GZD 32U, kmSQ ID PE, forming a 2x2 block.
    """
    gdf = gpd.GeoDataFrame(
        {
            "MGRS": ["32UPE0000", "32UPE0100", "32UPE0001", "32UPE0101"],
            "GZD": ["32U", "32U", "32U", "32U"],
            "kmSQ_ID": ["PE", "PE", "PE", "PE"],
            "EASTING": [0, 1, 0, 1],
            "NORTHING": [0, 0, 1, 1],
        },
        geometry=[
            box(9.0, 48.0, 9.01, 48.01),
            box(9.01, 48.0, 9.02, 48.01),
            box(9.0, 48.01, 9.01, 48.02),
            box(9.01, 48.01, 9.02, 48.02),
        ],
        crs="EPSG:4326",
    )
    path = tmp_path / "mgrs_1km.parquet"
    gdf.to_parquet(path)
    return path


@pytest.fixture
def sample_mgrs_1km_minimal_geoparquet(tmp_path: Path) -> Path:
    """Create MGRS 1km GeoParquet with only required columns (MGRS, GZD, geometry)."""
    gdf = gpd.GeoDataFrame(
        {
            "MGRS": ["32UPE0000", "32UPE0100", "32UPE0001", "32UPE0101"],
            "GZD": ["32U", "32U", "32U", "32U"],
        },
        geometry=[
            box(9.0, 48.0, 9.01, 48.01),
            box(9.01, 48.0, 9.02, 48.01),
            box(9.0, 48.01, 9.01, 48.02),
            box(9.01, 48.01, 9.02, 48.02),
        ],
        crs="EPSG:4326",
    )
    path = tmp_path / "mgrs_1km_minimal.parquet"
    gdf.to_parquet(path)
    return path


@pytest.fixture
def sample_mgrs_multiple_gzd_geoparquet(tmp_path: Path) -> Path:
    """Create MGRS 1km GeoParquet with multiple GZDs (should cause error)."""
    gdf = gpd.GeoDataFrame(
        {
            "MGRS": ["32UPE0000", "33UPE0000"],
            "GZD": ["32U", "33U"],
        },
        geometry=[
            box(9.0, 48.0, 9.01, 48.01),
            box(15.0, 48.0, 15.01, 48.01),
        ],
        crs="EPSG:4326",
    )
    path = tmp_path / "mgrs_multi_gzd.parquet"
    gdf.to_parquet(path)
    return path


@pytest.fixture
def sample_chips_with_coverage(tmp_path: Path) -> Path:
    """Create chips GeoParquet with grid_id, field_coverage_pct, geometry."""
    gdf = gpd.GeoDataFrame(
        {
            "id": ["grid_001", "grid_002", "grid_003"],
            "field_coverage_pct": [50.0, 25.0, 0.5],
        },
        geometry=[
            box(10.0, 50.0, 10.01, 50.01),
            box(10.01, 50.0, 10.02, 50.01),
            box(10.02, 50.0, 10.03, 50.01),
        ],
        crs="EPSG:4326",
    )
    path = tmp_path / "chips_with_coverage.parquet"
    gdf.to_parquet(path)
    return path


@pytest.fixture
def sample_boundaries_geoparquet(tmp_path: Path) -> Path:
    """Create boundary polygons GeoParquet matching chip locations."""
    gdf = gpd.GeoDataFrame(
        {"id": [1, 2, 3]},
        geometry=[
            box(10.0, 50.0, 10.005, 50.005),
            box(10.01, 50.0, 10.015, 50.005),
            box(10.02, 50.0, 10.025, 50.005),
        ],
        crs="EPSG:4326",
    )
    path = tmp_path / "boundaries.parquet"
    gdf.to_parquet(path)
    return path


@pytest.fixture
def sample_boundary_lines_geoparquet(tmp_path: Path) -> Path:
    """Create boundary lines GeoParquet from boundary polygons."""
    from shapely.geometry import LineString

    gdf = gpd.GeoDataFrame(
        {"id": [1, 2, 3]},
        geometry=[
            LineString([(10.0, 50.0), (10.005, 50.0), (10.005, 50.005)]),
            LineString([(10.01, 50.0), (10.015, 50.0), (10.015, 50.005)]),
            LineString([(10.02, 50.0), (10.025, 50.0), (10.025, 50.005)]),
        ],
        crs="EPSG:4326",
    )
    path = tmp_path / "boundary_lines.parquet"
    gdf.to_parquet(path)
    return path
