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
