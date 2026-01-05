"""Tests for MGRS grid generation utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import geopandas as gpd
import pytest
from shapely.geometry import Point, Polygon, box

from ftw_dataset_tools.api.grid import (
    DYNAMIC_GRID,
    CRSError,
    GetGridResult,
    get_grid,
    mgrs_tile_to_polygon,
    parse_mgrs_tile,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestParseMgrsTile:
    """Tests for parse_mgrs_tile function."""

    def test_valid_1km_tile(self):
        """Parse a valid 1km MGRS tile ID."""
        result = parse_mgrs_tile("33UUP5081")
        assert result is not None
        gzd, square_id, numeric = result
        assert gzd == "33U"
        assert square_id == "UP"
        assert numeric == "5081"

    def test_valid_10km_tile(self):
        """Parse a valid 10km MGRS tile ID."""
        result = parse_mgrs_tile("33UUP58")
        assert result is not None
        gzd, square_id, numeric = result
        assert gzd == "33U"
        assert square_id == "UP"
        assert numeric == "58"

    def test_valid_100km_tile(self):
        """Parse a valid 100km MGRS tile ID (no numeric part)."""
        result = parse_mgrs_tile("33UUP")
        assert result is not None
        gzd, square_id, numeric = result
        assert gzd == "33U"
        assert square_id == "UP"
        assert numeric == ""

    def test_valid_100m_tile(self):
        """Parse a valid 100m MGRS tile ID."""
        result = parse_mgrs_tile("33UUP508812")
        assert result is not None
        gzd, square_id, numeric = result
        assert gzd == "33U"
        assert square_id == "UP"
        assert numeric == "508812"

    def test_single_digit_zone(self):
        """Parse tile with single digit zone number."""
        result = parse_mgrs_tile("4QFJ1234")
        assert result is not None
        gzd, square_id, numeric = result
        assert gzd == "4Q"
        assert square_id == "FJ"
        assert numeric == "1234"

    def test_lowercase_input(self):
        """Parse lowercase input (should be case-insensitive)."""
        result = parse_mgrs_tile("33uup5081")
        assert result is not None
        gzd, square_id, _numeric = result
        assert gzd == "33U"
        assert square_id == "UP"

    def test_invalid_odd_numeric(self):
        """Reject tile with odd-length numeric part."""
        result = parse_mgrs_tile("33UUP508")
        assert result is None

    def test_invalid_format(self):
        """Reject completely invalid format."""
        assert parse_mgrs_tile("invalid") is None
        assert parse_mgrs_tile("") is None
        assert parse_mgrs_tile("123") is None

    def test_invalid_letters(self):
        """Reject tiles with invalid letter combinations."""
        # I and O are not used in MGRS
        assert parse_mgrs_tile("33UIO5081") is None


class TestMgrsTileToPolygon:
    """Tests for mgrs_tile_to_polygon function."""

    def test_valid_1km_tile_returns_polygon(self):
        """Convert a 1km tile to polygon."""
        polygon = mgrs_tile_to_polygon("33UUP5081")
        assert polygon is not None
        assert isinstance(polygon, Polygon)
        assert polygon.is_valid

    def test_polygon_has_correct_structure(self):
        """Verify polygon has correct structure (5 points, closed ring)."""
        polygon = mgrs_tile_to_polygon("33UUP5081")
        assert polygon is not None
        # Exterior ring should have 5 points (4 corners + closing point)
        assert len(polygon.exterior.coords) == 5

    def test_polygon_approximate_size(self):
        """Verify 1km tile polygon is approximately 1km x 1km."""
        polygon = mgrs_tile_to_polygon("33UUP5081")
        assert polygon is not None
        minx, miny, maxx, maxy = polygon.bounds
        # At mid-latitudes, 1km is roughly 0.009 degrees
        # Allow for some variation due to projection
        width = maxx - minx
        height = maxy - miny
        assert 0.005 < width < 0.02, f"Width {width} outside expected range"
        assert 0.005 < height < 0.02, f"Height {height} outside expected range"

    def test_10km_tile(self):
        """Convert a 10km tile to polygon."""
        polygon = mgrs_tile_to_polygon("33UUP58")
        assert polygon is not None
        assert polygon.is_valid
        minx, miny, maxx, maxy = polygon.bounds
        # 10km is roughly 0.09 degrees
        width = maxx - minx
        height = maxy - miny
        assert 0.05 < width < 0.2, f"Width {width} outside expected range"
        assert 0.05 < height < 0.2, f"Height {height} outside expected range"

    def test_invalid_tile_returns_none(self):
        """Invalid tile ID should return None."""
        result = mgrs_tile_to_polygon("invalid")
        assert result is None


class TestGetGridDynamic:
    """Tests for get_grid with DYNAMIC_GRID source."""

    @pytest.fixture
    def sample_input_file(self, tmp_path: Path) -> Path:
        """Create a sample input GeoParquet file."""
        # Create a small GeoDataFrame with a few points
        gdf = gpd.GeoDataFrame(
            {"id": [1, 2, 3]},
            geometry=[
                Point(11.0, 48.0),
                Point(11.05, 48.05),
                Point(11.1, 48.1),
            ],
            crs="EPSG:4326",
        )
        output_path = tmp_path / "input.parquet"
        gdf.to_parquet(output_path)
        return output_path

    @pytest.fixture
    def sample_polygon_file(self, tmp_path: Path) -> Path:
        """Create a sample input file with polygon geometry."""
        gdf = gpd.GeoDataFrame(
            {"id": [1]},
            geometry=[box(11.0, 48.0, 11.1, 48.1)],
            crs="EPSG:4326",
        )
        output_path = tmp_path / "polygon_input.parquet"
        gdf.to_parquet(output_path)
        return output_path

    def test_generates_grid_for_input_file(self, sample_input_file: Path, tmp_path: Path):
        """Generate grid tiles covering input file geometries."""
        output_path = tmp_path / "output_grid.parquet"
        result = get_grid(
            sample_input_file,
            output_file=output_path,
            grid_source=DYNAMIC_GRID,
            precision=2,
        )

        assert isinstance(result, GetGridResult)
        assert result.grid_count > 0
        assert output_path.exists()

        # Verify output has tile_id column
        gdf = gpd.read_parquet(output_path)
        assert "tile_id" in gdf.columns
        assert len(gdf) == result.grid_count

    def test_default_output_filename(self, sample_input_file: Path):
        """Default output filename is <input>_grid.parquet."""
        result = get_grid(
            sample_input_file,
            grid_source=DYNAMIC_GRID,
            precision=2,
        )

        expected_path = sample_input_file.parent / "input_grid.parquet"
        assert result.output_path == expected_path
        assert expected_path.exists()

    def test_file_not_found_error(self, tmp_path: Path):
        """Raise FileNotFoundError for missing input file."""
        with pytest.raises(FileNotFoundError):
            get_grid(tmp_path / "nonexistent.parquet", grid_source=DYNAMIC_GRID)

    def test_crs_error_for_non_wgs84(self, tmp_path: Path):
        """Raise CRSError for non-WGS84 input file."""
        # Create file with UTM CRS
        gdf = gpd.GeoDataFrame(
            {"id": [1]},
            geometry=[Point(500000, 5300000)],
            crs="EPSG:32633",  # UTM zone 33N
        )
        input_path = tmp_path / "utm_input.parquet"
        gdf.to_parquet(input_path)

        with pytest.raises(CRSError):
            get_grid(input_path, grid_source=DYNAMIC_GRID)

    def test_polygon_input(self, sample_polygon_file: Path, tmp_path: Path):
        """Generate grid for polygon input file."""
        output_path = tmp_path / "polygon_grid.parquet"
        result = get_grid(
            sample_polygon_file,
            output_file=output_path,
            grid_source=DYNAMIC_GRID,
            precision=2,
        )

        assert result.grid_count > 0
        # Should cover the full polygon extent
        gdf = gpd.read_parquet(output_path)
        assert len(gdf) > 50  # ~100 1km tiles for 10km x 10km area

    def test_precision_affects_tile_count(self, sample_polygon_file: Path, tmp_path: Path):
        """Higher precision should generate more tiles."""
        result_1km = get_grid(
            sample_polygon_file,
            output_file=tmp_path / "grid_1km.parquet",
            grid_source=DYNAMIC_GRID,
            precision=2,
        )
        result_10km = get_grid(
            sample_polygon_file,
            output_file=tmp_path / "grid_10km.parquet",
            grid_source=DYNAMIC_GRID,
            precision=1,
        )

        assert result_1km.grid_count > result_10km.grid_count

    def test_progress_callback(self, sample_input_file: Path, tmp_path: Path):
        """Progress callback is called."""
        messages = []

        def on_progress(msg: str) -> None:
            messages.append(msg)

        get_grid(
            sample_input_file,
            output_file=tmp_path / "grid.parquet",
            grid_source=DYNAMIC_GRID,
            on_progress=on_progress,
        )

        assert len(messages) > 0
        assert any("Generating" in msg for msg in messages)


@pytest.mark.network
class TestGetGridRemote:
    """
    Tests for get_grid with remote Source Coop source.

    These tests require network access and are skipped by default.
    Run with: pytest -m network
    """

    @pytest.fixture
    def sample_input_file(self, tmp_path: Path) -> Path:
        """Create a sample input GeoParquet file."""
        gdf = gpd.GeoDataFrame(
            {"id": [1]},
            geometry=[box(11.0, 48.0, 11.1, 48.1)],
            crs="EPSG:4326",
        )
        output_path = tmp_path / "input.parquet"
        gdf.to_parquet(output_path)
        return output_path

    def test_fetches_from_source_coop(self, sample_input_file: Path, tmp_path: Path):
        """Fetch grid tiles from Source Coop."""
        output_path = tmp_path / "grid.parquet"
        result = get_grid(
            sample_input_file,
            output_file=output_path,
        )

        assert isinstance(result, GetGridResult)
        assert result.grid_count > 0
        assert output_path.exists()

    def test_dynamic_matches_remote(self, sample_input_file: Path, tmp_path: Path):
        """Dynamic generation should match remote Source Coop tiles."""
        # Fetch from remote
        get_grid(
            sample_input_file,
            output_file=tmp_path / "remote.parquet",
        )

        # Generate dynamically
        get_grid(
            sample_input_file,
            output_file=tmp_path / "dynamic.parquet",
            grid_source=DYNAMIC_GRID,
            precision=2,
        )

        # Load and compare
        gdf_remote = gpd.read_parquet(tmp_path / "remote.parquet")
        gdf_dynamic = gpd.read_parquet(tmp_path / "dynamic.parquet")

        remote_tiles = set(gdf_remote["MGRS"].tolist())
        dynamic_tiles = set(gdf_dynamic["tile_id"].tolist())

        # Tile sets should be identical
        assert remote_tiles == dynamic_tiles

    @pytest.mark.parametrize(
        "bbox",
        [
            # Portland, Oregon area
            (-122.85, 45.40, -122.45, 45.60),
            # Berlin, Germany
            (13.2, 52.4, 13.6, 52.6),
            # Tokyo, Japan
            (139.6, 35.5, 139.9, 35.8),
            # São Paulo, Brazil (southern hemisphere)
            (-46.8, -23.7, -46.5, -23.4),
            # Cape Town, South Africa (southern hemisphere)
            (18.3, -34.1, 18.6, -33.8),
            # Small area crossing UTM zone boundary (edge case)
            (5.9, 50.0, 6.1, 50.2),
        ],
        ids=["portland", "berlin", "tokyo", "sao_paulo", "cape_town", "utm_boundary"],
    )
    def test_dynamic_matches_remote_various_locations(
        self, bbox: tuple[float, float, float, float], tmp_path: Path
    ):
        """Dynamic generation should exactly match remote tiles for various global locations."""
        xmin, ymin, xmax, ymax = bbox

        # Create input file
        gdf = gpd.GeoDataFrame(
            {"id": [1]},
            geometry=[box(xmin, ymin, xmax, ymax)],
            crs="EPSG:4326",
        )
        input_path = tmp_path / "input.parquet"
        gdf.to_parquet(input_path)

        # Fetch from remote
        get_grid(
            input_path,
            output_file=tmp_path / "remote.parquet",
        )

        # Generate dynamically
        get_grid(
            input_path,
            output_file=tmp_path / "dynamic.parquet",
            grid_source=DYNAMIC_GRID,
            precision=2,
        )

        # Load and compare
        gdf_remote = gpd.read_parquet(tmp_path / "remote.parquet")
        gdf_dynamic = gpd.read_parquet(tmp_path / "dynamic.parquet")

        remote_tiles = set(gdf_remote["MGRS"].tolist())
        dynamic_tiles = set(gdf_dynamic["tile_id"].tolist())

        # Tile sets should be identical
        assert remote_tiles == dynamic_tiles, (
            f"Mismatch for bbox {bbox}: "
            f"remote={len(remote_tiles)}, dynamic={len(dynamic_tiles)}, "
            f"only_remote={remote_tiles - dynamic_tiles}, "
            f"only_dynamic={dynamic_tiles - remote_tiles}"
        )
