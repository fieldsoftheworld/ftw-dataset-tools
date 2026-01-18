"""Tests for the geo API module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import duckdb
import geopandas as gpd
import pyarrow.parquet as pq
import pytest
from shapely.geometry import box

from ftw_dataset_tools.api.geo import ReprojectResult, reproject, write_geoparquet


class TestReprojectResult:
    """Tests for ReprojectResult dataclass."""

    def test_dataclass_fields(self) -> None:
        """Test that ReprojectResult has the expected fields."""
        result = ReprojectResult(
            output_path=Path("/tmp/test.parquet"),
            source_crs="EPSG:3035",
            target_crs="EPSG:4326",
            feature_count=100,
        )
        assert result.output_path == Path("/tmp/test.parquet")
        assert result.source_crs == "EPSG:3035"
        assert result.target_crs == "EPSG:4326"
        assert result.feature_count == 100


class TestReproject:
    """Tests for reproject function."""

    def test_file_not_found(self) -> None:
        """Test that FileNotFoundError is raised for missing input file."""
        with pytest.raises(FileNotFoundError, match="Input file not found"):
            reproject("/nonexistent/input.parquet")

    @patch("ftw_dataset_tools.api.geo.gpio")
    def test_reproject_with_epsg_crs(self, mock_gpio: MagicMock, tmp_path: Path) -> None:
        """Test reprojection with EPSG code in PROJJSON format."""
        # Create a dummy input file
        input_file = tmp_path / "input.parquet"
        input_file.touch()

        # Mock the gpio.read and Table
        # First read: for the initial table
        mock_table = MagicMock()
        mock_table.crs = {
            "id": {"authority": "EPSG", "code": 3035},
            "name": "ETRS89-extended / LAEA Europe",
        }
        mock_table.num_rows = 100
        mock_reprojected = MagicMock()
        mock_table.reproject.return_value = mock_reprojected

        # Second read: for adding bbox after writing temp file
        mock_temp_table = MagicMock()
        mock_gpio.read.side_effect = [mock_table, mock_temp_table]

        output_file = tmp_path / "output.parquet"

        result = reproject(str(input_file), str(output_file), "EPSG:4326")

        assert result.source_crs == "EPSG:3035"
        assert result.target_crs == "EPSG:4326"
        assert result.feature_count == 100
        mock_table.reproject.assert_called_once_with("EPSG:4326")
        # Reprojected table is written to temp file
        mock_reprojected.write.assert_called_once()
        # Then temp file is read, bbox added, and written to output
        mock_temp_table.add_bbox.assert_called_once()
        mock_temp_table.add_bbox.return_value.write.assert_called_once()

    @patch("ftw_dataset_tools.api.geo.gpio")
    def test_reproject_with_string_crs(self, mock_gpio: MagicMock, tmp_path: Path) -> None:
        """Test reprojection with string CRS identifier."""
        input_file = tmp_path / "input.parquet"
        input_file.touch()

        mock_table = MagicMock()
        mock_table.crs = "EPSG:4326"
        mock_table.num_rows = 50
        mock_reprojected = MagicMock()
        mock_table.reproject.return_value = mock_reprojected
        mock_gpio.read.return_value = mock_table

        output_file = tmp_path / "output.parquet"

        result = reproject(str(input_file), str(output_file), "EPSG:32610")

        assert result.source_crs == "EPSG:4326"
        assert result.target_crs == "EPSG:32610"
        assert result.feature_count == 50

    @patch("ftw_dataset_tools.api.geo.gpio")
    def test_reproject_with_none_crs(self, mock_gpio: MagicMock, tmp_path: Path) -> None:
        """Test reprojection when source CRS is None."""
        input_file = tmp_path / "input.parquet"
        input_file.touch()

        mock_table = MagicMock()
        mock_table.crs = None
        mock_table.num_rows = 25
        mock_reprojected = MagicMock()
        mock_table.reproject.return_value = mock_reprojected
        mock_gpio.read.return_value = mock_table

        output_file = tmp_path / "output.parquet"

        result = reproject(str(input_file), str(output_file), "EPSG:4326")

        assert result.source_crs == "unknown"

    @patch("ftw_dataset_tools.api.geo.gpio")
    def test_reproject_with_projjson_without_id(self, mock_gpio: MagicMock, tmp_path: Path) -> None:
        """Test reprojection with PROJJSON that lacks an id field."""
        input_file = tmp_path / "input.parquet"
        input_file.touch()

        mock_table = MagicMock()
        mock_table.crs = {"name": "Custom CRS"}
        mock_table.num_rows = 10
        mock_reprojected = MagicMock()
        mock_table.reproject.return_value = mock_reprojected
        mock_gpio.read.return_value = mock_table

        output_file = tmp_path / "output.parquet"

        result = reproject(str(input_file), str(output_file), "EPSG:4326")

        assert result.source_crs == "Custom CRS"

    @patch("ftw_dataset_tools.api.geo.gpio")
    def test_reproject_auto_generated_output_path(
        self, mock_gpio: MagicMock, tmp_path: Path
    ) -> None:
        """Test that output path is auto-generated when not provided."""
        input_file = tmp_path / "my_data.parquet"
        input_file.touch()

        mock_table = MagicMock()
        mock_table.crs = "EPSG:3035"
        mock_table.num_rows = 10
        mock_reprojected = MagicMock()
        mock_table.reproject.return_value = mock_reprojected
        mock_gpio.read.return_value = mock_table

        result = reproject(str(input_file), target_crs="EPSG:4326")

        expected_output = tmp_path / "my_data_epsg_4326.parquet"
        assert result.output_path == expected_output

    @patch("ftw_dataset_tools.api.geo.gpio")
    def test_reproject_progress_callback(self, mock_gpio: MagicMock, tmp_path: Path) -> None:
        """Test that progress callback is invoked."""
        input_file = tmp_path / "input.parquet"
        input_file.touch()

        mock_table = MagicMock()
        mock_table.crs = "EPSG:3035"
        mock_table.num_rows = 100
        mock_reprojected = MagicMock()
        mock_table.reproject.return_value = mock_reprojected
        mock_gpio.read.return_value = mock_table

        output_file = tmp_path / "output.parquet"
        progress_messages = []

        def on_progress(msg: str) -> None:
            progress_messages.append(msg)

        reproject(str(input_file), str(output_file), "EPSG:4326", on_progress)

        assert len(progress_messages) > 0
        assert any("Loading" in msg for msg in progress_messages)
        assert any("Source CRS" in msg for msg in progress_messages)
        assert any("Target CRS" in msg for msg in progress_messages)
        assert any("Reprojecting" in msg for msg in progress_messages)


class TestReprojectIntegration:
    """Integration tests for reproject that verify actual data transformation."""

    def test_reproject_updates_bbox_column(self, tmp_path: Path) -> None:
        """Test that bbox column is recalculated after reprojection.

        This test catches the bug where geometry was reprojected but bbox
        column retained coordinates from the original CRS.
        """
        import geoparquet_io as gpio

        # Create test file in EPSG:3035 using geopandas (produces proper GeoParquet with CRS)
        # This is a polygon in Lithuania area (EPSG:3035 uses meters)
        gdf = gpd.GeoDataFrame(
            {"id": [1]},
            geometry=[box(5150000, 3540000, 5160000, 3550000)],
            crs="EPSG:3035",
        )

        input_file = tmp_path / "input_3035.parquet"
        gdf.to_parquet(input_file)

        # Add bbox column using gpio (reads the file and adds bbox)
        gpio.read(str(input_file)).add_bbox().write(str(input_file))

        # Verify input has bbox in EPSG:3035 range (millions of meters)
        input_table = pq.read_table(input_file)
        input_bbox = input_table.column("bbox")[0].as_py()
        assert input_bbox["xmin"] > 1_000_000, "Input bbox should be in meters"
        assert input_bbox["ymin"] > 1_000_000, "Input bbox should be in meters"

        # Reproject to EPSG:4326
        output_file = tmp_path / "output_4326.parquet"
        result = reproject(str(input_file), str(output_file), "EPSG:4326")

        assert result.output_path == output_file

        # Read output and verify bbox is in EPSG:4326 range (degrees: -180 to 180, -90 to 90)
        output_table = pq.read_table(output_file)
        output_bbox = output_table.column("bbox")[0].as_py()

        # bbox should now be in degrees, not meters
        assert -180 <= output_bbox["xmin"] <= 180, (
            f"bbox xmin should be in degrees, got {output_bbox['xmin']}"
        )
        assert -180 <= output_bbox["xmax"] <= 180, (
            f"bbox xmax should be in degrees, got {output_bbox['xmax']}"
        )
        assert -90 <= output_bbox["ymin"] <= 90, (
            f"bbox ymin should be in degrees, got {output_bbox['ymin']}"
        )
        assert -90 <= output_bbox["ymax"] <= 90, (
            f"bbox ymax should be in degrees, got {output_bbox['ymax']}"
        )

        # Verify geometry is also in correct range using DuckDB
        con = duckdb.connect()
        con.install_extension("spatial")
        con.load_extension("spatial")
        bounds = con.execute(f"""
            SELECT
                min(ST_XMin(geometry)) as min_x,
                max(ST_XMax(geometry)) as max_x
            FROM read_parquet('{output_file}')
        """).fetchone()

        assert -180 <= bounds[0] <= 180, "Geometry xmin should be in degrees"
        assert -180 <= bounds[1] <= 180, "Geometry xmax should be in degrees"


class TestWriteGeoparquetIntegration:
    """Integration tests for write_geoparquet with real DuckDB queries."""

    def test_write_from_duckdb_query_produces_valid_geoparquet(self, tmp_path: Path) -> None:
        """Test that write_geoparquet from DuckDB query produces readable GeoParquet.

        This test catches the bug where using Arrow table directly from DuckDB
        resulted in geometry format that geoparquet-io couldn't parse.
        """
        # Create a test GeoDataFrame and save it
        gdf = gpd.GeoDataFrame(
            {"id": [1, 2, 3], "name": ["a", "b", "c"]},
            geometry=[
                box(0, 0, 1, 1),
                box(1, 1, 2, 2),
                box(2, 2, 3, 3),
            ],
            crs="EPSG:4326",
        )
        source_file = tmp_path / "source.parquet"
        gdf.to_parquet(source_file)

        # Use DuckDB to query and write via write_geoparquet
        con = duckdb.connect()
        con.install_extension("spatial")
        con.load_extension("spatial")

        output_file = tmp_path / "output.parquet"
        write_geoparquet(
            output_file,
            conn=con,
            query=f"SELECT * FROM read_parquet('{source_file}')",
        )

        # Verify output is valid GeoParquet that can be read
        assert output_file.exists()

        # Should be readable by geoparquet-io
        import geoparquet_io as gpio

        table = gpio.read(str(output_file))
        assert table.num_rows == 3

        # Should have bbox column
        output_table = pq.read_table(output_file)
        assert "bbox" in output_table.schema.names

        # bbox values should be valid
        bbox = output_table.column("bbox")[0].as_py()
        assert bbox["xmin"] == 0
        assert bbox["ymin"] == 0
        assert bbox["xmax"] == 1
        assert bbox["ymax"] == 1

    def test_write_from_geodataframe_produces_valid_geoparquet(self, tmp_path: Path) -> None:
        """Test that write_geoparquet from GeoDataFrame produces readable GeoParquet."""
        gdf = gpd.GeoDataFrame(
            {"id": [1]},
            geometry=[box(10, 20, 30, 40)],
            crs="EPSG:4326",
        )

        output_file = tmp_path / "output.parquet"
        write_geoparquet(output_file, gdf=gdf)

        # Verify output has bbox column with correct values
        output_table = pq.read_table(output_file)
        assert "bbox" in output_table.schema.names

        bbox = output_table.column("bbox")[0].as_py()
        assert bbox["xmin"] == 10
        assert bbox["ymin"] == 20
        assert bbox["xmax"] == 30
        assert bbox["ymax"] == 40


class TestDetectCRS:
    """Tests for detect_crs function."""

    def test_detect_crs_epsg_4326(self, sample_geoparquet_4326: Path) -> None:
        """Test detecting EPSG:4326 CRS."""
        from ftw_dataset_tools.api.geo import detect_crs

        crs_info = detect_crs(sample_geoparquet_4326)
        assert crs_info.authority_code == "EPSG:4326"

    def test_detect_crs_epsg_3035(self, sample_geoparquet_3035: Path) -> None:
        """Test detecting EPSG:3035 CRS."""
        from ftw_dataset_tools.api.geo import detect_crs

        crs_info = detect_crs(sample_geoparquet_3035)
        assert crs_info.authority == "EPSG"
        assert crs_info.code == "3035"

    def test_detect_crs_with_existing_connection(self, sample_geoparquet_4326: Path) -> None:
        """Test detect_crs with a provided DuckDB connection."""
        import duckdb

        from ftw_dataset_tools.api.geo import detect_crs

        conn = duckdb.connect(":memory:")
        crs_info = detect_crs(sample_geoparquet_4326, conn=conn)
        assert crs_info.authority_code == "EPSG:4326"
        conn.close()


class TestCRSInfo:
    """Tests for CRSInfo dataclass."""

    def test_authority_code_property(self) -> None:
        """Test authority_code combines authority and code."""
        from ftw_dataset_tools.api.geo import CRSInfo

        crs = CRSInfo(authority="EPSG", code="4326", wkt=None, projjson=None)
        assert crs.authority_code == "EPSG:4326"

    def test_authority_code_none_when_missing(self) -> None:
        """Test authority_code is None when components missing."""
        from ftw_dataset_tools.api.geo import CRSInfo

        crs = CRSInfo(authority=None, code="4326", wkt=None, projjson=None)
        assert crs.authority_code is None

    def test_is_equivalent_to_same_authority(self) -> None:
        """Test CRS equivalence with same authority code."""
        from ftw_dataset_tools.api.geo import CRSInfo

        crs1 = CRSInfo(authority="EPSG", code="4326", wkt=None, projjson=None)
        crs2 = CRSInfo(authority="epsg", code="4326", wkt=None, projjson=None)
        assert crs1.is_equivalent_to(crs2)

    def test_is_equivalent_to_different(self) -> None:
        """Test CRS non-equivalence with different codes."""
        from ftw_dataset_tools.api.geo import CRSInfo

        crs1 = CRSInfo(authority="EPSG", code="4326", wkt=None, projjson=None)
        crs2 = CRSInfo(authority="EPSG", code="3035", wkt=None, projjson=None)
        assert not crs1.is_equivalent_to(crs2)

    def test_str_representation(self) -> None:
        """Test string representation of CRSInfo."""
        from ftw_dataset_tools.api.geo import CRSInfo

        crs = CRSInfo(authority="EPSG", code="4326", wkt=None, projjson=None)
        assert str(crs) == "EPSG:4326"


class TestDetectGeometryColumn:
    """Tests for detect_geometry_column function."""

    def test_detect_geometry_column_default(self, sample_geoparquet_4326: Path) -> None:
        """Test detecting default geometry column name."""
        from ftw_dataset_tools.api.geo import detect_geometry_column

        geom_col = detect_geometry_column(sample_geoparquet_4326)
        assert geom_col == "geometry"

    def test_detect_geometry_column_nonexistent_file(self) -> None:
        """Test with nonexistent file returns None."""
        from ftw_dataset_tools.api.geo import detect_geometry_column

        result = detect_geometry_column(Path("/nonexistent/file.parquet"))
        # Should handle gracefully
        assert result is None


class TestHasBboxColumn:
    """Tests for has_bbox_column function."""

    def test_has_bbox_column_without_bbox(self, sample_geoparquet_4326: Path) -> None:
        """Test file without bbox column returns False."""
        from ftw_dataset_tools.api.geo import has_bbox_column

        # Standard geopandas output doesn't have bbox column
        assert has_bbox_column(sample_geoparquet_4326) is False

    def test_has_bbox_column_with_bbox(self, tmp_path: Path) -> None:
        """Test file with bbox column returns True."""
        import geoparquet_io as gpio

        from ftw_dataset_tools.api.geo import has_bbox_column

        # Create file and add bbox
        gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:4326")
        path = tmp_path / "with_bbox.parquet"
        gdf.to_parquet(path)
        gpio.read(str(path)).add_bbox().write(str(path))

        assert has_bbox_column(path) is True


class TestGetBboxColumnName:
    """Tests for get_bbox_column_name function."""

    def test_get_bbox_column_name_returns_none(self, sample_geoparquet_4326: Path) -> None:
        """Test returns None when no bbox column exists."""
        from ftw_dataset_tools.api.geo import get_bbox_column_name

        result = get_bbox_column_name(sample_geoparquet_4326)
        assert result is None

    def test_get_bbox_column_name_returns_name(self, tmp_path: Path) -> None:
        """Test returns bbox column name when it exists."""
        import geoparquet_io as gpio

        from ftw_dataset_tools.api.geo import get_bbox_column_name

        gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:4326")
        path = tmp_path / "with_bbox.parquet"
        gdf.to_parquet(path)
        gpio.read(str(path)).add_bbox().write(str(path))

        result = get_bbox_column_name(path)
        assert result == "bbox"
