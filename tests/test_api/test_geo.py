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

    @pytest.mark.skip(
        reason="Requires geoparquet-io >0.8.0 which fixes reproject bbox computation issues"
    )
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
