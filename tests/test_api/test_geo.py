"""Tests for the geo API module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ftw_dataset_tools.api.geo import ReprojectResult, reproject


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
        mock_table = MagicMock()
        mock_table.crs = {
            "id": {"authority": "EPSG", "code": 3035},
            "name": "ETRS89-extended / LAEA Europe",
        }
        mock_table.num_rows = 100
        mock_reprojected = MagicMock()
        mock_table.reproject.return_value = mock_reprojected
        mock_gpio.read.return_value = mock_table

        output_file = tmp_path / "output.parquet"

        result = reproject(str(input_file), str(output_file), "EPSG:4326")

        assert result.source_crs == "EPSG:3035"
        assert result.target_crs == "EPSG:4326"
        assert result.feature_count == 100
        mock_table.reproject.assert_called_once_with("EPSG:4326")
        mock_reprojected.add_bbox.assert_called_once()
        mock_reprojected.add_bbox.return_value.write.assert_called_once()

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
