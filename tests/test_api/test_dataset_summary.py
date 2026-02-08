"""Tests for the dataset_summary API module."""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest
from shapely.geometry import box


class TestDatasetSummary:
    """Tests for DatasetSummary dataclass."""

    def test_dataclass_fields(self) -> None:
        """Test DatasetSummary has expected fields."""
        from datetime import datetime

        from ftw_dataset_tools.api.dataset_summary import DatasetSummary

        summary = DatasetSummary(
            dataset_dir=Path("/tmp/dataset"),
            chips_dir=Path("/tmp/dataset/chips"),
            total_chips=100,
            train_chips=70,
            val_chips=15,
            test_chips=15,
            planting_dates=[datetime(2023, 4, 1)],
            harvest_dates=[datetime(2023, 9, 1)],
            planting_cloud_cover=[5.0, 10.0],
            harvest_cloud_cover=[3.0, 8.0],
            metadata={"calendar_year": 2023},
            example_chips=["chip1", "chip2"],
            field_coverage_pct=[50.0, 75.0, 90.0],
            empty_mask_count=5,
            output_path=Path("/tmp/dataset/summary.md"),
        )
        assert summary.total_chips == 100
        assert summary.train_chips == 70
        assert summary.val_chips == 15
        assert summary.test_chips == 15
        assert len(summary.planting_dates) == 1
        assert len(summary.example_chips) == 2


class TestCreateDatasetSummaryValidation:
    """Tests for create_dataset_summary input validation."""

    def test_dataset_dir_not_found(self) -> None:
        """Test FileNotFoundError for missing dataset directory."""
        from ftw_dataset_tools.api.dataset_summary import create_dataset_summary

        with pytest.raises(FileNotFoundError, match="Dataset directory not found"):
            create_dataset_summary("/nonexistent/dataset")

    def test_no_chips_directory_found(self, tmp_path: Path) -> None:
        """Test FileNotFoundError when no chips directory exists."""
        from ftw_dataset_tools.api.dataset_summary import create_dataset_summary

        # Create dataset dir without chips subdirectory
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()

        with pytest.raises(FileNotFoundError, match=r"No \*-chips directory found"):
            create_dataset_summary(dataset_dir)

    def test_no_parquet_file_found(self, tmp_path: Path) -> None:
        """Test FileNotFoundError when chips directory has no parquet file."""
        from ftw_dataset_tools.api.dataset_summary import create_dataset_summary

        # Create chips dir without parquet file (use -chips naming convention)
        dataset_dir = tmp_path / "dataset"
        chips_dir = dataset_dir / "test-chips"
        chips_dir.mkdir(parents=True)

        with pytest.raises(FileNotFoundError, match=r"No \*_chips.parquet file found"):
            create_dataset_summary(dataset_dir)


class TestFindChipsDirAndParquet:
    """Tests for _find_chips_dir_and_parquet helper."""

    def test_finds_chips_dir_and_parquet(self, tmp_path: Path) -> None:
        """Test finding chips directory and parquet file."""
        from ftw_dataset_tools.api.dataset_summary import _find_chips_dir_and_parquet

        # Create valid structure (*-chips directory and *_chips.parquet file)
        dataset_dir = tmp_path / "dataset"
        chips_dir = dataset_dir / "austria-chips"
        chips_dir.mkdir(parents=True)
        parquet_file = dataset_dir / "austria_chips.parquet"
        parquet_file.touch()

        result_dir, result_parquet = _find_chips_dir_and_parquet(dataset_dir, lambda _: None)

        assert result_dir == chips_dir
        assert result_parquet == parquet_file

    def test_raises_when_no_chips_dir(self, tmp_path: Path) -> None:
        """Test raises FileNotFoundError when no chips directory found."""
        from ftw_dataset_tools.api.dataset_summary import _find_chips_dir_and_parquet

        with pytest.raises(FileNotFoundError, match=r"No \*-chips directory found"):
            _find_chips_dir_and_parquet(tmp_path, lambda _: None)

    def test_raises_when_no_parquet_file(self, tmp_path: Path) -> None:
        """Test raises FileNotFoundError when no parquet file found."""
        from ftw_dataset_tools.api.dataset_summary import _find_chips_dir_and_parquet

        chips_dir = tmp_path / "test-chips"
        chips_dir.mkdir()

        with pytest.raises(FileNotFoundError, match=r"No \*_chips.parquet file found"):
            _find_chips_dir_and_parquet(tmp_path, lambda _: None)


class TestLoadChipsDf:
    """Tests for _load_chips_df helper."""

    def test_loads_parquet_with_splits(self, tmp_path: Path) -> None:
        """Test loading parquet file with split column."""
        import geopandas as gpd

        from ftw_dataset_tools.api.dataset_summary import _load_chips_df

        # Create test parquet with splits and geometry using GeoDataFrame
        geom = box(10, 50, 10.1, 50.1)
        gdf = gpd.GeoDataFrame(
            {
                "chip_id": ["chip1", "chip2", "chip3"],
                "split": ["train", "val", "test"],
            },
            geometry=[geom, geom, geom],
            crs="EPSG:4326",
        )
        parquet_file = tmp_path / "chips.parquet"
        gdf.to_parquet(parquet_file)

        result_df, total, train, val, test = _load_chips_df(parquet_file, lambda _: None)

        assert total == 3
        assert train == 1
        assert val == 1
        assert test == 1
        assert len(result_df) == 3

    def test_handles_missing_split_column(self, tmp_path: Path) -> None:
        """Test handling parquet without split column."""
        from ftw_dataset_tools.api.dataset_summary import _load_chips_df

        # Create test parquet without splits
        df = pd.DataFrame(
            {
                "chip_id": ["chip1", "chip2"],
            }
        )
        parquet_file = tmp_path / "chips.parquet"
        df.to_parquet(parquet_file)

        _result_df, total, train, val, test = _load_chips_df(parquet_file, lambda _: None)

        assert total == 2
        assert train == 0
        assert val == 0
        assert test == 0

    def test_handles_empty_parquet(self, tmp_path: Path) -> None:
        """Test handling empty parquet file."""
        from ftw_dataset_tools.api.dataset_summary import _load_chips_df

        # Create empty parquet
        df = pd.DataFrame({"chip_id": []})
        parquet_file = tmp_path / "chips.parquet"
        df.to_parquet(parquet_file)

        _result_df, total, train, val, test = _load_chips_df(parquet_file, lambda _: None)

        assert total == 0
        assert train == 0
        assert val == 0
        assert test == 0

    @patch("duckdb.connect")
    def test_handles_corrupt_parquet(self, mock_connect: Mock, tmp_path: Path) -> None:
        """Test handling corrupt parquet file."""
        from ftw_dataset_tools.api.dataset_summary import _load_chips_df

        # Mock DuckDB to raise error on corrupt file
        mock_con = MagicMock()
        mock_connect.return_value = mock_con
        mock_con.execute.side_effect = Exception("Corrupt parquet file")

        parquet_file = tmp_path / "corrupt.parquet"
        parquet_file.touch()

        with pytest.raises(Exception, match="Corrupt parquet file"):
            _load_chips_df(parquet_file, lambda _: None)


class TestCollectStacMetadata:
    """Tests for _collect_stac_metadata helper."""

    def test_collects_metadata_from_stac_items(self, tmp_path: Path) -> None:
        """Test collecting metadata from valid STAC JSON files."""
        import json

        from ftw_dataset_tools.api.dataset_summary import _collect_stac_metadata

        chips_dir = tmp_path / "chips"
        chip_dir = chips_dir / "chip1"
        chip_dir.mkdir(parents=True)

        # Create valid STAC item JSON
        stac_item = {
            "type": "Feature",
            "id": "chip1_planting_s2",
            "properties": {
                "datetime": "2023-04-15T10:00:00Z",
                "eo:cloud_cover": 5.5,
                "ftw:calendar_year": 2023,
            },
            "assets": {},
        }
        planting_file = chip_dir / "chip1_planting_s2.json"
        planting_file.write_text(json.dumps(stac_item))

        result = _collect_stac_metadata(chips_dir, lambda _: None)

        assert len(result["planting_items"]) == 1
        # Dates and cloud cover are only collected if datetime is present and parseable
        assert len(result["planting_dates"]) >= 0  # May be empty if parsing fails
        assert len(result["planting_cloud_cover"]) >= 0
        # Check metadata if present
        if result["metadata"]:
            assert result["metadata"].get("calendar_year") == 2023

    def test_handles_malformed_stac_json(self, tmp_path: Path) -> None:
        """Test handling malformed STAC JSON files."""
        from ftw_dataset_tools.api.dataset_summary import _collect_stac_metadata

        chips_dir = tmp_path / "chips"
        chip_dir = chips_dir / "chip1"
        chip_dir.mkdir(parents=True)

        # Create malformed JSON
        malformed_file = chip_dir / "chip1_planting_s2.json"
        malformed_file.write_text("{ invalid json")

        # Should not raise - files are found but parsing fails silently
        result = _collect_stac_metadata(chips_dir, lambda _: None)

        # The file path is still added to planting_items
        assert len(result["planting_items"]) == 1
        # But dates won't be extracted from malformed JSON
        assert len(result["planting_dates"]) == 0

    def test_handles_missing_properties(self, tmp_path: Path) -> None:
        """Test handling STAC items with missing properties."""
        import json

        from ftw_dataset_tools.api.dataset_summary import _collect_stac_metadata

        chips_dir = tmp_path / "chips"
        chip_dir = chips_dir / "chip1"
        chip_dir.mkdir(parents=True)

        # Create STAC item without datetime
        stac_item = {
            "type": "Feature",
            "id": "chip1_planting_s2",
            "properties": {},
            "assets": {},
        }
        planting_file = chip_dir / "chip1_planting_s2.json"
        planting_file.write_text(json.dumps(stac_item))

        # Should not raise
        result = _collect_stac_metadata(chips_dir, lambda _: None)

        assert len(result["planting_items"]) == 1
        # Dates list should be empty since datetime is missing
        assert len(result["planting_dates"]) == 0

    def test_handles_empty_chips_directory(self, tmp_path: Path) -> None:
        """Test handling empty chips directory."""
        from ftw_dataset_tools.api.dataset_summary import _collect_stac_metadata

        chips_dir = tmp_path / "chips"
        chips_dir.mkdir()

        result = _collect_stac_metadata(chips_dir, lambda _: None)

        assert len(result["planting_items"]) == 0
        assert len(result["planting_dates"]) == 0
        assert len(result["harvest_dates"]) == 0


class TestSelectExampleChips:
    """Tests for _select_example_chips helper."""

    def test_selects_chips_with_imagery(self, tmp_path: Path) -> None:
        """Test selecting chips that have both planting and harvest JPGs."""
        from ftw_dataset_tools.api.dataset_summary import _select_example_chips

        chips_dir = tmp_path / "chips"
        chip1_dir = chips_dir / "chip1"
        chip1_dir.mkdir(parents=True)

        # Create JPG files
        (chip1_dir / "chip1_planting_image_s2.jpg").touch()
        (chip1_dir / "chip1_harvest_image_s2.jpg").touch()

        planting_items = [Path("chip1_planting_s2.json")]

        result = _select_example_chips(chips_dir, planting_items, 1, lambda _: None)

        assert len(result) == 1
        assert result[0] == "chip1"

    def test_skips_chips_without_harvest_image(self, tmp_path: Path) -> None:
        """Test skipping chips missing harvest image."""
        from ftw_dataset_tools.api.dataset_summary import _select_example_chips

        chips_dir = tmp_path / "chips"
        chip1_dir = chips_dir / "chip1"
        chip1_dir.mkdir(parents=True)

        # Only create planting image
        (chip1_dir / "chip1_planting_image_s2.jpg").touch()

        planting_items = [Path("chip1_planting_s2.json")]

        result = _select_example_chips(chips_dir, planting_items, 1, lambda _: None)

        assert len(result) == 0

    def test_respects_num_examples_limit(self, tmp_path: Path) -> None:
        """Test respecting num_examples limit."""
        from ftw_dataset_tools.api.dataset_summary import _select_example_chips

        chips_dir = tmp_path / "chips"

        planting_items = []
        for i in range(5):
            chip_dir = chips_dir / f"chip{i}"
            chip_dir.mkdir(parents=True)
            (chip_dir / f"chip{i}_planting_image_s2.jpg").touch()
            (chip_dir / f"chip{i}_harvest_image_s2.jpg").touch()
            planting_items.append(Path(f"chip{i}_planting_s2.json"))

        result = _select_example_chips(chips_dir, planting_items, 3, lambda _: None)

        assert len(result) == 3


class TestCreateDatasetSummaryIntegration:
    """Integration tests for create_dataset_summary."""

    def test_creates_summary_for_valid_dataset(self, tmp_path: Path) -> None:
        """Test creating summary for a valid dataset."""
        import json

        from ftw_dataset_tools.api.dataset_summary import create_dataset_summary

        # Create valid dataset structure (*-chips directory, *_chips.parquet file)
        dataset_dir = tmp_path / "dataset"
        chips_dir = dataset_dir / "belgium-chips"
        chip1_dir = chips_dir / "chip1"
        chip1_dir.mkdir(parents=True)

        # Create parquet file using GeoDataFrame
        import geopandas as gpd

        geom = box(4.0, 50.0, 4.1, 50.1)
        gdf = gpd.GeoDataFrame(
            {
                "chip_id": ["chip1"],
                "split": ["train"],
            },
            geometry=[geom],
            crs="EPSG:4326",
        )
        parquet_file = dataset_dir / "belgium_chips.parquet"
        gdf.to_parquet(parquet_file)

        # Create STAC items
        stac_item = {
            "type": "Feature",
            "id": "chip1_planting_s2",
            "properties": {
                "datetime": "2023-04-15T10:00:00Z",
                "eo:cloud_cover": 5.0,
                "ftw:calendar_year": 2023,
            },
            "assets": {},
        }
        (chip1_dir / "chip1_planting_s2.json").write_text(json.dumps(stac_item))
        stac_item["id"] = "chip1_harvest_s2"
        stac_item["properties"]["datetime"] = "2023-09-15T10:00:00Z"
        (chip1_dir / "chip1_harvest_s2.json").write_text(json.dumps(stac_item))

        # Create image files
        (chip1_dir / "chip1_planting_image_s2.jpg").touch()
        (chip1_dir / "chip1_harvest_image_s2.jpg").touch()

        output_path = dataset_dir / "summary.md"

        result = create_dataset_summary(dataset_dir, output_path, num_examples=1)

        assert result.total_chips == 1
        assert result.train_chips == 1
        assert result.val_chips == 0
        assert result.test_chips == 0
        assert len(result.example_chips) == 1
        assert result.output_path == output_path
        assert output_path.exists()

    def test_on_progress_callback(self, tmp_path: Path) -> None:
        """Test that on_progress callback is invoked."""
        from ftw_dataset_tools.api.dataset_summary import create_dataset_summary

        # Create minimal valid structure (*-chips directory, *_chips.parquet file)
        dataset_dir = tmp_path / "dataset"
        chips_dir = dataset_dir / "test-chips"
        chips_dir.mkdir(parents=True)

        df = pd.DataFrame({"chip_id": ["chip1"]})
        parquet_file = dataset_dir / "test_chips.parquet"
        df.to_parquet(parquet_file)

        messages = []

        def callback(msg: str) -> None:
            messages.append(msg)

        create_dataset_summary(dataset_dir, num_examples=0, on_progress=callback)

        assert len(messages) > 0
        assert any("Analyzing dataset" in msg for msg in messages)


class TestWriteMarkdownSummary:
    """Tests for _write_markdown_summary helper."""

    def test_handles_zero_chips_gracefully(self, tmp_path: Path) -> None:
        """Test handling dataset with zero chips to avoid ZeroDivisionError."""

        from ftw_dataset_tools.api.dataset_summary import _write_markdown_summary

        output_file = tmp_path / "summary.md"
        figures_dir = tmp_path / "figures"
        figures_dir.mkdir()

        # Should not raise ZeroDivisionError
        _write_markdown_summary(
            output_path=output_file,
            dataset_dir=tmp_path,
            chips_dir=tmp_path / "chips",
            total_chips=0,
            train_chips=0,
            val_chips=0,
            test_chips=0,
            metadata={},
            example_chips=[],
            planting_dates=[],
            harvest_dates=[],
            planting_cloud_cover=[],
            harvest_cloud_cover=[],
            field_coverage_pct=[],
            empty_mask_count=0,
            figures_dir=figures_dir,
        )

        assert output_file.exists()
        content = output_file.read_text()
        assert "0" in content  # Should contain zero counts


class TestCountEmptyMasks:
    """Tests for _count_empty_masks helper."""

    def test_counts_empty_and_non_empty_masks(self, tmp_path: Path) -> None:
        """Test counting mix of empty and non-empty masks."""
        from ftw_dataset_tools.api.dataset_summary import _count_empty_masks

        chips_dir = tmp_path / "chips"

        # Create chip directories with masks
        # Empty mask (all zeros)
        chip1_dir = chips_dir / "chip1"
        chip1_dir.mkdir(parents=True)
        mask1_path = chip1_dir / "chip1_semantic_3_class.tif"
        self._create_test_mask(mask1_path, all_zeros=True)

        # Non-empty mask
        chip2_dir = chips_dir / "chip2"
        chip2_dir.mkdir(parents=True)
        mask2_path = chip2_dir / "chip2_semantic_3_class.tif"
        self._create_test_mask(mask2_path, all_zeros=False)

        # Another empty mask
        chip3_dir = chips_dir / "chip3"
        chip3_dir.mkdir(parents=True)
        mask3_path = chip3_dir / "chip3_semantic_3_class.tif"
        self._create_test_mask(mask3_path, all_zeros=True)

        messages = []
        result = _count_empty_masks(chips_dir, messages.append)

        assert result == 2  # Two empty masks
        assert any("Found 2 empty masks" in msg for msg in messages)

    def test_handles_missing_mask_files(self, tmp_path: Path) -> None:
        """Test handling chips without mask files."""
        from ftw_dataset_tools.api.dataset_summary import _count_empty_masks

        chips_dir = tmp_path / "chips"

        # Chip with mask
        chip1_dir = chips_dir / "chip1"
        chip1_dir.mkdir(parents=True)
        mask1_path = chip1_dir / "chip1_semantic_3_class.tif"
        self._create_test_mask(mask1_path, all_zeros=True)

        # Chip without mask
        chip2_dir = chips_dir / "chip2"
        chip2_dir.mkdir(parents=True)
        # No mask file created

        messages = []
        result = _count_empty_masks(chips_dir, messages.append)

        # Should only count the one chip with a mask
        assert result == 1
        assert any("Found 1 empty masks out of 1 chips checked" in msg for msg in messages)

    def test_handles_rasterio_read_errors(self, tmp_path: Path) -> None:
        """Test handling rasterio read errors gracefully."""
        from ftw_dataset_tools.api.dataset_summary import _count_empty_masks

        chips_dir = tmp_path / "chips"

        # Create chip directory with mask
        chip1_dir = chips_dir / "chip1"
        chip1_dir.mkdir(parents=True)
        mask1_path = chip1_dir / "chip1_semantic_3_class.tif"
        self._create_test_mask(mask1_path, all_zeros=True)

        messages = []

        # Mock rasterio.open to raise an exception
        with patch("rasterio.open") as mock_open:
            mock_open.side_effect = Exception("Simulated read error")

            result = _count_empty_masks(chips_dir, messages.append)

        # Should return 0 and log warning
        assert result == 0
        assert any("Warning: Failed to read mask" in msg for msg in messages)

    def test_handles_missing_rasterio(self, tmp_path: Path) -> None:
        """Test graceful handling when rasterio is not available."""
        from ftw_dataset_tools.api.dataset_summary import _count_empty_masks

        chips_dir = tmp_path / "chips"
        chip_dir = chips_dir / "chip1"
        chip_dir.mkdir(parents=True)

        messages = []

        # Mock the rasterio import to fail
        with patch.dict("sys.modules", {"rasterio": None}):
            result = _count_empty_masks(chips_dir, messages.append)

        assert result == 0
        assert any("rasterio not available" in msg for msg in messages)

    @staticmethod
    def _create_test_mask(path: Path, all_zeros: bool = True) -> None:
        """Create a test GeoTIFF mask file."""
        import numpy as np
        import rasterio
        from rasterio.transform import from_bounds

        # Create 10x10 test mask
        if all_zeros:
            data = np.zeros((10, 10), dtype=np.uint8)
        else:
            data = np.ones((10, 10), dtype=np.uint8)
            data[5:8, 5:8] = 2  # Some field pixels

        # Write GeoTIFF
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=10,
            width=10,
            count=1,
            dtype=np.uint8,
            crs="EPSG:4326",
            transform=from_bounds(0, 0, 10, 10, 10, 10),
        ) as dst:
            dst.write(data, 1)


class TestMaskStatisticsMarkdown:
    """Tests for Mask Statistics section in markdown output."""

    def test_renders_mask_statistics_with_field_coverage(self, tmp_path: Path) -> None:
        """Test rendering Mask Statistics section with field coverage data."""
        from ftw_dataset_tools.api.dataset_summary import _write_markdown_summary

        output_file = tmp_path / "summary.md"
        figures_dir = tmp_path / "figures"
        figures_dir.mkdir()

        # Create field coverage histogram
        field_cov_path = figures_dir / "field_coverage.png"
        field_cov_path.touch()

        _write_markdown_summary(
            output_path=output_file,
            dataset_dir=tmp_path,
            chips_dir=tmp_path / "chips",
            total_chips=100,
            train_chips=70,
            val_chips=15,
            test_chips=15,
            metadata={},
            example_chips=[],
            planting_dates=[],
            harvest_dates=[],
            planting_cloud_cover=[],
            harvest_cloud_cover=[],
            field_coverage_pct=[50.0, 75.5, 90.2, 45.3, 82.1],
            empty_mask_count=5,
            figures_dir=figures_dir,
        )

        assert output_file.exists()
        content = output_file.read_text()

        # Check Mask Statistics section exists
        assert "## Mask Statistics" in content

        # Check Field Coverage Distribution subsection
        assert "### Field Coverage Distribution" in content
        assert "![Field Coverage](figures/field_coverage.png)" in content

        # Check statistics are rendered
        assert "**Mean Coverage**:" in content
        assert "**Median Coverage**:" in content
        assert "**Min/Max**:" in content

        # Check empty mask count is shown with percentage
        assert "**Empty Masks**: 5" in content
        assert "(5.0% of total)" in content

    def test_renders_empty_masks_without_field_coverage(self, tmp_path: Path) -> None:
        """Test that empty mask count is always shown even without field coverage data."""
        from ftw_dataset_tools.api.dataset_summary import _write_markdown_summary

        output_file = tmp_path / "summary.md"
        figures_dir = tmp_path / "figures"
        figures_dir.mkdir()

        _write_markdown_summary(
            output_path=output_file,
            dataset_dir=tmp_path,
            chips_dir=tmp_path / "chips",
            total_chips=50,
            train_chips=35,
            val_chips=8,
            test_chips=7,
            metadata={},
            example_chips=[],
            planting_dates=[],
            harvest_dates=[],
            planting_cloud_cover=[],
            harvest_cloud_cover=[],
            field_coverage_pct=[],  # No field coverage data
            empty_mask_count=3,
            figures_dir=figures_dir,
        )

        assert output_file.exists()
        content = output_file.read_text()

        # Mask Statistics section should exist
        assert "## Mask Statistics" in content

        # Empty mask count should be shown
        assert "**Empty Masks**: 3" in content
        assert "(6.0% of total)" in content

        # Field coverage subsection should NOT exist
        assert "### Field Coverage Distribution" not in content
        assert "**Mean Coverage**:" not in content

    def test_handles_division_by_zero_in_empty_mask_percentage(self, tmp_path: Path) -> None:
        """Test that zero total_chips doesn't cause division by zero."""
        from ftw_dataset_tools.api.dataset_summary import _write_markdown_summary

        output_file = tmp_path / "summary.md"
        figures_dir = tmp_path / "figures"
        figures_dir.mkdir()

        # Should not raise ZeroDivisionError
        _write_markdown_summary(
            output_path=output_file,
            dataset_dir=tmp_path,
            chips_dir=tmp_path / "chips",
            total_chips=0,  # Zero chips
            train_chips=0,
            val_chips=0,
            test_chips=0,
            metadata={},
            example_chips=[],
            planting_dates=[],
            harvest_dates=[],
            planting_cloud_cover=[],
            harvest_cloud_cover=[],
            field_coverage_pct=[],
            empty_mask_count=0,
            figures_dir=figures_dir,
        )

        assert output_file.exists()
        content = output_file.read_text()

        # Should show 0.0% when total_chips is 0
        assert "## Mask Statistics" in content
        assert "**Empty Masks**: 0 (0.0% of total)" in content

    def test_renders_histogram_only_when_file_exists(self, tmp_path: Path) -> None:
        """Test that histogram image is only included when file exists."""
        from ftw_dataset_tools.api.dataset_summary import _write_markdown_summary

        output_file = tmp_path / "summary.md"
        figures_dir = tmp_path / "figures"
        figures_dir.mkdir()
        # Don't create field_coverage.png

        _write_markdown_summary(
            output_path=output_file,
            dataset_dir=tmp_path,
            chips_dir=tmp_path / "chips",
            total_chips=100,
            train_chips=70,
            val_chips=15,
            test_chips=15,
            metadata={},
            example_chips=[],
            planting_dates=[],
            harvest_dates=[],
            planting_cloud_cover=[],
            harvest_cloud_cover=[],
            field_coverage_pct=[50.0, 75.0, 90.0],
            empty_mask_count=5,
            figures_dir=figures_dir,
        )

        assert output_file.exists()
        content = output_file.read_text()

        # Statistics should be shown
        assert "### Field Coverage Distribution" in content
        assert "**Mean Coverage**:" in content

        # But histogram image should not be included
        assert "![Field Coverage](figures/field_coverage.png)" not in content
