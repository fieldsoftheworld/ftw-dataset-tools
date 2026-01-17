"""Tests for the dataset API module."""

from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import box


class TestCreateDatasetResult:
    """Tests for CreateDatasetResult dataclass."""

    def test_dataclass_fields(self) -> None:
        """Test CreateDatasetResult has expected fields."""
        from ftw_dataset_tools.api.dataset import CreateDatasetResult

        result = CreateDatasetResult(
            output_dir=Path("/tmp/output"),
            field_dataset="test_dataset",
            fields_file=Path("/tmp/fields.parquet"),
            chips_file=Path("/tmp/chips.parquet"),
            boundary_lines_file=Path("/tmp/boundary_lines.parquet"),
        )
        assert result.output_dir == Path("/tmp/output")
        assert result.field_dataset == "test_dataset"

    def test_total_masks_created(self) -> None:
        """Test total_masks_created property sums all mask types."""
        from ftw_dataset_tools.api.dataset import CreateDatasetResult
        from ftw_dataset_tools.api.masks import CreateMasksResult, MaskResult

        result = CreateDatasetResult(
            output_dir=Path("/tmp"),
            field_dataset="test",
            fields_file=Path("/tmp/fields.parquet"),
            chips_file=Path("/tmp/chips.parquet"),
            boundary_lines_file=Path("/tmp/boundary_lines.parquet"),
            masks_results={
                "instance": CreateMasksResult(
                    masks_created=[
                        MaskResult("a", Path("a.tif"), 512, 512),
                    ],
                    masks_skipped=[],
                    field_dataset="test",
                ),
                "semantic_2class": CreateMasksResult(
                    masks_created=[
                        MaskResult("a", Path("a.tif"), 512, 512),
                        MaskResult("b", Path("b.tif"), 512, 512),
                    ],
                    masks_skipped=[],
                    field_dataset="test",
                ),
            },
        )
        assert result.total_masks_created == 3


class TestCreateDatasetInputValidation:
    """Tests for create_dataset input validation."""

    def test_fields_file_not_found(self) -> None:
        """Test FileNotFoundError for missing fields file."""
        from ftw_dataset_tools.api.dataset import create_dataset

        with pytest.raises(FileNotFoundError, match="Fields file not found"):
            create_dataset("/nonexistent/fields.parquet")

    def test_year_required_without_datetime_column(self, tmp_path: Path) -> None:
        """Test ValueError when year not provided and no datetime column."""
        from ftw_dataset_tools.api.dataset import create_dataset

        # Create fields file without determination_datetime
        gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[box(10, 50, 10.01, 50.01)], crs="EPSG:4326")
        fields_file = tmp_path / "fields.parquet"
        gdf.to_parquet(fields_file)

        with pytest.raises(ValueError, match="Cannot determine temporal extent"):
            create_dataset(fields_file, year=None)

    def test_skip_reproject_error_non_4326(self, tmp_path: Path) -> None:
        """Test ValueError when skip_reproject=True with non-4326 input."""
        from ftw_dataset_tools.api.dataset import create_dataset

        # Create fields file in EPSG:3035
        gdf = gpd.GeoDataFrame(
            {"id": [1]},
            geometry=[box(5150000, 3540000, 5160000, 3550000)],
            crs="EPSG:3035",
        )
        fields_file = tmp_path / "fields_3035.parquet"
        gdf.to_parquet(fields_file)

        with pytest.raises(ValueError, match="EPSG:4326 is required"):
            create_dataset(fields_file, year=2023, skip_reproject=True)
