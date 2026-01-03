"""Tests for the splits API."""

from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import Point

from ftw_dataset_tools.api.splits import assign_splits, validate_split_percents


class TestValidateSplitPercents:
    """Tests for validate_split_percents function."""

    def test_valid_split_percents(self) -> None:
        """Test that valid split percentages are accepted."""
        result = validate_split_percents((80, 10, 10))
        assert result == (80, 10, 10)

    def test_invalid_sum_not_100(self) -> None:
        """Test that values not summing to 100 raise error."""
        with pytest.raises(ValueError, match="must sum to 100"):
            validate_split_percents((80, 10, 5))


class TestAssignSplits:
    """Tests for assign_splits function."""

    def test_invalid_split_type(self) -> None:
        """Test that invalid split type raises error immediately."""
        with pytest.raises(ValueError, match=r"Unsupported split_type.*Must be one of"):
            assign_splits(
                chips_file="/any/path.parquet",  # File doesn't need to exist - validation is first
                split_type="invalid-type",
            )

    def test_file_not_found(self) -> None:
        """Test that FileNotFoundError is raised for missing chips file."""
        with pytest.raises(FileNotFoundError, match="Chips file not found"):
            assign_splits(
                chips_file="/nonexistent/chips.parquet",
                split_type="random-uniform",
            )

    def test_random_uniform_split_basic(self, tmp_path: Path) -> None:
        """Test basic random-uniform split assignment."""
        # Create test chips
        chips_file = tmp_path / "chips.parquet"
        n_chips = 100
        chip_ids = [f"ftw-36NXF{i:04d}" for i in range(n_chips)]
        gdf = gpd.GeoDataFrame(
            {"id": chip_ids, "geometry": [Point(i, i) for i in range(n_chips)]},
            crs="EPSG:4326",
        )
        gdf.to_parquet(chips_file)

        # Assign splits
        result = assign_splits(
            chips_file=chips_file,
            split_type="random-uniform",
            split_percents=(80, 10, 10),
            random_seed=42,
        )

        # Verify result
        assert result.total_chips == n_chips
        assert result.train_count == 80
        assert result.val_count == 10
        assert result.test_count == 10
        assert result.train_count + result.val_count + result.test_count == n_chips

        # Verify file was updated
        updated_gdf = gpd.read_parquet(chips_file)
        assert "split" in updated_gdf.columns
        assert set(updated_gdf["split"]) == {"train", "val", "test"}

    def test_block3x3_split_basic(self, tmp_path: Path) -> None:
        """Test basic block3x3 split assignment with spatial coherence."""
        chips_file = tmp_path / "chips.parquet"

        # Create chips in a 9x9 grid (should create 3x3 = 9 blocks)
        chip_ids = []
        for easting in range(9):
            for northing in range(9):
                chip_ids.append(f"ftw-36NXF{easting:02d}{northing:02d}")

        n_chips = len(chip_ids)
        gdf = gpd.GeoDataFrame(
            {"id": chip_ids, "geometry": [Point(i, i) for i in range(n_chips)]},
            crs="EPSG:4326",
        )
        gdf.to_parquet(chips_file)

        result = assign_splits(
            chips_file=chips_file,
            split_type="block3x3",
            split_percents=(70, 20, 10),
            random_seed=42,
        )

        # Verify counts
        assert result.total_chips == 81
        assert result.train_count + result.val_count + result.test_count == 81

        # Verify spatial coherence - chips in same 3x3 block should have same split
        updated_gdf = gpd.read_parquet(chips_file)
        updated_gdf["easting"] = updated_gdf["id"].str[-4:-2].astype(int)
        updated_gdf["northing"] = updated_gdf["id"].str[-2:].astype(int)
        updated_gdf["block_east"] = updated_gdf["easting"] // 3
        updated_gdf["block_north"] = updated_gdf["northing"] // 3

        # Check that all chips in each block have the same split
        for block_east in range(3):
            for block_north in range(3):
                block_mask = (
                    (updated_gdf["block_east"] == block_east)
                    & (updated_gdf["block_north"] == block_north)
                )
                block_splits = updated_gdf[block_mask]["split"]
                assert len(block_splits.unique()) == 1, f"Block ({block_east}, {block_north}) has mixed splits"

    def test_block3x3_invalid_chip_id_format(self, tmp_path: Path) -> None:
        """Test that malformed chip IDs raise an error in block3x3."""
        chips_file = tmp_path / "chips.parquet"

        # Create chips with invalid ID format (too short)
        chip_ids = ["short-id", "another-bad"]
        gdf = gpd.GeoDataFrame(
            {"id": chip_ids, "geometry": [Point(0, 0), Point(1, 1)]},
            crs="EPSG:4326",
        )
        gdf.to_parquet(chips_file)

        with pytest.raises(ValueError, match=r"Invalid chip ID format.*at least 13 characters"):
            assign_splits(
                chips_file=chips_file,
                split_type="block3x3",
                split_percents=(80, 10, 10),
                random_seed=42,
            )

    def test_missing_id_column(self, tmp_path: Path) -> None:
        """Test that missing 'id' column raises an error."""
        chips_file = tmp_path / "chips.parquet"

        # Create chips without 'id' column
        gdf = gpd.GeoDataFrame(
            {"name": ["chip1", "chip2"], "geometry": [Point(0, 0), Point(1, 1)]},
            crs="EPSG:4326",
        )
        gdf.to_parquet(chips_file)

        with pytest.raises(ValueError, match="Chips file must contain an 'id' column"):
            assign_splits(
                chips_file=chips_file,
                split_type="random-uniform",
                split_percents=(80, 10, 10),
                random_seed=42,
            )
