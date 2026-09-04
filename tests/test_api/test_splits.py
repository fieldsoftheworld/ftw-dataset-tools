"""Tests for the splits API."""

from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, box

from ftw_dataset_tools.api.splits import _infer_grid_step, assign_splits, validate_split_percents


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


class TestInferGridStep:
    """Tests for _infer_grid_step function."""

    def test_dense_coverage(self) -> None:
        """Adjacent cells present: step is the common gap."""
        assert _infer_grid_step(pd.Series([0, 2, 4, 6])) == 2

    def test_sparse_coverage_uses_gcd_of_gaps(self) -> None:
        """No adjacent pair present: GCD of gaps still recovers the step."""
        assert _infer_grid_step(pd.Series([0, 6, 10])) == 2

    def test_single_value_is_unobservable(self) -> None:
        """Fewer than two distinct values: spacing cannot be observed."""
        assert _infer_grid_step(pd.Series([10, 10])) == 0


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

    def test_empty_chips_file(self, tmp_path: Path) -> None:
        """Test that empty chips file raises an error."""
        chips_file = tmp_path / "chips.parquet"

        # Create empty GeoDataFrame
        gdf = gpd.GeoDataFrame(
            {"id": [], "geometry": []},
            crs="EPSG:4326",
        )
        gdf.to_parquet(chips_file)

        with pytest.raises(ValueError, match=r"Chips file is empty.*Cannot assign splits"):
            assign_splits(
                chips_file=chips_file,
                split_type="random-uniform",
                split_percents=(80, 10, 10),
                random_seed=42,
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
                block_mask = (updated_gdf["block_east"] == block_east) & (
                    updated_gdf["block_north"] == block_north
                )
                block_splits = updated_gdf[block_mask]["split"]
                assert len(block_splits.unique()) == 1, (
                    f"Block ({block_east}, {block_north}) has mixed splits"
                )

    def test_block3x3_split_with_km_size_spacing(self, tmp_path: Path) -> None:
        """Test block3x3 with real FTW grid coordinate spacing (km_size=2).

        Real FTW grid IDs encode easting/northing as multiples of km_size
        (e.g. 0, 2, 4, 6...), not sequential integers. Blocks must group every
        3 grid cells regardless of the coordinate step size, or they end up
        lopsided (see issue #31).
        """
        chips_file = tmp_path / "chips.parquet"

        # 18x18 km grid at km_size=2 -> coordinates 0,2,4,...,34 (18 steps -> 6 blocks)
        step = 2
        n_steps = 18
        chip_ids = []
        for i in range(n_steps):
            for j in range(n_steps):
                chip_ids.append(f"ftw-36NXF{i * step:02d}{j * step:02d}")

        n_chips = len(chip_ids)
        gdf = gpd.GeoDataFrame(
            {"id": chip_ids, "geometry": [Point(k, k) for k in range(n_chips)]},
            crs="EPSG:4326",
        )
        gdf.to_parquet(chips_file)

        assign_splits(
            chips_file=chips_file,
            split_type="block3x3",
            split_percents=(70, 20, 10),
            random_seed=42,
        )

        updated_gdf = gpd.read_parquet(chips_file)
        updated_gdf["easting"] = updated_gdf["id"].str[-4:-2].astype(int)
        updated_gdf["northing"] = updated_gdf["id"].str[-2:].astype(int)
        updated_gdf["block_east"] = (updated_gdf["easting"] // step) // 3
        updated_gdf["block_north"] = (updated_gdf["northing"] // step) // 3

        n_blocks_per_side = n_steps // 3
        for block_east in range(n_blocks_per_side):
            for block_north in range(n_blocks_per_side):
                block_mask = (updated_gdf["block_east"] == block_east) & (
                    updated_gdf["block_north"] == block_north
                )
                block_chips = updated_gdf[block_mask]
                # Every full block should contain exactly 9 chips (3x3)
                assert len(block_chips) == 9, (
                    f"Block ({block_east}, {block_north}) has {len(block_chips)} chips, expected 9"
                )
                assert len(block_chips["split"].unique()) == 1, (
                    f"Block ({block_east}, {block_north}) has mixed splits"
                )

    def test_block3x3_single_column_uses_other_axis_step(self, tmp_path: Path) -> None:
        """Test block3x3 when one axis has a single distinct coordinate.

        A north-south strip of chips gives the easting axis only one unique
        value, so its spacing cannot be observed there and the step inferred
        from the northing axis is shared. Blocking must still group the
        populated axis correctly.
        """
        chips_file = tmp_path / "chips.parquet"

        # Single easting column; northings spaced by 2 -> 9 chips in 3 blocks
        step = 2
        n_steps = 9
        fixed_easting = 10
        chip_ids = [f"ftw-36NXF{fixed_easting:02d}{j * step:02d}" for j in range(n_steps)]

        gdf = gpd.GeoDataFrame(
            {"id": chip_ids, "geometry": [Point(k, k) for k in range(len(chip_ids))]},
            crs="EPSG:4326",
        )
        gdf.to_parquet(chips_file)

        result = assign_splits(
            chips_file=chips_file,
            split_type="block3x3",
            split_percents=(70, 20, 10),
            random_seed=42,
        )

        assert result.total_chips == n_steps

        updated_gdf = gpd.read_parquet(chips_file)
        updated_gdf["northing"] = updated_gdf["id"].str[-2:].astype(int)
        updated_gdf["block_north"] = (updated_gdf["northing"] // step) // 3

        # 9 chips spaced by 2 along one axis -> 3 blocks of 3, each internally consistent
        assert updated_gdf["block_north"].nunique() == 3
        for block_north, block_chips in updated_gdf.groupby("block_north"):
            assert len(block_chips) == 3, (
                f"Block {block_north} has {len(block_chips)} chips, expected 3"
            )
            assert len(block_chips["split"].unique()) == 1, f"Block {block_north} has mixed splits"

    def test_block3x3_sparse_coverage_infers_step_from_gcd(self, tmp_path: Path) -> None:
        """Test step inference when no two adjacent cells are populated.

        With sparse coverage the smallest gap between populated coordinates can
        be a multiple of the true km_size (eastings 00, 06, 10 have gaps 6 and
        4, but the grid step is 2). The GCD of the gaps recovers the true step,
        so chips 6+ cells apart must not collapse into one block.
        """
        chips_file = tmp_path / "chips.parquet"

        # km_size=2 grid, populated eastings 0, 6, 10 (no adjacent pair);
        # northings dense 0..16. Cell coords: eastings 0, 3, 5 -> blocks 0, 1, 1.
        step = 2
        eastings = [0, 6, 10]
        northings = [j * step for j in range(9)]
        chip_ids = [f"ftw-36NXF{e:02d}{n:02d}" for e in eastings for n in northings]

        gdf = gpd.GeoDataFrame(
            {"id": chip_ids, "geometry": [Point(k, k) for k in range(len(chip_ids))]},
            crs="EPSG:4326",
        )
        gdf.to_parquet(chips_file)

        assign_splits(
            chips_file=chips_file,
            split_type="block3x3",
            split_percents=(70, 20, 10),
            random_seed=42,
        )

        updated_gdf = gpd.read_parquet(chips_file)
        updated_gdf["easting"] = updated_gdf["id"].str[-4:-2].astype(int)
        updated_gdf["northing"] = updated_gdf["id"].str[-2:].astype(int)
        updated_gdf["block_east"] = (updated_gdf["easting"] // step) // 3
        updated_gdf["block_north"] = (updated_gdf["northing"] // step) // 3

        # Easting 0 is in block 0; eastings 6 and 10 are both in block 1.
        # A min-gap heuristic would infer step 4 and merge all three into
        # blocks {0, 0, 0} or split them irregularly.
        assert set(updated_gdf["block_east"].unique()) == {0, 1}
        for (block_east, block_north), block_chips in updated_gdf.groupby(
            ["block_east", "block_north"]
        ):
            assert len(block_chips["split"].unique()) == 1, (
                f"Block ({block_east}, {block_north}) has mixed splits"
            )

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

    def test_predefined_requires_fields_file(self, tmp_path: Path) -> None:
        """Test predefined split type requires fields_file."""
        chips_file = tmp_path / "chips.parquet"

        gdf = gpd.GeoDataFrame(
            {"id": ["ftw-36NXF0000"], "geometry": [box(0, 0, 1, 1)]},
            crs="EPSG:4326",
        )
        gdf.to_parquet(chips_file)

        with pytest.raises(ValueError, match="fields_file is required"):
            assign_splits(
                chips_file=chips_file,
                split_type="predefined",
                split_percents=(80, 10, 10),
                random_seed=42,
            )

    def test_predefined_missing_split_column(self, tmp_path: Path) -> None:
        """Test predefined split raises when fields file lacks split column."""
        chips_file = tmp_path / "chips.parquet"
        fields_file = tmp_path / "fields.parquet"

        chips_gdf = gpd.GeoDataFrame(
            {"id": ["ftw-36NXF0000"], "geometry": [box(0, 0, 1, 1)]},
            crs="EPSG:4326",
        )
        chips_gdf.to_parquet(chips_file)

        fields_gdf = gpd.GeoDataFrame(
            {"name": ["field1"], "geometry": [box(0.1, 0.1, 0.2, 0.2)]},
            crs="EPSG:4326",
        )
        fields_gdf.to_parquet(fields_file)

        with pytest.raises(ValueError, match="must contain a 'split' column"):
            assign_splits(
                chips_file=chips_file,
                split_type="predefined",
                split_percents=(80, 10, 10),
                random_seed=42,
                fields_file=fields_file,
            )

    def test_predefined_invalid_split_value(self, tmp_path: Path) -> None:
        """Test predefined split raises on invalid split values."""
        chips_file = tmp_path / "chips.parquet"
        fields_file = tmp_path / "fields.parquet"

        chips_gdf = gpd.GeoDataFrame(
            {"id": ["ftw-36NXF0000"], "geometry": [box(0, 0, 1, 1)]},
            crs="EPSG:4326",
        )
        chips_gdf.to_parquet(chips_file)

        fields_gdf = gpd.GeoDataFrame(
            {"split": ["not-a-split"], "geometry": [box(0.1, 0.1, 0.2, 0.2)]},
            crs="EPSG:4326",
        )
        fields_gdf.to_parquet(fields_file)

        with pytest.raises(ValueError, match="Invalid split values"):
            assign_splits(
                chips_file=chips_file,
                split_type="predefined",
                split_percents=(80, 10, 10),
                random_seed=42,
                fields_file=fields_file,
            )

    def test_predefined_majority_and_tiebreak(self, tmp_path: Path) -> None:
        """Test predefined split assigns majority and applies deterministic tie-break."""
        chips_file = tmp_path / "chips.parquet"
        fields_file = tmp_path / "fields.parquet"

        chips_gdf = gpd.GeoDataFrame(
            {
                "id": ["ftw-36NXF0000", "ftw-36NXF0001"],
                "geometry": [box(0, 0, 1, 1), box(2, 0, 3, 1)],
            },
            crs="EPSG:4326",
        )
        chips_gdf.to_parquet(chips_file)

        fields_gdf = gpd.GeoDataFrame(
            {
                "split": [
                    "train",
                    "Training",
                    "validation",
                    "testing",
                    "TRAIN",
                ],
                "geometry": [
                    box(0.1, 0.1, 0.2, 0.2),
                    box(0.2, 0.2, 0.3, 0.3),
                    box(0.3, 0.3, 0.4, 0.4),
                    box(2.1, 0.1, 2.2, 0.2),
                    box(2.2, 0.2, 2.3, 0.3),
                ],
            },
            crs="EPSG:4326",
        )
        fields_gdf.to_parquet(fields_file)

        result = assign_splits(
            chips_file=chips_file,
            split_type="predefined",
            split_percents=(80, 10, 10),
            random_seed=42,
            fields_file=fields_file,
        )

        assert result.total_chips == 2
        updated_gdf = gpd.read_parquet(chips_file)
        split_map = dict(zip(updated_gdf["id"], updated_gdf["split"], strict=True))

        # Chip 1: 2 train vs 1 val -> train
        assert split_map["ftw-36NXF0000"] == "train"
        # Chip 2: 1 train vs 1 test -> tie-break to train
        assert split_map["ftw-36NXF0001"] == "train"

    def test_predefined_creates_validation_when_missing(self, tmp_path: Path) -> None:
        """Test predefined split promotes 20% of train chips to validation when no val labels."""
        chips_file = tmp_path / "chips.parquet"
        fields_file = tmp_path / "fields.parquet"

        chip_ids = [f"ftw-36NXF{idx:04d}" for idx in range(12)]
        chips_gdf = gpd.GeoDataFrame(
            {
                "id": chip_ids,
                "geometry": [box(idx, 0, idx + 1, 1) for idx in range(12)],
            },
            crs="EPSG:4326",
        )
        chips_gdf.to_parquet(chips_file)

        split_labels = ["train"] * 10 + ["test"] * 2
        fields_gdf = gpd.GeoDataFrame(
            {
                "split": split_labels,
                "geometry": [box(idx + 0.1, 0.1, idx + 0.2, 0.2) for idx in range(12)],
            },
            crs="EPSG:4326",
        )
        fields_gdf.to_parquet(fields_file)

        messages: list[str] = []
        result = assign_splits(
            chips_file=chips_file,
            split_type="predefined",
            split_percents=(80, 10, 10),
            random_seed=42,
            fields_file=fields_file,
            on_progress=messages.append,
        )

        assert result.train_count == 8
        assert result.val_count == 2
        assert result.test_count == 2
        assert any("No validation labels found" in msg for msg in messages)
