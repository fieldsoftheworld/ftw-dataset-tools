"""Core API for assigning train/val/test splits to chips."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np

from ftw_dataset_tools.api.geo import write_geoparquet

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Sequence


SPLIT_TYPE_CHOICES: tuple[str, ...] = (
    "block3x3",
    "random-uniform",
)
SPLIT_TYPE_CHOICES_STR = ", ".join(SPLIT_TYPE_CHOICES)


def validate_split_percents(
    split_percents: Sequence[int] | tuple[int, int, int],
) -> tuple[int, int, int]:
    """Validate train/val/test split percentages.

    Ensures three integers in [0, 100] that sum to 100.
    """

    if len(split_percents) != 3:
        raise ValueError("split_percents must have exactly three values (train, val, test)")

    if any(p < 0 or p > 100 for p in split_percents):
        raise ValueError("split_percents values must be between 0 and 100")

    if sum(split_percents) != 100:
        raise ValueError("split_percents must sum to 100")

    return tuple(int(p) for p in split_percents)


@dataclass
class CreateSplitsResult:
    """Result of split assignment operation."""

    chips_file: Path
    split_type: str
    split_percents: tuple[int, int, int]
    total_chips: int
    train_count: int
    val_count: int
    test_count: int


def assign_splits(
    chips_file: str | Path,
    split_type: str,
    split_percents: tuple[int, int, int] = (80, 10, 10),
    random_seed: int = 42,
    on_progress: Callable[[str], None] | None = None,
) -> CreateSplitsResult:
    """
    Assign train/val/test splits to chips file.

    Adds a 'split' column to the chips parquet file with values 'train', 'val', or 'test'.

    Args:
        chips_file: Path to chips parquet file
        split_type: Split strategy ('random-uniform' or 'block3x3')
        split_percents: Tuple of (train_pct, val_pct, test_pct) summing to 100
        random_seed: Random seed for reproducibility (used for random-uniform)
        on_progress: Optional callback for progress messages

    Returns:
        CreateSplitsResult with statistics about the split assignment

    Raises:
        ValueError: If split_type is not supported
        FileNotFoundError: If chips_file doesn't exist
    """
    chips_path = Path(chips_file).resolve()

    if not chips_path.exists():
        raise FileNotFoundError(f"Chips file not found: {chips_path}")

    def log(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    log(f"Assigning {split_type} splits to {chips_path.name}")

    # Read chips geoparquet
    gdf = gpd.read_parquet(chips_path)
    n_chips = len(gdf)

    if split_type == "random-uniform":
        splits = _assign_random_uniform(gdf, split_percents, random_seed)
    elif split_type == "block3x3":
        splits = _assign_block3x3(gdf, split_percents, random_seed)
    else:
        raise ValueError(f"Unsupported split_type: {split_type}")

    # Assign to geodataframe
    gdf["split"] = splits

    # Write back to geoparquet (preserves metadata)
    write_geoparquet(chips_path, gdf=gdf)

    # Count assignments
    train_count = int((splits == "train").sum())
    val_count = int((splits == "val").sum())
    test_count = int((splits == "test").sum())

    log(f"Assigned {train_count} train, {val_count} val, {test_count} test")

    return CreateSplitsResult(
        chips_file=chips_path,
        split_type=split_type,
        split_percents=split_percents,
        total_chips=n_chips,
        train_count=train_count,
        val_count=val_count,
        test_count=test_count,
    )


def _assign_random_uniform(
    gdf: gpd.GeoDataFrame,
    split_percents: tuple[int, int, int],
    random_seed: int,
) -> np.ndarray:
    """Assign splits randomly and uniformly across all chips."""
    n_chips = len(gdf)
    train_pct, val_pct, test_pct = split_percents

    # Set random seed
    np.random.seed(random_seed)

    # Calculate number of chips for each split
    n_train = int(n_chips * train_pct / 100)
    n_val = int(n_chips * val_pct / 100)
    n_test = n_chips - n_train - n_val  # Remainder goes to test to ensure exact total

    # Create split labels
    splits = np.array(
        ["train"] * n_train + ["val"] * n_val + ["test"] * n_test
    )

    # Shuffle randomly
    np.random.shuffle(splits)

    return splits


def _assign_block3x3(
    gdf: gpd.GeoDataFrame,
    split_percents: tuple[int, int, int],
    random_seed: int,
) -> np.ndarray:
    """Assign splits using 3x3 block pattern.
    
    Groups chips into 3x3 blocks based on their grid coordinates (extracted from chip ID),
    then randomly assigns each block to train/val/test. This ensures spatial coherence
    within each split.
    """
    # Extract easting and northing from chip IDs (last 4 digits: EENN)
    # IDs follow format: ftw-<zone><band><grid><easting><northing>
    # Example: ftw-36NXF6658 -> zone=36N, grid=XF, easting=66, northing=58
    chip_ids = gdf["id"].astype(str)
    eastings = chip_ids.str[-4:-2].astype(int)
    northings = chip_ids.str[-2:].astype(int)
    
    # Extract the full MGRS grid identifier (zone + band + 100km grid square)
    # This is characters 4-9 of the chip ID (after "ftw-")
    # Example: ftw-36NXF6658 -> 36NXF
    mgrs_grids = chip_ids.str[4:9]
    
    # Create 3x3 block IDs by dividing coordinates by 3 (integer division)
    # This groups coordinates 0-2, 3-5, 6-8, etc. into the same block
    block_east = eastings // 3
    block_north = northings // 3
    
    # Create unique block identifier combining MGRS grid and block coordinates
    block_ids = mgrs_grids + "_" + block_east.astype(str) + "_" + block_north.astype(str)
    
    # Get unique blocks and their counts
    unique_blocks = block_ids.unique()
    n_blocks = len(unique_blocks)
    
    # Set random seed
    np.random.seed(random_seed)
    
    # Calculate number of blocks for each split
    train_pct, val_pct, test_pct = split_percents
    n_train = int(n_blocks * train_pct / 100)
    n_val = int(n_blocks * val_pct / 100)
    n_test = n_blocks - n_train - n_val  # Remainder goes to test
    
    # Create block split labels
    block_splits = np.array(
        ["train"] * n_train + ["val"] * n_val + ["test"] * n_test
    )
    
    # Shuffle block assignments randomly
    np.random.shuffle(block_splits)
    
    # Create mapping from block ID to split
    block_to_split = dict(zip(unique_blocks, block_splits))
    
    # Map each chip to its block's split assignment
    chip_splits = block_ids.map(block_to_split).values
    
    return chip_splits
