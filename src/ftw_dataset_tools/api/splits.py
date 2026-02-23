"""Core API for assigning train/val/test splits to chips."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import geopandas as gpd
import geoparquet_io as gpio
import numpy as np
import pandas as pd

from ftw_dataset_tools.api.geo import write_geoparquet

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


SPLIT_TYPE_CHOICES: tuple[str, ...] = (
    "block3x3",
    "predefined",
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
    split_percents: tuple[float, float, float]
    total_chips: int
    train_count: int
    val_count: int
    test_count: int


def assign_splits(
    chips_file: str | Path,
    split_type: str,
    split_percents: tuple[int, int, int] = (80, 10, 10),
    random_seed: int = 42,
    fields_file: str | Path | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> CreateSplitsResult:
    """
    Assign train/val/test splits to chips file.

    Adds a 'split' column to the chips parquet file with values 'train', 'val', or 'test'.

    Args:
        chips_file: Path to chips parquet file
        split_type: Split strategy (required). Must be one of: 'random-uniform' (random
            individual chip assignment), 'block3x3' (3x3 spatial block assignment),
            or 'predefined' (use split column from fields file with majority vote per chip)
        split_percents: Tuple of (train_pct, val_pct, test_pct) summing to 100.
            Default: (80, 10, 10)
        random_seed: Random seed for reproducibility. Default: 42
        fields_file: Fields GeoParquet path (required for split_type='predefined').
        on_progress: Optional callback for progress messages

    Returns:
        CreateSplitsResult with statistics about the split assignment

    Raises:
        ValueError: If split_type is not supported, required 'id' column is missing,
            or chips file is empty
        FileNotFoundError: If chips_file doesn't exist
    """
    # Validate split_type early before any processing
    if split_type not in SPLIT_TYPE_CHOICES:
        raise ValueError(
            f"Unsupported split_type: {split_type}. Must be one of: {SPLIT_TYPE_CHOICES_STR}"
        )

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

    # Validate non-empty dataframe
    if n_chips == 0:
        raise ValueError(
            "Chips file is empty (contains 0 rows). Cannot assign splits to an empty dataset."
        )

    # Validate required columns
    if "id" not in gdf.columns:
        raise ValueError(
            f"Chips file must contain an 'id' column. Found columns: {list(gdf.columns)}"
        )

    if split_type == "random-uniform":
        splits = _assign_random_uniform(gdf, split_percents, random_seed)
    elif split_type == "block3x3":
        splits = _assign_block3x3(gdf, split_percents, random_seed)
    elif split_type == "predefined":
        splits = _assign_predefined(gdf, fields_file, random_seed, log)
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

    actual_split_percents = split_percents
    if split_type == "predefined":
        total = max(n_chips, 1)
        actual_split_percents = (
            100.0 * train_count / total,
            100.0 * val_count / total,
            100.0 * test_count / total,
        )

    log(f"Assigned {train_count} train, {val_count} val, {test_count} test")

    return CreateSplitsResult(
        chips_file=chips_path,
        split_type=split_type,
        split_percents=actual_split_percents,
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
    train_pct, val_pct = split_percents[0], split_percents[1]

    # Create random number generator for reproducibility without global state
    rng = np.random.default_rng(random_seed)

    # Calculate number of chips for each split
    n_train = int(n_chips * train_pct / 100)
    n_val = int(n_chips * val_pct / 100)
    n_test = n_chips - n_train - n_val  # Remainder goes to test to ensure exact total

    # Create split labels
    splits = np.array(["train"] * n_train + ["val"] * n_val + ["test"] * n_test)

    # Shuffle randomly using generator
    rng.shuffle(splits)

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
    # Validate chip ID format
    # IDs should follow format: ftw-<zone><band><grid><easting><northing>
    # Example: ftw-36NXF6658 (minimum length 13)
    chip_ids = gdf["id"].astype(str)
    min_length = chip_ids.str.len().min()
    if min_length < 13:
        raise ValueError(
            f"Invalid chip ID format: IDs must be at least 13 characters (e.g., 'ftw-36NXF6658'). "
            f"Found chip ID with length {min_length}"
        )

    # Validate prefix
    if not all(chip_ids.str.startswith("ftw-")):
        invalid_ids = chip_ids[~chip_ids.str.startswith("ftw-")].tolist()
        raise ValueError(
            f"Invalid chip ID format: IDs must start with 'ftw-'. "
            f"Found invalid IDs: {invalid_ids[:5]}"  # Show first 5 for brevity
        )

    # Extract easting and northing from chip IDs (last 4 digits: EENN)
    # Example: ftw-36NXF6658 -> zone=36N, grid=XF, easting=66, northing=58
    try:
        eastings = chip_ids.str[-4:-2].astype(int)
        northings = chip_ids.str[-2:].astype(int)
    except (ValueError, TypeError) as e:
        raise ValueError(
            "Invalid chip ID format: Unable to extract numeric easting/northing from last "
            f"4 characters. Expected format: ftw-<zone><band><grid><EENN>. Error: {e}"
        ) from e

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

    # Create random number generator for reproducibility without global state
    rng = np.random.default_rng(random_seed)

    # Calculate number of blocks for each split
    train_pct, val_pct = split_percents[0], split_percents[1]
    n_train = int(n_blocks * train_pct / 100)
    n_val = int(n_blocks * val_pct / 100)
    n_test = n_blocks - n_train - n_val  # Remainder goes to test

    # Create block split labels
    block_splits = np.array(["train"] * n_train + ["val"] * n_val + ["test"] * n_test)

    # Shuffle block assignments randomly using generator
    rng.shuffle(block_splits)

    # Create mapping from block ID to split
    block_to_split = dict(zip(unique_blocks, block_splits, strict=True))

    # Map each chip to its block's split assignment
    chip_splits = block_ids.map(block_to_split).values

    return chip_splits


def _normalize_predefined_split(value: object) -> str | None:
    """Normalize user-provided split labels to train/val/test."""
    if pd.isna(value):
        return None

    text = str(value).strip().lower()
    mapping = {
        "train": "train",
        "training": "train",
        "val": "val",
        "valid": "val",
        "validation": "val",
        "test": "test",
        "testing": "test",
    }
    return mapping.get(text)


def _validate_fields_file(fields_file: str | Path | None) -> Path:
    if fields_file is None:
        raise ValueError("fields_file is required when split_type is 'predefined'")

    fields_path = Path(fields_file).resolve()
    if not fields_path.exists():
        raise FileNotFoundError(f"Fields file not found: {fields_path}")

    return fields_path


def _load_and_validate_fields(fields_path: Path) -> gpd.GeoDataFrame:
    table = gpio.read(str(fields_path))
    fields_gdf = gpd.GeoDataFrame.from_arrow(
        table.to_arrow(),
        geometry=table.geometry_column,
    )
    if "split" not in fields_gdf.columns:
        raise ValueError(
            "Fields file must contain a 'split' column when split_type is 'predefined'. "
            f"Found columns: {list(fields_gdf.columns)}"
        )
    return fields_gdf


def _normalize_and_validate_splits(
    fields_gdf: gpd.GeoDataFrame,
    log: Callable[[str], None],
) -> gpd.GeoDataFrame:
    fields_gdf = fields_gdf.copy()
    fields_gdf["_split_norm"] = fields_gdf["split"].map(_normalize_predefined_split)

    null_mask = fields_gdf["split"].isna()
    if null_mask.any():
        null_count = int(null_mask.sum())
        example_indices = fields_gdf.index[null_mask][:5].tolist()
        log(
            "Warning: Found null split values in fields file. "
            f"Count: {null_count}. Example row indices: {example_indices}"
        )

    invalid = fields_gdf[fields_gdf["_split_norm"].isna()]["split"].dropna().unique()
    if len(invalid) > 0:
        invalid_list = ", ".join(map(str, invalid[:5]))
        raise ValueError(
            "Invalid split values in fields file. Expected variants of train/val/test. "
            f"Found: {invalid_list}"
        )

    return fields_gdf


def _ensure_crs_alignment(
    gdf: gpd.GeoDataFrame,
    fields_gdf: gpd.GeoDataFrame,
    log: Callable[[str], None],
) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        raise ValueError("Chips file has no CRS information; cannot align with fields CRS.")
    if fields_gdf.crs is None:
        raise ValueError("Fields file has no CRS information; cannot align with chips CRS.")

    if fields_gdf.crs != gdf.crs:
        log("Reprojecting fields to match chips CRS for predefined splits...")
        fields_gdf = fields_gdf.to_crs(gdf.crs)

    return fields_gdf


def _compute_chip_majority_splits(
    fields_gdf: gpd.GeoDataFrame,
    gdf: gpd.GeoDataFrame,
) -> tuple[pd.Series, bool]:
    joined = gpd.sjoin(
        fields_gdf[["_split_norm", "geometry"]],
        gdf[["id", "geometry"]],
        how="inner",
        predicate="intersects",
    )

    if joined.empty:
        raise ValueError(
            "No fields intersected chips when assigning predefined splits. "
            "Check CRS alignment and geometry validity."
        )

    counts = (
        joined.groupby("id")["_split_norm"]
        .value_counts()
        .unstack(fill_value=0)
        .reindex(columns=["train", "val", "test"], fill_value=0)
    )

    has_val_labels = bool(counts.get("val", pd.Series(dtype=int)).sum() > 0)

    priority = ["train", "val", "test"]
    chip_to_split: dict[str, str] = {}
    for chip_id, row in counts.iterrows():
        max_count = int(row.max())
        if max_count == 0:
            continue
        for split in priority:
            if int(row[split]) == max_count:
                chip_to_split[chip_id] = split
                break

    splits = gdf["id"].map(chip_to_split)
    missing = gdf.loc[splits.isna(), "id"].astype(str).tolist()
    if missing:
        missing_preview = ", ".join(missing[:5])
        raise ValueError(
            "No predefined split assignments found for some chips. "
            f"Example missing chip IDs: {missing_preview}"
        )

    return splits, has_val_labels


def _assign_predefined(
    gdf: gpd.GeoDataFrame,
    fields_file: str | Path | None,
    random_seed: int,
    log: Callable[[str], None],
) -> np.ndarray:
    """Assign splits by majority vote using a predefined split column in fields."""
    fields_path = _validate_fields_file(fields_file)
    fields_gdf = _load_and_validate_fields(fields_path)
    fields_gdf = _normalize_and_validate_splits(fields_gdf, log)
    fields_gdf = _ensure_crs_alignment(gdf, fields_gdf, log)
    splits, has_val_labels = _compute_chip_majority_splits(fields_gdf, gdf)

    if not has_val_labels:
        train_mask = splits.eq("train")
        train_indices = splits[train_mask].index.to_numpy()
        train_count = len(train_indices)
        n_val = int(train_count * 0.2)

        if n_val > 0:
            rng = np.random.default_rng(random_seed)
            val_indices = rng.choice(train_indices, size=n_val, replace=False)
            splits.loc[val_indices] = "val"
            log(
                "Warning: No validation labels found in fields split column. "
                f"Promoted {n_val} of {train_count} training chips to validation (20% of train)."
            )
        else:
            log(
                "Warning: No validation labels found in fields split column, "
                "and training set is too small to allocate 20% to validation."
            )

    return splits.to_numpy()
