"""Generate dataset summaries with visualizations and statistics."""

from __future__ import annotations

import datetime  # noqa: TC003
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import duckdb
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pystac

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd

__all__ = [
    "DatasetSummary",
    "create_dataset_summary",
]


@dataclass
class DatasetSummary:
    """Summary statistics and visualizations for a dataset."""

    dataset_dir: Path
    chips_dir: Path
    total_chips: int
    train_chips: int
    val_chips: int
    test_chips: int
    planting_dates: list[datetime.datetime]
    harvest_dates: list[datetime.datetime]
    planting_cloud_cover: list[float]
    harvest_cloud_cover: list[float]
    metadata: dict
    example_chips: list[str]
    field_coverage_pct: list[float]
    empty_mask_count: int
    output_path: Path


def create_dataset_summary(
    dataset_dir: str | Path,
    output_path: str | Path | None = None,
    num_examples: int = 10,
    on_progress: Callable[[str], None] | None = None,
) -> DatasetSummary:
    """
    Create a markdown summary report for a dataset.

    Args:
        dataset_dir: Path to dataset directory containing *-chips/ subdirectory
        output_path: Output path for markdown file (default: dataset_dir/summary.md)
        num_examples: Number of example chips to include (default: 10)
        on_progress: Optional callback for progress messages

    Returns:
        DatasetSummary with statistics and paths

    Raises:
        FileNotFoundError: If dataset directory or chips directory not found
        ValueError: If no chips found or required files missing
    """
    dataset_dir = Path(dataset_dir).resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    def log(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    log(f"Analyzing dataset: {dataset_dir}")

    # Find chips directory and parquet file
    chips_dir, chips_parquet = _find_chips_dir_and_parquet(dataset_dir, log)

    # Load chips dataframe and compute split counts
    df, total_chips, train_chips, val_chips, test_chips = _load_chips_df(chips_parquet, log)

    # Collect STAC metadata from chip items
    stac_metadata = _collect_stac_metadata(chips_dir, log)

    # Select example chips with imagery
    example_chips = _select_example_chips(
        chips_dir, stac_metadata["planting_items"], num_examples, log
    )

    # Get field coverage statistics
    field_coverage_pct = (
        df["field_coverage_pct"].tolist() if "field_coverage_pct" in df.columns else []
    )

    # Count empty masks
    empty_mask_count = _count_empty_masks(chips_dir, log)

    # Generate visualizations
    figures_dir = dataset_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    _generate_visualizations(
        df=df,
        figures_dir=figures_dir,
        planting_dates=stac_metadata["planting_dates"],
        harvest_dates=stac_metadata["harvest_dates"],
        planting_cloud_cover=stac_metadata["planting_cloud_cover"],
        harvest_cloud_cover=stac_metadata["harvest_cloud_cover"],
        field_coverage_pct=field_coverage_pct,
        log=log,
    )

    # Create markdown report
    output_path = dataset_dir / "summary.md" if output_path is None else Path(output_path)

    log("Generating markdown report...")
    _write_markdown_summary(
        output_path=output_path,
        dataset_dir=dataset_dir,
        chips_dir=chips_dir,
        total_chips=total_chips,
        train_chips=train_chips,
        val_chips=val_chips,
        test_chips=test_chips,
        metadata=stac_metadata["metadata"],
        example_chips=example_chips,
        planting_dates=stac_metadata["planting_dates"],
        harvest_dates=stac_metadata["harvest_dates"],
        planting_cloud_cover=stac_metadata["planting_cloud_cover"],
        harvest_cloud_cover=stac_metadata["harvest_cloud_cover"],
        field_coverage_pct=field_coverage_pct,
        empty_mask_count=empty_mask_count,
        figures_dir=figures_dir,
    )

    log(f"Summary written to: {output_path}")

    return DatasetSummary(
        dataset_dir=dataset_dir,
        chips_dir=chips_dir,
        total_chips=total_chips,
        train_chips=train_chips,
        val_chips=val_chips,
        test_chips=test_chips,
        planting_dates=stac_metadata["planting_dates"],
        harvest_dates=stac_metadata["harvest_dates"],
        planting_cloud_cover=stac_metadata["planting_cloud_cover"],
        harvest_cloud_cover=stac_metadata["harvest_cloud_cover"],
        metadata=stac_metadata["metadata"],
        example_chips=example_chips,
        field_coverage_pct=field_coverage_pct,
        empty_mask_count=empty_mask_count,
        output_path=output_path,
    )


def _find_chips_dir_and_parquet(dataset_dir: Path, log: Callable[[str], None]) -> tuple[Path, Path]:
    """Find chips directory and parquet file in dataset directory."""
    chips_dirs = list(dataset_dir.glob("*-chips"))
    if not chips_dirs:
        raise FileNotFoundError(f"No *-chips directory found in {dataset_dir}")
    chips_dir = chips_dirs[0]
    log(f"Found chips directory: {chips_dir.name}")

    chips_parquet_files = list(dataset_dir.glob("*_chips.parquet"))
    if not chips_parquet_files:
        raise FileNotFoundError(f"No *_chips.parquet file found in {dataset_dir}")
    chips_parquet = chips_parquet_files[0]

    return chips_dir, chips_parquet


def _load_chips_df(
    chips_parquet: Path, log: Callable[[str], None]
) -> tuple[pd.DataFrame, int, int, int, int]:
    """Load chips dataframe and compute split counts."""
    log("Loading chips data...")

    # Use DuckDB with spatial extension to load parquet file
    con = duckdb.connect(":memory:")
    con.execute("INSTALL spatial")
    con.execute("LOAD spatial")

    # Check if geometry column exists and convert it to WKB format
    columns_query = "SELECT * FROM read_parquet(?) LIMIT 0"
    columns_df = con.execute(columns_query, [str(chips_parquet)]).fetchdf()

    if "geometry" in columns_df.columns:
        # Convert geometry column to WKB bytes for compatibility with shapely
        query = "SELECT * EXCLUDE (geometry), ST_AsWKB(geometry) as geometry FROM read_parquet(?)"
    else:
        query = "SELECT * FROM read_parquet(?)"

    df = con.execute(query, [str(chips_parquet)]).fetchdf()
    con.close()

    total_chips = len(df)
    train_chips = int((df["split"] == "train").sum()) if "split" in df.columns else 0
    val_chips = int((df["split"] == "val").sum()) if "split" in df.columns else 0
    test_chips = int((df["split"] == "test").sum()) if "split" in df.columns else 0

    log(f"Total chips: {total_chips} (train={train_chips}, val={val_chips}, test={test_chips})")

    return df, total_chips, train_chips, val_chips, test_chips


def _extract_dates_and_cloud_cover(
    items: list[Path], log: Callable[[str], None]
) -> tuple[list[datetime], list[float]]:
    """Extract dates and cloud cover from STAC items.

    Args:
        items: List of paths to STAC item JSON files
        log: Logging callback function

    Returns:
        Tuple of (dates, cloud_cover_values)
    """
    dates = []
    cloud_cover = []

    for json_file in items:
        try:
            item = pystac.Item.from_file(str(json_file))
            if item.datetime:
                dates.append(item.datetime)
            if "eo:cloud_cover" in item.properties:
                cloud_cover.append(item.properties["eo:cloud_cover"])
        except Exception as e:
            log(f"Warning: Could not parse {json_file.name}: {e}")

    return dates, cloud_cover


def _collect_stac_metadata(chips_dir: Path, log: Callable[[str], None]) -> dict:
    """Collect STAC metadata from chip items."""
    log("Scanning STAC items...")

    # Find all chip subdirectories
    chip_subdirs = [d for d in chips_dir.iterdir() if d.is_dir()]
    log(f"Found {len(chip_subdirs)} chip subdirectories")

    # Collect all JSON files from chip subdirectories
    chip_json_files = []
    for chip_subdir in chip_subdirs:
        chip_json_files.extend(chip_subdir.glob("*.json"))

    log(f"Found {len(chip_json_files)} JSON files in chip subdirectories")

    # Separate parent items and child items
    parent_items = [
        f
        for f in chip_json_files
        if not ("_planting_s2" in f.stem or "_harvest_s2" in f.stem or f.name == "collection.json")
    ]
    planting_items = [f for f in chip_json_files if "_planting_s2" in f.stem]
    harvest_items = [f for f in chip_json_files if "_harvest_s2" in f.stem]

    log(
        f"Found {len(parent_items)} parent items, {len(planting_items)} planting items, {len(harvest_items)} harvest items"
    )

    # Process planting and harvest child items
    planting_dates, planting_cloud_cover = _extract_dates_and_cloud_cover(planting_items, log)
    harvest_dates, harvest_cloud_cover = _extract_dates_and_cloud_cover(harvest_items, log)

    # Process parent items for metadata
    metadata = {}
    for json_file in parent_items:
        try:
            item = pystac.Item.from_file(str(json_file))
            if not metadata:
                metadata = {
                    "calendar_year": item.properties.get("ftw:calendar_year"),
                    "planting_day": item.properties.get("ftw:planting_day"),
                    "harvest_day": item.properties.get("ftw:harvest_day"),
                    "cloud_cover_threshold": item.properties.get("ftw:cloud_cover_chip_threshold"),
                    "buffer_days": item.properties.get("ftw:buffer_days"),
                    "stac_host": item.properties.get("ftw:stac_host"),
                }
                break
        except Exception as e:
            log(f"Warning: Could not parse {json_file.name}: {e}")

    # Fallback to child items if no parent metadata
    if not metadata and (planting_items or harvest_items):
        try:
            sample_item_file = planting_items[0] if planting_items else harvest_items[0]
            sample_item = pystac.Item.from_file(str(sample_item_file))
            metadata = {
                "calendar_year": sample_item.properties.get("ftw:calendar_year"),
                "planting_day": None,
                "harvest_day": None,
                "cloud_cover_threshold": None,
                "buffer_days": None,
                "stac_host": None,
            }
        except Exception:
            pass

    log(f"Found {len(planting_dates)} planting dates, {len(harvest_dates)} harvest dates")

    return {
        "planting_items": planting_items,
        "planting_dates": planting_dates,
        "harvest_dates": harvest_dates,
        "planting_cloud_cover": planting_cloud_cover,
        "harvest_cloud_cover": harvest_cloud_cover,
        "metadata": metadata,
    }


def _select_example_chips(
    chips_dir: Path, planting_items: list[Path], num_examples: int, log: Callable[[str], None]
) -> list[str]:
    """Select example chips with imagery."""
    example_chips = []
    chip_ids_with_planting = set()

    for planting_file in planting_items:
        chip_id = planting_file.stem.replace("_planting_s2", "")
        chip_ids_with_planting.add(chip_id)

    # Randomly sample chip IDs to get diverse examples
    chip_ids_list = list(chip_ids_with_planting)

    # Guard against empty chip list
    if not chip_ids_list:
        log("Warning: No chip IDs found with planting items, returning empty example list")
        return []

    rng = np.random.default_rng(42)
    sample_size = min(num_examples * 3, len(chip_ids_list))
    chip_ids_to_check = rng.choice(chip_ids_list, size=sample_size, replace=False)

    for chip_id in chip_ids_to_check:
        preview_dir = chips_dir / chip_id
        if preview_dir.exists() and preview_dir.is_dir():
            planting_jpg = preview_dir / f"{chip_id}_planting_image_s2.jpg"
            harvest_jpg = preview_dir / f"{chip_id}_harvest_image_s2.jpg"
            if planting_jpg.exists() and harvest_jpg.exists():
                example_chips.append(chip_id)
                if len(example_chips) >= num_examples:
                    break

    log(f"Selected {len(example_chips)} example chips with imagery")
    return example_chips


def _count_empty_masks(chips_dir: Path, log: Callable[[str], None]) -> int:
    """Count the number of chips with empty masks (no field pixels).

    Args:
        chips_dir: Directory containing chip subdirectories
        log: Logging function

    Returns:
        Number of chips with empty masks
    """
    log("Counting empty masks...")

    try:
        import rasterio
    except ImportError:
        log("Warning: rasterio not available, skipping empty mask count")
        return 0

    empty_count = 0
    total_checked = 0

    for chip_dir in sorted(chips_dir.iterdir()):
        if not chip_dir.is_dir():
            continue

        # Look for mask file
        mask_files = list(chip_dir.glob("*_semantic_3_class.tif"))
        if not mask_files:
            continue

        mask_file = mask_files[0]
        try:
            with rasterio.open(mask_file) as src:
                data = src.read(1)  # Read first band
                # Check if all pixels are 0 (no fields)
                if np.all(data == 0):
                    empty_count += 1
                total_checked += 1
        except Exception as e:
            log(f"Warning: Failed to read mask {mask_file.name}: {e}")
            continue

    log(f"Found {empty_count} empty masks out of {total_checked} chips checked")
    return empty_count


def _generate_visualizations(
    df: pd.DataFrame,
    figures_dir: Path,
    planting_dates: list[datetime.datetime],
    harvest_dates: list[datetime.datetime],
    planting_cloud_cover: list[float],
    harvest_cloud_cover: list[float],
    field_coverage_pct: list[float],
    log: Callable[[str], None],
) -> None:
    """Generate all visualization plots."""
    log(f"Creating visualizations in {figures_dir.name}/")

    _create_split_map(df, figures_dir / "split_map.png", log)

    if planting_dates:
        _create_date_histogram(planting_dates, "Planting", figures_dir / "planting_dates.png", log)
    if harvest_dates:
        _create_date_histogram(harvest_dates, "Harvest", figures_dir / "harvest_dates.png", log)

    if planting_cloud_cover:
        _create_histogram(
            planting_cloud_cover,
            "Planting Cloud Cover (%)",
            figures_dir / "planting_cloud_cover.png",
            log,
        )
    if harvest_cloud_cover:
        _create_histogram(
            harvest_cloud_cover,
            "Harvest Cloud Cover (%)",
            figures_dir / "harvest_cloud_cover.png",
            log,
        )

    if field_coverage_pct:
        _create_histogram(
            field_coverage_pct,
            "Field Coverage (%)",
            figures_dir / "field_coverage.png",
            log,
        )


def _create_split_map(df: pd.DataFrame, output_path: Path, log: Callable[[str], None]) -> None:
    """Create a map visualization of train/val/test splits."""
    log("Creating split map...")

    # Ensure we have geometry column
    if "geometry" not in df.columns:
        log("Warning: No geometry column found, skipping split map")
        return

    try:
        import geopandas as gpd
        from shapely import wkb

        # Convert to GeoDataFrame - handle WKB format if needed
        if not isinstance(df, gpd.GeoDataFrame):
            # Check if DataFrame is empty before accessing iloc[0]
            if df.empty:
                log("Warning: Empty DataFrame, skipping split map")
                return

            # Check if geometry is in WKB format (bytes or bytearray)
            first_geom = df["geometry"].iloc[0]

            if isinstance(first_geom, (bytes, bytearray)):
                df = df.copy()

                # Convert WKB to geometry, handling None/invalid values
                def safe_wkb_load(x):
                    if isinstance(x, (bytes, bytearray)):
                        try:
                            # Convert bytearray to bytes if needed
                            if isinstance(x, bytearray):
                                x = bytes(x)
                            return wkb.loads(x)
                        except Exception:
                            return None
                    return None

                df["geometry"] = df["geometry"].apply(safe_wkb_load)
            df = gpd.GeoDataFrame(df, geometry="geometry")

        # Filter out rows with None geometry
        df = df[df["geometry"].notna()]

        if len(df) == 0:
            log("Warning: No valid geometries found after conversion, skipping split map")
            return

        log(f"Plotting {len(df)} chips on map...")

        # Plot splits
        fig, ax = plt.subplots(figsize=(12, 8))

        # Define colors
        colors = {"train": "#1f77b4", "val": "#ff7f0e", "test": "#2ca02c"}

        if "split" in df.columns:
            # Create custom legend handles
            from matplotlib.patches import Patch

            legend_handles = []

            for split_name, color in colors.items():
                split_df = df[df["split"] == split_name]
                if len(split_df) > 0:
                    split_df.plot(ax=ax, color=color, alpha=0.6, edgecolor="black", linewidth=0.5)
                    legend_handles.append(
                        Patch(facecolor=color, edgecolor="black", label=split_name, alpha=0.6)
                    )

            if legend_handles:
                ax.legend(handles=legend_handles, title="Split")
        else:
            df.plot(ax=ax, alpha=0.6, edgecolor="black", linewidth=0.5)

        ax.set_title("Dataset Splits - Geographic Distribution")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log(f"  Saved: {output_path.name}")

    except Exception as e:
        import traceback

        log(f"Warning: Could not create split map: {e}")
        log(f"Traceback: {traceback.format_exc()}")


def _create_date_histogram(
    dates: list[datetime.datetime], label: str, output_path: Path, log: Callable[[str], None]
) -> None:
    """Create histogram of dates."""
    log(f"Creating {label.lower()} date histogram...")

    try:
        import matplotlib.dates as mdates

        fig, ax = plt.subplots(figsize=(10, 5))

        # Convert to matplotlib dates using date2num
        date_nums = mdates.date2num(dates)

        ax.hist(date_nums, bins=30, edgecolor="black", alpha=0.7)
        ax.set_xlabel("Date")
        ax.set_ylabel("Count")
        ax.set_title(f"{label} Image Dates")

        # Format x-axis as dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        fig.autofmt_xdate()  # Rotate date labels automatically

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log(f"  Saved: {output_path.name}")

    except Exception as e:
        import traceback

        log(f"Warning: Could not create date histogram: {e}")
        log(f"Traceback: {traceback.format_exc()}")


def _create_histogram(
    values: list[float], label: str, output_path: Path, log: Callable[[str], None], bins: int = 20
) -> None:
    """Create histogram of numerical values."""
    log(f"Creating {label.lower()} histogram...")

    try:
        fig, ax = plt.subplots(figsize=(10, 5))

        ax.hist(values, bins=bins, edgecolor="black", alpha=0.7)
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.set_title(f"{label} Distribution")

        # Add statistics text
        stats_text = f"Mean: {np.mean(values):.2f}\nMedian: {np.median(values):.2f}\nStd: {np.std(values):.2f}"
        ax.text(
            0.98,
            0.98,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
        )

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log(f"  Saved: {output_path.name}")

    except Exception as e:
        import traceback

        log(f"Warning: Could not create histogram: {e}")
        log(f"Traceback: {traceback.format_exc()}")


def _write_markdown_summary(
    output_path: Path,
    dataset_dir: Path,
    chips_dir: Path,
    total_chips: int,
    train_chips: int,
    val_chips: int,
    test_chips: int,
    metadata: dict,
    example_chips: list[str],
    planting_dates: list[datetime],
    harvest_dates: list[datetime],
    planting_cloud_cover: list[float],
    harvest_cloud_cover: list[float],
    field_coverage_pct: list[float],
    empty_mask_count: int,
    figures_dir: Path,
) -> None:
    """Write the markdown summary report."""
    with output_path.open("w") as f:
        f.write(f"# Dataset Summary: {dataset_dir.name}\n\n")

        # Overview
        f.write("## Overview\n\n")
        f.write(f"- **Dataset Directory**: `{dataset_dir.name}`\n")
        f.write(f"- **Total Chips**: {total_chips:,}\n")

        # Avoid division by zero
        if total_chips > 0:
            f.write(f"- **Train**: {train_chips:,} ({train_chips / total_chips * 100:.1f}%)\n")
            f.write(f"- **Validation**: {val_chips:,} ({val_chips / total_chips * 100:.1f}%)\n")
            f.write(f"- **Test**: {test_chips:,} ({test_chips / total_chips * 100:.1f}%)\n\n")
        else:
            f.write(f"- **Train**: {train_chips:,}\n")
            f.write(f"- **Validation**: {val_chips:,}\n")
            f.write(f"- **Test**: {test_chips:,}\n\n")

        # Metadata
        if metadata:
            f.write("## Configuration\n\n")
            if metadata.get("calendar_year"):
                f.write(f"- **Calendar Year**: {metadata['calendar_year']}\n")
            if metadata.get("planting_day"):
                f.write(f"- **Planting Day**: {metadata['planting_day']}\n")
            if metadata.get("harvest_day"):
                f.write(f"- **Harvest Day**: {metadata['harvest_day']}\n")
            if metadata.get("cloud_cover_threshold"):
                f.write(f"- **Cloud Cover Threshold**: {metadata['cloud_cover_threshold']}%\n")
            if metadata.get("buffer_days"):
                f.write(f"- **Buffer Days**: {metadata['buffer_days']}\n")
            if metadata.get("stac_host"):
                f.write(f"- **STAC Host**: {metadata['stac_host']}\n")
            f.write("\n")

        # Split map
        split_map_path = figures_dir / "split_map.png"
        if split_map_path.exists():
            f.write("## Geographic Distribution\n\n")
            f.write(f"![Split Map](figures/{split_map_path.name})\n\n")

        # Date distributions
        f.write("## Temporal Distribution\n\n")

        if planting_dates:
            planting_date_path = figures_dir / "planting_dates.png"
            if planting_date_path.exists():
                f.write(f"### Planting Dates ({len(planting_dates)} images)\n\n")
                f.write(f"![Planting Dates](figures/{planting_date_path.name})\n\n")

        if harvest_dates:
            harvest_date_path = figures_dir / "harvest_dates.png"
            if harvest_date_path.exists():
                f.write(f"### Harvest Dates ({len(harvest_dates)} images)\n\n")
                f.write(f"![Harvest Dates](figures/{harvest_date_path.name})\n\n")

        # Cloud cover distributions
        f.write("## Image Quality Metrics\n\n")

        if planting_cloud_cover:
            planting_cc_path = figures_dir / "planting_cloud_cover.png"
            if planting_cc_path.exists():
                f.write(f"### Planting Cloud Cover ({len(planting_cloud_cover)} images)\n\n")
                f.write(f"![Planting Cloud Cover](figures/{planting_cc_path.name})\n\n")

        if harvest_cloud_cover:
            harvest_cc_path = figures_dir / "harvest_cloud_cover.png"
            if harvest_cc_path.exists():
                f.write(f"### Harvest Cloud Cover ({len(harvest_cloud_cover)} images)\n\n")
                f.write(f"![Harvest Cloud Cover](figures/{harvest_cc_path.name})\n\n")

        # Mask statistics
        f.write("## Mask Statistics\n\n")

        if field_coverage_pct:
            f.write("### Field Coverage Distribution\n\n")
            field_cov_path = figures_dir / "field_coverage.png"
            if field_cov_path.exists():
                f.write(f"![Field Coverage](figures/{field_cov_path.name})\n\n")

            f.write(f"- **Mean Coverage**: {np.mean(field_coverage_pct):.2f}%\n")
            f.write(f"- **Median Coverage**: {np.median(field_coverage_pct):.2f}%\n")
            f.write(
                f"- **Min/Max**: {np.min(field_coverage_pct):.2f}% / {np.max(field_coverage_pct):.2f}%\n"
            )
            f.write(
                f"- **Empty Masks**: {empty_mask_count:,} ({empty_mask_count / total_chips * 100:.1f}% of total)\n\n"
            )

        # Example chips
        if example_chips:
            f.write(f"## Example Chips ({len(example_chips)} samples)\n\n")
            f.write("| Chip ID | Planting | Harvest | Overlay |\n")
            f.write("|---------|----------|---------|----------|\n")

            for chip_id in example_chips:
                chip_img_dir = chips_dir / chip_id
                planting_img = f"{chips_dir.name}/{chip_id}/{chip_id}_planting_image_s2.jpg"
                harvest_img = f"{chips_dir.name}/{chip_id}/{chip_id}_harvest_image_s2.jpg"
                overlay_img = f"{chips_dir.name}/{chip_id}/{chip_id}_overlay.jpg"

                # Check if images exist
                planting_exists = (chip_img_dir / f"{chip_id}_planting_image_s2.jpg").exists()
                harvest_exists = (chip_img_dir / f"{chip_id}_harvest_image_s2.jpg").exists()
                overlay_exists = (chip_img_dir / f"{chip_id}_overlay.jpg").exists()

                planting_cell = (
                    f"![{chip_id} planting]({planting_img})" if planting_exists else "N/A"
                )
                harvest_cell = f"![{chip_id} harvest]({harvest_img})" if harvest_exists else "N/A"
                overlay_cell = f"![{chip_id} overlay]({overlay_img})" if overlay_exists else "N/A"

                f.write(f"| `{chip_id}` | {planting_cell} | {harvest_cell} | {overlay_cell} |\n")

            f.write("\n")

        # Statistics summary
        f.write("## Statistics Summary\n\n")

        if planting_dates:
            f.write("### Planting Season\n\n")
            f.write(f"- **Images**: {len(planting_dates)}\n")
            f.write(
                f"- **Date Range**: {min(planting_dates).date()} to {max(planting_dates).date()}\n"
            )
            if planting_cloud_cover:
                f.write(
                    f"- **Cloud Cover**: {np.mean(planting_cloud_cover):.2f}% ± {np.std(planting_cloud_cover):.2f}%\n"
                )
                f.write(
                    f"  - Min: {np.min(planting_cloud_cover):.2f}%, Max: {np.max(planting_cloud_cover):.2f}%\n"
                )
            f.write("\n")

        if harvest_dates:
            f.write("### Harvest Season\n\n")
            f.write(f"- **Images**: {len(harvest_dates)}\n")
            f.write(
                f"- **Date Range**: {min(harvest_dates).date()} to {max(harvest_dates).date()}\n"
            )
            if harvest_cloud_cover:
                f.write(
                    f"- **Cloud Cover**: {np.mean(harvest_cloud_cover):.2f}% ± {np.std(harvest_cloud_cover):.2f}%\n"
                )
                f.write(
                    f"  - Min: {np.min(harvest_cloud_cover):.2f}%, Max: {np.max(harvest_cloud_cover):.2f}%\n"
                )
            f.write("\n")
