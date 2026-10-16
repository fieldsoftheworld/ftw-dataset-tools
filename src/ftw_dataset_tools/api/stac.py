"""Core API for generating STAC static catalogs from dataset outputs."""

from __future__ import annotations

import asyncio
import json
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import duckdb
import pystac
from pystac import (
    Asset,
    Catalog,
    Collection,
    Extent,
    Item,
    ItemAssetDefinition,
    Link,
    Provider,
    ProviderRole,
    SpatialExtent,
    TemporalExtent,
)
from pystac.extensions.version import VersionExtension
from pystac.layout import TemplateLayoutStrategy

from ftw_dataset_tools.api import land_cover
from ftw_dataset_tools.api.assets import (
    add_file_info,
    add_mask_classification,
    add_raster_bands,
    add_table_columns,
)
from ftw_dataset_tools.api.geo import detect_geometry_column, ensure_spatial_loaded, sql_path
from ftw_dataset_tools.api.logging_config import get_logger, success
from ftw_dataset_tools.api.masks import MaskType, get_mgrs_square, instance_labels_path
from ftw_dataset_tools.api.renders import (
    RENDER_ORDER_PROP,
    add_render_schema,
    build_collection_renders,
    build_item_renders,
    build_render_order,
)
from ftw_dataset_tools.api.scale import SCALE_VERSION

if TYPE_CHECKING:
    from ftw_dataset_tools.api.config import DatasetConfig, MetadataConfig

logger = get_logger(__name__)

# Media types
MEDIA_TYPE_PARQUET = "application/vnd.apache.parquet"
MEDIA_TYPE_COG = "image/tiff; application=geotiff; profile=cloud-optimized"

# Layout strategy for the single collection: sub-catalogs per MGRS square, items
# co-located with their assets inside the square's sub-catalog directory.
CHIP_LAYOUT = TemplateLayoutStrategy(
    catalog_template="chips/${id}/catalog.json", item_template="${id}/${id}.json"
)

# Portolan schema declared on every STAC object this tool writes.
PORTOLAN_SCHEMA_URI = "https://schemas.portolan-sdi.org/portolan/v0.1.2/schema.json"


def chips_base_dir_for(output_dir: Path | str) -> Path:
    """Return the chips base directory :data:`CHIP_LAYOUT` writes items into.

    Single source of truth for the ``<output>/chips`` convention, so the stage
    that writes masks and the one that builds the catalog cannot disagree about
    where a chip's files live.
    """
    return Path(output_dir) / "chips"


__all__ = [
    "PORTOLAN_SCHEMA_URI",
    "STACGenerationResult",
    "TemporalExtentError",
    "chips_base_dir_for",
    "generate_stac_catalog",
    "get_temporal_extent_from_year",
    "get_year_from_datetime_column",
    "resolve_year",
]


class TemporalExtentError(ValueError):
    """A fields file's datetime column exists but yields no year."""

    def __init__(self, path: str | Path, message: str) -> None:
        super().__init__(message)
        self.path = path


# Registry of mask asset names. Module-level so the drift-guard tests can
# assert it stays in sync with the other mask-type registries: a missing entry
# here silently drops that mask from the STAC items.
_MASK_TYPE_BY_ASSET_NAME = {
    "instance": MaskType.INSTANCE,
    "semantic_2class": MaskType.SEMANTIC_2_CLASS,
    "semantic_3class": MaskType.SEMANTIC_3_CLASS,
    "decode_boundary": MaskType.DECODE_BOUNDARY,
    "decode_distance": MaskType.DECODE_DISTANCE,
}

_MASK_TITLES = {
    "instance": "Instance segmentation mask",
    "semantic_2class": "Binary semantic mask (field/background)",
    "semantic_3class": "3-class semantic mask (field/boundary/background)",
    "decode_boundary": "DECODE field boundary mask",
    "decode_distance": "DECODE normalized distance-to-boundary map",
}


@dataclass
class STACGenerationResult:
    """Result of STAC collection generation."""

    collection_path: Path
    items_parquet_path: Path | None
    subcatalog_paths: dict[str, Path]
    total_items: int
    temporal_extent: tuple[datetime, datetime]


@dataclass
class ChipInfo:
    """Information about a single chip for STAC item creation."""

    grid_id: str
    geometry: dict  # GeoJSON geometry
    bbox: tuple[float, float, float, float]  # xmin, ymin, xmax, ymax
    year: int | None = None  # Optional year for year-based naming
    properties: dict = field(default_factory=dict)

    @property
    def item_id(self) -> str:
        """Get the STAC item ID, including year if set."""
        if self.year is not None:
            return f"{self.grid_id}_{self.year}"
        return self.grid_id

    @property
    def dir_name(self) -> str:
        """Get the directory name for this chip, including year if set."""
        return self.item_id


def detect_datetime_column(file_path: str | Path) -> str | None:
    """
    Check if file has determination_datetime column.

    Args:
        file_path: Path to parquet file

    Returns:
        Column name if found, None otherwise
    """
    conn = duckdb.connect(":memory:")
    try:
        schema = conn.execute(f"DESCRIBE SELECT * FROM '{sql_path(file_path)}'").fetchall()
        col_names = [row[0].lower() for row in schema]

        # Check for fiboa determination_datetime column
        if "determination_datetime" in col_names:
            return "determination_datetime"

        return None
    finally:
        conn.close()


def get_temporal_extent_from_data(
    file_path: str | Path,
    datetime_col: str = "determination_datetime",
) -> tuple[datetime, datetime]:
    """
    Extract min/max datetime from data column.

    Args:
        file_path: Path to parquet file
        datetime_col: Name of datetime column

    Returns:
        Tuple of (start_datetime, end_datetime)
    """
    conn = duckdb.connect(":memory:")
    try:
        result = conn.execute(f"""
            SELECT
                MIN("{datetime_col}") as min_dt,
                MAX("{datetime_col}") as max_dt
            FROM '{sql_path(file_path)}'
        """).fetchone()

        if result and result[0] and result[1]:
            min_dt = result[0]
            max_dt = result[1]

            # Ensure timezone aware
            if min_dt.tzinfo is None:
                min_dt = min_dt.replace(tzinfo=UTC)
            if max_dt.tzinfo is None:
                max_dt = max_dt.replace(tzinfo=UTC)

            return (min_dt, max_dt)

        raise ValueError(f"Could not extract datetime range from column '{datetime_col}'")
    finally:
        conn.close()


def get_year_from_datetime_column(
    file_path: str | Path,
    datetime_col: str = "determination_datetime",
) -> int | None:
    """
    Extract year from datetime column using the most common year.

    This is useful for image selection where we need a single year
    to query crop calendar and STAC catalogs.

    Args:
        file_path: Path to parquet file
        datetime_col: Name of datetime column

    Returns:
        Most common year in the column, or None if extraction fails
    """
    conn = duckdb.connect(":memory:")
    try:
        # Get the most common year (mode) from the datetime column
        result = conn.execute(f"""
            SELECT EXTRACT(YEAR FROM "{datetime_col}") as year, COUNT(*) as cnt
            FROM '{sql_path(file_path)}'
            WHERE "{datetime_col}" IS NOT NULL
            GROUP BY year
            ORDER BY cnt DESC
            LIMIT 1
        """).fetchone()

        if result and result[0]:
            return int(result[0])

        return None
    except Exception:
        return None
    finally:
        conn.close()


def resolve_year(fields_file: str | Path, year: int | None = None) -> int | None:
    """Return ``year``, else the most common year in the fields file's datetime column.

    Returns None when no year is given and the file has no datetime column.

    Raises:
        TemporalExtentError: If no year is given and the datetime column has no values.
    """
    if year is not None:
        return year
    datetime_col = detect_datetime_column(fields_file)
    if datetime_col is None:
        return None
    derived = get_year_from_datetime_column(fields_file, datetime_col)
    if derived is None:
        raise TemporalExtentError(
            fields_file,
            f"Cannot determine a year: '{datetime_col}' in {fields_file} has no "
            "usable values. Provide a year.",
        )
    return derived


def get_temporal_extent_from_year(year: int) -> tuple[datetime, datetime]:
    """
    Create temporal extent spanning full year.

    Args:
        year: The year to create extent for

    Returns:
        Tuple of (start_datetime, end_datetime) for Jan 1 to Dec 31
    """
    start = datetime(year, 1, 1, 0, 0, 0, tzinfo=UTC)
    end = datetime(year, 12, 31, 23, 59, 59, tzinfo=UTC)
    return (start, end)


def _crop_season_extent(
    bbox: tuple[float, float, float, float],
    year: int,
) -> tuple[datetime, datetime] | None:
    """Planting day to the end of harvest day for ``bbox``, or None outside the crop calendar."""
    # Imported here to avoid a circular import: api.imagery imports back into this module.
    from ftw_dataset_tools.api.imagery.crop_calendar import get_crop_calendar_dates

    try:
        planting, harvest = get_crop_calendar_dates(bbox).to_datetime(year)
    except ValueError:
        return None
    return (
        planting.replace(tzinfo=UTC),
        harvest.replace(hour=23, minute=59, second=59, tzinfo=UTC),
    )


def _covering_extent(
    extent: tuple[datetime, datetime], items: list[Item]
) -> tuple[datetime, datetime]:
    """Widen ``extent`` to cover every item's start/end datetimes."""
    starts = [extent[0], *(item.common_metadata.start_datetime for item in items)]
    ends = [extent[1], *(item.common_metadata.end_datetime for item in items)]
    return (min(s for s in starts if s), max(e for e in ends if e))


def _get_dataset_bounds(file_path: Path, geom_col: str = "geometry") -> list[float]:
    """Get overall bounding box from a parquet file."""
    conn = duckdb.connect(":memory:")
    ensure_spatial_loaded(conn)
    try:
        result = conn.execute(f"""
            SELECT
                MIN(ST_XMin("{geom_col}")) as xmin,
                MIN(ST_YMin("{geom_col}")) as ymin,
                MAX(ST_XMax("{geom_col}")) as xmax,
                MAX(ST_YMax("{geom_col}")) as ymax
            FROM '{sql_path(file_path)}'
        """).fetchone()

        if result:
            return [result[0], result[1], result[2], result[3]]
        return [-180, -90, 180, 90]
    finally:
        conn.close()


#: Chips columns that, when present, are copied onto item properties (NULLs omitted).
OPTIONAL_CHIP_COLUMNS = {
    "split": "ftw:split",
    "field_coverage_pct": "ftw:field_coverage_pct",
    "hcat_dominant_code": "ftw:hcat_dominant_code",
    "hcat_dominant_name_en": "ftw:hcat_dominant_name_en",
    "hcat_dominant_pct": "ftw:hcat_dominant_pct",
    "hcat_top": "ftw:hcat_top",
    "scale_score": "ftw:scale_score",
    **{col: f"ftw:{col}" for col in land_cover.OUTPUT_COLUMNS},
}

#: Columns that need an explicit cast (test fixtures built with geopandas can store
#: integer/float columns containing None as float64; DuckDB otherwise returns those
#: as-is, which would make item JSON carry a float instead of an int).
_OPTIONAL_COLUMN_CASTS = {
    "hcat_dominant_code": "BIGINT",
    "hcat_dominant_pct": "DOUBLE",
    "field_coverage_pct": "DOUBLE",
    "landcover_year": "INTEGER",
    "landcover_year_exact": "BOOLEAN",
    "landcover_dominant_code": "BIGINT",
    "landcover_dominant_pct": "DOUBLE",
}


def _is_publishable(value: object) -> bool:
    """Whether an optional chip column value belongs on the item.

    NULLs are omitted, and so are NaN floats: JSON has no NaN, so publishing one
    would produce an item that no strict JSON parser can read back.
    """
    if value is None:
        return False
    return not (isinstance(value, float) and math.isnan(value))


def _optional_column_select(col: str) -> str:
    """SQL select expression for one optional chip column, casting where needed."""
    cast = _OPTIONAL_COLUMN_CASTS.get(col)
    if cast:
        return f'CAST("{col}" AS {cast}) AS "{col}"'
    return f'"{col}"'


def _extract_chips_info(
    chips_file: Path,
    grid_id_col: str = "id",
    geom_col: str = "geometry",
    year: int | None = None,
) -> list[ChipInfo]:
    """
    Extract chip information from chips parquet file.

    Args:
        chips_file: Path to chips parquet file
        grid_id_col: Column name for grid ID
        geom_col: Column name for geometry
        year: Optional year for year-based naming convention

    Returns:
        List of ChipInfo objects
    """
    conn = duckdb.connect(":memory:")
    ensure_spatial_loaded(conn)
    chips_sql = sql_path(chips_file)
    try:
        existing_cols = {
            row[0] for row in conn.execute(f"DESCRIBE SELECT * FROM '{chips_sql}'").fetchall()
        }
        optional_cols = [col for col in OPTIONAL_CHIP_COLUMNS if col in existing_cols]
        optional_select = "".join(f", {_optional_column_select(col)}" for col in optional_cols)

        # Get chip info with geometry as GeoJSON
        results = conn.execute(f"""
            SELECT
                "{grid_id_col}" as grid_id,
                ST_AsGeoJSON("{geom_col}") as geojson,
                ST_XMin("{geom_col}") as xmin,
                ST_YMin("{geom_col}") as ymin,
                ST_XMax("{geom_col}") as xmax,
                ST_YMax("{geom_col}") as ymax
                {optional_select}
            FROM '{chips_sql}'
        """).fetchall()

        chips = []
        for row in results:
            grid_id, geojson, xmin, ymin, xmax, ymax, *optional_values = row
            geometry = json.loads(geojson)
            properties = {
                OPTIONAL_CHIP_COLUMNS[col]: value
                for col, value in zip(optional_cols, optional_values, strict=True)
                if _is_publishable(value)
            }
            chips.append(
                ChipInfo(
                    grid_id=str(grid_id),
                    geometry=geometry,
                    bbox=(xmin, ymin, xmax, ymax),
                    year=year,
                    properties=properties,
                )
            )
        return chips
    finally:
        conn.close()


def _add_portolan_schema(obj: pystac.STACObject) -> None:
    """Declare the Portolan schema on a STAC object exactly once."""
    if PORTOLAN_SCHEMA_URI not in obj.stac_extensions:
        obj.stac_extensions.append(PORTOLAN_SCHEMA_URI)


def _updated_stamp(provenance: dict | None) -> str:
    """ISO 8601 UTC 'updated' value: the build's generated_at, else now (trailing Z)."""
    raw = (provenance or {}).get("generated_at")
    stamp = datetime.fromisoformat(raw) if raw else datetime.now(UTC)
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=UTC)
    return stamp.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _apply_collection_metadata(
    collection: Collection,
    metadata: MetadataConfig | None,
    *,
    updated: str,
    title: str | None = None,
    description: str | None = None,
) -> None:
    """Stamp config metadata, the version extension and 'updated' on a collection."""
    collection.extra_fields["updated"] = updated
    if metadata is None:
        return
    if title:
        collection.title = title
    if description:
        collection.description = description
    if metadata.license:
        collection.license = metadata.license
    if metadata.license_url:
        collection.add_link(Link(rel="license", target=metadata.license_url, title="License"))
    if metadata.keywords:
        collection.keywords = list(metadata.keywords)
    if metadata.providers:
        collection.providers = [
            Provider(name=p.name, roles=[ProviderRole(r) for r in p.roles], url=p.url)
            for p in metadata.providers
        ]
    if metadata.version:
        VersionExtension.ext(collection, add_if_missing=True).version = metadata.version


def _collection_ftw_properties(config: DatasetConfig) -> dict:
    """Build settings a consumer needs to interpret the chips, as ftw: fields."""
    stages = config.stages
    props: dict = {
        "ftw:split_type": stages.splits.split_type,
        "ftw:split_seed": stages.splits.random_seed,
        "ftw:split_percents": list(stages.splits.split_percents),
        "ftw:chip_size_km": stages.chips.km_size,
        "ftw:mask_types": list(stages.masks.mask_types),
        "ftw:mask_resolution_m": stages.masks.resolution,
        "ftw:presence_only": stages.masks.presence_only,
        "ftw:min_coverage_pct": stages.chips.min_coverage,
    }
    if stages.scale.percent < 100:
        props.update(
            {
                "ftw:scale_percent": stages.scale.percent,
                "ftw:scale_min_blocks_per_square": stages.scale.min_blocks_per_square,
                "ftw:scale_version": SCALE_VERSION,
            }
        )
    if stages.chips.land_cover:
        props["ftw:landcover_product"] = land_cover.PRODUCT
    if stages.select_images.enabled:
        sel = stages.select_images
        props.update(
            {
                "ftw:cloud_cover_chip_threshold": sel.cloud_cover_chip,
                "ftw:nodata_max": sel.nodata_max,
                "ftw:buffer_days": sel.buffer_days,
                "ftw:num_buffer_expansions": sel.num_buffer_expansions,
                "ftw:buffer_expansion_size": sel.buffer_expansion_size,
            }
        )
    return {k: v for k, v in props.items() if v is not None}


def _read_previous_collection(output_dir: Path, collection_id: str) -> dict | None:
    """The collection.json an earlier run wrote for this dataset, if any."""
    path = output_dir / "collection.json"
    if not path.is_file():
        return None
    try:
        previous = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    return previous if previous.get("id") == collection_id else None


def _previous_filtered_fields(previous: dict, output_dir: Path) -> Path | None:
    """The filtered fields file the earlier collection published, if still on disk."""
    asset = previous.get("assets", {}).get("fields_filtered")
    if not asset:
        return None
    path = output_dir / asset["href"]
    return path if path.is_file() else None


def _has_license(collection: dict | None) -> bool:
    """Whether a collection dict carries a license beyond pystac's 'other' default."""
    if not collection:
        return False
    links = collection.get("links", [])
    return collection.get("license", "other") != "other" or any(
        link.get("rel") == "license" for link in links
    )


def _restore_collection_metadata(collection: Collection, previous: dict) -> None:
    """Carry over the metadata only a config-driven build can produce."""
    for key in ("title", "description", "license"):
        if previous.get(key):
            setattr(collection, key, previous[key])
    if previous.get("keywords"):
        collection.keywords = list(previous["keywords"])
    if previous.get("providers"):
        collection.providers = [Provider.from_dict(p) for p in previous["providers"]]
    if previous.get("version"):
        VersionExtension.ext(collection, add_if_missing=True).version = previous["version"]
    for link in previous.get("links", []):
        if link.get("rel") in ("license", "via"):
            collection.add_link(Link.from_dict(link))
    collection.extra_fields.update({k: v for k, v in previous.items() if k.startswith("ftw:")})


def _group_items_by_square(items: list[Item], squares: dict[str, str]) -> dict[str, list[Item]]:
    """Group chip items by MGRS 100 km square, keyed by the square id, sorted.

    Args:
        items: Chip items to group.
        squares: Mapping of item id to MGRS square, as derived when the items were
            created (so grouping agrees with where each item's directory lives).
    """
    groups: dict[str, list[Item]] = {}
    for item in items:
        groups.setdefault(squares.get(item.id, "other"), []).append(item)
    return {square: groups[square] for square in sorted(groups)}


def _build_item_assets() -> dict[str, ItemAssetDefinition]:
    """Declare the assets a chip item may carry (core ``item_assets`` collection field)."""
    defs: dict[str, ItemAssetDefinition] = {}
    for mask_name in _MASK_TYPE_BY_ASSET_NAME:
        defs[f"{mask_name}_mask"] = ItemAssetDefinition(
            {"type": MEDIA_TYPE_COG, "roles": ["labels"], "title": _MASK_TITLES[mask_name]}
        )
    for season in ("planting", "harvest"):
        defs[f"{season}_image"] = ItemAssetDefinition(
            {
                "type": MEDIA_TYPE_COG,
                "roles": ["data"],
                "title": f"{season.capitalize()} season imagery",
            }
        )
        defs[f"{season}_visual"] = ItemAssetDefinition(
            {
                "type": MEDIA_TYPE_COG,
                "roles": ["visual"],
                "title": f"{season.capitalize()} season scene (true colour)",
            }
        )
    defs["thumbnail"] = ItemAssetDefinition({"roles": ["thumbnail"], "title": "Chip preview"})
    return defs


def _add_parquet_asset(
    collection: Collection,
    key: str,
    path: Path,
    title: str,
    *,
    checksums: bool = False,
    roles: list[str] | None = None,
) -> None:
    """Add a GeoParquet asset to a collection, with file info and column stats."""
    collection.add_asset(
        key=key,
        asset=Asset(
            href=f"./{path.name}",
            media_type=MEDIA_TYPE_PARQUET,
            title=title,
            roles=roles if roles is not None else ["data"],
        ),
    )
    add_file_info(collection.assets[key], path, checksum=checksums)
    add_table_columns(collection.assets[key], path)


def _create_collection(
    dataset_name: str,
    fields_file: Path,
    boundary_lines_file: Path,
    chips_file: Path,
    temporal_extent: tuple[datetime, datetime],
    spatial_extent: list[float],
    *,
    filtered_fields_file: Path | None = None,
    checksums: bool = False,
    background_class_value: int = 0,
) -> Collection:
    """
    Create the single dataset collection with source-data and chip-definition assets.

    Args:
        dataset_name: Name of the dataset (used as the collection id)
        fields_file: Path to fields parquet file
        boundary_lines_file: Path to boundary lines parquet file
        chips_file: Path to chips parquet file
        temporal_extent: Tuple of (start, end) datetime
        spatial_extent: Bounding box [xmin, ymin, xmax, ymax]
        filtered_fields_file: Optional path to the class-filtered fields parquet file
        checksums: Compute file:checksum (multihash sha256) for every asset.
        background_class_value: Pixel value used for background in masks
            (3 for presence-only); the collection renders hide it.

    Returns:
        pystac Collection
    """
    collection = Collection(
        id=dataset_name,
        description=f"Benchmark chips with label masks for {dataset_name}",
        title=dataset_name,
        extent=Extent(
            spatial=SpatialExtent(bboxes=[spatial_extent]),
            temporal=TemporalExtent(intervals=[[temporal_extent[0], temporal_extent[1]]]),
        ),
    )

    _add_parquet_asset(
        collection, "fields", fields_file, "Field boundary polygons", checksums=checksums
    )

    if filtered_fields_file is not None:
        _add_parquet_asset(
            collection,
            "fields_filtered",
            filtered_fields_file,
            "Field polygons after the class filter",
            checksums=checksums,
        )

    _add_parquet_asset(
        collection,
        "boundary_lines",
        boundary_lines_file,
        "Field boundary lines",
        checksums=checksums,
    )

    _add_parquet_asset(
        collection,
        "chips",
        chips_file,
        "Chip definitions with field coverage",
        checksums=checksums,
    )

    collection.item_assets = _build_item_assets()
    # A Collection carries renders at its top level (the schema's Collection branch).
    collection.extra_fields["renders"] = build_collection_renders(
        background_value=background_class_value
    )
    add_render_schema(collection)

    return collection


def _create_chip_item(
    chip_info: ChipInfo,
    temporal_extent: tuple[datetime, datetime],
    chip_dir: Path,
    checksums: bool = False,
    background_class_value: int = 0,
) -> Item | None:
    """
    Create a STAC Item for a single chip.

    Args:
        chip_info: ChipInfo with geometry and bbox (includes optional year)
        temporal_extent: Tuple of (start, end) datetime
        chip_dir: Directory containing co-located masks
        checksums: Compute file:checksum (multihash sha256) for every mask asset.
        background_class_value: Pixel value used for background in masks
            (3 for presence-only).

    Returns:
        pystac Item, or None if no mask files exist
    """
    grid_id = chip_info.grid_id
    item_id = chip_info.item_id  # Includes year if set
    year = chip_info.year

    # Check which mask files exist (masks co-located with the item)
    mask_assets = {}
    for mask_name, mask_type in _MASK_TYPE_BY_ASSET_NAME.items():
        # Filename includes year if year is set
        if year is not None:
            mask_filename = f"{grid_id}_{year}_{mask_type.value}.tif"
        else:
            mask_filename = f"{grid_id}_{mask_type.value}.tif"
        mask_path = chip_dir / mask_filename
        if mask_path.exists():
            mask_assets[f"{mask_name}_mask"] = (
                Asset(
                    href=f"./{mask_filename}",
                    media_type=MEDIA_TYPE_COG,
                    title=_get_mask_title(mask_name),
                    roles=["labels"],
                ),
                mask_path,
            )

    # Skip if no masks exist
    if not mask_assets:
        return None

    # Build properties
    properties = {
        "start_datetime": temporal_extent[0].isoformat(),
        "end_datetime": temporal_extent[1].isoformat(),
    }

    # Add FTW extension properties if year is set
    if year is not None:
        properties["ftw:calendar_year"] = year

    # Merge in split, coverage and crop composition properties, if the chips file had them
    properties.update(chip_info.properties)

    # Create item with datetime range
    item = Item(
        id=item_id,  # Use item_id which includes year if set
        geometry=chip_info.geometry,
        bbox=list(chip_info.bbox),
        datetime=None,  # Use start/end instead
        properties=properties,
    )

    # Add mask assets, then decorate them (owner must be set first)
    for key, (asset, mask_path) in mask_assets.items():
        item.add_asset(key=key, asset=asset)
        add_file_info(asset, mask_path, checksum=checksums)
        add_raster_bands(asset, mask_path)
        add_mask_classification(
            asset, key.removesuffix("_mask"), background_value=background_class_value
        )
        if key == "instance_mask":
            labels_path = instance_labels_path(mask_path)
            if labels_path.exists():
                labels = Asset(
                    href=f"./{labels_path.name}",
                    media_type="application/json",
                    title="Field instance crop labels",
                    roles=["labels", "metadata"],
                )
                item.add_asset("instance_labels", labels)
                add_file_info(labels, labels_path, checksum=checksums)

    # Season children survive a STAC rerun on disk; put their links and imagery
    # assets back onto the item this run rebuilt from the mask files alone.
    _reattach_existing_seasons(item, chip_dir, checksums=checksums)

    # The render extension's Feature branch requires properties.renders, not a
    # top-level key, so this must not go through extra_fields.
    renders = build_item_renders(item, background_value=background_class_value)
    if renders:
        item.properties["renders"] = renders
        add_render_schema(item)
        # The default layer stack a Portolan browser opens the chip on: fields over
        # the season's imagery. Absent on a chip with no imagery to stack them over.
        render_order = build_render_order(renders)
        if render_order:
            item.properties[RENDER_ORDER_PROP] = render_order

    return item


def _reattach_existing_seasons(item: Item, chip_dir: Path, *, checksums: bool = False) -> None:
    """Restore the season links and imagery assets left on disk by the imagery stages.

    Imported inside the function: ``api.imagery.stac_child_items`` imports this
    module, so a module-level import would be circular.
    """
    from ftw_dataset_tools.api.imagery.stac_child_items import attach_existing_seasons

    attach_existing_seasons(item, chip_dir, checksums=checksums)


def _get_mask_title(mask_name: str) -> str:
    """Get human-readable title for mask type."""
    return _MASK_TITLES.get(mask_name, f"{mask_name} mask")


async def _write_items_parquet_async(
    items: list[Item],
    output_path: Path,
) -> None:
    """Write STAC items to stac-geoparquet format using rustac."""
    import rustac

    # Convert pystac Items to dicts. include_self_link=False matches what
    # SELF_CONTAINED save() writes to disk: a self link would be the one href
    # pystac leaves absolute, putting the build machine's paths in the mirror.
    item_dicts = [item.to_dict(include_self_link=False) for item in items]

    # Write using rustac async API
    await rustac.write(str(output_path), item_dicts)


def _write_items_parquet(
    items: list[Item],
    output_path: Path,
) -> None:
    """Synchronous wrapper for writing stac-geoparquet."""
    asyncio.run(_write_items_parquet_async(items, output_path))


def _resolve_temporal_extent(fields_file: Path, year: int | None) -> tuple[datetime, datetime]:
    """Dataset temporal extent from the fields' datetime column, else from ``year``."""
    logger.info("Determining temporal extent...")
    datetime_col = detect_datetime_column(fields_file)
    # An all-NULL column has no extent to read, so an explicit year takes over.
    if (
        datetime_col
        and year is not None
        and get_year_from_datetime_column(fields_file, datetime_col) is None
    ):
        datetime_col = None

    if datetime_col:
        logger.info(f"Using '{datetime_col}' column for temporal extent")
        return get_temporal_extent_from_data(fields_file, datetime_col)
    if year is not None:
        logger.info(f"Using year {year} for temporal extent")
        return get_temporal_extent_from_year(year)
    raise ValueError(
        "Cannot determine temporal extent. Either provide --year parameter "
        "or ensure fields file has 'determination_datetime' column."
    )


def _build_chip_items(
    chip_infos: list[ChipInfo],
    chips_base_dir: Path,
    temporal_extent: tuple[datetime, datetime],
    *,
    crop_calendar: bool,
    checksums: bool,
    background_class_value: int,
) -> tuple[list[pystac.Item], dict[str, str]]:
    """One item per chip that has a mask directory, and the MGRS square of each item."""
    logger.info("Creating STAC items...")
    # Imported here to avoid a circular import: api.imagery imports back into
    # this module.
    from ftw_dataset_tools.api.imagery.catalog_ops import preserve_imagery_selection

    items = []
    resumed = 0
    outside_calendar = 0
    item_squares: dict[str, str] = {}
    for chip_info in chip_infos:
        square = get_mgrs_square(chip_info.grid_id)
        chip_dir = chips_base_dir / square / chip_info.dir_name
        if not chip_dir.exists():
            # Skip chips without directories (no masks generated)
            continue

        chip_extent = temporal_extent
        if crop_calendar and chip_info.year is not None:
            season = _crop_season_extent(chip_info.bbox, chip_info.year)
            if season is None:
                outside_calendar += 1
            else:
                chip_extent = season

        item = _create_chip_item(
            chip_info=chip_info,
            temporal_extent=chip_extent,
            chip_dir=chip_dir,
            checksums=checksums,
            background_class_value=background_class_value,
        )
        if item:
            # Items are rebuilt from scratch, so carry over any imagery
            # selection the previous run recorded; otherwise saving the catalog
            # would wipe it and every chip would re-select.
            # Read the previous run's item from the chip directory actually in
            # use: CHIP_LAYOUT writes each item next to its masks, nested under
            # its MGRS square, so that is where the last run's JSON is.
            if preserve_imagery_selection(item, chip_dir / f"{item.id}.json"):
                resumed += 1
            items.append(item)
            item_squares[item.id] = square

    logger.info(f"Created {len(items)} items with mask assets")
    if resumed:
        logger.info(f"Preserved existing imagery selections for {resumed} items")
    if outside_calendar:
        logger.info(f"{outside_calendar} chip(s) outside the crop calendar keep the dataset extent")
    return items, item_squares


def _apply_collection_context(
    collection: pystac.Collection,
    config: DatasetConfig | None,
    metadata: MetadataConfig | None,
    previous: dict | None,
    provenance: dict | None,
) -> None:
    """Configured metadata, source link, ftw: properties and provenance on the collection."""
    updated = _updated_stamp(provenance)
    # No configured title means the title the collection constructor gave it stands;
    # passing an absent title through would leave the collection untitled.
    base_title = metadata.title if metadata and metadata.title else None
    _apply_collection_metadata(
        collection,
        metadata,
        updated=updated,
        title=base_title,
        description=metadata.description if metadata else None,
    )
    if config is not None and config.source_via:
        collection.add_link(
            Link(
                rel="via",
                target=config.source_via,
                media_type="application/json",
                title="Source field boundary collection",
            )
        )
    if config is not None and config.stages.chips.land_cover:
        collection.add_link(
            Link(
                rel="derived_from",
                target=land_cover.COLLECTION_URL,
                media_type="application/json",
                title=land_cover.ATTRIBUTION,
            )
        )
    if config is not None:
        collection.extra_fields.update(_collection_ftw_properties(config))
    if previous is not None:
        _restore_collection_metadata(collection, previous)
    if provenance is not None:
        collection.extra_fields["ftw:config"] = provenance


def _add_square_subcatalogs(
    collection: pystac.Collection,
    items: list[pystac.Item],
    item_squares: dict[str, str],
) -> dict[str, list[pystac.Item]]:
    """Group items by MGRS square under one sub-catalog per square."""
    groups = _group_items_by_square(items, item_squares)
    for square, square_items in groups.items():
        sub = Catalog(
            id=square,
            description=f"Chips in MGRS 100 km square {square}",
            title=square,
        )
        _add_portolan_schema(sub)
        for item in square_items:
            sub.add_item(item)
            item.set_collection(collection)
            _add_portolan_schema(item)
        collection.add_child(sub)
    return groups


def generate_stac_catalog(
    output_dir: Path | str,
    field_dataset: str,
    fields_file: Path | str,
    chips_file: Path | str,
    boundary_lines_file: Path | str,
    *,
    filtered_fields_file: Path | None = None,
    grid_id_col: str = "id",
    year: int | None = None,
    provenance: dict | None = None,
    config: DatasetConfig | None = None,
    checksums: bool = False,
    background_class_value: int = 0,
    crop_calendar: bool = True,
) -> STACGenerationResult:
    """
    Generate a single self-contained STAC collection from dataset outputs.

    The output is one collection at ``output_dir/collection.json`` whose children
    are per-MGRS-square sub-catalogs holding the chip items (custom, non-FTW grid
    ids are grouped under an ``other`` sub-catalog).

    Chip assets are read from, and written back to, ``output_dir/chips/{square}/
    {item_id}/``. That directory is derived from ``output_dir`` rather than passed
    in, because :data:`CHIP_LAYOUT` writes each item there: a caller-supplied base
    that pointed elsewhere would leave every asset href dangling.

    Args:
        output_dir: Base directory for dataset and STAC output
        field_dataset: Dataset name (used as the collection id)
        fields_file: Path to fields parquet file
        chips_file: Path to chips parquet file
        boundary_lines_file: Path to boundary lines parquet file
        filtered_fields_file: Optional path to the class-filtered fields parquet file;
            when given, a ``fields_filtered`` asset is added to the collection.
        grid_id_col: Column in the chips file holding the grid cell id. Must match the
            column the masks were written from, or no item lines up with its chip dir.
        year: Optional year for temporal extent (required if no determination_datetime)
        provenance: Optional resolved-config record embedded on the collection under
                    the ``ftw:config`` extra field for reproducibility.
        config: Resolved dataset config; supplies metadata and the ftw: build properties
            written on the collection.
        checksums: Compute file:checksum (multihash sha256) for every asset. Slow; default False.
        background_class_value: Pixel value used for background in masks (3 for presence-only).
        crop_calendar: Date each year-tagged chip item by its crop season (planting to
            harvest) rather than the dataset extent. Chips the calendar does not cover
            keep the dataset extent.

    Returns:
        STACGenerationResult with paths to generated files

    Raises:
        ValueError: If year not provided and no determination_datetime column
    """
    output_dir = Path(output_dir)
    fields_file = Path(fields_file)
    chips_file = Path(chips_file)
    boundary_lines_file = Path(boundary_lines_file)
    chips_base_dir = chips_base_dir_for(output_dir)
    filtered_fields_file = Path(filtered_fields_file) if filtered_fields_file else None

    temporal_extent = _resolve_temporal_extent(fields_file, year)

    # Get spatial extent from fields
    logger.info("Calculating spatial extent...")
    spatial_extent = _get_dataset_bounds(
        fields_file, detect_geometry_column(fields_file) or "geometry"
    )

    # Extract chip info (pass year for year-based naming). The geometry column is
    # detected rather than assumed, matching how the masks were rasterized.
    logger.info("Extracting chip information...")
    chip_infos = _extract_chips_info(
        chips_file,
        grid_id_col=grid_id_col,
        geom_col=detect_geometry_column(chips_file) or "geometry",
        year=year,
    )
    logger.info(f"Found {len(chip_infos)} chips")

    # Create items for each chip, nested by MGRS square
    items, item_squares = _build_chip_items(
        chip_infos,
        chips_base_dir,
        temporal_extent,
        crop_calendar=crop_calendar,
        checksums=checksums,
        background_class_value=background_class_value,
    )

    # A southern-hemisphere harvest falls in the following year.
    temporal_extent = _covering_extent(temporal_extent, items)

    # Create the single collection
    logger.info("Creating collection...")
    # Without a config (e.g. create-masks over a pipeline output), keep what the
    # earlier build published rather than silently dropping it.
    previous = _read_previous_collection(output_dir, field_dataset) if config is None else None
    if previous is not None and filtered_fields_file is None:
        filtered_fields_file = _previous_filtered_fields(previous, output_dir)

    metadata = config.metadata if config is not None else None
    if not (metadata and metadata.license) and not _has_license(previous):
        logger.warning(
            "no metadata.license; the collection is not Portolan-publishable without one"
        )

    collection = _create_collection(
        dataset_name=field_dataset,
        fields_file=fields_file,
        boundary_lines_file=boundary_lines_file,
        chips_file=chips_file,
        temporal_extent=temporal_extent,
        spatial_extent=spatial_extent,
        filtered_fields_file=filtered_fields_file,
        checksums=checksums,
        background_class_value=background_class_value,
    )

    _apply_collection_context(collection, config, metadata, previous, provenance)

    groups = _add_square_subcatalogs(collection, items, item_squares)

    _add_portolan_schema(collection)

    # normalize_hrefs assigns each item's self href in-memory (needed for rustac to
    # serialize them below) without writing any files yet.
    logger.info("Writing STAC catalog...")
    collection.normalize_hrefs(str(output_dir), strategy=CHIP_LAYOUT)
    # Set the catalog type before serializing the items below: item.to_dict()
    # renders hierarchical link hrefs relative only when the root catalog is a
    # relative one, so leaving it until save() would bake this build machine's
    # absolute filesystem paths into the published parquet mirror.
    collection.catalog_type = pystac.CatalogType.SELF_CONTAINED

    # Write stac-geoparquet before the single collection.save() below so its file:size
    # lands in the collection.json that save writes; a second save_object() call
    # would otherwise write an absolute filesystem self link into that file.
    items_parquet_path: Path | None = None
    if items:
        logger.info("Writing stac-geoparquet...")
        items_parquet_path = output_dir / "items.parquet"
        _write_items_parquet(items, items_parquet_path)
        _add_parquet_asset(
            collection,
            "items",
            items_parquet_path,
            "STAC items in GeoParquet format (collection mirror)",
            checksums=checksums,
            roles=["collection-mirror"],
        )

    collection.save(catalog_type=collection.catalog_type)

    success(logger, "STAC catalog generation complete")

    return STACGenerationResult(
        collection_path=output_dir / "collection.json",
        items_parquet_path=items_parquet_path,
        subcatalog_paths={
            square: output_dir / "chips" / square / "catalog.json" for square in groups
        },
        total_items=len(items),
        temporal_extent=temporal_extent,
    )
