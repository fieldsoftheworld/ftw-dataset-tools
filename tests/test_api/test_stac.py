"""Tests for the STAC API."""

from datetime import UTC, datetime
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box


class TestTemporalExtent:
    """Tests for temporal extent functions."""

    def test_get_temporal_extent_from_year(self) -> None:
        """Test temporal extent from year."""
        from ftw_dataset_tools.api.stac import get_temporal_extent_from_year

        start, end = get_temporal_extent_from_year(2023)

        assert start == datetime(2023, 1, 1, 0, 0, 0, tzinfo=UTC)
        assert end == datetime(2023, 12, 31, 23, 59, 59, tzinfo=UTC)

    def test_get_temporal_extent_from_year_timezone_aware(self) -> None:
        """Test temporal extent datetimes are timezone aware."""
        from ftw_dataset_tools.api.stac import get_temporal_extent_from_year

        start, end = get_temporal_extent_from_year(2024)

        assert start.tzinfo is not None
        assert end.tzinfo is not None


class TestDetectDatetimeColumn:
    """Tests for detect_datetime_column function."""

    def test_returns_none_for_no_datetime(self, sample_geoparquet_4326: Path) -> None:
        """Test returns None when no datetime column exists."""
        from ftw_dataset_tools.api.stac import detect_datetime_column

        result = detect_datetime_column(sample_geoparquet_4326)
        assert result is None

    def test_detects_determination_datetime(self, tmp_path: Path) -> None:
        """Test detects determination_datetime column."""
        from ftw_dataset_tools.api.stac import detect_datetime_column

        gdf = gpd.GeoDataFrame(
            {"id": [1], "determination_datetime": [datetime(2023, 6, 15)]},
            geometry=[box(0, 0, 1, 1)],
            crs="EPSG:4326",
        )
        path = tmp_path / "with_datetime.parquet"
        gdf.to_parquet(path)

        result = detect_datetime_column(path)
        assert result == "determination_datetime"


class TestSTACGenerationResult:
    """Tests for STACGenerationResult dataclass."""

    def test_dataclass_fields(self) -> None:
        """Test STACGenerationResult has expected fields."""
        from ftw_dataset_tools.api.stac import STACGenerationResult

        result = STACGenerationResult(
            catalog_path=Path("/tmp/catalog.json"),
            source_collection_path=Path("/tmp/source/collection.json"),
            chips_collection_path=Path("/tmp/chips/collection.json"),
            items_parquet_path=Path("/tmp/chips/items.parquet"),
            total_items=100,
            temporal_extent=(
                datetime(2023, 1, 1, tzinfo=UTC),
                datetime(2023, 12, 31, tzinfo=UTC),
            ),
        )
        assert result.total_items == 100
        assert result.catalog_path.name == "catalog.json"


class TestGetMaskTitle:
    """Tests for _get_mask_title helper."""

    def test_instance_title(self) -> None:
        """Test instance mask title."""
        from ftw_dataset_tools.api.stac import _get_mask_title

        assert _get_mask_title("instance") == "Instance segmentation mask"

    def test_semantic_2class_title(self) -> None:
        """Test semantic 2-class mask title."""
        from ftw_dataset_tools.api.stac import _get_mask_title

        assert "Binary" in _get_mask_title("semantic_2class")

    def test_unknown_returns_formatted(self) -> None:
        """Test unknown mask type returns formatted string."""
        from ftw_dataset_tools.api.stac import _get_mask_title

        assert "custom" in _get_mask_title("custom")


class TestChipItemAssetHrefs:
    """Tests for STAC item asset href generation."""

    def test_asset_href_colocated(self, tmp_path: Path) -> None:
        """Test asset hrefs are relative to item directory when co-located."""
        from ftw_dataset_tools.api.stac import ChipInfo, _create_chip_item

        # Create chip directory with mask files
        chip_dir = tmp_path / "chips" / "grid_001"
        chip_dir.mkdir(parents=True)

        # Create dummy mask files with NEW naming convention (no dataset prefix)
        (chip_dir / "grid_001_instance.tif").touch()
        (chip_dir / "grid_001_semantic_2_class.tif").touch()
        (chip_dir / "grid_001_semantic_3_class.tif").touch()

        chip_info = ChipInfo(
            grid_id="grid_001",
            geometry={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
            bbox=(0.0, 0.0, 1.0, 1.0),
        )

        temporal_extent = (
            datetime(2023, 1, 1, tzinfo=UTC),
            datetime(2023, 12, 31, tzinfo=UTC),
        )

        # Call with chip_dir for co-located assets
        item = _create_chip_item(
            chip_info=chip_info,
            field_dataset="test_dataset",
            chip_dir=chip_dir,
            temporal_extent=temporal_extent,
        )

        assert item is not None
        # Verify relative paths are simple (same directory)
        assert item.assets["instance_mask"].href == "./grid_001_instance.tif"
        assert item.assets["semantic_2class_mask"].href == "./grid_001_semantic_2_class.tif"
        assert item.assets["semantic_3class_mask"].href == "./grid_001_semantic_3_class.tif"

    def test_asset_href_legacy_mask_dirs(self, tmp_path: Path) -> None:
        """Test asset hrefs use legacy paths when mask_dirs is provided."""
        from ftw_dataset_tools.api.stac import ChipInfo, _create_chip_item

        # Create mask directories (legacy structure)
        instance_dir = tmp_path / "label_masks" / "instance"
        semantic_2class_dir = tmp_path / "label_masks" / "semantic_2class"
        instance_dir.mkdir(parents=True)
        semantic_2class_dir.mkdir(parents=True)

        # Create dummy mask files with legacy naming convention (with dataset prefix)
        (instance_dir / "test_dataset_grid_001_instance.tif").touch()
        (semantic_2class_dir / "test_dataset_grid_001_semantic_2_class.tif").touch()

        chip_info = ChipInfo(
            grid_id="grid_001",
            geometry={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
            bbox=(0.0, 0.0, 1.0, 1.0),
        )

        temporal_extent = (
            datetime(2023, 1, 1, tzinfo=UTC),
            datetime(2023, 12, 31, tzinfo=UTC),
        )

        mask_dirs = {
            "instance": instance_dir,
            "semantic_2class": semantic_2class_dir,
        }

        # Call with mask_dirs for legacy structure
        item = _create_chip_item(
            chip_info=chip_info,
            field_dataset="test_dataset",
            mask_dirs=mask_dirs,
            temporal_extent=temporal_extent,
        )

        assert item is not None
        # Verify legacy relative paths
        assert (
            item.assets["instance_mask"].href
            == "../../label_masks/instance/test_dataset_grid_001_instance.tif"
        )
        assert (
            item.assets["semantic_2class_mask"].href
            == "../../label_masks/semantic_2class/test_dataset_grid_001_semantic_2_class.tif"
        )

    def test_returns_none_when_no_masks_exist(self, tmp_path: Path) -> None:
        """Test that None is returned when no mask files exist."""
        from ftw_dataset_tools.api.stac import ChipInfo, _create_chip_item

        # Create empty chip directory
        chip_dir = tmp_path / "chips" / "grid_001"
        chip_dir.mkdir(parents=True)

        chip_info = ChipInfo(
            grid_id="grid_001",
            geometry={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
            bbox=(0.0, 0.0, 1.0, 1.0),
        )

        temporal_extent = (
            datetime(2023, 1, 1, tzinfo=UTC),
            datetime(2023, 12, 31, tzinfo=UTC),
        )

        # Call with chip_dir but no mask files
        item = _create_chip_item(
            chip_info=chip_info,
            field_dataset="test_dataset",
            chip_dir=chip_dir,
            temporal_extent=temporal_extent,
        )

        assert item is None


class TestGenerateStacCatalogSignature:
    """Tests for generate_stac_catalog function signature."""

    def test_accepts_chips_base_dir_parameter(self) -> None:
        """Test that generate_stac_catalog accepts chips_base_dir parameter."""
        import inspect

        from ftw_dataset_tools.api.stac import generate_stac_catalog

        sig = inspect.signature(generate_stac_catalog)
        assert "chips_base_dir" in sig.parameters


class TestTemporalExtentFromData:
    """Tests for get_temporal_extent_from_data function."""

    def test_extracts_datetime_range(self, tmp_path: Path) -> None:
        """Test extracting min/max datetime from data."""
        from ftw_dataset_tools.api.stac import get_temporal_extent_from_data

        gdf = gpd.GeoDataFrame(
            {
                "id": [1, 2, 3],
                "determination_datetime": [
                    datetime(2023, 3, 15),
                    datetime(2023, 6, 20),
                    datetime(2023, 9, 10),
                ],
            },
            geometry=[box(0, 0, 1, 1), box(1, 1, 2, 2), box(2, 2, 3, 3)],
            crs="EPSG:4326",
        )
        path = tmp_path / "with_datetime.parquet"
        gdf.to_parquet(path)

        start, end = get_temporal_extent_from_data(path, "determination_datetime")

        assert start.year == 2023
        assert start.month == 3
        assert end.year == 2023
        assert end.month == 9
        assert start.tzinfo is not None
        assert end.tzinfo is not None


class TestGetDatasetBounds:
    """Tests for _get_dataset_bounds function."""

    def test_calculates_bounds(self, sample_geoparquet_4326: Path) -> None:
        """Test calculating spatial bounds from geometry."""
        from ftw_dataset_tools.api.stac import _get_dataset_bounds

        bounds = _get_dataset_bounds(sample_geoparquet_4326)

        assert len(bounds) == 4
        assert bounds[0] <= bounds[2]  # xmin <= xmax
        assert bounds[1] <= bounds[3]  # ymin <= ymax

    def test_bounds_match_data(self, tmp_path: Path) -> None:
        """Test that bounds match the actual data extent."""
        from ftw_dataset_tools.api.stac import _get_dataset_bounds

        gdf = gpd.GeoDataFrame(
            {"id": [1, 2]},
            geometry=[box(10.0, 50.0, 11.0, 51.0), box(12.0, 52.0, 13.0, 53.0)],
            crs="EPSG:4326",
        )
        path = tmp_path / "bounds_test.parquet"
        gdf.to_parquet(path)

        bounds = _get_dataset_bounds(path)

        assert bounds[0] == 10.0  # xmin
        assert bounds[1] == 50.0  # ymin
        assert bounds[2] == 13.0  # xmax
        assert bounds[3] == 53.0  # ymax


class TestExtractChipsInfo:
    """Tests for _extract_chips_info function."""

    def test_extracts_chip_info(self, sample_grid_geoparquet: Path) -> None:
        """Test extracting chip info from parquet file."""
        from ftw_dataset_tools.api.stac import _extract_chips_info

        chips = _extract_chips_info(sample_grid_geoparquet)

        assert len(chips) == 2
        assert chips[0].grid_id == "grid_001"
        assert chips[1].grid_id == "grid_002"
        assert "type" in chips[0].geometry
        assert len(chips[0].bbox) == 4


class TestCreateRootCatalog:
    """Tests for _create_root_catalog function."""

    def test_creates_catalog_with_name(self) -> None:
        """Test creating root catalog with dataset name."""
        from ftw_dataset_tools.api.stac import _create_root_catalog

        catalog = _create_root_catalog("test_dataset")

        assert catalog.id == "test_dataset"
        assert "test_dataset" in catalog.description

    def test_creates_catalog_with_custom_description(self) -> None:
        """Test creating root catalog with custom description."""
        from ftw_dataset_tools.api.stac import _create_root_catalog

        catalog = _create_root_catalog("test_dataset", "Custom description")

        assert catalog.description == "Custom description"


class TestChipInfo:
    """Tests for ChipInfo dataclass."""

    def test_chip_info_fields(self) -> None:
        """Test ChipInfo has expected fields."""
        from ftw_dataset_tools.api.stac import ChipInfo

        chip = ChipInfo(
            grid_id="grid_001",
            geometry={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
            bbox=(0.0, 0.0, 1.0, 1.0),
        )

        assert chip.grid_id == "grid_001"
        assert chip.geometry["type"] == "Polygon"
        assert chip.bbox == (0.0, 0.0, 1.0, 1.0)
        assert chip.properties == {}  # Default

    def test_chip_info_with_properties(self) -> None:
        """Test ChipInfo with custom properties."""
        from ftw_dataset_tools.api.stac import ChipInfo

        chip = ChipInfo(
            grid_id="grid_001",
            geometry={"type": "Point", "coordinates": [0, 0]},
            bbox=(0.0, 0.0, 0.0, 0.0),
            properties={"coverage": 50.0},
        )

        assert chip.properties["coverage"] == 50.0
