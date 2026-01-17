"""Tests for the STAC API."""

from datetime import UTC, datetime
from pathlib import Path


class TestChipInfoWithYear:
    """Tests for ChipInfo year-based naming."""

    def test_item_id_without_year(self) -> None:
        """Test item_id property without year returns grid_id."""
        from ftw_dataset_tools.api.stac import ChipInfo

        chip_info = ChipInfo(
            grid_id="ftw-34UFF1628",
            geometry={"type": "Polygon", "coordinates": []},
            bbox=(0.0, 0.0, 1.0, 1.0),
        )

        assert chip_info.item_id == "ftw-34UFF1628"
        assert chip_info.dir_name == "ftw-34UFF1628"

    def test_item_id_with_year(self) -> None:
        """Test item_id property with year includes year suffix."""
        from ftw_dataset_tools.api.stac import ChipInfo

        chip_info = ChipInfo(
            grid_id="ftw-34UFF1628",
            geometry={"type": "Polygon", "coordinates": []},
            bbox=(0.0, 0.0, 1.0, 1.0),
            year=2024,
        )

        assert chip_info.item_id == "ftw-34UFF1628_2024"
        assert chip_info.dir_name == "ftw-34UFF1628_2024"

    def test_year_property_stored(self) -> None:
        """Test that year property is stored correctly."""
        from ftw_dataset_tools.api.stac import ChipInfo

        chip_info = ChipInfo(
            grid_id="grid_001",
            geometry={"type": "Polygon", "coordinates": []},
            bbox=(0.0, 0.0, 1.0, 1.0),
            year=2023,
        )

        assert chip_info.year == 2023


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

    def test_asset_href_with_year(self, tmp_path: Path) -> None:
        """Test asset hrefs include year in filenames when year is set."""
        from ftw_dataset_tools.api.stac import ChipInfo, _create_chip_item

        # Create chip directory with year-based mask files
        chip_dir = tmp_path / "chips" / "grid_001_2024"
        chip_dir.mkdir(parents=True)

        # Create dummy mask files with year in filename
        (chip_dir / "grid_001_2024_instance.tif").touch()
        (chip_dir / "grid_001_2024_semantic_2_class.tif").touch()

        chip_info = ChipInfo(
            grid_id="grid_001",
            geometry={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
            bbox=(0.0, 0.0, 1.0, 1.0),
            year=2024,
        )

        temporal_extent = (
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 12, 31, tzinfo=UTC),
        )

        item = _create_chip_item(
            chip_info=chip_info,
            field_dataset="test_dataset",
            chip_dir=chip_dir,
            temporal_extent=temporal_extent,
        )

        assert item is not None
        # Verify item ID includes year
        assert item.id == "grid_001_2024"
        # Verify asset hrefs include year
        assert item.assets["instance_mask"].href == "./grid_001_2024_instance.tif"
        assert item.assets["semantic_2class_mask"].href == "./grid_001_2024_semantic_2_class.tif"
        # Verify FTW extension property
        assert item.properties.get("ftw:calendar_year") == 2024


class TestGenerateStacCatalogSignature:
    """Tests for generate_stac_catalog function signature."""

    def test_accepts_chips_base_dir_parameter(self) -> None:
        """Test that generate_stac_catalog accepts chips_base_dir parameter."""
        import inspect

        from ftw_dataset_tools.api.stac import generate_stac_catalog

        sig = inspect.signature(generate_stac_catalog)
        assert "chips_base_dir" in sig.parameters
