"""Tests for the STAC API."""

from datetime import UTC, datetime
from pathlib import Path


def _write_mask(path: Path, values: list[list[int]], dtype: str = "uint8") -> None:
    import numpy as np
    import rasterio
    from rasterio.transform import from_bounds

    from ftw_dataset_tools.api.raster_stats import compute_band_stats, embed_band_stats

    data = np.array(values, dtype=dtype)
    with rasterio.open(
        path,
        "w",
        driver="COG",
        width=data.shape[1],
        height=data.shape[0],
        count=1,
        dtype=dtype,
        crs="EPSG:4326",
        transform=from_bounds(0, 0, 1, 1, data.shape[1], data.shape[0]),
        compress="deflate",
    ) as dst:
        dst.write(data, 1)
        embed_band_stats(dst, 1, compute_band_stats(data))


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
        _write_mask(chip_dir / "grid_001_instance.tif", [[0, 1], [1, 0]], dtype="uint32")
        _write_mask(chip_dir / "grid_001_semantic_2_class.tif", [[0, 1], [1, 0]])
        _write_mask(chip_dir / "grid_001_semantic_3_class.tif", [[0, 1], [2, 0]])

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

    def test_decode_masks_registered_as_assets(self, tmp_path: Path) -> None:
        """DECODE layers become STAC assets with their own titles when present."""
        from ftw_dataset_tools.api.stac import ChipInfo, _create_chip_item

        chip_dir = tmp_path / "chips" / "grid_001"
        chip_dir.mkdir(parents=True)
        _write_mask(chip_dir / "grid_001_semantic_2_class.tif", [[0, 1], [1, 0]])
        _write_mask(chip_dir / "grid_001_decode_boundary.tif", [[0, 1], [1, 0]])
        _write_mask(
            chip_dir / "grid_001_decode_distance.tif",
            [[0.0, 0.5], [1.0, 0.0]],
            dtype="float32",
        )

        item = _create_chip_item(
            chip_info=ChipInfo(
                grid_id="grid_001",
                geometry={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
                bbox=(0.0, 0.0, 1.0, 1.0),
            ),
            field_dataset="test_dataset",
            chip_dir=chip_dir,
            temporal_extent=(
                datetime(2023, 1, 1, tzinfo=UTC),
                datetime(2023, 12, 31, tzinfo=UTC),
            ),
        )

        assert item is not None
        assert item.assets["decode_boundary_mask"].href == "./grid_001_decode_boundary.tif"
        assert item.assets["decode_distance_mask"].href == "./grid_001_decode_distance.tif"
        # Titles are specific, not the generic "<name> mask" fallback.
        assert item.assets["decode_boundary_mask"].title == "DECODE field boundary mask"
        assert item.assets["decode_distance_mask"].title == (
            "DECODE normalized distance-to-boundary map"
        )
        # Mask types that were not written must not appear.
        assert "instance_mask" not in item.assets

    def test_decode_masks_absent_when_not_generated(self, tmp_path: Path) -> None:
        """A dataset built without the DECODE layers gets no DECODE assets."""
        from ftw_dataset_tools.api.stac import ChipInfo, _create_chip_item

        chip_dir = tmp_path / "chips" / "grid_001"
        chip_dir.mkdir(parents=True)
        _write_mask(chip_dir / "grid_001_semantic_2_class.tif", [[0, 1], [1, 0]])

        item = _create_chip_item(
            chip_info=ChipInfo(
                grid_id="grid_001",
                geometry={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
                bbox=(0.0, 0.0, 1.0, 1.0),
            ),
            field_dataset="test_dataset",
            chip_dir=chip_dir,
            temporal_extent=(
                datetime(2023, 1, 1, tzinfo=UTC),
                datetime(2023, 12, 31, tzinfo=UTC),
            ),
        )

        assert item is not None
        assert "decode_boundary_mask" not in item.assets
        assert "decode_distance_mask" not in item.assets

    def test_asset_href_legacy_mask_dirs(self, tmp_path: Path) -> None:
        """Test asset hrefs use legacy paths when mask_dirs is provided."""
        from ftw_dataset_tools.api.stac import ChipInfo, _create_chip_item

        # Create mask directories (legacy structure)
        instance_dir = tmp_path / "label_masks" / "instance"
        semantic_2class_dir = tmp_path / "label_masks" / "semantic_2class"
        instance_dir.mkdir(parents=True)
        semantic_2class_dir.mkdir(parents=True)

        # Create dummy mask files with legacy naming convention (with dataset prefix)
        _write_mask(
            instance_dir / "test_dataset_grid_001_instance.tif",
            [[0, 1], [1, 0]],
            dtype="uint32",
        )
        _write_mask(
            semantic_2class_dir / "test_dataset_grid_001_semantic_2_class.tif",
            [[0, 1], [1, 0]],
        )

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
        _write_mask(chip_dir / "grid_001_2024_instance.tif", [[0, 1], [1, 0]], dtype="uint32")
        _write_mask(chip_dir / "grid_001_2024_semantic_2_class.tif", [[0, 1], [1, 0]])

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


class TestChipItemAssetMetadata:
    def _chip_item(self, tmp_path: Path, checksums: bool = False):
        from ftw_dataset_tools.api.stac import ChipInfo, _create_chip_item

        chip_dir = tmp_path / "ftw-1"
        chip_dir.mkdir()
        _write_mask(chip_dir / "ftw-1_semantic_3_class.tif", [[0, 1], [2, 0]])
        _write_mask(chip_dir / "ftw-1_instance.tif", [[0, 4], [4, 0]], dtype="uint32")
        _write_mask(chip_dir / "ftw-1_decode_boundary.tif", [[0, 1], [1, 0]])
        _write_mask(
            chip_dir / "ftw-1_decode_distance.tif",
            [[0.0, 0.5], [1.0, 0.0]],
            dtype="float32",
        )

        chip_info = ChipInfo(
            grid_id="ftw-1",
            geometry={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
            bbox=(0.0, 0.0, 1.0, 1.0),
        )
        return _create_chip_item(
            chip_info=chip_info,
            field_dataset="ds",
            temporal_extent=(datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 12, 31, tzinfo=UTC)),
            chip_dir=chip_dir,
            checksums=checksums,
        )

    def test_mask_assets_have_size_type_roles_and_bands(self, tmp_path: Path) -> None:
        item = self._chip_item(tmp_path)
        assert item is not None

        semantic = item.assets["semantic_3class_mask"]
        assert semantic.media_type == "image/tiff; application=geotiff; profile=cloud-optimized"
        assert semantic.roles == ["labels"]
        assert semantic.extra_fields["file:size"] > 0
        assert "file:checksum" not in semantic.extra_fields
        band = semantic.extra_fields["raster:bands"][0]
        assert band["data_type"] == "uint8"
        assert band["statistics"]["maximum"] == 2
        assert [c["value"] for c in band["classification:classes"]] == [0, 1, 2]

        instance = item.assets["instance_mask"]
        iband = instance.extra_fields["raster:bands"][0]
        assert iband["data_type"] == "uint32"
        assert "classification:classes" not in iband

    def test_checksums_when_enabled(self, tmp_path: Path) -> None:
        item = self._chip_item(tmp_path, checksums=True)
        assert item is not None

        checksum = item.assets["semantic_3class_mask"].extra_fields["file:checksum"]
        assert checksum.startswith("1220")
        assert len(checksum) == 4 + 64

    def test_extensions_registered_on_item(self, tmp_path: Path) -> None:
        item = self._chip_item(tmp_path)
        assert item is not None

        assert "https://stac-extensions.github.io/file/v2.1.0/schema.json" in item.stac_extensions
        assert "https://stac-extensions.github.io/raster/v1.1.0/schema.json" in item.stac_extensions
        assert (
            "https://stac-extensions.github.io/classification/v2.0.0/schema.json"
            in item.stac_extensions
        )

    def test_decode_assets_classified_or_described(self, tmp_path: Path) -> None:
        item = self._chip_item(tmp_path)
        assert item is not None

        boundary = item.assets["decode_boundary_mask"].extra_fields["raster:bands"][0]
        assert [c["name"] for c in boundary["classification:classes"]] == ["background", "boundary"]

        distance = item.assets["decode_distance_mask"].extra_fields["raster:bands"][0]
        assert distance["data_type"] == "float32"
        assert "classification:classes" not in distance
        assert "decode_distance_max_px" in distance["description"]


class TestCollectionAssetMetadata:
    def _build_catalog(
        self,
        tmp_path: Path,
        checksums: bool = False,
        config=None,
        provenance=None,
        on_progress=None,
    ):
        import geopandas as gpd
        from shapely.geometry import box

        from ftw_dataset_tools.api.stac import generate_stac_catalog

        fields = gpd.GeoDataFrame(
            {"id": [1]},
            geometry=[box(0, 0, 1, 1)],
            crs="EPSG:4326",
        )
        fields_path = tmp_path / "ds_fields.parquet"
        fields.to_parquet(fields_path)
        lines_path = tmp_path / "ds_boundary_lines.parquet"
        fields.to_parquet(lines_path)
        chips = gpd.GeoDataFrame(
            {"id": ["ftw-1"], "field_coverage_pct": [50.0]},
            geometry=[box(0, 0, 1, 1)],
            crs="EPSG:4326",
        )
        chips_path = tmp_path / "ds_chips.parquet"
        chips.to_parquet(chips_path)

        chips_base = tmp_path / "ds-chips"
        chip_dir = chips_base / "ftw-1_2024"
        chip_dir.mkdir(parents=True)
        _write_mask(chip_dir / "ftw-1_2024_semantic_2_class.tif", [[0, 1], [1, 0]])

        return generate_stac_catalog(
            output_dir=tmp_path,
            field_dataset="ds",
            fields_file=fields_path,
            chips_file=chips_path,
            boundary_lines_file=lines_path,
            chips_base_dir=chips_base,
            year=2024,
            checksums=checksums,
            config=config,
            provenance=provenance,
            on_progress=on_progress,
        )

    def test_parquet_and_items_assets(self, tmp_path: Path) -> None:
        result = self._build_catalog(tmp_path)
        fields_path = tmp_path / "ds_fields.parquet"
        chips_path = tmp_path / "ds_chips.parquet"

        import pystac

        source = pystac.Collection.from_file(str(result.source_collection_path))
        assert source.assets["fields"].extra_fields["file:size"] == fields_path.stat().st_size
        assert source.assets["fields"].media_type == "application/vnd.apache.parquet"

        chips_coll = pystac.Collection.from_file(str(result.chips_collection_path))
        items_asset = chips_coll.assets["items"]
        assert items_asset.media_type == "application/vnd.apache.parquet"
        assert items_asset.roles == ["collection-mirror"]
        assert items_asset.extra_fields["file:size"] == result.items_parquet_path.stat().st_size
        assert chips_coll.assets["chips"].extra_fields["file:size"] == chips_path.stat().st_size

    def test_collections_have_no_self_link(self, tmp_path: Path) -> None:
        import json

        result = self._build_catalog(tmp_path)
        for path in (result.chips_collection_path, result.source_collection_path):
            rels = [link["rel"] for link in json.loads(path.read_text())["links"]]
            assert "self" not in rels

    def test_checksums_on_collection_and_items_assets(self, tmp_path: Path) -> None:
        import pystac

        result = self._build_catalog(tmp_path, checksums=True)

        source = pystac.Collection.from_file(str(result.source_collection_path))
        assert source.assets["fields"].extra_fields["file:checksum"].startswith("1220")
        assert source.assets["boundary_lines"].extra_fields["file:checksum"].startswith("1220")

        chips_coll = pystac.Collection.from_file(str(result.chips_collection_path))
        assert chips_coll.assets["chips"].extra_fields["file:checksum"].startswith("1220")
        assert chips_coll.assets["items"].extra_fields["file:checksum"].startswith("1220")


class TestCollectionMetadata:
    def _config(self, **metadata):
        from ftw_dataset_tools.api.config import DatasetConfig

        data = {
            "fields_file": "unused.parquet",
            "stages": {"splits": {"split_type": "block3x3", "random_seed": 7}},
        }
        if metadata:
            data["metadata"] = metadata
        return DatasetConfig.from_dict(data)

    def test_metadata_lands_on_collections(self, tmp_path: Path) -> None:
        import json

        config = self._config(
            title="Austria",
            description="Chips for Austria",
            license="CC-BY-4.0",
            version="2.0.0-alpha.1",
            keywords=["austria"],
            providers=[
                {
                    "name": "Agrarmarkt Austria",
                    "roles": ["producer", "licensor"],
                    "url": "https://x",
                }
            ],
        )
        result = TestCollectionAssetMetadata()._build_catalog(tmp_path, config=config)

        chips = json.loads(result.chips_collection_path.read_text())
        assert chips["title"] == "Austria"
        assert chips["description"] == "Chips for Austria"
        assert chips["license"] == "CC-BY-4.0"
        assert chips["keywords"] == ["austria"]
        assert chips["version"] == "2.0.0-alpha.1"
        assert (
            "https://stac-extensions.github.io/version/v1.2.0/schema.json"
            in chips["stac_extensions"]
        )
        assert [p["name"] for p in chips["providers"]] == ["Agrarmarkt Austria"]
        assert chips["providers"][0]["roles"] == ["producer", "licensor"]
        assert all(p["roles"] != ["host"] for p in chips["providers"])
        assert chips["updated"].endswith("Z")

        source = json.loads(result.source_collection_path.read_text())
        assert source["license"] == "CC-BY-4.0"
        assert source["title"] == "Austria source fields"
        assert source["version"] == "2.0.0-alpha.1"

    def test_license_link_when_other(self, tmp_path: Path) -> None:
        import json

        config = self._config(license="other", license_url="https://rkg.gov.si/vstop/")
        result = TestCollectionAssetMetadata()._build_catalog(tmp_path, config=config)

        chips = json.loads(result.chips_collection_path.read_text())
        assert chips["license"] == "other"
        links = [link for link in chips["links"] if link["rel"] == "license"]
        assert links and links[0]["href"] == "https://rkg.gov.si/vstop/"

    def test_ftw_properties_and_provenance_on_chips_collection(self, tmp_path: Path) -> None:
        import json

        config = self._config(license="CC0-1.0")
        provenance = config.provenance_dict()
        result = TestCollectionAssetMetadata()._build_catalog(
            tmp_path, config=config, provenance=provenance
        )

        chips = json.loads(result.chips_collection_path.read_text())
        assert chips["ftw:split_type"] == "block3x3"
        assert chips["ftw:split_seed"] == 7
        assert chips["ftw:split_percents"] == [80, 10, 10]
        assert chips["ftw:mask_types"] == ["instance", "semantic_2_class", "semantic_3_class"]
        assert chips["ftw:mask_resolution_m"] == 10.0
        assert "ftw:cloud_cover_chip_threshold" in chips  # select_images enabled by default
        assert chips["ftw:config"]["config"]["metadata"]["license"] == "CC0-1.0"
        assert chips["updated"] == provenance["generated_at"].replace("+00:00", "Z")

        root = json.loads(result.catalog_path.read_text())
        assert root["ftw:config"]["config"]["metadata"]["license"] == "CC0-1.0"

    def test_table_columns_on_parquet_assets(self, tmp_path: Path) -> None:
        import json

        result = TestCollectionAssetMetadata()._build_catalog(tmp_path)

        chips = json.loads(result.chips_collection_path.read_text())
        chip_cols = {c["name"]: c["type"] for c in chips["assets"]["chips"]["table:columns"]}
        assert chip_cols["id"] == "varchar"
        assert chip_cols["field_coverage_pct"] == "double"
        assert chip_cols["geometry"] == "geometry"
        assert chips["assets"]["chips"]["table:row_count"] == 1

        source = json.loads(result.source_collection_path.read_text())
        assert source["assets"]["fields"]["table:row_count"] == 1
        assert (
            "https://stac-extensions.github.io/table/v1.2.0/schema.json"
            in source["stac_extensions"]
        )

    def test_warns_without_license(self, tmp_path: Path) -> None:
        messages: list[str] = []
        TestCollectionAssetMetadata()._build_catalog(tmp_path, on_progress=messages.append)

        assert any("not Portolan-publishable" in m for m in messages)
