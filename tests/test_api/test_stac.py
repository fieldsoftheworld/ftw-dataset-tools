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
            chip_dir=chip_dir,
            temporal_extent=(
                datetime(2023, 1, 1, tzinfo=UTC),
                datetime(2023, 12, 31, tzinfo=UTC),
            ),
        )

        assert item is not None
        assert "decode_boundary_mask" not in item.assets
        assert "decode_distance_mask" not in item.assets

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
        filtered: bool = False,
        with_masks: bool = True,
        grid_id: str = "ftw-33UXP0410",
    ):
        import geopandas as gpd
        from shapely.geometry import box

        from ftw_dataset_tools.api.masks import get_mgrs_square
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
            {"id": [grid_id], "field_coverage_pct": [50.0]},
            geometry=[box(0, 0, 1, 1)],
            crs="EPSG:4326",
        )
        chips_path = tmp_path / "ds_chips.parquet"
        chips.to_parquet(chips_path)

        filtered_fields_path = None
        if filtered:
            filtered_fields_path = tmp_path / "ds_fields_filtered.parquet"
            fields.to_parquet(filtered_fields_path)

        chips_base = tmp_path / "chips"
        square = get_mgrs_square(grid_id)
        chip_dir = chips_base / square / f"{grid_id}_2024"
        chip_dir.mkdir(parents=True)
        if with_masks:
            _write_mask(chip_dir / f"{grid_id}_2024_semantic_2_class.tif", [[0, 1], [1, 0]])

        return generate_stac_catalog(
            output_dir=tmp_path,
            field_dataset="ds",
            fields_file=fields_path,
            chips_file=chips_path,
            boundary_lines_file=lines_path,
            chips_base_dir=chips_base,
            filtered_fields_file=filtered_fields_path,
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

        coll = pystac.Collection.from_file(str(result.collection_path))
        assert coll.assets["fields"].extra_fields["file:size"] == fields_path.stat().st_size
        assert coll.assets["fields"].media_type == "application/vnd.apache.parquet"

        items_asset = coll.assets["items"]
        assert items_asset.media_type == "application/vnd.apache.parquet"
        assert items_asset.roles == ["collection-mirror"]
        assert items_asset.extra_fields["file:size"] == result.items_parquet_path.stat().st_size
        assert coll.assets["chips"].extra_fields["file:size"] == chips_path.stat().st_size

    def test_collections_have_no_self_link(self, tmp_path: Path) -> None:
        import json

        result = self._build_catalog(tmp_path)
        rels = [link["rel"] for link in json.loads(result.collection_path.read_text())["links"]]
        assert "self" not in rels

    def test_checksums_on_collection_and_items_assets(self, tmp_path: Path) -> None:
        import pystac

        result = self._build_catalog(tmp_path, checksums=True)

        coll = pystac.Collection.from_file(str(result.collection_path))
        assert coll.assets["fields"].extra_fields["file:checksum"].startswith("1220")
        assert coll.assets["boundary_lines"].extra_fields["file:checksum"].startswith("1220")
        assert coll.assets["chips"].extra_fields["file:checksum"].startswith("1220")
        assert coll.assets["items"].extra_fields["file:checksum"].startswith("1220")


class TestSingleCollectionLayout:
    def test_tree_and_links(self, tmp_path: Path) -> None:
        import json

        result = TestCollectionAssetMetadata()._build_catalog(tmp_path)

        assert result.collection_path == tmp_path / "collection.json"
        assert not (tmp_path / "catalog.json").exists()
        assert not any(p.name.endswith("-source") for p in tmp_path.iterdir())
        assert result.subcatalog_paths == {"33UXP": tmp_path / "chips" / "33UXP" / "catalog.json"}

        coll = json.loads(result.collection_path.read_text())
        assert coll["id"] == "ds"
        children = [link["href"] for link in coll["links"] if link["rel"] == "child"]
        assert children == ["./chips/33UXP/catalog.json"]
        assert not [link for link in coll["links"] if link["rel"] == "item"]
        assert set(coll["assets"]) >= {"fields", "boundary_lines", "chips", "items"}
        assert coll["assets"]["fields"]["href"] == "./ds_fields.parquet"
        assert coll["assets"]["items"]["href"] == "./items.parquet"
        assert "ftw:config" not in coll

        sub = json.loads(result.subcatalog_paths["33UXP"].read_text())
        items = [link["href"] for link in sub["links"] if link["rel"] == "item"]
        assert items == ["./ftw-33UXP0410_2024/ftw-33UXP0410_2024.json"]
        parent_links = [link["href"] for link in sub["links"] if link["rel"] == "parent"]
        assert parent_links == ["../../collection.json"]
        item_path = tmp_path / "chips" / "33UXP" / "ftw-33UXP0410_2024" / "ftw-33UXP0410_2024.json"
        assert item_path.exists()
        item = json.loads(item_path.read_text())
        assert (
            item["assets"]["semantic_2class_mask"]["href"]
            == "./ftw-33UXP0410_2024_semantic_2_class.tif"
        )
        assert item["collection"] == "ds"
        collection_links = [link["href"] for link in item["links"] if link["rel"] == "collection"]
        assert collection_links == ["../../../collection.json"]

        import duckdb

        con = duckdb.connect()
        con.install_extension("spatial")
        con.load_extension("spatial")
        distinct_collections = con.execute(
            f"SELECT DISTINCT collection FROM read_parquet('{result.items_parquet_path}')"
        ).fetchall()
        assert distinct_collections == [("ds",)]

    def test_filtered_fields_asset(self, tmp_path: Path) -> None:
        import json

        result = TestCollectionAssetMetadata()._build_catalog(tmp_path, filtered=True)

        coll = json.loads(result.collection_path.read_text())
        assert coll["assets"]["fields_filtered"]["href"] == "./ds_fields_filtered.parquet"
        assert coll["assets"]["fields_filtered"]["roles"] == ["data"]

    def test_no_items_means_no_mirror(self, tmp_path: Path) -> None:
        import json

        result = TestCollectionAssetMetadata()._build_catalog(tmp_path, with_masks=False)

        assert result.items_parquet_path is None
        assert result.subcatalog_paths == {}
        coll = json.loads(result.collection_path.read_text())
        assert "items" not in coll["assets"]

    def test_item_assets_on_collection(self, tmp_path: Path) -> None:
        import json

        result = TestCollectionAssetMetadata()._build_catalog(tmp_path)

        coll = json.loads(result.collection_path.read_text())
        ia = coll["item_assets"]
        assert set(ia) >= {
            "instance_mask",
            "semantic_2class_mask",
            "semantic_3class_mask",
            "planting_image",
            "thumbnail",
        }
        assert ia["semantic_3class_mask"]["roles"] == ["labels"]
        assert ia["semantic_3class_mask"]["type"].startswith("image/tiff")
        assert "item-assets" not in " ".join(coll.get("stac_extensions", []))

    def test_custom_grid_ids_go_under_other(self, tmp_path: Path) -> None:
        result = TestCollectionAssetMetadata()._build_catalog(tmp_path, grid_id="grid_001")

        assert list(result.subcatalog_paths) == ["other"]
        assert (tmp_path / "chips" / "other" / "grid_001_2024" / "grid_001_2024.json").exists()


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

        coll = json.loads(result.collection_path.read_text())
        assert coll["title"] == "Austria"
        assert coll["description"] == "Chips for Austria"
        assert coll["license"] == "CC-BY-4.0"
        assert coll["keywords"] == ["austria"]
        assert coll["version"] == "2.0.0-alpha.1"
        assert (
            "https://stac-extensions.github.io/version/v1.2.0/schema.json"
            in coll["stac_extensions"]
        )
        assert [p["name"] for p in coll["providers"]] == ["Agrarmarkt Austria"]
        assert coll["providers"][0]["roles"] == ["producer", "licensor"]
        assert all(p["roles"] != ["host"] for p in coll["providers"])
        assert coll["updated"].endswith("Z")

    def test_license_link_when_other(self, tmp_path: Path) -> None:
        import json

        config = self._config(license="other", license_url="https://rkg.gov.si/vstop/")
        result = TestCollectionAssetMetadata()._build_catalog(tmp_path, config=config)

        coll = json.loads(result.collection_path.read_text())
        assert coll["license"] == "other"
        links = [link for link in coll["links"] if link["rel"] == "license"]
        assert links and links[0]["href"] == "https://rkg.gov.si/vstop/"

    def test_ftw_properties_and_provenance_on_collection(self, tmp_path: Path) -> None:
        import json

        config = self._config(license="CC0-1.0")
        provenance = config.provenance_dict()
        result = TestCollectionAssetMetadata()._build_catalog(
            tmp_path, config=config, provenance=provenance
        )

        coll = json.loads(result.collection_path.read_text())
        assert coll["ftw:split_type"] == "block3x3"
        assert coll["ftw:split_seed"] == 7
        assert coll["ftw:split_percents"] == [80, 10, 10]
        assert coll["ftw:mask_types"] == ["instance", "semantic_2_class", "semantic_3_class"]
        assert coll["ftw:mask_resolution_m"] == 10.0
        assert "ftw:cloud_cover_chip_threshold" in coll  # select_images enabled by default
        assert coll["ftw:config"]["config"]["metadata"]["license"] == "CC0-1.0"
        assert coll["updated"] == provenance["generated_at"].replace("+00:00", "Z")

    def test_table_columns_on_parquet_assets(self, tmp_path: Path) -> None:
        import json

        result = TestCollectionAssetMetadata()._build_catalog(tmp_path)

        coll = json.loads(result.collection_path.read_text())
        chip_cols = {c["name"]: c["type"] for c in coll["assets"]["chips"]["table:columns"]}
        assert chip_cols["id"] == "varchar"
        assert chip_cols["field_coverage_pct"] == "double"
        assert chip_cols["geometry"] == "geometry"
        assert coll["assets"]["chips"]["table:row_count"] == 1

        assert coll["assets"]["fields"]["table:row_count"] == 1
        assert (
            "https://stac-extensions.github.io/table/v1.2.0/schema.json" in coll["stac_extensions"]
        )

        assert coll["assets"]["items"]["table:row_count"] == 1
        items_cols = {c["name"] for c in coll["assets"]["items"]["table:columns"]}
        assert "id" in items_cols

    def test_no_split_type_omitted_from_ftw_properties(self, tmp_path: Path) -> None:
        import json

        from ftw_dataset_tools.api.config import DatasetConfig

        config = DatasetConfig.from_dict({"fields_file": "unused.parquet"})
        result = TestCollectionAssetMetadata()._build_catalog(tmp_path, config=config)

        coll = json.loads(result.collection_path.read_text())
        assert "ftw:split_type" not in coll

    def test_warns_without_license(self, tmp_path: Path) -> None:
        messages: list[str] = []
        TestCollectionAssetMetadata()._build_catalog(tmp_path, on_progress=messages.append)

        assert any("not Portolan-publishable" in m for m in messages)
