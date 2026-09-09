"""Tests for preserving imagery selections across STAC catalog regeneration.

`generate_stac_catalog` rebuilds every parent chip item from scratch and
overwrites the item JSON, which used to wipe the ftw:planting/ftw:harvest links
that image selection relies on to skip already-selected chips.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import geopandas as gpd
import pystac
import pytest
from shapely.geometry import box

from ftw_dataset_tools.api.imagery.catalog_ops import (
    IMAGERY_PROPERTIES,
    IMAGERY_TEMPORAL_PROPERTIES,
    has_existing_scenes,
    preserve_imagery_selection,
)
from ftw_dataset_tools.api.imagery.selection_workflow import select_imagery_for_catalog
from ftw_dataset_tools.api.imagery.stac_child_items import create_child_items_from_selection
from ftw_dataset_tools.api.stac import generate_stac_catalog

if TYPE_CHECKING:
    from pathlib import Path

    from ftw_dataset_tools.api.imagery.scene_selection import SceneSelectionResult

FIELD_DATASET = "test_dataset"
GRID_ID = "chip_a"
YEAR = 2024
ITEM_ID = f"{GRID_ID}_{YEAR}"
CHIP_BOX = (10.0, 50.0, 10.02, 50.02)


def _write_instance_mask(path: Path) -> None:
    """Write a tiny real instance mask.

    Catalog generation reads every mask asset for its raster metadata, so the
    file has to be an openable COG rather than an empty placeholder.
    """
    import numpy as np
    import rasterio
    from rasterio.transform import from_bounds

    from ftw_dataset_tools.api.raster_stats import compute_band_stats, embed_band_stats

    data = np.array([[0, 1], [1, 0]], dtype="uint32")
    with rasterio.open(
        path,
        "w",
        driver="COG",
        width=data.shape[1],
        height=data.shape[0],
        count=1,
        dtype="uint32",
        crs="EPSG:4326",
        transform=from_bounds(*CHIP_BOX, data.shape[1], data.shape[0]),
        compress="deflate",
    ) as dst:
        dst.write(data, 1)
        embed_band_stats(dst, 1, compute_band_stats(data))


def _make_item(item_id: str, properties: dict | None = None) -> pystac.Item:
    """Create a bare parent chip item."""
    return pystac.Item(
        id=item_id,
        geometry={
            "type": "Polygon",
            "coordinates": [[[10.0, 50.0], [10.02, 50.0], [10.02, 50.02], [10.0, 50.0]]],
        },
        bbox=list(CHIP_BOX),
        datetime=datetime(YEAR, 1, 1, tzinfo=UTC),
        properties=properties or {},
    )


def _make_selected_item(item_path: Path, properties: dict | None = None) -> pystac.Item:
    """Create a parent item that carries a completed imagery selection.

    The child season items the links point at are written too, since
    preservation only trusts a link whose target is still on disk.
    """
    item = _make_item(item_path.stem, properties)
    item.properties["ftw:calendar_year"] = YEAR
    item.properties["ftw:planting_day"] = 150
    item.properties["ftw:harvest_cloud_cover"] = 2.0
    item.properties["start_datetime"] = "2024-06-15T10:00:00+00:00"
    item.properties["end_datetime"] = "2024-09-15T10:00:00+00:00"
    item_path.parent.mkdir(parents=True, exist_ok=True)
    for season in ("planting", "harvest"):
        item.add_link(
            pystac.Link(
                rel=f"ftw:{season}",
                target=f"./{item.id}_{season}_s2.json",
                media_type="application/json",
            )
        )
        (item_path.parent / f"{item.id}_{season}_s2.json").touch()
    item.set_self_href(str(item_path))
    item.save_object(dest_href=str(item_path))
    return item


class TestPreserveImagerySelection:
    """Tests for preserve_imagery_selection."""

    def test_returns_false_when_no_existing_item(self, tmp_path: Path) -> None:
        """A first run has nothing to carry over."""
        item = _make_item(ITEM_ID)

        assert preserve_imagery_selection(item, tmp_path / f"{ITEM_ID}.json") is False
        assert not has_existing_scenes(item)

    def test_returns_false_when_existing_item_has_no_selection(self, tmp_path: Path) -> None:
        """An item without both scene links is not a resumable selection."""
        item_path = tmp_path / f"{ITEM_ID}.json"
        existing = _make_item(ITEM_ID)
        existing.set_self_href(str(item_path))
        existing.save_object(dest_href=str(item_path))

        assert preserve_imagery_selection(_make_item(ITEM_ID), item_path) is False

    def test_returns_false_on_unreadable_item(self, tmp_path: Path) -> None:
        """A corrupt item re-selects instead of failing catalog generation."""
        item_path = tmp_path / f"{ITEM_ID}.json"
        item_path.write_text("{ not json")

        assert preserve_imagery_selection(_make_item(ITEM_ID), item_path) is False

    def test_carries_over_links_and_properties(self, tmp_path: Path) -> None:
        """Scene links, ftw: properties and the scene-derived extent survive."""
        item_path = tmp_path / f"{ITEM_ID}.json"
        _make_selected_item(item_path)

        fresh = _make_item(
            ITEM_ID,
            properties={
                "start_datetime": "2024-01-01T00:00:00+00:00",
                "end_datetime": "2024-12-31T23:59:59+00:00",
            },
        )

        assert preserve_imagery_selection(fresh, item_path) is True
        assert has_existing_scenes(fresh)
        assert fresh.properties["ftw:planting_day"] == 150
        assert fresh.properties["ftw:harvest_cloud_cover"] == 2.0
        # The dataset-wide extent is replaced by the actual scene dates
        assert fresh.properties["start_datetime"] == "2024-06-15T10:00:00+00:00"
        assert fresh.properties["end_datetime"] == "2024-09-15T10:00:00+00:00"

    def test_preserved_links_stay_relative(self, tmp_path: Path) -> None:
        """Carried-over links must not become absolute local paths."""
        item_path = tmp_path / f"{ITEM_ID}.json"
        _make_selected_item(item_path)

        fresh = _make_item(ITEM_ID)
        preserve_imagery_selection(fresh, item_path)

        hrefs = {link.rel: link.href for link in fresh.links if link.rel.startswith("ftw:")}
        assert hrefs["ftw:planting"] == f"./{ITEM_ID}_planting_s2.json"
        assert hrefs["ftw:harvest"] == f"./{ITEM_ID}_harvest_s2.json"

    def test_drops_links_whose_child_item_is_gone(self, tmp_path: Path) -> None:
        """A link to a deleted child item is not carried over.

        Otherwise the chip looks selected forever while the scene it points at
        no longer exists, and selection can never repair it.
        """
        item_path = tmp_path / f"{ITEM_ID}.json"
        _make_selected_item(item_path)
        (tmp_path / f"{ITEM_ID}_planting_s2.json").unlink()

        fresh = _make_item(ITEM_ID)
        preserve_imagery_selection(fresh, item_path)

        rels = {link.rel for link in fresh.links}
        assert "ftw:planting" not in rels
        assert "ftw:harvest" in rels
        # ...so the chip is re-selected rather than skipped
        assert not has_existing_scenes(fresh)

    def test_carries_over_only_assets_still_on_disk(self, tmp_path: Path) -> None:
        """Downloaded imagery is re-advertised, dangling assets are dropped."""
        item_path = tmp_path / f"{ITEM_ID}.json"
        existing = _make_selected_item(item_path)
        for season in ("planting", "harvest"):
            existing.assets[f"{season}_image"] = pystac.Asset(
                href=f"./{ITEM_ID}_{season}_image_s2.tif",
                media_type="image/tiff; application=geotiff",
                roles=["data"],
            )
        existing.save_object(dest_href=str(item_path))
        # Only the planting image was actually downloaded
        (tmp_path / f"{ITEM_ID}_planting_image_s2.tif").touch()

        fresh = _make_item(ITEM_ID)
        assert preserve_imagery_selection(fresh, item_path) is True
        assert "planting_image" in fresh.assets
        assert "harvest_image" not in fresh.assets

    def test_keeps_freshly_generated_mask_assets(self, tmp_path: Path) -> None:
        """Preserving a selection must not clobber regenerated mask assets."""
        item_path = tmp_path / f"{ITEM_ID}.json"
        _make_selected_item(item_path)

        fresh = _make_item(ITEM_ID)
        fresh.add_asset(
            "instance_mask",
            pystac.Asset(href=f"./{ITEM_ID}_instance.tif", roles=["labels"]),
        )

        preserve_imagery_selection(fresh, item_path)

        assert "instance_mask" in fresh.assets


def _build_inputs(tmp_path: Path, chips_base_dir: Path | None = None) -> dict[str, Path]:
    """Minimal fields/chips/boundaries inputs plus one chip dir with a mask.

    ``chips_base_dir`` defaults to the flat ``{dataset}-chips`` layout; pass a
    nested one to exercise the MGRS-square layout, where the chip directory is
    not where ``catalog.save()`` writes the item.
    """
    output_dir = tmp_path / "dataset"
    if chips_base_dir is None:
        chips_base_dir = output_dir / f"{FIELD_DATASET}-chips"
    chip_dir = chips_base_dir / ITEM_ID
    chip_dir.mkdir(parents=True)
    _write_instance_mask(chip_dir / f"{ITEM_ID}_instance.tif")

    fields_file = tmp_path / "fields.parquet"
    gpd.GeoDataFrame(
        {"id": [1]}, geometry=[box(10.001, 50.001, 10.01, 50.01)], crs="EPSG:4326"
    ).to_parquet(fields_file)

    chips_file = tmp_path / "chips.parquet"
    gpd.GeoDataFrame({"id": [GRID_ID]}, geometry=[box(*CHIP_BOX)], crs="EPSG:4326").to_parquet(
        chips_file
    )

    boundary_lines_file = tmp_path / "boundaries.parquet"
    gpd.GeoDataFrame(
        {"id": [1]}, geometry=[box(10.001, 50.001, 10.01, 50.01).boundary], crs="EPSG:4326"
    ).to_parquet(boundary_lines_file)

    return {
        "output_dir": output_dir,
        "chips_base_dir": chips_base_dir,
        "chip_dir": chip_dir,
        "fields_file": fields_file,
        "chips_file": chips_file,
        "boundary_lines_file": boundary_lines_file,
    }


@pytest.fixture
def catalog_inputs(tmp_path: Path) -> dict[str, Path]:
    """Catalog generation inputs in the default flat chip layout."""
    return _build_inputs(tmp_path)


def _generate(inputs: dict[str, Path]) -> None:
    """Run catalog generation against the shared test inputs."""
    generate_stac_catalog(
        output_dir=inputs["output_dir"],
        field_dataset=FIELD_DATASET,
        fields_file=inputs["fields_file"],
        chips_file=inputs["chips_file"],
        boundary_lines_file=inputs["boundary_lines_file"],
        chips_base_dir=inputs["chips_base_dir"],
        year=YEAR,
    )


class TestCatalogRegenerationResume:
    """End-to-end: regenerate a catalog over an existing selection."""

    def test_regeneration_preserves_selection(
        self,
        catalog_inputs: dict[str, Path],
        mock_selection_result: SceneSelectionResult,
    ) -> None:
        """A second create-dataset run keeps the selection the first one made."""
        _generate(catalog_inputs)

        item_path = catalog_inputs["chip_dir"] / f"{ITEM_ID}.json"
        parent = pystac.Item.from_file(str(item_path))
        create_child_items_from_selection(
            chip_dir=catalog_inputs["chip_dir"],
            parent_item=parent,
            result=mock_selection_result,
            year=YEAR,
            cloud_cover_chip=2.0,
            buffer_days=14,
        )
        assert has_existing_scenes(pystac.Item.from_file(str(item_path)))

        _generate(catalog_inputs)

        regenerated = pystac.Item.from_file(str(item_path))
        assert has_existing_scenes(regenerated)
        assert regenerated.properties["ftw:planting_day"] == 150
        assert regenerated.properties["ftw:stac_host"] == "earthsearch"
        # Mask assets are still rebuilt from disk
        assert "instance_mask" in regenerated.assets
        # Child items were never regenerated
        assert (catalog_inputs["chip_dir"] / f"{ITEM_ID}_planting_s2.json").exists()

    def test_selection_skips_after_regeneration(
        self,
        catalog_inputs: dict[str, Path],
        mock_selection_result: SceneSelectionResult,
    ) -> None:
        """Selection after a rebuild resumes instead of re-selecting every chip.

        Regression test for the bug where catalog regeneration cleared the scene
        links moments before selection read them, so every chip re-selected and
        --force-image-selection had no observable effect. No network access is
        needed: a resumed chip is skipped before any STAC query is made.
        """
        _generate(catalog_inputs)

        item_path = catalog_inputs["chip_dir"] / f"{ITEM_ID}.json"
        create_child_items_from_selection(
            chip_dir=catalog_inputs["chip_dir"],
            parent_item=pystac.Item.from_file(str(item_path)),
            result=mock_selection_result,
            year=YEAR,
            cloud_cover_chip=2.0,
            buffer_days=14,
        )

        _generate(catalog_inputs)

        result = select_imagery_for_catalog(
            catalog_dir=catalog_inputs["chips_base_dir"],
            year=YEAR,
        )

        assert result.successful == 0
        assert result.skipped == 1
        assert "Already has imagery selections" in result.skipped_details[0]["reason"]

    def test_first_run_has_no_selection(self, catalog_inputs: dict[str, Path]) -> None:
        """Generating a catalog from scratch leaves items without scene links."""
        _generate(catalog_inputs)

        item = pystac.Item.from_file(str(catalog_inputs["chip_dir"] / f"{ITEM_ID}.json"))

        assert not has_existing_scenes(item)

    def test_saved_links_stay_relative(
        self,
        catalog_inputs: dict[str, Path],
        mock_selection_result: SceneSelectionResult,
    ) -> None:
        """The regenerated item JSON on disk keeps relative child hrefs.

        Read straight from the file rather than the in-memory item: an href
        rewrite by ``catalog.save`` is exactly what would break a copied or
        relocated catalog.
        """
        _generate(catalog_inputs)

        item_path = catalog_inputs["chip_dir"] / f"{ITEM_ID}.json"
        create_child_items_from_selection(
            chip_dir=catalog_inputs["chip_dir"],
            parent_item=pystac.Item.from_file(str(item_path)),
            result=mock_selection_result,
            year=YEAR,
            cloud_cover_chip=2.0,
            buffer_days=14,
        )

        _generate(catalog_inputs)

        saved = json.loads(item_path.read_text())
        hrefs = {
            link["rel"]: link["href"] for link in saved["links"] if link["rel"].startswith("ftw:")
        }
        assert hrefs["ftw:planting"] == f"./{ITEM_ID}_planting_s2.json"
        assert hrefs["ftw:harvest"] == f"./{ITEM_ID}_harvest_s2.json"

    def test_regeneration_drops_links_to_deleted_children(
        self,
        catalog_inputs: dict[str, Path],
        mock_selection_result: SceneSelectionResult,
    ) -> None:
        """A chip whose child items were deleted re-selects instead of resuming."""
        _generate(catalog_inputs)

        item_path = catalog_inputs["chip_dir"] / f"{ITEM_ID}.json"
        create_child_items_from_selection(
            chip_dir=catalog_inputs["chip_dir"],
            parent_item=pystac.Item.from_file(str(item_path)),
            result=mock_selection_result,
            year=YEAR,
            cloud_cover_chip=2.0,
            buffer_days=14,
        )
        # An interrupted run, a manual cleanup or a partially copied catalog
        (catalog_inputs["chip_dir"] / f"{ITEM_ID}_planting_s2.json").unlink()

        _generate(catalog_inputs)

        regenerated = pystac.Item.from_file(str(item_path))
        rels = {link.rel for link in regenerated.links}
        assert "ftw:planting" not in rels
        assert not has_existing_scenes(regenerated)

    def test_preserves_selection_under_nested_chip_layout(self, tmp_path: Path) -> None:
        """Resuming works when chip dirs are nested, not at {dataset}-chips/{id}.

        Under the MGRS-square layout the previous run's item lives in the chip
        directory the pipeline was handed, which is not where the flat layout
        would look for it.
        """
        inputs = _build_inputs(
            tmp_path,
            chips_base_dir=tmp_path / "dataset" / f"{FIELD_DATASET}-chips" / "32UPU",
        )
        _generate(inputs)

        # Record a selection where a real select-images run would write it
        _make_selected_item(inputs["chip_dir"] / f"{ITEM_ID}.json")

        _generate(inputs)

        saved_path = inputs["output_dir"] / f"{FIELD_DATASET}-chips" / ITEM_ID / f"{ITEM_ID}.json"
        regenerated = pystac.Item.from_file(str(saved_path))
        assert has_existing_scenes(regenerated)
        assert regenerated.properties["ftw:planting_day"] == 150


class TestImageryPropertyAllowList:
    """Guard the hand-maintained allow-lists against drift."""

    def test_allow_list_covers_every_written_property(
        self,
        tmp_path: Path,
        mock_selection_result: SceneSelectionResult,
    ) -> None:
        """Every ftw: property a selection writes must survive regeneration.

        A key missing from IMAGERY_PROPERTIES is silently dropped on a re-run,
        so assert the list still covers what selection actually writes.
        """
        chip_dir = tmp_path / ITEM_ID
        chip_dir.mkdir()
        parent = _make_item(ITEM_ID)

        create_child_items_from_selection(
            chip_dir=chip_dir,
            parent_item=parent,
            result=mock_selection_result,
            year=YEAR,
            cloud_cover_chip=2.0,
            buffer_days=14,
        )

        written = {key for key in parent.properties if key.startswith("ftw:")}
        missing = written - set(IMAGERY_PROPERTIES)
        assert not missing, f"IMAGERY_PROPERTIES is missing {sorted(missing)}"
        assert set(IMAGERY_TEMPORAL_PROPERTIES) <= set(parent.properties)
