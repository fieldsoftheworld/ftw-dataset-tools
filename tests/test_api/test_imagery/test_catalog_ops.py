"""Tests for catalog-tree walking helpers in catalog_ops."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


class TestIterChipDirs:
    def test_yields_nested_item_dirs_sorted(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.imagery.catalog_ops import iter_chip_dirs

        for square, chip in (("33UXQ", "b"), ("33UXP", "a"), ("33UXP", "c")):
            (tmp_path / "chips" / square / chip).mkdir(parents=True)
        (tmp_path / "chips" / "33UXP" / ".hidden").mkdir()
        (tmp_path / "chips" / "33UXP" / "catalog.json").write_text("{}")

        dirs = iter_chip_dirs(tmp_path)

        assert [d.relative_to(tmp_path).as_posix() for d in dirs] == [
            "chips/33UXP/a",
            "chips/33UXP/c",
            "chips/33UXQ/b",
        ]

    def test_missing_chips_dir_is_empty(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.imagery.catalog_ops import iter_chip_dirs

        assert iter_chip_dirs(tmp_path) == []


class TestFindCollectionDir:
    def test_returns_dir_with_collection(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.imagery.catalog_ops import find_collection_dir

        (tmp_path / "collection.json").write_text("{}")
        assert find_collection_dir(tmp_path) == tmp_path

    def test_raises_without_collection(self, tmp_path: Path) -> None:
        import pytest

        from ftw_dataset_tools.api.imagery.catalog_ops import find_collection_dir

        with pytest.raises(FileNotFoundError, match=r"collection\.json"):
            find_collection_dir(tmp_path)


class TestClearChipSelections:
    """Tests for clear_chip_selections against a nested chips/<square>/<item>/ layout."""

    def test_clears_child_items_files_and_links(self, tmp_path: Path) -> None:
        from datetime import UTC, datetime

        import pystac

        from ftw_dataset_tools.api.imagery.catalog_ops import clear_chip_selections

        chip_dir = tmp_path / "chips" / "33UXP" / "chip1"
        chip_dir.mkdir(parents=True)
        # A sibling chip dir whose files must survive untouched.
        other_chip_dir = tmp_path / "chips" / "33UXP" / "chip2"
        other_chip_dir.mkdir(parents=True)

        # Parent item with existing planting/harvest scene links, saved to disk
        # the way the CLI does (self href set to its real location).
        parent = pystac.Item(
            id="chip1",
            geometry={
                "type": "Polygon",
                "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
            },
            bbox=(0.0, 0.0, 1.0, 1.0),
            datetime=datetime(2024, 1, 1, tzinfo=UTC),
            properties={"ftw:calendar_year": 2024},
        )
        parent.add_link(pystac.Link(rel="ftw:planting", target="./chip1_planting_s2.json"))
        parent.add_link(pystac.Link(rel="ftw:harvest", target="./chip1_harvest_s2.json"))
        parent_path = chip_dir / "chip1.json"
        parent.set_self_href(str(parent_path))
        parent.save_object(dest_href=str(parent_path))

        # Child STAC items, downloaded imagery and thumbnails.
        (chip_dir / "chip1_planting_s2.json").write_text("{}")
        (chip_dir / "chip1_harvest_s2.json").write_text("{}")
        (chip_dir / "chip1_planting_image_s2.tif").write_bytes(b"tif")
        (chip_dir / "chip1_harvest_image_s2.tif").write_bytes(b"tif")
        (chip_dir / "chip1_planting_image_s2.jpg").write_bytes(b"jpg")
        (chip_dir / "chip1_harvest_image_s2.jpg").write_bytes(b"jpg")
        (chip_dir / "chip1_overlay.jpg").write_bytes(b"jpg")

        # Unrelated files in a sibling chip dir must not be touched.
        (other_chip_dir / "chip2_planting_s2.json").write_text("{}")
        (other_chip_dir / "chip2_planting_image_s2.tif").write_bytes(b"tif")

        # Call the way the CLI does: pass the collection (dataset) dir plus
        # the in-memory parent item.
        result = clear_chip_selections(tmp_path, parent)

        assert result.stac_items_deleted == 2
        assert result.geotiffs_deleted == 2

        assert not (chip_dir / "chip1_planting_s2.json").exists()
        assert not (chip_dir / "chip1_harvest_s2.json").exists()
        assert not (chip_dir / "chip1_planting_image_s2.tif").exists()
        assert not (chip_dir / "chip1_harvest_image_s2.tif").exists()
        assert not (chip_dir / "chip1_planting_image_s2.jpg").exists()
        assert not (chip_dir / "chip1_harvest_image_s2.jpg").exists()
        assert not (chip_dir / "chip1_overlay.jpg").exists()

        # Links removed from the in-memory item...
        assert not any(link.rel in ("ftw:planting", "ftw:harvest") for link in parent.links)
        # ...and from what was saved to disk.
        saved = pystac.Item.from_file(str(parent_path))
        assert not any(link.rel in ("ftw:planting", "ftw:harvest") for link in saved.links)

        # Sibling chip's files are untouched.
        assert (other_chip_dir / "chip2_planting_s2.json").exists()
        assert (other_chip_dir / "chip2_planting_image_s2.tif").exists()


class TestClearChipSelectionsWithoutAResolvableRoot:
    """Clearing must work in a staging tree whose `rel: root` target is missing."""

    def test_clears_and_rewrites_the_item(self, tmp_path: Path) -> None:
        import json
        from datetime import UTC, datetime

        import pystac

        from ftw_dataset_tools.api.imagery.catalog_ops import clear_chip_selections

        chip_dir = tmp_path / "chips" / "33UXP" / "chip1"
        chip_dir.mkdir(parents=True)
        parent_path = chip_dir / "chip1.json"

        parent = pystac.Item(
            id="chip1",
            geometry={"type": "Point", "coordinates": [0.5, 0.5]},
            bbox=(0.0, 0.0, 1.0, 1.0),
            datetime=datetime(2024, 1, 1, tzinfo=UTC),
            properties={"ftw:calendar_year": 2024},
        )
        parent.add_link(pystac.Link(rel="root", target="../../../../catalog.json"))
        parent.add_link(pystac.Link(rel="ftw:planting", target="./chip1_planting_s2.json"))
        parent_path.write_text(
            json.dumps(parent.to_dict(include_self_link=False, transform_hrefs=False), indent=2)
        )
        (chip_dir / "chip1_planting_s2.json").write_text("{}")

        staged = pystac.Item.from_file(str(parent_path))
        result = clear_chip_selections(tmp_path, staged)

        assert result.stac_items_deleted == 1
        written = json.loads(parent_path.read_text())
        rels = {link["rel"]: link["href"] for link in written["links"]}
        assert rels["root"] == "../../../../catalog.json"
        assert "ftw:planting" not in rels
