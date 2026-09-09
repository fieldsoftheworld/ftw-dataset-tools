"""Tests for STAC item save/update helpers."""

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pystac
import pytest
from rasterio.transform import from_bounds


def _parent(tmp_path: Path) -> tuple[pystac.Item, Path]:
    item = pystac.Item(
        id="chip",
        geometry={"type": "Point", "coordinates": [0.5, 0.5]},
        bbox=[0, 0, 1, 1],
        datetime=None,
        properties={
            "start_datetime": "2024-01-01T00:00:00Z",
            "end_datetime": "2024-12-31T00:00:00Z",
        },
    )
    path = tmp_path / "chip.json"
    item.set_self_href(str(path))
    item.save_object(dest_href=str(path))
    return item, path


def _cog(path: Path) -> None:
    import rasterio

    with rasterio.open(
        path,
        "w",
        driver="COG",
        width=2,
        height=2,
        count=1,
        dtype="uint16",
        crs="EPSG:4326",
        transform=from_bounds(0, 0, 1, 1, 2, 2),
        compress="deflate",
    ) as dst:
        dst.write(np.zeros((1, 2, 2), dtype=np.uint16))


class TestUpdateParentItem:
    def test_decorates_image_and_thumbnail(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.stac_items import update_parent_item

        parent, path = _parent(tmp_path)
        _cog(tmp_path / "chip_planting_image_s2.tif")
        (tmp_path / "chip_planting_image_s2.jpg").write_bytes(b"\xff\xd8\xff\xd9")

        update_parent_item(
            parent,
            path,
            "planting",
            "chip_planting_image_s2.tif",
            ["red", "green", "blue", "nir"],
            thumbnail_filename="chip_planting_image_s2.jpg",
        )

        saved = pystac.Item.from_file(str(path))
        assert saved.assets["planting_image"].extra_fields["file:size"] > 0
        assert "raster:bands" in saved.assets["planting_image"].extra_fields
        assert saved.assets["thumbnail"].extra_fields["file:size"] == 4

    def test_missing_files_leave_assets_undecorated(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.stac_items import update_parent_item

        parent, path = _parent(tmp_path)

        update_parent_item(parent, path, "harvest", "missing.tif", ["red"])

        saved = pystac.Item.from_file(str(path))
        assert "file:size" not in saved.assets["harvest_image"].extra_fields

    def test_rollback_removes_both_assets_when_save_fails(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        from ftw_dataset_tools.api import stac_items as stac_items_module
        from ftw_dataset_tools.api.stac_items import STACSaveError, update_parent_item

        parent, path = _parent(tmp_path)
        _cog(tmp_path / "chip_planting_image_s2.tif")
        (tmp_path / "chip_planting_image_s2.jpg").write_bytes(b"\xff\xd8\xff\xd9")

        def boom(*_args, **_kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(stac_items_module, "write_item", boom)

        with pytest.raises(STACSaveError):
            update_parent_item(
                parent,
                path,
                "planting",
                "chip_planting_image_s2.tif",
                ["red"],
                thumbnail_filename="chip_planting_image_s2.jpg",
            )

        assert "planting_image" not in parent.assets
        assert "thumbnail" not in parent.assets

    def test_rollback_when_thumbnail_decoration_fails(self, tmp_path: Path, monkeypatch) -> None:
        from ftw_dataset_tools.api import stac_items
        from ftw_dataset_tools.api.stac_items import STACSaveError, update_parent_item

        parent, path = _parent(tmp_path)
        _cog(tmp_path / "chip_planting_image_s2.tif")
        (tmp_path / "chip_planting_image_s2.jpg").write_bytes(b"\xff\xd8\xff\xd9")
        calls = {"n": 0}
        real = stac_items.add_file_info

        def flaky(asset, file_path, **kwargs):
            calls["n"] += 1
            if calls["n"] == 2:  # the thumbnail decoration
                raise OSError("stat failed")
            return real(asset, file_path, **kwargs)

        monkeypatch.setattr(stac_items, "add_file_info", flaky)

        with pytest.raises(STACSaveError):
            update_parent_item(
                parent,
                path,
                "planting",
                "chip_planting_image_s2.tif",
                ["red"],
                thumbnail_filename="chip_planting_image_s2.jpg",
            )

        assert "thumbnail" not in parent.assets
        assert "planting_image" not in parent.assets


def _write_item_json(path: Path, item: pystac.Item) -> str:
    """Serialize `item` the way a staged catalog stores it and return the text."""
    text = json.dumps(item.to_dict(include_self_link=False, transform_hrefs=False), indent=2) + "\n"
    path.write_text(text)
    return text


def _staged_chip_item(tmp_path: Path) -> tuple[Path, dict[str, str], dict[str, str]]:
    """Write a chip item whose `root` link points at a file that does not exist.

    Mirrors the real staging tree: `chips/<square>/<item>/<item>.json` with the
    hierarchical links rewritten to the *published* root.
    """
    item_dir = tmp_path / "chips" / "31UFR" / "ftw-chip"
    item_dir.mkdir(parents=True)
    path = item_dir / "ftw-chip.json"

    item = pystac.Item(
        id="ftw-chip",
        geometry={"type": "Point", "coordinates": [0.5, 0.5]},
        bbox=[0, 0, 1, 1],
        datetime=datetime(2024, 1, 1, tzinfo=UTC),
        properties={},
        collection="lu",
    )
    item.add_link(pystac.Link(rel="root", target="../../../collection.json"))
    item.add_link(pystac.Link(rel="parent", target="../catalog.json"))
    item.add_link(pystac.Link(rel="collection", target="../../../collection.json"))
    item.add_asset("mask", pystac.Asset(href="./x.tif", media_type="image/tiff"))
    item.add_asset("remote", pystac.Asset(href="https://example.com/scene.tif"))

    _write_item_json(path, item)
    links = {link["rel"]: link["href"] for link in json.loads(path.read_text())["links"]}
    assets = {k: a["href"] for k, a in json.loads(path.read_text())["assets"].items()}
    return path, links, assets


class TestWriteItem:
    """`write_item` writes an item without resolving its root link."""

    def test_save_object_still_fails_on_a_missing_root(self, tmp_path: Path) -> None:
        """The bug this function exists for: pystac resolves the root when saving."""
        path, _links, _assets = _staged_chip_item(tmp_path)
        item = pystac.Item.from_file(str(path))

        with pytest.raises(pystac.STACError):
            item.save_object(dest_href=str(path))

    def test_roundtrips_relative_links_with_an_unresolvable_root(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.stac_items import write_item

        path, links, assets = _staged_chip_item(tmp_path)
        item = pystac.Item.from_file(str(path))

        returned = write_item(item, path)

        assert returned == path
        written = json.loads(path.read_text())
        assert {link["rel"]: link["href"] for link in written["links"]} == links
        assert {k: a["href"] for k, a in written["assets"].items()} == assets
        # Still a readable STAC item afterwards.
        assert pystac.Item.from_file(str(path)).id == "ftw-chip"

    def test_keeps_remote_asset_hrefs_absolute(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.stac_items import write_item

        path, _links, _assets = _staged_chip_item(tmp_path)
        item = pystac.Item.from_file(str(path))

        write_item(item, path)

        written = json.loads(path.read_text())
        assert written["assets"]["remote"]["href"] == "https://example.com/scene.tif"

    def test_relativizes_absolute_local_hrefs(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.stac_items import write_item

        path, _links, _assets = _staged_chip_item(tmp_path)
        item = pystac.Item.from_file(str(path))
        item.add_link(pystac.Link(rel="ftw:planting", target=str(path.parent / "child.json")))
        item.add_asset("local", pystac.Asset(href=f"file://{path.parent}/local.tif"))
        item.add_asset("sibling", pystac.Asset(href=str(path.parent.parent / "other.tif")))

        write_item(item, path)

        written = json.loads(path.read_text())
        hrefs = {link["rel"]: link["href"] for link in written["links"]}
        assert hrefs["ftw:planting"] == "./child.json"
        assert written["assets"]["local"]["href"] == "./local.tif"
        assert written["assets"]["sibling"]["href"] == "../other.tif"

    def test_writes_pretty_json_with_trailing_newline(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.stac_items import write_item

        path, _links, _assets = _staged_chip_item(tmp_path)
        item = pystac.Item.from_file(str(path))

        write_item(item, path)

        text = path.read_text()
        assert text.endswith("}\n")
        assert "\n  " in text

    def test_omits_the_self_link(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.stac_items import write_item

        path, _links, _assets = _staged_chip_item(tmp_path)
        item = pystac.Item.from_file(str(path))

        write_item(item, path)

        rels = [link["rel"] for link in json.loads(path.read_text())["links"]]
        assert "self" not in rels
