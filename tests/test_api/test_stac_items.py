"""Tests for STAC item save/update helpers."""

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
        from ftw_dataset_tools.api.stac_items import STACSaveError, update_parent_item

        parent, path = _parent(tmp_path)
        _cog(tmp_path / "chip_planting_image_s2.tif")
        (tmp_path / "chip_planting_image_s2.jpg").write_bytes(b"\xff\xd8\xff\xd9")

        def boom(*_args, **_kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(parent, "save_object", boom)

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
