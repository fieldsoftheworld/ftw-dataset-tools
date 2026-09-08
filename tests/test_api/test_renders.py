"""Tests for the render definitions published on chip items and the collection."""

from __future__ import annotations

import pystac

GEOMETRY = {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]}


def _item(assets: dict[str, list[dict] | None]) -> pystac.Item:
    """Build a chip-like item whose assets carry the given ``raster:bands``."""
    item = pystac.Item(
        id="chip",
        geometry=GEOMETRY,
        bbox=[0.0, 0.0, 1.0, 1.0],
        datetime=None,
        properties={
            "start_datetime": "2024-01-01T00:00:00Z",
            "end_datetime": "2024-12-31T00:00:00Z",
        },
    )
    for key, bands in assets.items():
        asset = pystac.Asset(href=f"./{key}.tif", media_type="image/tiff", roles=["labels"])
        if bands is not None:
            asset.extra_fields["raster:bands"] = bands
        item.add_asset(key, asset)
    return item


class TestBuildItemRenders:
    def test_only_assets_present_get_renders(self) -> None:
        from ftw_dataset_tools.api.renders import build_item_renders

        renders = build_item_renders(_item({"semantic_2class_mask": None}))

        assert set(renders) == {"semantic_2class"}
        assert renders["semantic_2class"]["assets"] == ["semantic_2class_mask"]
        assert renders["semantic_2class"]["title"]

    def test_no_label_assets_means_no_renders(self) -> None:
        from ftw_dataset_tools.api.renders import build_item_renders

        assert build_item_renders(_item({"thumbnail": None})) == {}

    def test_categorical_renders_carry_no_colormap(self) -> None:
        """Colours for categorical masks come from classification:classes, not renders."""
        from ftw_dataset_tools.api.renders import build_item_renders

        renders = build_item_renders(
            _item(
                {
                    "semantic_2class_mask": None,
                    "semantic_3class_mask": None,
                    "decode_boundary_mask": None,
                }
            )
        )

        for key in ("semantic_2class", "semantic_3class", "decode_boundary"):
            assert set(renders[key]) == {"assets", "title", "nodata"}
            assert renders[key]["nodata"] == [0]
            assert "colormap" not in renders[key]
            assert "colormap_name" not in renders[key]

    def test_instance_rescales_to_band_maximum(self) -> None:
        from ftw_dataset_tools.api.renders import build_item_renders

        renders = build_item_renders(
            _item({"instance_mask": [{"statistics": {"minimum": 0, "maximum": 945174}}]})
        )

        assert renders["instance"]["rescale"] == [[0, 945174]]
        assert renders["instance"]["nodata"] == [0]
        assert renders["instance"]["colormap_name"] == "viridis"

    def test_instance_falls_back_to_one_without_statistics(self) -> None:
        from ftw_dataset_tools.api.renders import build_item_renders

        assert build_item_renders(_item({"instance_mask": None}))["instance"]["rescale"] == [[0, 1]]
        bandless = _item({"instance_mask": [{"data_type": "uint32"}]})
        assert build_item_renders(bandless)["instance"]["rescale"] == [[0, 1]]

    def test_decode_distance_is_a_continuous_ramp(self) -> None:
        from ftw_dataset_tools.api.renders import build_item_renders

        renders = build_item_renders(_item({"decode_distance_mask": None}))

        assert renders["decode_distance"]["rescale"] == [[0, 1]]
        assert renders["decode_distance"]["nodata"] == [0]
        assert renders["decode_distance"]["colormap_name"] == "viridis"

    def test_decode_distance_uses_declared_band_nodata(self) -> None:
        from ftw_dataset_tools.api.renders import build_item_renders

        renders = build_item_renders(
            _item({"decode_distance_mask": [{"nodata": -1, "data_type": "float32"}]})
        )

        assert renders["decode_distance"]["nodata"] == [-1]


class TestBuildCollectionRenders:
    def test_keyed_by_asset_name_and_mirrors_item_renders(self) -> None:
        from ftw_dataset_tools.api.renders import build_collection_renders

        renders = build_collection_renders()

        assert set(renders) == {
            "semantic_2class_mask",
            "semantic_3class_mask",
            "decode_boundary_mask",
            "decode_distance_mask",
            "instance_mask",
        }
        assert renders["semantic_3class_mask"]["assets"] == ["semantic_3class_mask"]
        assert "colormap" not in renders["semantic_3class_mask"]
        assert renders["instance_mask"]["colormap_name"] == "viridis"


class TestLabelColors:
    def test_colours_are_bare_six_digit_hex(self) -> None:
        from ftw_dataset_tools.api.assets import LABEL_COLORS

        assert LABEL_COLORS["field"] == "009E73"
        assert LABEL_COLORS["boundary"] == "D55E00"
        for value in LABEL_COLORS.values():
            assert len(value) == 6
            assert not value.startswith("#")
            int(value, 16)
