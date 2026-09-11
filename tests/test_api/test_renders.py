"""Tests for the render definitions published on chip items and the collection."""

from __future__ import annotations

import pystac

GEOMETRY = {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]}


#: A four-band clipped season image as the download path writes it: the requested
#: bands in order (red, green, blue, nir), each named by its GDAL description, with
#: 0 declared as the reflectance fill. Mean and stddev are left out so the stretch
#: is the raw extremes; ``TestTrueColourStretch`` covers the statistics that have them.
IMAGE_BANDS = [
    {"description": "red", "nodata": 0, "statistics": {"minimum": 1067, "maximum": 7045}},
    {"description": "green", "nodata": 0, "statistics": {"minimum": 1077, "maximum": 3400}},
    {"description": "blue", "nodata": 0, "statistics": {"minimum": 1050, "maximum": 3295}},
    {"description": "nir", "nodata": 0, "statistics": {"minimum": 2000, "maximum": 9000}},
]


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
            assert renders[key]["nodata"] == 0
            assert "colormap" not in renders[key]
            assert "colormap_name" not in renders[key]

    def test_instance_stretches_over_the_ids_actually_present(self) -> None:
        """Instance ids are global, so the stretch must start at the chip's own minimum."""
        from ftw_dataset_tools.api.renders import build_item_renders

        renders = build_item_renders(
            _item({"instance_mask": [{"statistics": {"minimum": 1000, "maximum": 1010}}]})
        )

        assert renders["instance"]["rescale"] == [[1000, 1010]]
        assert renders["instance"]["nodata"] == 0
        assert renders["instance"]["colormap_name"] == "viridis"

    def test_instance_stretch_from_zero_when_the_minimum_is_zero(self) -> None:
        from ftw_dataset_tools.api.renders import build_item_renders

        renders = build_item_renders(
            _item({"instance_mask": [{"statistics": {"minimum": 0, "maximum": 945174}}]})
        )

        assert renders["instance"]["rescale"] == [[0, 945174]]

    def test_instance_falls_back_to_one_without_usable_statistics(self) -> None:
        from ftw_dataset_tools.api.renders import build_item_renders

        assert build_item_renders(_item({"instance_mask": None}))["instance"]["rescale"] == [[0, 1]]
        bandless = _item({"instance_mask": [{"data_type": "uint32"}]})
        assert build_item_renders(bandless)["instance"]["rescale"] == [[0, 1]]
        flat = _item({"instance_mask": [{"statistics": {"minimum": 7, "maximum": 7}}]})
        assert build_item_renders(flat)["instance"]["rescale"] == [[0, 1]]

    def test_decode_distance_is_a_continuous_ramp(self) -> None:
        from ftw_dataset_tools.api.renders import build_item_renders

        renders = build_item_renders(_item({"decode_distance_mask": None}))

        assert renders["decode_distance"]["rescale"] == [[0, 1]]
        assert renders["decode_distance"]["nodata"] == 0
        assert renders["decode_distance"]["colormap_name"] == "viridis"

    def test_decode_distance_uses_declared_band_nodata(self) -> None:
        from ftw_dataset_tools.api.renders import build_item_renders

        renders = build_item_renders(
            _item({"decode_distance_mask": [{"nodata": -1, "data_type": "float32"}]})
        )

        assert renders["decode_distance"]["nodata"] == -1


class TestBackgroundValue:
    """Presence-only datasets label background 3, so the renders must hide 3, not 0."""

    def test_dataset_background_reaches_the_class_valued_masks(self) -> None:
        from ftw_dataset_tools.api.renders import build_item_renders

        renders = build_item_renders(
            _item(
                {
                    "semantic_2class_mask": None,
                    "semantic_3class_mask": None,
                    "instance_mask": None,
                }
            ),
            background_value=3,
        )

        assert renders["semantic_2class"]["nodata"] == 3
        assert renders["semantic_3class"]["nodata"] == 3
        assert renders["instance"]["nodata"] == 3

    def test_decode_layers_always_use_zero(self) -> None:
        """The DECODE layers fold presence-only background into 0, so 0 it stays."""
        from ftw_dataset_tools.api.renders import build_item_renders

        renders = build_item_renders(
            _item({"decode_boundary_mask": None, "decode_distance_mask": None}),
            background_value=3,
        )

        assert renders["decode_boundary"]["nodata"] == 0
        assert renders["decode_distance"]["nodata"] == 0

    def test_declared_band_nodata_still_wins_for_decode_distance(self) -> None:
        from ftw_dataset_tools.api.renders import build_item_renders

        renders = build_item_renders(
            _item({"decode_distance_mask": [{"nodata": -1}]}), background_value=3
        )

        assert renders["decode_distance"]["nodata"] == -1

    def test_collection_renders_take_the_background_too(self) -> None:
        from ftw_dataset_tools.api.renders import build_collection_renders

        renders = build_collection_renders(background_value=3)

        assert renders["semantic_2class_mask"]["nodata"] == 3
        assert renders["instance_mask"]["nodata"] == 3
        assert renders["decode_boundary_mask"]["nodata"] == 0


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
        assert renders["instance_mask"]["nodata"] == 0
        assert renders["decode_distance_mask"]["rescale"] == [[0, 1]]

    def test_instance_has_no_global_stretch(self) -> None:
        """There is no meaningful collection-wide instance id range, so no rescale."""
        from ftw_dataset_tools.api.renders import build_collection_renders

        assert "rescale" not in build_collection_renders()["instance_mask"]


class TestBoundaryOverlayColouring:
    """The chosen overlay draws from the class hints, with its background transparent."""

    def _overlay(self, assets: dict) -> tuple[dict, pystac.Item]:
        """The render ``build_render_order`` picks as the overlay, and the item it came from."""
        from ftw_dataset_tools.api.renders import build_item_renders, build_render_order

        item = _item({"planting_image": IMAGE_BANDS, **assets})
        renders = build_item_renders(item)
        overlay_key = build_render_order(renders)[1]
        return renders[overlay_key], item

    def test_the_chosen_overlay_is_coloured_by_the_boundary_class_hint(self) -> None:
        from ftw_dataset_tools.api.assets import LABEL_COLORS, MASK_CLASSES

        overlay, item = self._overlay({"decode_boundary_mask": None})

        assert overlay["assets"] == ["decode_boundary_mask"]
        # No colormap on the render: colour comes from the band's classes, which
        # api.assets writes from this table when the asset is decorated.
        assert "colormap" not in overlay
        assert "colormap_name" not in overlay
        assert item.assets[overlay["assets"][0]].href.endswith(".tif")
        hints = {name: LABEL_COLORS.get(name) for _, name, _ in MASK_CLASSES["decode_boundary"]}
        assert hints == {"background": None, "boundary": "D55E00"}

    def test_the_chosen_overlay_hides_its_background(self) -> None:
        overlay, _ = self._overlay({"decode_boundary_mask": None})

        assert overlay["nodata"] == 0

    def test_the_fallback_overlay_hides_its_background_too(self) -> None:
        overlay, _ = self._overlay({"semantic_2class_mask": None})

        assert overlay["assets"] == ["semantic_2class_mask"]
        assert overlay["nodata"] == 0


class TestLabelColors:
    def test_colours_are_bare_six_digit_hex(self) -> None:
        from ftw_dataset_tools.api.assets import LABEL_COLORS

        assert LABEL_COLORS["field"] == "009E73"
        assert LABEL_COLORS["boundary"] == "D55E00"
        for value in LABEL_COLORS.values():
            assert len(value) == 6
            assert not value.startswith("#")
            int(value, 16)


class TestSeasonImageryRenders:
    """The true-colour base layer, from the chip's own image or the scene's visual COG."""

    def test_local_image_uses_named_bands_and_its_own_statistics(self) -> None:
        from ftw_dataset_tools.api.renders import build_item_renders

        renders = build_item_renders(_item({"planting_image": IMAGE_BANDS}))

        assert renders["planting_rgb"]["assets"] == ["planting_image"]
        assert renders["planting_rgb"]["bidx"] == [1, 2, 3]
        assert renders["planting_rgb"]["rescale"] == [[1067, 7045], [1077, 3400], [1050, 3295]]
        assert renders["planting_rgb"]["nodata"] == 0
        assert renders["planting_rgb"]["title"] == "Planting season (true colour)"

    def test_bidx_follows_the_band_names_not_their_position(self) -> None:
        """A stack written in another order must still be drawn as true colour."""
        from ftw_dataset_tools.api.renders import build_item_renders

        bands = [
            {"description": "nir", "statistics": {"minimum": 1, "maximum": 2}},
            {"description": "blue", "statistics": {"minimum": 3, "maximum": 4}},
            {"description": "green", "statistics": {"minimum": 5, "maximum": 6}},
            {"description": "red", "statistics": {"minimum": 7, "maximum": 8}},
        ]

        renders = build_item_renders(_item({"harvest_image": bands}))

        assert renders["harvest_rgb"]["bidx"] == [4, 3, 2]
        assert renders["harvest_rgb"]["rescale"] == [[7, 8], [5, 6], [3, 4]]

    def test_unnamed_bands_fall_back_to_the_first_three(self) -> None:
        from ftw_dataset_tools.api.renders import build_item_renders

        bands = [{"statistics": {"minimum": 1, "maximum": 2}} for _ in range(4)]

        assert build_item_renders(_item({"planting_image": bands}))["planting_rgb"]["bidx"] == [
            1,
            2,
            3,
        ]

    def test_a_named_stack_without_rgb_is_never_guessed_at(self) -> None:
        """``--bands nir,red,green`` is a supported download; drawing band 1 as red there
        would publish infrared under a "true colour" title."""
        from ftw_dataset_tools.api.renders import build_item_renders

        false_colour = [
            {"description": name, "statistics": {"minimum": 1, "maximum": 2}}
            for name in ("nir", "red", "green")
        ]

        assert build_item_renders(_item({"planting_image": false_colour})) == {}

    def test_a_named_stack_without_rgb_falls_through_to_the_scene_asset(self) -> None:
        from ftw_dataset_tools.api.renders import build_item_renders

        false_colour = [{"description": name} for name in ("nir", "red", "green")]

        renders = build_item_renders(
            _item({"planting_image": false_colour, "planting_visual": None})
        )

        assert renders["planting_rgb"]["assets"] == ["planting_visual"]
        assert renders["planting_rgb"]["bidx"] == [1, 2, 3]

    def test_a_partially_named_stack_is_not_guessed_at_either(self) -> None:
        """Some bands named and red missing is a stack nobody can draw as true colour."""
        from ftw_dataset_tools.api.renders import build_item_renders

        bands = [{"description": "swir16"}, {"description": "nir"}, {}, {}]

        assert build_item_renders(_item({"planting_image": bands})) == {}

    def test_nodata_is_published_as_an_integer(self) -> None:
        """rasterio reports an integer fill as a float; every other render publishes 0."""
        from ftw_dataset_tools.api.renders import build_item_renders

        bands = [{"description": name, "nodata": 0.0} for name in ("red", "green", "blue")]

        nodata = build_item_renders(_item({"planting_image": bands}))["planting_rgb"]["nodata"]
        assert nodata == 0
        assert isinstance(nodata, int)

    def test_nodata_is_omitted_when_the_colour_bands_disagree(self) -> None:
        """One value is published for all three bands, so it is only published if it holds."""
        from ftw_dataset_tools.api.renders import build_item_renders

        bands = [
            {"description": "red", "nodata": 0},
            {"description": "green", "nodata": 0},
            {"description": "blue"},
        ]

        assert "nodata" not in build_item_renders(_item({"planting_image": bands}))["planting_rgb"]

    def test_undeclared_nodata_is_not_invented(self) -> None:
        """A stack sharing non-reflectance bands declares no nodata; 0 is a real value there."""
        from ftw_dataset_tools.api.renders import build_item_renders

        bands = [{"description": name} for name in ("red", "green", "blue", "scl")]

        assert "nodata" not in build_item_renders(_item({"planting_image": bands}))["planting_rgb"]

    def test_missing_statistics_leave_the_stretch_to_the_client(self) -> None:
        from ftw_dataset_tools.api.renders import build_item_renders

        bands = [{"description": name, "nodata": 0} for name in ("red", "green", "blue")]

        render = build_item_renders(_item({"planting_image": bands}))["planting_rgb"]
        assert "rescale" not in render
        assert render["bidx"] == [1, 2, 3]

    def test_visual_scene_asset_is_the_fallback_base_layer(self) -> None:
        """Most chips have a selection but no local download, so the scene COG is drawn."""
        from ftw_dataset_tools.api.renders import build_item_renders

        renders = build_item_renders(_item({"planting_visual": None}))

        assert renders["planting_rgb"]["assets"] == ["planting_visual"]
        assert renders["planting_rgb"]["bidx"] == [1, 2, 3]
        assert renders["planting_rgb"]["rescale"] == [[0, 255], [0, 255], [0, 255]]
        assert renders["planting_rgb"]["nodata"] == 0

    def test_local_image_wins_over_the_scene_asset(self) -> None:
        from ftw_dataset_tools.api.renders import build_item_renders

        renders = build_item_renders(
            _item({"planting_image": IMAGE_BANDS, "planting_visual": None})
        )

        assert renders["planting_rgb"]["assets"] == ["planting_image"]

    def test_unusable_local_image_falls_through_to_the_scene_asset(self) -> None:
        """An image asset added before its file existed carries no bands to select from."""
        from ftw_dataset_tools.api.renders import build_item_renders

        renders = build_item_renders(_item({"planting_image": None, "planting_visual": None}))

        assert renders["planting_rgb"]["assets"] == ["planting_visual"]

    def test_unusable_local_image_alone_gets_no_render(self) -> None:
        from ftw_dataset_tools.api.renders import build_item_renders

        assert build_item_renders(_item({"planting_image": [{"description": "red"}]})) == {}

    def test_both_seasons_get_their_own_render(self) -> None:
        from ftw_dataset_tools.api.renders import build_item_renders

        renders = build_item_renders(_item({"planting_image": IMAGE_BANDS, "harvest_visual": None}))

        assert renders["planting_rgb"]["assets"] == ["planting_image"]
        assert renders["harvest_rgb"]["assets"] == ["harvest_visual"]
        assert renders["harvest_rgb"]["title"] == "Harvest season (true colour)"


class TestTrueColourStretch:
    """A raw min/max stretch lets one bright pixel crush the image, so clip to mean +/- 2 sigma."""

    def _rescale(self, statistics: dict) -> list[list[float]]:
        from ftw_dataset_tools.api.renders import build_item_renders

        bands = [
            {"description": name, "statistics": dict(statistics)}
            for name in ("red", "green", "blue")
        ]
        return build_item_renders(_item({"planting_image": bands}))["planting_rgb"]["rescale"]

    def test_an_outlier_maximum_does_not_crush_the_stretch(self) -> None:
        statistics = {"minimum": 100, "maximum": 9000, "mean": 1000, "stddev": 200}

        assert self._rescale(statistics) == [[600, 1400]] * 3

    def test_the_stretch_never_runs_past_the_band_extremes(self) -> None:
        statistics = {"minimum": 900, "maximum": 1100, "mean": 1000, "stddev": 500}

        assert self._rescale(statistics) == [[900, 1100]] * 3

    def test_a_flat_band_falls_back_to_its_extremes(self) -> None:
        """Nearly-identical values collapse mean +/- 2 sigma to nothing."""
        statistics = {"minimum": 5, "maximum": 10, "mean": 7, "stddev": 0}

        assert self._rescale(statistics) == [[5, 10]] * 3

    def test_statistics_without_a_mean_fall_back_to_the_extremes(self) -> None:
        assert self._rescale({"minimum": 5, "maximum": 10}) == [[5, 10]] * 3


class TestBuildRenderOrder:
    """The default stack a Portolan browser opens a chip on: fields over imagery."""

    def test_boundary_outline_is_drawn_over_the_planting_image(self) -> None:
        from ftw_dataset_tools.api.renders import build_item_renders, build_render_order

        renders = build_item_renders(
            _item(
                {
                    "planting_image": IMAGE_BANDS,
                    "semantic_2class_mask": None,
                    "decode_boundary_mask": None,
                }
            )
        )

        assert build_render_order(renders) == ["planting_rgb", "decode_boundary"]

    def test_binary_mask_is_the_overlay_without_a_decode_boundary(self) -> None:
        from ftw_dataset_tools.api.renders import build_item_renders, build_render_order

        renders = build_item_renders(
            _item({"planting_image": IMAGE_BANDS, "semantic_2class_mask": None})
        )

        assert build_render_order(renders) == ["planting_rgb", "semantic_2class"]

    def test_harvest_is_the_base_when_only_harvest_has_imagery(self) -> None:
        from ftw_dataset_tools.api.renders import build_item_renders, build_render_order

        renders = build_item_renders(
            _item({"harvest_image": IMAGE_BANDS, "decode_boundary_mask": None})
        )

        assert build_render_order(renders) == ["harvest_rgb", "decode_boundary"]

    def test_planting_is_the_base_when_both_seasons_have_imagery(self) -> None:
        from ftw_dataset_tools.api.renders import build_item_renders, build_render_order

        renders = build_item_renders(
            _item(
                {
                    "planting_visual": None,
                    "harvest_image": IMAGE_BANDS,
                    "decode_boundary_mask": None,
                }
            )
        )

        assert build_render_order(renders) == ["planting_rgb", "decode_boundary"]

    def test_no_imagery_means_no_stack_at_all(self) -> None:
        """A mask-only stack would change nothing a viewer's default does not already do."""
        from ftw_dataset_tools.api.renders import build_item_renders, build_render_order

        renders = build_item_renders(
            _item({"semantic_2class_mask": None, "decode_boundary_mask": None})
        )

        assert build_render_order(renders) == []

    def test_imagery_without_an_overlay_still_opens_on_the_imagery(self) -> None:
        from ftw_dataset_tools.api.renders import build_item_renders, build_render_order

        renders = build_item_renders(
            _item({"planting_image": IMAGE_BANDS, "semantic_3class_mask": None})
        )

        assert build_render_order(renders) == ["planting_rgb"]

    def test_every_key_names_a_render_of_the_same_item(self) -> None:
        from ftw_dataset_tools.api.renders import build_item_renders, build_render_order

        renders = build_item_renders(
            _item({"planting_image": IMAGE_BANDS, "decode_boundary_mask": None})
        )

        assert all(key in renders for key in build_render_order(renders))
