"""Tests for chip previews rendered from a remote scene, with no local clip."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 - used at runtime in helpers

import numpy as np
import pystac
import pytest
import rasterio
from rasterio.transform import from_bounds

from ftw_dataset_tools.api.imagery.preview_workflow import (
    build_preview_task,
    preview_imagery_for_catalog,
    preview_summary_line,
)

CRS = "EPSG:32633"


def _scene(path: Path) -> Path:
    """A 'scene' COG far larger than one chip."""
    width = height = 300
    transform = from_bounds(500000, 5000000, 503000, 5003000, width, height)
    data = np.zeros((3, height, width), dtype="uint8")
    ramp = np.linspace(0, 255, width, dtype="uint8")
    data[0, :, :] = ramp[None, :]
    data[1, :, :] = ramp[::-1][None, :]
    data[2, :, :] = 100
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=3,
        dtype="uint8",
        crs=CRS,
        transform=transform,
    ) as dst:
        dst.write(data)
    return path


def _chip(collection_dir: Path, chip_id: str, *, scene: Path | None, mask: bool = True) -> Path:
    """A chip directory with a parent item, a mask, and optionally a season child."""
    chip_dir = collection_dir / "chips" / "33TXM" / chip_id
    chip_dir.mkdir(parents=True, exist_ok=True)

    if mask:
        size = 32
        transform = from_bounds(501000, 5001000, 501320, 5001320, size, size)
        arr = np.zeros((size, size), dtype="uint8")
        arr[5:20, 5:20] = 1
        arr[20:24, 5:20] = 2
        with rasterio.open(
            chip_dir / f"{chip_id}_semantic_3_class.tif",
            "w",
            driver="GTiff",
            width=size,
            height=size,
            count=1,
            dtype="uint8",
            crs=CRS,
            transform=transform,
        ) as dst:
            dst.write(arr, 1)

    item = pystac.Item(
        id=chip_id,
        geometry={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
        bbox=[0.0, 0.0, 1.0, 1.0],
        datetime=None,
        start_datetime=__import__("datetime").datetime(
            2024, 1, 1, tzinfo=__import__("datetime").UTC
        ),
        end_datetime=__import__("datetime").datetime(
            2024, 12, 31, tzinfo=__import__("datetime").UTC
        ),
        properties={},
    )
    item.set_self_href(str(chip_dir / f"{chip_id}.json"))
    item.save_object(include_self_link=False, dest_href=str(chip_dir / f"{chip_id}.json"))

    if scene is not None:
        child = pystac.Item(
            id=f"{chip_id}_planting_s2",
            geometry=item.geometry,
            bbox=item.bbox,
            datetime=None,
            start_datetime=item.common_metadata.start_datetime,
            end_datetime=item.common_metadata.end_datetime,
            properties={},
        )
        child.add_asset("visual", pystac.Asset(href=str(scene), roles=["visual"]))
        child_path = chip_dir / f"{chip_id}_planting_s2.json"
        child.set_self_href(str(child_path))
        child.save_object(include_self_link=False, dest_href=str(child_path))

    return chip_dir


def _collection(tmp_path: Path) -> Path:
    out = tmp_path / "ds"
    out.mkdir(parents=True, exist_ok=True)
    (out / "collection.json").write_text("{}")
    return out


class TestBuildPreviewTask:
    def test_skips_a_chip_with_no_mask(self, tmp_path: Path) -> None:
        out = _collection(tmp_path)
        scene = _scene(tmp_path / "scene.tif")
        chip_dir = _chip(out, "chip_a", scene=scene, mask=False)
        item = pystac.Item.from_file(str(chip_dir / "chip_a.json"))
        assert build_preview_task(item, chip_dir / "chip_a.json") == (
            "No semantic mask to use as the preview grid"
        )

    def test_skips_a_chip_with_no_selected_scene(self, tmp_path: Path) -> None:
        out = _collection(tmp_path)
        chip_dir = _chip(out, "chip_b", scene=None)
        item = pystac.Item.from_file(str(chip_dir / "chip_b.json"))
        assert build_preview_task(item, chip_dir / "chip_b.json") == (
            "No remote true-colour scene selected"
        )

    def test_rejects_a_relative_href(self, tmp_path: Path) -> None:
        """A relative href is a locally written file, not a scene to read a window from."""
        out = _collection(tmp_path)
        scene = _scene(tmp_path / "scene.tif")
        chip_dir = _chip(out, "chip_c", scene=scene)
        child_path = chip_dir / "chip_c_planting_s2.json"
        child = pystac.Item.from_file(str(child_path))
        child.assets["visual"].href = "./chip_c_planting_image_s2.tif"
        child.save_object(include_self_link=False, dest_href=str(child_path))

        item = pystac.Item.from_file(str(chip_dir / "chip_c.json"))
        assert build_preview_task(item, chip_dir / "chip_c.json") == (
            "No remote true-colour scene selected"
        )


class TestPreviewImageryForCatalog:
    def test_renders_the_overlay_and_attaches_it_to_the_parent(self, tmp_path: Path) -> None:
        out = _collection(tmp_path)
        scene = _scene(tmp_path / "scene.tif")
        _chip(out, "chip_a", scene=scene)

        result = preview_imagery_for_catalog(out, show_progress_bar=False, workers=1)

        assert (result.successful, result.skipped, result.failed) == (1, 0, 0)
        overlay = out / "chips" / "33TXM" / "chip_a" / "chip_a_overlay.jpg"
        assert overlay.exists() and overlay.stat().st_size > 0

        item = pystac.Item.from_file(str(out / "chips" / "33TXM" / "chip_a" / "chip_a.json"))
        assert "thumbnail" in item.assets
        assert item.assets["thumbnail"].href == "./chip_a_overlay.jpg"

    def test_leaves_no_intermediate_base_file(self, tmp_path: Path) -> None:
        out = _collection(tmp_path)
        scene = _scene(tmp_path / "scene.tif")
        chip_dir = _chip(out, "chip_a", scene=scene)
        preview_imagery_for_catalog(out, show_progress_bar=False, workers=1)
        leftovers = [p.name for p in chip_dir.iterdir() if p.name.startswith(".")]
        assert leftovers == [], leftovers

    def test_resume_skips_an_existing_preview(self, tmp_path: Path) -> None:
        out = _collection(tmp_path)
        scene = _scene(tmp_path / "scene.tif")
        _chip(out, "chip_a", scene=scene)
        preview_imagery_for_catalog(out, show_progress_bar=False, workers=1)

        again = preview_imagery_for_catalog(out, resume=True, show_progress_bar=False, workers=1)
        assert again.successful == 0
        assert again.skipped == 1
        assert again.skipped_details[0]["reason"] == "Already rendered"

    def test_an_unreadable_scene_is_reported_not_swallowed(self, tmp_path: Path) -> None:
        out = _collection(tmp_path)
        broken = tmp_path / "broken.tif"
        broken.write_bytes(b"not a raster")
        _chip(out, "chip_a", scene=broken)

        result = preview_imagery_for_catalog(out, show_progress_bar=False, workers=1)

        assert result.successful == 0
        assert result.failed == 1
        assert result.failed_details[0]["chip"] == "chip_a"

    def test_summary_line_reports_all_three_counts(self) -> None:
        from ftw_dataset_tools.api.imagery.preview_workflow import PreviewWorkflowResult

        line = preview_summary_line(PreviewWorkflowResult(successful=3, skipped=2, failed=1))
        assert line == "Chip previews: 3 ok, 2 skipped, 1 failed"


class TestPipelineDispatchesOnMode:
    def test_preview_mode_renders_previews_instead_of_downloading(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from ftw_dataset_tools.api import pipeline

        out = _collection(tmp_path)
        called: list[str] = []
        monkeypatch.setattr(
            pipeline,
            "download_imagery_for_catalog",
            lambda **_k: called.append("download"),
        )
        monkeypatch.setattr(
            pipeline,
            "preview_imagery_for_catalog",
            lambda **_k: called.append("preview"),
        )

        ctx = _ctx_with_mode(pipeline, tmp_path, out, "preview")
        pipeline.stage_download_images(ctx)
        assert called == ["preview"]

    def test_clip_mode_still_downloads(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from ftw_dataset_tools.api import pipeline

        out = _collection(tmp_path)
        called: list[str] = []
        monkeypatch.setattr(
            pipeline,
            "download_imagery_for_catalog",
            lambda **_k: called.append("download"),
        )
        monkeypatch.setattr(
            pipeline,
            "preview_imagery_for_catalog",
            lambda **_k: called.append("preview"),
        )

        ctx = _ctx_with_mode(pipeline, tmp_path, out, "clip")
        pipeline.stage_download_images(ctx)
        assert called == ["download"]


def _ctx_with_mode(pipeline_module, tmp_path: Path, out: Path, mode: str):
    """A context whose download stage is configured with the given mode."""
    import geopandas as gpd
    from shapely.geometry import box

    from ftw_dataset_tools.api.config import DatasetConfig

    fields = tmp_path / "fields.parquet"
    gpd.GeoDataFrame({"id": [1]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:4326").to_parquet(fields)
    config = DatasetConfig.from_dict(
        {
            "fields_file": str(fields),
            "output_dir": str(out),
            "year": 2024,
            "stages": {"download_images": {"enabled": True, "mode": mode}},
        }
    )
    return pipeline_module.build_context(config)


class TestModeValidation:
    def test_an_unknown_mode_is_rejected(self, tmp_path: Path) -> None:
        import geopandas as gpd
        from shapely.geometry import box

        from ftw_dataset_tools.api.config import ConfigError, DatasetConfig

        fields = tmp_path / "f.parquet"
        gpd.GeoDataFrame({"id": [1]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:4326").to_parquet(
            fields
        )
        with pytest.raises(ConfigError, match=r"download_images\.mode"):
            DatasetConfig.from_dict(
                {
                    "fields_file": str(fields),
                    "output_dir": str(tmp_path / "o"),
                    "year": 2024,
                    "stages": {"download_images": {"mode": "bogus"}},
                }
            )
