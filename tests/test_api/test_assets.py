"""Tests for STAC asset decoration helpers."""

import hashlib
from pathlib import Path

import numpy as np
import pystac
import rasterio
from rasterio.transform import from_bounds


def _write_cog(
    path: Path,
    data: np.ndarray,
    nodata: int | float | None = None,
    crs: str = "EPSG:4326",
    transform: rasterio.Affine | None = None,
) -> None:
    from ftw_dataset_tools.api.raster_stats import compute_band_stats, embed_band_stats

    profile = {
        "driver": "COG",
        "width": data.shape[1],
        "height": data.shape[0],
        "count": 1,
        "dtype": data.dtype,
        "crs": crs,
        "transform": transform or from_bounds(10, 50, 10.02, 50.02, data.shape[1], data.shape[0]),
        "compress": "deflate",
    }
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)
        embed_band_stats(dst, 1, compute_band_stats(data, nodata=nodata))


def _item_with_asset(href_path: Path, roles: list[str]) -> tuple[pystac.Item, pystac.Asset]:
    item = pystac.Item(
        id="x",
        geometry={"type": "Point", "coordinates": [10, 50]},
        bbox=[10, 50, 10, 50],
        datetime=None,
        properties={
            "start_datetime": "2024-01-01T00:00:00Z",
            "end_datetime": "2024-12-31T00:00:00Z",
        },
    )
    asset = pystac.Asset(href=f"./{href_path.name}", media_type="image/tiff", roles=roles)
    item.add_asset("a", asset)
    return item, asset


class TestMultihash:
    def test_sha256_multihash_prefix(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.assets import multihash_sha256

        p = tmp_path / "f.bin"
        p.write_bytes(b"hello")
        expected = "1220" + hashlib.sha256(b"hello").hexdigest()

        assert multihash_sha256(p) == expected


class TestAddFileInfo:
    def test_size_only_by_default(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.assets import add_file_info

        p = tmp_path / "f.bin"
        p.write_bytes(b"abc")
        item, asset = _item_with_asset(p, ["data"])

        add_file_info(asset, p)

        assert asset.extra_fields["file:size"] == 3
        assert "file:checksum" not in asset.extra_fields
        assert "https://stac-extensions.github.io/file/v2.1.0/schema.json" in item.stac_extensions

    def test_checksum_when_requested(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.assets import add_file_info, multihash_sha256

        p = tmp_path / "f.bin"
        p.write_bytes(b"abc")
        _, asset = _item_with_asset(p, ["data"])

        add_file_info(asset, p, checksum=True)

        assert asset.extra_fields["file:checksum"] == multihash_sha256(p)


class TestAddRasterBands:
    def test_band_fields_from_file(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.assets import add_raster_bands

        p = tmp_path / "r.tif"
        _write_cog(p, np.array([[0, 1], [2, 0]], dtype=np.uint8))
        item, asset = _item_with_asset(p, ["labels"])

        add_raster_bands(asset, p)

        bands = asset.extra_fields["raster:bands"]
        assert len(bands) == 1
        assert bands[0]["data_type"] == "uint8"
        assert "nodata" not in bands[0]
        # 0.01 degrees at EPSG:4326, converted to metres per the raster extension's
        # spatial_resolution contract (matching masks._grid_raster_geometry's factor).
        assert abs(bands[0]["spatial_resolution"] - 1110.0) < 1e-6
        assert bands[0]["statistics"]["minimum"] == 0
        assert bands[0]["statistics"]["maximum"] == 2
        assert "https://stac-extensions.github.io/raster/v1.1.0/schema.json" in item.stac_extensions

    def test_spatial_resolution_projected_crs_stays_in_metres(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.assets import add_raster_bands

        p = tmp_path / "utm.tif"
        _write_cog(
            p,
            np.array([[0, 1], [2, 0]], dtype=np.uint8),
            crs="EPSG:32633",
            transform=from_bounds(500000, 5000000, 500020, 5000020, 2, 2),
        )
        _, asset = _item_with_asset(p, ["data"])

        add_raster_bands(asset, p)

        band = asset.extra_fields["raster:bands"][0]
        assert band["spatial_resolution"] == 10.0

    def test_nodata_and_valid_percent(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.assets import add_raster_bands

        p = tmp_path / "n.tif"
        _write_cog(p, np.array([[9, 1], [2, 9]], dtype=np.uint8), nodata=9)
        _, asset = _item_with_asset(p, ["data"])

        add_raster_bands(asset, p)

        band = asset.extra_fields["raster:bands"][0]
        assert band["nodata"] == 9
        assert band["statistics"]["valid_percent"] == 50.0

    def test_nan_nodata_serializes_as_string(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.assets import add_raster_bands

        p = tmp_path / "nan.tif"
        _write_cog(
            p,
            np.array([[0.0, 1.0], [2.0, 0.0]], dtype=np.float32),
            nodata=float("nan"),
        )
        _, asset = _item_with_asset(p, ["data"])

        add_raster_bands(asset, p)

        band = asset.extra_fields["raster:bands"][0]
        assert band["nodata"] == "nan"


class TestAddMaskClassification:
    def test_semantic_3class_classes(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.assets import add_mask_classification, add_raster_bands

        p = tmp_path / "m.tif"
        _write_cog(p, np.array([[0, 1], [2, 0]], dtype=np.uint8))
        item, asset = _item_with_asset(p, ["labels"])
        add_raster_bands(asset, p)

        add_mask_classification(asset, "semantic_3class")

        classes = asset.extra_fields["raster:bands"][0]["classification:classes"]
        assert [(c["value"], c["name"]) for c in classes] == [
            (0, "background"),
            (1, "field"),
            (2, "boundary"),
        ]
        assert (
            "https://stac-extensions.github.io/classification/v2.0.0/schema.json"
            in item.stac_extensions
        )

    def test_presence_only_background_value(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.assets import add_mask_classification, add_raster_bands

        p = tmp_path / "m.tif"
        _write_cog(p, np.array([[3, 1], [1, 3]], dtype=np.uint8))
        _, asset = _item_with_asset(p, ["labels"])
        add_raster_bands(asset, p)

        add_mask_classification(asset, "semantic_2class", background_value=3)

        classes = asset.extra_fields["raster:bands"][0]["classification:classes"]
        assert [(c["value"], c["name"]) for c in classes] == [(3, "background"), (1, "field")]

    def test_instance_gets_description_not_classes(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.assets import add_mask_classification, add_raster_bands

        p = tmp_path / "i.tif"
        _write_cog(p, np.array([[0, 5], [7, 0]], dtype=np.uint32))
        _, asset = _item_with_asset(p, ["labels"])
        add_raster_bands(asset, p)

        add_mask_classification(asset, "instance")

        band = asset.extra_fields["raster:bands"][0]
        assert "classification:classes" not in band
        assert "instance" in band["description"].lower()


class TestDecodeMaskClassification:
    def test_decode_boundary_classes_ignore_background_value(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.assets import add_mask_classification, add_raster_bands

        p = tmp_path / "b.tif"
        _write_cog(p, np.array([[0, 1], [1, 0]], dtype=np.uint8))
        _, asset = _item_with_asset(p, ["labels"])
        add_raster_bands(asset, p)

        add_mask_classification(asset, "decode_boundary", background_value=3)

        classes = asset.extra_fields["raster:bands"][0]["classification:classes"]
        assert [(c["value"], c["name"]) for c in classes] == [(0, "background"), (1, "boundary")]

    def test_decode_distance_gets_description_not_classes(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.assets import add_mask_classification, add_raster_bands

        p = tmp_path / "d.tif"
        _write_cog(p, np.array([[0.0, 0.5], [1.0, 0.0]], dtype=np.float32))
        _, asset = _item_with_asset(p, ["labels"])
        add_raster_bands(asset, p)

        add_mask_classification(asset, "decode_distance")

        band = asset.extra_fields["raster:bands"][0]
        assert "classification:classes" not in band
        assert "decode_distance_max_px" in band["description"]
        assert band["data_type"] == "float32"

    def test_unknown_kind_raises(self, tmp_path: Path) -> None:
        import pytest

        from ftw_dataset_tools.api.assets import add_mask_classification, add_raster_bands

        p = tmp_path / "u.tif"
        _write_cog(p, np.array([[0, 1], [1, 0]], dtype=np.uint8))
        _, asset = _item_with_asset(p, ["labels"])
        add_raster_bands(asset, p)

        with pytest.raises(ValueError, match="Unknown mask kind"):
            add_mask_classification(asset, "bogus")


class TestMaskKindRegistry:
    """Drift guard: every STAC mask asset kind must be classifiable or described."""

    def test_every_stac_mask_asset_kind_is_handled(self) -> None:
        from ftw_dataset_tools.api.assets import MASK_CLASSES, MASK_DESCRIPTIONS
        from ftw_dataset_tools.api.stac import _MASK_TYPE_BY_ASSET_NAME

        assert set(_MASK_TYPE_BY_ASSET_NAME) == set(MASK_CLASSES) | set(MASK_DESCRIPTIONS)
        assert not set(MASK_CLASSES) & set(MASK_DESCRIPTIONS)
