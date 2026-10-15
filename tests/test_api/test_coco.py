"""Tests for COCO instance export."""

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pystac
import pytest
import rasterio
from rasterio.transform import from_origin

from ftw_dataset_tools.api.assets import MaskReadError
from ftw_dataset_tools.api.coco import encode_rle, export_coco, mask_annotations


def decode_rle(rle: dict) -> np.ndarray:
    flat = np.zeros(sum(rle["counts"]), dtype=bool)
    pos = 0
    for i, count in enumerate(rle["counts"]):
        flat[pos : pos + count] = i % 2 == 1
        pos += count
    return flat.reshape(rle["size"], order="F")


def write_chip(dataset: Path, item_id: str, mask: np.ndarray | None, split: str | None) -> Path:
    chip_dir = dataset / "chips" / "32VJH" / item_id
    chip_dir.mkdir(parents=True)
    properties = {"ftw:split": split} if split else {}
    item = pystac.Item(item_id, None, None, datetime(2024, 1, 1, tzinfo=UTC), properties)
    if mask is not None:
        mask_path = chip_dir / f"{item_id}_instance.tif"
        with rasterio.open(
            mask_path,
            "w",
            driver="GTiff",
            height=mask.shape[0],
            width=mask.shape[1],
            count=1,
            dtype="uint32",
            crs="EPSG:32632",
            transform=from_origin(0, 0, 10, 10),
        ) as dst:
            dst.write(mask.astype("uint32"), 1)
        item.add_asset("instance_mask", pystac.Asset(href=f"./{mask_path.name}"))
    item_path = chip_dir / f"{item_id}.json"
    item_path.write_text(json.dumps(item.to_dict(include_self_link=False)))
    return item_path


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    root = tmp_path / "ds"
    root.mkdir()
    (root / "collection.json").write_text("{}")
    mask = np.zeros((6, 5), dtype=np.uint32)
    mask[0:2, 0:2] = 111205887
    mask[3:6, 2:5] = 7
    write_chip(root, "ftw-a", mask, "train")
    write_chip(root, "ftw-b", np.zeros((6, 5), dtype=np.uint32), "val")
    write_chip(root, "ftw-c", None, "train")
    return root


class TestEncodeRle:
    @pytest.mark.parametrize(
        "mask",
        [
            np.zeros((3, 4), dtype=bool),
            np.ones((3, 4), dtype=bool),
            np.array([[1, 0, 1], [0, 1, 1]], dtype=bool),
        ],
    )
    def test_round_trip(self, mask: np.ndarray) -> None:
        rle = encode_rle(mask)
        assert rle["size"] == list(mask.shape)
        np.testing.assert_array_equal(decode_rle(rle), mask)

    def test_starts_with_background_run(self) -> None:
        assert encode_rle(np.ones((2, 2), dtype=bool))["counts"] == [0, 4]


class TestMaskAnnotations:
    def test_bbox_area_and_mask(self) -> None:
        mask = np.zeros((4, 6), dtype=np.uint32)
        mask[1:3, 2:5] = 9
        (ann,) = mask_annotations(mask)
        assert ann["bbox"] == [2, 1, 3, 2]
        assert ann["area"] == 6
        assert ann["iscrowd"] == 0
        np.testing.assert_array_equal(decode_rle(ann["segmentation"]), mask == 9)

    def test_min_area_drops_small_instances(self) -> None:
        mask = np.zeros((4, 4), dtype=np.uint32)
        mask[0, 0] = 1
        mask[2:4, 2:4] = 2
        assert [a["area"] for a in mask_annotations(mask, min_area=2)] == [4]

    def test_background_only(self) -> None:
        assert mask_annotations(np.zeros((3, 3), dtype=np.uint32)) == []


class TestExportCoco:
    def test_writes_one_file_per_split(self, dataset: Path) -> None:
        result = export_coco(dataset)

        assert set(result.files) == {"train", "val"}
        assert result.skipped_chips == ["ftw-c"]
        train = json.loads((dataset / "coco" / "instances_train.json").read_text())
        assert train["categories"] == [{"id": 1, "name": "field", "supercategory": "field"}]
        (image,) = train["images"]
        assert image == {
            "id": 1,
            "file_name": "chips/32VJH/ftw-a/ftw-a.json",
            "height": 6,
            "width": 5,
        }
        assert [a["id"] for a in train["annotations"]] == [1, 2]
        assert {a["image_id"] for a in train["annotations"]} == {1}
        assert sorted(a["area"] for a in train["annotations"]) == [4, 9]

        val = json.loads(result.files["val"].read_text())
        assert len(val["images"]) == 1
        assert val["annotations"] == []

    def test_unsplit_chips_and_output_dir(self, tmp_path: Path) -> None:
        root = tmp_path / "ds"
        root.mkdir()
        (root / "collection.json").write_text("{}")
        write_chip(root, "ftw-a", np.ones((2, 2), dtype=np.uint32), None)

        result = export_coco(root, output_dir=tmp_path / "out")

        assert result.files == {"all": tmp_path / "out" / "instances_all.json"}
        assert result.annotations == {"all": 1}

    def test_missing_collection(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            export_coco(tmp_path)

    def test_unreadable_mask(self, dataset: Path) -> None:
        (dataset / "chips" / "32VJH" / "ftw-a" / "ftw-a_instance.tif").write_text("bad")
        with pytest.raises(MaskReadError):
            export_coco(dataset)
