"""Export instance masks as COCO instance segmentation annotations."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import rasterio
from rasterio.errors import RasterioIOError

from ftw_dataset_tools.api.assets import MaskReadError
from ftw_dataset_tools.api.imagery.catalog_ops import find_collection_dir
from ftw_dataset_tools.api.imagery.selection_workflow import find_chip_items

if TYPE_CHECKING:
    import pystac

FIELD_CATEGORY = {"id": 1, "name": "field", "supercategory": "field"}
UNSPLIT = "all"


@dataclass
class CocoExportResult:
    """Paths written and counts per split."""

    files: dict[str, Path] = field(default_factory=dict)
    images: dict[str, int] = field(default_factory=dict)
    annotations: dict[str, int] = field(default_factory=dict)
    skipped_chips: list[str] = field(default_factory=list)


def encode_rle(mask: np.ndarray) -> dict:
    """Uncompressed COCO RLE (column-major run lengths, starting with a 0-run)."""
    flat = mask.ravel(order="F").astype(np.uint8)
    edges = np.flatnonzero(np.diff(flat)) + 1
    counts = np.diff(np.concatenate(([0], edges, [flat.size]))).tolist()
    if flat.size and flat[0]:
        counts.insert(0, 0)
    return {"size": [int(mask.shape[0]), int(mask.shape[1])], "counts": counts}


def mask_annotations(instance: np.ndarray, min_area: int = 0) -> list[dict]:
    """One annotation (without ids) per non-zero instance value."""
    annotations = []
    for value in np.unique(instance):
        if value == 0:
            continue
        mask = instance == value
        area = int(mask.sum())
        if area < max(min_area, 1):
            continue
        rows = np.flatnonzero(mask.any(axis=1))
        cols = np.flatnonzero(mask.any(axis=0))
        annotations.append(
            {
                "category_id": FIELD_CATEGORY["id"],
                "segmentation": encode_rle(mask),
                "area": area,
                "bbox": [
                    int(cols[0]),
                    int(rows[0]),
                    int(cols[-1] - cols[0] + 1),
                    int(rows[-1] - rows[0] + 1),
                ],
                "iscrowd": 0,
            }
        )
    return annotations


def _read_instance_mask(path: Path) -> np.ndarray:
    try:
        with rasterio.open(path) as src:
            return src.read(1)
    except RasterioIOError as err:
        raise MaskReadError(path, f"Cannot read instance mask {path}: {err}") from err


def _instance_mask_path(item: pystac.Item, item_path: Path) -> Path | None:
    asset = item.assets.get("instance_mask")
    if asset is None:
        return None
    return (item_path.parent / asset.href).resolve()


def _empty_dataset() -> dict:
    return {"images": [], "annotations": [], "categories": [FIELD_CATEGORY]}


def _add_chip(
    dataset: dict, item_path: Path, mask_path: Path, collection_dir: Path, min_area: int
) -> None:
    instance = _read_instance_mask(mask_path)
    image_id = len(dataset["images"]) + 1
    dataset["images"].append(
        {
            "id": image_id,
            "file_name": item_path.resolve().relative_to(collection_dir).as_posix(),
            "height": int(instance.shape[0]),
            "width": int(instance.shape[1]),
        }
    )
    for ann in mask_annotations(instance, min_area):
        ann["id"] = len(dataset["annotations"]) + 1
        ann["image_id"] = image_id
        dataset["annotations"].append(ann)


def export_coco(
    dataset_dir: str | Path,
    output_dir: str | Path | None = None,
    min_area: int = 0,
) -> CocoExportResult:
    """Write ``instances_<split>.json`` for every split in a dataset's instance masks.

    Each chip becomes one COCO image whose ``file_name`` is its STAC item path,
    relative to the dataset directory. Chips without a split go to ``instances_all.json``.
    Instance masks are expected to use 0 as background.
    """
    collection_dir = find_collection_dir(Path(dataset_dir)).resolve()
    out = Path(output_dir) if output_dir else collection_dir / "coco"
    result = CocoExportResult()
    datasets: dict[str, dict] = {}

    for item, item_path in sorted(find_chip_items(collection_dir), key=lambda x: x[0].id):
        mask_path = _instance_mask_path(item, item_path)
        if mask_path is None or not mask_path.exists():
            result.skipped_chips.append(item.id)
            continue
        split = item.properties.get("ftw:split") or UNSPLIT
        dataset = datasets.setdefault(split, _empty_dataset())
        _add_chip(dataset, item_path, mask_path, collection_dir, min_area)

    out.mkdir(parents=True, exist_ok=True)
    for split, dataset in sorted(datasets.items()):
        path = out / f"instances_{split}.json"
        path.write_text(json.dumps(dataset))
        result.files[split] = path
        result.images[split] = len(dataset["images"])
        result.annotations[split] = len(dataset["annotations"])
    return result
