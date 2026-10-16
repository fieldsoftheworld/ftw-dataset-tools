"""Cut selected PlanetScope scenes onto each chip's own 3 m UTM grid.

Selection already downloaded the clipped scenes (see ``planet_prefetch``), so this
is local work: each ``{chip}_{slot}_planet`` child's 4-band surface reflectance file
is warped onto the chip's ``_planet`` mask grid - whatever UTM zone Planet delivered
it in - and written as ``{chip}_{slot}_image_planet.tif``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pystac
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

from ftw_dataset_tools.api.imagery.catalog_ops import image_asset_key, iter_chip_dirs
from ftw_dataset_tools.api.imagery.download_workflow import DownloadWorkflowResult
from ftw_dataset_tools.api.imagery.image_download import compute_target_grid, write_cog
from ftw_dataset_tools.api.imagery.parallel import ParallelOutcome, run_in_parallel
from ftw_dataset_tools.api.imagery.planet_items import PLANET_SOURCE
from ftw_dataset_tools.api.imagery.slots import image_filename, parse_child, slot_title
from ftw_dataset_tools.api.logging_config import get_logger
from ftw_dataset_tools.api.stac import MEDIA_TYPE_COG
from ftw_dataset_tools.api.stac_items import write_item

logger = get_logger(__name__)

__all__ = ["PLANET_BANDS", "download_planet_imagery_for_catalog"]

#: Bands of ortho_analytic_4b_sr, in file order.
PLANET_BANDS = ("blue", "green", "red", "nir")
#: Surface reflectance is stored as reflectance x 10000.
PLANET_SCALE = 0.0001
_NODATA = 0
_NO_MASKS = (
    "no PlanetScope masks for this chip; run the masks stage with source planetscope "
    "or create-masks --planet-grid first"
)


@dataclass(frozen=True)
class _Task:
    child: pystac.Item
    child_path: Path
    parent_path: Path
    slot: str
    output_path: Path


def _find_tasks(catalog_dir: Path) -> list[_Task]:
    tasks = []
    for chip_dir in iter_chip_dirs(catalog_dir):
        for child_path in sorted(chip_dir.glob("*_planet.json")):
            parsed = parse_child(child_path.stem)
            if parsed is None or parsed[2] != PLANET_SOURCE:
                continue
            chip_id, slot, _source = parsed
            tasks.append(
                _Task(
                    child=pystac.Item.from_file(str(child_path)),
                    child_path=child_path,
                    parent_path=chip_dir / f"{chip_id}.json",
                    slot=slot,
                    output_path=chip_dir / image_filename(chip_id, slot, PLANET_SOURCE),
                )
            )
    return tasks


def _cut(task: _Task) -> None:
    """Warp the child's delivered scene onto the chip's PlanetScope mask grid."""
    analytic = task.child.assets.get("analytic")
    if analytic is None:
        raise ValueError("child item has no 'analytic' asset")
    source_path = (task.child_path.parent / analytic.href).resolve()
    grid, error = compute_target_grid(
        tuple(task.child.bbox), task.output_path, resolution=3.0, reference_raster=None
    )
    if error is not None:
        raise ValueError(error)
    crs, transform, width, height, reference = grid
    if reference is None:
        raise ValueError(_NO_MASKS)

    with (
        rasterio.open(source_path) as src,
        WarpedVRT(
            src,
            crs=crs,
            transform=transform,
            width=width,
            height=height,
            resampling=Resampling.bilinear,
            src_nodata=_NODATA,
            nodata=_NODATA,
        ) as warped,
    ):
        stacked = warped.read(list(range(1, len(PLANET_BANDS) + 1))).astype(np.uint16)

    profile = {
        "driver": "COG",
        "dtype": "uint16",
        "width": width,
        "height": height,
        "count": len(PLANET_BANDS),
        "crs": crs,
        "transform": transform,
        "compress": "deflate",
        "nodata": _NODATA,
    }
    write_error = write_cog(
        task.output_path,
        stacked,
        list(PLANET_BANDS),
        profile,
        nodata=_NODATA,
        scale=PLANET_SCALE,
    )
    if write_error is not None:
        raise ValueError(write_error)


def _record(task: _Task) -> None:
    """Point the child and its parent chip at the written image (calling thread only)."""
    href = f"./{task.output_path.name}"
    asset = pystac.Asset(
        href=href,
        media_type=MEDIA_TYPE_COG,
        roles=["data"],
        title=f"{slot_title(task.slot)} PlanetScope surface reflectance (B, G, R, NIR), 3 m",
        extra_fields={"raster:scale": PLANET_SCALE},
    )
    task.child.add_asset("image", asset)
    write_item(task.child, task.child_path)

    parent = pystac.Item.from_file(str(task.parent_path))
    parent.add_asset(image_asset_key(task.slot, PLANET_SOURCE), asset.clone())
    write_item(parent, task.parent_path)


def download_planet_imagery_for_catalog(
    catalog_dir: Path,
    *,
    workers: int,
    resume: bool = True,
) -> DownloadWorkflowResult:
    """Write every selected PlanetScope scene onto its chip's grid.

    With ``resume``, chips whose image is already on disk are skipped.
    """
    result = DownloadWorkflowResult()
    tasks = []
    for task in _find_tasks(Path(catalog_dir)):
        if resume and task.output_path.exists():
            result.skipped += 1
            result.skipped_details.append({"item": task.child.id, "reason": "Already downloaded"})
        else:
            tasks.append(task)

    def apply(outcome: ParallelOutcome[_Task, None]) -> None:
        task = outcome.task
        if outcome.error is not None:
            result.failed += 1
            result.failed_details.append({"item": task.child.id, "error": str(outcome.error)})
            logger.warning(f"{task.child.id}: {outcome.error}")
            return
        _record(task)
        result.successful += 1
        logger.debug(f"Wrote {os.path.relpath(task.output_path, catalog_dir)}")

    run_in_parallel(tasks, work=_cut, apply=apply, workers=workers)
    return result
