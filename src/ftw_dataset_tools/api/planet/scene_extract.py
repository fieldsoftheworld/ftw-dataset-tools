"""Scene-grouped range reads, with independent recovery of SR and UDM2."""

from pathlib import Path

import rasterio

from ftw_dataset_tools.api.planet.http import PlanetClient, PlanetError
from ftw_dataset_tools.api.planet.raster import (
    COG_OPTIONS,
    crop_window,
    output_matches,
    write_raster,
)
from ftw_dataset_tools.api.planet.scenes import ASSETS, asset_url
from ftw_dataset_tools.api.planet.state import digest


def signature(row: dict, item_id: str, asset: str) -> str:
    """Identify exactly which AOI/scene/asset a crop represents."""
    return digest({"geometry": row["geometry"], "item_id": item_id, "asset": asset, "version": 1})


def output_path(workdir: Path, identifier: str, asset: str) -> Path:
    suffix = "sr" if asset == ASSETS[0] else "udm2"
    return workdir / "images" / f"{identifier}_{suffix}.tif"


def crop_scene(
    source: rasterio.io.DatasetReader, row: dict, path: Path, item_id: str, asset: str
) -> None:
    expected_bands = 4 if asset == ASSETS[0] else 8
    if source.count != expected_bands:
        raise ValueError(f"Expected {expected_bands} bands for {asset}")
    window = crop_window(source, row["geometry"])
    data = source.read(window=window, boundless=True, fill_value=0)
    if asset == ASSETS[1]:
        height, width = data.shape[1:]
        import numpy as np

        rows, cols = np.ogrid[:height, :width]
        outside = (
            (rows + window.row_off < 0)
            | (rows + window.row_off >= source.height)
            | (cols + window.col_off < 0)
            | (cols + window.col_off >= source.width)
        )
        data[7][outside] |= 1
    profile = dict(source.profile, transform=source.window_transform(window))
    if asset == ASSETS[1]:
        profile["nodata"] = None
    write_raster(
        path,
        data,
        profile,
        signature(row, item_id, asset),
        tags={"planet_item_id": item_id, "planet_asset": asset},
    )


def extract_group(
    client: PlanetClient, task: dict, workdir: Path, activation_timeout: float
) -> dict:
    """Open each scene/asset once for all AOIs, retrying with a fresh URL once."""
    item_id = task["id"]
    failed = []
    for asset in ASSETS:
        todo = [
            row
            for row in task["members"]
            if not output_matches(
                output_path(workdir, row["id"], asset), signature(row, item_id, asset)
            )
        ]
        for attempt in range(2):
            if not todo:
                break
            remaining = []
            try:
                url = asset_url(
                    client, item_id, asset, timeout=activation_timeout, refresh=attempt > 0
                )
                with rasterio.Env(**COG_OPTIONS), rasterio.open(url) as source:
                    for row in todo:
                        try:
                            crop_scene(
                                source, row, output_path(workdir, row["id"], asset), item_id, asset
                            )
                        except (OSError, ValueError, rasterio.errors.RasterioError):
                            remaining.append(row)
                todo = remaining
            except (PlanetError, OSError, ValueError, rasterio.errors.RasterioError):
                pass
        failed.extend({"id": row["id"], "asset": asset} for row in todo)
    return {"id": item_id, "status": "failed" if failed else "ok", "pending": failed}
