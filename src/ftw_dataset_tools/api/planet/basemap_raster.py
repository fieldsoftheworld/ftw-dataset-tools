"""Assemble explicit mosaic/band order on a common native pixel grid."""

import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import bounds as window_bounds
from rasterio.windows import from_bounds

from ftw_dataset_tools.api.planet.basemaps import quad_path
from ftw_dataset_tools.api.planet.raster import (
    aoi_mask,
    crop_window,
    output_matches,
    write_raster,
)
from ftw_dataset_tools.api.planet.state import digest


def patch_grid(path: Path, geometry: dict) -> dict:
    with rasterio.open(path) as source:
        window = crop_window(source, geometry)
        return {
            "crs": source.crs,
            "transform": source.window_transform(window),
            "height": int(window.height),
            "width": int(window.width),
            "dtype": source.dtypes[0],
        }


def read_quad(
    source: rasterio.io.DatasetReader, grid: dict, bands: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Read a native window, rejecting grids that would require implicit resampling."""
    transform = grid["transform"]
    if source.crs != grid["crs"] or not np.allclose(
        (source.transform.a, source.transform.b, source.transform.d, source.transform.e),
        (transform.a, transform.b, transform.d, transform.e),
        rtol=0,
        atol=1e-8,
    ):
        raise ValueError(
            "Mosaics must share a CRS and pixel resolution; run differing products separately"
        )
    if any(band < 1 or band > source.count for band in bands):
        raise ValueError("Requested band is unavailable in a quad")
    if any(source.dtypes[band - 1] != grid["dtype"] for band in bands):
        raise ValueError("Mosaics must share a data type")
    bounds = window_bounds(rasterio.windows.Window(0, 0, grid["width"], grid["height"]), transform)
    window = from_bounds(*bounds, transform=source.transform)
    if not np.allclose(
        (window.col_off, window.row_off),
        (round(window.col_off), round(window.row_off)),
        rtol=0,
        atol=1e-5,
    ):
        raise ValueError("Mosaics have different pixel alignment")
    window = window.round_offsets().round_lengths()
    data = source.read(indexes=list(bands), window=window, boundless=True, fill_value=0)
    valid = source.dataset_mask(window=window, boundless=True) > 0
    return data, valid


def crop_mosaic(
    paths: list[Path], grid: dict, bands: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Merge AOI windows only, respecting alpha/nodata while preserving valid black pixels."""
    result = np.zeros((len(bands), grid["height"], grid["width"]), dtype=grid["dtype"])
    filled = np.zeros((grid["height"], grid["width"]), dtype=bool)
    for path in paths:
        with rasterio.open(path) as source:
            data, valid = read_quad(source, grid, bands)
        use = valid & ~filled
        result[:, use] = data[:, use]
        filled |= valid
    return result, filled


def assemble_patch(workdir: Path, task: dict, bands: tuple[int, ...]) -> dict:
    """Write one ordered seasonal stack. Missing periods/quads fail, never zero-fill silently."""
    plans = task["plans"]
    if any(row.get("status") != "ok" or not row.get("quad_ids") for row in plans):
        raise ValueError("A requested mosaic has no completed quad plan")
    signature = digest({"plans": plans, "bands": bands, "version": 1})
    destination = workdir / "images" / f"{task['id']}.tif"
    if output_matches(destination, signature):
        with rasterio.open(destination) as source:
            tags = source.tags()
        return {
            "id": task["id"],
            "status": "ok",
            "valid_fractions": json.loads(tags["planet_valid_fractions"]),
            "joint_valid_fraction": float(tags["planet_joint_valid_fraction"]),
        }
    paths_by_mosaic = [
        [quad_path(workdir, row["mosaic_id"], qid) for qid in row["quad_ids"]] for row in plans
    ]
    grid = patch_grid(paths_by_mosaic[0][0], task["geometry"])
    if grid["height"] * grid["width"] * len(bands) * len(plans) > 100_000_000:
        raise ValueError("Stack exceeds 100 million samples; split the AOI or mosaic list")
    inside = aoi_mask(
        task["geometry"], grid["crs"], grid["transform"], grid["height"], grid["width"]
    )
    if not inside.any():
        raise ValueError("AOI contains no pixel centers")
    arrays, coverage, descriptions = [], {}, []
    valid = inside.copy()
    for row, paths in zip(plans, paths_by_mosaic, strict=True):
        data, mask = crop_mosaic(paths, grid, bands)
        arrays.append(data)
        valid &= mask
        coverage[row["mosaic_name"]] = float((mask & inside).sum() / inside.sum())
        descriptions.extend(f"{row['mosaic_name']}:band_{band}" for band in bands)
    write_raster(
        destination,
        np.concatenate(arrays),
        dict(grid, nodata=None),
        signature,
        mask=valid,
        descriptions=tuple(descriptions),
        tags={
            "planet_mosaics": json.dumps([row["mosaic_name"] for row in plans]),
            "planet_mosaic_ids": json.dumps([row["mosaic_id"] for row in plans]),
            "planet_valid_fractions": json.dumps(coverage),
            "planet_joint_valid_fraction": str(float(valid.sum() / inside.sum())),
        },
    )
    return {
        "id": task["id"],
        "status": "ok",
        "valid_fractions": coverage,
        "joint_valid_fraction": float(valid.sum() / inside.sum()),
    }
