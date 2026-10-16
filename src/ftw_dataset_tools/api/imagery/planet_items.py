"""STAC items for a chip's PlanetScope selection.

PlanetScope children sit beside any Sentinel-2 ones on the same chip: ids end in
``_planet``, the parent links them as ``ftw:<slot>_planet`` and keeps its Planet
bookkeeping under ``ftw:planet:``, so selecting one source never disturbs the other.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pystac

from ftw_dataset_tools.api.imagery import planet_api
from ftw_dataset_tools.api.imagery.catalog_ops import PLANET_PROPERTY_PREFIX, link_rel
from ftw_dataset_tools.api.imagery.slots import SCENE_SLOTS, child_item_id, slot_title
from ftw_dataset_tools.api.imagery.stac_child_items import _copy_hierarchical_links
from ftw_dataset_tools.api.stac import _add_portolan_schema
from ftw_dataset_tools.api.stac_items import write_item

if TYPE_CHECKING:
    from pathlib import Path

    from ftw_dataset_tools.api.imagery.scene_selection import SceneSelectionResult, SelectedScene

__all__ = ["PLANET_SOURCE", "create_planet_child_items"]

PLANET_SOURCE = "planetscope"
_P = PLANET_PROPERTY_PREFIX


def _relative(path: str, chip_dir: Path) -> str:
    return os.path.relpath(path, chip_dir)


def _child_item(
    chip_dir: Path, parent_item: pystac.Item, scene: SelectedScene, year: int
) -> pystac.Item:
    child_id = child_item_id(parent_item.id, scene.season, PLANET_SOURCE)
    child_path = chip_dir / f"{child_id}.json"
    props = scene.item.properties
    properties: dict = {
        "ftw:season": scene.season,
        "ftw:source": PLANET_SOURCE,
        "ftw:calendar_year": year,
        "ftw:scene_id": scene.id,
        "ftw:planet_item_type": planet_api.ITEM_TYPE,
        "ftw:planet_asset": planet_api.SCENE_ASSET,
        "ftw:planet_order_id": props.get("ftw:planet_order_id"),
        "ftw:cloud_mask": "udm2",
        "eo:cloud_cover": round(scene.cloud_cover, 2),
        "constellation": "planetscope",
    }
    if props.get("pl:clear_percent") is not None:
        properties["ftw:scene_clear_percent"] = props["pl:clear_percent"]
    if props.get("pl:satellite_id"):
        properties["platform"] = props["pl:satellite_id"]
    if props.get("pl:gsd") is not None:
        properties["gsd"] = props["pl:gsd"]

    child = pystac.Item(
        id=child_id,
        geometry=parent_item.geometry,
        bbox=parent_item.bbox,
        datetime=scene.datetime,
        properties=properties,
    )
    child.set_self_href(str(child_path))
    analytic = scene.item.assets["analytic"]
    child.add_asset(
        "analytic",
        pystac.Asset(
            href=_relative(analytic.href, chip_dir),
            media_type=pystac.MediaType.GEOTIFF,
            roles=["data"],
            title="PlanetScope 4-band surface reflectance (blue, green, red, NIR), clipped",
        ),
    )
    child.add_asset(
        "udm2",
        pystac.Asset(
            href=_relative(scene.item.assets["udm2"].href, chip_dir),
            media_type=pystac.MediaType.GEOTIFF,
            roles=["cloud"],
            title="PlanetScope usable data mask (UDM2), clipped",
        ),
    )
    child.add_link(
        pystac.Link(
            rel="ftw:parent_chip", target=f"./{parent_item.id}.json", media_type="application/json"
        )
    )
    if scene.stac_url:
        child.add_link(pystac.Link(rel="via", target=scene.stac_url, media_type="application/json"))
    _copy_hierarchical_links(parent_item, child)
    _add_portolan_schema(child)
    write_item(child, child_path)
    return child


def _drop_previous_planet_selection(parent_item: pystac.Item) -> None:
    rels = {link_rel(slot, PLANET_SOURCE) for slot in SCENE_SLOTS}
    parent_item.links = [link for link in parent_item.links if link.rel not in rels]
    for key in [k for k in parent_item.properties if k.startswith(_P)]:
        del parent_item.properties[key]


def create_planet_child_items(
    chip_dir: Path,
    parent_item: pystac.Item,
    result: SceneSelectionResult,
    year: int,
) -> None:
    """Write the chip's planting and harvest PlanetScope items and link them from the parent.

    Only the parent's Planet links and ``ftw:planet:`` properties change; the chip's
    temporal extent and any Sentinel-2 selection stay as they are.
    """
    parent_path = chip_dir / f"{parent_item.id}.json"
    if parent_item.get_self_href() is None:
        parent_item.set_self_href(str(parent_path))

    _drop_previous_planet_selection(parent_item)
    params = result.selection_params or {}
    parent_item.properties.update(
        {
            f"{_P}imagery_mode": "scenes",
            f"{_P}calendar_year": year,
            f"{_P}planting_day": result.crop_calendar.planting_day,
            f"{_P}harvest_day": result.crop_calendar.harvest_day,
            f"{_P}cloud_cover_chip_threshold": params.get("cloud_cover_chip_threshold"),
            f"{_P}buffer_days": params.get("buffer_days"),
            f"{_P}planting_buffer_used": result.planting_buffer_used,
            f"{_P}harvest_buffer_used": result.harvest_buffer_used,
            f"{_P}expansions_performed": result.expansions_performed,
        }
    )

    scenes = {"planting": result.planting_scene, "harvest": result.harvest_scene}
    try:
        for season in SCENE_SLOTS:
            scene = scenes[season]
            if scene is None:
                continue
            child = _child_item(chip_dir, parent_item, scene, year)
            parent_item.properties[f"{_P}{season}_cloud_cover"] = round(scene.cloud_cover, 2)
            parent_item.add_link(
                pystac.Link(
                    rel=link_rel(season, PLANET_SOURCE),
                    target=f"./{child.id}.json",
                    media_type="application/json",
                    title=f"{slot_title(season)} PlanetScope imagery",
                )
            )
    finally:
        write_item(parent_item, parent_path)
