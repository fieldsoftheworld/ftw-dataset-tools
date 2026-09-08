"""Render definitions for the label rasters on chip items and the collection.

A chip item's label COGs are single-band and low-valued, so a viewer that draws
them raw shows a near-black square. The
`render extension <https://github.com/stac-extensions/render>`_ tells a client how
to draw each one instead.

Colour is *not* set here for the categorical masks: their colours live in
``classification:classes[].color_hint`` (see :mod:`ftw_dataset_tools.api.assets`),
which is the primary rendering mechanism for them. The render entries for those
masks carry only the asset, a title and ``nodata`` so background pixels are drawn
transparent. Only the genuinely continuous rasters -- the normalized DECODE
distance map and the id-valued instance mask -- get a colour ramp
(``colormap_name``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pystac

RENDER_SCHEMA_URI = "https://stac-extensions.github.io/render/v2.0.0/schema.json"

__all__ = [
    "RENDER_SCHEMA_URI",
    "add_render_schema",
    "build_collection_renders",
    "build_item_renders",
]

#: Render key -> the item asset key it draws. The render key is the mask kind, so
#: it matches ``assets.MASK_CLASSES`` / ``assets.MASK_DESCRIPTIONS``.
_RENDER_ASSETS = {
    "semantic_2class": "semantic_2class_mask",
    "semantic_3class": "semantic_3class_mask",
    "decode_boundary": "decode_boundary_mask",
    "decode_distance": "decode_distance_mask",
    "instance": "instance_mask",
}

_RENDER_TITLES = {
    "semantic_2class": "Fields (binary mask)",
    "semantic_3class": "Fields and boundaries (3-class mask)",
    "decode_boundary": "DECODE field boundaries",
    "decode_distance": "DECODE distance to boundary",
    "instance": "Field instances",
}

#: Masks whose colours come from ``classification:classes[].color_hint``.
_CATEGORICAL = frozenset({"semantic_2class", "semantic_3class", "decode_boundary"})

#: Background pixel value, drawn transparent by every render.
_BACKGROUND = 0

#: Ramp for the continuous rasters; a built-in of the render extension.
_CONTINUOUS_COLORMAP = "viridis"


def _first_band(asset: pystac.Asset | None) -> dict:
    """Return the asset's first ``raster:bands`` entry, or an empty dict."""
    if asset is None:
        return {}
    bands = asset.extra_fields.get("raster:bands") or []
    return bands[0] if bands else {}


def _categorical_render(render_key: str, asset_key: str) -> dict:
    """A render that only hides the background; colours come from the class hints."""
    return {
        "title": _RENDER_TITLES[render_key],
        "assets": [asset_key],
        "nodata": _BACKGROUND,
    }


def _decode_distance_render(asset_key: str, band: dict) -> dict:
    """The normalized [0, 1] distance map, drawn as a continuous ramp."""
    nodata = band.get("nodata")
    return {
        "title": _RENDER_TITLES["decode_distance"],
        "assets": [asset_key],
        "rescale": [[0, 1]],
        "nodata": nodata if nodata is not None else _BACKGROUND,
        "colormap_name": _CONTINUOUS_COLORMAP,
    }


def _instance_rescale(band: dict) -> list[list[float]]:
    """The chip's own id range, or the default stretch when the band cannot supply one.

    Instance ids are global rather than per-chip, so a chip whose ids all sit in
    the millions needs its own minimum, not zero, as the low end of the ramp. The
    mask declares its background as nodata, so these statistics already exclude it.
    """
    statistics = band.get("statistics") or {}
    minimum = statistics.get("minimum")
    maximum = statistics.get("maximum")
    if minimum is None or maximum is None or maximum <= minimum:
        return [[0, 1]]
    return [[minimum, maximum]]


def _instance_render(asset_key: str, band: dict | None) -> dict:
    """Instance ids drawn as a ramp; ``band`` is None where no stretch is knowable."""
    render = {
        "title": _RENDER_TITLES["instance"],
        "assets": [asset_key],
    }
    if band is not None:
        render["rescale"] = _instance_rescale(band)
    render["nodata"] = _BACKGROUND
    render["colormap_name"] = _CONTINUOUS_COLORMAP
    return render


def _render_for(render_key: str, asset_key: str, asset: pystac.Asset | None) -> dict:
    """Build one render definition, reading band metadata from the asset when present.

    ``asset`` is None on the collection, where per-chip band statistics do not
    exist; the instance render then carries no ``rescale`` at all rather than a
    stretch that would be wrong for every chip.
    """
    if render_key in _CATEGORICAL:
        return _categorical_render(render_key, asset_key)
    if render_key == "decode_distance":
        return _decode_distance_render(asset_key, _first_band(asset))
    return _instance_render(asset_key, _first_band(asset) if asset is not None else None)


def build_item_renders(item: pystac.Item) -> dict:
    """Build the ``renders`` object for a chip item, one entry per label asset present.

    Args:
        item: Chip item whose label assets have already been added and decorated
            with ``raster:bands``.

    Returns:
        Render definitions keyed by mask kind; empty when the item carries no
        label assets.
    """
    return {
        render_key: _render_for(render_key, asset_key, item.assets[asset_key])
        for render_key, asset_key in _RENDER_ASSETS.items()
        if asset_key in item.assets
    }


def build_collection_renders() -> dict:
    """Build collection-level ``renders``, keyed by the asset name each one draws.

    Per-item band statistics are not available here, so the instance render omits
    ``rescale`` entirely; the item's own render carries the real stretch.
    """
    return {
        asset_key: _render_for(render_key, asset_key, None)
        for render_key, asset_key in _RENDER_ASSETS.items()
    }


def add_render_schema(obj: pystac.STACObject) -> None:
    """Declare the render extension on a STAC object exactly once."""
    if RENDER_SCHEMA_URI not in obj.stac_extensions:
        obj.stac_extensions.append(RENDER_SCHEMA_URI)
