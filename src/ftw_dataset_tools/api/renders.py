"""Render definitions for the rasters on chip items and the collection.

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

A chip that has season imagery also gets a true-colour render per season, and
:func:`build_render_order` names the stack a viewer should open the item on:
the fields drawn over that season's imagery. See that function for the
convention.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pystac

RENDER_SCHEMA_URI = "https://stac-extensions.github.io/render/v2.0.0/schema.json"

#: Item property naming the render keys to draw, bottom first. A browser-side
#: Portolan convention pending standardisation (portolan-spec issue #41); a reader
#: that does not know it keeps its single-asset default.
RENDER_ORDER_PROP = "portolan:render_order"

__all__ = [
    "RENDER_ORDER_PROP",
    "RENDER_SCHEMA_URI",
    "add_render_schema",
    "build_collection_renders",
    "build_item_renders",
    "build_render_order",
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

#: Default background pixel value; ``background_class_value`` overrides it per dataset.
_DEFAULT_BACKGROUND = 0

#: Mask kinds whose background pixel is the dataset's ``background_class_value``
#: (3 for presence-only labels). The DECODE layers always fold their background
#: into 0 -- see ``assets.MASK_CLASSES`` and ``api.decode`` -- so they are absent.
_DATASET_BACKGROUND_KINDS = frozenset({"semantic_2class", "semantic_3class", "instance"})

#: Ramp for the continuous rasters; a built-in of the render extension.
_CONTINUOUS_COLORMAP = "viridis"

#: Seasons a chip can carry imagery for, in the order the render stack prefers them
#: as its base layer.
_SEASONS = ("planting", "harvest")

#: Bands the true-colour render selects, in display order.
_RGB_BANDS = ("red", "green", "blue")

#: bidx for an image whose bands are not named at all; the download path writes the
#: requested bands in order, and the RGB ones are requested first. Never used as a
#: guess for a *named* stack -- see ``_rgb_bidx``.
_DEFAULT_RGB_BIDX = [1, 2, 3]

#: Half-width of the true-colour stretch, in standard deviations about the mean.
#: A raw min/max stretch lets one bright cloud edge crush the whole image; the
#: thumbnail of the same GeoTIFF clips at the 2-98 percentile for the same reason
#: (``api.imagery.thumbnails._normalize_for_display``).
_STRETCH_SIGMA = 2

#: The scene's ``visual`` asset is an 8-bit true-colour composite whose fill is 0,
#: so its stretch is fixed rather than read from per-chip statistics (a remote
#: scene asset carries no band metadata on the chip item).
_VISUAL_RESCALE = [[0, 255], [0, 255], [0, 255]]
_VISUAL_NODATA = 0

#: Overlay candidates for the default stack, best first. The DECODE boundary is an
#: outline, so the imagery stays visible inside each field; the binary mask fills
#: them and is only the fallback for a dataset built without DECODE layers.
_OVERLAY_PREFERENCE = ("decode_boundary", "semantic_2class")


def _first_band(asset: pystac.Asset | None) -> dict:
    """Return the asset's first ``raster:bands`` entry, or an empty dict."""
    if asset is None:
        return {}
    bands = asset.extra_fields.get("raster:bands") or []
    return bands[0] if bands else {}


def _background_for(render_key: str, background_value: int) -> int:
    """The pixel value this mask kind actually uses for background."""
    if render_key in _DATASET_BACKGROUND_KINDS:
        return background_value
    return _DEFAULT_BACKGROUND


def _categorical_render(render_key: str, asset_key: str, background: int) -> dict:
    """A render that only hides the background; colours come from the class hints."""
    return {
        "title": _RENDER_TITLES[render_key],
        "assets": [asset_key],
        "nodata": background,
    }


def _decode_distance_render(asset_key: str, band: dict, background: int) -> dict:
    """The normalized [0, 1] distance map, drawn as a continuous ramp."""
    nodata = band.get("nodata")
    return {
        "title": _RENDER_TITLES["decode_distance"],
        "assets": [asset_key],
        "rescale": [[0, 1]],
        "nodata": nodata if nodata is not None else background,
        "colormap_name": _CONTINUOUS_COLORMAP,
    }


def _instance_rescale(band: dict) -> list[list[float]]:
    """The chip's own id range, or the default stretch when the band cannot supply one.

    Instance ids are global rather than per-chip, so a chip whose ids all sit in
    the millions needs its own minimum, not zero, as the low end of the ramp. The
    background is excluded when the mask's statistics are computed, whether or not
    the band declares it as nodata: a presence-only mask leaves the band's nodata
    unset, because its background value can collide with a real instance id.
    """
    statistics = band.get("statistics") or {}
    minimum = statistics.get("minimum")
    maximum = statistics.get("maximum")
    if minimum is None or maximum is None or maximum <= minimum:
        return [[0, 1]]
    return [[minimum, maximum]]


def _instance_render(asset_key: str, band: dict | None, background: int) -> dict:
    """Instance ids drawn as a ramp; ``band`` is None where no stretch is knowable."""
    render = {
        "title": _RENDER_TITLES["instance"],
        "assets": [asset_key],
    }
    if band is not None:
        render["rescale"] = _instance_rescale(band)
    render["nodata"] = background
    render["colormap_name"] = _CONTINUOUS_COLORMAP
    return render


def _render_for(
    render_key: str,
    asset_key: str,
    asset: pystac.Asset | None,
    background_value: int,
) -> dict:
    """Build one render definition, reading band metadata from the asset when present.

    ``asset`` is None on the collection, where per-chip band statistics do not
    exist; the instance render then carries no ``rescale`` at all rather than a
    stretch that would be wrong for every chip.
    """
    background = _background_for(render_key, background_value)
    if render_key in _CATEGORICAL:
        return _categorical_render(render_key, asset_key, background)
    if render_key == "decode_distance":
        return _decode_distance_render(asset_key, _first_band(asset), background)
    band = _first_band(asset) if asset is not None else None
    return _instance_render(asset_key, band, background)


def _bands_of(asset: pystac.Asset) -> list[dict]:
    """Every ``raster:bands`` entry of an asset, or an empty list."""
    return asset.extra_fields.get("raster:bands") or []


def _rgb_bidx(bands: list[dict]) -> list[int] | None:
    """1-based band indices for red, green, blue, or None when there is no true colour.

    The clipped season COG records each band's name as its GDAL description, which
    ``assets.add_raster_bands`` copies into ``raster:bands``, so the RGB bands are
    looked up by name. The positional fallback is only for a file written without
    descriptions at all: ``ftwd download-images --bands`` takes any band list, and a
    named stack that has no red/green/blue (nir,red,green, say) must not be guessed
    at and published under a "true colour" title. The season falls back to its
    ``visual`` asset instead.
    """
    by_name = {
        str(band.get("description") or "").lower(): index
        for index, band in enumerate(bands, start=1)
    }
    named = [by_name.get(name) for name in _RGB_BANDS]
    if all(index is not None for index in named):
        return [index for index in named if index is not None]
    if any(band.get("description") for band in bands):
        return None
    return list(_DEFAULT_RGB_BIDX) if len(bands) >= len(_RGB_BANDS) else None


def _band_stretch(statistics: dict) -> list[float] | None:
    """One band's display range, or None when its statistics cannot supply one.

    ``mean +/- 2 sigma`` clamped to the band's own extremes, so a single bright or
    dark pixel does not crush the image. Falls back to the raw extremes when the
    band reports no mean/stddev, or when the clamped range collapses (a band whose
    values are nearly all identical).
    """
    minimum = statistics.get("minimum")
    maximum = statistics.get("maximum")
    if minimum is None or maximum is None or maximum <= minimum:
        return None

    mean = statistics.get("mean")
    stddev = statistics.get("stddev")
    if mean is None or stddev is None:
        return [minimum, maximum]

    lower = max(minimum, mean - _STRETCH_SIGMA * stddev)
    upper = min(maximum, mean + _STRETCH_SIGMA * stddev)
    if lower >= upper:
        return [minimum, maximum]
    return [lower, upper]


def _rgb_rescale(bands: list[dict], bidx: list[int]) -> list[list[float]] | None:
    """Per-band display range from the asset's own statistics, or None if any band lacks them."""
    ranges = []
    for index in bidx:
        stretch = _band_stretch(bands[index - 1].get("statistics") or {})
        if stretch is None:
            return None
        ranges.append(stretch)
    return ranges


def _rgb_nodata(bands: list[dict], bidx: list[int]) -> float | int | str | None:
    """The fill value the three colour bands agree on, or None when they do not.

    GeoTIFF nodata is per-dataset, so the bands of a stack ftwd wrote always agree;
    the render carries one value for all three, so it is only emitted when they do.
    """
    declared = {bands[index - 1].get("nodata") for index in bidx}
    if len(declared) != 1:
        return None
    value = declared.pop()
    if isinstance(value, float) and value.is_integer():
        # rasterio reports an integer fill as a float; every other render publishes
        # a plain 0, so publish one here too.
        return int(value)
    return value


def _season_title(season: str) -> str:
    return f"{season.capitalize()} season (true colour)"


def _local_image_render(season: str, asset: pystac.Asset) -> dict | None:
    """True colour over the chip's own clipped GeoTIFF, stretched to its band statistics.

    Returns None when the asset cannot supply three colour bands -- a stack whose
    named bands are not red/green/blue, one with fewer than three bands, or an asset
    added before the file was on disk, so ``raster:bands`` never got read. The caller
    then falls back to the scene asset.
    """
    bands = _bands_of(asset)
    bidx = _rgb_bidx(bands)
    if bidx is None:
        return None

    render = {
        "title": _season_title(season),
        "assets": [f"{season}_image"],
        "bidx": bidx,
    }
    rescale = _rgb_rescale(bands, bidx)
    if rescale is not None:
        render["rescale"] = rescale
    nodata = _rgb_nodata(bands, bidx)
    if nodata is not None:
        render["nodata"] = nodata
    return render


def _visual_render(season: str) -> dict:
    """True colour over the full scene's remote ``visual`` COG, at its fixed 8-bit stretch."""
    return {
        "title": _season_title(season),
        "assets": [f"{season}_visual"],
        "bidx": list(_DEFAULT_RGB_BIDX),
        "rescale": [list(band) for band in _VISUAL_RESCALE],
        "nodata": _VISUAL_NODATA,
    }


def _season_render(item: pystac.Item, season: str) -> dict | None:
    """The season's true-colour render, or None when the chip has no imagery for it.

    The chip's own clipped image is preferred: it covers exactly the chip and is
    stretched to what is actually in it. Most chips have a selection but no local
    download, so the scene's true-colour COG is the fallback -- a COG reader only
    fetches the tiles the chip covers.
    """
    image = item.assets.get(f"{season}_image")
    if image is not None:
        render = _local_image_render(season, image)
        if render is not None:
            return render
    if f"{season}_visual" in item.assets:
        return _visual_render(season)
    return None


def build_render_order(renders: dict) -> list[str]:
    """Name the renders a viewer should stack when it opens the item, bottom first.

    The default view of a chip is its fields drawn over that season's true-colour
    imagery: a season render at the bottom, the field overlay above it. A chip with
    no imagery of any kind gets no order at all, because a mask-only stack would
    show exactly what a viewer's single-asset default already shows.

    Args:
        renders: The item's own ``renders`` object; every entry returned is a key
            of it, as the convention requires.

    Returns:
        Render keys bottom first, or an empty list when the item has no imagery.
    """
    base = next((f"{season}_rgb" for season in _SEASONS if f"{season}_rgb" in renders), None)
    if base is None:
        return []
    overlay = next((key for key in _OVERLAY_PREFERENCE if key in renders), None)
    return [base, overlay] if overlay else [base]


def build_item_renders(item: pystac.Item, background_value: int = _DEFAULT_BACKGROUND) -> dict:
    """Build the ``renders`` object for a chip item: one entry per label asset, plus imagery.

    Belongs under ``item.properties``: the render extension's Feature branch
    requires ``properties.renders``, not a top-level key.

    Args:
        item: Chip item whose label and imagery assets have already been added and
            decorated with ``raster:bands``.
        background_value: Pixel value the class-valued masks use for background
            (3 for presence-only labels).

    Returns:
        Render definitions keyed by mask kind, plus ``<season>_rgb`` for each season
        the chip has imagery for; empty when the item carries neither.
    """
    renders = {
        render_key: _render_for(render_key, asset_key, item.assets[asset_key], background_value)
        for render_key, asset_key in _RENDER_ASSETS.items()
        if asset_key in item.assets
    }
    for season in _SEASONS:
        render = _season_render(item, season)
        if render is not None:
            renders[f"{season}_rgb"] = render
    return renders


def build_collection_renders(background_value: int = _DEFAULT_BACKGROUND) -> dict:
    """Build collection-level ``renders``, keyed by the asset name each one draws.

    Per-item band statistics are not available here, so the instance render omits
    ``rescale`` entirely; the item's own render carries the real stretch.

    Args:
        background_value: Pixel value the class-valued masks use for background.
    """
    return {
        asset_key: _render_for(render_key, asset_key, None, background_value)
        for render_key, asset_key in _RENDER_ASSETS.items()
    }


def add_render_schema(obj: pystac.STACObject) -> None:
    """Declare the render extension on a STAC object exactly once."""
    if RENDER_SCHEMA_URI not in obj.stac_extensions:
        obj.stac_extensions.append(RENDER_SCHEMA_URI)
