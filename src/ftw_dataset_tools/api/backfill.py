"""Bring a published collection up to date with chip and field steps added after its build.

The admin names step and ``ftwd add-field-embeddings`` write new columns and files. A
collection built before them also needs those on its STAC documents. Rerunning the
``stac`` stage would need every mask and image on disk, so these functions edit the
documents instead, and leave everything else in them as it was:

- ``chip_item_properties`` and ``update_item``: the ``ftw:admin*`` properties of each
  chip item, from the chips file;
- ``update_collection``: the chips asset's size and columns, the GAUL link, the field
  embeddings asset with its links, and a record of the backfill in ``ftw:config``;
- ``rewrite_docs``: the styles, README.md and AGENTS.md, measured from the updated files.

Each function gives the same result when run again, so a backfill can be repeated.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pystac

from ftw_dataset_tools import __version__
from ftw_dataset_tools.api import admin_names, docs, field_embeddings, stac, styles
from ftw_dataset_tools.api.assets import add_file_info, add_table_columns
from ftw_dataset_tools.api.source import installed_git_commit

if TYPE_CHECKING:
    from collections.abc import Iterable

#: The ``ftw:config`` key a backfill records itself under, one entry per run.
BACKFILL_KEY = "backfills"

#: Item properties a backfill owns: removed, then set again from the chips file.
ADMIN_PROPERTIES: tuple[str, ...] = tuple(f"ftw:{col}" for col in admin_names.OUTPUT_COLUMNS)

#: Link rels the backfill inserts its links ahead of, so they sit with the other
#: collection-level links instead of after hundreds of child links.
_TRAILING_RELS = frozenset({"child", "item"})


class BackfillError(ValueError):
    """A published collection lacks something the backfill needs."""

    def __init__(self, what: str, message: str) -> None:
        super().__init__(message)
        self.what = what


def chip_item_properties(
    chips_file: Path | str, *, year: int | None, keys: Iterable[str] = ADMIN_PROPERTIES
) -> dict[str, dict[str, Any]]:
    """Item id to the chip properties among ``keys``, as the stac stage would publish them.

    Empty values are left out, the same way the stac stage omits them.
    """
    wanted = set(keys)
    return {
        chip.item_id: {k: v for k, v in chip.properties.items() if k in wanted}
        for chip in stac._extract_chips_info(Path(chips_file), year=year)
    }


def update_item(
    item: dict, properties: dict[str, Any], keys: Iterable[str] = ADMIN_PROPERTIES
) -> bool:
    """Replace ``keys`` in an item's properties with ``properties``; True if it changed."""
    props = item.setdefault("properties", {})
    keys = tuple(keys)
    after = _pick(properties, keys)
    if _pick(props, keys) == after:
        return False
    for key in keys:
        props.pop(key, None)
    props.update(after)
    return True


def _pick(mapping: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {k: mapping[k] for k in keys if k in mapping}


def _holder(assets: dict[str, dict] | None = None) -> pystac.Collection:
    """A throwaway collection to run the stac stage's asset and link helpers on."""
    return pystac.Collection.from_dict(
        {
            "type": "Collection",
            "stac_version": pystac.get_stac_version(),
            "id": "backfill",
            "description": "backfill",
            "license": "other",
            "extent": {
                "spatial": {"bbox": [[-180, -90, 180, 90]]},
                "temporal": {"interval": [[None, None]]},
            },
            "links": [],
            "assets": assets or {},
        }
    )


def refresh_parquet_asset(doc: dict, key: str, path: Path | str) -> bool:
    """An asset's size, checksum (when it had one) and columns, from its file, as built.

    Returns whether the asset carries a checksum, so new assets can match it.
    """
    current = doc.get("assets", {}).get(key)
    if current is None:
        raise BackfillError(key, f"the collection has no {key} asset")
    checksum = "file:checksum" in current
    holder = _holder({key: current})
    asset = holder.assets[key]
    add_file_info(asset, Path(path), checksum=checksum)
    add_table_columns(asset, Path(path))
    doc["assets"][key] = asset.to_dict()
    _add_extensions(doc, holder.stac_extensions)
    return checksum


def _add_extensions(doc: dict, uris: Iterable[str]) -> None:
    extensions = doc.setdefault("stac_extensions", [])
    extensions.extend(uri for uri in uris if uri not in extensions)


def _insert_links(doc: dict, new: list[dict]) -> None:
    """Add links ahead of the first child or item link."""
    links = doc.setdefault("links", [])
    at = next((i for i, link in enumerate(links) if link.get("rel") in _TRAILING_RELS), len(links))
    links[at:at] = new


def _is_backfill_link(link: dict) -> bool:
    """A link this module adds, and so removes before adding again."""
    href = link.get("href")
    if link.get("rel") == "derived_from":
        return href == admin_names.GAUL_URL or href in {
            p.url for p in field_embeddings.PRODUCTS.values()
        }
    return link.get(stac.ASSET_LICENSE_KEY) == stac.FIELD_EMBEDDINGS_ASSET


def _embedding_entries(path: Path, *, checksum: bool) -> tuple[dict, list[dict], list[str]]:
    """The field embeddings asset, its links and its extensions, as the stac stage makes them."""
    holder = _holder()
    stac._add_field_embeddings(holder, path, checksums=checksum)
    asset = holder.assets[stac.FIELD_EMBEDDINGS_ASSET].to_dict()
    links = [
        link.to_dict(transform_href=False)
        for link in holder.links
        if link.rel in ("derived_from", "license")
    ]
    return asset, links, list(holder.stac_extensions)


def _gaul_link() -> dict:
    return {
        "rel": "derived_from",
        "href": admin_names.GAUL_URL,
        "type": "text/html",
        "title": admin_names.ATTRIBUTION,
    }


def _record(doc: dict, *, steps: list[str]) -> None:
    """Note in ``ftw:config`` what the backfill added, and with which ftwd.

    The config itself stays as the build ran it. Each run replaces the record, so it
    never claims a step a later run took out; a run that adds nothing removes it.
    """
    record = doc.setdefault("ftw:config", {})
    if not steps:
        record.pop(BACKFILL_KEY, None)
        return
    record[BACKFILL_KEY] = [
        {
            "steps": steps,
            "ftwd_version": __version__,
            "ftwd_git_commit": installed_git_commit(),
            "generated_at": datetime.now(UTC).isoformat(),
        }
    ]


def update_collection(
    doc: dict,
    *,
    chips_file: Path | str,
    admin_names_added: bool,
    field_embeddings_file: Path | str | None,
) -> None:
    """Make a published collection describe the backfilled chips and field embeddings.

    Args:
        doc: The collection, edited in place.
        chips_file: The chips GeoParquet as it will be published.
        admin_names_added: Whether the chips carry the admin name columns now.
        field_embeddings_file: The field embeddings file to publish beside the
            collection, or None to publish none.
    """
    checksum = refresh_parquet_asset(doc, "chips", chips_file)
    doc["links"] = [link for link in doc.get("links", []) if not _is_backfill_link(link)]
    doc.get("assets", {}).pop(stac.FIELD_EMBEDDINGS_ASSET, None)

    new_links = [_gaul_link()] if admin_names_added else []
    if field_embeddings_file is not None:
        new_links += _add_embeddings(doc, Path(field_embeddings_file), checksum=checksum)
    _insert_links(doc, new_links)
    steps = [
        step
        for step, done in (
            ("admin_names", admin_names_added),
            ("field_embeddings", field_embeddings_file is not None),
        )
        if done
    ]
    _record(doc, steps=steps)


def _add_embeddings(doc: dict, path: Path, *, checksum: bool) -> list[dict]:
    """Add the field embeddings asset and its extensions; returns the links to add."""
    asset, links, extensions = _embedding_entries(path, checksum=checksum)
    doc.setdefault("assets", {})[stac.FIELD_EMBEDDINGS_ASSET] = asset
    _add_extensions(doc, extensions)
    return links


def field_polygons_file(doc: dict, output_dir: Path) -> Path:
    """The field polygons the docs measure: the filtered set when the build filtered classes."""
    assets = doc.get("assets", {})
    key = "fields_filtered" if "fields_filtered" in assets else "fields"
    if key not in assets:
        raise BackfillError("fields", "the collection has no fields asset")
    return output_dir / str(assets[key]["href"]).removeprefix("./")


def rewrite_docs(output_dir: Path | str, collection_json: Path | str) -> list[Path]:
    """Rewrite the styles, README.md and AGENTS.md from the files in ``output_dir``.

    Uses the tile archives the collection already names, so no tiles are rebuilt, and
    the build's own config from ``ftw:config``. Returns the documents written.
    """
    import json

    output_dir = Path(output_dir)
    collection_json = Path(collection_json)
    doc = json.loads(collection_json.read_text(encoding="utf-8"))
    chips = output_dir / str(doc["assets"]["chips"]["href"]).removeprefix("./")
    fields = field_polygons_file(doc, output_dir)
    style_results = styles.write_styles(
        output_dir, doc["id"], chips, fields, **_tile_refs(doc["assets"])
    )
    config_dict = (doc.get("ftw:config") or {}).get("config") or {}
    docs_cfg = (config_dict.get("stages") or {}).get("docs") or {}
    return docs.write_docs(
        output_dir,
        collection_json,
        chips,
        fields,
        style_results,
        config_dict,
        readme=docs_cfg.get("readme", True),
        agents=docs_cfg.get("agents", True),
    )


def _tile_refs(assets: dict) -> dict[str, str | None]:
    """The tile archives as the styles reference them, from one level below the collection."""
    refs: dict[str, str | None] = {}
    for key in ("chips_tiles", "fields_tiles"):
        asset = assets.get(key)
        refs[key] = "../" + str(asset["href"]).removeprefix("./") if asset else None
    return refs
