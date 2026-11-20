"""Bring a published collection up to date with chip and field steps added after its build.

The admin names step and ``ftwd add-field-embeddings`` write new columns and files. A
collection built before them also needs those on its STAC documents. Rerunning the
``stac`` stage would need every mask and image on disk, so these functions edit the
documents instead, and leave everything else in them as it was:

- ``chip_item_properties`` and ``update_item``: the ``ftw:admin*`` properties of each
  chip item, from the chips file;
- ``update_collection``: the chips asset's size and columns, the GAUL link, the field
  embeddings asset with its links, the field assets as their files now are (which
  may have gained crop names, see ``crop_names.add_crop_names``), and a record of the
  backfill in ``ftw:config``;
- ``rewrite_docs``: the styles, README.md and AGENTS.md, measured from the updated files,
  and the style assets listed on the collection.

Running a function again gives the same result, apart from the time stamp in the
backfill record, so a backfill can be repeated. The benchmark catalog's
``tools/backfill.py`` drives these functions for each published dataset.
"""

from __future__ import annotations

import copy
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pystac

from ftw_dataset_tools import __version__
from ftw_dataset_tools.api import admin_names, docs, field_embeddings, stac, styles
from ftw_dataset_tools.api.assets import CROP_NAME_NOTES, add_file_info, add_table_columns
from ftw_dataset_tools.api.source import installed_git_commit

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

#: The ``ftw:config`` key a backfill records itself under: the latest run's record.
BACKFILL_KEY = "backfills"

#: Item properties a backfill owns: removed, then set again from the chips file.
ADMIN_PROPERTIES: tuple[str, ...] = tuple(f"ftw:{col}" for col in admin_names.OUTPUT_COLUMNS)

#: Link rels the backfill inserts its links ahead of, so they sit with the other
#: collection-level links instead of after hundreds of child links.
_TRAILING_RELS = frozenset({"child", "item"})


class BackfillError(Exception):
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


def _record(doc: dict, *, steps: list[str], details: Mapping[str, Any] | None = None) -> None:
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
            **dict(details or {}),
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
    field_files: Mapping[str, Path | str] | None = None,
    crop_names: Mapping[str, str] | None = None,
) -> None:
    """Make a published collection describe the backfilled chips and field embeddings.

    Each run states every step the collection has had, since the record it writes
    replaces the last one.

    Args:
        doc: The collection, edited in place.
        chips_file: The chips GeoParquet as it will be published.
        admin_names_added: Whether the chips carry the admin name columns now.
        field_embeddings_file: The field embeddings file to publish beside the
            collection, or None to publish none.
        field_files: Asset key to field file (``fields``, ``fields_filtered``,
            ``boundary_lines``) as it will be published. Each asset's size, checksum
            and columns are read from its file, so they stay true whether or not the
            run changed the file.
        crop_names: Where this run's crop names came from (e.g. ``{"source": ...,
            "by": "code"}``), recorded with the ``crop_names`` step and in the name
            columns' descriptions; None when the run joined none. Needs
            ``field_files``.

    Raises:
        BackfillError: A file or asset is missing. ``doc`` is left as it was, as it
            is when reading a file fails part way.
    """
    work = copy.deepcopy(doc)
    _update_collection(
        work,
        chips_file=chips_file,
        admin_names_added=admin_names_added,
        field_embeddings_file=field_embeddings_file,
        field_files=field_files,
        crop_names=crop_names,
    )
    doc.clear()
    doc.update(work)


def _checked_field_files(
    doc: dict,
    field_files: Mapping[str, Path | str] | None,
    crop_names: Mapping[str, str] | None,
) -> dict[str, Path | str]:
    """The field files to refresh, each with its asset and on disk; raises otherwise."""
    field_files = dict(field_files or {})
    if crop_names is not None and not field_files:
        raise BackfillError("fields", "crop names were joined but no field file was given")
    for key, path in field_files.items():
        if key not in doc.get("assets", {}):
            raise BackfillError(key, f"the collection has no {key} asset")
        if not Path(path).is_file():
            raise BackfillError(key, f"no {key} file at {path}")
    return field_files


def _update_collection(
    doc: dict,
    *,
    chips_file: Path | str,
    admin_names_added: bool,
    field_embeddings_file: Path | str | None,
    field_files: Mapping[str, Path | str] | None,
    crop_names: Mapping[str, str] | None,
) -> None:
    if field_embeddings_file is not None and not Path(field_embeddings_file).is_file():
        raise BackfillError(
            stac.FIELD_EMBEDDINGS_ASSET, f"no field embeddings file at {field_embeddings_file}"
        )
    field_files = _checked_field_files(doc, field_files, crop_names)
    checksum = refresh_parquet_asset(doc, "chips", chips_file)
    _refresh_field_assets(doc, field_files, crop_names)
    doc["links"] = [link for link in doc.get("links", []) if not _is_backfill_link(link)]
    doc.get("assets", {}).pop(stac.FIELD_EMBEDDINGS_ASSET, None)

    new_links = [stac.gaul_link().to_dict()] if admin_names_added else []
    if field_embeddings_file is not None:
        new_links += _add_embeddings(doc, Path(field_embeddings_file), checksum=checksum)
    _insert_links(doc, new_links)
    steps = []
    if admin_names_added:
        steps.append("admin_names")
    if field_embeddings_file is not None:
        steps.append("field_embeddings")
    if crop_names is not None:
        steps.append("crop_names")
    _record(
        doc, steps=steps, details=None if crop_names is None else {"crop_names": dict(crop_names)}
    )


def _refresh_field_assets(
    doc: dict, field_files: Mapping[str, Path | str], crop_names: Mapping[str, str] | None
) -> None:
    for key, path in field_files.items():
        refresh_parquet_asset(doc, key, path)
        if crop_names is not None:
            _describe_crop_names(doc["assets"][key], crop_names)


def crop_name_notes(crop_names: Mapping[str, str]) -> dict[str, str]:
    """What each crop name column holds, given where a run's names came from."""
    source = crop_names.get("source")
    origin = f"the harmonized edition {source}" if source else "the harmonized edition"
    if crop_names.get("by") == "code":
        origin += ", matched by crop:code as that edition's field ids differ from this dataset's"
    return {name: f"{note}, from {origin}" for name, note in CROP_NAME_NOTES.items()}


def _describe_crop_names(asset: dict, crop_names: Mapping[str, str]) -> None:
    notes = crop_name_notes(crop_names)
    for column in asset.get("table:columns", []):
        if column.get("name") in notes:
            column["description"] = notes[column["name"]]


def _add_embeddings(doc: dict, path: Path, *, checksum: bool) -> list[dict]:
    """Add the field embeddings asset and its extensions; returns the links to add."""
    asset, links, extensions = _embedding_entries(path, checksum=checksum)
    doc.setdefault("assets", {})[stac.FIELD_EMBEDDINGS_ASSET] = asset
    _add_extensions(doc, extensions)
    return links


def asset_path(doc: dict, key: str, output_dir: Path) -> Path:
    """Where an asset's file is in ``output_dir``, from its relative href."""
    asset = doc.get("assets", {}).get(key)
    if asset is None:
        raise BackfillError(key, f"the collection has no {key} asset")
    return output_dir / str(asset["href"]).removeprefix("./")


def field_polygons_file(doc: dict, output_dir: Path) -> Path:
    """The field polygons the docs measure: the filtered set when the build filtered classes."""
    assets = doc.get("assets", {})
    key = "fields_filtered" if "fields_filtered" in assets else "fields"
    return asset_path(doc, key, output_dir)


def rewrite_docs(output_dir: Path | str, collection_json: Path | str) -> list[Path]:
    """Rewrite the styles, README.md and AGENTS.md from the files in ``output_dir``.

    Uses the tile archives the collection already names, so no tiles are rebuilt, and
    the build's own config from ``ftw:config``. As in the docs stage, styles are
    written only when there are tile archives, and the collection's style assets are
    made to match the styles written; its tile assets are left as they are. Returns
    the documents written.
    """
    output_dir = Path(output_dir)
    collection_json = Path(collection_json)
    doc = json.loads(collection_json.read_text(encoding="utf-8"))
    chips = asset_path(doc, "chips", output_dir)
    fields = field_polygons_file(doc, output_dir)
    style_results = _write_styles(doc, output_dir, chips, fields)
    if docs.sync_style_assets(doc, style_results):
        collection_json.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return _write_docs(doc, output_dir, collection_json, chips, fields, style_results)


def _write_styles(
    doc: dict, output_dir: Path, chips: Path, fields: Path
) -> list[styles.StyleResult]:
    tiles = _tile_refs(doc.get("assets", {}))
    if not any(tiles.values()):
        return []
    return styles.write_styles(output_dir, doc["id"], chips, fields, **tiles)


def _write_docs(
    doc: dict,
    output_dir: Path,
    collection_json: Path,
    chips: Path,
    fields: Path,
    style_results: list[styles.StyleResult],
) -> list[Path]:
    config_dict = (doc.get("ftw:config") or {}).get("config") or {}
    docs_cfg = (config_dict.get("stages") or {}).get("docs") or {}
    readme = docs_cfg.get("readme", True)
    agents = docs_cfg.get("agents", True)
    if not (readme or agents):
        return []
    return docs.write_docs(
        output_dir,
        collection_json,
        chips,
        fields,
        style_results,
        config_dict,
        readme=readme,
        agents=agents,
    )


def _tile_refs(assets: dict) -> dict[str, str | None]:
    """The tile archives as the styles reference them, from one level below the collection."""
    refs: dict[str, str | None] = {}
    for key in ("chips_tiles", "fields_tiles"):
        asset = assets.get(key)
        refs[key] = "../" + str(asset["href"]).removeprefix("./") if asset else None
    return refs
