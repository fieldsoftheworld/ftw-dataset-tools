"""README.md and AGENTS.md for a collection, written from its measured contents.

Nothing here is boilerplate: every count, percentile and crop share is computed
with DuckDB against the files that were just written, sections whose data is
absent are omitted rather than filled with placeholders, and each example query
in AGENTS.md is executed against the collection before it is printed, with its
first rows inlined underneath. A reader can therefore trust that a documented
query runs and returns what the document says it returns.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import duckdb

from ftw_dataset_tools.api.geo import ensure_spatial_loaded
from ftw_dataset_tools.api.imagery.catalog_ops import iter_chip_dirs
from ftw_dataset_tools.api.styles import split_counts, top_codes

if TYPE_CHECKING:
    from collections.abc import Callable

    from ftw_dataset_tools.api.styles import StyleResult

QUANTILES = (5, 25, 50, 75, 95)
TOP_CROP_LIMIT = 10
RESULT_ROWS = 3
SEASONS = ("planting", "harvest")

# Columns that would mark a row in items.parquet as a season child item rather
# than a chip. The catalogue written today mirrors chips only, so the child
# items are normally read from the item JSON files instead.
PARENT_COLUMNS = ("parent", "ftw:parent_id", "ftw:parent_chip", "parent_id")

_MISSING_COLUMN = "not found in FROM clause"

AGENTS_QUERIES: list[tuple[str, str]] = [
    (
        "Chips per split",
        "SELECT \"ftw:split\" AS split, count(*) AS chips FROM read_parquet('items.parquet') "
        "GROUP BY 1 ORDER BY 1",
    ),
    (
        "Chips with the highest field coverage",
        "SELECT id, field_coverage_pct FROM read_parquet('{chips}') "
        "ORDER BY field_coverage_pct DESC LIMIT 5",
    ),
    (
        "Dominant crops across chips",
        "SELECT hcat_dominant_name_en AS crop, count(*) AS chips FROM read_parquet('{chips}') "
        "WHERE hcat_dominant_code IS NOT NULL GROUP BY 1 ORDER BY 2 DESC LIMIT 10",
    ),
    (
        "Field polygons intersecting one chip",
        "SELECT count(*) AS fields FROM read_parquet('{fields}') f, "
        "(SELECT geometry FROM read_parquet('{chips}') LIMIT 1) c "
        "WHERE ST_Intersects(f.geometry, c.geometry)",
    ),
]

# Plain-language meanings for the columns and item properties FTW writes. Anything
# not listed is carried through from the source dataset and described as such.
CHIP_COLUMN_NOTES = {
    "id": "chip identifier; matches the STAC item id (with the year suffix stripped)",
    "geometry": "the chip footprint, a square polygon in EPSG:4326",
    "bbox": "bounding box of the chip footprint",
    "split": "which benchmark split the chip belongs to (train / val / test)",
    "field_coverage_pct": "percent of the chip's area covered by mapped field polygons",
    "field_count": "number of field polygons intersecting the chip",
    "hcat_dominant_code": "EuroCrops HCAT code of the crop covering the most of the chip's fields",
    "hcat_dominant_name_en": "English name for hcat_dominant_code",
    "hcat_dominant_pct": "share of the chip's field area under the dominant crop",
    "grid_id": "identifier of the FTW grid cell the chip was cut from",
    "year": "calendar year the field boundaries were declared for",
}

ITEM_PROPERTY_NOTES = {
    "ftw:split": "which benchmark split the chip belongs to (train / val / test)",
    "ftw:calendar_year": "calendar year of the crop cycle the chip documents",
    "ftw:planting_day": "day of year the planting window is centred on",
    "ftw:harvest_day": "day of year the harvest window is centred on",
    "ftw:planting_cloud_cover": "cloud cover of the selected planting scene, in percent",
    "ftw:harvest_cloud_cover": "cloud cover of the selected harvest scene, in percent",
    "ftw:stac_host": "STAC API the imagery was selected from",
    "ftw:season": "which crop-calendar window a child imagery item covers",
    "ftw:source": "satellite mission the imagery came from",
    "ftw:buffer_days": "half-width of the search window around the target day, in days",
    "ftw:field_coverage_pct": "percent of the chip's area covered by mapped field polygons",
}

ASSET_NOTES = {
    "fields": "field boundary polygons, one row per field (GeoParquet)",
    "chips": "one row per chip, with its split, field coverage and dominant crop (GeoParquet)",
    "items": "stac-geoparquet mirror of every chip item, for querying the whole collection",
    "chips_tiles": "vector tiles of the chips, for maps (PMTiles)",
    "fields_tiles": "vector tiles of the fields, for maps (PMTiles)",
    "boundary_lines": "field boundaries as lines rather than polygons (GeoParquet)",
}

STYLE_BLURBS = {
    "split": "chips coloured by their train / val / test assignment",
    "field-coverage": "chips shaded light to dark by the share of their area covered by fields",
    "dominant-crop": "chips coloured by the crop covering the most of their field area",
    "crops": "fields coloured by their harmonized crop (EuroCrops HCAT)",
    "outline": "every field in one colour, for reading the boundaries themselves",
}


# --------------------------------------------------------------------------- #
# DuckDB helpers
# --------------------------------------------------------------------------- #


def _sql_path(path: Path | str) -> str:
    """Escape a path for interpolation into a single-quoted SQL string literal."""
    return str(path).replace("'", "''")


def _connect(working_dir: Path | None = None) -> duckdb.DuckDBPyConnection:
    """A spatial connection whose relative paths resolve inside ``working_dir``."""
    con = duckdb.connect(":memory:")
    ensure_spatial_loaded(con)
    if working_dir is not None:
        con.execute(f"SET file_search_path='{_sql_path(working_dir)}'")
    return con


def _columns(con: duckdb.DuckDBPyConnection, path: Path) -> list[str]:
    rows = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{_sql_path(path)}')").fetchall()
    return [row[0] for row in rows]


def _count(con: duckdb.DuckDBPyConnection, path: Path) -> int:
    row = con.execute(f"SELECT COUNT(*) FROM read_parquet('{_sql_path(path)}')").fetchone()
    return int(row[0]) if row else 0


def run_query(sql: str, output_dir: Path | str) -> list[tuple]:
    """Run one SQL statement with the collection directory as the working directory."""
    con = _connect(Path(output_dir))
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()


# --------------------------------------------------------------------------- #
# Measurement
# --------------------------------------------------------------------------- #


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _coverage_quantiles(con: duckdb.DuckDBPyConnection, chips: Path) -> dict[int, float]:
    """Field coverage at the 5/25/50/75/95th percentiles, as raw percentages."""
    if "field_coverage_pct" not in _columns(con, chips):
        return {}
    fractions = ", ".join(str(q / 100) for q in QUANTILES)
    row = con.execute(
        f"SELECT quantile_cont(field_coverage_pct, [{fractions}]) "
        f"FROM read_parquet('{_sql_path(chips)}') WHERE field_coverage_pct IS NOT NULL"
    ).fetchone()
    if not row or row[0] is None:
        return {}
    return {q: float(v) for q, v in zip(QUANTILES, row[0], strict=False)}


def _top_crops(chips: Path, fields: Path) -> list[tuple[int, str | None, float]]:
    """The most prominent crops as ``(code, name, share)``, fields first then chips."""
    rows: list[tuple[int, str | None, float]] = []
    if fields.exists():
        rows = top_codes(fields, "hcat:code", "hcat:name_en", "area", TOP_CROP_LIMIT)
    if not rows and chips.exists():
        rows = top_codes(
            chips, "hcat_dominant_code", "hcat_dominant_name_en", "count", TOP_CROP_LIMIT
        )
    total = sum(weight for _, _, weight in rows) or 1.0
    return [(code, name, weight / total) for code, name, weight in rows]


def _observed_asset_keys(output_dir: Path) -> set[str]:
    """Asset keys carried by the chip items actually written under ``chips/``."""
    keys: set[str] = set()
    for chip_dir in iter_chip_dirs(output_dir):
        for item_path in sorted(chip_dir.glob("*.json")):
            keys.update(_read_json(item_path).get("assets") or {})
        if keys:
            break
    return keys


def _mask_types(collection: dict, output_dir: Path) -> list[str]:
    """Declared mask assets, narrowed to the ones the written items actually carry."""
    declared = [key for key in (collection.get("item_assets") or {}) if key.endswith("_mask")]
    observed = _observed_asset_keys(output_dir)
    if not observed:
        return declared
    return [key for key in declared if key in observed]


def _item_properties(output_dir: Path) -> list[str]:
    """The ``ftw:*`` item properties present, from items.parquet or a sample item."""
    items = output_dir / "items.parquet"
    if items.exists():
        con = _connect()
        try:
            return sorted(c for c in _columns(con, items) if c.startswith("ftw:"))
        finally:
            con.close()
    for chip_dir in iter_chip_dirs(output_dir):
        for item_path in sorted(chip_dir.glob("*.json")):
            props = _read_json(item_path).get("properties") or {}
            return sorted(key for key in props if key.startswith("ftw:"))
    return []


def _record(parent: Any, season: Any, when: Any, cloud: Any) -> dict:
    return {
        "parent": str(parent),
        "season": str(season),
        "datetime": str(when) if when else None,
        "cloud_cover": float(cloud) if cloud is not None else None,
    }


def _child_records_from_parquet(items_parquet: Path) -> list[dict]:
    """Season child items mirrored into items.parquet, when the mirror carries them."""
    if not items_parquet.exists():
        return []
    con = _connect()
    try:
        columns = _columns(con, items_parquet)
        parent = next((c for c in PARENT_COLUMNS if c in columns), None)
        if parent is None or "ftw:season" not in columns:
            return []
        when = "datetime" if "datetime" in columns else "CAST(NULL AS VARCHAR)"
        cloud = '"eo:cloud_cover"' if "eo:cloud_cover" in columns else "CAST(NULL AS DOUBLE)"
        rows = con.execute(
            f'SELECT "{parent}", "ftw:season", CAST({when} AS VARCHAR), {cloud} '
            f"FROM read_parquet('{_sql_path(items_parquet)}') WHERE \"ftw:season\" IS NOT NULL"
        ).fetchall()
    finally:
        con.close()
    return [_record(*row) for row in rows]


def _child_records_from_json(output_dir: Path) -> list[dict]:
    """Season child items read from the item JSON files under ``chips/<square>/<chip>/``."""
    records: list[dict] = []
    for chip_dir in iter_chip_dirs(output_dir):
        for item_path in sorted(chip_dir.glob("*_s2.json")):
            props = _read_json(item_path).get("properties") or {}
            season = props.get("ftw:season")
            if season:
                cloud = props.get("eo:cloud_cover")
                records.append(_record(chip_dir.name, season, props.get("datetime"), cloud))
    return records


def _season_summary(records: list[dict], season: str) -> dict | None:
    rows = [r for r in records if r["season"] == season]
    if not rows:
        return None
    dates = sorted(r["datetime"] for r in rows if r["datetime"])
    clouds = [r["cloud_cover"] for r in rows if r["cloud_cover"] is not None]
    summary: dict = {"min": dates[0] if dates else None, "max": dates[-1] if dates else None}
    if clouds:
        summary["cloud_cover_avg"] = sum(clouds) / len(clouds)
        summary["cloud_cover_max"] = max(clouds)
    return summary


def imagery_stats(output_dir: Path | str) -> dict | None:
    """Imagery coverage and acquisition windows, or ``None`` when no scenes were selected."""
    output_dir = Path(output_dir)
    records = _child_records_from_parquet(output_dir / "items.parquet")
    records = records or _child_records_from_json(output_dir)
    if not records:
        return None
    stats: dict = {"chips_with_imagery": len({r["parent"] for r in records})}
    for season in SEASONS:
        summary = _season_summary(records, season)
        if summary:
            stats[season] = summary
    return stats


def collect_stats(
    output_dir: Path | str, chips_parquet: Path | str, fields_parquet: Path | str
) -> dict:
    """Every number the documents quote, measured against the written collection."""
    output_dir = Path(output_dir)
    chips, fields = Path(chips_parquet), Path(fields_parquet)
    collection = _read_json(output_dir / "collection.json")

    con = _connect()
    try:
        chips_total = _count(con, chips) if chips.exists() else 0
        fields_total = _count(con, fields) if fields.exists() else 0
        quantiles = _coverage_quantiles(con, chips) if chips.exists() else {}
        chip_columns = _columns(con, chips) if chips.exists() else []
    finally:
        con.close()

    return {
        "chips_total": chips_total,
        "split_counts": split_counts(chips) if chips.exists() else {},
        "coverage_quantiles": quantiles,
        "fields_total": fields_total,
        "top_crops": _top_crops(chips, fields),
        "imagery": imagery_stats(output_dir),
        "mask_types": _mask_types(collection, output_dir),
        "chip_columns": chip_columns,
        "item_properties": _item_properties(output_dir),
    }


# --------------------------------------------------------------------------- #
# Query execution for AGENTS.md
# --------------------------------------------------------------------------- #


def _asset_href(collection: dict, key: str, default: str) -> str:
    asset = (collection.get("assets") or {}).get(key) or {}
    return str(asset.get("href") or default).removeprefix("./")


def run_agents_queries(
    output_dir: Path | str, collection: dict
) -> list[tuple[str, str, list[tuple]]]:
    """Execute every documented query, dropping the ones whose columns are absent."""
    chips = _asset_href(collection, "chips", "chips.parquet")
    fields = _asset_href(collection, "fields", "fields.parquet")
    executed: list[tuple[str, str, list[tuple]]] = []
    for title, template in AGENTS_QUERIES:
        sql = template.format(chips=chips, fields=fields)
        try:
            rows = run_query(sql, output_dir)
        except duckdb.BinderException as exc:
            if _MISSING_COLUMN in str(exc):
                continue
            raise
        executed.append((title, sql, rows))
    return executed


# --------------------------------------------------------------------------- #
# Markdown helpers
# --------------------------------------------------------------------------- #


def _link(text: str, url: str | None) -> str:
    """A markdown link, or plain text when there is nothing to link to."""
    return f"[{text}]({url})" if url else text


def _table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines += ["| " + " | ".join(cells) + " |" for cells in rows]
    return "\n".join(lines)


def _join(sections: list[str]) -> str:
    return "\n\n".join(s.strip() for s in sections if s and s.strip()) + "\n"


def _sentence_list(parts: list[str]) -> str:
    if len(parts) <= 1:
        return parts[0] if parts else ""
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def _ordinal(percentile: int) -> str:
    if percentile % 100 in (11, 12, 13):
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(percentile % 10, "th")


def _via_link(collection: dict) -> tuple[str, str] | None:
    """``(title, href)`` of the collection's ``via`` link, when it has one."""
    for link in collection.get("links") or []:
        if link.get("rel") == "via" and link.get("href"):
            return str(link.get("title") or "the source dataset"), str(link["href"])
    return None


# --------------------------------------------------------------------------- #
# README sections
# --------------------------------------------------------------------------- #


def _readme_title(collection: dict) -> str:
    title = collection.get("title") or collection.get("id") or "Collection"
    parts = [f"# {title}"]
    if collection.get("description"):
        parts.append(str(collection["description"]))
    return "\n\n".join(parts)


def _contents_paragraph(stats: dict) -> str:
    counts = [f"{stats['chips_total']:,} chips"]
    if stats["fields_total"]:
        counts.append(f"{stats['fields_total']:,} field polygons")
    return (
        f"This collection holds {_sentence_list(counts)}. Each chip is a square STAC item "
        "carrying the label masks rasterized from the field boundaries that fall inside it."
    )


def _split_block(stats: dict) -> str:
    counts = stats["split_counts"]
    if not counts:
        return ""
    table = _table(["split", "chips"], [[name, f"{count:,}"] for name, count in counts.items()])
    return "Chips per benchmark split:\n\n" + table


def _coverage_block(stats: dict) -> str:
    quantiles = stats["coverage_quantiles"]
    if not quantiles:
        return ""
    rows = [[f"{q}{_ordinal(q)}", f"{value:.1f}%"] for q, value in sorted(quantiles.items())]
    table = _table(["percentile", "field coverage"], rows)
    return (
        "How much of a chip is mapped field, as a distribution over the chips "
        "(`field_coverage_pct`):\n\n" + table
    )


def _mask_block(stats: dict) -> str:
    masks = stats["mask_types"]
    if not masks:
        return ""
    names = ", ".join(f"`{m}`" for m in masks)
    return f"Every chip carries these label rasters as STAC assets: {names}."


def _season_sentence(season: str, summary: dict) -> str:
    text = f"{season.capitalize()} imagery"
    if summary.get("min") and summary.get("max"):
        text += f" was acquired between {summary['min']} and {summary['max']}"
    if summary.get("cloud_cover_avg") is not None:
        text += (
            f", averaging {summary['cloud_cover_avg']:.1f}% cloud cover "
            f"(worst {summary['cloud_cover_max']:.1f}%)"
        )
    return text + "."


def _imagery_block(stats: dict) -> str:
    imagery = stats["imagery"]
    if not imagery:
        return ""
    lines = [
        f"{imagery['chips_with_imagery']:,} chips have Sentinel-2 scenes selected for them, "
        "one in the planting window and one in the harvest window of the crop calendar."
    ]
    lines += [_season_sentence(s, imagery[s]) for s in SEASONS if imagery.get(s)]
    return " ".join(lines)


def _readme_contents(stats: dict) -> str:
    blocks = [
        "## What is in this collection",
        _contents_paragraph(stats),
        _split_block(stats),
        _coverage_block(stats),
        _mask_block(stats),
        _imagery_block(stats),
    ]
    return "\n\n".join(b for b in blocks if b)


def _readme_crops(stats: dict) -> str:
    crops = stats["top_crops"]
    if not crops:
        return ""
    rows = [[name or f"HCAT {code}", str(code), f"{share:.1%}"] for code, name, share in crops]
    return (
        "## Crops\n\n"
        "Crops are harmonized to the EuroCrops HCAT taxonomy. The classes covering the most "
        "of this collection:\n\n" + _table(["crop", "HCAT code", "share"], rows)
    )


def _readme_styles(styles: list[StyleResult]) -> str:
    if not styles:
        return ""
    lines = ["## Styles", "", "Ready-made map styles ship with the collection:", ""]
    for style in styles:
        blurb = STYLE_BLURBS.get(style.style_id, "a map view of this collection")
        default = " (the default view)" if style.default else ""
        lines.append(f"- **{style.title}**{default}: {blurb}.")
    return "\n".join(lines)


def _provider_lines(collection: dict) -> list[str]:
    lines = []
    for provider in collection.get("providers") or []:
        name = str(provider.get("name") or "").strip()
        if not name:
            continue
        roles = ", ".join(provider.get("roles") or []) or "provider"
        lines.append(f"- {_link(name, provider.get('url'))}: {roles}")
    return lines


def _config_lines(collection: dict, config_dict: dict) -> list[str]:
    stages = config_dict.get("stages") or {}
    splits = stages.get("splits") or {}
    masks = stages.get("masks") or {}
    lines = []
    split_type = splits.get("split_type") or collection.get("ftw:split_type")
    if split_type:
        seed = splits.get("random_seed")
        suffix = f", random seed {seed}" if seed is not None else ""
        lines.append(f"- Splits assigned with the `{split_type}` strategy{suffix}")
    if masks.get("resolution"):
        lines.append(f"- Masks rasterized at {masks['resolution']:g} m per pixel")
    return lines


def _readme_provenance(collection: dict, config_dict: dict) -> str:
    lines = _provider_lines(collection)
    if collection.get("license"):
        lines.append(f"- License: `{collection['license']}`")
    via = _via_link(collection)
    if via:
        lines.append(f"- Derived from {_link(*via)}")
    lines += _config_lines(collection, config_dict)
    if not lines:
        return ""
    return "## Provenance\n\n" + "\n".join(lines)


def _readme_uses() -> str:
    return (
        "## Suggested uses\n\n"
        "- Training and evaluating field boundary delineation models on a fixed, "
        "reproducible split.\n"
        "- Comparing model performance across regions by filtering chips on their grid id.\n"
        "- Sampling field polygons for crop-type work, using the harmonized HCAT codes."
    )


def _readme_limitations() -> str:
    return (
        "## Limitations\n\n"
        "- Masks are derived from field boundaries declared for a given year; parcels that "
        "changed shape, split or merged after that declaration are not reflected.\n"
        "- Imagery windows follow a crop calendar rather than a fixed date, so acquisition "
        "dates differ between chips and cloud-free scenes are not guaranteed.\n"
        "- Chips on the border of the source dataset may be only partly covered by field "
        "boundaries, and empty area there means unmapped, not fieldless."
    )


def _readme_access(collection: dict) -> str:
    items = _asset_href(collection, "items", "items.parquet")
    return (
        "## Access\n\n"
        f"`{items}` mirrors every chip item, so the whole collection can be queried without "
        "walking the catalog:\n\n"
        "```sql\n"
        "INSTALL spatial; LOAD spatial;\n"
        f"SELECT * FROM read_parquet('{items}') LIMIT 5;\n"
        "```\n\n"
        f"See {_link('AGENTS.md', 'AGENTS.md')} for the schema, field notes and more queries."
    )


def render_readme(
    collection: dict, stats: dict, styles: list[StyleResult], config_dict: dict
) -> str:
    """The collection's README, with every section backed by a measured number."""
    return _join(
        [
            _readme_title(collection),
            _readme_contents(stats),
            _readme_crops(stats),
            _readme_styles(styles),
            _readme_provenance(collection, config_dict),
            _readme_uses(),
            _readme_limitations(),
            _readme_access(collection),
        ]
    )


# --------------------------------------------------------------------------- #
# AGENTS sections
# --------------------------------------------------------------------------- #


def _agents_overview(collection: dict, stats: dict) -> str:
    title = collection.get("title") or collection.get("id") or "Collection"
    lines = [f"# {title}", "", "## Overview", "", str(collection.get("description") or title)]
    counts = [f"{stats['chips_total']:,} chips"]
    if stats["fields_total"]:
        counts.append(f"{stats['fields_total']:,} field polygons")
    detail = f"It contains {_sentence_list(counts)}."
    if stats["split_counts"]:
        splits = ", ".join(f"{name} {count:,}" for name, count in stats["split_counts"].items())
        detail += f" Chips are pre-assigned to benchmark splits ({splits})."
    lines += ["", detail]
    if collection.get("license"):
        lines.append(f"\nLicensed `{collection['license']}`.")
    return "\n".join(lines)


def _agents_access(collection: dict) -> str:
    assets = collection.get("assets") or {}
    lines = ["## Accessing the data", ""]
    if assets:
        lines.append("The collection ships these files alongside `collection.json`:")
        lines.append("")
        rows = [
            [
                f"`{_asset_href(collection, key, key)}`",
                str(asset.get("title") or ASSET_NOTES.get(key, key)),
            ]
            for key, asset in assets.items()
        ]
        lines.append(_table(["file", "what it is"], rows))
        lines.append("")
    lines += [
        "Query them with DuckDB from inside the collection directory, so the relative paths "
        "below resolve:",
        "",
        "```sql",
        "INSTALL spatial; LOAD spatial;",
        f"SELECT * FROM read_parquet('{_asset_href(collection, 'items', 'items.parquet')}') "
        "LIMIT 5;",
        "```",
    ]
    return "\n".join(lines)


def _note_lines(names: list[str], notes: dict[str, str], fallback: str) -> list[str]:
    return [f"- `{name}`: {notes.get(name, fallback)}" for name in names]


def _agents_schema(stats: dict) -> str:
    lines = ["## Schema & field notes", ""]
    columns = stats.get("chip_columns") or []
    if columns:
        lines += ["Columns in the chips table:", ""]
        lines += _note_lines(columns, CHIP_COLUMN_NOTES, "carried through from the source dataset")
        lines.append("")
    properties = stats.get("item_properties") or []
    if properties:
        lines += ["FTW properties on each STAC item:", ""]
        lines += _note_lines(
            properties, ITEM_PROPERTY_NOTES, "set by the FTW pipeline; see the item JSON"
        )
        lines.append("")
    masks = stats.get("mask_types") or []
    if masks:
        names = ", ".join(f"`{m}`" for m in masks)
        lines.append(f"Label rasters available as item assets: {names}.")
    if len(lines) == 2:
        return ""
    return "\n".join(lines).strip()


def _agents_quality(stats: dict) -> str:
    lines = ["## Data quality & usage notes", ""]
    quantiles = stats.get("coverage_quantiles") or {}
    if quantiles:
        median = quantiles.get(50)
        low = quantiles.get(5)
        lines.append(
            f"- Field coverage is uneven: the median chip is {median:.1f}% mapped field while "
            f"the bottom 5% sit at or below {low:.1f}%. Filter on `field_coverage_pct` when "
            "sparse chips would skew an evaluation."
        )
    lines += [
        "- Masks are derived from boundaries declared for one year; later parcel changes are "
        "not reflected.",
        "- Empty area inside a chip means unmapped, not necessarily fieldless.",
        "- Chips on the dataset border may be only partly covered by the source boundaries.",
    ]
    if stats.get("imagery"):
        lines.append(
            "- Imagery is chosen against a crop calendar, so acquisition dates and cloud cover "
            "vary between chips; check `eo:cloud_cover` on the season child items."
        )
    if stats.get("split_counts"):
        lines.append(
            "- Respect the pre-assigned splits: they are spatially blocked, so resampling "
            "chips at random leaks information between train and test."
        )
    return "\n".join(lines)


def _cell(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, bytes | memoryview):
        return f"<{len(bytes(value))} bytes>"
    text = str(value)
    return text if len(text) <= 60 else text[:57] + "..."


def _query_block(title: str, sql: str, rows: list[tuple]) -> str:
    lines = [f"### {title}", "", "```sql", sql + ";"]
    if rows:
        lines += [f"-- result: {' | '.join(_cell(v) for v in row)}" for row in rows[:RESULT_ROWS]]
        if len(rows) > RESULT_ROWS:
            lines.append(f"-- result: ... {len(rows) - RESULT_ROWS} more rows")
    else:
        lines.append("-- result: (no rows)")
    lines.append("```")
    return "\n".join(lines)


def _agents_queries(queries: list[tuple[str, str, list[tuple]]]) -> str:
    intro = (
        "## Example queries\n\n"
        "Every query below was run against this collection when this file was written; the "
        "`-- result:` lines are its first rows. Run them from the collection directory."
    )
    return "\n\n".join([intro, *(_query_block(t, sql, rows) for t, sql, rows in queries)])


def _agents_related(collection: dict) -> str:
    via = _via_link(collection)
    if via:
        return (
            "## Related collections\n\n"
            f"The field boundaries here come from {_link(*via)}; consult it for the original "
            "attributes, licensing terms and update cadence."
        )
    return (
        "## Related collections\n\n"
        "This collection stands alone: no upstream or sibling collection is declared in its "
        "STAC links."
    )


def render_agents(
    collection: dict, stats: dict, queries: list[tuple[str, str, list[tuple]]]
) -> str:
    """The collection's agent guide, using the Portolan headings."""
    return _join(
        [
            _agents_overview(collection, stats),
            _agents_access(collection),
            _agents_schema(stats),
            _agents_quality(stats),
            _agents_queries(queries),
            _agents_related(collection),
        ]
    )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def write_docs(
    output_dir: Path | str,
    collection_json_path: Path | str,
    chips_parquet: Path | str,
    fields_parquet: Path | str,
    styles: list[StyleResult],
    config_dict: dict,
    *,
    readme: bool = True,
    agents: bool = True,
    on_progress: Callable[[str], None] | None = None,
) -> list[Path]:
    """Measure the collection, run the documented queries and write the documents."""
    output_dir = Path(output_dir)
    collection = _read_json(Path(collection_json_path))
    stats = collect_stats(output_dir, chips_parquet, fields_parquet)

    written: list[Path] = []
    if readme:
        path = output_dir / "README.md"
        path.write_text(render_readme(collection, stats, styles, config_dict), encoding="utf-8")
        written.append(path)
        if on_progress:
            on_progress("docs: README.md")
    if agents:
        queries = run_agents_queries(output_dir, collection)
        path = output_dir / "AGENTS.md"
        path.write_text(render_agents(collection, stats, queries), encoding="utf-8")
        written.append(path)
        if on_progress:
            on_progress("docs: AGENTS.md")
    return written


# --------------------------------------------------------------------------- #
# Registration on the saved collection
# --------------------------------------------------------------------------- #

TILE_TITLES = {
    "chips_tiles": "Chips (PMTiles)",
    "fields_tiles": "Field boundaries (PMTiles)",
}

DOC_LINKS = {
    "README.md": ("describedby", "Collection README"),
    "AGENTS.md": ("agents", "Collection agent guide"),
}

PMTILES_MEDIA_TYPE = "application/vnd.pmtiles"
STYLE_MEDIA_TYPE = "application/vnd.mapbox.style+json"


def _tile_asset(key: str, path: Path) -> dict:
    return {
        "href": f"./{path.name}",
        "type": PMTILES_MEDIA_TYPE,
        "title": TILE_TITLES.get(key, key),
        "roles": ["visual"],
        "file:size": path.stat().st_size,
    }


def _style_asset(style: StyleResult) -> dict:
    return {
        "href": f"./styles/{style.style_id}.json",
        "type": STYLE_MEDIA_TYPE,
        "title": style.title,
        "roles": ["style", "default"] if style.default else ["style"],
    }


def _doc_links(docs: list[Path]) -> list[dict]:
    links = []
    for doc in docs:
        entry = DOC_LINKS.get(Path(doc).name)
        if entry is None:
            continue
        rel, title = entry
        links.append(
            {"rel": rel, "href": f"./{Path(doc).name}", "type": "text/markdown", "title": title}
        )
    return links


def _merge_links(existing: list[dict], new: list[dict]) -> list[dict]:
    """Append links, replacing in place any that share a rel and href."""
    merged = list(existing)
    for link in new:
        key = (link["rel"], link["href"])
        for index, current in enumerate(merged):
            if (current.get("rel"), current.get("href")) == key:
                merged[index] = link
                break
        else:
            merged.append(link)
    return merged


def register_docs_assets(
    collection_json_path: Path | str,
    *,
    tiles: dict[str, Path],
    styles: list[StyleResult],
    docs: list[Path],
) -> None:
    """Add the tiles, styles and documents to an already-written ``collection.json``.

    The collection is edited as JSON rather than re-serialised through pystac, so
    everything the stac stage wrote (key order, extension fields) survives
    untouched. Re-running replaces assets with the same key and links with the
    same rel and href instead of duplicating them.
    """
    path = Path(collection_json_path)
    collection = json.loads(path.read_text(encoding="utf-8"))

    assets = collection.setdefault("assets", {})
    for key, tile_path in tiles.items():
        assets[key] = _tile_asset(key, Path(tile_path))
    for style in styles:
        assets[f"style-{style.style_id}"] = _style_asset(style)

    collection["links"] = _merge_links(collection.get("links", []), _doc_links(docs))
    path.write_text(json.dumps(collection, indent=2) + "\n", encoding="utf-8")
