"""MapLibre GL styles for a collection's PMTiles, built from measured distributions.

Every legend entry corresponds to values that actually exist in the data: the
splits, coverage quantiles and crop codes are measured with DuckDB before a
style is written, and a style is skipped entirely when its prerequisites are
absent. The browser derives a legend only from a ``fill`` layer whose
``fill-color`` is a top-level ``match`` or ``step`` expression, so the
legend-bearing styles are written in exactly that shape.

Colours follow the palette fiboa.org uses on its crop map (hcat_palette.json,
vendored from fieldsoftheworld/harmonized-field-data-catalog
tools/hcat_palette.json, originally fiboa/fiboa.github.io map/crop/codes2.js).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING

import duckdb

from ftw_dataset_tools.api.geo import detect_geometry_column, ensure_spatial_loaded
from ftw_dataset_tools.api.tiles import CHIPS_TILES, FIELDS_TILES

if TYPE_CHECKING:
    from collections.abc import Callable

SPLIT_ORDER = ("train", "val", "test")
SPLIT_COLORS = {"train": "#1b9e77", "val": "#d95f02", "test": "#7570b3"}
COVERAGE_RAMP = ["#f7fcf5", "#c7e9c0", "#74c476", "#238b45", "#00441b"]
COVERAGE_STOP_CANDIDATES = [1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]

HCAT_MAX_LEGEND = 12
OTHER_COLOR = "#c8c8c8"

# Outline colours, one per collection, so sibling datasets do not all read as
# the same pale blue in a card grid.
OUTLINE_PALETTE = [
    "#1b7837",
    "#2166ac",
    "#b2182b",
    "#762a83",
    "#e08214",
    "#35978f",
    "#8c510a",
    "#4d4d4d",
    "#c51b7d",
    "#01665e",
    "#5e3c99",
    "#d6604d",
]

# ColorBrewer Paired (12) — used where the HCAT palette is generic or repeats.
FALLBACK_PALETTE = [
    "#a6cee3",
    "#1f78b4",
    "#b2df8a",
    "#33a02c",
    "#fb9a99",
    "#e31a1c",
    "#fdbf6f",
    "#ff7f00",
    "#cab2d6",
    "#6a3d9a",
    "#ffff99",
    "#b15928",
]

GENERIC_HCAT_COLORS = {"#cc8c32"}  # the palette's "arable crops" default, reused by 41 groups
MIN_COLOR_DISTANCE = 48  # euclidean RGB distance below which two legend colours read as one


@dataclass(frozen=True)
class StyleResult:
    """One written style document and the legend the browser will show for it."""

    style_id: str
    path: Path
    title: str
    legend_rows: list[dict]
    default: bool


@lru_cache(maxsize=1)
def load_palette() -> dict[str, dict]:
    """The vendored fiboa.org HCAT palette, keyed by 10-digit code string."""
    raw = files("ftw_dataset_tools.api.data").joinpath("hcat_palette.json").read_text()
    return {entry["code"]: entry for entry in json.loads(raw)}


def _rgb(color: str) -> tuple[int, int, int]:
    c = color.lstrip("#")
    return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)


def _too_close(color: str, used: set[str]) -> bool:
    r, g, b = _rgb(color)
    return any(
        ((r - r2) ** 2 + (g - g2) ** 2 + (b - b2) ** 2) ** 0.5 < MIN_COLOR_DISTANCE
        for r2, g2, b2 in (_rgb(u) for u in used)
    )


def distinct_color(preferred: str | None, used: set[str]) -> str:
    """The palette colour when it is specific and distinct from the legend so far.

    Falls back to the next qualitative colour that reads as distinct, and finally
    to the neutral "other" grey.
    """
    if (
        preferred
        and preferred.lower() not in GENERIC_HCAT_COLORS
        and not _too_close(preferred, used)
    ):
        return preferred
    for color in FALLBACK_PALETTE:
        if not _too_close(color, used):
            return color
    return OTHER_COLOR


def _sql_path(path: Path | str) -> str:
    """Escape a path for interpolation into a single-quoted SQL string literal."""
    return str(path).replace("'", "''")


def _connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    ensure_spatial_loaded(con)
    return con


def _columns(con: duckdb.DuckDBPyConnection, path: Path) -> set[str]:
    rows = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{_sql_path(path)}')").fetchall()
    return {row[0] for row in rows}


def split_counts(chips: Path | str) -> dict[str, int]:
    """Chip counts per split, in train/val/test order. Empty when there is no split column."""
    chips = Path(chips)
    con = _connect()
    try:
        if "split" not in _columns(con, chips):
            return {}
        rows = con.execute(
            f"SELECT split, COUNT(*) FROM read_parquet('{_sql_path(chips)}') "
            "WHERE split IS NOT NULL GROUP BY split"
        ).fetchall()
    finally:
        con.close()
    counts = {str(name): int(count) for name, count in rows}
    ordered = {s: counts.pop(s) for s in SPLIT_ORDER if s in counts}
    ordered.update(sorted(counts.items()))
    return ordered


def _snap_stops(values: list[float]) -> list[float]:
    stops: list[float] = []
    for value in values:
        if value is None:
            continue
        nearest = min(COVERAGE_STOP_CANDIDATES, key=lambda c: abs(c - float(value)))
        if nearest <= 0 or nearest >= 100 or nearest in stops:
            continue
        if stops and nearest < stops[-1]:
            continue
        stops.append(nearest)
    return stops


def coverage_quantiles(chips: Path | str) -> list[float]:
    """Up to four ascending field-coverage stops at the 20/40/60/80th percentiles.

    Each percentile is snapped to the nearest "nice" value so the legend reads
    cleanly. Empty when the collection carries no ``field_coverage_pct``.
    """
    chips = Path(chips)
    con = _connect()
    try:
        if "field_coverage_pct" not in _columns(con, chips):
            return []
        row = con.execute(
            "SELECT quantile_cont(field_coverage_pct, [0.2, 0.4, 0.6, 0.8]) "
            f"FROM read_parquet('{_sql_path(chips)}') WHERE field_coverage_pct IS NOT NULL"
        ).fetchone()
    finally:
        con.close()
    if not row or row[0] is None:
        return []
    return _snap_stops(list(row[0])) or [50]


def top_codes(
    parquet: Path | str,
    code_col: str,
    name_col: str,
    weight: str = "count",
    limit: int = HCAT_MAX_LEGEND,
) -> list[tuple[int, str | None, float]]:
    """The most prominent HCAT codes in a file as ``(code, name, weight)``, largest first.

    ``weight`` is ``"count"`` (rows carrying the code) or ``"area"`` (summed
    geometry area, in the file's native units — degrees² for an EPSG:4326 file;
    only meaningful for ranking codes within this one file, not for comparing
    across files with different CRSs). Empty when the code column is absent.
    """
    parquet = Path(parquet)
    con = _connect()
    try:
        columns = _columns(con, parquet)
        if code_col not in columns:
            return []
        name_expr = f'MIN("{name_col}")' if name_col in columns else "CAST(NULL AS VARCHAR)"
        geom_col = detect_geometry_column(parquet, con) or "geometry"
        weight_expr = "COUNT(*)" if weight == "count" else f'SUM(ST_Area("{geom_col}"))'
        rows = con.execute(
            f'SELECT TRY_CAST("{code_col}" AS BIGINT) AS code, {name_expr} AS name_en, '
            f"{weight_expr} AS weight "
            f"FROM read_parquet('{_sql_path(parquet)}') "
            f'WHERE "{code_col}" IS NOT NULL '
            "GROUP BY code HAVING code IS NOT NULL "
            f"ORDER BY weight DESC, code LIMIT {int(limit)}"
        ).fetchall()
    finally:
        con.close()
    return [(int(code), name, float(w or 0.0)) for code, name, w in rows]


def _base(name: str, tiles_href: str, description: str) -> dict:
    return {
        "version": 8,
        "name": name,
        "metadata": {"description": description},
        "sources": {"data": {"type": "vector", "url": f"pmtiles://{tiles_href}"}},
        "layers": [],
    }


def _outline_layer(layer: str, layer_id: str, *, color: str = "rgba(60,60,60,0.5)") -> dict:
    return {
        "id": layer_id,
        "type": "line",
        "source": "data",
        "source-layer": layer,
        "paint": {"line-color": color, "line-width": 0.5},
    }


def split_style(
    counts: dict[str, int], *, tiles_href: str, layer: str = CHIPS_TILES.layer
) -> tuple[str, dict, list[dict]]:
    """Chips coloured by their train / val / test assignment."""
    expr: list = ["match", ["get", "split"]]
    legend: list[dict] = []
    for split in SPLIT_ORDER:
        if counts.get(split):
            expr += [split, SPLIT_COLORS[split]]
            legend.append({"label": split, "color": SPLIT_COLORS[split], "count": counts[split]})
    expr.append(OTHER_COLOR)
    style = _base(
        "Chips by split",
        tiles_href,
        "Chips coloured by their train / val / test assignment.",
    )
    style["layers"] = [
        {
            "id": "chips-by-split",
            "type": "fill",
            "source": "data",
            "source-layer": layer,
            "paint": {"fill-color": expr, "fill-opacity": 0.65},
        },
        _outline_layer(layer, "chips-outline", color="rgba(51,51,51,0.6)"),
    ]
    return "split", style, legend


def _coverage_labels(stops: list[float]) -> list[str]:
    labels = [f"< {stops[0]:g}%"]
    labels += [f"{a:g}-{b:g}%" for a, b in pairwise(stops)]
    labels.append(f">= {stops[-1]:g}%")
    return labels


def coverage_style(
    stops: list[float], *, tiles_href: str, layer: str = CHIPS_TILES.layer
) -> tuple[str, dict, list[dict]]:
    """Chips shaded light to dark by the share of their area covered by fields."""
    ramp = COVERAGE_RAMP[: len(stops) + 1]
    expr: list = ["step", ["get", "field_coverage_pct"], ramp[0]]
    for stop, color in zip(stops, ramp[1:], strict=False):
        expr.extend([stop, color])
    style = _base(
        "Field coverage",
        tiles_href,
        "Chips shaded light to dark by field_coverage_pct, with stops at the nice values "
        "nearest this collection's 20th/40th/60th/80th percentiles so each class holds a "
        "similar share of chips.",
    )
    style["layers"] = [
        {
            "id": "chips-by-coverage",
            "type": "fill",
            "source": "data",
            "source-layer": layer,
            "paint": {"fill-color": expr, "fill-opacity": 0.8},
        },
        _outline_layer(layer, "chips-outline", color="rgba(60,60,60,0.4)"),
    ]
    legend = [
        {"label": label, "color": color}
        for label, color in zip(_coverage_labels(stops), ramp, strict=False)
    ]
    return "field-coverage", style, legend


def _crop_label(code: int, name: str | None, entry: dict | None, taken: set[str]) -> str:
    label = name or (entry["name"].replace("_", " ").capitalize() if entry else f"HCAT {code}")
    if label in taken:
        label = f"{label} ({code})"
    return label


def _crop_style(
    rows: list[tuple[int, str | None, float]],
    *,
    style_id: str,
    name: str,
    description: str,
    code_property: str,
    layer_id: str,
    tiles_href: str,
    layer: str,
) -> tuple[str, dict, list[dict]]:
    """Outer ``match`` on crop labels carrying the legend, inner ``match`` on the code."""
    palette = load_palette()
    total = sum(w for _, _, w in rows) or 1.0
    inner: list = ["match", ["get", code_property]]
    outer: list = ["match", inner]
    legend: list[dict] = []
    used: set[str] = set()
    labels: set[str] = set()
    for code, crop_name, weight in rows[:HCAT_MAX_LEGEND]:
        entry = palette.get(str(code))
        label = _crop_label(code, crop_name, entry, labels)
        labels.add(label)
        color = distinct_color(entry.get("color") if entry else None, used)
        used.add(color.lower())
        inner.extend([int(code), label])
        outer.extend([label, color])
        legend.append({"label": label, "code": int(code), "color": color, "share": weight / total})
    inner.append("Other")
    outer.append(OTHER_COLOR)
    legend.append({"label": "Other", "color": OTHER_COLOR})

    style = _base(name, tiles_href, description)
    style["layers"] = [
        {
            "id": layer_id,
            "type": "fill",
            "source": "data",
            "source-layer": layer,
            "paint": {"fill-color": outer, "fill-opacity": 0.75},
        },
        _outline_layer(layer, f"{layer_id}-outline"),
    ]
    return style_id, style, legend


def dominant_crop_style(
    rows: list[tuple[int, str | None, float]], *, tiles_href: str, layer: str = CHIPS_TILES.layer
) -> tuple[str, dict, list[dict]]:
    """Chips coloured by the crop covering most of their field area."""
    return _crop_style(
        rows,
        style_id="dominant-crop",
        name="Dominant crop per chip",
        description=(
            "Chips coloured by hcat_dominant_code, the harmonized crop covering most of "
            "the chip's field area. The most common crops are named, everything else is grey. "
            "Colours follow the fiboa.org crop map palette."
        ),
        code_property="hcat_dominant_code",
        layer_id="chips-by-dominant-crop",
        tiles_href=tiles_href,
        layer=layer,
    )


def crops_style(
    rows: list[tuple[int, str | None, float]], *, tiles_href: str, layer: str = FIELDS_TILES.layer
) -> tuple[str, dict, list[dict]]:
    """Fields coloured by their harmonized crop (EuroCrops HCAT code)."""
    return _crop_style(
        rows,
        style_id="crops",
        name="Crops",
        description=(
            "Fields coloured by harmonized crop (EuroCrops HCAT code, `hcat:code`). The crops "
            "covering the most area are named, everything else is grey. Colours follow the "
            "fiboa.org crop map palette."
        ),
        code_property="hcat:code",
        layer_id="fields-by-crop",
        tiles_href=tiles_href,
        layer=layer,
    )


def outline_style(
    collection_id: str, *, tiles_href: str, layer: str = FIELDS_TILES.layer
) -> tuple[str, dict, list[dict]]:
    """Every field in one per-collection colour, for reading the boundaries themselves."""
    idx = int(hashlib.md5(collection_id.encode()).hexdigest(), 16) % len(OUTLINE_PALETTE)
    color = OUTLINE_PALETTE[idx]
    style = _base(
        "Field outlines",
        tiles_href,
        "Every field in one colour with a thin outline; for reading the boundaries themselves.",
    )
    style["layers"] = [
        {
            "id": "fields-fill",
            "type": "fill",
            "source": "data",
            "source-layer": layer,
            "paint": {"fill-color": color, "fill-opacity": 0.25},
        },
        {
            "id": "fields-outline",
            "type": "line",
            "source": "data",
            "source-layer": layer,
            "paint": {"line-color": color, "line-width": 0.8},
        },
    ]
    return "outline", style, []


def _write_style(styles_dir: Path, style_id: str, style: dict) -> Path:
    path = styles_dir / f"{style_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(style, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _chip_styles(
    chips_parquet: Path | None, tiles_href: str | None
) -> list[tuple[str, dict, list[dict], str]]:
    """Built chip styles as ``(style_id, style, legend, title)``, in presentation order."""
    if not tiles_href or chips_parquet is None or not Path(chips_parquet).exists():
        return []
    built: list[tuple[str, dict, list[dict], str]] = []
    counts = split_counts(chips_parquet)
    if counts:
        built.append((*split_style(counts, tiles_href=tiles_href), "Chips by split"))
    stops = coverage_quantiles(chips_parquet)
    if stops:
        built.append((*coverage_style(stops, tiles_href=tiles_href), "Field coverage"))
    dominant = top_codes(chips_parquet, "hcat_dominant_code", "hcat_dominant_name_en", "count")
    if dominant:
        built.append(
            (*dominant_crop_style(dominant, tiles_href=tiles_href), "Dominant crop per chip")
        )
    return built


def _field_styles(
    collection_id: str, fields_parquet: Path | None, tiles_href: str | None
) -> list[tuple[str, dict, list[dict], str]]:
    """Built field styles as ``(style_id, style, legend, title)``, in presentation order."""
    if not tiles_href:
        return []
    built: list[tuple[str, dict, list[dict], str]] = []
    if fields_parquet is not None and Path(fields_parquet).exists():
        crops = top_codes(fields_parquet, "hcat:code", "hcat:name_en", "area")
        if crops:
            built.append((*crops_style(crops, tiles_href=tiles_href), "Crops"))
    built.append((*outline_style(collection_id, tiles_href=tiles_href), "Field outlines"))
    return built


def write_styles(
    output_dir: Path | str,
    collection_id: str,
    chips_parquet: Path | str | None,
    fields_parquet: Path | str | None,
    *,
    chips_tiles: str | None,
    fields_tiles: str | None,
    on_progress: Callable[[str], None] | None = None,
) -> list[StyleResult]:
    """Write ``styles/<id>.json`` for every style whose data and tiles are present.

    The first style written is marked as the collection's default.
    """
    styles_dir = Path(output_dir) / "styles"
    chips = Path(chips_parquet) if chips_parquet else None
    fields = Path(fields_parquet) if fields_parquet else None
    built = _chip_styles(chips, chips_tiles) + _field_styles(collection_id, fields, fields_tiles)

    results: list[StyleResult] = []
    for style_id, style, legend, title in built:
        path = _write_style(styles_dir, style_id, style)
        if on_progress:
            on_progress(f"style: {style_id}")
        results.append(
            StyleResult(
                style_id=style_id,
                path=path,
                title=title,
                legend_rows=legend,
                default=not results,
            )
        )
    return results
