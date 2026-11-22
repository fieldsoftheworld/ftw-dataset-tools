"""Tile the root chips table into ``chips.pmtiles``: every chip location, at every zoom.

Needs tippecanoe and tile-join on PATH. A location is in every collection that covers it
(other years, and a few overlapping regions), so the table repeats it. Four layers, none
dropping a location:

``chips`` (zoom 7-12): one polygon per chip id, with

- ``id``, ``block``, ``split_hash_v1`` and ``split_admin2_v1`` (the same in every
  dataset, by construction);
- ``datasets``: how many datasets have the chip;
- ``presence_only``: true when any of them is presence-only;
- ``legacy_conflict``: true when those datasets' ``split_legacy`` values disagree (any
  two of train, val and test);
- ``admin0_name``.

``overview_train``, ``overview_val`` and ``overview_test`` (zoom 0-6): the chip centres of
each ``split_admin2_v1``, clustered where they crowd, each split on its own so a cluster
never mixes splits and the whole catalog shows at once. Each point (or cluster) sums
``chips`` (locations), ``conflicts`` (locations with ``legacy_conflict``) and
``datasets`` (chip-collection pairs).
"""

from __future__ import annotations

import io
import json
import subprocess
import tempfile
import urllib.request
from pathlib import Path

import duckdb

from ftw_dataset_tools.api import admin_names
from ftw_dataset_tools.api.geo import sql_path
from ftw_dataset_tools.api.tiles import TileSpec, build_pmtiles

TILES_NAME = "chips.pmtiles"
DEFAULT_SERVER = "http://localhost:13579"
# The whole catalog in one frame, rendered large and shrunk so the clusters stay crisp.
THUMBNAIL_BBOX = (-130, -45, 150, 72)
THUMBNAIL_RENDER_SIZE = 2400
THUMBNAIL_SIZE = 1200
ATTRIBUTES = (
    "id",
    "block",
    "split_hash_v1",
    "split_admin2_v1",
    "datasets",
    "presence_only",
    "legacy_conflict",
    "admin0_name",
)
SPLITS = ("train", "val", "test")
OVERVIEW_LAYERS = tuple(f"overview_{split}" for split in SPLITS)
OVERVIEW_SUMS = ("chips", "conflicts", "datasets")
REQUIRED_COLUMNS = (
    "id",
    "block",
    "split_legacy",
    "split_hash_v1",
    "split_admin2_v1",
    "presence_only",
    "admin0_name",
    "geometry",
)
CHIPS_ZOOMS = (7, 12)
# tippecanoe turns sub-pixel polygons into dust, which can lose them. At zoom 7 a chip of
# this area is about 10 tile units across, far above that; real chips are 4 km2 or more.
MIN_CHIP_KM2 = 0.5
OVERVIEW_ZOOMS = (0, 6)


class ChipsTilesError(Exception):
    """The table can't be tiled."""


def _sql(value: str) -> str:
    return f"'{sql_path(value)}'"


def _check_columns(con: duckdb.DuckDBPyConnection, table: str) -> None:
    source = f"read_parquet({_sql(table)})"
    columns = {r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {source}").fetchall()}
    missing = [c for c in REQUIRED_COLUMNS if c not in columns]
    if missing:
        raise ChipsTilesError(
            f"{table} has no {', '.join(missing)}; build it with ftwd chips-table first"
        )


def _check_footprints(con: duckdb.DuckDBPyConnection, source: str) -> None:
    area = "ST_Area_Spheroid(ST_FlipCoordinates(geometry)) / 1e6"
    tiny = con.execute(
        f"SELECT id, {area} AS km2 FROM {source} WHERE {area} < {MIN_CHIP_KM2} LIMIT 1"
    ).fetchone()
    if tiny:
        raise ChipsTilesError(
            f"chip {tiny[0]} covers {tiny[1]:.3f} km2, below {MIN_CHIP_KM2} km2: the tiles "
            f"could lose it at zoom {CHIPS_ZOOMS[0]}"
        )


def _check_table(con: duckdb.DuckDBPyConnection, table: str) -> None:
    """Refuse a table the one-feature-per-location tiles would misrepresent."""
    _check_columns(con, table)
    source = f"read_parquet({_sql(table)})"
    bad = con.execute(f"""
        SELECT id FROM {source} GROUP BY id
        HAVING count(DISTINCT split_hash_v1) > 1 OR count(DISTINCT split_admin2_v1) > 1
            OR count(DISTINCT ST_AsWKB(geometry)) > 1
        LIMIT 1""").fetchone()
    if bad:
        raise ChipsTilesError(f"chip {bad[0]} has more than one hash or admin2 split or footprint")
    splits = ", ".join(_sql(s) for s in SPLITS)
    unsplit = con.execute(f"""
        SELECT id FROM {source}
        WHERE split_admin2_v1 IS NULL OR split_admin2_v1 NOT IN ({splits}) LIMIT 1""").fetchone()
    if unsplit:
        raise ChipsTilesError(
            f"chip {unsplit[0]} has no admin2 split; the overview would leave it out"
        )
    _check_footprints(con, source)


def write_locations(table: str, out: Path) -> int:
    """One row per chip id from ``table`` as GeoParquet at ``out``; the number of rows.

    Refuses a table without REQUIRED_COLUMNS, one where a chip id has more than one hash
    or admin2 split or geometry (which the one-feature-per-location tiles would hide), a
    chip without an admin2 split, or a footprint below MIN_CHIP_KM2.
    """
    import geoparquet_io as gpio

    con = admin_names.connect()
    try:
        _check_table(con, table)
        tmp = out.with_suffix(".tmp.parquet")
        con.execute(f"""
            COPY (SELECT id, any_value(block) AS block,
                         any_value(split_hash_v1) AS split_hash_v1,
                         any_value(split_admin2_v1) AS split_admin2_v1,
                         count(*)::INTEGER AS datasets,
                         bool_or(presence_only) AS presence_only,
                         count(DISTINCT split_legacy) > 1 AS legacy_conflict,
                         any_value(admin0_name) AS admin0_name,
                         any_value(geometry) AS geometry
                  FROM read_parquet({_sql(table)}) GROUP BY id ORDER BY id)
            TO {_sql(str(tmp))} (FORMAT parquet)""")
        gpio.read(str(tmp)).write(str(out))
        tmp.unlink()
        return con.execute(f"SELECT count(*) FROM read_parquet({_sql(str(out))})").fetchone()[0]
    except duckdb.Error as err:
        raise ChipsTilesError(f"can't read {table}: {err}") from err
    finally:
        con.close()


def write_overview(locations: Path, out: Path, split: str) -> int:
    """The centres of ``split``'s chips with the counts the overview sums, as GeoJSONSeq;
    the number of points."""
    con = admin_names.connect()
    where = f"FROM read_parquet({_sql(str(locations))}) WHERE split_admin2_v1 = {_sql(split)}"
    try:
        con.execute(f"""
            COPY (SELECT 1 AS chips, legacy_conflict::INTEGER AS conflicts, datasets,
                         ST_Centroid(geometry) AS geometry {where})
            TO {_sql(str(out))} WITH (FORMAT GDAL, DRIVER 'GeoJSONSeq')""")
        return con.execute(f"SELECT count(*) {where}").fetchone()[0]
    except duckdb.Error as err:
        raise ChipsTilesError(f"can't write the {split} overview: {err}") from err
    finally:
        con.close()


def _run(cmd: list[str]) -> None:
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as err:
        detail = getattr(err, "stderr", "") or str(err)
        raise ChipsTilesError(f"{cmd[0]} failed: {detail.strip()[:500]}") from err


def overview_command(seq: Path, out: Path, layer: str) -> list[str]:
    """tippecanoe for an overview layer: cluster crowded points, summing every count, drop none."""
    cmd = [
        "tippecanoe",
        "-o",
        str(out),
        "-l",
        layer,
        "--force",
        "--quiet",
        f"-Z{OVERVIEW_ZOOMS[0]}",
        f"-z{OVERVIEW_ZOOMS[1]}",
        "-r1",
        "--cluster-densest-as-needed",
        "--cluster-distance=2",
    ]
    cmd += [f"--accumulate-attribute={name}:sum" for name in OVERVIEW_SUMS]
    return [*cmd, str(seq)]


def _dropped(strategy: dict | None) -> dict:
    return {k: v for k, v in (strategy or {}).items() if k.startswith("dropped")}


def check_no_drops(archive: Path, layer: str) -> None:
    """Refuse an archive whose tippecanoe run dropped features at any zoom."""
    from pmtiles.reader import MmapSource, Reader

    with archive.open("rb") as f:
        strategies = Reader(MmapSource(f)).metadata().get("strategies", [])
    if isinstance(strategies, str):
        strategies = json.loads(strategies)
    for zoom, strategy in enumerate(strategies):
        dropped = _dropped(strategy)
        if dropped:
            raise ChipsTilesError(
                f"tippecanoe dropped {layer} features at zoom {zoom}: {dropped}; "
                f"every location must be kept"
            )


def _build_overviews(locations: Path, tmp: Path, count: int) -> list[str]:
    """One clustered archive per split that has chips; refuses a location left out."""
    overviews, points = [], 0
    for split, layer in zip(SPLITS, OVERVIEW_LAYERS, strict=True):
        seq, archive = tmp / f"{layer}.geojsonseq", tmp / f"{layer}.pmtiles"
        n = write_overview(locations, seq, split)
        if not n:
            continue
        points += n
        _run(overview_command(seq, archive, layer))
        check_no_drops(archive, layer)
        overviews.append(str(archive))
    if points != count:
        raise ChipsTilesError(f"the overview has {points:,} of {count:,} locations")
    return overviews


def build_tiles(table: str, out: Path | str) -> int:
    """Tile ``table``'s chip locations into ``out``; the number of locations."""
    out = Path(out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp_dir:
        tmp = Path(tmp_dir)
        locations = tmp / "locations.parquet"
        count = write_locations(table, locations)
        chips = tmp / "chips.pmtiles"
        # from zoom 7 the densest tile is a quarter of the size limit; a drop is refused
        spec = TileSpec("chips", ATTRIBUTES, *CHIPS_ZOOMS, False, True)
        try:
            build_pmtiles(locations, chips, spec)
        except RuntimeError as err:
            raise ChipsTilesError(str(err)) from err
        check_no_drops(chips, "chips")
        overviews = _build_overviews(locations, tmp, count)
        _run(
            [
                "tile-join",
                "-o",
                str(out),
                "--force",
                "--quiet",
                "--no-tile-size-limit",
                "-n",
                "Chips of every collection",
                "-N",
                "Fields of the World benchmark chip locations with their splits",
                str(chips),
                *overviews,
            ]
        )
    return count


def render_thumbnail(
    out: Path | str, *, style: Path | str, tiles_url: str, server: str = DEFAULT_SERVER
) -> Path:
    """Render ``style`` over the tiles at ``tiles_url`` with chiitiler into ``out`` (WebP).

    The style's ``data`` source is pointed at ``tiles_url``. Needs a running chiitiler
    at ``server``. Refuses an image with no data in it.
    """
    from PIL import Image

    out = Path(out)
    doc = json.loads(Path(style).read_text())
    doc["sources"]["data"] = {
        "type": "vector",
        "minzoom": OVERVIEW_ZOOMS[0],
        "maxzoom": CHIPS_ZOOMS[1],
        "tiles": [f"pmtiles://{tiles_url}/{{z}}/{{x}}/{{y}}"],
    }
    bbox = ",".join(str(v) for v in THUMBNAIL_BBOX)
    request = urllib.request.Request(
        f"{server}/clip.png?bbox={bbox}&size={THUMBNAIL_RENDER_SIZE}",
        data=json.dumps({"style": doc}).encode(),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            image = Image.open(io.BytesIO(response.read())).convert("RGB")
    except OSError as err:
        raise ChipsTilesError(f"chiitiler at {server} failed: {err}") from err
    if all(low == high for low, high in image.getextrema()):
        raise ChipsTilesError("the thumbnail is blank: no tiles rendered")
    image.thumbnail((THUMBNAIL_SIZE, THUMBNAIL_SIZE))
    image.save(out, "WEBP", quality=85)
    return out
