"""The root chips table: every chip of every dataset in a benchmark catalog, with its splits.

One row per (dataset, chip), sorted by group, dataset and chip id, so a reader that
filters on ``dataset`` skips most row groups. Columns, in order:

- ``dataset``, ``group`` (the folder above the dataset in the catalog), ``id``, ``block``
  (the chip's 3x3 grid block) and the admin names.
- ``license`` and ``license_url`` (the collection's license and its own license link, or
  the SPDX page for a standard id), ``license_category`` (``LICENSE_CATEGORIES``) and
  ``presence_only`` (the collection's ``ftw:presence_only``).
- One column per split scheme, all ``train`` / ``val`` / ``test`` or null:

  - ``split_legacy``: the split each dataset published before the table existed. Frozen:
    every build carries it over from the previous table, and chips added later get null.
  - ``split_predefined``: the source's own split, for collections with
    ``ftw:split_type: predefined``; null elsewhere. Read from the chips file of a local
    dataset; carried over from the previous table for the others.
  - ``split_hash_v1``: ``block3x3-hash`` at 80/10/10 with the salt pinned here
    (``HASH_SALT``). A block's split comes from a stable hash of its id, so a location has
    the same split in every dataset and year, and new chips never move existing ones; a
    build refuses to change a carried chip's block or hash split.
  - ``split_admin2_v1``, with its unit in ``unit_admin2_v1``: whole admin2 units, balanced
    to 80/10/10 of locations within each country (largest units first). A unit with any
    presence-only chip is train. A row with no admin2 name is in its admin1's unit; a row
    without a country or admin1 uses its block as the unit, and so does a country with
    fewer than three admin2 units to balance. Frozen once published (see
    :func:`admin2_splits`).

- ``geometry`` and ``bbox``.

The frozen columns exist nowhere else once the per-dataset files stop carrying ``split``,
so a build refuses anything that would lose them: a dataset of the previous table left
out, or chips of one gone (unless ``drop`` names the dataset), and a local predefined
dataset whose chips file has no ``split``.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

import duckdb
import numpy as np
import pandas as pd

from ftw_dataset_tools.api.blocks import block_scores, chip_block_ids
from ftw_dataset_tools.api.geo import configure_source_coop_s3, ensure_spatial_loaded, sql_path
from ftw_dataset_tools.api.logging_config import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

logger = get_logger(__name__)

TABLE_NAME = "chips.parquet"
SPLITS = ("train", "val", "test")
HASH_PERCENTS = (80, 10, 10)
HASH_SALT = "ftw-split-v1"  # SPLIT_HASH_VERSION when v1 was defined; never change it
ADMIN_COLUMNS = ("admin0_iso3", "admin0_name", "admin1_name", "admin2_name")
ADMIN2_SALT = "ftw-split-admin2-v1"  # never change it; a new scheme is split_admin2_v2
# A country needs this many admin2 units to give each split one; with fewer, blocks are units.
MIN_ADMIN2_UNITS = 3
DEFAULT_CHIP_SIZE_KM = 2.0
# license_category, most to least permissive. The first three are open: any use, including
# commercial, with at most attribution or share-alike.
LICENSE_CATEGORIES = (
    "public-domain",
    "attribution",
    "share-alike",
    "non-commercial",
    "restricted",
    "unknown",
)
# By SPDX id, or by license link for "other". Anything not listed is "unknown" until judged.
LICENSES = {
    "CC0-1.0": "public-domain",
    "DL-DE-ZERO-2.0": "public-domain",
    "https://inspire.ec.europa.eu/metadata-codelist/ConditionsApplyingToAccessAndUse/"
    "noConditionsApply": "public-domain",
    "CC-BY-4.0": "attribution",
    "DL-DE-BY-2.0": "attribution",
    "https://etalab.gouv.fr/licence-ouverte-open-licence": "attribution",
    "https://data.vlaanderen.be/id/licentie/modellicentie-gratis-hergebruik/v1.0": "attribution",
    "CC-BY-SA-4.0": "share-alike",
    "CC-BY-SA-3.0": "share-alike",
    "https://go.planet.com/nicfi-pla-2024": "non-commercial",  # NICFI imagery terms
    # St. Gallen: free public apps only; passing the data on needs approval
    "https://www.sg.ch/bauen/geoinformation/datenbezug/agb.html": "restricted",
}


class ChipsTableError(Exception):
    """The inputs can't make a consistent table."""


@dataclass(frozen=True)
class DatasetInput:
    """Where one dataset's chips come from and how its splits were made."""

    dataset: str
    group: str
    chips: str  # path or URL of the chips file this build reads
    split_type: str | None
    km_size: float
    license: str | None = None
    license_url: str | None = None
    presence_only: bool = False
    local: bool = False  # read from a local build rather than the published files


def license_category(license: str | None, license_url: str | None) -> str:
    """A dataset's LICENSE_CATEGORIES entry, from LICENSES."""
    if license in LICENSES:
        return LICENSES[license]
    return LICENSES.get(license_url, "unknown") if license == "other" else "unknown"


def license_fields(collection: dict) -> dict:
    """The dataset's license: its id, its own license link (not the embeddings'; the SPDX
    page for a standard id without one), and presence_only."""
    license = collection.get("license")
    links = [
        link["href"]
        for link in collection.get("links", [])
        if link.get("rel") == "license" and "olmoearth" not in link["href"].lower()
    ]
    url = links[0] if links else _spdx_url(license)
    return {
        "license": license,
        "license_url": url,
        "presence_only": bool(collection.get("ftw:presence_only", False)),
    }


def _spdx_url(license: str | None) -> str | None:
    return f"https://spdx.org/licenses/{license}.html" if license and license != "other" else None


def _sql(value: str) -> str:
    return f"'{sql_path(value)}'"


def _join(base: str, href: str) -> str:
    """``href`` resolved against the folder ``base`` (a path or URL), unless absolute."""
    return href if "://" in href else f"{base.rstrip('/')}/{href.removeprefix('./')}"


def _location(path_or_url: str | Path) -> str:
    """A URL as given, a path made absolute, so its folder is always ``rsplit("/")[0]``."""
    text = str(path_or_url)
    return text if "://" in text else str(Path(text).resolve())


def _read_json(location: str) -> dict:
    try:
        if "://" not in location:
            return json.loads(Path(location).read_text())
        with urllib.request.urlopen(location, timeout=60) as response:
            return json.load(response)
    except json.JSONDecodeError as err:
        raise ChipsTableError(f"{location} is not valid JSON: {err}") from err


def published_datasets(catalog_json: str, rel: str = "") -> list[str]:
    """``<group>/<id>`` for every collection the root catalog links, through its groups.

    Follows ``rel: child`` links: a child ``catalog.json`` (a group) is walked in turn,
    a child ``collection.json`` in a group is a published dataset. Datasets live in
    groups, so a collection the root links directly (the all-chips collection that holds
    this table) is not one.
    """
    catalog_json = _location(catalog_json)
    links = _read_json(catalog_json).get("links", [])
    base = catalog_json.rsplit("/", 1)[0]
    found: list[str] = []
    for link in links:
        if link["rel"] != "child":
            continue
        child = link["href"].removeprefix("./")
        folder = child.rsplit("/", 1)[0]
        if child.endswith("/collection.json"):
            if rel:
                found.append(f"{rel}{folder}")
        elif child.endswith("/catalog.json"):
            found += published_datasets(_join(base, child), f"{rel}{folder}/")
    return sorted(found)


def _collection_input(
    collection: dict, dataset: str, group: str, chips: str, *, local: bool
) -> DatasetInput:
    return DatasetInput(
        dataset,
        group,
        chips,
        collection.get("ftw:split_type"),
        float(collection.get("ftw:chip_size_km", DEFAULT_CHIP_SIZE_KM)),
        **license_fields(collection),
        local=local,
    )


def _chips_href(collection: dict) -> str | None:
    return ((collection.get("assets") or {}).get("chips") or {}).get("href")


def _read_local(folder: Path) -> tuple[dict, str]:
    """The collection of a dataset written by ``ftwd run`` into ``folder``, and its id."""
    collection_path = folder / "collection.json"
    if not collection_path.is_file():
        raise ChipsTableError(
            f"{folder}: {collection_path} does not exist (the table needs its license and "
            f"presence_only)"
        )
    collection = json.loads(collection_path.read_text())
    return collection, collection.get("id") or folder.name


def _local_input(folder: Path, collection: dict, dataset: str, group: str) -> DatasetInput:
    chips = Path(_join(str(folder), _chips_href(collection) or f"{dataset}_chips.parquet"))
    if not chips.is_file():
        raise ChipsTableError(f"{dataset}: --local, but {chips} does not exist")
    return _collection_input(collection, dataset, group, str(chips), local=True)


def _published_inputs(catalog_json: str, data_base_url: str) -> dict[str, DatasetInput]:
    root = catalog_json.rsplit("/", 1)[0]
    inputs: dict[str, DatasetInput] = {}
    for relpath in published_datasets(catalog_json):
        group, _, folder = relpath.rpartition("/")
        collection = _read_json(_join(root, f"{relpath}/collection.json"))
        dataset = collection.get("id") or folder
        if dataset in inputs:
            raise ChipsTableError(
                f"{dataset} is published twice: in {inputs[dataset].group} and {group}"
            )
        href = _chips_href(collection) or f"./{dataset}_chips.parquet"
        chips = _join(f"{data_base_url.rstrip('/')}/{relpath}", href)
        inputs[dataset] = _collection_input(collection, dataset, group, chips, local=False)
    return inputs


def _local_group(dataset: str, published: DatasetInput | None, groups: Mapping[str, str]) -> str:
    given = groups.get(dataset)
    if published is None:
        if given is None:
            raise ChipsTableError(
                f"{dataset} is not in the catalog; give its group with --group {dataset}=<group>"
            )
        return given
    if given is not None and given != published.group:
        raise ChipsTableError(
            f"{dataset}: --group says {given}, but the catalog has it in {published.group}"
        )
    return published.group


def _add_local(
    inputs: dict[str, DatasetInput], local_dirs: Iterable[Path | str], groups: dict[str, str]
) -> None:
    """Add (or replace) each local build in ``inputs``."""
    local_ids = []
    for folder in map(Path, local_dirs):
        collection, dataset = _read_local(folder)
        local_ids.append(dataset)
        group = _local_group(dataset, inputs.get(dataset), groups)
        inputs[dataset] = _local_input(folder, collection, dataset, group)
    unknown = sorted(set(groups) - set(local_ids))
    if unknown:
        raise ChipsTableError(f"--group names {', '.join(unknown)}, which no --local dataset has")


def resolve_inputs(
    catalog_json: str | Path,
    data_base_url: str | None = None,
    local_dirs: Iterable[Path | str] = (),
    groups: Mapping[str, str] | None = None,
) -> list[DatasetInput]:
    """One input per dataset the root catalog links, plus each local build.

    Published chips files are read from ``data_base_url`` (default: the catalog's own
    folder) joined with the collection's path relative to the catalog root. A local
    build (a directory written by ``ftwd run``) replaces the published dataset of the
    same id; one the catalog doesn't have needs its group in ``groups``.
    """
    catalog_json = _location(catalog_json)
    base = data_base_url or catalog_json.rsplit("/", 1)[0]
    inputs = _published_inputs(catalog_json, base)
    _add_local(inputs, local_dirs, dict(groups or {}))
    return [inputs[k] for k in sorted(inputs)]


def _columns(con: duckdb.DuckDBPyConnection, source: str) -> set[str]:
    return {
        r[0] for r in con.execute(f"DESCRIBE SELECT * FROM read_parquet({_sql(source)})").fetchall()
    }


def _read_dataset(item: DatasetInput, out: Path) -> None:
    """One dataset's rows as a parquet file in the table's column order (no hash yet)."""
    try:
        _copy_dataset(item, out)
    except duckdb.Error as err:
        raise ChipsTableError(f"{item.dataset}: {err}") from err


def _license_select(item: DatasetInput) -> str:
    def literal(value: str | None) -> str:
        return _sql(value) if value else "NULL"

    category = license_category(item.license, item.license_url)
    return ", ".join(
        [
            f"{literal(item.license)}::VARCHAR AS license",
            f"{literal(item.license_url)}::VARCHAR AS license_url",
            f"{_sql(category)} AS license_category",
            f"{'true' if item.presence_only else 'false'} AS presence_only",
        ]
    )


def _check_local_split(item: DatasetInput, cols: set[str]) -> None:
    """Refuse a local predefined dataset whose chips file lost its split."""
    if item.local and item.split_type == "predefined" and "split" not in cols:
        raise ChipsTableError(
            f"{item.dataset}: the collection uses predefined splits, but "
            f"{item.chips} has no split column"
        )


def _connect(source: str) -> duckdb.DuckDBPyConnection:
    """A connection that reads ``source``; httpfs only for an S3 source, so threads reading
    local files never race to install it."""
    con = duckdb.connect(":memory:")
    con.execute("SET enable_progress_bar = false")
    ensure_spatial_loaded(con)
    if source.startswith("s3://"):
        configure_source_coop_s3(con)
    return con


def _install_extensions(sources: Sequence[str]) -> None:
    """Install spatial (and httpfs for remote sources) once, before the pool's threads
    each load them: concurrent INSTALLs race on a machine that has neither yet."""
    con = duckdb.connect(":memory:")
    try:
        con.execute("INSTALL spatial")
        if any("://" in s for s in sources):
            con.execute("INSTALL httpfs")
    finally:
        con.close()


def _copy_dataset(item: DatasetInput, out: Path) -> None:
    con = _connect(item.chips)
    try:
        cols = _columns(con, item.chips)
        _check_local_split(item, cols)
        admin = ", ".join(f"c.{c}" if c in cols else f"NULL::VARCHAR AS {c}" for c in ADMIN_COLUMNS)
        split = "c.split" if "split" in cols else "NULL::VARCHAR"
        predefined = split if item.split_type == "predefined" else "NULL::VARCHAR"
        con.execute(f"""
            COPY (SELECT {_sql(item.dataset)} AS dataset, {_sql(item.group)} AS "group",
                         c.id, {admin}, {_license_select(item)}, c.geometry, c.bbox,
                         NULL::VARCHAR AS split_legacy, {predefined} AS split_predefined
                  FROM read_parquet({_sql(item.chips)}) c)
            TO {_sql(str(out))} (FORMAT parquet)""")
    finally:
        con.close()


def hash_splits(ids: Sequence[str], km_size: float) -> tuple[np.ndarray, np.ndarray]:
    """``(block, split_hash_v1)`` for each chip id: block3x3-hash at HASH_PERCENTS."""
    blocks = chip_block_ids(pd.Series(ids), km_size)
    scores = block_scores(blocks, HASH_SALT).to_numpy()
    train, val = HASH_PERCENTS[0] / 100, (HASH_PERCENTS[0] + HASH_PERCENTS[1]) / 100
    return blocks.to_numpy(), np.where(
        scores < train, "train", np.where(scores < val, "val", "test")
    )


def _score(key: str) -> float:
    """A stable score in [0, 1) for an admin2 unit, the same on every machine and run."""
    digest = hashlib.blake2b(f"{ADMIN2_SALT}:{key}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2**64


def _admin2_keys(loc: pd.DataFrame) -> pd.Series:
    return (
        loc["country"].fillna("") + "|" + loc["admin1"].fillna("") + "|" + loc["admin2"].fillna("")
    )


def _units(locations: pd.DataFrame, frozen_blocks: dict[str, bool]) -> np.ndarray:
    """Each location's admin2 unit, or its block where the country has too few units.

    ``frozen_blocks`` (country -> uses blocks) keeps a carried country's kind of unit, so
    a country never switches between admin2 units and blocks once published.
    """
    loc = locations
    known = loc["country"].notna() & loc["admin1"].notna()
    admin2 = _admin2_keys(loc)
    eligible = admin2[known & ~loc["unit_po"]].groupby(loc["country"]).nunique()
    few = loc["country"].map(eligible).fillna(0) < MIN_ADMIN2_UNITS
    frozen = loc["country"].map(frozen_blocks)
    few = np.where(frozen.notna(), frozen, few).astype(bool)
    block = loc["country"].fillna("") + "|block:" + loc["block"]
    return np.where(known & ~few, admin2, block)


def _fill(units: dict[str, int], counts: dict[str, int]) -> dict[str, str]:
    """Splits for new ``units`` (unit -> locations) of one country.

    Largest first (then in hash order), each to the split furthest below its share,
    counting ``counts`` (split -> locations) already there. A split left with no
    locations then takes the smallest new unit of a split that has more than one, so a
    country with three units or more gets all three splits.
    """
    shares = dict(zip(SPLITS, (p / 100 for p in HASH_PERCENTS), strict=True))
    total = sum(counts.values()) + sum(units.values())
    have = {s: counts.get(s, 0) for s in SPLITS}
    out = {}
    for unit in sorted(units, key=lambda u: (-units[u], _score(u), u)):
        split = max(SPLITS, key=lambda s: shares[s] * total - have[s])
        out[unit] = split
        have[split] += units[unit]
    _fill_missing(out, units, counts, have)
    return out


def _donors(out: dict[str, str], counts: dict[str, int]) -> list[str]:
    """Splits with more than one unit, counting the country's carried locations as one."""
    per_split = Counter(out.values())
    return [s for s in SPLITS if per_split[s] + (counts.get(s, 0) > 0) > 1]


def _fill_missing(
    out: dict[str, str], units: dict[str, int], counts: dict[str, int], have: dict[str, int]
) -> None:
    """Give each split without locations the smallest new unit of a split with several."""
    for missing in [s for s in SPLITS if have[s] == 0]:
        donors = _donors(out, counts)
        if not donors:
            break
        donor = max(donors, key=lambda s: have[s])
        unit = min((u for u, s in out.items() if s == donor), key=lambda u: (units[u], u))
        out[unit], have[donor], have[missing] = missing, have[donor] - units[unit], units[unit]


def _frozen_blocks(loc: pd.DataFrame, is_carried: pd.Series, carried: pd.DataFrame) -> dict:
    """Country -> whether its carried locations use blocks as units."""
    frozen = loc[is_carried & loc["country"].notna() & loc["admin1"].notna()]
    return (
        frozen["id"]
        .map(carried["unit"])
        .astype(str)
        .str.contains("|block:", regex=False)
        .groupby(frozen["country"])
        .all()
        .to_dict()
    )


def _check_frozen_presence_only(loc: pd.DataFrame, known: dict[str, str]) -> None:
    frozen_po = loc[loc["unit_po"] & loc["unit"].map(known).isin(["val", "test"])]
    if len(frozen_po):
        row = frozen_po.iloc[0]
        raise ChipsTableError(
            f"admin2 unit {row['unit']} is frozen as {known[row['unit']]} but now has "
            f"presence-only chips (e.g. {row['id']}); presence-only chips can't be scored"
        )


def admin2_splits(locations: pd.DataFrame, previous: pd.DataFrame | None = None) -> pd.DataFrame:
    """``unit_admin2_v1`` and ``split_admin2_v1`` per location.

    ``locations``: one row per id with ``country``, ``admin1``, ``admin2``, ``block`` and
    ``presence_only`` (any of its rows). ``previous``: ``id``, ``unit``, ``split`` from the
    published table, or None for a first build. Returns ``id``, ``unit``, ``split``.

    A later build carries every chip's unit and split and each country's kind of unit,
    gives a new chip of a known unit that unit's split, and fills new units into the split
    furthest below its share in their country. It refuses a frozen val or test unit that
    gains a presence-only chip.
    """
    loc = locations.reset_index(drop=True).copy()
    prev = previous if previous is not None else pd.DataFrame(columns=["id", "unit", "split"])
    carried = prev.set_index("id")
    # a unit is presence-only when any location in it is; first by admin2, to decide the units
    loc["unit_po"] = loc.groupby(_admin2_keys(loc))["presence_only"].transform("any")
    is_carried = loc["id"].isin(carried.index)
    loc["unit"] = _units(loc, _frozen_blocks(loc, is_carried, carried))
    loc.loc[is_carried, "unit"] = loc.loc[is_carried, "id"].map(carried["unit"])
    loc["unit_po"] = loc.groupby("unit")["presence_only"].transform("any")
    known = dict(zip(prev["unit"], prev["split"], strict=True))
    _check_frozen_presence_only(loc, known)
    loc["split"] = loc["unit"].map(known).astype(object)
    loc.loc[loc["unit_po"] & loc["split"].isna(), "split"] = "train"
    open_ = loc[loc["split"].isna()]
    country = loc["country"].fillna("")
    for c, new in open_.groupby(country[open_.index]):
        base = loc[(country == c) & loc["split"].notna() & ~loc["unit_po"]]
        filled = _fill(new.groupby("unit").size().to_dict(), base["split"].value_counts().to_dict())
        loc.loc[new.index, "split"] = new["unit"].map(filled)
    return loc[["id", "unit", "split"]]


def _admin2_table(con: duckdb.DuckDBPyConnection, previous: str) -> None:
    """Table ``admin2(id, unit_admin2_v1, split_admin2_v1)`` over the chips in ``rows``."""
    # one row's whole admin tuple (the smallest), so a location's unit is stable across runs
    locations = con.execute("""
        SELECT id, a[1] AS country, a[2] AS admin1, a[3] AS admin2, block, presence_only
        FROM (SELECT r.id, min([r.admin0_iso3, r.admin1_name, r.admin2_name]) AS a,
                     min(h.block) AS block, bool_or(r.presence_only) AS presence_only
              FROM rows r JOIN sizes s USING (dataset)
              JOIN hashed h ON h.id = r.id AND h.km_size = s.km_size GROUP BY r.id)""").df()
    if "split_admin2_v1" not in _columns(con, previous):
        raise ChipsTableError(
            f"the previous table {previous} has no split_admin2_v1, so the split would start "
            f"from scratch"
        )
    prev = con.execute(f"""
        SELECT id, any_value(unit_admin2_v1) AS unit, any_value(split_admin2_v1) AS split
        FROM read_parquet({_sql(previous)}) GROUP BY id""").df()
    out = admin2_splits(locations, prev)
    con.register(
        "admin2_df", out.rename(columns={"unit": "unit_admin2_v1", "split": "split_admin2_v1"})
    )
    con.execute("CREATE TABLE admin2 AS SELECT * FROM admin2_df")


def _distinct_ids(con: duckdb.DuckDBPyConnection, datasets: list[str]) -> list[str]:
    rows = con.execute(
        "SELECT DISTINCT id FROM rows WHERE dataset IN (SELECT unnest(?))", [datasets]
    ).fetchall()
    return [r[0] for r in rows]


def _hash_table(con: duckdb.DuckDBPyConnection, inputs: list[DatasetInput]) -> None:
    """Table ``hashed(id, km_size, block, split_hash_v1)`` over the chips in ``rows``."""
    sizes = {item.dataset: item.km_size for item in inputs}
    frames = []
    for km_size in sorted(set(sizes.values())):
        ids = _distinct_ids(con, [d for d, k in sizes.items() if k == km_size])
        blocks, splits = hash_splits(ids, km_size)
        frames.append(
            pd.DataFrame({"id": ids, "km_size": km_size, "block": blocks, "split_hash_v1": splits})
        )
    hashed = (
        pd.concat(frames)
        if frames
        else pd.DataFrame(columns=["id", "km_size", "block", "split_hash_v1"])
    )
    con.register("hashed_df", hashed)
    con.execute("CREATE TABLE hashed AS SELECT * FROM hashed_df")
    con.register(
        "sizes_df", pd.DataFrame({"dataset": list(sizes), "km_size": list(sizes.values())})
    )
    con.execute("CREATE TABLE sizes AS SELECT * FROM sizes_df")


def _check_left_out(con: duckdb.DuckDBPyConnection, names: set[str], drop: set[str]) -> None:
    missing = {r[0] for r in con.execute("SELECT DISTINCT dataset FROM prev").fetchall()} - names
    if missing - drop:
        raise ChipsTableError(
            f"the previous table has {', '.join(sorted(missing - drop))}, which this build "
            f"leaves out; their frozen splits would be lost (--drop them if that is intended)"
        )


def _check_gone(con: duckdb.DuckDBPyConnection, names: set[str], drop: set[str]) -> None:
    """Refuse chips of the previous table that are gone with frozen splits; note the others."""
    for dataset, gone, frozen in con.execute(
        """
            SELECT dataset, count(*), count(split_legacy) + count(split_predefined)
            FROM prev ANTI JOIN rows USING (dataset, id)
            WHERE dataset IN (SELECT unnest(?)) GROUP BY ALL ORDER BY ALL""",
        [sorted(names)],
    ).fetchall():
        if frozen and dataset not in drop:
            raise ChipsTableError(
                f"{dataset}: {gone:,} chips of the previous table are gone, with their frozen "
                f"splits (changed chip ids, or an empty or wrong chips file?); --drop {dataset} "
                f"if that is intended"
            )
        logger.info(f"{dataset}: {gone:,} chips of the previous table are gone")


def _carry_frozen(
    con: duckdb.DuckDBPyConnection, previous: str, inputs: list[DatasetInput], drop: set[str]
) -> None:
    """Fill ``split_legacy`` from the previous table, and ``split_predefined`` for published sets.

    Refuses when a dataset of the previous table is not an input (its frozen splits would be
    lost), unless ``drop`` names it. Reports chips a local rebuild no longer has.
    """
    try:
        con.execute(
            f"CREATE TABLE prev AS SELECT dataset, id, split_legacy, split_predefined "
            f"FROM read_parquet({_sql(previous)})"
        )
    except duckdb.Error as err:
        raise ChipsTableError(f"can't read the previous table {previous}: {err}") from err
    names = {item.dataset for item in inputs}
    _check_left_out(con, names, drop)
    _check_gone(con, names, drop)
    con.execute("""UPDATE rows SET split_legacy = prev.split_legacy FROM prev
                   WHERE rows.dataset = prev.dataset AND rows.id = prev.id""")
    predefined = sorted(i.dataset for i in inputs if i.split_type == "predefined" and not i.local)
    con.execute(
        """UPDATE rows SET split_predefined = prev.split_predefined FROM prev
                   WHERE rows.dataset = prev.dataset AND rows.id = prev.id
                     AND rows.dataset IN (SELECT unnest(?))""",
        [predefined],
    )


def check_hash_unchanged(con: duckdb.DuckDBPyConnection) -> None:
    """Refuse when a chip carried from the previous table gets another block or hash split."""
    moved = con.execute("""
        SELECT r.dataset, r.id FROM rows r JOIN sizes s USING (dataset)
        JOIN hashed h ON h.id = r.id AND h.km_size = s.km_size
        JOIN prev_hash p ON p.dataset = r.dataset AND p.id = r.id
        WHERE p.block IS DISTINCT FROM h.block OR p.split_hash_v1 IS DISTINCT FROM h.split_hash_v1
        LIMIT 1""").fetchone()
    if moved:
        raise ChipsTableError(
            f"{moved[0]}: chip {moved[1]} would change its block or split_hash_v1; a hash "
            f"version never moves a chip (did chip_block_ids change? then define split_hash_v2)"
        )


def _check_values(con: duckdb.DuckDBPyConnection) -> None:
    """Refuse split values outside train/val/test in the columns read from files."""
    for column in ("split_legacy", "split_predefined"):  # the hash is ours
        bad = con.execute(
            f"SELECT dataset, {column} FROM rows WHERE {column} IS NOT NULL "
            f"AND {column} NOT IN {SPLITS} LIMIT 1"
        ).fetchone()
        if bad:
            raise ChipsTableError(f"{bad[0]}: {column} has the value {bad[1]!r}")


def check_rows(con: duckdb.DuckDBPyConnection, inputs: list[DatasetInput]) -> None:
    """Refuse duplicate rows, split values outside train/val/test, and a predefined dataset
    left with no ``split_predefined`` (its file has no ``split`` and nothing was carried)."""
    dup = con.execute(
        "SELECT dataset, id FROM rows GROUP BY ALL HAVING count(*) > 1 LIMIT 1"
    ).fetchone()
    if dup:
        raise ChipsTableError(f"{dup[0]}: chip {dup[1]} appears more than once")
    _check_values(con)
    predefined = sorted(i.dataset for i in inputs if i.split_type == "predefined")
    empty = con.execute(
        """SELECT dataset FROM rows WHERE dataset IN (SELECT unnest(?))
           GROUP BY ALL HAVING count(split_predefined) = 0 ORDER BY ALL LIMIT 1""",
        [predefined],
    ).fetchone()
    if empty:
        raise ChipsTableError(
            f"{empty[0]}: the collection uses predefined splits, but no chip has one "
            f"(add it with --local from a build whose chips file has split)"
        )


_FINAL_SELECT = """
    SELECT r.dataset, r."group", r.id, h.block, {admin},
           r.license, r.license_url, r.license_category, r.presence_only,
           r.split_legacy, r.split_predefined, h.split_hash_v1,
           a.unit_admin2_v1, a.split_admin2_v1,
           r.geometry, r.bbox
    FROM rows r JOIN sizes s USING (dataset)
    JOIN hashed h ON h.id = r.id AND h.km_size = s.km_size
    JOIN admin2 a ON a.id = r.id
    ORDER BY r."group", r.dataset, r.id"""


def _assemble(
    con: duckdb.DuckDBPyConnection,
    parts: list[Path],
    inputs: list[DatasetInput],
    previous: str,
    drop: set[str],
) -> None:
    """Load the per-dataset parts, carry and check the splits, and add the derived tables."""
    files = ", ".join(_sql(str(p)) for p in parts)
    con.execute(f"CREATE TABLE rows AS SELECT * FROM read_parquet([{files}])")
    _carry_frozen(con, previous, inputs, drop)
    _hash_table(con, inputs)
    check_rows(con, inputs)
    con.execute(
        f"CREATE TABLE prev_hash AS SELECT dataset, id, block, split_hash_v1 "
        f"FROM read_parquet({_sql(previous)})"
    )
    check_hash_unchanged(con)
    _admin2_table(con, previous)


def _write(con: duckdb.DuckDBPyConnection, joined: Path, out: Path) -> int:
    import geoparquet_io as gpio

    admin = ", ".join(f"r.{c}" for c in ADMIN_COLUMNS)
    con.execute(
        f"COPY ({_FINAL_SELECT.format(admin=admin)}) TO {_sql(str(joined))} (FORMAT parquet)"
    )
    gpio.read(str(joined)).write(str(out))
    written = con.execute(f"SELECT count(*) FROM read_parquet({_sql(str(out))})").fetchone()[0]
    expected = con.execute("SELECT count(*) FROM rows").fetchone()[0]
    if written != expected:
        out.unlink()
        raise ChipsTableError(f"wrote {written:,} rows from {expected:,} chips")
    return written


def _keep_predefined(
    inputs: list[DatasetInput], previous: str, drop: frozenset[str] = frozenset()
) -> list[DatasetInput]:
    """Treat a dataset with a published ``split_predefined`` as predefined.

    A collection built with the splits stage off has no ``ftw:split_type``, so without
    this its frozen ``split_predefined`` would be dropped without a word. As predefined,
    a published one is carried from ``previous`` and a local one must have ``split``.
    One whose collection now names another split type is refused unless ``drop`` names it.
    """
    con = duckdb.connect(":memory:")
    try:
        rows = con.execute(
            f"SELECT DISTINCT dataset FROM read_parquet({_sql(previous)}) "
            f"WHERE split_predefined IS NOT NULL"
        ).fetchall()
    except duckdb.Error as err:
        raise ChipsTableError(f"can't read the previous table {previous}: {err}") from err
    finally:
        con.close()
    had = {r[0] for r in rows}
    changed = sorted(
        i.dataset
        for i in inputs
        if i.dataset in had and i.split_type not in (None, "predefined") and i.dataset not in drop
    )
    if changed:
        raise ChipsTableError(
            f"{', '.join(changed)}: the previous table has split_predefined, but the collection "
            f"now says another split type, so it would be lost (--drop it if that is intended)"
        )
    return [
        replace(i, split_type="predefined") if i.dataset in had and i.split_type is None else i
        for i in inputs
    ]


def build_chips_table(
    inputs: list[DatasetInput],
    out: Path | str,
    previous: str | Path,
    drop: Iterable[str] = (),
    workers: int = 8,
) -> int:
    """Write the table to ``out``, carrying frozen splits from ``previous``; the number of rows.

    ``previous`` (a path or URL) is the published table, which must have
    ``split_admin2_v1``. ``drop`` names datasets whose frozen splits may be lost: left
    out entirely, or chips a rebuild no longer has.
    """
    inputs = _keep_predefined(inputs, str(previous), frozenset(drop))
    _install_extensions([item.chips for item in inputs] + [str(previous)])
    out = Path(out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        parts = [Path(tmp) / f"{item.dataset}.parquet" for item in inputs]
        with ThreadPoolExecutor(workers) as pool:
            list(pool.map(_read_dataset, inputs, parts))
        con = _connect(str(previous))
        try:
            _assemble(con, parts, inputs, str(previous), set(drop))
            return _write(con, Path(tmp) / "joined.parquet", out)
        finally:
            con.close()
