"""Tests for the root chips table, from local stand-in datasets.

Two published datasets (one seeded block3x3, one predefined) that share a chip id, and a
local rebuild of the first with one extra chip and reshuffled splits.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import duckdb
import pandas as pd
import pytest

from ftw_dataset_tools.api import chips_table
from ftw_dataset_tools.api.chips_table import ChipsTableError, build_chips_table, resolve_inputs

if TYPE_CHECKING:
    from pathlib import Path

SHARED = "ftw-33TUN0000"
AA = [(SHARED, "train"), ("ftw-33TUN0002", "test"), ("ftw-33TUN0602", "val")]
BB = [(SHARED, "test"), ("ftw-33TUN2020", "val")]
STAGED_ROWS = [
    (SHARED, "val"),
    ("ftw-33TUN0002", "train"),
    ("ftw-33TUN0602", "val"),
    ("ftw-33TUN4040", "train"),
]
PRESENCE_ONLY = {"bb_2022"}


def write_chips(path: Path, chips: list[tuple[str, str]], admin: bool = True) -> None:
    """chips: (id, split); each chip a small square so the file has real geometry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial")
    rows = ", ".join(f"('{i}', '{s}', {n})" for n, (i, s) in enumerate(chips))
    admin_cols = "'AUT' AS admin0_iso3, 'Austria' AS admin0_name, " if admin else ""
    con.execute(f"""
        COPY (SELECT id, split, {admin_cols}
                     ST_MakeEnvelope(n, 0, n + 0.01, 0.01) AS geometry,
                     {{'xmin': n, 'ymin': 0, 'xmax': n + 0.01, 'ymax': 0.01}} AS bbox
              FROM (VALUES {rows}) t(id, split, n))
        TO '{path}' (FORMAT parquet)""")
    con.close()


def strip_split(source: Path, out: Path) -> None:
    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial")
    con.execute(
        f"COPY (SELECT * EXCLUDE (split) FROM read_parquet('{source}')) TO '{out}' (FORMAT parquet)"
    )
    con.close()


def rewrite_previous(source: Path, out: Path, select: str) -> None:
    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial")
    con.execute(f"COPY (SELECT {select} FROM read_parquet('{source}')) TO '{out}' (FORMAT parquet)")
    con.close()


def table(path: Path) -> dict[tuple[str, str], tuple]:
    con = duckdb.connect()
    rows = con.execute(
        "SELECT dataset, id, split_legacy, split_predefined, split_hash_v1, block "
        f"FROM read_parquet('{path}')"
    ).fetchall()
    con.close()
    return {(r[0], r[1]): r[2:] for r in rows}


def write_previous(path: Path, published: list[tuple[str, str, list, bool]]) -> None:
    """A stand-in for the published table: published is (dataset, split_type, chips, admin)."""
    rows = pd.DataFrame(
        [
            {
                "dataset": d,
                "id": i,
                "split_legacy": s,
                "split_predefined": s if split_type == "predefined" else None,
                "country": "AUT" if admin else None,
                "presence_only": d in PRESENCE_ONLY,
            }
            for d, split_type, chips, admin in published
            for i, s in chips
        ]
    )
    ids = sorted(set(rows["id"]))
    blocks, hashes = chips_table.hash_splits(ids, 2.0)
    rows = rows.merge(pd.DataFrame({"id": ids, "block": blocks, "split_hash_v1": hashes}), on="id")
    locations = (
        rows.groupby("id")
        .agg(
            country=("country", lambda s: s.dropna().min() if s.notna().any() else None),
            block=("block", "first"),
            presence_only=("presence_only", "any"),
        )
        .reset_index()
    )
    locations["admin1"] = locations["admin2"] = None
    admin2 = chips_table.admin2_splits(locations).rename(
        columns={"unit": "unit_admin2_v1", "split": "split_admin2_v1"}
    )
    rows = rows.merge(admin2, on="id").drop(columns=["country", "presence_only"])
    con = duckdb.connect()
    con.register("rows_df", rows)
    con.execute(f"COPY (SELECT * FROM rows_df) TO '{path}' (FORMAT parquet)")
    con.close()


class World:
    """A catalog tree, the bucket it publishes to, and local builds next to it."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.catalog = root / "catalog"
        self.base = root / "bucket"
        self.staging = root / "staging"

    def write_catalog(self, datasets: dict[str, tuple[str, str | None]]) -> None:
        """datasets: {id: (group, split_type)}; the linked catalog tree."""
        groups: dict[str, list[str]] = {}
        for dataset_id, (group, split_type) in datasets.items():
            groups.setdefault(group, []).append(dataset_id)
            self.write_collection(self.catalog / group / dataset_id, dataset_id, split_type)
        for group, ids in groups.items():
            links = [{"rel": "child", "href": f"./{d}/collection.json"} for d in ids]
            (self.catalog / group / "catalog.json").write_text(json.dumps({"links": links}))
        links = [{"rel": "child", "href": f"./{g}/catalog.json"} for g in groups]
        (self.catalog / "catalog.json").write_text(json.dumps({"links": links}))

    @staticmethod
    def write_collection(folder: Path, dataset_id: str, split_type: str | None) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        collection = {
            "id": dataset_id,
            "ftw:chip_size_km": 2.0,
            "license": "other",
            "links": [
                {
                    "rel": "license",
                    "href": "https://data.source.coop/olmoearth/"
                    "olmoearth-v1-3-embeddings/LICENSE.txt",
                },
                {"rel": "license", "href": "https://go.planet.com/nicfi-pla-2024"},
            ],
            "ftw:presence_only": dataset_id in PRESENCE_ONLY,
            "assets": {"chips": {"href": f"./{dataset_id}_chips.parquet"}},
        }
        if split_type:
            collection["ftw:split_type"] = split_type
        (folder / "collection.json").write_text(json.dumps(collection))

    def publish_chips(self, group: str, dataset: str, chips: list, admin: bool = True) -> None:
        write_chips(self.base / group / dataset / f"{dataset}_chips.parquet", chips, admin)

    def local(
        self, dataset: str, chips: list | None, split_type: str | None = "block3x3", **kw
    ) -> Path:
        """A directory as ``ftwd run`` writes it: collection.json and the chips file."""
        folder = self.staging / dataset
        folder.mkdir(parents=True, exist_ok=True)
        collection = {
            "id": dataset,
            "ftw:chip_size_km": 2.0,
            "license": "CC-BY-4.0",
            "ftw:presence_only": False,
            "assets": {"chips": {"href": f"./{dataset}_chips.parquet"}},
        }
        if split_type:
            collection["ftw:split_type"] = split_type
        (folder / "collection.json").write_text(json.dumps(collection))
        if chips is not None:
            write_chips(folder / f"{dataset}_chips.parquet", chips, **kw)
        return folder

    def inputs(self, local: tuple[str, ...] = (), groups: dict | None = None) -> list:
        return resolve_inputs(
            self.catalog / "catalog.json",
            str(self.base),
            [self.staging / d for d in local],
            groups,
        )

    def build(self, previous: Path, local: tuple[str, ...] = (), drop=(), name="x") -> Path:
        out = self.root / f"{name}.parquet"
        build_chips_table(self.inputs(local), out, previous, drop=drop, workers=2)
        return out


@pytest.fixture
def world(tmp_path: Path) -> World:
    """Published aa_2021 (block3x3) and bb_2022 (predefined, presence-only), and the
    table frozen from them (``boot.parquet``)."""
    w = World(tmp_path)
    w.write_catalog({"aa_2021": ("zz", "block3x3"), "bb_2022": ("bb", "predefined")})
    w.publish_chips("zz", "aa_2021", AA)
    w.publish_chips("bb", "bb_2022", BB, admin=False)
    write_previous(
        tmp_path / "boot.parquet",
        [("aa_2021", "block3x3", AA, True), ("bb_2022", "predefined", BB, False)],
    )
    return w


@pytest.fixture
def later(world: World) -> Path:
    """A build with a local rebuild of aa_2021 (one new chip, reshuffled splits) over
    published files that now say something else."""
    world.local("aa_2021", STAGED_ROWS)
    world.publish_chips("bb", "bb_2022", [(SHARED, "train"), ("ftw-33TUN2020", "train")], False)
    return world.build(world.root / "boot.parquet", local=("aa_2021",), name="later")


# --------------------------------------------------------------------------- #
# A build
# --------------------------------------------------------------------------- #


def test_build_carries_frozen_splits(later: Path) -> None:
    t = table(later)
    assert len(t) == 6, "one row per dataset chip, with the new chip"
    # split_legacy comes from the previous table, not the reshuffled files
    assert t[("aa_2021", SHARED)][0] == "train"
    assert t[("aa_2021", "ftw-33TUN0002")][0] == "test"
    assert t[("bb_2022", SHARED)][0] == "test"
    assert t[("aa_2021", "ftw-33TUN4040")][0] is None, "a chip added later has null legacy"
    assert all(v[1] is None for k, v in t.items() if k[0] == "aa_2021"), (
        "split_predefined is null for a block3x3 dataset"
    )
    assert t[("bb_2022", "ftw-33TUN2020")][1] == "val", (
        "split_predefined of a published dataset is carried over"
    )
    assert t[("aa_2021", SHARED)][2] == t[("bb_2022", SHARED)][2], (
        "a shared location has one split_hash_v1 in every dataset"
    )
    assert all(v[2] in chips_table.SPLITS and v[3] for v in t.values())


def test_build_returns_the_row_count(world: World) -> None:
    world.local("aa_2021", STAGED_ROWS)
    rows = build_chips_table(
        world.inputs(("aa_2021",)), world.root / "n.parquet", world.root / "boot.parquet"
    )
    assert rows == 6


def test_build_columns_order_and_geoparquet(later: Path) -> None:
    con = duckdb.connect()
    columns = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM '{later}'").fetchall()]
    order = con.execute(f"SELECT \"group\", dataset, id FROM read_parquet('{later}')").fetchall()
    meta = con.execute(f"SELECT decode(key) FROM parquet_kv_metadata('{later}')").fetchall()
    con.close()
    assert columns == [
        "dataset",
        "group",
        "id",
        "block",
        "admin0_iso3",
        "admin0_name",
        "admin1_name",
        "admin2_name",
        "license",
        "license_url",
        "license_category",
        "presence_only",
        "split_legacy",
        "split_predefined",
        "split_hash_v1",
        "unit_admin2_v1",
        "split_admin2_v1",
        "geometry",
        "bbox",
    ]
    assert order == sorted(order) and order[0][1] == "bb_2022", (
        "rows are sorted by group (bb before zz), then dataset and id"
    )
    assert ("geo",) in meta, "the table is GeoParquet"


def test_build_license_presence_only_and_admin2(later: Path) -> None:
    con = duckdb.connect()
    extra = {
        r[0]: r[1:]
        for r in con.execute(f"""
        SELECT dataset || ' ' || id, license, license_url, license_category, presence_only,
               unit_admin2_v1, split_admin2_v1 FROM read_parquet('{later}')""").fetchall()
    }
    con.close()
    # license, its own link (not the embeddings'), its category and presence_only
    assert extra[f"bb_2022 {SHARED}"][:4] == (
        "other",
        "https://go.planet.com/nicfi-pla-2024",
        "non-commercial",
        True,
    )
    assert extra[f"aa_2021 {SHARED}"][:4] == (
        "CC-BY-4.0",
        "https://spdx.org/licenses/CC-BY-4.0.html",
        "attribution",
        False,
    ), "a local dataset's license comes from its own collection"
    # a location in a presence-only dataset is train in every dataset
    assert extra[f"aa_2021 {SHARED}"][4:] == extra[f"bb_2022 {SHARED}"][4:]
    assert extra[f"aa_2021 {SHARED}"][5] == "train"
    assert all(v[5] in chips_table.SPLITS and v[4] for v in extra.values())


def test_restaged_predefined_takes_new_splits(world: World, later: Path) -> None:
    world.local("bb_2022", [(SHARED, "val"), ("ftw-33TUN2020", "test")], "predefined", admin=False)
    t = table(world.build(later, local=("aa_2021", "bb_2022")))
    assert t[("bb_2022", "ftw-33TUN2020")][1] == "test"
    assert t[("bb_2022", SHARED)][1] == "val"
    assert t[("bb_2022", SHARED)][0] == "test", "its legacy split stays frozen"


def test_published_predefined_without_split_type_keeps_its_split(world: World, later: Path) -> None:
    """A predefined dataset republished with the splits stage off has no ftw:split_type;
    its split_predefined is still carried from the previous table."""
    world.write_catalog({"aa_2021": ("zz", "block3x3"), "bb_2022": ("bb", None)})
    t = table(world.build(later, local=("aa_2021",)))
    assert t[("bb_2022", "ftw-33TUN2020")][1] == "val"


def test_predefined_switched_to_another_split_type_is_refused(world: World, later: Path) -> None:
    """A dataset with split_predefined whose collection now names another split type
    would lose that column; it is refused unless --drop names it."""
    world.write_catalog({"aa_2021": ("zz", "block3x3"), "bb_2022": ("bb", "block3x3-hash")})
    with pytest.raises(ChipsTableError, match="bb_2022: the previous table has split_predefined"):
        world.build(later, local=("aa_2021",))
    t = table(world.build(later, local=("aa_2021",), drop={"bb_2022"}))
    assert t[("bb_2022", "ftw-33TUN2020")][1] is None


def test_local_predefined_without_split_type_needs_a_split(world: World, later: Path) -> None:
    world.local("bb_2022", None, None)
    strip_split(
        world.base / "bb/bb_2022/bb_2022_chips.parquet",
        world.staging / "bb_2022" / "bb_2022_chips.parquet",
    )
    with pytest.raises(ChipsTableError, match="bb_2022: the collection uses predefined"):
        world.build(later, local=("aa_2021", "bb_2022"))


def test_drop_leaves_a_dataset_out(world: World) -> None:
    world.write_catalog({"aa_2021": ("zz", "block3x3")})
    out = world.build(world.root / "boot.parquet", drop={"bb_2022"})
    assert {k[0] for k in table(out)} == {"aa_2021"}


def test_gone_chip_without_frozen_split_is_a_note(
    world: World, later: Path, caplog: pytest.LogCaptureFixture
) -> None:
    world.local("aa_2021", STAGED_ROWS[:3])
    with caplog.at_level(logging.INFO):
        world.build(later, local=("aa_2021",), drop={"bb_2022"})
    assert "aa_2021: 1 chips of the previous table are gone" in caplog.text


def test_drop_allows_losing_frozen_chips(world: World, later: Path) -> None:
    world.local("aa_2021", STAGED_ROWS[1:])
    out = world.build(later, local=("aa_2021",), drop={"bb_2022", "aa_2021"})
    assert (("aa_2021", SHARED)) not in table(out)


# --------------------------------------------------------------------------- #
# Refusals
# --------------------------------------------------------------------------- #


def test_refuses_unreadable_previous(world: World) -> None:
    with pytest.raises(ChipsTableError, match="can't read the previous table"):
        world.build(world.root / "nope.parquet")


def test_refuses_previous_without_admin2(world: World) -> None:
    old = world.root / "old.parquet"
    rewrite_previous(
        world.root / "boot.parquet", old, "* EXCLUDE (unit_admin2_v1, split_admin2_v1)"
    )
    world.local("aa_2021", STAGED_ROWS)
    with pytest.raises(ChipsTableError, match="has no split_admin2_v1"):
        world.build(old, local=("aa_2021",))


def test_refuses_local_without_chips_file(world: World) -> None:
    world.local("bb_2022", None, "predefined")
    with pytest.raises(ChipsTableError, match="does not exist"):
        world.inputs(("bb_2022",))


def test_refuses_local_without_collection(world: World) -> None:
    folder = world.local("aa_2021", STAGED_ROWS)
    (folder / "collection.json").unlink()
    with pytest.raises(ChipsTableError, match="needs its license"):
        world.inputs(("aa_2021",))


def test_refuses_leaving_out_a_dataset(world: World) -> None:
    world.write_catalog({"aa_2021": ("zz", "block3x3")})
    with pytest.raises(ChipsTableError, match="leaves out"):
        world.build(world.root / "boot.parquet")


def test_refuses_losing_a_frozen_chip(world: World, later: Path) -> None:
    world.local("aa_2021", STAGED_ROWS[1:])
    with pytest.raises(
        ChipsTableError, match="1 chips of the previous table are gone, with their frozen splits"
    ):
        world.build(later, local=("aa_2021",), drop={"bb_2022"})


@pytest.mark.parametrize(
    "select",
    [
        "* REPLACE (CASE WHEN split_hash_v1 = 'train' THEN 'test' ELSE 'train' END "
        "AS split_hash_v1)",
        "* REPLACE (block || '_x' AS block)",
    ],
    ids=["hash", "block"],
)
def test_refuses_a_moved_chip(world: World, later: Path, select: str) -> None:
    moved = world.root / "moved.parquet"
    rewrite_previous(later, moved, select)
    with pytest.raises(ChipsTableError, match="would change its block or split_hash_v1"):
        world.build(moved, local=("aa_2021",), drop={"bb_2022"})


def test_refuses_local_predefined_without_split(world: World, later: Path) -> None:
    world.write_catalog({"aa_2021": ("zz", "predefined"), "bb_2022": ("bb", "predefined")})
    folder = world.local("aa_2021", None, "predefined")
    strip_split(world.base / "zz/aa_2021/aa_2021_chips.parquet", folder / "aa_2021_chips.parquet")
    with pytest.raises(ChipsTableError, match="has no split column"):
        world.build(later, local=("aa_2021",))


def test_refuses_new_published_predefined_without_split(world: World, later: Path) -> None:
    world.write_catalog(
        {
            "aa_2021": ("zz", "block3x3"),
            "bb_2022": ("bb", "predefined"),
            "cc_2023": ("cc", "predefined"),
        }
    )
    (world.base / "cc/cc_2023").mkdir(parents=True)
    strip_split(
        world.base / "zz/aa_2021/aa_2021_chips.parquet",
        world.base / "cc/cc_2023/cc_2023_chips.parquet",
    )
    with pytest.raises(ChipsTableError, match="cc_2023: the collection uses predefined splits"):
        world.build(later, local=("aa_2021",))


def test_refuses_a_dataset_published_twice(world: World) -> None:
    World.write_collection(world.catalog / "bb" / "aa_2021", "aa_2021", "block3x3")
    links = [
        {"rel": "child", "href": "./bb_2022/collection.json"},
        {"rel": "child", "href": "./aa_2021/collection.json"},
    ]
    (world.catalog / "bb" / "catalog.json").write_text(json.dumps({"links": links}))
    with pytest.raises(ChipsTableError, match="published twice"):
        world.inputs()


def test_refuses_a_duplicate_chip(world: World, later: Path) -> None:
    world.local("aa_2021", [(SHARED, "train"), (SHARED, "val")])
    with pytest.raises(ChipsTableError, match="more than once"):
        world.build(later, local=("aa_2021",), drop={"aa_2021"})


def test_refuses_a_bad_split_value(world: World, later: Path) -> None:
    world.local("aa_2021", [(SHARED, "training")], "predefined")
    with pytest.raises(ChipsTableError, match="has the value"):
        world.build(later, local=("aa_2021",), drop={"aa_2021"})


def test_refuses_an_unreadable_chips_file(world: World, later: Path) -> None:
    (world.base / "bb/bb_2022/bb_2022_chips.parquet").write_text("not parquet")
    with pytest.raises(ChipsTableError, match="bb_2022"):
        world.build(later, local=("aa_2021",))


# --------------------------------------------------------------------------- #
# Input resolution
# --------------------------------------------------------------------------- #


def test_inputs_from_the_catalog(world: World) -> None:
    inputs = {i.dataset: i for i in world.inputs()}
    assert set(inputs) == {"aa_2021", "bb_2022"}
    aa = inputs["aa_2021"]
    assert aa.group == "zz" and aa.split_type == "block3x3" and not aa.local
    assert aa.chips == f"{world.base}/zz/aa_2021/aa_2021_chips.parquet"
    assert inputs["bb_2022"].presence_only and inputs["bb_2022"].split_type == "predefined"


def test_relative_chips_resolve_against_the_data_base_url(world: World) -> None:
    inputs = resolve_inputs(
        world.catalog / "catalog.json", "https://data.example.org/bench/", (), None
    )
    assert inputs[0].chips == "https://data.example.org/bench/zz/aa_2021/aa_2021_chips.parquet"


def test_relative_catalog_path(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(world.catalog)
    inputs = resolve_inputs("catalog.json")
    assert inputs[0].chips == f"{world.catalog.resolve()}/zz/aa_2021/aa_2021_chips.parquet"


def test_invalid_catalog_json(world: World) -> None:
    (world.catalog / "zz" / "catalog.json").write_text("{not json")
    with pytest.raises(ChipsTableError, match="is not valid JSON"):
        world.inputs()


def test_data_base_url_defaults_to_the_catalog_folder(world: World) -> None:
    inputs = resolve_inputs(world.catalog / "catalog.json")
    assert inputs[0].chips == f"{world.catalog.resolve()}/zz/aa_2021/aa_2021_chips.parquet"


def test_absolute_chips_href_is_kept(world: World) -> None:
    path = world.catalog / "zz/aa_2021/collection.json"
    collection = json.loads(path.read_text())
    collection["assets"]["chips"]["href"] = "s3://bucket/aa.parquet"
    path.write_text(json.dumps(collection))
    assert world.inputs()[0].chips == "s3://bucket/aa.parquet"


def test_local_overrides_the_published_dataset(world: World) -> None:
    folder = world.local("aa_2021", STAGED_ROWS)
    aa = next(i for i in world.inputs(("aa_2021",)) if i.dataset == "aa_2021")
    assert aa.local and aa.group == "zz"
    assert aa.chips == str(folder / "aa_2021_chips.parquet")


def test_new_local_dataset_needs_a_group(world: World) -> None:
    world.local("dd_2024", [("ftw-33TUN8080", "train")])
    with pytest.raises(ChipsTableError, match="--group dd_2024=<group>"):
        world.inputs(("dd_2024",))


def test_new_local_dataset_with_a_group(world: World, later: Path) -> None:
    world.local("dd_2024", [("ftw-33TUN8080", "train")])
    inputs = world.inputs(("aa_2021", "dd_2024"), {"dd_2024": "dd"})
    assert next(i for i in inputs if i.dataset == "dd_2024").group == "dd"
    out = world.root / "dd.parquet"
    build_chips_table(inputs, out, later, workers=2)
    t = table(out)
    assert t[("dd_2024", "ftw-33TUN8080")][0] is None, "a new dataset has no legacy split"


def test_group_must_match_the_catalog(world: World) -> None:
    world.local("aa_2021", STAGED_ROWS)
    with pytest.raises(ChipsTableError, match="the catalog has it in zz"):
        world.inputs(("aa_2021",), {"aa_2021": "other"})


def test_group_for_a_dataset_that_is_not_local(world: World) -> None:
    with pytest.raises(ChipsTableError, match="no --local dataset has"):
        world.inputs((), {"ee_2020": "ee"})


def test_published_datasets_walks_groups(world: World) -> None:
    assert chips_table.published_datasets(str(world.catalog / "catalog.json")) == [
        "bb/bb_2022",
        "zz/aa_2021",
    ]


# --------------------------------------------------------------------------- #
# License categories
# --------------------------------------------------------------------------- #


def test_license_categories() -> None:
    cat = chips_table.license_category
    assert cat("CC-BY-4.0", None) == "attribution"
    assert cat("CC0-1.0", None) == "public-domain"
    assert cat("CC-BY-SA-4.0", None) == "share-alike"
    assert cat("other", "https://go.planet.com/nicfi-pla-2024") == "non-commercial"
    assert cat("other", "https://example.com/terms") == "unknown"
    assert cat(None, None) == "unknown"
    assert cat("MIT-ish", None) == "unknown"
    assert set(chips_table.LICENSES.values()) <= set(chips_table.LICENSE_CATEGORIES)


def test_license_links() -> None:
    fields = chips_table.license_fields(
        {
            "license": "CC-BY-4.0",
            "links": [
                {
                    "rel": "license",
                    "href": "https://data.source.coop/olmoearth/olmoearth-v1-3-embeddings/"
                    "LICENSE.txt",
                }
            ],
        }
    )
    assert fields["license_url"] == "https://spdx.org/licenses/CC-BY-4.0.html"
    assert chips_table.license_fields({"license": "other", "links": []})["license_url"] is None


# --------------------------------------------------------------------------- #
# admin2 splits, on stand-in locations
# --------------------------------------------------------------------------- #


def locs(rows: list[tuple]) -> pd.DataFrame:
    """rows: (id, country, admin1, admin2, presence_only); block = id's first 4 chars."""
    return pd.DataFrame(
        [
            {"id": i, "country": c, "admin1": a1, "admin2": a2, "block": i[:4], "presence_only": po}
            for i, c, a1, a2, po in rows
        ]
    )


def splits_of(out: pd.DataFrame) -> dict:
    return dict(zip(out["id"], out["split"], strict=True))


BIG = [(f"a{u:02d}{k:03d}", "AAA", "r", f"u{u}", False) for u in range(20) for k in range(5 + u)]
FEW = [
    ("b1x01", "BBB", "r", "u1", False),
    ("b2x01", "BBB", "r", "u2", False),
    ("b3x01", "BBB", "r", "u2", False),
    ("zzzz1", None, None, None, False),
    ("yyyy1", "CCC", None, None, False),
]


def test_admin2_balanced_by_country() -> None:
    out = chips_table.admin2_splits(locs(BIG))
    assert (out.groupby("unit")["split"].nunique() == 1).all(), "a unit has one split"
    share = out["split"].value_counts(normalize=True)
    assert set(share.index) == set(chips_table.SPLITS)
    assert abs(share["train"] - 0.8) < 0.08
    assert splits_of(chips_table.admin2_splits(locs(BIG[::-1]))) == splits_of(out), (
        "the result does not depend on row order"
    )


def test_admin2_presence_only_unit_is_train() -> None:
    po = [
        *BIG,
        ("p0001", "AAA", "r", "u0", True),
        ("q0001", "AAA", "r", "pq", True),
        ("q0002", "AAA", "r", "pq", False),
    ]
    out = splits_of(chips_table.admin2_splits(locs(po)))
    assert all(out[i] == "train" for i in ("p0001", "q0001", "q0002"))
    assert all(out[r[0]] == "train" for r in BIG if r[3] == "u0")


def test_admin2_block_units() -> None:
    out = chips_table.admin2_splits(locs(FEW)).set_index("id")["unit"]
    assert out["b1x01"] == "BBB|block:b1x0"
    assert out["zzzz1"] == "|block:zzzz"
    assert out["yyyy1"] == "CCC|block:yyyy"
    mixed = [*FEW, ("zzzz2", None, None, None, True)]
    s = splits_of(chips_table.admin2_splits(locs(mixed)))
    assert s["zzzz1"] == s["zzzz2"] == "train", "a block unit with a presence-only chip is train"


def test_admin2_three_units_give_all_splits() -> None:
    lopsided = [
        (f"k{u}{k:04d}", "KKK", "r", f"u{u}", False)
        for u, n in ((0, 300), (1, 5), (2, 3))
        for k in range(n)
    ]
    out = chips_table.admin2_splits(locs(lopsided))
    per = out.groupby("unit")["split"].first().to_dict()
    assert per["KKK|r|u0"] == "train"
    assert {per["KKK|r|u1"], per["KKK|r|u2"]} == {"val", "test"}


def test_admin2_country_keeps_admin2_units() -> None:
    xx = [(f"x{u}{k:03d}", "XXX", "r", f"u{u}", False) for u in range(3) for k in range(10)]
    first = chips_table.admin2_splits(locs(xx))
    per = dict(zip(first["unit"], first["split"], strict=True))
    train_unit = next(u for u, s in per.items() if s == "train").split("|")[2]
    val_unit = next(u for u, s in per.items() if s == "val").split("|")[2]
    grown = [*xx, ("pzzz1", "XXX", "r", train_unit, True), ("nnnn1", "XXX", "r", val_unit, False)]
    out = chips_table.admin2_splits(locs(grown), first).set_index("id")
    assert out.loc["nnnn1", "split"] == "val"
    assert "|block:" not in out.loc["nnnn1", "unit"]


def test_admin2_country_keeps_block_units() -> None:
    yy = [
        ("y1aa1", "YYY", "r", "a", False),
        ("y2bb1", "YYY", "r", "b", False),
        ("y3cc1", "YYY", "r", "b", False),
    ]
    first = chips_table.admin2_splits(locs(yy))
    grown = [*yy, ("y3cc2", "YYY", "r", "c", False), ("y4dd1", "YYY", "r", "d", False)]
    out = chips_table.admin2_splits(locs(grown), first).set_index("id")
    assert out.loc["y3cc2", "unit"] == "YYY|block:y3cc"
    assert out.loc["y3cc2", "split"] == splits_of(first)["y3cc1"]


def test_admin2_new_units_fill_missing_splits() -> None:
    only_train = pd.DataFrame(
        {"id": [r[0] for r in BIG], "unit": [f"AAA|r|{r[3]}" for r in BIG], "split": "train"}
    )
    grown = [
        *BIG,
        *[(f"m{u}{k:03d}", "AAA", "r", f"more{u}", False) for u in range(4) for k in range(3)],
    ]
    out = chips_table.admin2_splits(locs(grown), only_train)
    assert {"val", "test"} <= set(out[out["id"].str.startswith("m")]["split"])


def test_admin2_carried_chips_never_move() -> None:
    first = chips_table.admin2_splits(locs(BIG))
    grown = [
        *BIG,
        ("a00999", "AAA", "r", "u0", False),
        *[(f"n{u}{k:03d}", "AAA", "r", f"new{u}", False) for u in range(6) for k in range(8)],
    ]
    s1, s2 = splits_of(first), splits_of(chips_table.admin2_splits(locs(grown), first))
    assert all(s2[i] == s1[i] for i in s1), "a later build moves no carried chip"
    assert s2["a00999"] == s1["a00000"], "a new chip of a known unit takes that unit's split"
    assert all(s2[i] in chips_table.SPLITS for i in s2)


def test_admin2_carried_unit_survives_a_rename() -> None:
    first = chips_table.admin2_splits(locs(BIG))
    renamed = first.copy()
    renamed.loc[renamed["id"] == "a00000", "unit"] = "AAA|r|old-name"
    out = chips_table.admin2_splits(locs(BIG), renamed).set_index("id")
    assert out.loc["a00000", "unit"] == "AAA|r|old-name"


def test_admin2_refuses_presence_only_in_a_frozen_val_unit() -> None:
    first = chips_table.admin2_splits(locs(BIG))
    frozen = first[first["split"] == "val"]["unit"].iloc[0]
    name = frozen.split("|")[2]
    with pytest.raises(ChipsTableError, match=frozen.replace("|", r"\|")):
        chips_table.admin2_splits(locs([*BIG, ("x0001", "AAA", "r", name, True)]), first)


def test_build_leaves_no_temp_files(world: World, later: Path) -> None:
    assert later.exists()
    leftovers = [p for p in world.root.iterdir() if p.name.startswith("tmp")]
    assert not leftovers


def test_local_reads_never_load_httpfs(monkeypatch: pytest.MonkeyPatch) -> None:
    # threads reading local files used to race to INSTALL httpfs on a fresh machine
    calls: list[str] = []
    monkeypatch.setattr(chips_table, "configure_source_coop_s3", lambda _con: calls.append("s3"))
    chips_table._connect("/data/aa_2021_chips.parquet").close()
    chips_table._connect("https://data.example.org/aa_2021_chips.parquet").close()
    assert calls == []
    chips_table._connect("s3://bucket/aa_2021_chips.parquet").close()
    assert calls == ["s3"]


def test_extensions_are_installed_once_before_the_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    run: list[str] = []

    class Con:
        def execute(self, sql: str) -> None:
            run.append(sql)

        def close(self) -> None:
            pass

    monkeypatch.setattr(chips_table.duckdb, "connect", lambda *_a: Con())
    chips_table._install_extensions(["/data/a.parquet"])
    assert run == ["INSTALL spatial"]
    run.clear()
    chips_table._install_extensions(["/data/a.parquet", "https://x.org/b.parquet"])
    assert run == ["INSTALL spatial", "INSTALL httpfs"]
