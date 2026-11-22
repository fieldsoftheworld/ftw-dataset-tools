"""Tests for the root chips tiles: one feature per chip location, from a small stand-in table."""

from __future__ import annotations

import io
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import duckdb
import pytest
from PIL import Image
from pmtiles.tile import Compression, TileType, zxy_to_tileid
from pmtiles.writer import Writer

from ftw_dataset_tools.api import chips_tiles
from ftw_dataset_tools.api.chips_tiles import ChipsTilesError
from ftw_dataset_tools.api.tiles import tippecanoe_available

REPO_ROOT = Path(__file__).resolve().parents[2]


def write_table(path: Path, rows: list[tuple], size: float = 0.01) -> None:
    """rows: (dataset, id, split_legacy, split_hash_v1, x[, split_admin2_v1]) with a square of
    ``size`` degrees at x; the admin2 split defaults to the hash split."""
    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial")

    def admin2(row: tuple) -> str | None:
        return row[3] if len(row) < 6 else row[5]

    values = ", ".join(
        f"('{r[0]}', '{r[1]}', '{r[2]}', '{r[3]}', {r[4]}, "
        f"{'NULL' if admin2(r) is None else repr(admin2(r))}, {str(r[0] == 'po').lower()})"
        for r in rows
    )
    con.execute(f"""
        COPY (SELECT dataset, id, id || '_b' AS block, 'Austria' AS admin0_name,
                     split_legacy, split_hash_v1, split_admin2_v1::VARCHAR AS split_admin2_v1,
                     presence_only, ST_MakeEnvelope(x, 0, x + {size}, {size}) AS geometry
              FROM (VALUES {values}) t(dataset, id, split_legacy, split_hash_v1, x,
                                       split_admin2_v1, presence_only))
        TO '{path}' (FORMAT parquet)""")
    con.close()


# each admin2 split differs from the hash split, so the tiles must read the right one;
# c2 is also in a presence-only dataset ("po")
ROWS = [
    ("aa_2021", "c1", "train", "val", 0, "test"),
    ("aa_2022", "c1", "test", "val", 0, "test"),
    ("aa_2021", "c2", "train", "train", 1, "val"),
    ("aa_2022", "c2", "train", "train", 1, "val"),
    ("po", "c2", "train", "train", 1, "val"),
    ("bb_2020", "c3", "val", "test", 2, "train"),
]


@pytest.fixture
def table(tmp_path: Path) -> Path:
    path = tmp_path / "chips.parquet"
    write_table(path, ROWS)
    return path


@pytest.fixture
def locations(tmp_path: Path, table: Path) -> Path:
    out = tmp_path / "locations.parquet"
    assert chips_tiles.write_locations(str(table), out) == 3
    return out


def test_one_location_per_chip(locations: Path) -> None:
    con = duckdb.connect()
    rows = {
        r[0]: r[1:]
        for r in con.execute(
            "SELECT id, datasets, legacy_conflict, split_hash_v1, split_admin2_v1, presence_only "
            f"FROM read_parquet('{locations}')"
        ).fetchall()
    }
    meta = con.execute(f"SELECT decode(key) FROM parquet_kv_metadata('{locations}')").fetchall()
    columns = {
        r[0] for r in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{locations}')").fetchall()
    }
    con.close()
    assert set(rows) == {"c1", "c2", "c3"}
    assert rows["c1"] == (2, True, "val", "test", False), "2 datasets whose legacy splits differ"
    assert rows["c2"] == (3, False, "train", "val", True), "one of its datasets is presence-only"
    assert rows["c3"] == (1, False, "test", "train", False)
    assert ("geo",) in meta, "the locations file is GeoParquet"
    assert set(chips_tiles.ATTRIBUTES) <= columns, "every tiled attribute is a column"


def test_overview_points_sum_every_count(tmp_path: Path, locations: Path) -> None:
    expected = {
        "train": {"chips": 1, "conflicts": 0, "datasets": 1},  # c3
        "val": {"chips": 1, "conflicts": 0, "datasets": 3},  # c2
        "test": {"chips": 1, "conflicts": 1, "datasets": 2},  # c1
    }
    for split in chips_tiles.SPLITS:
        seq = tmp_path / f"overview_{split}.geojsonseq"
        n = chips_tiles.write_overview(locations, seq, split)
        points = [json.loads(line) for line in seq.read_text().splitlines() if line.strip()]
        sums = {k: sum(p["properties"][k] for p in points) for k in chips_tiles.OVERVIEW_SUMS}
        assert n == len(points) and sums == expected[split]
        assert all(p["geometry"]["type"] == "Point" for p in points)


def test_overview_read_error_is_a_domain_error(tmp_path: Path) -> None:
    with pytest.raises(chips_tiles.ChipsTilesError, match="can't write the val overview"):
        chips_tiles.write_overview(tmp_path / "missing.parquet", tmp_path / "o.seq", "val")


def test_overview_command(tmp_path: Path) -> None:
    cmd = chips_tiles.overview_command(tmp_path / "s", tmp_path / "o.pmtiles", "overview_test")
    assert all(f"--accumulate-attribute={k}:sum" in cmd for k in chips_tiles.OVERVIEW_SUMS)
    assert "-r1" in cmd and "--cluster-densest-as-needed" in cmd
    assert cmd[cmd.index("-l") + 1] == "overview_test"


@pytest.mark.parametrize(
    ("rows", "size"),
    [
        ([("aa_2021", "c1", "train", "val", 0), ("aa_2022", "c1", "train", "test", 0)], 0.01),
        ([("aa_2021", "c1", "train", "val", 0), ("aa_2022", "c1", "train", "val", 5)], 0.01),
        ([("aa_2021", "c1", "train", "val", 0)], 0.001),
        (
            [
                ("aa_2021", "c1", "train", "val", 0, "val"),
                ("aa_2022", "c1", "train", "val", 0, "test"),
            ],
            0.01,
        ),
        ([("aa_2021", "c1", "train", "val", 0, None)], 0.01),
    ],
    ids=["two hash splits", "two footprints", "tiny footprint", "two admin2 splits", "no admin2"],
)
def test_refuses_a_bad_location(tmp_path: Path, rows: list, size: float) -> None:
    bad = tmp_path / "bad.parquet"
    write_table(bad, rows, size=size)
    with pytest.raises(ChipsTilesError, match="c1"):
        chips_tiles.write_locations(str(bad), tmp_path / "x.parquet")


def test_refuses_a_table_without_required_columns(tmp_path: Path, table: Path) -> None:
    old = tmp_path / "old.parquet"
    con = duckdb.connect()
    con.execute(
        f"COPY (SELECT * EXCLUDE (split_admin2_v1, presence_only) FROM read_parquet('{table}')) "
        f"TO '{old}' (FORMAT parquet)"
    )
    con.close()
    with pytest.raises(ChipsTilesError, match="split_admin2_v1, presence_only"):
        chips_tiles.write_locations(str(old), tmp_path / "x.parquet")


# --------------------------------------------------------------------------- #
# The thumbnail, against a stand-in chiitiler that returns a fixed PNG
# --------------------------------------------------------------------------- #


def _png(blank: bool) -> bytes:
    im = Image.new("RGB", (60, 40), (244, 244, 242))
    if not blank:
        im.putpixel((5, 5), (215, 48, 31))
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture
def chiitiler():
    seen: dict = {"body": b""}

    class Fake(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            seen["style"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen["path"] = self.path
            self.send_response(200)
            self.end_headers()
            self.wfile.write(seen["body"])

        def log_message(self, *args) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    seen["url"] = f"http://127.0.0.1:{server.server_port}"
    yield seen
    server.shutdown()


@pytest.fixture
def style(tmp_path: Path) -> Path:
    path = tmp_path / "chips.style.json"
    path.write_text(json.dumps({"version": 8, "sources": {"data": {}}, "layers": []}))
    return path


def test_render_thumbnail(tmp_path: Path, chiitiler: dict, style: Path) -> None:
    chiitiler["body"] = _png(blank=False)
    thumb = chips_tiles.render_thumbnail(
        tmp_path / "t.webp",
        style=style,
        tiles_url="https://x/chips.pmtiles",
        server=chiitiler["url"],
    )
    source = chiitiler["style"]["style"]["sources"]["data"]
    assert Image.open(thumb).format == "WEBP"
    assert source["tiles"] == ["pmtiles://https://x/chips.pmtiles/{z}/{x}/{y}"]
    assert source["minzoom"] == chips_tiles.OVERVIEW_ZOOMS[0]
    assert chiitiler["path"].startswith("/clip.png?bbox=")


def test_render_thumbnail_refuses_blank(tmp_path: Path, chiitiler: dict, style: Path) -> None:
    chiitiler["body"] = _png(blank=True)
    with pytest.raises(ChipsTilesError, match="blank"):
        chips_tiles.render_thumbnail(
            tmp_path / "b.webp",
            style=style,
            tiles_url="https://x/c.pmtiles",
            server=chiitiler["url"],
        )


def test_render_thumbnail_without_server(tmp_path: Path, style: Path) -> None:
    with pytest.raises(ChipsTilesError, match="chiitiler at"):
        chips_tiles.render_thumbnail(
            tmp_path / "b.webp",
            style=style,
            tiles_url="https://x/c.pmtiles",
            server="http://127.0.0.1:9",
        )


# --------------------------------------------------------------------------- #
# The drop check reads tippecanoe's per-zoom strategies
# --------------------------------------------------------------------------- #


def archive(path: Path, strategies: list) -> Path:
    with path.open("wb") as f:
        w = Writer(f)
        w.write_tile(zxy_to_tileid(0, 0, 0), b"x")
        w.finalize(
            {
                "tile_type": TileType.MVT,
                "tile_compression": Compression.NONE,
                "min_zoom": 0,
                "max_zoom": 0,
                "min_lon_e7": 0,
                "min_lat_e7": 0,
                "max_lon_e7": 0,
                "max_lat_e7": 0,
                "center_zoom": 0,
                "center_lon_e7": 0,
                "center_lat_e7": 0,
            },
            {"strategies": json.dumps(strategies)},
        )
    return path


# the strategies of a published archive, as tippecanoe writes them
LIVE = [
    {"coalesced_as_needed": 620167},
    {"coalesced_as_needed": 754580},
    {"coalesced_as_needed": 690679},
    {"coalesced_as_needed": 684300},
    {"coalesced_as_needed": 639525},
    {"coalesced_as_needed": 572239},
    {"coalesced_as_needed": 419031},
    {"tiny_polygons": 215},
    {"tiny_polygons": 177},
    {"tiny_polygons": 236},
    {"tiny_polygons": 213},
    {"tiny_polygons": 258},
    {"tiny_polygons": 270},
]


def test_check_no_drops_accepts(tmp_path: Path) -> None:
    chips_tiles.check_no_drops(archive(tmp_path / "ok.pmtiles", [{}, {"tiny_polygons": 3}]), "c")
    chips_tiles.check_no_drops(archive(tmp_path / "live.pmtiles", LIVE), "chips")


@pytest.mark.parametrize(
    ("strategies", "zoom"),
    [
        ([*LIVE[:6], {**LIVE[6], "dropped_by_rate": 9}], 6),
        ([{}, {"dropped_as_needed": 5}], 1),
    ],
)
def test_check_no_drops_refuses(tmp_path: Path, strategies: list, zoom: int) -> None:
    with pytest.raises(ChipsTilesError, match=f"zoom {zoom}"):
        chips_tiles.check_no_drops(archive(tmp_path / "drop.pmtiles", strategies), "chips")


def test_refuses_an_unreadable_table(tmp_path: Path) -> None:
    bad = tmp_path / "bad.parquet"
    bad.write_text("not parquet")
    with pytest.raises(ChipsTilesError, match=r"bad\.parquet"):
        chips_tiles.write_locations(str(bad), tmp_path / "x.parquet")


def test_refuses_an_overview_missing_locations(
    tmp_path: Path, locations: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(chips_tiles, "write_overview", lambda *_args: 0)
    with pytest.raises(ChipsTilesError, match="the overview has 0 of 3 locations"):
        chips_tiles._build_overviews(locations, tmp_path, 3)


def test_readme_gives_the_zooms() -> None:
    readme = " ".join((REPO_ROOT / "README.md").read_text().split())
    (c0, c1), (o0, o1) = chips_tiles.CHIPS_ZOOMS, chips_tiles.OVERVIEW_ZOOMS
    assert f"`chips` (zoom {c0}-{c1}" in readme
    assert f"`overview_test` (zoom {o0}-{o1}" in readme


def test_build_tiles_without_tippecanoe(
    tmp_path: Path, table: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("ftw_dataset_tools.api.tiles.tippecanoe_available", lambda: False)
    with pytest.raises(ChipsTilesError, match="tippecanoe is not installed"):
        chips_tiles.build_tiles(str(table), tmp_path / "chips.pmtiles")


@pytest.mark.skipif(not tippecanoe_available(), reason="tippecanoe is not installed")
def test_build_tiles(tmp_path: Path, table: Path) -> None:
    from pmtiles.reader import MmapSource, Reader

    out = tmp_path / "chips.pmtiles"
    assert chips_tiles.build_tiles(str(table), out) == 3
    with out.open("rb") as f:
        reader = Reader(MmapSource(f))
        header, meta = reader.header(), reader.metadata()
    layers = {layer["id"] for layer in meta.get("vector_layers", [])}
    assert layers == {"chips", *chips_tiles.OVERVIEW_LAYERS}
    assert (header["min_zoom"], header["max_zoom"]) == (
        chips_tiles.OVERVIEW_ZOOMS[0],
        chips_tiles.CHIPS_ZOOMS[1],
    )
