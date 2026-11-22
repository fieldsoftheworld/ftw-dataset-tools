"""Tests for the chips-table and chips-tiles commands."""

from __future__ import annotations

from typing import TYPE_CHECKING

import duckdb
import pytest
from click.testing import CliRunner

from ftw_dataset_tools.cli import cli
from tests.test_api.test_chips_table import AA, BB, STAGED_ROWS, World, write_previous
from tests.test_api.test_chips_tiles import ROWS, write_table

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def world(tmp_path: Path) -> World:
    w = World(tmp_path)
    w.write_catalog({"aa_2021": ("zz", "block3x3"), "bb_2022": ("bb", "predefined")})
    w.publish_chips("zz", "aa_2021", AA)
    w.publish_chips("bb", "bb_2022", BB, admin=False)
    write_previous(
        tmp_path / "boot.parquet",
        [("aa_2021", "block3x3", AA, True), ("bb_2022", "predefined", BB, False)],
    )
    return w


def _args(world: World, *extra: str) -> list[str]:
    return [
        "chips-table",
        str(world.catalog / "catalog.json"),
        "--data-base-url",
        str(world.base),
        "--previous",
        str(world.root / "boot.parquet"),
        "--out",
        str(world.root / "out" / "chips.parquet"),
        "--workers",
        "2",
        *extra,
    ]


def test_chips_table(world: World) -> None:
    world.local("aa_2021", STAGED_ROWS)
    world.local("dd_2024", [("ftw-33TUN8080", "train")])
    result = CliRunner().invoke(
        cli,
        _args(
            world,
            "--local",
            str(world.staging / "aa_2021"),
            "--local",
            str(world.staging / "dd_2024"),
            "--group",
            "dd_2024=dd",
        ),
    )
    assert result.exit_code == 0, result.output
    assert "7 chips from 3 datasets" in result.output
    out = world.root / "out" / "chips.parquet"
    groups = duckdb.sql(f"SELECT DISTINCT dataset, \"group\" FROM '{out}' ORDER BY 1").fetchall()
    assert groups == [("aa_2021", "zz"), ("bb_2022", "bb"), ("dd_2024", "dd")]


def test_chips_table_refusal_is_a_click_error(world: World) -> None:
    world.write_catalog({"aa_2021": ("zz", "block3x3")})
    result = CliRunner().invoke(cli, _args(world))
    assert result.exit_code == 1
    assert "Error: the previous table has bb_2022, which this build leaves out" in result.output
    result = CliRunner().invoke(cli, [*_args(world), "--drop", "bb_2022"])
    assert result.exit_code == 0, result.output


def test_chips_table_new_local_needs_group(world: World) -> None:
    world.local("dd_2024", [("ftw-33TUN8080", "train")])
    result = CliRunner().invoke(cli, _args(world, "--local", str(world.staging / "dd_2024")))
    assert result.exit_code == 1
    assert "--group dd_2024=<group>" in result.output


def test_chips_table_bad_group(world: World) -> None:
    result = CliRunner().invoke(cli, _args(world, "--group", "dd_2024"))
    assert result.exit_code == 2
    assert "is not <id>=<group>" in result.output


def test_chips_table_missing_catalog(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        cli,
        [
            "chips-table",
            str(tmp_path / "catalog.json"),
            "--previous",
            "p.parquet",
            "--out",
            str(tmp_path / "o.parquet"),
        ],
    )
    assert result.exit_code == 1
    assert "No such file" in result.output


def test_chips_table_needs_previous(world: World) -> None:
    result = CliRunner().invoke(
        cli, ["chips-table", str(world.catalog / "catalog.json"), "--out", "x.parquet"]
    )
    assert result.exit_code == 2
    assert "--previous" in result.output


def test_chips_tiles_refuses_old_table(tmp_path: Path) -> None:
    table = tmp_path / "chips.parquet"
    write_table(table, ROWS)
    old = tmp_path / "old.parquet"
    duckdb.sql(
        f"COPY (SELECT * EXCLUDE (split_admin2_v1) FROM '{table}') TO '{old}' (FORMAT parquet)"
    )
    result = CliRunner().invoke(
        cli, ["chips-tiles", "--table", str(old), "--out", str(tmp_path / "c.pmtiles")]
    )
    assert result.exit_code == 1
    assert "has no split_admin2_v1" in result.output


def test_chips_tiles_needs_table_and_out() -> None:
    result = CliRunner().invoke(cli, ["chips-tiles"])
    assert result.exit_code == 2
    assert "--table and --out" in result.output


def test_chips_tiles_thumbnail_needs_style(tmp_path: Path) -> None:
    result = CliRunner().invoke(cli, ["chips-tiles", "--thumbnail", str(tmp_path / "t.webp")])
    assert result.exit_code == 2
    assert "--style and --tiles-url" in result.output


def test_chips_tiles_thumbnail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    style = tmp_path / "style.json"
    style.write_text("{}")
    calls = {}

    def fake(out, *, style, tiles_url, server):
        calls.update(out=out, style=style, tiles_url=tiles_url, server=server)
        return out

    monkeypatch.setattr("ftw_dataset_tools.api.chips_tiles.render_thumbnail", fake)
    result = CliRunner().invoke(
        cli,
        [
            "chips-tiles",
            "--thumbnail",
            str(tmp_path / "t.webp"),
            "--style",
            str(style),
            "--tiles-url",
            "https://x/chips.pmtiles",
        ],
    )
    assert result.exit_code == 0, result.output
    assert calls["tiles_url"] == "https://x/chips.pmtiles"
    assert calls["server"] == "http://localhost:13579"
