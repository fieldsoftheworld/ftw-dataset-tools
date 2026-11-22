"""Tests for the chips-tiles command."""

from __future__ import annotations

from typing import TYPE_CHECKING

import duckdb
from click.testing import CliRunner

from ftw_dataset_tools.cli import cli
from tests.test_api.test_chips_tiles import ROWS, write_table

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


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
