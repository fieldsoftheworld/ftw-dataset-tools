"""Tests for per-cluster border chip detection."""

from __future__ import annotations

import duckdb
import pytest

from ftw_dataset_tools.api.chip_borders import find_border_chips

W = 100.0  # chip width in projected units


def _chips(cells: set[tuple[int, int]], occupied: set[tuple[int, int]] | None = None):
    """Build a chips table from lattice coordinates; ``occupied`` defaults to all cells."""
    occupied = cells if occupied is None else occupied
    conn = duckdb.connect()
    conn.execute("INSTALL spatial; LOAD spatial;")
    conn.execute("CREATE TABLE chips (cx INTEGER, cy INTEGER, cov DOUBLE, geom GEOMETRY)")
    for cx, cy in sorted(cells):
        conn.execute(
            "INSERT INTO chips VALUES (?, ?, ?, ST_MakeEnvelope(?, ?, ?, ?))",
            [
                cx,
                cy,
                100.0 if (cx, cy) in occupied else 0.0,
                cx * W,
                cy * W,
                (cx + 1) * W,
                (cy + 1) * W,
            ],
        )
    return conn


def _block(x0: int, y0: int, w: int, h: int) -> set[tuple[int, int]]:
    return {(x0 + i, y0 + j) for i in range(w) for j in range(h)}


def _ring(x0: int, y0: int, w: int, h: int) -> set[tuple[int, int]]:
    return _block(x0, y0, w, h) - _block(x0 + 1, y0 + 1, w - 2, h - 2)


def _dropped_cells(conn, result) -> set[tuple[int, int]]:
    if not result.border_rowids:
        return set()
    rows = conn.execute(
        "SELECT cx, cy FROM chips WHERE rowid IN (SELECT unnest(?::BIGINT[]))",
        [result.border_rowids],
    ).fetchall()
    return {(r[0], r[1]) for r in rows}


class TestFindBorderChips:
    """Border detection across the cluster layouts the FTW datasets actually have."""

    def test_two_blocks_drop_both_interior_edges(self) -> None:
        """Estonia's shape: the edges facing the gap must go, not just the outer ring."""
        cells = _block(0, 0, 6, 6) | _block(12, 0, 6, 6)
        conn = _chips(cells)
        result = find_border_chips(conn, "chips", "geom", "cov")

        assert result.cluster_count == 2
        assert _dropped_cells(conn, result) == _ring(0, 0, 6, 6) | _ring(12, 0, 6, 6)
        conn.close()

    def test_many_small_blocks_each_get_their_own_border(self) -> None:
        """Cambodia's shape: every block is bordered independently."""
        origins = [(0, 0), (10, 0), (20, 0), (0, 10), (10, 10)]
        cells: set[tuple[int, int]] = set()
        expected: set[tuple[int, int]] = set()
        for x0, y0 in origins:
            cells |= _block(x0, y0, 5, 5)
            expected |= _ring(x0, y0, 5, 5)
        conn = _chips(cells)
        result = find_border_chips(conn, "chips", "geom", "cov")

        assert result.cluster_count == len(origins)
        assert _dropped_cells(conn, result) == expected
        conn.close()

    def test_enclosed_hole_is_not_a_border(self) -> None:
        """A lake inside a labelled block must not pull the border inward."""
        cells = _block(0, 0, 9, 9)
        occupied = cells - _block(4, 4, 2, 2)
        conn = _chips(cells, occupied=occupied)
        result = find_border_chips(conn, "chips", "geom", "cov")

        assert result.cluster_count == 1
        assert _dropped_cells(conn, result) == _ring(0, 0, 9, 9)
        conn.close()

    def test_wider_gap_setting_merges_neighbouring_blocks(self) -> None:
        """A gap narrower than the setting reads as one cluster, not two."""
        cells = _block(0, 0, 6, 6) | _block(8, 0, 6, 6)  # two-chip gap

        conn = _chips(cells)
        assert find_border_chips(conn, "chips", "geom", "cov", gap_chips=0).cluster_count == 2
        conn.close()

        conn = _chips(cells)
        assert find_border_chips(conn, "chips", "geom", "cov", gap_chips=6).cluster_count == 1
        conn.close()

    def test_unlabelled_chips_outside_every_cluster_are_dropped(self) -> None:
        """Chips holding no fields are not part of the labelled region."""
        cells = _block(0, 0, 5, 5) | {(20, 20)}
        conn = _chips(cells, occupied=_block(0, 0, 5, 5))
        result = find_border_chips(conn, "chips", "geom", "cov")

        assert (20, 20) in _dropped_cells(conn, result)
        conn.close()

    def test_no_occupied_chips_returns_nothing(self) -> None:
        conn = _chips(_block(0, 0, 3, 3), occupied=set())
        result = find_border_chips(conn, "chips", "geom", "cov")

        assert result.border_rowids == []
        assert result.cluster_count == 0
        conn.close()

    def test_single_chip_is_all_border(self) -> None:
        conn = _chips({(0, 0)})
        result = find_border_chips(conn, "chips", "geom", "cov")

        assert _dropped_cells(conn, result) == {(0, 0)}
        conn.close()

    def test_negative_gap_chips_is_rejected(self) -> None:
        conn = _chips(_block(0, 0, 3, 3))
        with pytest.raises(ValueError, match="gap_chips must be 0 or greater"):
            find_border_chips(conn, "chips", "geom", "cov", gap_chips=-1)
        conn.close()
