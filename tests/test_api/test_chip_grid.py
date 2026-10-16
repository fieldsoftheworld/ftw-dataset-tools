"""Tests for chip sizes, chip ids and cutting the FTW grid to a custom chip size."""

from __future__ import annotations

import math

import duckdb
import pandas as pd
import pytest
from pyproj import Transformer
from shapely import wkt
from shapely.geometry import MultiPolygon, box
from shapely.ops import transform

from ftw_dataset_tools.api.chip_grid import (
    GridCutError,
    InvalidChipSizeError,
    chip_size_label,
    chip_size_m,
    cut_grid,
    expand_bounds,
    id_digits,
    parse_chip_ids,
)

# MGRS square 33UXP, anchored at a 100 km multiple in UTM zone 33N.
X0, Y0 = 500_000, 5_300_000
UTM_33N = "EPSG:32633"


class TestChipSizeM:
    @pytest.mark.parametrize(
        ("km", "metres"),
        [(2, 2000), (2.0, 2000), (0.5, 500), (0.1, 100), (0.256, 256), (100, 100_000)],
    )
    def test_valid_sizes(self, km: float, metres: int) -> None:
        assert chip_size_m(km) == metres

    @pytest.mark.parametrize(
        ("bad", "match"),
        [
            (0, "positive"),
            (-1, "positive"),
            (math.nan, "positive"),
            (True, "number"),
            ("2", "number"),
            (0.1234, "whole number of metres"),
            (0.0005, "whole number of metres"),
        ],
    )
    def test_invalid_sizes(self, bad: object, match: str) -> None:
        with pytest.raises(InvalidChipSizeError, match=match):
            chip_size_m(bad)  # type: ignore[arg-type]

    def test_metres_passed_as_km_get_a_hint(self) -> None:
        with pytest.raises(InvalidChipSizeError, match=r"kilometres.*For 512 m chips pass 0.512"):
            chip_size_m(512)

    def test_error_carries_the_value(self) -> None:
        with pytest.raises(InvalidChipSizeError) as info:
            chip_size_m(150)
        assert info.value.km_size == 150


@pytest.mark.parametrize(
    ("metres", "digits"), [(2000, 2), (1000, 2), (5000, 2), (500, 3), (100, 3), (250, 4), (256, 5)]
)
def test_id_digits(metres: int, digits: int) -> None:
    assert id_digits(metres) == digits


def test_chip_size_label() -> None:
    assert chip_size_label(0.1) == "0.1 km (100 m)"
    assert chip_size_label(2) == "2 km (2,000 m)"


class TestParseChipIds:
    def test_every_precision_lands_on_one_metre_scale(self) -> None:
        ids = pd.Series(
            ["ftw-33UXP6658", "ftw-33UXP665581", "ftw-1CDE66505810", "ftw-33UXP6650458102"]
        )
        parts = parse_chip_ids(ids)
        assert parts["square"].tolist() == ["33UXP", "33UXP", "1CDE", "33UXP"]
        assert parts["east_m"].tolist() == [66000, 66500, 66500, 66504]
        assert parts["north_m"].tolist() == [58000, 58100, 58100, 58102]
        assert parts["unit_m"].tolist() == [1000, 100, 10, 1]

    def test_keeps_the_input_index(self) -> None:
        parts = parse_chip_ids(pd.Series(["ftw-33UXP0000"], index=[7]))
        assert parts.index.tolist() == [7]

    @pytest.mark.parametrize("bad", ["ftw-33UXP00AB", "ftw-33UXP000", "grid_001", "ftw-33UXP00"])
    def test_rejects_non_chip_ids(self, bad: str) -> None:
        with pytest.raises(ValueError, match="numeric easting/northing"):
            parse_chip_ids(pd.Series(["ftw-33UXP0000", bad]))


def test_expand_bounds_grows_by_one_chip() -> None:
    xmin, ymin, xmax, ymax = expand_bounds((10.0, 50.0, 10.1, 50.1), 2)
    assert 50.0 - ymin == pytest.approx(2000 / 111_320)
    # A degree of longitude is shorter at 50 degrees, so the margin is wider in degrees.
    assert 10.0 - xmin > 50.0 - ymin
    assert (xmax - 10.1, ymax - 50.1) == pytest.approx((10.0 - xmin, 50.0 - ymin))


def test_expand_bounds_clamps_to_the_globe() -> None:
    assert expand_bounds((-180.0, -90.0, 180.0, 90.0), 100) == (-180.0, -90.0, 180.0, 90.0)


def _to_lonlat(geom, utm: str = UTM_33N):
    return transform(Transformer.from_crs(utm, "OGC:CRS84", always_xy=True).transform, geom)


def _grid(cells: dict[str, object], utm: str = UTM_33N) -> duckdb.DuckDBPyConnection:
    """A connection holding ``grid_table``: lon/lat cells built from UTM geometries."""
    conn = duckdb.connect()
    conn.execute("INSTALL spatial; LOAD spatial;")
    conn.execute("CREATE TABLE grid_table (id VARCHAR, geometry GEOMETRY)")
    conn.executemany(
        "INSERT INTO grid_table VALUES (?, ST_GeomFromText(?::VARCHAR))",
        [(chip_id, _to_lonlat(g, utm).wkt) for chip_id, g in cells.items()],
    )
    return conn


def _cell(e_km: float, n_km: float, size_km: float = 2, x0: int = X0, y0: int = Y0):
    x, y = x0 + e_km * 1000, y0 + n_km * 1000
    return box(x, y, x + size_km * 1000, y + size_km * 1000)


def _two_by_two() -> dict[str, object]:
    """Four 2 km source cells covering 4 x 4 km at the square's corner."""
    return {f"ftw-33UXP{e:02d}{n:02d}": _cell(e, n) for e in (0, 2) for n in (0, 2)}


def _cut(conn: duckdb.DuckDBPyConnection, utm: str = UTM_33N) -> dict[str, object]:
    """The cut grid as {id: UTM geometry}."""
    rows = conn.execute(
        f"SELECT id, ST_AsText(ST_Transform(geometry, 'OGC:CRS84', '{utm}')) FROM grid_table"
    ).fetchall()
    return {chip_id: wkt.loads(text) for chip_id, text in rows}


class TestCutGrid:
    def test_whole_km_size_reproduces_the_source_ids(self) -> None:
        conn = _grid(_two_by_two())
        assert cut_grid(conn, "grid_table", 2) == 4
        cells = _cut(conn)
        assert set(cells) == set(_two_by_two())
        assert all(g.area == pytest.approx(4e6, rel=1e-6) for g in cells.values())

    def test_sub_km_chips_get_longer_ids(self) -> None:
        conn = _grid(_two_by_two())
        assert cut_grid(conn, "grid_table", 0.5) == 64
        cells = _cut(conn)
        expected = {f"ftw-33UXP{e:03d}{n:03d}" for e in range(0, 40, 5) for n in range(0, 40, 5)}
        assert set(cells) == expected
        assert cells["ftw-33UXP035015"].bounds == pytest.approx(
            (X0 + 3500, Y0 + 1500, X0 + 4000, Y0 + 2000), abs=0.01
        )
        assert all(g.area == pytest.approx(250_000, rel=1e-6) for g in cells.values())

    def test_metre_precision_ids(self) -> None:
        conn = _grid({"ftw-33UXP0000": _cell(0, 0)})
        cut_grid(conn, "grid_table", 0.256)
        cells = _cut(conn)
        assert "ftw-33UXP0025600512" in cells
        assert cells["ftw-33UXP0025600512"].area == pytest.approx(256**2, rel=1e-6)

    def test_chips_larger_than_the_source_merge_cells(self) -> None:
        conn = _grid(_two_by_two())
        assert cut_grid(conn, "grid_table", 4) == 1
        (chip_id, geom), *_ = _cut(conn).items()
        assert chip_id == "ftw-33UXP0000"
        assert geom.area == pytest.approx(16e6, rel=1e-6)
        assert len(geom.exterior.coords) > 5  # densified so it follows the UTM lines

    def test_size_not_dividing_the_source(self) -> None:
        conn = _grid(_two_by_two())
        cut_grid(conn, "grid_table", 3)
        areas = {k: round(g.area / 1e6, 3) for k, g in _cut(conn).items()}
        # The source stops at 4 km, so the chips starting at 3 km are 1 km deep.
        assert areas == {
            "ftw-33UXP0000": 9.0,
            "ftw-33UXP0300": 3.0,
            "ftw-33UXP0003": 3.0,
            "ftw-33UXP0303": 1.0,
        }

    def test_clipped_source_cell_stays_clipped(self) -> None:
        # A cell cut at a zone boundary 600 m into its 2 km.
        conn = _grid({"ftw-33UXP0400": box(X0 + 4000, Y0, X0 + 4600, Y0 + 2000)})
        cut_grid(conn, "grid_table", 0.5)
        cells = _cut(conn)
        assert set(cells) == {f"ftw-33UXP{e:03d}{n:03d}" for e in (40, 45) for n in (0, 5, 10, 15)}
        assert cells["ftw-33UXP040000"].area == pytest.approx(250_000, rel=1e-6)
        assert cells["ftw-33UXP045000"].area == pytest.approx(50_000, rel=1e-6)

    def test_last_column_is_clipped_when_size_does_not_divide_100_km(self) -> None:
        conn = _grid({"ftw-33UXP9800": _cell(98, 0)})
        cut_grid(conn, "grid_table", 0.3)
        cells = _cut(conn)
        last = cells["ftw-33UXP999000"]
        assert last.bounds == pytest.approx((X0 + 99_900, Y0, X0 + 100_000, Y0 + 300), abs=0.01)
        assert max(cells) < "ftw-33UXP999999"  # nothing runs past the square

    def test_stray_sliver_outside_the_square_is_dropped(self) -> None:
        # Some published cells carry a sliver just across their square's edge.
        stray = MultiPolygon([_cell(0, 0), box(X0, Y0 - 2, X0 + 2000, Y0)])
        conn = _grid({"ftw-33UXP0000": stray})
        assert cut_grid(conn, "grid_table", 1) == 4
        assert set(_cut(conn)) == {
            "ftw-33UXP0000",
            "ftw-33UXP0001",
            "ftw-33UXP0100",
            "ftw-33UXP0101",
        }

    def test_southern_hemisphere(self) -> None:
        utm_55s = "EPSG:32755"
        conn = _grid({"ftw-55HFA0204": _cell(2, 4, x0=600_000, y0=6_100_000)}, utm=utm_55s)
        cut_grid(conn, "grid_table", 1)
        cells = _cut(conn, utm=utm_55s)
        assert set(cells) == {"ftw-55HFA0204", "ftw-55HFA0205", "ftw-55HFA0304", "ftw-55HFA0305"}
        assert cells["ftw-55HFA0305"].bounds == pytest.approx(
            (603_000, 6_105_000, 604_000, 6_106_000), abs=0.01
        )

    def test_published_grid_columns(self) -> None:
        conn = _grid({"ftw-33UXP1202": _cell(12, 2)})
        cut_grid(conn, "grid_table", 0.5)
        row = conn.execute(
            "SELECT gzd, mgrs_10km, bbox, ST_XMin(geometry), ST_YMax(geometry) FROM grid_table "
            "WHERE id = 'ftw-33UXP135035'"
        ).fetchone()
        gzd, mgrs_10km, bbox, xmin, ymax = row
        assert (gzd, mgrs_10km) == ("33U", "33UXP10")
        assert bbox["xmin"] == xmin
        assert bbox["ymax"] == ymax

    def test_keep_bounds(self) -> None:
        conn = _grid(_two_by_two())
        corner = _to_lonlat(box(X0 + 100, Y0 + 100, X0 + 200, Y0 + 200)).bounds
        assert cut_grid(conn, "grid_table", 1, keep_bounds=corner) == 1
        assert set(_cut(conn)) == {"ftw-33UXP0000"}

    def test_rejects_non_ftw_ids(self) -> None:
        conn = _grid({"grid_001": _cell(0, 0)})
        with pytest.raises(GridCutError, match="grid_001") as info:
            cut_grid(conn, "grid_table", 1)
        assert info.value.bad_ids == ["grid_001"]

    def test_rejects_invalid_size(self) -> None:
        conn = _grid(_two_by_two())
        with pytest.raises(InvalidChipSizeError):
            cut_grid(conn, "grid_table", 0.0001)
