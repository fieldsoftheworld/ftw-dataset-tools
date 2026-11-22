"""Tests for chip sizes, chip ids and cutting the FTW grid to a custom chip size."""

from __future__ import annotations

import math

import duckdb
import pandas as pd
import pytest
import shapely.affinity
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
    drop_stray_parts,
    expand_bounds,
    id_digits,
    join_band_pieces,
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


def _band_line_pieces() -> tuple[object, object, object]:
    """A 2 km chip of zone 33 that 48°N cuts in two: (full, southern, northern) in lon/lat."""
    _, y48 = Transformer.from_crs("OGC:CRS84", UTM_33N, always_xy=True).transform(15.0, 48.0)
    y = y48 // 2000 * 2000
    full = _to_lonlat(box(X0, y, X0 + 2000, y + 2000))
    return (
        full,
        full.intersection(box(-180, -90, 180, 48)),
        full.intersection(box(-180, 48, 180, 90)),
    )


def _lonlat_grid(cells: dict[str, object]) -> duckdb.DuckDBPyConnection:
    """A connection holding ``grid_table`` with lon/lat cells and a published-style bbox."""
    conn = duckdb.connect()
    conn.execute("INSTALL spatial; LOAD spatial;")
    conn.execute("""
        CREATE TABLE grid_table (
            gzd VARCHAR, id VARCHAR, geometry GEOMETRY,
            bbox STRUCT(xmin DOUBLE, ymin DOUBLE, xmax DOUBLE, ymax DOUBLE)
        )
    """)
    conn.executemany(
        "INSERT INTO grid_table VALUES (?, ?, ST_GeomFromText(?::VARCHAR), "
        "STRUCT_PACK(xmin := ?, ymin := ?, xmax := ?, ymax := ?))",
        [(cid[4:7], cid, g.wkt, *g.bounds) for cid, g in cells.items()],
    )
    return conn


def _rows(conn: duckdb.DuckDBPyConnection) -> dict[str, tuple]:
    rows = conn.execute("SELECT id, gzd, ST_AsText(geometry), bbox FROM grid_table").fetchall()
    return {cid: (gzd, wkt.loads(text), bbox) for cid, gzd, text, bbox in rows}


# Pieces are joined only when both are under 99.5% of a 2 km chip.
_MAX_AREA = 0.995 * 4_000_000


def _join(conn: duckdb.DuckDBPyConnection) -> int:
    return join_band_pieces(
        conn, "grid_table", _MAX_AREA, 'ST_Area_Spheroid(ST_FlipCoordinates("g"))'
    )


class TestJoinBandPieces:
    def test_pieces_join_under_the_southern_id(self) -> None:
        full, south, north = _band_line_pieces()
        below = _to_lonlat(box(X0, 5_000_000, X0 + 2000, 5_002_000))
        conn = _lonlat_grid(
            {"ftw-33TWP0016": south, "ftw-33UWP0016": north, "ftw-33TWP0014": below}
        )
        assert _join(conn) == 1
        rows = _rows(conn)
        assert set(rows) == {"ftw-33TWP0016", "ftw-33TWP0014"}
        gzd, geom, bbox = rows["ftw-33TWP0016"]
        assert gzd == "33T"
        assert geom.symmetric_difference(full).area < 1e-12 * full.area
        assert tuple(bbox.values()) == pytest.approx(geom.bounds)
        assert rows["ftw-33TWP0014"][1].equals_exact(below, 1e-12)

    def test_only_neighbouring_bands_of_one_chip_join(self) -> None:
        _, south, north = _band_line_pieces()
        cells = {
            "ftw-33TWP0016": south,
            "ftw-33UWP0216": north,  # other easting
            "ftw-32UWP0016": north,  # other zone
            "ftw-33VWP0016": north,  # band two up
            "grid_7": north,
        }
        conn = _lonlat_grid(cells)
        assert _join(conn) == 0
        assert set(_rows(conn)) == set(cells)

    def test_far_apart_chips_with_the_same_rest_do_not_join(self) -> None:
        # Row letters repeat every 2,000 km, so a W chip at 64.5°N and an X chip at
        # 82.4°N of zone 25 share everything but the band. Both small, but apart.
        conn = _lonlat_grid(
            {
                "ftw-25WEN0010": box(-33.0, 64.5, -32.99, 64.501),
                "ftw-25XEN0010": box(-33.0, 82.43, -32.99, 82.431),
            }
        )
        assert _join(conn) == 0

    @pytest.mark.parametrize("north_share", [0.002, 0.998])
    def test_a_full_size_piece_is_never_joined(self, north_share: float) -> None:
        # A band line that clips only a corner off a chip leaves one piece that passes
        # the area filter on its own; that chip must keep its id and extent.
        full, _, _ = _band_line_pieces()
        _, ymin, _, ymax = full.bounds
        line = ymax - north_share * (ymax - ymin)
        south = full.intersection(box(-180, -90, 180, line))
        north = full.intersection(box(-180, line, 180, 90))
        conn = _lonlat_grid({"ftw-33TWP0016": south, "ftw-33UWP0016": north})
        assert _join(conn) == 0
        assert set(_rows(conn)) == {"ftw-33TWP0016", "ftw-33UWP0016"}

    def test_gap_between_the_pieces_is_closed(self) -> None:
        _, south, north = _band_line_pieces()
        # A notch in the middle of the northern piece's bottom edge: the union then
        # has a hole where neither piece reaches the line.
        x0, y0, x1, _ = north.bounds
        mid = (x0 + x1) / 2
        notch = box(mid - 0.001, y0 - 0.0001, mid + 0.001, y0 + 0.00001)
        conn = _lonlat_grid({"ftw-33TWP0016": south, "ftw-33UWP0016": north.difference(notch)})
        assert _join(conn) == 1
        geom = _rows(conn)["ftw-33TWP0016"][1]
        assert len(geom.interiors) == 0

    def test_sub_km_chip_ids(self) -> None:
        _, south, north = _band_line_pieces()
        conn = _lonlat_grid({"ftw-33TWP005160": south, "ftw-33UWP005160": north})
        assert _join(conn) == 1

    def test_grid_without_ids_is_left_alone(self) -> None:
        conn = duckdb.connect()
        conn.execute("INSTALL spatial; LOAD spatial;")
        conn.execute("CREATE TABLE grid_table AS SELECT ST_Point(0, 0) AS geometry")
        assert _join(conn) == 0


def _stray_cell() -> tuple[object, object]:
    """A 2 km cell of 33UXP and the same cell with a near-empty sliver 100 km south."""
    cell = _to_lonlat(_cell(10, 98))
    sliver = _to_lonlat(box(X0 + 10_000, Y0 - 98_000, X0 + 12_000, Y0 - 97_999.9))
    return cell, MultiPolygon([cell, sliver])


class TestDropStrayParts:
    def test_sliver_is_cut_off_and_bbox_follows(self) -> None:
        cell, stray = _stray_cell()
        other = _to_lonlat(_cell(20, 20))
        conn = _lonlat_grid({"ftw-33UXP1098": stray, "ftw-33UXP2020": other})
        assert drop_stray_parts(conn, "grid_table") == (1, 0)
        rows = _rows(conn)
        _, geom, bbox = rows["ftw-33UXP1098"]
        assert geom.geom_type == "Polygon"
        assert geom.equals_exact(cell, 1e-9)
        assert tuple(bbox.values()) == pytest.approx(cell.bounds)
        assert rows["ftw-33UXP2020"][1].equals_exact(other, 1e-9)

    def test_touching_parts_stay(self) -> None:
        # Row 90 of the square is inside band U (the square's origin is at 47.8°N).
        left, right = _to_lonlat(_cell(0, 90, 1)), _to_lonlat(_cell(1, 90, 1))
        conn = _lonlat_grid({"ftw-33UXP0090": MultiPolygon([left, right])})
        assert drop_stray_parts(conn, "grid_table") == (1, 0)
        assert _rows(conn)["ftw-33UXP0090"][1].area == pytest.approx((left.union(right)).area)

    def test_part_in_another_hemisphere_is_cut_off(self) -> None:
        # As in GZD 33C of grid v0.2: a speck in Antarctica and a bigger part at 72°N.
        speck = box(12.0, -72.26, 12.0000001, -72.2599999)
        far = box(12.0, 71.88, 12.06, 71.9)
        conn = _lonlat_grid({"ftw-33CUV9678": MultiPolygon([speck, far])})
        assert drop_stray_parts(conn, "grid_table") == (1, 0)
        assert _rows(conn)["ftw-33CUV9678"][1].equals_exact(speck, 1e-12)

    def test_cell_with_no_part_in_its_gzd_is_dropped(self) -> None:
        _, stray = _stray_cell()
        outside = MultiPolygon([box(68.0, -0.09, 68.01, -0.07), box(40.0, 72.0, 40.04, 72.01)])
        conn = _lonlat_grid({"ftw-42XUE9690": outside, "ftw-33UXP1098": stray})
        assert drop_stray_parts(conn, "grid_table") == (1, 1)
        assert set(_rows(conn)) == {"ftw-33UXP1098"}

    def test_copy_in_another_zone_is_cut_off_even_when_listed_first(self) -> None:
        # As in GZDs 18K, 18L and 20E of grid v0.2: an equal copy 72 degrees west.
        cell = _to_lonlat(_cell(10, 90))
        copy = shapely.affinity.translate(cell, xoff=-72)
        conn = _lonlat_grid({"ftw-33UXP1090": MultiPolygon([copy, cell])})
        assert drop_stray_parts(conn, "grid_table") == (1, 0)
        assert _rows(conn)["ftw-33UXP1090"][1].equals_exact(cell, 1e-9)

    def test_single_part_cells_are_untouched(self) -> None:
        conn = _lonlat_grid({"ftw-33UXP0000": _to_lonlat(_cell(0, 0))})
        assert drop_stray_parts(conn, "grid_table") == (0, 0)

    def test_grid_without_ids_or_bbox(self) -> None:
        _, stray = _stray_cell()
        conn = duckdb.connect()
        conn.execute("INSTALL spatial; LOAD spatial;")
        conn.execute(
            "CREATE TABLE grid_table AS SELECT ST_GeomFromText(?::VARCHAR) AS geometry", [stray.wkt]
        )
        assert drop_stray_parts(conn, "grid_table") == (1, 0)
        geom_type = conn.execute("SELECT ST_GeometryType(geometry) FROM grid_table").fetchone()[0]
        assert geom_type == "POLYGON"
