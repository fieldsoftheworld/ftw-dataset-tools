"""Tests for per-chip admin names from a GAUL level-2 layer (a local stand-in)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import duckdb
import geopandas as gpd
import pytest
from shapely.geometry import Polygon, box

from ftw_dataset_tools.api.admin_names import (
    OUTPUT_COLUMNS,
    UNNAMED,
    AdminNamesError,
    add_admin_names,
    admin_names_summary,
    connect,
    drop_admin_names,
    name_chips,
)
from ftw_dataset_tools.api.geo import ensure_spatial_loaded, write_geoparquet
from ftw_dataset_tools.api.source import SourceFetchError

if TYPE_CHECKING:
    from pathlib import Path

# Two countries side by side along x = 10; B has two level-2 units split at y = 5.
UNITS = [
    ("AAA", "Aland", "North A", "A one", 1, box(0, 0, 10, 10)),
    ("BBB", "Bland", "North B", "B low", 2, box(10, 0, 20, 5)),
    ("BBB", "Bland", "North B", UNNAMED[0], 3, box(10, 5, 20, 10)),
]


def write_gaul(path: Path, units: list[tuple] = UNITS) -> Path:
    """A GAUL-shaped GeoParquet: names, codes, geometry and a float bbox struct."""
    con = duckdb.connect()
    ensure_spatial_loaded(con)
    rows = ", ".join(
        f"('{iso}', '{n0}', '{n1}', '{n2}', {code}, ST_GeomFromText('{geom.wkt}'))"
        for iso, n0, n1, n2, code, geom in units
    )
    con.execute(f"""
        CREATE TABLE gaul AS SELECT iso3_code, gaul0_name, gaul1_name, gaul2_name,
            gaul2_code, geometry,
            {{'xmin': ST_XMin(geometry)::FLOAT, 'ymin': ST_YMin(geometry)::FLOAT,
              'xmax': ST_XMax(geometry)::FLOAT, 'ymax': ST_YMax(geometry)::FLOAT}}
              AS geometry_bbox
        FROM (VALUES {rows}) AS t(iso3_code, gaul0_name, gaul1_name, gaul2_name,
            gaul2_code, geometry)""")
    write_geoparquet(path, conn=con, query="SELECT * FROM gaul")
    con.close()
    return path


def write_chips(path: Path, chips: dict[str, Polygon], crs: str = "EPSG:4326") -> Path:
    gpd.GeoDataFrame({"id": list(chips)}, geometry=list(chips.values()), crs=crs).to_parquet(path)
    return path


def read_names(chips: Path) -> dict[str, tuple]:
    cols = ", ".join(OUTPUT_COLUMNS)
    rows = duckdb.sql(f"SELECT id, {cols} FROM '{chips}' ORDER BY id").fetchall()
    return {r[0]: r[1:] for r in rows}


class TestAddAdminNames:
    def test_each_chip_takes_the_unit_of_its_centre(self, tmp_path: Path) -> None:
        gaul = write_gaul(tmp_path / "gaul.parquet")
        # c2 straddles the border but its centre (11, 2) is in B.
        chips = write_chips(
            tmp_path / "chips.parquet",
            {"c1": box(1, 1, 2, 2), "c2": box(9.5, 1.5, 12.5, 2.5)},
        )

        result = add_admin_names(chips, source=str(gaul))

        assert read_names(chips) == {
            "c1": ("AAA", "Aland", "North A", "A one"),
            "c2": ("BBB", "Bland", "North B", "B low"),
        }
        assert (result.total_chips, result.by_centre, result.by_overlap) == (2, 2, 0)
        assert result.countries == ("Aland", "Bland")

    def test_unnamed_placeholder_is_left_empty(self, tmp_path: Path) -> None:
        gaul = write_gaul(tmp_path / "gaul.parquet")
        chips = write_chips(tmp_path / "chips.parquet", {"c": box(15, 7, 16, 8)})

        add_admin_names(chips, source=str(gaul))

        assert read_names(chips) == {"c": ("BBB", "Bland", "North B", None)}

    def test_centre_outside_every_unit_takes_the_largest_overlap(self, tmp_path: Path) -> None:
        gaul = write_gaul(tmp_path / "gaul.parquet")
        # Centre (20.5, 2) is offshore; the chip overlaps "B low" only.
        chips = write_chips(tmp_path / "chips.parquet", {"c": box(19, 1, 22, 3)})

        result = add_admin_names(chips, source=str(gaul))

        assert read_names(chips)["c"][3] == "B low"
        assert (result.by_centre, result.by_overlap, result.unnamed) == (0, 1, 0)

    def test_chip_touching_no_unit_stays_blank(self, tmp_path: Path) -> None:
        gaul = write_gaul(tmp_path / "gaul.parquet")
        chips = write_chips(
            tmp_path / "chips.parquet", {"c": box(30, 30, 31, 31), "d": box(1, 1, 2, 2)}
        )

        result = add_admin_names(chips, source=str(gaul))

        assert read_names(chips)["c"] == (None, None, None, None)
        assert result.unnamed == 1
        assert "1 touch no unit" in admin_names_summary(result)

    def test_overlapping_units_pick_the_lowest_code(self, tmp_path: Path) -> None:
        units = [
            ("AAA", "Aland", "R", "second", 9, box(0, 0, 10, 10)),
            ("CCC", "Cland", "R", "first", 4, box(0, 0, 10, 10)),
        ]
        gaul = write_gaul(tmp_path / "gaul.parquet", units)
        chips = write_chips(tmp_path / "chips.parquet", {"c": box(1, 1, 2, 2)})

        add_admin_names(chips, source=str(gaul))

        assert read_names(chips)["c"][3] == "first"

    def test_rerun_replaces_columns_and_keeps_order(self, tmp_path: Path) -> None:
        gaul = write_gaul(tmp_path / "gaul.parquet")
        chips = write_chips(
            tmp_path / "chips.parquet", {"z": box(1, 1, 2, 2), "a": box(15, 1, 16, 2)}
        )

        add_admin_names(chips, source=str(gaul))
        add_admin_names(chips, source=str(gaul))

        columns = [c[0] for c in duckdb.sql(f"DESCRIBE SELECT * FROM '{chips}'").fetchall()]
        assert columns.count("admin0_name") == 1
        assert [r[0] for r in duckdb.sql(f"SELECT id FROM '{chips}'").fetchall()] == ["a", "z"]

    def test_empty_chips_file(self, tmp_path: Path) -> None:
        gaul = write_gaul(tmp_path / "gaul.parquet")
        chips = tmp_path / "chips.parquet"
        gpd.GeoDataFrame({"id": []}, geometry=[], crs="EPSG:4326").to_parquet(chips)

        result = add_admin_names(chips, source=str(gaul))

        assert result.total_chips == 0
        assert set(OUTPUT_COLUMNS) <= set(gpd.read_parquet(chips).columns)

    def test_projected_chips_are_refused(self, tmp_path: Path) -> None:
        gaul = write_gaul(tmp_path / "gaul.parquet")
        chips = write_chips(
            tmp_path / "chips.parquet", {"c": box(500000, 0, 501000, 1000)}, crs="EPSG:32631"
        )

        with pytest.raises(AdminNamesError, match="EPSG:32631"):
            add_admin_names(chips, source=str(gaul))

    def test_drop(self, tmp_path: Path) -> None:
        gaul = write_gaul(tmp_path / "gaul.parquet")
        chips = write_chips(tmp_path / "chips.parquet", {"c": box(1, 1, 2, 2)})
        add_admin_names(chips, source=str(gaul))

        assert drop_admin_names(chips) is True
        assert not set(OUTPUT_COLUMNS) & set(gpd.read_parquet(chips).columns)
        assert drop_admin_names(chips) is False


class TestMatching:
    def test_overlap_picks_the_larger_share(self, tmp_path: Path) -> None:
        gaul = write_gaul(tmp_path / "gaul.parquet")
        # The chip spans y 4-8 across B's east edge (x 20): it covers "B low" (y < 5)
        # by 1 and B's unnamed unit (y > 5) by 3; its centre (20.5, 6) is outside both.
        chips = write_chips(tmp_path / "chips.parquet", {"c": box(19, 4, 22, 8)})

        add_admin_names(chips, source=str(gaul))

        assert read_names(chips)["c"] == ("BBB", "Bland", "North B", None)

    def test_overlap_tie_picks_the_lowest_code(self, tmp_path: Path) -> None:
        units = [
            ("AAA", "Aland", "R", "high", 7, box(0, 0, 10, 5)),
            ("AAA", "Aland", "R", "low", 6, box(0, 5, 10, 10)),
        ]
        gaul = write_gaul(tmp_path / "gaul.parquet", units)
        # Centre (11, 5) is outside both; the chip overlaps each by the same area.
        chips = write_chips(tmp_path / "chips.parquet", {"c": box(9, 3, 13, 7)})

        add_admin_names(chips, source=str(gaul))

        assert read_names(chips)["c"][3] == "low"

    def test_not_available_placeholder(self, tmp_path: Path) -> None:
        units = [("COL", "Colombia", "Not Available", "Area", 1, box(0, 0, 10, 10))]
        gaul = write_gaul(tmp_path / "gaul.parquet", units)
        chips = write_chips(tmp_path / "chips.parquet", {"c": box(1, 1, 2, 2)})

        add_admin_names(chips, source=str(gaul))

        assert read_names(chips)["c"] == ("COL", "Colombia", None, "Area")

    def test_null_geometry_and_null_id_are_skipped(self, tmp_path: Path) -> None:
        gaul = write_gaul(tmp_path / "gaul.parquet")
        chips = tmp_path / "chips.parquet"
        gpd.GeoDataFrame(
            {"id": ["a", None, "nogeom", "off"]},
            geometry=[box(1, 1, 2, 2), box(1, 1, 2, 2), None, box(19, 1, 22, 3)],
            crs="EPSG:4326",
        ).to_parquet(chips)

        result = add_admin_names(chips, source=str(gaul))

        names = read_names(chips)
        assert names["a"][3] == "A one"
        assert names["off"][3] == "B low"  # a NULL id must not block the overlap step
        assert names["nogeom"] == (None, None, None, None)
        assert (result.total_chips, result.by_centre, result.by_overlap) == (4, 1, 1)

    def test_all_null_geometry(self, tmp_path: Path) -> None:
        gaul = write_gaul(tmp_path / "gaul.parquet")
        chips = tmp_path / "chips.parquet"
        gpd.GeoDataFrame({"id": ["a"]}, geometry=[None], crs="EPSG:4326").to_parquet(chips)

        result = add_admin_names(chips, source=str(gaul))

        assert result.unnamed == 1


class TestNameChips:
    def test_custom_columns_and_bbox_prefilter(self, tmp_path: Path) -> None:
        units = [
            *UNITS,
            ("FAR", "Farland", "F", "far", 9, box(100, 50, 110, 60)),
            ("NEA", "Nearland", "N", "near", 8, box(-5, 0, -0.005, 10)),
        ]
        gaul = write_gaul(tmp_path / "gaul.parquet", units)
        chips = tmp_path / "chips.parquet"
        gpd.GeoDataFrame(
            {"chip": ["c1"]}, geometry=gpd.GeoSeries([box(0, 1, 1, 2)]), crs="EPSG:4326"
        ).rename_geometry("geom").to_parquet(chips)

        con = connect(str(gaul))
        name_chips(
            con, f"SELECT * FROM '{chips}'", geom_col="geom", id_col="chip", source=str(gaul)
        )

        loaded = {r[0] for r in con.execute("SELECT admin0_iso3 FROM _units").fetchall()}
        # Units within the margin of the chips' bbox are read; far ones are not.
        assert loaded == {"AAA", "NEA"}
        assert con.execute("SELECT id, admin2_name, how FROM _named").fetchall() == [
            ("c1", "A one", "centre")
        ]
        con.close()

    def test_missing_source_is_a_fetch_error(self, tmp_path: Path) -> None:
        chips = write_chips(tmp_path / "chips.parquet", {"c": box(1, 1, 2, 2)})

        with pytest.raises(SourceFetchError, match="GAUL"):
            add_admin_names(chips, source=str(tmp_path / "missing" / "*.parquet"))

    def test_chips_without_the_id_column_are_refused(self, tmp_path: Path) -> None:
        gaul = write_gaul(tmp_path / "gaul.parquet")
        chips = write_chips(tmp_path / "chips.parquet", {"c": box(1, 1, 2, 2)})

        with pytest.raises(AdminNamesError, match="no 'chip_id' column"):
            add_admin_names(chips, source=str(gaul), chips_id_col="chip_id")
