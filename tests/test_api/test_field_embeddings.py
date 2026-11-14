"""Tests for per-field embedding statistics, against local stand-ins for both products."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
import pyarrow.parquet as pq
import pytest
import rasterio
from affine import Affine
from pyproj import Transformer
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from shapely.geometry import Polygon, box
from shapely.ops import transform as shapely_transform

from ftw_dataset_tools.api import field_embeddings as fe
from ftw_dataset_tools.api.field_embeddings import (
    BANDS,
    NODATA,
    AefSource,
    FieldEmbeddingsError,
    OlmoEarthSource,
    Window,
    add_field_embeddings,
    dequantize,
    field_embeddings_summary,
    read_description,
)

if TYPE_CHECKING:
    from pathlib import Path

ZONE = 31
UTM = f"EPSG:326{ZONE}"


def codes_at(cols: np.ndarray, rows: np.ndarray, seed: int) -> np.ndarray:
    """Deterministic int8 codes for absolute pixel columns/rows (x/10, -y/10), per band."""
    band = np.arange(BANDS)[:, None, None]
    values = (cols[None] * 7 + rows[None] * 13 + band * (3 + seed) + seed * 11) % 255 - 127
    return values.astype(np.int8)


class FakeSource:
    """A product with a code for every pixel of zone 31, minus an optional no-data area."""

    def __init__(self, seed: int = 0, nodata_west_of: float | None = None) -> None:
        self.seed = seed
        self.nodata_west_of = nodata_west_of
        self.reads: list[Window] = []

    def grid(self, window: Window) -> tuple[np.ndarray, np.ndarray]:
        cols = np.arange(window.width) + round(window.x0 / 10)
        rows = np.arange(window.height) + round(-window.y1 / 10)
        return np.meshgrid(cols, rows)

    def read(self, window: Window) -> np.ndarray:
        self.reads.append(window)
        if window.zone != ZONE:
            return np.full((BANDS, window.height, window.width), NODATA, np.int8)
        cols, rows = self.grid(window)
        out = codes_at(cols, rows, self.seed)
        if self.nodata_west_of is not None:
            out[:, cols * 10 + 5 < self.nodata_west_of] = NODATA
        return out


def expected(
    geom: Polygon, source: FakeSource, *, all_touched: bool = False
) -> tuple[int, np.ndarray, np.ndarray]:
    """Pixel count, mean and std computed directly over the field's own bbox."""
    xmin, ymin, xmax, ymax = geom.bounds
    x0, y1 = np.floor(xmin / 10) * 10, np.ceil(ymax / 10) * 10
    w = int(np.ceil(xmax / 10) - np.floor(xmin / 10))
    h = int(np.ceil(ymax / 10) - np.floor(ymin / 10))
    window = Window(ZONE, x0, y1, w, h)
    inside = ~geometry_mask([geom], (h, w), from_origin(x0, y1, 10, 10), all_touched=all_touched)
    codes = FakeSource(source.seed, source.nodata_west_of).read(window)
    picked = codes[:, inside]
    picked = picked[:, picked[0] != NODATA]
    values = dequantize(picked).astype(np.float64)
    return picked.shape[1], values.mean(axis=1), values.std(axis=1)


def write_fields(path: Path, geoms: dict, crs: str = UTM) -> Path:
    gpd.GeoDataFrame({"id": list(geoms)}, geometry=list(geoms.values()), crs=crs).to_parquet(path)
    return path


def rows_by_id(path: Path) -> dict:
    return {r["id"]: r for r in pq.read_table(path).to_pylist()}


# Fields around (500 km, 5,000 km) in zone 31 (about 3 E, 45 N).
X, Y = 500_000.0, 5_000_000.0


class TestDequantize:
    def test_matches_the_provider_formula(self) -> None:
        q = np.array([0, 127, -127, 64, -128], dtype=np.int8)
        out = dequantize(q)
        np.testing.assert_allclose(
            out[:4], [0, (127 / 127.5) ** 2, -((127 / 127.5) ** 2), (64 / 127.5) ** 2], rtol=1e-6
        )
        assert np.isnan(out[4])


class TestTiles:
    def test_tiles_cover_the_bounds_once_on_the_tile_grid(self) -> None:
        bounds = (1003.0, -2017.0, 1497.0, -1001.0)
        tiles = list(fe._tiles(bounds, ZONE, tile_px=16))
        covered = set()
        for t in tiles:
            assert t.width <= 16 and t.height <= 16
            for c in range(t.width):
                for r in range(t.height):
                    cell = (round(t.x0 / 10) + c, round(-t.y1 / 10) + r)
                    assert cell not in covered
                    covered.add(cell)
        cols = range(100, 150)
        rows = range(100, 202)
        assert covered == {(c, r) for c in cols for r in rows}


class TestAddFieldEmbeddings:
    def run(self, tmp_path: Path, geoms: dict, crs: str = UTM, **kwargs):
        fields = write_fields(tmp_path / "fields.parquet", geoms, crs)
        out = tmp_path / "emb.parquet"
        sources = kwargs.pop("sources", {"aef": FakeSource(0), "olmoearth": FakeSource(1)})
        result = add_field_embeddings(fields, out, year=2022, sources=sources, workers=2, **kwargs)
        return result, rows_by_id(out), sources

    def assert_stats(self, row: dict, product: str, source: FakeSource, geom, **kw) -> None:
        n, mean, std = expected(geom, source, **kw)
        assert row[f"{product}_pixels"] == n
        np.testing.assert_allclose(row[f"{product}_mean"], mean, atol=1e-6)
        np.testing.assert_allclose(row[f"{product}_std"], std, atol=1e-6)

    def test_fields_crossing_windows_match_a_direct_computation(self, tmp_path: Path) -> None:
        geoms = {
            "a": box(X + 3, Y + 7, X + 437, Y + 251),
            "b": Polygon([(X + 500, Y), (X + 900, Y + 40), (X + 700, Y + 380)]),
        }
        # 8-pixel windows: every field spans many of them.
        result, rows, sources = self.run(tmp_path, geoms, tile_px=8)

        for fid, geom in geoms.items():
            assert rows[fid]["pixel_rule"] == "centre"
            self.assert_stats(rows[fid], "aef", sources["aef"], geom)
            self.assert_stats(rows[fid], "olmoearth", sources["olmoearth"], geom)
        assert (result.total_fields, result.by_centre, result.by_touch) == (2, 2, 0)

    def test_overlapping_fields_each_count_the_shared_pixels(self, tmp_path: Path) -> None:
        geoms = {
            "a": box(X, Y, X + 300, Y + 300),
            "b": box(X + 150, Y + 150, X + 450, Y + 450),
            "c": box(X + 100, Y + 100, X + 200, Y + 200),
        }
        _, rows, sources = self.run(tmp_path, geoms, tile_px=16)

        for fid, geom in geoms.items():
            self.assert_stats(rows[fid], "aef", sources["aef"], geom)
        assert rows["a"]["pixels"] == 900

    def test_a_field_without_a_pixel_centre_uses_touched_pixels(self, tmp_path: Path) -> None:
        geoms = {"tiny": box(X + 1, Y + 1, X + 4, Y + 4), "thin": box(X + 101, Y, X + 104, Y + 95)}
        result, rows, sources = self.run(tmp_path, geoms, tile_px=4)

        assert rows["tiny"]["pixel_rule"] == "touched"
        assert rows["tiny"]["pixels"] == 1
        self.assert_stats(rows["thin"], "aef", sources["aef"], geoms["thin"], all_touched=True)
        assert rows["thin"]["pixels"] == 10
        assert result.by_touch == 2

    def test_no_data_pixels_are_left_out_per_product(self, tmp_path: Path) -> None:
        geom = box(X, Y, X + 200, Y + 100)
        sources = {"aef": FakeSource(0, nodata_west_of=X + 50), "olmoearth": FakeSource(1)}
        _, rows, _ = self.run(tmp_path, {"f": geom}, sources=sources)

        assert rows["f"]["pixels"] == 200
        assert rows["f"]["aef_pixels"] == 150
        assert rows["f"]["olmoearth_pixels"] == 200
        self.assert_stats(rows["f"], "aef", sources["aef"], geom)

    def test_a_field_with_no_data_at_all_has_no_values(self, tmp_path: Path) -> None:
        sources = {"aef": FakeSource(0, nodata_west_of=X + 1000), "olmoearth": FakeSource(1)}
        _, rows, _ = self.run(tmp_path, {"f": box(X, Y, X + 100, Y + 100)}, sources=sources)

        assert rows["f"]["aef_pixels"] == 0
        assert rows["f"]["aef_mean"] is None
        assert rows["f"]["olmoearth_mean"] is not None

    def test_a_missing_product_leaves_its_columns_empty(self, tmp_path: Path) -> None:
        _, rows, _ = self.run(
            tmp_path, {"f": box(X, Y, X + 100, Y + 100)}, sources={"aef": FakeSource(0)}
        )

        assert rows["f"]["olmoearth_pixels"] is None
        assert rows["f"]["olmoearth_mean"] is None
        assert len(rows["f"]["aef_mean"]) == BANDS

    def test_lon_lat_fields_are_read_in_the_zone_of_their_centre(self, tmp_path: Path) -> None:
        to_lonlat = Transformer.from_crs(UTM, "EPSG:4326", always_xy=True)
        utm_geom = box(X, Y, X + 300, Y + 200)
        lonlat = shapely_transform(to_lonlat.transform, utm_geom)
        _, rows, sources = self.run(tmp_path, {"f": lonlat}, crs="EPSG:4326")

        # Reprojected back, the polygon's edges are no longer exactly on the grid; the
        # direct computation uses the same reprojected polygon.
        back = shapely_transform(
            Transformer.from_crs("EPSG:4326", UTM, always_xy=True).transform, lonlat
        )
        self.assert_stats(rows["f"], "aef", sources["aef"], back)
        assert all(w.zone == ZONE for w in sources["aef"].reads)

    def test_output_keeps_the_fields_order_ids_and_empty_geometries(self, tmp_path: Path) -> None:
        fields = tmp_path / "fields.parquet"
        gpd.GeoDataFrame(
            {"id": [30, 10, 20]},
            geometry=[box(X, Y, X + 50, Y + 50), None, box(X + 900, Y, X + 950, Y + 50)],
            crs=UTM,
        ).to_parquet(fields)
        out = tmp_path / "emb.parquet"

        result = add_field_embeddings(
            fields, out, year=2022, sources={"aef": FakeSource(0)}, workers=1
        )

        table = pq.read_table(out)
        assert table.column("id").to_pylist() == [30, 10, 20]
        assert table.column("pixels").to_pylist() == [25, None, 25]
        assert result.without_pixels == 1
        description = read_description(out)
        assert description["year"] == 2022
        assert description["products"]["aef"]["included"] is True
        assert description["products"]["olmoearth"]["included"] is False
        assert "Section 2" in description["products"]["olmoearth"]["attribution"]

    def test_missing_id_column_is_refused(self, tmp_path: Path) -> None:
        fields = tmp_path / "fields.parquet"
        gpd.GeoDataFrame({"name": ["a"]}, geometry=[box(X, Y, X + 1, Y + 1)], crs=UTM).to_parquet(
            fields
        )
        with pytest.raises(FieldEmbeddingsError, match="no 'id' column"):
            add_field_embeddings(fields, tmp_path / "o.parquet", year=2022, sources={})

    def test_summary(self) -> None:
        result = fe.FieldEmbeddingsResult(10, 7, 2, ("aef", "olmoearth"), 2022)
        line = field_embeddings_summary(result)
        assert "9 of 10 fields" in line
        assert "2 from touched pixels" in line
        assert "1 without pixels" in line
        assert field_embeddings_summary(None) == "Field embeddings: none written this run"


def write_aef_tile(path: Path, x0: float, y0: float, size: int, *, south_up: bool, epsg: int):
    """An AEF-shaped COG whose codes follow codes_at over absolute EPSG:326NN pixels."""
    shift = 10_000_000 if epsg // 100 == 327 else 0
    cols = np.arange(size) + round(x0 / 10)
    top_rows = np.arange(size) + round(-(y0 - shift + size * 10) / 10)
    c, r = np.meshgrid(cols, top_rows)
    data = codes_at(c, r, seed=5)  # north-up
    if south_up:
        transform = Affine(10, 0, x0, 0, 10, y0)
        data = data[:, ::-1, :]
    else:
        transform = from_origin(x0, y0 + size * 10, 10, 10)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=size,
        height=size,
        count=BANDS,
        dtype="int8",
        crs=f"EPSG:{epsg}",
        transform=transform,
        nodata=NODATA,
    ) as dst:
        dst.write(data)
    return fe._AefTile(
        str(path), epsg % 100, epsg // 100 == 327, x0, y0, x0 + size * 10, y0 + size * 10
    )


class TestAefSource:
    def expected(self, window: Window) -> np.ndarray:
        return FakeSource(5).read(window)

    def test_a_south_up_tile_reads_north_up(self, tmp_path: Path) -> None:
        tile = write_aef_tile(tmp_path / "t.tif", 1000, 2000, 32, south_up=True, epsg=32631)
        source = AefSource(2022, [tile])
        window = Window(31, 1050, 2250, 10, 12)

        np.testing.assert_array_equal(source.read(window), self.expected(window))

    def test_windows_spanning_tiles_and_hemispheres(self, tmp_path: Path) -> None:
        # Two south-up tiles side by side, one in EPSG:32731 (y + 10,000 km).
        north = write_aef_tile(tmp_path / "n.tif", 1000, -320, 32, south_up=True, epsg=32631)
        south = write_aef_tile(
            tmp_path / "s.tif", 1000, 10_000_000 - 640, 32, south_up=True, epsg=32731
        )
        source = AefSource(2022, [north, south])
        window = Window(31, 1100, -100, 8, 40)  # rows from -100 down to -500

        out = source.read(window)
        np.testing.assert_array_equal(out, self.expected(window))

    def test_outside_every_tile_is_no_data(self, tmp_path: Path) -> None:
        tile = write_aef_tile(tmp_path / "t.tif", 1000, 2000, 16, south_up=False, epsg=32631)
        source = AefSource(2022, [tile])
        out = source.read(Window(31, 1100, 2100, 10, 10))  # half inside, half below
        expected = self.expected(Window(31, 1100, 2100, 10, 10))
        np.testing.assert_array_equal(out[:, :, :6], expected[:, :, :6])
        assert (out[:, :, 6:] == NODATA).all()
        assert (AefSource(2022, [tile]).read(Window(32, 1100, 2100, 4, 4)) == NODATA).all()

    def test_open_reads_the_index_for_the_year(self, tmp_path: Path) -> None:
        import duckdb

        index = tmp_path / "index.parquet"
        duckdb.sql(
            "COPY (SELECT * FROM (VALUES ('a.tif', 'EPSG:32736', 2021, 0.0, 1.0, 2.0, 3.0), "
            "('b.tif', 'EPSG:32636', 2022, 0.0, 1.0, 2.0, 3.0)) "
            "t(path, crs, year, utm_west, utm_south, utm_east, utm_north)) "
            f"TO '{index}' (FORMAT parquet)"
        )
        source = AefSource.open(2021, index=str(index))
        assert [(t.path, t.zone, t.south) for t in source._tiles[36]] == [("a.tif", 36, True)]


class TestOlmoEarthSource:
    def make_store(self, tmp_path: Path):
        import zarr

        root = zarr.open_group(str(tmp_path / "emb.zarr"), mode="w")
        group = root.create_group("utm31")
        # 20 x 30 pixels starting at x = 1000, y = 3000 (top), 128 bands, 2 years.
        group.attrs["spatial:transform"] = [10, 0.0, 1000, 0.0, -10, 3000]
        group.create_array("time", data=np.array([2021, 2022]))
        c, r = np.meshgrid(np.arange(30) + 100, np.arange(20) - 300)
        full = np.concatenate([codes_at(c, r, seed=2), np.zeros((64, 20, 30), np.int8)])
        array = group.create_array(
            "embeddings", shape=(2, 128, 20, 30), dtype="int8", fill_value=NODATA
        )
        array[1] = full
        return zarr.open_group(str(tmp_path / "emb.zarr"), mode="r")

    def test_reads_the_first_64_dimensions_of_the_year(self, tmp_path: Path) -> None:
        source = OlmoEarthSource(2022, self.make_store(tmp_path))
        window = Window(31, 1050, 2900, 6, 5)

        np.testing.assert_array_equal(source.read(window), FakeSource(2).read(window))
        assert source.years() == [2021, 2022]

    def test_outside_the_array_zone_or_year_is_no_data(self, tmp_path: Path) -> None:
        store = self.make_store(tmp_path)
        source = OlmoEarthSource(2022, store)
        out = source.read(Window(31, 1250, 2900, 8, 3))  # 5 columns inside, 3 past the edge
        np.testing.assert_array_equal(
            out[:, :, :5], FakeSource(2).read(Window(31, 1250, 2900, 5, 3))
        )
        assert (out[:, :, 5:] == NODATA).all()
        assert (source.read(Window(32, 1050, 2900, 2, 2)) == NODATA).all()
        assert (OlmoEarthSource(2019, store).read(Window(31, 1050, 2900, 2, 2)) == NODATA).all()

    def test_a_window_off_the_grid_is_refused(self, tmp_path: Path) -> None:
        source = OlmoEarthSource(2022, self.make_store(tmp_path))
        with pytest.raises(ValueError, match="10 m grid"):
            source.read(Window(31, 1055, 2900, 2, 2))


class TestDescription:
    def test_round_trips_through_the_parquet_metadata(self, tmp_path: Path) -> None:
        import duckdb

        path = tmp_path / "x.parquet"
        meta = json.dumps(fe.describe(2020, "id", ("aef",))).replace("'", "''")
        duckdb.sql(
            f"COPY (SELECT 1 AS id) TO '{path}' "
            f"(FORMAT parquet, KV_METADATA {{'{fe.METADATA_KEY}': '{meta}'}})"
        )
        assert read_description(path)["dequantization"] == fe.DEQUANTIZATION
        assert read_description(tmp_path / "x.parquet")["year"] == 2020


class FlakySource(FakeSource):
    """Fails the first ``fail_first`` reads, then the ones listed in ``fail_at``."""

    def __init__(self, seed: int = 0, fail_first: int = 0, fail_after: int | None = None) -> None:
        super().__init__(seed)
        self.calls = 0
        self.fail_first = fail_first
        self.fail_after = fail_after

    def read(self, window: Window) -> np.ndarray:
        self.calls += 1
        if self.calls <= self.fail_first:
            raise OSError("connection reset")
        if self.fail_after is not None and self.calls > self.fail_after:
            raise OSError("connection reset")
        return super().read(window)


@pytest.fixture
def no_backoff(monkeypatch):
    monkeypatch.setattr(fe, "READ_BACKOFF_S", 0.0)


class TestReviewFindings:
    run = TestAddFieldEmbeddings.run
    assert_stats = TestAddFieldEmbeddings.assert_stats

    def test_constant_and_single_pixel_fields_have_zero_std(self) -> None:
        sums = fe._Sums.zeros(2)
        codes = np.full((BANDS, 5), 100, np.int8)
        sums.add(np.array([0, 0, 0, 0, 1]), dequantize(codes))
        _, _, std = fe._stats(sums, np.array([True, True]), sums)
        assert (std == 0).all()

    def test_a_proj_string_crs_is_read_in_full(self, tmp_path: Path) -> None:
        geom = box(X, Y, X + 200, Y + 100)
        crs = "+proj=utm +zone=31 +datum=WGS84 +units=m +no_defs"
        _, rows, sources = self.run(tmp_path, {"f": geom}, crs=crs)
        self.assert_stats(rows["f"], "aef", sources["aef"], geom)

    def test_an_explicit_null_crs_is_refused(self, tmp_path: Path) -> None:
        fields = write_fields(tmp_path / "f.parquet", {"f": box(X, Y, X + 10, Y + 10)})
        table = pq.read_table(fields)
        geo = json.loads(table.schema.metadata[b"geo"])
        geo["columns"]["geometry"]["crs"] = None
        pq.write_table(
            table.replace_schema_metadata({**table.schema.metadata, b"geo": json.dumps(geo)}),
            fields,
        )
        with pytest.raises(FieldEmbeddingsError, match="declares no CRS"):
            add_field_embeddings(fields, tmp_path / "o.parquet", year=2022, sources={})

    @pytest.mark.parametrize("id_col", ["pixels", "aef_mean", 'a"b'])
    def test_an_id_column_clashing_with_the_output_is_refused(
        self, tmp_path: Path, id_col: str
    ) -> None:
        fields = write_fields(tmp_path / "f.parquet", {"f": box(X, Y, X + 10, Y + 10)})
        with pytest.raises(FieldEmbeddingsError, match="cannot be the id column"):
            add_field_embeddings(
                fields, tmp_path / "o.parquet", year=2022, sources={}, id_col=id_col
            )

    def test_fields_off_any_utm_grid_get_no_pixels(self, tmp_path: Path) -> None:
        geoms = {
            "ok": box(3.0, 45.0, 3.001, 45.001),
            "antimeridian": box(-179.9, -17.0, 179.9, -16.9),
        }
        result, rows, _ = self.run(tmp_path, geoms, crs="EPSG:4326")
        assert rows["ok"]["pixels"] > 0
        assert not rows["antimeridian"]["pixels"]
        assert rows["antimeridian"]["pixel_rule"] is None
        assert result.without_pixels == 1

    def test_many_blocks_and_workers_match_a_direct_computation(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(3)
        geoms = {}
        for i in range(40):
            x, y = X + rng.uniform(0, 90_000), Y + rng.uniform(0, 90_000)
            geoms[f"f{i}"] = box(x, y, x + rng.uniform(5, 400), y + rng.uniform(5, 400))
        fields = write_fields(tmp_path / "fields.parquet", geoms)
        sources = {"aef": FakeSource(0)}
        add_field_embeddings(
            fields, tmp_path / "o.parquet", year=2022, sources=sources, workers=4, tile_px=64
        )
        table = pq.read_table(tmp_path / "o.parquet")
        assert table.column("id").to_pylist() == list(geoms)
        rows = rows_by_id(tmp_path / "o.parquet")
        for fid, geom in geoms.items():
            self.assert_stats(
                rows[fid],
                "aef",
                sources["aef"],
                geom,
                all_touched=rows[fid]["pixel_rule"] == "touched",
            )

    def test_reads_are_clipped_to_the_fields(self, tmp_path: Path) -> None:
        # A small field near a tile corner reads its own few pixels, not whole tiles.
        _, _, sources = self.run(tmp_path, {"f": box(X + 10235, Y + 10235, X + 10245, Y + 10245)})
        assert sum(w.width * w.height for w in sources["aef"].reads) <= 4

    @pytest.mark.usefixtures("no_backoff")
    def test_a_transient_read_error_is_retried(self, tmp_path: Path) -> None:
        geom = box(X, Y, X + 100, Y + 100)
        sources = {"aef": FlakySource(0, fail_first=2)}
        _, rows, _ = self.run(tmp_path, {"f": geom}, sources=sources)
        self.assert_stats(rows["f"], "aef", FakeSource(0), geom)

    @pytest.mark.usefixtures("no_backoff")
    def test_a_stopped_run_resumes_where_it_stopped(self, tmp_path: Path, monkeypatch) -> None:
        from ftw_dataset_tools.api.source import SourceFetchError

        monkeypatch.setattr(fe, "PART_ROWS", 1)
        geoms = {f"f{i}": box(X + i * 30_000, Y, X + i * 30_000 + 50, Y + 50) for i in range(6)}
        fields = write_fields(tmp_path / "fields.parquet", geoms)
        out = tmp_path / "o.parquet"

        with pytest.raises(SourceFetchError):
            add_field_embeddings(
                fields, out, year=2022, workers=1, sources={"aef": FlakySource(0, fail_after=3)}
            )
        assert not out.exists()
        assert len(list((tmp_path / "o.parquet.partial").glob("part-*.parquet"))) >= 2

        again = FakeSource(0)
        add_field_embeddings(fields, out, year=2022, workers=1, sources={"aef": again})

        assert len(again.reads) <= 6 - 2
        rows = rows_by_id(out)
        for fid, geom in geoms.items():
            self.assert_stats(rows[fid], "aef", again, geom)
        assert not (tmp_path / "o.parquet.partial").exists()

    def test_parts_of_another_run_are_not_reused(self, tmp_path: Path) -> None:
        geoms = {"f": box(X, Y, X + 50, Y + 50)}
        fields = write_fields(tmp_path / "fields.parquet", geoms)
        out = tmp_path / "o.parquet"
        stale = tmp_path / "o.parquet.partial"
        stale.mkdir()
        (stale / "run.json").write_text(json.dumps({"year": 1999}))
        (stale / "part-000000.parquet").write_bytes(b"junk")

        add_field_embeddings(fields, out, year=2022, sources={"aef": FakeSource(0)})

        assert rows_by_id(out)["f"]["pixels"] == 25


class TestIsCurrent:
    def test_same_fields_year_and_products(self, tmp_path: Path) -> None:
        fields = write_fields(tmp_path / "f.parquet", {"a": box(X, Y, X + 50, Y + 50)})
        out = tmp_path / "o.parquet"
        add_field_embeddings(
            fields, out, year=2022, sources={"aef": FakeSource(0), "olmoearth": FakeSource(1)}
        )

        assert fe.is_current(out, fields, year=2022)
        assert not fe.is_current(out, fields, year=2023)
        assert not fe.is_current(tmp_path / "missing.parquet", fields, year=2022)

        # Same ids, one geometry changed: no longer current, whatever the timestamps.
        write_fields(fields, {"a": box(X, Y, X + 60, Y + 50)})
        assert not fe.is_current(out, fields, year=2022)

    def test_a_file_missing_a_product_is_not_current(self, tmp_path: Path) -> None:
        fields = write_fields(tmp_path / "f.parquet", {"a": box(X, Y, X + 50, Y + 50)})
        out = tmp_path / "o.parquet"
        add_field_embeddings(fields, out, year=2022, sources={"aef": FakeSource(0)})

        assert not fe.is_current(out, fields, year=2022)


BLOCK = fe.BLOCK_M * 2


class TestSecondReviewFindings:
    run = TestAddFieldEmbeddings.run

    def test_a_no_data_field_between_fields_with_data_is_written(self, tmp_path: Path) -> None:
        # a and c have AEF data, b sits in the no-data area: v-N-v in one batch.
        geoms = {
            "a": box(X + 100, Y, X + 150, Y + 50),
            "b": box(X - 300, Y, X - 250, Y + 50),
            "c": box(X + 200, Y, X + 250, Y + 50),
        }
        sources = {"aef": FakeSource(0, nodata_west_of=X), "olmoearth": FakeSource(1)}
        _, rows, _ = self.run(tmp_path, geoms, sources=sources)

        assert rows["b"]["aef_mean"] is None
        assert len(rows["a"]["aef_mean"]) == len(rows["c"]["aef_mean"]) == BANDS

    def test_a_second_run_on_the_same_output_is_refused(self, tmp_path: Path) -> None:
        import fcntl

        fields = write_fields(tmp_path / "f.parquet", {"a": box(X, Y, X + 50, Y + 50)})
        out = tmp_path / "o.parquet"
        with (tmp_path / "o.parquet.lock").open("w") as held:
            fcntl.flock(held, fcntl.LOCK_EX)
            with pytest.raises(FieldEmbeddingsError, match="another run"):
                add_field_embeddings(fields, out, year=2022, sources={"aef": FakeSource(0)})
        add_field_embeddings(fields, out, year=2022, sources={"aef": FakeSource(0)})
        assert out.exists()

    def test_parts_that_do_not_add_up_are_never_published(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        fields = write_fields(tmp_path / "f.parquet", {"a": box(X, Y, X + 50, Y + 50)})
        out = tmp_path / "o.parquet"
        real_write = fe._Parts.write

        def write_twice(self, batches):
            real_write(self, batches)
            real_write(self, batches)

        monkeypatch.setattr(fe._Parts, "write", write_twice)
        with pytest.raises(FieldEmbeddingsError, match="expected 1"):
            add_field_embeddings(fields, out, year=2022, sources={"aef": FakeSource(0)})
        assert not out.exists()

    def test_a_zero_area_field_is_placed_by_its_bbox(self, tmp_path: Path) -> None:
        line = Polygon([(X + 1, Y + 1), (X + 31, Y + 31), (X + 61, Y + 61), (X + 1, Y + 1)])
        _, rows, sources = self.run(tmp_path, {"flat": line, "ok": box(X, Y, X + 20, Y + 20)})
        # No area, so no pixels, but read where it is rather than at (0, 0).
        assert not rows["flat"]["pixels"]
        assert rows["ok"]["pixels"] == 4
        assert sources["aef"].reads
        assert all(abs(w.x0 - X) < BLOCK and abs(w.y1 - Y) < BLOCK for w in sources["aef"].reads)

    def test_without_any_product_the_ids_are_written_without_reading(self, tmp_path: Path) -> None:
        fields = write_fields(tmp_path / "f.parquet", {"a": box(X, Y, X + 50, Y + 50)})
        out = tmp_path / "o.parquet"
        add_field_embeddings(fields, out, year=2016, sources={})
        rows = rows_by_id(out)
        assert rows["a"]["aef_mean"] is None
        assert read_description(out)["requested_products"] == []

    def test_a_year_no_product_has_counts_as_current(self, tmp_path: Path, monkeypatch) -> None:
        fields = write_fields(tmp_path / "f.parquet", {"a": box(X, Y, X + 50, Y + 50)})
        out = tmp_path / "o.parquet"
        monkeypatch.setattr(fe, "open_sources", lambda _year: {})

        add_field_embeddings(fields, out, year=2016)

        assert fe.is_current(out, fields, year=2016)

    def test_the_fingerprint_follows_the_crs_and_ignores_scan_order(self, tmp_path: Path) -> None:
        geoms = {"a": box(X, Y, X + 50, Y + 50), "b": box(X + 90, Y, X + 99, Y + 9)}
        utm = fe.fields_fingerprint(write_fields(tmp_path / "u.parquet", geoms))
        again = fe.fields_fingerprint(write_fields(tmp_path / "v.parquet", geoms))
        other_crs = fe.fields_fingerprint(write_fields(tmp_path / "w.parquet", geoms, "EPSG:32632"))
        assert utm == again
        assert utm != other_crs

    def test_an_older_algorithm_version_is_not_current(self, tmp_path: Path, monkeypatch) -> None:
        fields = write_fields(tmp_path / "f.parquet", {"a": box(X, Y, X + 50, Y + 50)})
        out = tmp_path / "o.parquet"
        monkeypatch.setattr(fe, "open_sources", lambda _year: {})
        add_field_embeddings(fields, out, year=2016)
        monkeypatch.setattr(fe, "ALGORITHM_VERSION", fe.ALGORITHM_VERSION + 1)
        assert not fe.is_current(out, fields, year=2016)
