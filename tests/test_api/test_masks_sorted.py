"""The masks stage queries Hilbert-sorted copies of the fields and lines, in source order."""

from __future__ import annotations

import random
from pathlib import Path

import duckdb
import geopandas as gpd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import rasterio
from shapely.geometry import LineString, box

from ftw_dataset_tools.api import masks
from ftw_dataset_tools.api.geo import ensure_spatial_loaded

CRS = "EPSG:4326"


def _shuffled_grid(path: Path, n: int = 12, seed: int = 0) -> list[str]:
    """n x n small boxes over 10-11E, 50-51N in random row order; returns the ids in file order."""
    cells = [(i, j) for i in range(n) for j in range(n)]
    random.Random(seed).shuffle(cells)
    step = 1 / n
    geoms = [
        box(10 + i * step, 50 + j * step, 10 + (i + 0.8) * step, 50 + (j + 0.8) * step)
        for i, j in cells
    ]
    ids = [f"f{i}_{j}" for i, j in cells]
    gpd.GeoDataFrame({"id": ids}, geometry=geoms, crs=CRS).to_parquet(
        path, write_covering_bbox=True, row_group_size=10
    )
    return ids


class TestSpatiallySortedCopy:
    def test_row_groups_cover_small_areas(self, tmp_path, monkeypatch) -> None:
        # DuckDB rounds row groups up to 2048 rows, so eight of them need this many cells.
        monkeypatch.setattr(masks, "_SORTED_ROW_GROUP", 2048)
        _shuffled_grid(tmp_path / "fields.parquet", n=128)

        copy = masks._spatially_sorted_copy(
            tmp_path / "fields.parquet", "geometry", tmp_path / "copy.parquet", 1024
        )

        assert copy == tmp_path / "copy.parquet"
        meta = pq.ParquetFile(copy).metadata
        names = [meta.row_group(0).column(j).path_in_schema for j in range(meta.num_columns)]
        xmin = names.index("bbox.xmin")
        spans = []
        for i in range(meta.num_row_groups):
            stats = meta.row_group(i).column(xmin).statistics
            spans.append(stats.max - stats.min)
        # Shuffled, every group spans nearly the whole degree; sorted, about a quarter of it.
        assert len(spans) == 8
        assert max(spans) < 0.6

    def test_copy_records_each_rows_source_position(self, tmp_path) -> None:
        ids = _shuffled_grid(tmp_path / "fields.parquet")

        copy = masks._spatially_sorted_copy(
            tmp_path / "fields.parquet", "geometry", tmp_path / "copy.parquet", 1024
        )

        table = pq.read_table(copy, columns=["id", masks._SOURCE_ROW_COL]).to_pylist()
        assert sorted(r[masks._SOURCE_ROW_COL] for r in table) == list(range(len(ids)))
        assert all(ids[r[masks._SOURCE_ROW_COL]] == r["id"] for r in table)
        assert [r["id"] for r in table] != ids

    def test_query_restores_source_order(self, tmp_path) -> None:
        ids = _shuffled_grid(tmp_path / "fields.parquet")
        copy = masks._spatially_sorted_copy(
            tmp_path / "fields.parquet", "geometry", tmp_path / "copy.parquet", 1024
        )
        conn = duckdb.connect(":memory:")
        ensure_spatial_loaded(conn)
        bounds = (10.0, 50.0, 10.5, 50.5)

        rows = masks._get_geometries_in_bounds(
            conn, copy, "geometry", bounds, id_col="id", order_col=masks._SOURCE_ROW_COL
        )
        original = masks._get_geometries_in_bounds(
            conn, tmp_path / "fields.parquet", "geometry", bounds, id_col="id"
        )

        assert [r[0] for r in rows] == [r[0] for r in original]
        assert [r[0] for r in rows] == [i for i in ids if i in {r[0] for r in rows}]

    def test_no_bbox_column_keeps_the_original(self, tmp_path) -> None:
        path = tmp_path / "fields.parquet"
        gpd.GeoDataFrame({"id": [1]}, geometry=[box(10, 50, 11, 51)], crs=CRS).to_parquet(path)

        assert masks._spatially_sorted_copy(path, "geometry", tmp_path / "c.parquet", 1024) is None
        assert not (tmp_path / "c.parquet").exists()

    def test_empty_file_keeps_the_original(self, tmp_path) -> None:
        path = tmp_path / "fields.parquet"
        frame = gpd.GeoDataFrame({"id": [1]}, geometry=[box(10, 50, 11, 51)], crs=CRS)
        frame.iloc[:0].to_parquet(path, write_covering_bbox=True)

        assert masks._spatially_sorted_copy(path, "geometry", tmp_path / "c.parquet", 1024) is None

    def test_projected_file_keeps_its_crs_and_is_used(self, tmp_path) -> None:
        path = tmp_path / "fields.parquet"
        geoms = [box(500000 + i * 100, 5600000, 500050 + i * 100, 5600050) for i in range(50)]
        gpd.GeoDataFrame({"id": range(50)}, geometry=geoms, crs="EPSG:32632").to_parquet(
            path, write_covering_bbox=True
        )

        copy = masks._spatially_sorted_copy(path, "geometry", tmp_path / "c.parquet", 1024)

        assert copy is not None
        assert gpd.read_parquet(copy).crs.to_epsg() == 32632

    def test_nan_bbox_past_the_sample_keeps_the_original(self, tmp_path) -> None:
        path = tmp_path / "fields.parquet"
        n = masks._BBOX_CHECK_ROWS + 10
        frame = gpd.GeoDataFrame({"id": range(n)}, geometry=[box(10, 50, 10.1, 50.1)] * n, crs=CRS)
        frame.to_parquet(path, write_covering_bbox=True)
        table = pq.read_table(path)
        bbox = table.column("bbox").to_pylist()
        bbox[-1] = {**bbox[-1], "xmax": float("nan")}
        index = table.schema.get_field_index("bbox")
        table = table.set_column(
            index, "bbox", pa.array(bbox, type=table.schema.field("bbox").type)
        )
        pq.write_table(table, path)

        assert masks._spatially_sorted_copy(path, "geometry", tmp_path / "c.parquet", 1024) is None
        assert not (tmp_path / "c.parquet").exists()

    def test_failed_copy_keeps_the_original(self, tmp_path) -> None:
        path = tmp_path / "fields.parquet"
        gpd.GeoDataFrame(
            {"file_row_number": [1]}, geometry=[box(10, 50, 11, 51)], crs=CRS
        ).to_parquet(path, write_covering_bbox=True)

        assert masks._spatially_sorted_copy(path, "geometry", tmp_path / "c.parquet", 1024) is None
        assert not (tmp_path / "c.parquet").exists()


def _overlapping_inputs(tmp_path: Path, write_covering_bbox: bool) -> tuple[Path, Path, Path]:
    """One chip; overlapping fields whose file order differs from their Hilbert order."""
    chips = tmp_path / "chips.parquet"
    gpd.GeoDataFrame(
        {"id": ["c1"], "field_coverage_pct": [90.0]},
        geometry=[box(10.0, 50.0, 10.01, 50.01)],
        crs=CRS,
    ).to_parquet(chips)
    # Each field overlaps the next; where they overlap, the later row's id must win.
    fields = [
        box(10.006, 50.006, 10.0095, 50.0095),
        box(10.0005, 50.0005, 10.007, 50.007),
        box(10.004, 50.0005, 10.0095, 50.005),
        box(10.0005, 50.004, 10.005, 50.0095),
    ]
    ids = [101, 102, 103, 104]
    paths = []
    for name, geoms in (
        ("fields", fields),
        ("lines", [LineString(f.exterior.coords) for f in fields]),
    ):
        path = tmp_path / f"{name}.parquet"
        gpd.GeoDataFrame({"id": ids}, geometry=geoms, crs=CRS).to_parquet(
            path, write_covering_bbox=write_covering_bbox
        )
        paths.append(path)
    return chips, paths[0], paths[1]


def _instance_mask(tmp_path: Path, write_covering_bbox: bool = True) -> np.ndarray:
    tmp_path.mkdir(exist_ok=True)
    chips, fields, lines = _overlapping_inputs(tmp_path, write_covering_bbox)
    result = masks.create_masks(
        chips_file=chips,
        boundaries_file=fields,
        boundary_lines_file=lines,
        output_dir=tmp_path / "masks",
        field_dataset="test",
        mask_types=[masks.MaskType.INSTANCE],
        resolution=2.5,
        num_workers=1,
    )[masks.MaskType.INSTANCE]
    assert result.total_created == 1, result.masks_skipped
    with rasterio.open(result.masks_created[0].output_path) as src:
        return src.read(1)


class TestCreateMasksOnSortedCopies:
    def test_masks_match_the_unsorted_files(self, tmp_path, monkeypatch) -> None:
        sorted_mask = _instance_mask(tmp_path / "sorted")
        monkeypatch.setattr(masks, "_spatially_sorted_copy", lambda *_a, **_k: None)
        unsorted_mask = _instance_mask(tmp_path / "unsorted")

        assert set(np.unique(sorted_mask)) >= {101, 102, 103, 104}
        np.testing.assert_array_equal(sorted_mask, unsorted_mask)

    def test_workers_read_the_copies(self, tmp_path, monkeypatch) -> None:
        seen = []
        original = masks._run_work_items

        def spy(work_items, *args, **kwargs):
            seen.extend((Path(t.boundaries_path), t.boundaries_order_col) for t in work_items)
            return original(work_items, *args, **kwargs)

        monkeypatch.setattr(masks, "_run_work_items", spy)
        _instance_mask(tmp_path)

        assert [(p.name, col) for p, col in seen] == [("fields.parquet", masks._SOURCE_ROW_COL)]
        assert ".sorted-inputs-" in str(seen[0][0])

    def test_copies_are_removed_afterwards(self, tmp_path) -> None:
        _instance_mask(tmp_path)

        assert not list((tmp_path / "masks").glob(".sorted-inputs-*"))

    def test_files_without_bbox_are_read_directly(self, tmp_path, monkeypatch) -> None:
        seen = []
        original = masks._run_work_items

        def spy(work_items, *args, **kwargs):
            seen.extend((Path(t.boundaries_path), t.boundaries_order_col) for t in work_items)
            return original(work_items, *args, **kwargs)

        monkeypatch.setattr(masks, "_run_work_items", spy)
        _instance_mask(tmp_path, write_covering_bbox=False)

        assert seen == [(tmp_path / "fields.parquet", None)]

    def test_a_killed_runs_copies_are_removed(self, tmp_path) -> None:
        stale = tmp_path / "masks" / ".sorted-inputs-old"
        stale.mkdir(parents=True)
        (stale / "fields.parquet").write_bytes(b"x")

        _instance_mask(tmp_path)

        assert not stale.exists()
