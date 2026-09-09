"""Tests for the field_stats API."""

from pathlib import Path

import duckdb
import geopandas as gpd
import geoparquet_io as gpio
import pytest
from shapely.geometry import box

from ftw_dataset_tools.api.field_stats import FieldStatsResult


class TestDetectBboxColumn:
    """Tests for detect_bbox_column function."""

    def test_returns_none_without_bbox(self, sample_geoparquet_4326: Path) -> None:
        """Test returns None when no bbox column."""
        from ftw_dataset_tools.api.field_stats import detect_bbox_column

        conn = duckdb.connect(":memory:")
        conn.execute("INSTALL spatial; LOAD spatial;")

        result = detect_bbox_column(conn, sample_geoparquet_4326, "geometry")
        assert result is None
        conn.close()

    def test_returns_bbox_column_name(self, tmp_path: Path) -> None:
        """Test returns bbox column name when present."""
        from ftw_dataset_tools.api.field_stats import detect_bbox_column

        # Create file with bbox
        gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:4326")
        path = tmp_path / "with_bbox.parquet"
        gdf.to_parquet(path)
        gpio.read(str(path)).add_bbox().write(str(path))

        conn = duckdb.connect(":memory:")
        conn.execute("INSTALL spatial; LOAD spatial;")

        result = detect_bbox_column(conn, path, "geometry")
        assert result == "bbox"
        conn.close()


class TestBuildCoverageBatchQuery:
    """Tests for _build_coverage_batch_query and _build_result_query."""

    def test_query_with_bbox_optimization(self) -> None:
        from ftw_dataset_tools.api.field_stats import _build_coverage_batch_query

        query = _build_coverage_batch_query("geometry", "geometry", "bbox", "bbox", 10, 19)

        assert "bbox" in query.lower()
        assert "st_intersects" in query.lower()
        assert "between 10 and 19" in query.lower()
        assert "st_union_agg" in query.lower()

    def test_query_without_bbox_optimization(self) -> None:
        from ftw_dataset_tools.api.field_stats import _build_coverage_batch_query

        query = _build_coverage_batch_query("geometry", "geometry", None, None, 0, 0)

        assert "bbox" not in query.lower()
        assert "st_intersects" in query.lower()

    def test_result_query_joins_coverage(self) -> None:
        from ftw_dataset_tools.api.field_stats import _build_result_query

        query = _build_result_query("geometry", "coverage")

        assert "left join coverage" in query.lower()
        assert '"coverage"' in query


class TestFieldStatsResult:
    """Tests for FieldStatsResult dataclass."""

    def test_coverage_percentage_calculation(self) -> None:
        """Test that coverage_percentage is calculated correctly."""
        from pathlib import Path

        result = FieldStatsResult(
            output_path=Path("/tmp/test.parquet"),
            total_cells=100,
            cells_with_coverage=25,
            average_coverage=15.5,
            max_coverage=95.0,
        )
        assert result.coverage_percentage == 25.0

    def test_coverage_percentage_zero_cells(self) -> None:
        """Test that coverage_percentage handles zero total cells."""
        from pathlib import Path

        result = FieldStatsResult(
            output_path=Path("/tmp/test.parquet"),
            total_cells=0,
            cells_with_coverage=0,
            average_coverage=0.0,
            max_coverage=0.0,
        )
        assert result.coverage_percentage == 0.0


class TestAddFieldStats:
    """Tests for add_field_stats function."""

    def test_file_not_found_grid(self, tmp_path: pytest.TempPathFactory) -> None:
        """Test that FileNotFoundError is raised for missing grid file."""
        from ftw_dataset_tools.api.field_stats import add_field_stats

        # Create a dummy fields file (must exist to test grid file error)
        fields_file = tmp_path / "fields.parquet"
        fields_file.touch()

        with pytest.raises(FileNotFoundError, match="Grid file not found"):
            add_field_stats(
                grid_file="/nonexistent/grid.parquet",
                fields_file=str(fields_file),
            )

    def test_file_not_found_fields(self, tmp_path: Path) -> None:
        """Test that FileNotFoundError is raised for missing fields file."""
        from ftw_dataset_tools.api.field_stats import add_field_stats

        # Create a dummy grid file
        grid_file = tmp_path / "grid.parquet"
        grid_file.touch()

        with pytest.raises(FileNotFoundError, match="Fields file not found"):
            add_field_stats(
                grid_file=str(grid_file),
                fields_file="/nonexistent/fields.parquet",
            )

    def test_remote_grid_bounds_guard_for_mislabeled_crs(self, tmp_path: Path) -> None:
        """Test guard for projected bounds when EPSG:4326 is expected."""
        from ftw_dataset_tools.api.field_stats import add_field_stats

        fields_file = tmp_path / "fields.parquet"
        gdf = gpd.GeoDataFrame(
            {
                "id": ["field1"],
                "geometry": [box(408000, 5808000, 409000, 5809000)],
            },
            crs="EPSG:4326",
        )
        gdf.to_parquet(fields_file)

        messages: list[str] = []
        with pytest.raises(ValueError, match="Fields bounds appear to be in projected units"):
            add_field_stats(
                fields_file=str(fields_file),
                grid_file=None,
                on_progress=messages.append,
            )

        assert any("Warning: Fields bounds are outside degree ranges" in msg for msg in messages)


class TestAddFieldStatsWithLocalGrid:
    """Tests for add_field_stats with local grid file."""

    def test_add_field_stats_basic(
        self, sample_grid_geoparquet: Path, sample_fields_geoparquet: Path, tmp_path: Path
    ) -> None:
        """Test add_field_stats with local grid and fields files."""
        from ftw_dataset_tools.api.field_stats import add_field_stats

        output_file = tmp_path / "chips_output.parquet"
        result = add_field_stats(
            grid_file=sample_grid_geoparquet,
            fields_file=sample_fields_geoparquet,
            output_file=output_file,
        )

        assert result.output_path == output_file
        assert result.total_cells == 2  # Two grid cells
        assert output_file.exists()

    def test_batch_size_does_not_change_coverage(
        self, sample_grid_geoparquet: Path, sample_fields_geoparquet: Path, tmp_path: Path
    ) -> None:
        """Coverage is identical whether cells are aggregated one at a time or all at once."""
        import duckdb

        from ftw_dataset_tools.api.field_stats import add_field_stats

        outputs = {}
        for batch_size in (1, 1000):
            out = tmp_path / f"chips_{batch_size}.parquet"
            add_field_stats(
                grid_file=sample_grid_geoparquet,
                fields_file=sample_fields_geoparquet,
                output_file=out,
                batch_size=batch_size,
            )
            con = duckdb.connect()
            outputs[batch_size] = sorted(
                con.execute(f"SELECT field_coverage_pct FROM read_parquet('{out}')").fetchall()
            )
            con.close()

        assert outputs[1] == outputs[1000]
        assert len(outputs[1]) == 2
        assert any(v[0] > 0 for v in outputs[1])

    def test_batch_size_must_be_positive(
        self, sample_grid_geoparquet: Path, sample_fields_geoparquet: Path, tmp_path: Path
    ) -> None:
        from ftw_dataset_tools.api.field_stats import add_field_stats

        with pytest.raises(ValueError, match="batch_size must be at least 1, got 0"):
            add_field_stats(
                grid_file=sample_grid_geoparquet,
                fields_file=sample_fields_geoparquet,
                output_file=tmp_path / "x.parquet",
                batch_size=0,
            )

    def test_batch_size_validated_before_any_input_is_read(self, tmp_path: Path) -> None:
        """An invalid batch size must not cost a grid download (or any file read) first."""
        from ftw_dataset_tools.api.field_stats import add_field_stats

        with pytest.raises(ValueError, match="batch_size"):
            add_field_stats(
                fields_file=tmp_path / "does-not-exist.parquet",
                output_file=tmp_path / "x.parquet",
                batch_size=0,
            )

    def test_add_field_stats_with_progress(
        self, sample_grid_geoparquet: Path, sample_fields_geoparquet: Path, tmp_path: Path
    ) -> None:
        """Test add_field_stats with progress callback."""
        from ftw_dataset_tools.api.field_stats import add_field_stats

        progress_messages: list[str] = []

        def on_progress(msg: str) -> None:
            progress_messages.append(msg)

        output_file = tmp_path / "chips_output.parquet"
        add_field_stats(
            grid_file=sample_grid_geoparquet,
            fields_file=sample_fields_geoparquet,
            output_file=output_file,
            on_progress=on_progress,
        )

        assert len(progress_messages) > 0
        assert any("Loading" in msg for msg in progress_messages)

    def test_add_field_stats_default_output_name(
        self, sample_grid_geoparquet: Path, sample_fields_geoparquet: Path
    ) -> None:
        """Test default output filename is chips_<fields_basename>.parquet."""
        from ftw_dataset_tools.api.field_stats import add_field_stats

        result = add_field_stats(
            grid_file=sample_grid_geoparquet,
            fields_file=sample_fields_geoparquet,
        )

        expected_name = "chips_fields.parquet"
        assert result.output_path.name == expected_name
        # Cleanup
        if result.output_path.exists():
            result.output_path.unlink()

    def test_add_field_stats_min_coverage_filter(
        self, sample_grid_geoparquet: Path, sample_fields_geoparquet: Path, tmp_path: Path
    ) -> None:
        """Test min_coverage parameter filters low-coverage cells."""
        from ftw_dataset_tools.api.field_stats import add_field_stats

        output_file = tmp_path / "chips_filtered.parquet"
        result = add_field_stats(
            grid_file=sample_grid_geoparquet,
            fields_file=sample_fields_geoparquet,
            output_file=output_file,
            min_coverage=1.0,  # Filter cells with coverage < 1%
        )

        # Some cells may be filtered out
        assert result.output_path.exists()


class TestDetectBboxColumnFallback:
    """Tests for detect_bbox_column schema fallback behavior."""

    def test_detect_bbox_column_from_schema(self, tmp_path: Path) -> None:
        """Test bbox detection from schema when metadata not available."""
        from ftw_dataset_tools.api.field_stats import detect_bbox_column

        # Create file with bbox column structure
        gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:4326")
        path = tmp_path / "with_bbox_struct.parquet"
        gdf.to_parquet(path)

        # Add bbox using geoparquet-io
        gpio.read(str(path)).add_bbox().write(str(path))

        conn = duckdb.connect(":memory:")
        conn.execute("INSTALL spatial; LOAD spatial;")

        result = detect_bbox_column(conn, path, "geometry")
        conn.close()

        assert result == "bbox"


class TestCRSMismatchHandling:
    """Tests for CRS mismatch detection and handling."""

    def test_crs_mismatch_error(
        self, sample_geoparquet_3035: Path, sample_grid_geoparquet: Path, tmp_path: Path
    ) -> None:
        """Test CRSMismatchError when CRS don't match."""
        from ftw_dataset_tools.api.field_stats import add_field_stats
        from ftw_dataset_tools.api.geo import CRSMismatchError

        output_file = tmp_path / "chips_output.parquet"

        with pytest.raises(CRSMismatchError):
            add_field_stats(
                grid_file=sample_grid_geoparquet,  # EPSG:4326
                fields_file=sample_geoparquet_3035,  # EPSG:3035
                output_file=output_file,
                reproject_to_4326=False,
            )


class TestFieldStatsResultProperties:
    """Additional tests for FieldStatsResult."""

    def test_average_and_max_coverage(self) -> None:
        """Test average_coverage and max_coverage fields."""
        result = FieldStatsResult(
            output_path=Path("/tmp/test.parquet"),
            total_cells=10,
            cells_with_coverage=5,
            average_coverage=45.5,
            max_coverage=95.0,
        )
        assert result.average_coverage == 45.5
        assert result.max_coverage == 95.0
        assert result.coverage_percentage == 50.0


def _unsorted_grid(path: Path, side: int = 6, size: float = 0.02) -> Path:
    """Write a ``side`` x ``side`` grid whose rows are deliberately not in id order.

    Row order in the file must not leak into the output: the chips writer sorts by
    chip id, so a shuffled input still yields one canonical output order.
    """
    cells, ids = [], []
    for i in range(side):
        for j in range(side):
            x0, y0 = 10.0 + i * size, 50.0 + j * size
            cells.append(box(x0, y0, x0 + size, y0 + size))
            ids.append(f"ftw-{i:02d}{j:02d}")
    gdf = gpd.GeoDataFrame({"id": ids}, geometry=cells, crs="EPSG:4326")
    # Reverse, then interleave, so neither the original nor the sorted order survives.
    shuffled = gdf.iloc[::-1].iloc[[*range(1, len(gdf), 2), *range(0, len(gdf), 2)]]
    shuffled.reset_index(drop=True).to_parquet(path)
    return path


def _covering_fields(path: Path, lo: int = 1, hi: int = 5, size: float = 0.02) -> Path:
    """Fields filling the grid cells in the [lo, hi) index range on both axes."""
    polys = [
        box(10.0 + i * size, 50.0 + j * size, 10.0 + (i + 1) * size, 50.0 + (j + 1) * size)
        for i in range(lo, hi)
        for j in range(lo, hi)
    ]
    gpd.GeoDataFrame({"fid": range(len(polys))}, geometry=polys, crs="EPSG:4326").to_parquet(path)
    return path


def _chip_ids(chips_file: Path) -> list[str]:
    """Row order of the written chips, as stored in the file."""
    con = duckdb.connect()
    try:
        return [row[0] for row in con.execute(f"SELECT id FROM '{chips_file}'").fetchall()]
    finally:
        con.close()


class TestChipRowOrderIsReproducible:
    """Row order decides split assignment, so it must be identical run to run.

    ``splits._assign_random_uniform`` shuffles a label array with the seeded RNG and
    assigns it to rows *positionally*, and the block strategy maps blocks in
    first-appearance order. A chips file whose row order comes from engine internals
    therefore puts a chip in train one run and test the next at the same seed.
    """

    def test_rows_are_written_in_chip_id_order(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.field_stats import add_field_stats

        grid = _unsorted_grid(tmp_path / "grid.parquet")
        fields = _covering_fields(tmp_path / "fields.parquet")
        out = tmp_path / "chips.parquet"

        add_field_stats(grid_file=grid, fields_file=fields, output_file=out)

        ids = _chip_ids(out)
        assert ids == sorted(ids)
        assert ids != _chip_ids(grid)  # the input order was not simply passed through

    def test_row_order_identical_across_runs_and_batch_sizes(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.field_stats import add_field_stats

        grid = _unsorted_grid(tmp_path / "grid.parquet")
        fields = _covering_fields(tmp_path / "fields.parquet")

        orders = []
        for run, batch_size in enumerate((1, 7, 1000, 1, 7)):
            out = tmp_path / f"chips_{run}.parquet"
            add_field_stats(
                grid_file=grid, fields_file=fields, output_file=out, batch_size=batch_size
            )
            orders.append(_chip_ids(out))

        assert all(order == orders[0] for order in orders)

    def test_split_assignment_identical_across_runs(self, tmp_path: Path) -> None:
        """The end the reproducibility guarantee is actually about."""
        from ftw_dataset_tools.api.field_stats import add_field_stats
        from ftw_dataset_tools.api.splits import assign_splits

        grid = _unsorted_grid(tmp_path / "grid.parquet")
        fields = _covering_fields(tmp_path / "fields.parquet")

        assignments = []
        for run in range(3):
            out = tmp_path / f"chips_{run}.parquet"
            add_field_stats(grid_file=grid, fields_file=fields, output_file=out)
            assign_splits(
                chips_file=out,
                split_type="random-uniform",
                split_percents=(80, 10, 10),
                random_seed=42,
            )
            con = duckdb.connect()
            assignments.append(dict(con.execute(f"SELECT id, split FROM '{out}'").fetchall()))
            con.close()

        assert all(a == assignments[0] for a in assignments)
        assert len(set(assignments[0].values())) > 1  # the split is not degenerate


class TestSparseRowids:
    """Dropping border chips deletes rows without renumbering, leaving rowid gaps."""

    def test_border_chips_leave_sparse_rowids(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.field_stats import add_field_stats

        grid = _unsorted_grid(tmp_path / "grid.parquet")
        fields = _covering_fields(tmp_path / "fields.parquet")

        messages: list[str] = []
        result = add_field_stats(
            grid_file=grid,
            fields_file=fields,
            output_file=tmp_path / "chips.parquet",
            drop_border_chips=True,
            batch_size=2,
            on_progress=messages.append,
        )

        assert any("Removed" in msg and "border chips" in msg for msg in messages)
        assert result.total_cells == 16  # the inner 4x4 block survives the convex hull
        assert result.cells_with_coverage == 16

        # Progress counts the cells that exist, not the rowid span they are spread over.
        coverage_msgs = [m for m in messages if m.strip().startswith("Coverage:")]
        assert coverage_msgs[-1].strip() == f"Coverage: {result.total_cells:,}/16 grid cells"
        assert len(coverage_msgs) == 8  # 16 cells at 2 per batch

    def test_sparse_rowids_give_the_same_coverage_at_every_batch_size(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.field_stats import add_field_stats

        grid = _unsorted_grid(tmp_path / "grid.parquet")
        fields = _covering_fields(tmp_path / "fields.parquet")

        rows = []
        for batch_size in (1, 3, 1000):
            out = tmp_path / f"chips_{batch_size}.parquet"
            add_field_stats(
                grid_file=grid,
                fields_file=fields,
                output_file=out,
                drop_border_chips=True,
                batch_size=batch_size,
            )
            con = duckdb.connect()
            rows.append(con.execute(f"SELECT id, field_coverage_pct FROM '{out}'").fetchall())
            con.close()

        assert all(r == rows[0] for r in rows)
        assert len(rows[0]) == 16


class TestChipOrderByFallback:
    """Grids without an id column still work; they just cannot be ordered by chip id."""

    def test_warns_and_skips_ordering_without_id_column(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.field_stats import add_field_stats

        cells = [box(10.0 + i * 0.02, 50.0, 10.0 + (i + 1) * 0.02, 50.02) for i in range(3)]
        grid = tmp_path / "grid_no_id.parquet"
        gpd.GeoDataFrame({"name": ["a", "b", "c"]}, geometry=cells, crs="EPSG:4326").to_parquet(
            grid
        )
        fields = _covering_fields(tmp_path / "fields.parquet", lo=0, hi=1)

        messages: list[str] = []
        result = add_field_stats(
            grid_file=grid,
            fields_file=fields,
            output_file=tmp_path / "chips.parquet",
            on_progress=messages.append,
        )

        assert result.total_cells == 3
        assert any("no 'id' column" in msg for msg in messages)
