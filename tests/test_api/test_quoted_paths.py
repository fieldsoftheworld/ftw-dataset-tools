"""Regression tests for paths containing single quotes.

An unescaped apostrophe ends a DuckDB SQL literal early. Each test drives an
API entry point from a directory named ``o'brien data`` and fails with
``ParserException`` if its module drops the ``geo.sql_path`` escaping.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from ftw_dataset_tools.api.config import ClassFilter


@pytest.fixture
def quoted_dir(tmp_path: Path) -> Path:
    """A directory whose name contains a single quote."""
    path = tmp_path / "o'brien data"
    path.mkdir()
    return path


def _copy_into(source: Path, directory: Path) -> Path:
    """Copy ``source`` into ``directory``, returning the new path."""
    destination = directory / source.name
    shutil.copy(source, destination)
    return destination


class TestQuotedPathsBoundaries:
    """api.boundaries survives an apostrophe in the path."""

    def test_create_boundaries(self, sample_boundaries_geoparquet: Path, quoted_dir: Path) -> None:
        from ftw_dataset_tools.api.boundaries import create_boundaries

        fields = _copy_into(sample_boundaries_geoparquet, quoted_dir)

        result = create_boundaries(fields, output_dir=quoted_dir)

        assert result.total_processed == 1
        assert result.total_features > 0
        assert result.files_processed[0].output_path.exists()


class TestQuotedPathsClassFilter:
    """api.class_filter survives an apostrophe in the path."""

    @pytest.fixture
    def crop_filter(self) -> ClassFilter:
        return ClassFilter(column="id", include=["1", "2"], exclude=["3", "4"])

    def test_resolve_column(
        self, sample_fields_geoparquet: Path, quoted_dir: Path, crop_filter: ClassFilter
    ) -> None:
        from ftw_dataset_tools.api.class_filter import resolve_column

        fields = _copy_into(sample_fields_geoparquet, quoted_dir)

        assert resolve_column(fields, crop_filter) == "id"

    def test_get_distinct_classes(self, sample_fields_geoparquet: Path, quoted_dir: Path) -> None:
        from ftw_dataset_tools.api.class_filter import get_distinct_classes

        fields = _copy_into(sample_fields_geoparquet, quoted_dir)

        assert get_distinct_classes(fields, "id") == {"1", "2", "3", "4"}

    def test_write_filtered_fields(
        self, sample_fields_geoparquet: Path, quoted_dir: Path, crop_filter: ClassFilter
    ) -> None:
        import duckdb

        from ftw_dataset_tools.api.class_filter import write_filtered_fields
        from ftw_dataset_tools.api.geo import sql_path

        fields = _copy_into(sample_fields_geoparquet, quoted_dir)
        output = quoted_dir / "filtered.parquet"

        write_filtered_fields(fields, output, crop_filter)

        con = duckdb.connect(":memory:")
        try:
            count = con.execute(f"SELECT COUNT(*) FROM '{sql_path(output)}'").fetchone()[0]
        finally:
            con.close()
        assert count == 2


class TestQuotedPathsFieldSummary:
    """api.field_summary survives an apostrophe in the path."""

    def test_summarize_fields(self, sample_fields_geoparquet: Path, quoted_dir: Path) -> None:
        from ftw_dataset_tools.api.field_summary import summarize_fields

        fields = _copy_into(sample_fields_geoparquet, quoted_dir)

        summary = summarize_fields(fields)

        assert summary.num_rows == 4
        assert any(col.name == "id" for col in summary.columns)


class TestQuotedPathsFieldStats:
    """api.field_stats survives an apostrophe in the path."""

    def test_add_field_stats_with_local_grid(
        self, sample_fields_geoparquet: Path, sample_grid_geoparquet: Path, quoted_dir: Path
    ) -> None:
        from ftw_dataset_tools.api.field_stats import add_field_stats

        fields = _copy_into(sample_fields_geoparquet, quoted_dir)
        grid = _copy_into(sample_grid_geoparquet, quoted_dir)
        output = quoted_dir / "chips.parquet"

        result = add_field_stats(fields, grid_file=grid, output_file=output)

        assert Path(result.output_path).exists()


class TestQuotedPathsFtwGrid:
    """api.ftw_grid survives an apostrophe in the path."""

    def test_create_ftw_grid_parses_quoted_path(
        self, sample_mgrs_1km_geoparquet: Path, quoted_dir: Path
    ) -> None:
        import duckdb

        from ftw_dataset_tools.api.ftw_grid import create_ftw_grid

        mgrs = _copy_into(sample_mgrs_1km_geoparquet, quoted_dir)
        output = quoted_dir / "ftw_grid.parquet"

        # This module's Arrow write fails on some DuckDB builds, unrelated to
        # escaping, so only assert the load query parses.
        try:
            create_ftw_grid(mgrs, output_path=output, km_size=2)
        except duckdb.ParserException as exc:  # pragma: no cover - regression guard
            pytest.fail(f"path with an apostrophe was not escaped: {exc}")
        except duckdb.InternalException:
            pass


def _write_bbox_grid(path: Path) -> None:
    """Write a two-cell grid with only a ``bbox`` column; one cell overlaps the fixtures."""
    import duckdb

    from ftw_dataset_tools.api.geo import sql_path

    con = duckdb.connect(":memory:")
    try:
        con.execute(
            f"""
            COPY (
                SELECT * FROM (VALUES
                    ('a', {{'xmin': 10.0, 'ymin': 50.0, 'xmax': 10.04, 'ymax': 50.04}}),
                    ('b', {{'xmin': 20.0, 'ymin': 60.0, 'xmax': 20.04, 'ymax': 60.04}})
                ) AS t(id, bbox)
            ) TO '{sql_path(path)}' (FORMAT PARQUET)
            """
        )
    finally:
        con.close()


class TestQuotedPathsGrid:
    """api.grid survives an apostrophe in the output and cache paths."""

    def test_get_grid_writes_output_and_cache(
        self, sample_fields_geoparquet: Path, quoted_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from ftw_dataset_tools.api.grid import get_grid

        monkeypatch.setenv("FTW_CACHE_DIR", str(quoted_dir / "cache"))
        fields = _copy_into(sample_fields_geoparquet, quoted_dir)
        grid = quoted_dir / "grid.parquet"
        _write_bbox_grid(grid)

        fresh = get_grid(fields, output_file=quoted_dir / "fresh.parquet", grid_source=str(grid))
        cached = get_grid(fields, output_file=quoted_dir / "cached.parquet", grid_source=str(grid))

        assert fresh.grid_count == cached.grid_count == 1
        assert fresh.output_path.exists()
        assert cached.output_path.exists()
        assert list((quoted_dir / "cache" / "grid").glob("grid_*.parquet"))


class TestQuotedPathsPipeline:
    """api.pipeline local-grid subsetting survives an apostrophe in the path."""

    def test_subset_local_grid(self, quoted_dir: Path) -> None:
        import duckdb

        from ftw_dataset_tools.api import pipeline
        from ftw_dataset_tools.api.config import DatasetConfig
        from ftw_dataset_tools.api.geo import sql_path

        fields = quoted_dir / "fields.parquet"
        grid = quoted_dir / "grid.parquet"
        con = duckdb.connect(":memory:")
        try:
            con.execute(
                "COPY (SELECT 1 AS id, "
                "{'xmin': 10.0, 'ymin': 50.0, 'xmax': 10.03, 'ymax': 50.03} AS bbox) "
                f"TO '{sql_path(fields)}' (FORMAT PARQUET)"
            )
        finally:
            con.close()
        _write_bbox_grid(grid)
        ctx = pipeline.PipelineContext(
            config=DatasetConfig.from_dict({"fields_file": str(fields)}),
            fields_input=fields,
            output_dir=quoted_dir,
            field_dataset="t",
            effective_year=None,
            has_temporal=False,
        )
        ctx.field_polygons_path = fields

        subset = pipeline._subset_local_grid(ctx, str(grid))

        assert subset == str(quoted_dir / "t_grid.parquet")
        con = duckdb.connect(":memory:")
        try:
            ids = [r[0] for r in con.execute(f"SELECT id FROM '{sql_path(subset)}'").fetchall()]
        finally:
            con.close()
        assert ids == ["a"]


class TestQuotedPathsStac:
    """api.stac column/extent helpers survive an apostrophe in the path."""

    def test_detect_datetime_column(self, sample_fields_geoparquet: Path, quoted_dir: Path) -> None:
        from ftw_dataset_tools.api.stac import detect_datetime_column

        fields = _copy_into(sample_fields_geoparquet, quoted_dir)

        # No datetime column in the fixture; the point is that DESCRIBE parses.
        assert detect_datetime_column(fields) is None

    def test_get_dataset_bounds(self, sample_fields_geoparquet: Path, quoted_dir: Path) -> None:
        from ftw_dataset_tools.api.stac import _get_dataset_bounds

        fields = _copy_into(sample_fields_geoparquet, quoted_dir)

        bbox = _get_dataset_bounds(fields, "geometry")

        assert len(bbox) == 4
        assert bbox[0] == pytest.approx(10.0)


class TestQuotedPathsMasks:
    """api.masks survives an apostrophe in the path."""

    def test_create_masks(
        self,
        sample_chips_with_coverage: Path,
        sample_boundaries_geoparquet: Path,
        sample_boundary_lines_geoparquet: Path,
        quoted_dir: Path,
    ) -> None:
        from ftw_dataset_tools.api.masks import MaskType, create_masks

        chips = _copy_into(sample_chips_with_coverage, quoted_dir)
        boundaries = _copy_into(sample_boundaries_geoparquet, quoted_dir)
        lines = _copy_into(sample_boundary_lines_geoparquet, quoted_dir)

        results = create_masks(
            chips_file=chips,
            boundaries_file=boundaries,
            boundary_lines_file=lines,
            output_dir=quoted_dir / "masks",
            field_dataset="test",
            mask_types=[MaskType.SEMANTIC_2_CLASS],
            num_workers=1,
        )

        assert results[MaskType.SEMANTIC_2_CLASS].total_created > 0
