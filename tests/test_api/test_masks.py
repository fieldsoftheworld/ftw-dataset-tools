"""Tests for the masks API."""

from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest


class TestMaskFilenameConvention:
    """Tests for mask filename generation."""

    def test_mask_filename_uses_grid_id_only(self) -> None:
        """Test that mask filenames use grid_id without dataset prefix."""
        from ftw_dataset_tools.api.masks import MaskType, get_mask_filename

        filename = get_mask_filename("abc123", MaskType.INSTANCE)
        assert filename == "abc123_instance.tif"
        assert "dataset" not in filename.lower()

    def test_mask_filename_semantic_2class(self) -> None:
        """Test semantic 2-class mask filename."""
        from ftw_dataset_tools.api.masks import MaskType, get_mask_filename

        filename = get_mask_filename("grid_001", MaskType.SEMANTIC_2_CLASS)
        assert filename == "grid_001_semantic_2_class.tif"

    def test_mask_filename_semantic_3class(self) -> None:
        """Test semantic 3-class mask filename."""
        from ftw_dataset_tools.api.masks import MaskType, get_mask_filename

        filename = get_mask_filename("grid_001", MaskType.SEMANTIC_3_CLASS)
        assert filename == "grid_001_semantic_3_class.tif"

    def test_mask_filename_with_year(self) -> None:
        """Test that mask filenames include year when provided."""
        from ftw_dataset_tools.api.masks import MaskType, get_mask_filename

        filename = get_mask_filename("ftw-34UFF1628", MaskType.INSTANCE, year=2024)
        assert filename == "ftw-34UFF1628_2024_instance.tif"

    def test_mask_filename_with_year_semantic(self) -> None:
        """Test semantic mask filename with year."""
        from ftw_dataset_tools.api.masks import MaskType, get_mask_filename

        filename = get_mask_filename("grid_001", MaskType.SEMANTIC_2_CLASS, year=2023)
        assert filename == "grid_001_2023_semantic_2_class.tif"


class TestGetItemId:
    """Tests for get_item_id function."""

    def test_item_id_without_year(self) -> None:
        """Test item ID generation without year."""
        from ftw_dataset_tools.api.masks import get_item_id

        item_id = get_item_id("ftw-34UFF1628")
        assert item_id == "ftw-34UFF1628"

    def test_item_id_with_year(self) -> None:
        """Test item ID generation with year."""
        from ftw_dataset_tools.api.masks import get_item_id

        item_id = get_item_id("ftw-34UFF1628", year=2024)
        assert item_id == "ftw-34UFF1628_2024"

    def test_item_id_with_none_year(self) -> None:
        """Test item ID generation with explicit None year."""
        from ftw_dataset_tools.api.masks import get_item_id

        item_id = get_item_id("grid_001", year=None)
        assert item_id == "grid_001"


class TestMaskOutputPath:
    """Tests for mask output path generation."""

    def test_output_path_with_chip_dirs(self) -> None:
        """Test mask path uses chip_dirs when provided."""
        from pathlib import Path

        from ftw_dataset_tools.api.masks import MaskType, get_mask_output_path

        chip_dirs = {
            "grid_001": Path("/output/chips/grid_001"),
            "grid_002": Path("/output/chips/grid_002"),
        }

        path = get_mask_output_path(
            grid_id="grid_001",
            mask_type=MaskType.INSTANCE,
            chip_dirs=chip_dirs,
            output_dir=Path("/output/masks"),
            field_dataset="test_dataset",
        )

        assert path == Path("/output/chips/grid_001/grid_001_instance.tif")

    def test_output_path_without_chip_dirs(self) -> None:
        """Test mask path uses output_dir with dataset prefix when chip_dirs is None."""
        from pathlib import Path

        from ftw_dataset_tools.api.masks import MaskType, get_mask_output_path

        path = get_mask_output_path(
            grid_id="grid_001",
            mask_type=MaskType.INSTANCE,
            chip_dirs=None,
            output_dir=Path("/output/masks"),
            field_dataset="test_dataset",
        )

        assert path == Path("/output/masks/test_dataset_grid_001_instance.tif")

    def test_output_path_with_year_and_chip_dirs(self) -> None:
        """Test mask path with year uses item_id for chip_dirs lookup."""
        from pathlib import Path

        from ftw_dataset_tools.api.masks import MaskType, get_mask_output_path

        # chip_dirs keyed by item_id (grid_id_year)
        chip_dirs = {
            "grid_001_2024": Path("/output/chips/grid_001_2024"),
        }

        path = get_mask_output_path(
            grid_id="grid_001",
            mask_type=MaskType.INSTANCE,
            chip_dirs=chip_dirs,
            output_dir=Path("/output/masks"),
            field_dataset="test_dataset",
            year=2024,
        )

        assert path == Path("/output/chips/grid_001_2024/grid_001_2024_instance.tif")

    def test_output_path_with_year_without_chip_dirs(self) -> None:
        """Test mask path with year includes year in filename."""
        from pathlib import Path

        from ftw_dataset_tools.api.masks import MaskType, get_mask_output_path

        path = get_mask_output_path(
            grid_id="grid_001",
            mask_type=MaskType.INSTANCE,
            chip_dirs=None,
            output_dir=Path("/output/masks"),
            field_dataset="test_dataset",
            year=2024,
        )

        assert path == Path("/output/masks/test_dataset_grid_001_2024_instance.tif")


class TestCreateMasksChipDirs:
    """Tests for chip_dirs parameter in create_masks."""

    def test_create_masks_accepts_chip_dirs_parameter(self) -> None:
        """Test that create_masks signature accepts chip_dirs parameter."""
        import inspect

        from ftw_dataset_tools.api.masks import create_masks

        sig = inspect.signature(create_masks)
        assert "chip_dirs" in sig.parameters
        # Should be optional (has default None)
        assert sig.parameters["chip_dirs"].default is None


class TestBackgroundClassValue:
    """Tests for background_class_value parameter."""

    def test_create_masks_accepts_background_class_value_parameter(self) -> None:
        """Test that create_masks signature accepts background_class_value parameter."""
        import inspect

        from ftw_dataset_tools.api.masks import create_masks

        sig = inspect.signature(create_masks)
        assert "background_class_value" in sig.parameters
        # Should have default value of 0
        assert sig.parameters["background_class_value"].default == 0

    def test_create_masks_for_cell_accepts_background_class_value_parameter(self) -> None:
        """Test that _create_masks_for_cell signature accepts background_class_value parameter."""
        import inspect

        from ftw_dataset_tools.api.masks import _create_masks_for_cell

        sig = inspect.signature(_create_masks_for_cell)
        assert "background_class_value" in sig.parameters
        # Should have default value of 0
        assert sig.parameters["background_class_value"].default == 0


def _mask_task(tmp_path, grid_id="grid_1", outputs=None, memory_limit_mb=2048):
    """Build a _MaskTask for a one-output semantic_2_class group."""
    from rasterio.crs import CRS

    from ftw_dataset_tools.api.masks import MaskType, _MaskTask

    if outputs is None:
        outputs = ((MaskType.SEMANTIC_2_CLASS, str(Path(tmp_path) / f"{grid_id}.tif")),)
    return _MaskTask(
        grid_id=grid_id,
        bounds=(0.0, 0.0, 1.0, 1.0),
        crs_wkt=CRS.from_epsg(4326).to_wkt(),
        boundaries_path=str(Path(tmp_path) / "boundaries.parquet"),
        boundary_lines_path=str(Path(tmp_path) / "lines.parquet"),
        boundaries_geom_col="geometry",
        boundary_lines_geom_col="geometry",
        source_type=MaskType.SEMANTIC_2_CLASS,
        outputs=outputs,
        resolution=10.0,
        id_col=None,
        background_class_value=0,
        memory_limit_mb=memory_limit_mb,
    )


class TestMaskTaskHashable:
    """_run_work_items tracks finished tasks in a set, so tasks must hash."""

    def test_task_is_hashable(self, tmp_path) -> None:
        task = _mask_task(tmp_path)
        assert task in {task}


class TestProcessSingleGridCellRetry:
    """_process_single_grid_cell should retry the whole group with a fresh connection."""

    def test_retries_once_then_succeeds(self, tmp_path, monkeypatch) -> None:
        from ftw_dataset_tools.api import masks

        calls = {"n": 0}

        def fake_create_masks_for_cell(**kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise ValueError("boom")
            return [
                (
                    mask_type,
                    masks.MaskResult(
                        grid_id=kwargs["grid_id"], output_path=path, width=1, height=1
                    ),
                )
                for mask_type, path in kwargs["outputs"]
            ]

        monkeypatch.setattr(masks, "_create_masks_for_cell", fake_create_masks_for_cell)

        results, error = masks._process_single_grid_cell(_mask_task(tmp_path))

        assert error is None
        assert [mask_type for mask_type, _ in results] == [masks.MaskType.SEMANTIC_2_CLASS]
        assert calls["n"] == 2

    def test_fails_twice_returns_error_with_exception_class_prefix(
        self, tmp_path, monkeypatch
    ) -> None:
        from ftw_dataset_tools.api import masks

        calls = {"n": 0}

        def fake_create_masks_for_cell(**_kwargs):
            calls["n"] += 1
            raise ValueError("boom")

        monkeypatch.setattr(masks, "_create_masks_for_cell", fake_create_masks_for_cell)

        results, error = masks._process_single_grid_cell(_mask_task(tmp_path))

        assert results == []
        assert calls["n"] == 2  # retried once, still failed
        grid_id, reason = error
        assert grid_id == "grid_1"
        assert reason == "ValueError: boom"

    def test_partial_write_failure_reports_what_reached_disk(self, tmp_path, monkeypatch) -> None:
        """A group whose 2nd output fails still reports the 1st as created."""
        from ftw_dataset_tools.api import masks

        outputs = (
            (masks.MaskType.SEMANTIC_2_CLASS, str(tmp_path / "a.tif")),
            (masks.MaskType.DECODE_BOUNDARY, str(tmp_path / "b.tif")),
        )

        def fake_create_masks_for_cell(**kwargs):
            mask_type, path = kwargs["outputs"][0]
            written = [
                (mask_type, masks.MaskResult(grid_id="grid_1", output_path=path, width=1, height=1))
            ]
            raise masks._PartialCellFailure(written, "decode_boundary: disk full")

        monkeypatch.setattr(masks, "_create_masks_for_cell", fake_create_masks_for_cell)

        results, error = masks._process_single_grid_cell(_mask_task(tmp_path, outputs=outputs))

        assert [mask_type for mask_type, _ in results] == [masks.MaskType.SEMANTIC_2_CLASS]
        assert error[0] == "grid_1"
        assert "disk full" in error[1]


class _FakeFuture:
    """A real concurrent.futures.Future, already completed at construction time.

    as_completed() relies on Future internals (condition variable, state), so
    a lightweight duck-typed stand-in doesn't work; wrapping the stdlib class
    and completing it synchronously does.
    """

    def __new__(cls, *, result=None, exception=None):
        import concurrent.futures as cf

        future = cf.Future()
        if exception is not None:
            future.set_exception(exception)
        else:
            future.set_result(result)
        return future


class _FakeExecutor:
    """Runs submitted callables synchronously in-process; records submissions."""

    instances: ClassVar[list["_FakeExecutor"]] = []

    def __init__(self, max_workers=None) -> None:
        self.max_workers = max_workers
        self.submitted: list[tuple] = []
        _FakeExecutor.instances.append(self)

    def submit(self, fn, item):
        self.submitted.append(item)
        try:
            result = fn(item)
        except Exception as exc:
            return _FakeFuture(exception=exc)
        return _FakeFuture(result=result)

    def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
        # Mirrors ProcessPoolExecutor.shutdown's signature; nothing to clean up
        # since submit() already ran everything synchronously.
        self.shutdown_calls = [*getattr(self, "shutdown_calls", []), (wait, cancel_futures)]


class TestDefaultWorkerCount:
    """Tests for the default (unspecified) worker count."""

    def test_capped_at_eight_on_a_large_machine(self, monkeypatch) -> None:
        from ftw_dataset_tools.api import masks

        monkeypatch.setattr(masks.os, "cpu_count", lambda: 32)
        assert masks._default_num_workers() == 8

    def test_uses_cpu_count_when_below_the_cap(self, monkeypatch) -> None:
        from ftw_dataset_tools.api import masks

        monkeypatch.setattr(masks.os, "cpu_count", lambda: 4)
        assert masks._default_num_workers() == 4

    def test_floors_at_one_when_cpu_count_is_unknown(self, monkeypatch) -> None:
        from ftw_dataset_tools.api import masks

        monkeypatch.setattr(masks.os, "cpu_count", lambda: None)
        assert masks._default_num_workers() == 1


class TestWorkerMemoryLimit:
    """Tests for the per-worker DuckDB memory budget."""

    def test_formula(self) -> None:
        from ftw_dataset_tools.api.masks import _worker_memory_limit_mb

        # 0.6 * 16 GiB / 4 workers / 1 MiB = 2457.6, truncated.
        assert _worker_memory_limit_mb(16 * 2**30, 4) == 2457

    def test_has_a_floor(self) -> None:
        from ftw_dataset_tools.api.masks import _worker_memory_limit_mb

        # A tiny machine with many workers must not get an unusably small budget.
        assert _worker_memory_limit_mb(1 * 2**30, 100) == 512

    def test_task_carries_a_positive_memory_limit(self, tmp_path, monkeypatch) -> None:
        from ftw_dataset_tools.api import masks

        chips_path, fields_path, lines_path = TestCreateMasksSkipExisting._build_inputs(tmp_path)

        def fake_worker(task):
            return (
                [
                    (
                        mask_type,
                        masks.MaskResult(
                            grid_id=task.grid_id, output_path=Path(path), width=1, height=1
                        ),
                    )
                    for mask_type, path in task.outputs
                ],
                None,
            )

        monkeypatch.setattr(masks, "_process_single_grid_cell", fake_worker)
        monkeypatch.setattr(masks, "ProcessPoolExecutor", _FakeExecutor)
        _FakeExecutor.instances.clear()

        results = masks.create_masks(
            chips_file=chips_path,
            boundaries_file=fields_path,
            boundary_lines_file=lines_path,
            output_dir=tmp_path / "masks",
            field_dataset="test",
            mask_types=[masks.MaskType.SEMANTIC_2_CLASS],
            num_workers=2,
        )

        assert results[masks.MaskType.SEMANTIC_2_CLASS].total_created == 2
        submitted = _FakeExecutor.instances[0].submitted
        assert len(submitted) == 2
        for task in submitted:
            assert isinstance(task.memory_limit_mb, int)
            assert task.memory_limit_mb > 0


class TestRunWorkItemsBrokenPool:
    """Coverage for restarting a ProcessPoolExecutor that a crashed worker broke."""

    @staticmethod
    def _item(tmp_path, grid_id: str):
        return _mask_task(tmp_path, grid_id=grid_id)

    @staticmethod
    def _worker_result(task):
        from ftw_dataset_tools.api import masks

        return [
            (
                mask_type,
                masks.MaskResult(grid_id=task.grid_id, output_path=Path(path), width=1, height=1),
            )
            for mask_type, path in task.outputs
        ]

    def test_recovers_after_one_broken_pool(self, tmp_path, monkeypatch) -> None:
        from ftw_dataset_tools.api import masks

        attempts: dict[str, int] = {}

        def fake_worker(task):
            attempts[task.grid_id] = attempts.get(task.grid_id, 0) + 1
            if task.grid_id == "g2" and attempts[task.grid_id] == 1:
                raise BrokenProcessPool("boom")
            return (self._worker_result(task), None)

        monkeypatch.setattr(masks, "_process_single_grid_cell", fake_worker)
        monkeypatch.setattr(masks, "ProcessPoolExecutor", _FakeExecutor)

        work_items = [self._item(tmp_path, g) for g in ("g1", "g2", "g3")]
        created, failures, pool_restarts = masks._run_work_items(
            work_items, num_workers=1, total_tasks=3, on_progress=None
        )

        assert {result.grid_id for _mask_type, result in created} == {"g1", "g2", "g3"}
        assert failures == []
        assert pool_restarts == 1

    def test_gives_up_after_max_restarts(self, tmp_path, monkeypatch) -> None:
        from ftw_dataset_tools.api import masks

        def fake_worker(task):
            if task.grid_id == "bad":
                raise BrokenProcessPool("boom")
            return (self._worker_result(task), None)

        monkeypatch.setattr(masks, "_process_single_grid_cell", fake_worker)
        monkeypatch.setattr(masks, "ProcessPoolExecutor", _FakeExecutor)

        work_items = [self._item(tmp_path, "good"), self._item(tmp_path, "bad")]
        created, failures, pool_restarts = masks._run_work_items(
            work_items, num_workers=1, total_tasks=2, on_progress=None
        )

        assert {result.grid_id for _mask_type, result in created} == {"good"}
        assert [error for _task, error, _written in failures] == [
            ("bad", "BrokenProcessPool: worker died repeatedly")
        ]
        assert pool_restarts == masks._MAX_POOL_RESTARTS

    def test_give_up_is_attributed_to_every_type_in_the_group(self, tmp_path, monkeypatch) -> None:
        """A dead group loses all of its outputs, not just the source type."""
        from ftw_dataset_tools.api import masks

        def fake_worker(_task):
            raise BrokenProcessPool("boom")

        monkeypatch.setattr(masks, "_process_single_grid_cell", fake_worker)
        monkeypatch.setattr(masks, "ProcessPoolExecutor", _FakeExecutor)

        outputs = (
            (masks.MaskType.SEMANTIC_2_CLASS, str(tmp_path / "a.tif")),
            (masks.MaskType.DECODE_BOUNDARY, str(tmp_path / "b.tif")),
        )
        work_items = [_mask_task(tmp_path, grid_id="g1", outputs=outputs)]
        _created, failures, _restarts = masks._run_work_items(
            work_items, num_workers=1, total_tasks=1, on_progress=None
        )

        (task, _error, written) = failures[0]
        assert written == set()
        assert [mask_type for mask_type, _ in task.outputs] == [
            masks.MaskType.SEMANTIC_2_CLASS,
            masks.MaskType.DECODE_BOUNDARY,
        ]

    def test_pool_restarts_reflected_in_create_masks_result(self, tmp_path, monkeypatch) -> None:
        """create_masks surfaces pool_restarts from _run_work_items on every result."""
        from ftw_dataset_tools.api import masks

        chips_path, fields_path, lines_path = TestCreateMasksSkipExisting._build_inputs(tmp_path)

        def fake_run_work_items(_work_items, _num_workers, _total_tasks, _on_progress):
            return ([], [], 2)

        monkeypatch.setattr(masks, "_run_work_items", fake_run_work_items)

        results = masks.create_masks(
            chips_file=chips_path,
            boundaries_file=fields_path,
            boundary_lines_file=lines_path,
            output_dir=tmp_path / "masks",
            field_dataset="test",
            mask_types=[masks.MaskType.SEMANTIC_2_CLASS],
            num_workers=1,
        )

        assert results[masks.MaskType.SEMANTIC_2_CLASS].pool_restarts == 2


class TestCreateMasksSkipExisting:
    """Tests for the skip_existing parameter of create_masks."""

    @staticmethod
    def _build_inputs(tmp_path):
        import geopandas as gpd
        from shapely.geometry import LineString, box

        crs = "EPSG:4326"
        chips = gpd.GeoDataFrame(
            {"id": ["c1", "c2"], "field_coverage_pct": [50.0, 50.0]},
            geometry=[box(10.0, 50.0, 10.01, 50.01), box(11.0, 50.0, 11.01, 50.01)],
            crs=crs,
        )
        chips_path = tmp_path / "chips.parquet"
        chips.to_parquet(chips_path)

        fields = gpd.GeoDataFrame(
            {"id": [1]}, geometry=[box(10.002, 50.002, 10.006, 50.006)], crs=crs
        )
        fields_path = tmp_path / "fields.parquet"
        fields.to_parquet(fields_path)

        lines = gpd.GeoDataFrame(
            {"id": [1]},
            geometry=[LineString([(10.002, 50.002), (10.006, 50.002)])],
            crs=crs,
        )
        lines_path = tmp_path / "lines.parquet"
        lines.to_parquet(lines_path)
        return chips_path, fields_path, lines_path

    def test_skip_existing_default_recreates_everything(self, tmp_path) -> None:
        from ftw_dataset_tools.api.masks import MaskType, create_masks

        chips_path, fields_path, lines_path = self._build_inputs(tmp_path)
        output_dir = tmp_path / "masks"

        results = create_masks(
            chips_file=chips_path,
            boundaries_file=fields_path,
            boundary_lines_file=lines_path,
            output_dir=output_dir,
            field_dataset="test",
            mask_types=[MaskType.SEMANTIC_2_CLASS],
            num_workers=1,
        )

        result = results[MaskType.SEMANTIC_2_CLASS]
        assert result.total_created == 2
        assert result.masks_existing == 0

    def test_skip_existing_skips_non_empty_file_and_recreates_empty_one(self, tmp_path) -> None:
        from ftw_dataset_tools.api.masks import (
            MaskType,
            create_masks,
            get_mask_output_path,
        )

        chips_path, fields_path, lines_path = self._build_inputs(tmp_path)
        output_dir = tmp_path / "masks"
        output_dir.mkdir()

        existing_path = get_mask_output_path(
            grid_id="c1",
            mask_type=MaskType.SEMANTIC_2_CLASS,
            chip_dirs=None,
            output_dir=output_dir,
            field_dataset="test",
        )
        existing_path.write_bytes(b"not-really-a-tif-but-non-empty")

        empty_path = get_mask_output_path(
            grid_id="c2",
            mask_type=MaskType.SEMANTIC_2_CLASS,
            chip_dirs=None,
            output_dir=output_dir,
            field_dataset="test",
        )
        empty_path.write_bytes(b"")

        results = create_masks(
            chips_file=chips_path,
            boundaries_file=fields_path,
            boundary_lines_file=lines_path,
            output_dir=output_dir,
            field_dataset="test",
            mask_types=[MaskType.SEMANTIC_2_CLASS],
            num_workers=1,
            skip_existing=True,
        )

        result = results[MaskType.SEMANTIC_2_CLASS]
        assert result.masks_existing == 1
        assert result.total_created == 1
        assert result.masks_created[0].grid_id == "c2"
        # The non-empty file was left untouched (still not a valid raster).
        assert existing_path.read_bytes() == b"not-really-a-tif-but-non-empty"
        # The empty file was recreated into a real raster.
        assert empty_path.stat().st_size > 0

    def test_skip_existing_filters_per_output_not_per_group(self, tmp_path) -> None:
        """A group whose 2-class mask exists must still burn its missing DECODE layer."""
        from ftw_dataset_tools.api.masks import (
            MaskType,
            create_masks,
            get_mask_output_path,
        )

        chips_path, fields_path, lines_path = self._build_inputs(tmp_path)
        output_dir = tmp_path / "masks"
        output_dir.mkdir()

        paths = {
            mask_type: get_mask_output_path(
                grid_id=grid_id,
                mask_type=mask_type,
                chip_dirs=None,
                output_dir=output_dir,
                field_dataset="test",
            )
            for grid_id in ("c1",)
            for mask_type in (MaskType.SEMANTIC_2_CLASS, MaskType.DECODE_BOUNDARY)
        }
        # Only the 2-class output of c1's group is already on disk.
        paths[MaskType.SEMANTIC_2_CLASS].write_bytes(b"not-really-a-tif-but-non-empty")

        results = create_masks(
            chips_file=chips_path,
            boundaries_file=fields_path,
            boundary_lines_file=lines_path,
            output_dir=output_dir,
            field_dataset="test",
            mask_types=[MaskType.SEMANTIC_2_CLASS, MaskType.DECODE_BOUNDARY],
            num_workers=1,
            skip_existing=True,
        )

        two_class = results[MaskType.SEMANTIC_2_CLASS]
        boundary = results[MaskType.DECODE_BOUNDARY]
        assert two_class.masks_existing == 1
        assert two_class.total_created == 1  # only c2
        assert boundary.masks_existing == 0
        assert boundary.total_created == 2  # c1's group still burned for the DECODE layer
        assert paths[MaskType.SEMANTIC_2_CLASS].read_bytes() == b"not-really-a-tif-but-non-empty"
        assert paths[MaskType.DECODE_BOUNDARY].stat().st_size > 0


class TestDecodeMaskTypes:
    """End-to-end tests for the derived DECODE mask types."""

    @staticmethod
    def _build_inputs(tmp_path):
        """Write a one-cell chips file plus two adjacent fields and their lines."""
        import geopandas as gpd
        from shapely.geometry import LineString, box

        crs = "EPSG:3035"
        cell = box(4000000, 3000000, 4001000, 3001000)

        chips = tmp_path / "chips.parquet"
        gpd.GeoDataFrame(
            {"id": ["grid_001"], "field_coverage_pct": [20.0]},
            geometry=[cell],
            crs=crs,
        ).to_parquet(chips)

        # Two fields with a 100m gap, so the burned boundary lines separate them.
        fields = [
            box(4000100, 3000100, 4000400, 3000400),
            box(4000500, 3000100, 4000800, 3000400),
        ]
        boundaries = tmp_path / "fields.parquet"
        gpd.GeoDataFrame({"id": [1, 2]}, geometry=fields, crs=crs).to_parquet(boundaries)

        lines = tmp_path / "lines.parquet"
        gpd.GeoDataFrame(
            {"id": [1, 2]},
            geometry=[LineString(f.exterior.coords) for f in fields],
            crs=crs,
        ).to_parquet(lines)

        return chips, boundaries, lines

    def _create(self, tmp_path, mask_type, **kwargs):
        """Run create_masks for one mask type and return the written raster path."""
        from ftw_dataset_tools.api.masks import create_masks

        chips, boundaries, lines = self._build_inputs(tmp_path)
        output_dir = tmp_path / mask_type.value
        result = create_masks(
            chips_file=chips,
            boundaries_file=boundaries,
            boundary_lines_file=lines,
            output_dir=output_dir,
            field_dataset="test",
            mask_types=[mask_type],
            num_workers=1,
            **kwargs,
        )[mask_type]

        assert result.total_created == 1, result.masks_skipped
        return result.masks_created[0].output_path

    def test_boundary_mask_is_uint8_and_binary(self, tmp_path) -> None:
        """The boundary layer is stored as uint8 holding only 0 and 1."""
        import rasterio

        from ftw_dataset_tools.api.masks import MaskType

        path = self._create(tmp_path, MaskType.DECODE_BOUNDARY)

        with rasterio.open(path) as src:
            data = src.read(1)
            assert src.dtypes[0] == "uint8"

        assert set(np.unique(data).tolist()) == {0, 1}
        assert data.sum() > 0

    def test_boundary_mask_traces_both_fields(self, tmp_path) -> None:
        """Each of the two fields gets its own closed boundary ring."""
        import rasterio
        from scipy import ndimage

        from ftw_dataset_tools.api.masks import MaskType

        path = self._create(tmp_path, MaskType.DECODE_BOUNDARY)

        with rasterio.open(path) as src:
            data = src.read(1)

        _, num_rings = ndimage.label(data)
        assert num_rings == 2

    def test_distance_map_is_float32_in_unit_range(self, tmp_path) -> None:
        """The distance layer is float32 normalized into [0, 1]."""
        import rasterio

        from ftw_dataset_tools.api.masks import MaskType

        path = self._create(tmp_path, MaskType.DECODE_DISTANCE)

        with rasterio.open(path) as src:
            data = src.read(1)
            assert src.dtypes[0] == "float32"

        assert data.min() == 0.0
        assert data.max() == 1.0

    def test_distance_map_records_normalization_divisor(self, tmp_path) -> None:
        """The pre-normalization maximum survives into the COG tags."""
        import rasterio

        from ftw_dataset_tools.api.masks import MaskType

        path = self._create(tmp_path, MaskType.DECODE_DISTANCE)

        with rasterio.open(path) as src:
            max_px = float(src.tags()["decode_distance_max_px"])
            data = src.read(1)

        # Fields are 300m wide at 10m resolution, so the centre sits ~15px in.
        assert max_px == pytest.approx(15.0, abs=1.0)
        assert (data * max_px).max() == pytest.approx(max_px)

    def test_decode_layers_align_with_semantic_2_class(self, tmp_path) -> None:
        """Derived layers share the grid and georeferencing of their source mask."""
        import rasterio

        from ftw_dataset_tools.api.masks import MaskType

        semantic = self._create(tmp_path, MaskType.SEMANTIC_2_CLASS)
        boundary = self._create(tmp_path, MaskType.DECODE_BOUNDARY)
        distance = self._create(tmp_path, MaskType.DECODE_DISTANCE)

        with rasterio.open(semantic) as src:
            shape, transform, crs = src.shape, src.transform, src.crs
            field = src.read(1) > 0

        for path in (boundary, distance):
            with rasterio.open(path) as src:
                assert src.shape == shape
                assert src.transform == transform
                assert src.crs == crs
                # Neither layer may light up outside the field extent.
                assert not (src.read(1) > 0)[~field].any()

    def test_presence_only_background_excluded(self, tmp_path) -> None:
        """Background value 3 is treated as background, not as a field."""
        import rasterio

        from ftw_dataset_tools.api.masks import MaskType

        path = self._create(tmp_path, MaskType.DECODE_DISTANCE, background_class_value=3)

        with rasterio.open(path) as src:
            data = src.read(1)

        # Only the two fields carry distance; the rest of the chip stays at zero.
        assert data.max() == 1.0
        assert (data == 0).sum() > (data > 0).sum()

    def test_chip_with_no_fields_writes_empty_layers(self, tmp_path) -> None:
        """A chip whose cell contains no fields still writes valid, all-zero rasters."""
        import geopandas as gpd
        import rasterio
        from shapely.geometry import LineString, box

        from ftw_dataset_tools.api.masks import MaskType, create_masks

        crs = "EPSG:3035"
        # The chip cell and the fields do not overlap.
        chips = tmp_path / "chips.parquet"
        gpd.GeoDataFrame(
            {"id": ["empty_001"], "field_coverage_pct": [0.0]},
            geometry=[box(4000000, 3000000, 4001000, 3001000)],
            crs=crs,
        ).to_parquet(chips)

        far_away = box(4900000, 3900000, 4900300, 3900300)
        boundaries = tmp_path / "fields.parquet"
        gpd.GeoDataFrame({"id": [1]}, geometry=[far_away], crs=crs).to_parquet(boundaries)
        lines = tmp_path / "lines.parquet"
        gpd.GeoDataFrame(
            {"id": [1]}, geometry=[LineString(far_away.exterior.coords)], crs=crs
        ).to_parquet(lines)

        for mask_type in (MaskType.DECODE_BOUNDARY, MaskType.DECODE_DISTANCE):
            result = create_masks(
                chips_file=chips,
                boundaries_file=boundaries,
                boundary_lines_file=lines,
                output_dir=tmp_path / mask_type.value,
                field_dataset="test",
                mask_types=[mask_type],
                min_coverage=0.0,
                num_workers=1,
            )[mask_type]
            assert result.total_created == 1, result.masks_skipped

            with rasterio.open(result.masks_created[0].output_path) as src:
                assert src.read(1).max() == 0

    def test_empty_chip_records_zero_normalization_divisor(self, tmp_path) -> None:
        """The distance tag is still written (as 0) when there is nothing to normalize."""
        import geopandas as gpd
        import rasterio
        from shapely.geometry import LineString, box

        from ftw_dataset_tools.api.masks import MaskType, create_masks

        crs = "EPSG:3035"
        chips = tmp_path / "chips.parquet"
        gpd.GeoDataFrame(
            {"id": ["empty_001"], "field_coverage_pct": [0.0]},
            geometry=[box(4000000, 3000000, 4001000, 3001000)],
            crs=crs,
        ).to_parquet(chips)
        far_away = box(4900000, 3900000, 4900300, 3900300)
        boundaries = tmp_path / "fields.parquet"
        gpd.GeoDataFrame({"id": [1]}, geometry=[far_away], crs=crs).to_parquet(boundaries)
        lines = tmp_path / "lines.parquet"
        gpd.GeoDataFrame(
            {"id": [1]}, geometry=[LineString(far_away.exterior.coords)], crs=crs
        ).to_parquet(lines)

        result = create_masks(
            chips_file=chips,
            boundaries_file=boundaries,
            boundary_lines_file=lines,
            output_dir=tmp_path / "dist",
            field_dataset="test",
            mask_types=[MaskType.DECODE_DISTANCE],
            min_coverage=0.0,
            num_workers=1,
        )[MaskType.DECODE_DISTANCE]

        with rasterio.open(result.masks_created[0].output_path) as src:
            assert float(src.tags()["decode_distance_max_px"]) == 0.0


class TestGridRasterGeometry:
    """Tests for the extracted grid geometry helper."""

    def test_projected_crs_dimensions(self) -> None:
        """A 1km cell at 10m resolution is 100x100 pixels."""
        from rasterio.crs import CRS

        from ftw_dataset_tools.api.masks import _grid_raster_geometry

        _, width, height = _grid_raster_geometry(
            bounds=(4000000, 3000000, 4001000, 3001000),
            crs=CRS.from_epsg(3035),
            resolution=10.0,
        )

        assert (width, height) == (100, 100)

    def test_geographic_crs_converts_resolution_to_degrees(self) -> None:
        """Metres are approximated as degrees for a geographic CRS."""
        from rasterio.crs import CRS

        from ftw_dataset_tools.api.masks import _grid_raster_geometry

        _, width, height = _grid_raster_geometry(
            bounds=(10.0, 50.0, 10.01, 50.01),
            crs=CRS.from_epsg(4326),
            resolution=10.0,
        )

        # 0.01 degrees / (10 / 111000) degrees per pixel = 110.99..., truncated by int()
        assert width == height == 110

    def test_cell_too_small_for_resolution_raises(self) -> None:
        """A cell smaller than one pixel is an error, not a zero-sized raster."""
        from rasterio.crs import CRS

        from ftw_dataset_tools.api.masks import _grid_raster_geometry

        with pytest.raises(ValueError, match="too small for resolution"):
            _grid_raster_geometry(
                bounds=(4000000, 3000000, 4000005, 3000005),
                crs=CRS.from_epsg(3035),
                resolution=10.0,
            )


class TestDeriveDecodeLayer:
    """Tests for the dispatch between rasterized and derived mask types."""

    def test_boundary_returns_no_tags(self) -> None:
        """The boundary layer carries no extra metadata."""
        from ftw_dataset_tools.api.masks import MaskType, _derive_decode_layer

        source = np.zeros((8, 8), dtype=np.uint8)
        source[2:6, 2:6] = 1

        array, tags = _derive_decode_layer(MaskType.DECODE_BOUNDARY, source)

        assert array.dtype == np.uint8
        assert tags == {}

    def test_distance_returns_normalization_tag(self) -> None:
        """The distance layer records the divisor it used."""
        from ftw_dataset_tools.api.masks import MaskType, _derive_decode_layer

        source = np.zeros((12, 12), dtype=np.uint8)
        source[1:11, 1:11] = 1

        array, tags = _derive_decode_layer(MaskType.DECODE_DISTANCE, source)

        assert array.dtype == np.float32
        assert float(tags["decode_distance_max_px"]) == 5.0

    def test_derived_types_are_registered(self) -> None:
        """Both DECODE types must be in the derived set, or they'd be rasterized."""
        from ftw_dataset_tools.api.masks import _DERIVED_MASK_TYPES, MaskType

        assert MaskType.DECODE_BOUNDARY in _DERIVED_MASK_TYPES
        assert MaskType.DECODE_DISTANCE in _DERIVED_MASK_TYPES
        # The rasterized types must not be, or they'd be derived from 2-class.
        assert MaskType.INSTANCE not in _DERIVED_MASK_TYPES
        assert MaskType.SEMANTIC_2_CLASS not in _DERIVED_MASK_TYPES
        assert MaskType.SEMANTIC_3_CLASS not in _DERIVED_MASK_TYPES


class TestSharedRasterization:
    """The DECODE layers share one burn with the 2-class mask they derive from."""

    def _inputs(self, tmp_path):
        return TestDecodeMaskTypes._build_inputs(tmp_path)

    def test_grouping_matches_one_call_per_type_byte_for_byte(self, tmp_path) -> None:
        """Grouped output must be identical to rasterizing each type separately."""
        import hashlib

        from ftw_dataset_tools.api.masks import MaskType, create_masks

        requested = [
            MaskType.SEMANTIC_2_CLASS,
            MaskType.DECODE_BOUNDARY,
            MaskType.DECODE_DISTANCE,
        ]
        chips, boundaries, lines = self._inputs(tmp_path)

        def run(output_dir, mask_types):
            return create_masks(
                chips_file=chips,
                boundaries_file=boundaries,
                boundary_lines_file=lines,
                output_dir=output_dir,
                field_dataset="test",
                mask_types=mask_types,
                num_workers=1,
            )

        grouped = run(tmp_path / "grouped", requested)
        separate = {
            mask_type: run(tmp_path / f"separate_{mask_type.value}", [mask_type])[mask_type]
            for mask_type in requested
        }

        for mask_type in requested:
            grouped_path = grouped[mask_type].masks_created[0].output_path
            separate_path = separate[mask_type].masks_created[0].output_path
            assert (
                hashlib.sha256(grouped_path.read_bytes()).hexdigest()
                == hashlib.sha256(separate_path.read_bytes()).hexdigest()
            ), f"{mask_type.value} differs when grouped"

    def test_shared_group_rasterizes_once(self, tmp_path, monkeypatch) -> None:
        """Three shared outputs must cost one burn, not three."""
        import duckdb
        from rasterio.crs import CRS

        from ftw_dataset_tools.api import masks as masks_module
        from ftw_dataset_tools.api.geo import ensure_spatial_loaded
        from ftw_dataset_tools.api.masks import MaskType, _create_masks_for_cell

        calls = []
        original = masks_module._rasterize_mask

        def counting_rasterize(**kwargs):
            calls.append(kwargs["mask_type"])
            return original(**kwargs)

        monkeypatch.setattr(masks_module, "_rasterize_mask", counting_rasterize)

        # create_masks runs its workers in subprocesses, which would not see the
        # patch, so exercise the in-process function the workers call.
        _, boundaries, lines = self._inputs(tmp_path)
        outputs = [
            (MaskType.SEMANTIC_2_CLASS, tmp_path / "2class.tif"),
            (MaskType.DECODE_BOUNDARY, tmp_path / "boundary.tif"),
            (MaskType.DECODE_DISTANCE, tmp_path / "distance.tif"),
        ]

        conn = duckdb.connect(":memory:")
        ensure_spatial_loaded(conn)
        try:
            results = _create_masks_for_cell(
                conn=conn,
                grid_id="grid_001",
                bounds=(4000000, 3000000, 4001000, 3001000),
                crs=CRS.from_epsg(3035),
                boundaries_path=boundaries,
                boundary_lines_path=lines,
                boundaries_geom_col="geometry",
                boundary_lines_geom_col="geometry",
                source_type=MaskType.SEMANTIC_2_CLASS,
                outputs=outputs,
            )
        finally:
            conn.close()

        assert calls == [MaskType.SEMANTIC_2_CLASS], calls
        assert [mask_type for mask_type, _ in results] == [mask_type for mask_type, _ in outputs]
        for _, path in outputs:
            assert path.exists()

    def test_unrelated_types_keep_their_own_burn(self) -> None:
        """instance and 3-class need their own rasterization; they must not merge."""
        from ftw_dataset_tools.api.masks import MaskType, _group_by_source

        groups = _group_by_source(
            [
                MaskType.INSTANCE,
                MaskType.SEMANTIC_2_CLASS,
                MaskType.SEMANTIC_3_CLASS,
                MaskType.DECODE_BOUNDARY,
                MaskType.DECODE_DISTANCE,
            ]
        )

        assert groups == {
            MaskType.INSTANCE: [MaskType.INSTANCE],
            MaskType.SEMANTIC_2_CLASS: [
                MaskType.SEMANTIC_2_CLASS,
                MaskType.DECODE_BOUNDARY,
                MaskType.DECODE_DISTANCE,
            ],
            MaskType.SEMANTIC_3_CLASS: [MaskType.SEMANTIC_3_CLASS],
        }

    def test_decode_alone_still_burns_its_source(self) -> None:
        """The standalone command asks for one DECODE layer with no 2-class output."""
        from ftw_dataset_tools.api.masks import MaskType, _group_by_source

        assert _group_by_source([MaskType.DECODE_DISTANCE]) == {
            MaskType.SEMANTIC_2_CLASS: [MaskType.DECODE_DISTANCE]
        }

    def test_every_requested_type_gets_a_result(self, tmp_path) -> None:
        """Callers index the result by type, so every requested type must be a key."""
        from ftw_dataset_tools.api.masks import MaskType, create_masks

        requested = [
            MaskType.SEMANTIC_3_CLASS,
            MaskType.SEMANTIC_2_CLASS,
            MaskType.DECODE_BOUNDARY,
        ]
        chips, boundaries, lines = self._inputs(tmp_path)

        results = create_masks(
            chips_file=chips,
            boundaries_file=boundaries,
            boundary_lines_file=lines,
            output_dir=tmp_path / "out",
            field_dataset="test",
            mask_types=requested,
            num_workers=1,
        )

        assert set(results) == set(requested)
        for mask_type in requested:
            assert results[mask_type].total_created == 1, results[mask_type].masks_skipped
            assert results[mask_type].masks_created[0].output_path.exists()

    def test_write_failure_keeps_the_outputs_already_written(self, tmp_path) -> None:
        """A failed 2nd output must not discard the 1st, which is on disk."""
        from rasterio.crs import CRS

        from ftw_dataset_tools.api.masks import (
            MaskType,
            _MaskTask,
            _process_single_grid_cell,
        )

        _, boundaries, lines = self._inputs(tmp_path)
        written = tmp_path / "2class.tif"
        blocked = tmp_path / "boundary.tif"
        # A directory where the raster should go: rasterio cannot write over it.
        blocked.mkdir()

        results, error = _process_single_grid_cell(
            _MaskTask(
                grid_id="grid_001",
                bounds=(4000000, 3000000, 4001000, 3001000),
                crs_wkt=CRS.from_epsg(3035).to_wkt(),
                boundaries_path=str(boundaries),
                boundary_lines_path=str(lines),
                boundaries_geom_col="geometry",
                boundary_lines_geom_col="geometry",
                source_type=MaskType.SEMANTIC_2_CLASS,
                outputs=(
                    (MaskType.SEMANTIC_2_CLASS, str(written)),
                    (MaskType.DECODE_BOUNDARY, str(blocked)),
                ),
                resolution=10.0,
                id_col=None,
                background_class_value=0,
                memory_limit_mb=2048,
            )
        )

        assert [mask_type for mask_type, _ in results] == [MaskType.SEMANTIC_2_CLASS]
        assert written.exists()
        assert error is not None
        assert error[0] == "grid_001"
        assert "decode_boundary" in error[1]

    def test_write_failure_is_only_reported_for_the_unwritten_type(self, tmp_path) -> None:
        """The type that succeeded counts as created, not as skipped."""
        from ftw_dataset_tools.api.masks import MaskType, create_masks, get_mask_output_path

        chips, boundaries, lines = self._inputs(tmp_path)
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        blocked = get_mask_output_path(
            grid_id="grid_001",
            mask_type=MaskType.DECODE_BOUNDARY,
            chip_dirs=None,
            output_dir=output_dir,
            field_dataset="test",
        )
        blocked.mkdir(parents=True)

        results = create_masks(
            chips_file=chips,
            boundaries_file=boundaries,
            boundary_lines_file=lines,
            output_dir=output_dir,
            field_dataset="test",
            mask_types=[MaskType.SEMANTIC_2_CLASS, MaskType.DECODE_BOUNDARY],
            num_workers=1,
        )

        assert results[MaskType.SEMANTIC_2_CLASS].total_created == 1
        assert results[MaskType.SEMANTIC_2_CLASS].total_skipped == 0
        assert results[MaskType.DECODE_BOUNDARY].total_created == 0
        assert results[MaskType.DECODE_BOUNDARY].total_skipped == 1

    def test_duplicate_mask_types_are_written_and_counted_once(self, tmp_path) -> None:
        """A repeated type must not write the same path twice or double-count it."""
        from ftw_dataset_tools.api.masks import MaskType, create_masks

        chips, boundaries, lines = self._inputs(tmp_path)
        starts: list[tuple[int, int, int]] = []

        results = create_masks(
            chips_file=chips,
            boundaries_file=boundaries,
            boundary_lines_file=lines,
            output_dir=tmp_path / "out",
            field_dataset="test",
            mask_types=[
                MaskType.SEMANTIC_2_CLASS,
                MaskType.SEMANTIC_2_CLASS,
                MaskType.DECODE_BOUNDARY,
            ],
            num_workers=1,
            on_start=lambda *args: starts.append(args),
        )

        assert set(results) == {MaskType.SEMANTIC_2_CLASS, MaskType.DECODE_BOUNDARY}
        assert results[MaskType.SEMANTIC_2_CLASS].total_created == 1
        assert results[MaskType.DECODE_BOUNDARY].total_created == 1
        # One cell, one group after de-duplication: one task, not two.
        assert starts == [(1, 1, 1)]

    def test_on_start_announces_the_total_the_progress_bar_counts_to(self, tmp_path) -> None:
        """on_start's task total must be cells x groups, matching on_progress."""
        from ftw_dataset_tools.api.masks import MaskType, create_masks

        chips, boundaries, lines = self._inputs(tmp_path)
        starts: list[tuple[int, int, int]] = []
        progress: list[tuple[int, int]] = []

        create_masks(
            chips_file=chips,
            boundaries_file=boundaries,
            boundary_lines_file=lines,
            output_dir=tmp_path / "out",
            field_dataset="test",
            # Two groups: instance burns on its own, the DECODE layer shares 2-class.
            mask_types=[
                MaskType.INSTANCE,
                MaskType.SEMANTIC_2_CLASS,
                MaskType.DECODE_BOUNDARY,
            ],
            num_workers=1,
            on_start=lambda *args: starts.append(args),
            on_progress=lambda current, total: progress.append((current, total)),
        )

        # One cell x two groups.
        assert starts == [(1, 1, 2)]
        assert [total for _, total in progress] == [2, 2]
        assert progress[-1][0] == 2


class TestMaskCogStatistics:
    """Masks must carry embedded band statistics inside the COG."""

    def _write_inputs(self, tmp_path: Path) -> tuple[Path, Path]:
        import geopandas as gpd
        from shapely.geometry import LineString, box

        fields = gpd.GeoDataFrame(
            {"id": [1]}, geometry=[box(10.002, 50.002, 10.006, 50.006)], crs="EPSG:4326"
        )
        fields_path = tmp_path / "fields.parquet"
        fields.to_parquet(fields_path)

        lines = gpd.GeoDataFrame(
            {"id": [1]},
            geometry=[LineString([(10.002, 50.002), (10.006, 50.002)])],
            crs="EPSG:4326",
        )
        lines_path = tmp_path / "lines.parquet"
        lines.to_parquet(lines_path)
        return fields_path, lines_path

    def _write_one_mask(self, tmp_path: Path, mask_type, out: Path) -> None:
        """Write one mask through the shared-rasterization entry point."""
        import duckdb
        from rasterio.crs import CRS

        from ftw_dataset_tools.api.geo import ensure_spatial_loaded
        from ftw_dataset_tools.api.masks import _create_masks_for_cell, _source_mask_type

        fields_path, lines_path = self._write_inputs(tmp_path)
        conn = duckdb.connect(":memory:")
        ensure_spatial_loaded(conn)
        try:
            _create_masks_for_cell(
                conn=conn,
                grid_id="g1",
                bounds=(10.0, 50.0, 10.01, 50.01),
                crs=CRS.from_epsg(4326),
                boundaries_path=fields_path,
                boundary_lines_path=lines_path,
                boundaries_geom_col="geometry",
                boundary_lines_geom_col="geometry",
                source_type=_source_mask_type(mask_type),
                outputs=[(mask_type, out)],
                resolution=10.0,
            )
        finally:
            conn.close()

    def test_single_mask_has_embedded_stats(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.masks import MaskType
        from ftw_dataset_tools.api.raster_stats import read_band_stats

        out = tmp_path / "mask.tif"
        self._write_one_mask(tmp_path, MaskType.SEMANTIC_3_CLASS, out)

        stats = read_band_stats(out, 1)
        assert stats is not None
        assert stats.minimum == 0.0
        assert stats.maximum == 2.0
        assert 0.0 < stats.mean < 2.0
        assert stats.valid_percent is None
        assert not out.with_name(out.name + ".aux.xml").exists()
        assert not out.with_suffix(".tmp.tif").exists()

    def test_single_mask_is_cog(self, tmp_path: Path) -> None:
        import rasterio

        from ftw_dataset_tools.api.masks import MaskType

        out = tmp_path / "mask2.tif"
        self._write_one_mask(tmp_path, MaskType.SEMANTIC_2_CLASS, out)

        with rasterio.open(out) as src:
            assert src.profile["tiled"] is True
            assert src.profile["compress"] == "deflate"
            assert src.dtypes[0] == "uint8"

    def test_decode_distance_keeps_tags_and_stats(self, tmp_path: Path) -> None:
        import rasterio

        from ftw_dataset_tools.api.masks import MaskType
        from ftw_dataset_tools.api.raster_stats import read_band_stats

        out = tmp_path / "dist.tif"
        self._write_one_mask(tmp_path, MaskType.DECODE_DISTANCE, out)

        with rasterio.open(out) as src:
            assert "decode_distance_max_px" in src.tags()
            assert src.dtypes[0] == "float32"
        stats = read_band_stats(out, 1)
        assert stats is not None
        assert 0.0 <= stats.minimum <= stats.maximum <= 1.0


class TestInstanceValue:
    """Tests for coercing raw field ids into usable instance mask values."""

    def test_float_like_string_parses(self) -> None:
        from ftw_dataset_tools.api.masks import _instance_value

        assert _instance_value("111205887.0") == 111205887

    def test_plain_int_passthrough(self) -> None:
        from ftw_dataset_tools.api.masks import _instance_value

        assert _instance_value(42) == 42

    def test_float_is_truncated(self) -> None:
        from ftw_dataset_tools.api.masks import _instance_value

        assert _instance_value(42.9) == 42

    def test_non_numeric_string_is_none(self) -> None:
        from ftw_dataset_tools.api.masks import _instance_value

        assert _instance_value("abc") is None

    def test_none_is_none(self) -> None:
        from ftw_dataset_tools.api.masks import _instance_value

        assert _instance_value(None) is None

    def test_bool_is_none(self) -> None:
        """bool is an int subclass in Python; treat it as not-an-id anyway."""
        from ftw_dataset_tools.api.masks import _instance_value

        assert _instance_value(True) is None


class TestRasterizeMaskInstanceIds:
    """End-to-end coverage for the instance id coercion/fallback in _rasterize_mask."""

    @staticmethod
    def _build_inputs(tmp_path, ids):
        import geopandas as gpd
        from shapely.geometry import LineString, box

        crs = "EPSG:3035"
        fields = gpd.GeoDataFrame(
            {"id": ids},
            geometry=[
                box(4000000, 3000000, 4000100, 3000100),
                box(4000200, 3000000, 4000300, 3000100),
                box(4000400, 3000000, 4000500, 3000100),
            ],
            crs=crs,
        )
        fields_path = tmp_path / "fields.parquet"
        fields.to_parquet(fields_path)

        # Far away, so it never intersects the raster bounds used below.
        lines_path = tmp_path / "lines.parquet"
        gpd.GeoDataFrame({"id": [1]}, geometry=[LineString([(0, 0), (1, 1)])], crs=crs).to_parquet(
            lines_path
        )
        return fields_path, lines_path

    def _rasterize(self, tmp_path, ids, background_class_value: int = 0):
        import duckdb
        from rasterio.crs import CRS

        from ftw_dataset_tools.api.geo import ensure_spatial_loaded
        from ftw_dataset_tools.api.masks import MaskType, _grid_raster_geometry, _rasterize_mask

        fields_path, lines_path = self._build_inputs(tmp_path, ids)
        bounds = (4000000, 3000000, 4000600, 3000200)
        conn = duckdb.connect(":memory:")
        ensure_spatial_loaded(conn)
        transform, width, height = _grid_raster_geometry(
            bounds=bounds, crs=CRS.from_epsg(3035), resolution=10.0
        )
        return _rasterize_mask(
            conn=conn,
            boundaries_path=fields_path,
            boundary_lines_path=lines_path,
            boundaries_geom_col="geometry",
            boundary_lines_geom_col="geometry",
            bounds=bounds,
            transform=transform,
            width=width,
            height=height,
            mask_type=MaskType.INSTANCE,
            id_col="id",
            background_class_value=background_class_value,
        )

    def test_valid_ids_are_preserved(self, tmp_path) -> None:
        mask = self._rasterize(tmp_path, ["5", "10", "15"])
        unique_nonzero = set(np.unique(mask).tolist()) - {0}
        assert unique_nonzero == {5, 10, 15}

    def test_mixed_valid_and_invalid_ids_fall_back_to_sequential(self, tmp_path) -> None:
        """A non-numeric or float-like id (e.g. Austria's '111205887.0') must not

        crash the whole cell; every id in the cell falls back to sequential
        numbering so ids stay unique.
        """
        mask = self._rasterize(tmp_path, ["111205887.0", "abc", "3"])
        unique_nonzero = set(np.unique(mask).tolist()) - {0}
        assert unique_nonzero == {1, 2, 3}

    def test_fallback_ids_skip_a_presence_only_background(self, tmp_path) -> None:
        """With background 3, a plain 1..n fallback would burn the third field as

        background and make it vanish. The fallback has to step over 3.
        """
        mask = self._rasterize(tmp_path, ["abc", "abc", "abc"], background_class_value=3)
        burned = set(np.unique(mask).tolist()) - {3}
        assert burned == {1, 2, 4}


class TestFallbackInstanceIds:
    """The fallback id generator must never hand out the background value."""

    def test_default_background_is_a_plain_range(self) -> None:
        from ftw_dataset_tools.api.masks import _fallback_instance_ids

        assert _fallback_instance_ids(4, background_class_value=0) == [1, 2, 3, 4]

    def test_presence_only_background_is_skipped(self) -> None:
        from ftw_dataset_tools.api.masks import _fallback_instance_ids

        assert _fallback_instance_ids(5, background_class_value=3) == [1, 2, 4, 5, 6]

    def test_no_shapes_needs_no_ids(self) -> None:
        from ftw_dataset_tools.api.masks import _fallback_instance_ids

        assert _fallback_instance_ids(0, background_class_value=3) == []

    def test_ids_stay_unique(self) -> None:
        from ftw_dataset_tools.api.masks import _fallback_instance_ids

        ids = _fallback_instance_ids(20, background_class_value=7)
        assert len(set(ids)) == len(ids)
        assert 7 not in ids


class TestMgrsSquare:
    def test_ftw_grid_id(self) -> None:
        from ftw_dataset_tools.api.masks import get_mgrs_square

        assert get_mgrs_square("ftw-33UXP0410") == "33UXP"
        assert get_mgrs_square("ftw-1CDE0001") == "1CDE"

    def test_non_ftw_id_goes_to_other(self) -> None:
        from ftw_dataset_tools.api.masks import get_mgrs_square

        assert get_mgrs_square("grid_001") == "other"
        assert get_mgrs_square("ftw-abc") == "other"


class TestMaskWriteIsAtomic:
    """A mask must never be visible at its destination path until it is complete.

    ``skip_existing`` treats any non-empty file as finished, and the writer
    creates the destination the moment it opens it, so a worker killed mid-write
    would otherwise leave a truncated file that a gap-filling rerun reuses.
    """

    @staticmethod
    def _write_args():
        from affine import Affine
        from rasterio.crs import CRS

        mask = np.ones((8, 8), dtype=np.uint8)
        return mask, CRS.from_epsg(4326), Affine.translation(0, 1) * Affine.scale(0.1, -0.1)

    def test_successful_write_leaves_only_the_destination(self, tmp_path) -> None:
        from ftw_dataset_tools.api.masks import _write_mask_raster

        mask, crs, transform = self._write_args()
        output_path = tmp_path / "mask.tif"

        _write_mask_raster(mask, output_path, crs, transform)

        assert output_path.exists()
        assert [p.name for p in tmp_path.iterdir()] == ["mask.tif"]

    def test_failed_write_leaves_no_destination_file(self, tmp_path, monkeypatch) -> None:
        from ftw_dataset_tools.api import masks

        def boom(*_args, **_kwargs):
            raise RuntimeError("killed mid-write")

        monkeypatch.setattr(masks, "embed_band_stats", boom)

        mask, crs, transform = self._write_args()
        output_path = tmp_path / "mask.tif"

        with pytest.raises(RuntimeError):
            masks._write_mask_raster(mask, output_path, crs, transform)

        assert not output_path.exists()
        assert list(tmp_path.iterdir()) == []

    def test_a_crashed_run_does_not_leave_a_reusable_file(self, tmp_path, monkeypatch) -> None:
        """End to end: a run that dies while writing leaves nothing skip_existing

        could mistake for a finished mask.
        """
        from ftw_dataset_tools.api import masks

        chips_path, fields_path, lines_path = TestCreateMasksSkipExisting._build_inputs(tmp_path)
        output_dir = tmp_path / "masks"

        def boom(*_args, **_kwargs):
            raise RuntimeError("killed mid-write")

        monkeypatch.setattr(masks, "embed_band_stats", boom)
        monkeypatch.setattr(masks, "ProcessPoolExecutor", _FakeExecutor)
        _FakeExecutor.instances.clear()

        results = masks.create_masks(
            chips_file=chips_path,
            boundaries_file=fields_path,
            boundary_lines_file=lines_path,
            output_dir=output_dir,
            field_dataset="test",
            mask_types=[masks.MaskType.SEMANTIC_2_CLASS],
            num_workers=1,
        )

        assert results[masks.MaskType.SEMANTIC_2_CLASS].total_created == 0
        assert list(output_dir.iterdir()) == []


class TestOnStartCountsQueuedTasks:
    """The progress total has to be taken after the skip_existing filter."""

    def test_total_tasks_excludes_masks_already_on_disk(self, tmp_path) -> None:
        from ftw_dataset_tools.api.masks import MaskType, create_masks, get_mask_output_path

        chips_path, fields_path, lines_path = TestCreateMasksSkipExisting._build_inputs(tmp_path)
        output_dir = tmp_path / "masks"

        common = {
            "chips_file": chips_path,
            "boundaries_file": fields_path,
            "boundary_lines_file": lines_path,
            "output_dir": output_dir,
            "field_dataset": "test",
            "mask_types": [MaskType.SEMANTIC_2_CLASS],
            "num_workers": 1,
        }
        create_masks(**common)

        # Drop one of the two masks; only that one should be queued on the rerun.
        get_mask_output_path(
            grid_id="c2",
            mask_type=MaskType.SEMANTIC_2_CLASS,
            chip_dirs=None,
            output_dir=output_dir,
            field_dataset="test",
        ).unlink()

        seen: list[tuple[int, int, int]] = []
        progress: list[tuple[int, int]] = []
        results = create_masks(
            **common,
            skip_existing=True,
            on_start=lambda total, filtered, tasks: seen.append((total, filtered, tasks)),
            on_progress=lambda current, total: progress.append((current, total)),
        )

        assert seen == [(2, 2, 1)]
        assert progress[-1] == (1, 1)
        assert results[MaskType.SEMANTIC_2_CLASS].masks_existing == 1
        assert results[MaskType.SEMANTIC_2_CLASS].total_created == 1


class _SubmitBreaksExecutor(_FakeExecutor):
    """Dies inside submit() on the first pool created; later pools behave normally."""

    def submit(self, fn, item):
        if _FakeExecutor.instances.index(self) == 0:
            raise BrokenProcessPool("died while submitting")
        return super().submit(fn, item)


class TestRunWorkItemsSubmissionBreaks:
    """A pool that breaks while work is being submitted must restart, not abort."""

    def test_recovers_when_submit_raises(self, tmp_path, monkeypatch) -> None:
        from ftw_dataset_tools.api import masks

        def fake_worker(task):
            return (
                [
                    (
                        mask_type,
                        masks.MaskResult(
                            grid_id=task.grid_id, output_path=Path(path), width=1, height=1
                        ),
                    )
                    for mask_type, path in task.outputs
                ],
                None,
            )

        monkeypatch.setattr(masks, "_process_single_grid_cell", fake_worker)
        monkeypatch.setattr(masks, "ProcessPoolExecutor", _SubmitBreaksExecutor)
        _FakeExecutor.instances.clear()

        work_items = [_mask_task(tmp_path, grid_id=g) for g in ("g1", "g2")]
        created, failures, pool_restarts = masks._run_work_items(
            work_items, num_workers=2, total_tasks=2, on_progress=None
        )

        assert {result.grid_id for _mask_type, result in created} == {"g1", "g2"}
        assert failures == []
        assert pool_restarts == 1


class TestRunWorkItemsHalvesWorkers:
    """A broken pool usually means a memory shortfall, so each restart narrows it."""

    def test_worker_count_halves_on_every_restart(self, tmp_path, monkeypatch) -> None:
        from ftw_dataset_tools.api import masks

        def always_broken(_task):
            raise BrokenProcessPool("boom")

        monkeypatch.setattr(masks, "_process_single_grid_cell", always_broken)
        monkeypatch.setattr(masks, "ProcessPoolExecutor", _FakeExecutor)
        _FakeExecutor.instances.clear()

        _created, failures, pool_restarts = masks._run_work_items(
            [_mask_task(tmp_path, grid_id="g1")], num_workers=8, total_tasks=1, on_progress=None
        )

        assert pool_restarts == masks._MAX_POOL_RESTARTS
        assert len(failures) == 1
        assert [e.max_workers for e in _FakeExecutor.instances] == [8, 4, 2, 1]

    def test_never_narrows_below_one_worker(self, tmp_path, monkeypatch) -> None:
        from ftw_dataset_tools.api import masks

        def always_broken(_task):
            raise BrokenProcessPool("boom")

        monkeypatch.setattr(masks, "_process_single_grid_cell", always_broken)
        monkeypatch.setattr(masks, "ProcessPoolExecutor", _FakeExecutor)
        _FakeExecutor.instances.clear()

        masks._run_work_items(
            [_mask_task(tmp_path, grid_id="g1")], num_workers=1, total_tasks=1, on_progress=None
        )

        assert [e.max_workers for e in _FakeExecutor.instances] == [1, 1, 1, 1]


class TestMaskRunSummaryLines:
    """Lines shared by the standalone command and the pipeline."""

    @staticmethod
    def _result(existing: int = 0, restarts: int = 0):
        from ftw_dataset_tools.api.masks import CreateMasksResult

        return CreateMasksResult(
            masks_created=[],
            masks_skipped=[],
            field_dataset="test",
            masks_existing=existing,
            pool_restarts=restarts,
        )

    def test_clean_run_reports_nothing(self) -> None:
        from ftw_dataset_tools.api.masks import mask_run_summary_lines

        assert mask_run_summary_lines([self._result()]) == []

    def test_reused_masks_are_summed_across_types(self) -> None:
        from ftw_dataset_tools.api.masks import mask_run_summary_lines

        lines = mask_run_summary_lines([self._result(existing=2), self._result(existing=3)])
        assert lines == ["  Masks reused: 5"]

    def test_restarts_are_counted_once_not_summed(self) -> None:
        from ftw_dataset_tools.api.masks import mask_run_summary_lines

        lines = mask_run_summary_lines([self._result(restarts=2), self._result(restarts=2)])
        assert lines == ["  Worker pool restarts: 2"]

    def test_no_results_is_empty(self) -> None:
        from ftw_dataset_tools.api.masks import mask_run_summary_lines

        assert mask_run_summary_lines([]) == []


class TestInstanceMaskStatistics:
    """The instance mask declares its background as nodata, so stats cover only fields."""

    @staticmethod
    def _build_inputs(tmp_path, ids):
        """One-cell chips file plus two fields carrying the given instance ids."""
        import geopandas as gpd
        from shapely.geometry import LineString, box

        crs = "EPSG:3035"
        cell = box(4000000, 3000000, 4001000, 3001000)

        chips = tmp_path / "chips.parquet"
        gpd.GeoDataFrame(
            {"id": ["grid_001"], "field_coverage_pct": [20.0]},
            geometry=[cell],
            crs=crs,
        ).to_parquet(chips)

        fields = [
            box(4000100, 3000100, 4000400, 3000400),
            box(4000500, 3000100, 4000800, 3000400),
        ]
        boundaries = tmp_path / "fields.parquet"
        gpd.GeoDataFrame({"id": ids}, geometry=fields, crs=crs).to_parquet(boundaries)

        lines = tmp_path / "lines.parquet"
        gpd.GeoDataFrame(
            {"id": ids},
            geometry=[LineString(f.exterior.coords) for f in fields],
            crs=crs,
        ).to_parquet(lines)

        return chips, boundaries, lines

    def _create(self, tmp_path, mask_type, ids=(1000, 1010), **kwargs):
        from ftw_dataset_tools.api.masks import create_masks

        chips, boundaries, lines = self._build_inputs(tmp_path, list(ids))
        result = create_masks(
            chips_file=chips,
            boundaries_file=boundaries,
            boundary_lines_file=lines,
            output_dir=tmp_path / mask_type.value,
            field_dataset="test",
            mask_types=[mask_type],
            num_workers=1,
            **kwargs,
        )[mask_type]
        assert result.total_created == 1, result.masks_skipped
        return result.masks_created[0].output_path

    def test_instance_mask_declares_background_as_nodata(self, tmp_path) -> None:
        import rasterio

        from ftw_dataset_tools.api.masks import MaskType

        path = self._create(tmp_path, MaskType.INSTANCE)

        with rasterio.open(path) as src:
            assert src.nodata == 0
            tags = src.tags(1)

        assert float(tags["STATISTICS_MINIMUM"]) == 1000
        assert float(tags["STATISTICS_MAXIMUM"]) == 1010
        assert float(tags["STATISTICS_VALID_PERCENT"]) < 100

    def test_presence_only_background_is_excluded_but_not_declared(self, tmp_path) -> None:
        """Background 3 is a legal instance id, so it must not become the band nodata."""
        import rasterio

        from ftw_dataset_tools.api.masks import MaskType

        path = self._create(tmp_path, MaskType.INSTANCE, background_class_value=3)

        with rasterio.open(path) as src:
            assert src.nodata is None
            tags = src.tags(1)

        assert float(tags["STATISTICS_MINIMUM"]) == 1000
        assert float(tags["STATISTICS_MAXIMUM"]) == 1010

    def test_semantic_masks_keep_background_in_their_statistics(self, tmp_path) -> None:
        import rasterio

        from ftw_dataset_tools.api.masks import MaskType

        path = self._create(tmp_path, MaskType.SEMANTIC_2_CLASS)

        with rasterio.open(path) as src:
            assert src.nodata is None
            assert float(src.tags(1)["STATISTICS_MINIMUM"]) == 0
