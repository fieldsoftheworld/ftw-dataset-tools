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

    def test_create_single_mask_accepts_background_class_value_parameter(self) -> None:
        """Test that _create_single_mask signature accepts background_class_value parameter."""
        import inspect

        from ftw_dataset_tools.api.masks import _create_single_mask

        sig = inspect.signature(_create_single_mask)
        assert "background_class_value" in sig.parameters
        # Should have default value of 0
        assert sig.parameters["background_class_value"].default == 0


class TestProcessSingleGridCellRetry:
    """_process_single_grid_cell should retry once with a fresh connection."""

    @staticmethod
    def _args(tmp_path):
        from rasterio.crs import CRS

        from ftw_dataset_tools.api.masks import MaskType

        return (
            "grid_1",
            (0.0, 0.0, 1.0, 1.0),
            CRS.from_epsg(4326).to_wkt(),
            str(tmp_path / "boundaries.parquet"),
            str(tmp_path / "lines.parquet"),
            "geometry",
            "geometry",
            str(tmp_path / "out.tif"),
            MaskType.SEMANTIC_2_CLASS.value,
            10.0,
            None,
            0,
            2048,
        )

    def test_retries_once_then_succeeds(self, tmp_path, monkeypatch) -> None:
        from ftw_dataset_tools.api import masks

        calls = {"n": 0}

        def fake_create_single_mask(**kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise ValueError("boom")
            return masks.MaskResult(
                grid_id=kwargs["grid_id"], output_path=kwargs["output_path"], width=1, height=1
            )

        monkeypatch.setattr(masks, "_create_single_mask", fake_create_single_mask)

        result, error = masks._process_single_grid_cell(self._args(tmp_path))

        assert error is None
        assert result is not None
        assert calls["n"] == 2

    def test_fails_twice_returns_error_with_exception_class_prefix(
        self, tmp_path, monkeypatch
    ) -> None:
        from ftw_dataset_tools.api import masks

        calls = {"n": 0}

        def fake_create_single_mask(**_kwargs):
            calls["n"] += 1
            raise ValueError("boom")

        monkeypatch.setattr(masks, "_create_single_mask", fake_create_single_mask)

        result, error = masks._process_single_grid_cell(self._args(tmp_path))

        assert result is None
        assert calls["n"] == 2  # retried once, still failed
        grid_id, reason = error
        assert grid_id == "grid_1"
        assert reason == "ValueError: boom"


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

    def test_work_item_carries_a_positive_memory_limit(self, tmp_path, monkeypatch) -> None:
        from ftw_dataset_tools.api import masks

        chips_path, fields_path, lines_path = TestCreateMasksSkipExisting._build_inputs(tmp_path)

        def fake_worker(item):
            grid_id, output_path = item[0], item[7]
            return (
                masks.MaskResult(grid_id=grid_id, output_path=output_path, width=1, height=1),
                None,
            )

        monkeypatch.setattr(masks, "_process_single_grid_cell", fake_worker)
        monkeypatch.setattr(masks, "ProcessPoolExecutor", _FakeExecutor)
        _FakeExecutor.instances.clear()

        result = masks.create_masks(
            chips_file=chips_path,
            boundaries_file=fields_path,
            boundary_lines_file=lines_path,
            output_dir=tmp_path / "masks",
            field_dataset="test",
            mask_type=masks.MaskType.SEMANTIC_2_CLASS,
            num_workers=2,
        )

        assert result.total_created == 2
        submitted = _FakeExecutor.instances[0].submitted
        assert len(submitted) == 2
        for item in submitted:
            memory_limit_mb = item[-1]
            assert isinstance(memory_limit_mb, int)
            assert memory_limit_mb > 0


class TestRunWorkItemsBrokenPool:
    """Coverage for restarting a ProcessPoolExecutor that a crashed worker broke."""

    @staticmethod
    def _item(grid_id: str) -> tuple:
        return (
            grid_id,
            (0, 0, 1, 1),
            "wkt",
            "boundaries.parquet",
            "lines.parquet",
            "geometry",
            "geometry",
            f"{grid_id}.tif",
            "semantic_2_class",
            10.0,
            None,
            0,
            2048,
        )

    def test_recovers_after_one_broken_pool(self, monkeypatch) -> None:
        from ftw_dataset_tools.api import masks

        attempts: dict[str, int] = {}

        def fake_worker(item):
            grid_id = item[0]
            attempts[grid_id] = attempts.get(grid_id, 0) + 1
            if grid_id == "g2" and attempts[grid_id] == 1:
                raise BrokenProcessPool("boom")
            return (
                masks.MaskResult(
                    grid_id=grid_id, output_path=Path(f"{grid_id}.tif"), width=1, height=1
                ),
                None,
            )

        monkeypatch.setattr(masks, "_process_single_grid_cell", fake_worker)
        monkeypatch.setattr(masks, "ProcessPoolExecutor", _FakeExecutor)

        work_items = [self._item("g1"), self._item("g2"), self._item("g3")]
        created, skipped, pool_restarts = masks._run_work_items(
            work_items, num_workers=1, total_cells=3, on_progress=None
        )

        assert {r.grid_id for r in created} == {"g1", "g2", "g3"}
        assert skipped == []
        assert pool_restarts == 1

    def test_gives_up_after_max_restarts(self, monkeypatch) -> None:
        from ftw_dataset_tools.api import masks

        def fake_worker(item):
            grid_id = item[0]
            if grid_id == "bad":
                raise BrokenProcessPool("boom")
            return (
                masks.MaskResult(
                    grid_id=grid_id, output_path=Path(f"{grid_id}.tif"), width=1, height=1
                ),
                None,
            )

        monkeypatch.setattr(masks, "_process_single_grid_cell", fake_worker)
        monkeypatch.setattr(masks, "ProcessPoolExecutor", _FakeExecutor)

        work_items = [self._item("good"), self._item("bad")]
        created, skipped, pool_restarts = masks._run_work_items(
            work_items, num_workers=1, total_cells=2, on_progress=None
        )

        assert {r.grid_id for r in created} == {"good"}
        assert skipped == [("bad", "BrokenProcessPool: worker died repeatedly")]
        assert pool_restarts == masks._MAX_POOL_RESTARTS

    def test_pool_restarts_reflected_in_create_masks_result(self, tmp_path, monkeypatch) -> None:
        """create_masks surfaces pool_restarts from _run_work_items on its result."""
        from ftw_dataset_tools.api import masks

        chips_path, fields_path, lines_path = TestCreateMasksSkipExisting._build_inputs(tmp_path)

        def fake_run_work_items(_work_items, _num_workers, _total_cells, _on_progress):
            return ([], [], 2)

        monkeypatch.setattr(masks, "_run_work_items", fake_run_work_items)

        result = masks.create_masks(
            chips_file=chips_path,
            boundaries_file=fields_path,
            boundary_lines_file=lines_path,
            output_dir=tmp_path / "masks",
            field_dataset="test",
            mask_type=masks.MaskType.SEMANTIC_2_CLASS,
            num_workers=1,
        )

        assert result.pool_restarts == 2


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

        result = create_masks(
            chips_file=chips_path,
            boundaries_file=fields_path,
            boundary_lines_file=lines_path,
            output_dir=output_dir,
            field_dataset="test",
            mask_type=MaskType.SEMANTIC_2_CLASS,
            num_workers=1,
        )

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

        result = create_masks(
            chips_file=chips_path,
            boundaries_file=fields_path,
            boundary_lines_file=lines_path,
            output_dir=output_dir,
            field_dataset="test",
            mask_type=MaskType.SEMANTIC_2_CLASS,
            num_workers=1,
            skip_existing=True,
        )

        assert result.masks_existing == 1
        assert result.total_created == 1
        assert result.masks_created[0].grid_id == "c2"
        # The non-empty file was left untouched (still not a valid raster).
        assert existing_path.read_bytes() == b"not-really-a-tif-but-non-empty"
        # The empty file was recreated into a real raster.
        assert empty_path.stat().st_size > 0


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
            mask_type=mask_type,
            num_workers=1,
            **kwargs,
        )

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
                mask_type=mask_type,
                min_coverage=0.0,
                num_workers=1,
            )
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
            mask_type=MaskType.DECODE_DISTANCE,
            min_coverage=0.0,
            num_workers=1,
        )

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

    def test_single_mask_has_embedded_stats(self, tmp_path: Path) -> None:
        import duckdb
        from rasterio.crs import CRS

        from ftw_dataset_tools.api.geo import ensure_spatial_loaded
        from ftw_dataset_tools.api.masks import MaskType, _create_single_mask
        from ftw_dataset_tools.api.raster_stats import read_band_stats

        fields_path, lines_path = self._write_inputs(tmp_path)
        conn = duckdb.connect(":memory:")
        ensure_spatial_loaded(conn)
        out = tmp_path / "mask.tif"

        _create_single_mask(
            conn=conn,
            grid_id="g1",
            bounds=(10.0, 50.0, 10.01, 50.01),
            crs=CRS.from_epsg(4326),
            boundaries_path=fields_path,
            boundary_lines_path=lines_path,
            boundaries_geom_col="geometry",
            boundary_lines_geom_col="geometry",
            output_path=out,
            mask_type=MaskType.SEMANTIC_3_CLASS,
            resolution=10.0,
        )

        stats = read_band_stats(out, 1)
        assert stats is not None
        assert stats.minimum == 0.0
        assert stats.maximum == 2.0
        assert 0.0 < stats.mean < 2.0
        assert stats.valid_percent is None
        assert not out.with_name(out.name + ".aux.xml").exists()
        assert not out.with_suffix(".tmp.tif").exists()

    def test_single_mask_is_cog(self, tmp_path: Path) -> None:
        import duckdb
        import rasterio
        from rasterio.crs import CRS

        from ftw_dataset_tools.api.geo import ensure_spatial_loaded
        from ftw_dataset_tools.api.masks import MaskType, _create_single_mask

        fields_path, lines_path = self._write_inputs(tmp_path)
        conn = duckdb.connect(":memory:")
        ensure_spatial_loaded(conn)
        out = tmp_path / "mask2.tif"

        _create_single_mask(
            conn=conn,
            grid_id="g1",
            bounds=(10.0, 50.0, 10.01, 50.01),
            crs=CRS.from_epsg(4326),
            boundaries_path=fields_path,
            boundary_lines_path=lines_path,
            boundaries_geom_col="geometry",
            boundary_lines_geom_col="geometry",
            output_path=out,
            mask_type=MaskType.SEMANTIC_2_CLASS,
            resolution=10.0,
        )

        with rasterio.open(out) as src:
            assert src.profile["tiled"] is True
            assert src.profile["compress"] == "deflate"
            assert src.dtypes[0] == "uint8"

    def test_decode_distance_keeps_tags_and_stats(self, tmp_path: Path) -> None:
        import duckdb
        import rasterio
        from rasterio.crs import CRS

        from ftw_dataset_tools.api.geo import ensure_spatial_loaded
        from ftw_dataset_tools.api.masks import MaskType, _create_single_mask
        from ftw_dataset_tools.api.raster_stats import read_band_stats

        fields_path, lines_path = self._write_inputs(tmp_path)
        conn = duckdb.connect(":memory:")
        ensure_spatial_loaded(conn)
        out = tmp_path / "dist.tif"

        _create_single_mask(
            conn=conn,
            grid_id="g1",
            bounds=(10.0, 50.0, 10.01, 50.01),
            crs=CRS.from_epsg(4326),
            boundaries_path=fields_path,
            boundary_lines_path=lines_path,
            boundaries_geom_col="geometry",
            boundary_lines_geom_col="geometry",
            output_path=out,
            mask_type=MaskType.DECODE_DISTANCE,
            resolution=10.0,
        )

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

    def _rasterize(self, tmp_path, ids):
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
            background_class_value=0,
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


class TestMgrsSquare:
    def test_ftw_grid_id(self) -> None:
        from ftw_dataset_tools.api.masks import get_mgrs_square

        assert get_mgrs_square("ftw-33UXP0410") == "33UXP"
        assert get_mgrs_square("ftw-1CDE0001") == "1CDE"

    def test_non_ftw_id_goes_to_other(self) -> None:
        from ftw_dataset_tools.api.masks import get_mgrs_square

        assert get_mgrs_square("grid_001") == "other"
        assert get_mgrs_square("ftw-abc") == "other"
