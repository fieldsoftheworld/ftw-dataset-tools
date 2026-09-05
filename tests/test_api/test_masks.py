"""Tests for the masks API."""

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
        from ftw_dataset_tools.api.masks import MaskType, group_by_source

        groups = group_by_source(
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
        from ftw_dataset_tools.api.masks import MaskType, group_by_source

        assert group_by_source([MaskType.DECODE_DISTANCE]) == {
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
