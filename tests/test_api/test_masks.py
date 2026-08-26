"""Tests for the masks API."""

from pathlib import Path

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
        )

        assert path == Path("/output/chips/grid_001/grid_001_instance.tif")

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
            year=2024,
        )

        assert path == Path("/output/chips/grid_001_2024/grid_001_2024_instance.tif")

    def test_output_path_raises_when_item_id_missing(self) -> None:
        """Test mask path raises KeyError when the chip is absent from the mapping."""
        from pathlib import Path

        import pytest

        from ftw_dataset_tools.api.masks import MaskType, get_mask_output_path

        with pytest.raises(KeyError, match="grid_999"):
            get_mask_output_path(
                grid_id="grid_999",
                mask_type=MaskType.INSTANCE,
                chip_dirs={"grid_001": Path("/output/chips/grid_001")},
            )


class TestChipsBaseDir:
    """Tests for chips base directory resolution."""

    def test_chips_base_dir_matches_create_dataset_layout(self) -> None:
        """Test the chips base dir mirrors what create-dataset produces."""
        from pathlib import Path

        from ftw_dataset_tools.api.masks import get_chips_base_dir

        assert get_chips_base_dir("./output", "austria") == Path("output/austria-chips")


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


class TestBuildChipDirs:
    """Tests for build_chip_dirs."""

    def test_creates_one_directory_per_covered_chip(
        self, sample_chips_with_coverage: Path, tmp_path: Path
    ) -> None:
        """Test a directory is created per grid cell meeting the coverage threshold."""
        from ftw_dataset_tools.api.masks import build_chip_dirs

        base_dir = tmp_path / "austria-chips"
        chip_dirs = build_chip_dirs(
            chips_file=sample_chips_with_coverage,
            chips_base_dir=base_dir,
            min_coverage=0.01,
        )

        assert set(chip_dirs) == {"grid_001", "grid_002", "grid_003"}
        for item_id, chip_dir in chip_dirs.items():
            assert chip_dir == base_dir / item_id
            assert chip_dir.is_dir()

    def test_respects_min_coverage(self, sample_chips_with_coverage: Path, tmp_path: Path) -> None:
        """Test cells below the coverage threshold are excluded."""
        from ftw_dataset_tools.api.masks import build_chip_dirs

        # grid_003 has 0.5% coverage, below the 1.0 threshold
        chip_dirs = build_chip_dirs(
            chips_file=sample_chips_with_coverage,
            chips_base_dir=tmp_path / "chips",
            min_coverage=1.0,
        )

        assert set(chip_dirs) == {"grid_001", "grid_002"}
        assert not (tmp_path / "chips" / "grid_003").exists()

    def test_year_included_in_item_id(
        self, sample_chips_with_coverage: Path, tmp_path: Path
    ) -> None:
        """Test the year becomes part of the chip directory name."""
        from ftw_dataset_tools.api.masks import build_chip_dirs

        chip_dirs = build_chip_dirs(
            chips_file=sample_chips_with_coverage,
            chips_base_dir=tmp_path / "chips",
            min_coverage=0.01,
            year=2024,
        )

        assert set(chip_dirs) == {"grid_001_2024", "grid_002_2024", "grid_003_2024"}
        assert chip_dirs["grid_001_2024"].name == "grid_001_2024"

    def test_missing_grid_id_column_raises_value_error(
        self, sample_chips_with_coverage: Path, tmp_path: Path
    ) -> None:
        """Test an unknown grid ID column surfaces as a ValueError, not a DuckDB error."""
        import pytest

        from ftw_dataset_tools.api.masks import build_chip_dirs

        with pytest.raises(ValueError, match="Grid ID column 'not_a_column' not found"):
            build_chip_dirs(
                chips_file=sample_chips_with_coverage,
                chips_base_dir=tmp_path / "chips",
                min_coverage=0.01,
                grid_id_col="not_a_column",
            )

    def test_missing_coverage_column_raises_value_error(
        self, sample_chips_with_coverage: Path, tmp_path: Path
    ) -> None:
        """Test an unknown coverage column surfaces as a ValueError."""
        import pytest

        from ftw_dataset_tools.api.masks import build_chip_dirs

        with pytest.raises(ValueError, match="Coverage column 'nope' not found"):
            build_chip_dirs(
                chips_file=sample_chips_with_coverage,
                chips_base_dir=tmp_path / "chips",
                min_coverage=0.01,
                coverage_col="nope",
            )

    def test_handles_quote_in_chips_path(self, tmp_path: Path) -> None:
        """Test a path containing a single quote does not break the query."""
        import geopandas as gpd
        from shapely.geometry import box

        from ftw_dataset_tools.api.masks import build_chip_dirs

        quoted_dir = tmp_path / "O'Brien"
        quoted_dir.mkdir()
        chips_file = quoted_dir / "chips.parquet"
        gpd.GeoDataFrame(
            {"id": ["grid_001"], "field_coverage_pct": [50.0]},
            geometry=[box(10.0, 50.0, 10.01, 50.01)],
            crs="EPSG:4326",
        ).to_parquet(chips_file)

        chip_dirs = build_chip_dirs(
            chips_file=chips_file,
            chips_base_dir=tmp_path / "chips",
            min_coverage=0.01,
        )

        assert set(chip_dirs) == {"grid_001"}


class TestCreateMasksCatalogStructure:
    """Tests that standalone create_masks writes the create-dataset catalog structure."""

    def test_standalone_writes_into_per_chip_directories(
        self,
        sample_chips_with_coverage: Path,
        sample_boundaries_geoparquet: Path,
        sample_boundary_lines_geoparquet: Path,
        tmp_path: Path,
    ) -> None:
        """Test masks land in {output}/{dataset}-chips/{item_id}/{item_id}_{type}.tif."""
        from ftw_dataset_tools.api.masks import MaskType, create_masks

        output_dir = tmp_path / "output"
        result = create_masks(
            chips_file=sample_chips_with_coverage,
            boundaries_file=sample_boundaries_geoparquet,
            boundary_lines_file=sample_boundary_lines_geoparquet,
            output_dir=output_dir,
            field_dataset="austria",
            mask_type=MaskType.SEMANTIC_3_CLASS,
            num_workers=1,
        )

        assert result.total_created > 0
        base_dir = output_dir / "austria-chips"
        for mask in result.masks_created:
            expected = base_dir / mask.grid_id / f"{mask.grid_id}_semantic_3_class.tif"
            assert mask.output_path == expected
            assert mask.output_path.exists()
            # Dataset name must not be duplicated into the filename
            assert not mask.output_path.name.startswith("austria_")

    def test_standalone_includes_year_in_dir_and_filename(
        self,
        sample_chips_with_coverage: Path,
        sample_boundaries_geoparquet: Path,
        sample_boundary_lines_geoparquet: Path,
        tmp_path: Path,
    ) -> None:
        """Test the year appears in both the chip directory and the mask filename."""
        from ftw_dataset_tools.api.masks import MaskType, create_masks

        output_dir = tmp_path / "output"
        result = create_masks(
            chips_file=sample_chips_with_coverage,
            boundaries_file=sample_boundaries_geoparquet,
            boundary_lines_file=sample_boundary_lines_geoparquet,
            output_dir=output_dir,
            field_dataset="austria",
            mask_type=MaskType.SEMANTIC_3_CLASS,
            year=2024,
            num_workers=1,
        )

        assert result.total_created > 0
        for mask in result.masks_created:
            item_id = f"{mask.grid_id}_2024"
            expected = output_dir / "austria-chips" / item_id / f"{item_id}_semantic_3_class.tif"
            assert mask.output_path == expected
            assert mask.output_path.exists()

    def test_explicit_chip_dirs_still_honored(
        self,
        sample_chips_with_coverage: Path,
        sample_boundaries_geoparquet: Path,
        sample_boundary_lines_geoparquet: Path,
        tmp_path: Path,
    ) -> None:
        """Test a caller-supplied mapping (the pipeline path) takes precedence."""
        from ftw_dataset_tools.api.masks import MaskType, create_masks

        custom_dir = tmp_path / "custom" / "grid_001"
        custom_dir.mkdir(parents=True)
        chip_dirs = {"grid_001": custom_dir}

        result = create_masks(
            chips_file=sample_chips_with_coverage,
            boundaries_file=sample_boundaries_geoparquet,
            boundary_lines_file=sample_boundary_lines_geoparquet,
            output_dir=tmp_path / "output",
            field_dataset="austria",
            mask_type=MaskType.SEMANTIC_3_CLASS,
            coverage_col="field_coverage_pct",
            min_coverage=40.0,  # only grid_001 qualifies
            chip_dirs=chip_dirs,
            num_workers=1,
        )

        assert result.total_created == 1
        assert result.masks_created[0].output_path == custom_dir / "grid_001_semantic_3_class.tif"
        # Nothing should be written to the derived location
        assert not (tmp_path / "output" / "austria-chips").exists()

    def test_create_masks_handles_quote_in_input_paths(self, tmp_path: Path) -> None:
        """Test create_masks runs end-to-end when input paths contain a single quote."""
        import geopandas as gpd
        from shapely.geometry import LineString, box

        from ftw_dataset_tools.api.masks import MaskType, create_masks

        quoted_dir = tmp_path / "O'Brien"
        quoted_dir.mkdir()
        gpd.GeoDataFrame(
            {"id": ["grid_001"], "field_coverage_pct": [50.0]},
            geometry=[box(10.0, 50.0, 10.01, 50.01)],
            crs="EPSG:4326",
        ).to_parquet(quoted_dir / "chips.parquet")
        gpd.GeoDataFrame(
            {"id": [1]},
            geometry=[box(10.0, 50.0, 10.005, 50.005)],
            crs="EPSG:4326",
        ).to_parquet(quoted_dir / "fields.parquet")
        gpd.GeoDataFrame(
            {"id": [1]},
            geometry=[LineString([(10.0, 50.0), (10.005, 50.005)])],
            crs="EPSG:4326",
        ).to_parquet(quoted_dir / "lines.parquet")

        result = create_masks(
            chips_file=quoted_dir / "chips.parquet",
            boundaries_file=quoted_dir / "fields.parquet",
            boundary_lines_file=quoted_dir / "lines.parquet",
            output_dir=tmp_path / "output",
            field_dataset="austria",
            mask_type=MaskType.SEMANTIC_3_CLASS,
            num_workers=1,
        )

        assert result.total_created == 1


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
