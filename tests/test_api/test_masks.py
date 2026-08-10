"""Tests for the masks API."""

from pathlib import Path


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
