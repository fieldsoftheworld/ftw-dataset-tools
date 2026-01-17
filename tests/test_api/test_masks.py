"""Tests for the masks API."""

from pathlib import Path

import pytest


class TestMaskType:
    """Tests for MaskType enum."""

    def test_instance_value(self) -> None:
        """Test INSTANCE mask type value."""
        from ftw_dataset_tools.api.masks import MaskType

        assert MaskType.INSTANCE.value == "instance"

    def test_semantic_2_class_value(self) -> None:
        """Test SEMANTIC_2_CLASS mask type value."""
        from ftw_dataset_tools.api.masks import MaskType

        assert MaskType.SEMANTIC_2_CLASS.value == "semantic_2_class"

    def test_semantic_3_class_value(self) -> None:
        """Test SEMANTIC_3_CLASS mask type value."""
        from ftw_dataset_tools.api.masks import MaskType

        assert MaskType.SEMANTIC_3_CLASS.value == "semantic_3_class"


class TestMaskResult:
    """Tests for MaskResult dataclass."""

    def test_dataclass_fields(self) -> None:
        """Test MaskResult has expected fields."""
        from ftw_dataset_tools.api.masks import MaskResult

        result = MaskResult(
            grid_id="grid_001",
            output_path=Path("/tmp/mask.tif"),
            width=512,
            height=512,
        )
        assert result.grid_id == "grid_001"
        assert result.width == 512
        assert result.height == 512


class TestCreateMasksResult:
    """Tests for CreateMasksResult dataclass."""

    def test_total_created(self) -> None:
        """Test total_created property."""
        from ftw_dataset_tools.api.masks import CreateMasksResult, MaskResult

        result = CreateMasksResult(
            masks_created=[
                MaskResult("a", Path("a.tif"), 512, 512),
                MaskResult("b", Path("b.tif"), 512, 512),
            ],
            masks_skipped=[],
            field_dataset="test",
        )
        assert result.total_created == 2

    def test_total_skipped(self) -> None:
        """Test total_skipped property."""
        from ftw_dataset_tools.api.masks import CreateMasksResult

        result = CreateMasksResult(
            masks_created=[],
            masks_skipped=[("grid_001", "error1"), ("grid_002", "error2")],
            field_dataset="test",
        )
        assert result.total_skipped == 2


class TestCreateMasksInputValidation:
    """Tests for create_masks input validation."""

    def test_chips_file_not_found(self, tmp_path: Path) -> None:
        """Test FileNotFoundError for missing chips file."""
        from ftw_dataset_tools.api.masks import create_masks

        boundaries = tmp_path / "boundaries.parquet"
        boundary_lines = tmp_path / "boundary_lines.parquet"
        boundaries.touch()
        boundary_lines.touch()

        with pytest.raises(FileNotFoundError, match="Chips file not found"):
            create_masks(
                chips_file="/nonexistent/chips.parquet",
                boundaries_file=str(boundaries),
                boundary_lines_file=str(boundary_lines),
            )

    def test_boundaries_file_not_found(self, tmp_path: Path) -> None:
        """Test FileNotFoundError for missing boundaries file."""
        from ftw_dataset_tools.api.masks import create_masks

        chips = tmp_path / "chips.parquet"
        boundary_lines = tmp_path / "boundary_lines.parquet"
        chips.touch()
        boundary_lines.touch()

        with pytest.raises(FileNotFoundError, match="Boundaries file not found"):
            create_masks(
                chips_file=str(chips),
                boundaries_file="/nonexistent/boundaries.parquet",
                boundary_lines_file=str(boundary_lines),
            )


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
