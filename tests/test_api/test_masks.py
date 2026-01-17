"""Tests for the masks API."""


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
