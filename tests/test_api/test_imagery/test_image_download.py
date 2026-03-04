"""Tests for image_download module helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ftw_dataset_tools.api.imagery.image_download import find_reference_mask_for_output

if TYPE_CHECKING:
    from pathlib import Path


class TestFindReferenceMaskForOutput:
    """Tests for reference mask auto-detection."""

    def test_prefers_semantic_3_class(self, tmp_path: Path) -> None:
        """Picks semantic_3_class when multiple masks are available."""
        output_path = tmp_path / "chip_001_2024_planting_image_s2.tif"
        (tmp_path / "chip_001_2024_instance.tif").touch()
        expected = tmp_path / "chip_001_2024_semantic_3_class.tif"
        expected.touch()

        found = find_reference_mask_for_output(output_path)

        assert found == expected

    def test_falls_back_to_semantic_2_class(self, tmp_path: Path) -> None:
        """Falls back to semantic_2_class when semantic_3_class is missing."""
        output_path = tmp_path / "chip_001_2024_harvest_image_s2.tif"
        expected = tmp_path / "chip_001_2024_semantic_2_class.tif"
        expected.touch()

        found = find_reference_mask_for_output(output_path)

        assert found == expected

    def test_falls_back_to_instance(self, tmp_path: Path) -> None:
        """Falls back to instance mask when semantic masks are missing."""
        output_path = tmp_path / "chip_001_2024_planting_image_s2.tif"
        expected = tmp_path / "chip_001_2024_instance.tif"
        expected.touch()

        found = find_reference_mask_for_output(output_path)

        assert found == expected

    def test_returns_none_when_not_matching_pattern(self, tmp_path: Path) -> None:
        """Returns None for filenames that do not match imagery naming pattern."""
        output_path = tmp_path / "custom_output.tif"

        found = find_reference_mask_for_output(output_path)

        assert found is None
