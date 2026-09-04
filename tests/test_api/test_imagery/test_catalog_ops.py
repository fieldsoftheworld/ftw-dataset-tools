"""Tests for catalog-tree walking helpers in catalog_ops."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


class TestIterChipDirs:
    def test_yields_nested_item_dirs_sorted(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.imagery.catalog_ops import iter_chip_dirs

        for square, chip in (("33UXQ", "b"), ("33UXP", "a"), ("33UXP", "c")):
            (tmp_path / "chips" / square / chip).mkdir(parents=True)
        (tmp_path / "chips" / "33UXP" / ".hidden").mkdir()
        (tmp_path / "chips" / "33UXP" / "catalog.json").write_text("{}")

        dirs = iter_chip_dirs(tmp_path)

        assert [d.relative_to(tmp_path).as_posix() for d in dirs] == [
            "chips/33UXP/a",
            "chips/33UXP/c",
            "chips/33UXQ/b",
        ]

    def test_missing_chips_dir_is_empty(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.imagery.catalog_ops import iter_chip_dirs

        assert iter_chip_dirs(tmp_path) == []


class TestFindCollectionDir:
    def test_returns_dir_with_collection(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.imagery.catalog_ops import find_collection_dir

        (tmp_path / "collection.json").write_text("{}")
        assert find_collection_dir(tmp_path) == tmp_path

    def test_raises_without_collection(self, tmp_path: Path) -> None:
        import pytest

        from ftw_dataset_tools.api.imagery.catalog_ops import find_collection_dir

        with pytest.raises(FileNotFoundError, match=r"collection\.json"):
            find_collection_dir(tmp_path)
