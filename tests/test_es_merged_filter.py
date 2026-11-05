"""es_merged_filter.yaml stays consistent with the filters it is generated from."""

from __future__ import annotations

from pathlib import Path

import pytest

from ftw_dataset_tools.api.config import ClassFilter

FILTERS = Path(__file__).resolve().parent.parent / "configs" / "filters"
REGIONS = {
    "es_cb": "es_uso_north_filter.yaml",
    "es_pv": "es_uso_north_filter.yaml",
}


@pytest.fixture(scope="module")
def merged() -> ClassFilter:
    return ClassFilter.from_file(FILTERS / "es_merged_filter.yaml")


def _printable(values) -> set[str]:
    return {v for v in values if v.isprintable()}


class TestEsMergedFilter:
    def test_reads_the_merge_class_column(self, merged) -> None:
        assert merged.resolve_column(["ftw:class", "crop:code"]) == "ftw:class"

    def test_national_codes_keep_their_decision(self, merged) -> None:
        national = ClassFilter.from_file(FILTERS / "spain_filter.yaml")
        assert {f"es:{v}" for v in national.include} <= set(merged.include)
        assert {f"es:{v}" for v in national.exclude} <= set(merged.exclude)

    def test_pasture_is_field_only_in_the_atlantic_north(self, merged) -> None:
        for code in ("62", "63"):
            assert f"es:{code}" in merged.exclude
            assert f"es:{code}:atlantic" in merged.include

    @pytest.mark.parametrize(("region", "source"), sorted(REGIONS.items()))
    def test_regional_codes_follow_their_region_filter(self, merged, region, source) -> None:
        regional = ClassFilter.from_file(FILTERS / source)
        assert {f"{region}:{v}" for v in _printable(regional.include)} <= set(merged.include)
        assert {f"{region}:{v}" for v in _printable(regional.exclude)} <= set(merged.exclude)

    def test_nothing_else(self, merged) -> None:
        national = ClassFilter.from_file(FILTERS / "spain_filter.yaml")
        expected = {f"es:{v}" for v in [*national.include, *national.exclude]}
        expected |= {"es:62:atlantic", "es:63:atlantic"}
        for region, source in REGIONS.items():
            regional = ClassFilter.from_file(FILTERS / source)
            expected |= {
                f"{region}:{v}" for v in _printable([*regional.include, *regional.exclude])
            }
        assert set(merged.include) | set(merged.exclude) == expected
        assert not set(merged.include) & set(merged.exclude)
