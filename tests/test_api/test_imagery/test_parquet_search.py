"""Tests for the parquet-catalog scene search backend."""

from datetime import UTC, date, datetime
from pathlib import Path

import duckdb
import pytest

from ftw_dataset_tools.api.imagery.parquet_search import (
    part_urls_for_query,
    query_scenes,
    zones_for_bbox,
)

# Slovenia-ish chip bbox, UTM zone 33 (lon 14.9-15.1)
SI_BBOX = (14.9, 45.9, 15.1, 46.1)


class TestZonesForBbox:
    """Tests for zones_for_bbox."""

    def test_single_zone(self):
        assert zones_for_bbox(SI_BBOX) == {33}

    def test_bbox_spanning_zone_boundary(self):
        # Luxembourg sits on the 6 degree meridian: zones 31 and 32
        assert zones_for_bbox((5.9, 49.5, 6.1, 49.7)) == {31, 32}

    def test_norway_exception_adds_zone_32(self):
        # Western Norway (band V widening): lon 4E computes zone 31, but
        # Sentinel-2 tiles there are zone 32
        assert 32 in zones_for_bbox((4.5, 60.0, 4.7, 60.2))

    def test_svalbard_exception_adds_odd_zones(self):
        zones = zones_for_bbox((14.0, 78.0, 14.5, 78.3))
        assert {33} <= zones


class TestPartUrlsForQuery:
    """Tests for part_urls_for_query."""

    BASE = "https://example.com/s2"

    def test_2017_is_one_items_parquet(self):
        urls = part_urls_for_query(
            SI_BBOX,
            datetime(2017, 6, 1, tzinfo=UTC),
            datetime(2017, 7, 1, tzinfo=UTC),
            base_url=self.BASE,
            today=date(2026, 9, 21),
        )
        assert urls == [f"{self.BASE}/year=2017/items.parquet"]

    def test_2019_uses_four_zone_ranges(self):
        urls = part_urls_for_query(
            SI_BBOX,
            datetime(2019, 6, 1, tzinfo=UTC),
            datetime(2019, 7, 1, tzinfo=UTC),
            base_url=self.BASE,
            today=date(2026, 9, 21),
        )
        assert urls == [f"{self.BASE}/year=2019/z21-35.parquet"]

    def test_2021_uses_eight_zone_ranges(self):
        urls = part_urls_for_query(
            SI_BBOX,
            datetime(2021, 6, 1, tzinfo=UTC),
            datetime(2021, 7, 1, tzinfo=UTC),
            base_url=self.BASE,
            today=date(2026, 9, 21),
        )
        assert urls == [f"{self.BASE}/year=2021/z32-35.parquet"]

    def test_current_year_adds_live_part(self):
        urls = part_urls_for_query(
            SI_BBOX,
            datetime(2026, 8, 1, tzinfo=UTC),
            datetime(2026, 9, 1, tzinfo=UTC),
            base_url=self.BASE,
            today=date(2026, 9, 21),
        )
        assert urls == [
            f"{self.BASE}/year=2026/z32-35.parquet",
            f"{self.BASE}/year=2026/live.parquet",
        ]

    def test_window_spanning_years_queries_both(self):
        urls = part_urls_for_query(
            SI_BBOX,
            datetime(2024, 12, 10, tzinfo=UTC),
            datetime(2025, 1, 20, tzinfo=UTC),
            base_url=self.BASE,
            today=date(2026, 9, 21),
        )
        assert urls == [
            f"{self.BASE}/year=2024/z32-35.parquet",
            f"{self.BASE}/year=2025/z32-35.parquet",
        ]

    def test_years_outside_archive_are_dropped(self):
        urls = part_urls_for_query(
            SI_BBOX,
            datetime(2015, 6, 1, tzinfo=UTC),
            datetime(2015, 7, 1, tzinfo=UTC),
            base_url=self.BASE,
            today=date(2026, 9, 21),
        )
        assert urls == []

    def test_boundary_bbox_yields_two_parts(self):
        urls = part_urls_for_query(
            (5.9, 49.5, 6.1, 49.7),
            datetime(2021, 6, 1, tzinfo=UTC),
            datetime(2021, 7, 1, tzinfo=UTC),
            base_url=self.BASE,
            today=date(2026, 9, 21),
        )
        assert urls == [
            f"{self.BASE}/year=2021/z21-31.parquet",
            f"{self.BASE}/year=2021/z32-35.parquet",
        ]


def _write_part(path: Path, rows: list[dict]) -> None:
    """Write a catalog part file with the mirror's narrow columns.

    Deliberately omits the wide ``assets`` column: reading it in bulk is what
    makes mirror queries slow, so the implementation must never touch it, and
    a fixture without it proves that.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET TimeZone = 'UTC'")
    con.execute(
        """
        CREATE TABLE part (
            id VARCHAR,
            bbox DOUBLE[],
            datetime TIMESTAMP WITH TIME ZONE,
            "s2:mgrs_tile" VARCHAR,
            "eo:cloud_cover" DOUBLE,
            "s2:nodata_pixel_percentage" DOUBLE,
            _month TINYINT
        )
        """
    )
    for r in rows:
        con.execute(
            "INSERT INTO part VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                r["id"],
                r.get("bbox", [14.5, 45.5, 15.6, 46.5]),
                r["datetime"],
                r.get("tile", "33TVM"),
                r.get("cloud", 1.0),
                r.get("nodata"),
                r["datetime"].month,
            ],
        )
    con.execute(f"COPY part TO '{path}' (FORMAT PARQUET)")
    con.close()


class TestQueryScenes:
    """Tests for query_scenes against local fixture parts."""

    @pytest.fixture
    def base_dir(self, tmp_path):
        _write_part(
            tmp_path / "year=2021" / "z32-35.parquet",
            [
                {
                    "id": "S2A_33TVM_20210605_0_L2A",
                    "datetime": datetime(2021, 6, 5, 10, 0, tzinfo=UTC),
                    "cloud": 5.0,
                },
                {
                    "id": "S2B_33TVM_20210610_0_L2A",
                    "datetime": datetime(2021, 6, 10, 10, 0, tzinfo=UTC),
                    "cloud": 0.5,
                    "nodata": 3.2,
                },
                {
                    "id": "S2A_33TVM_20210615_0_L2A",
                    "datetime": datetime(2021, 6, 15, 10, 0, tzinfo=UTC),
                    "cloud": 90.0,
                },
                {
                    "id": "S2A_33TVM_20210705_0_L2A",
                    "datetime": datetime(2021, 7, 5, 10, 0, tzinfo=UTC),
                    "cloud": 1.0,
                },
                {
                    "id": "S2A_33TWM_20210605_0_L2A",
                    "datetime": datetime(2021, 6, 5, 10, 0, tzinfo=UTC),
                    "cloud": 1.0,
                    "bbox": [16.4, 45.5, 17.5, 46.5],
                },
            ],
        )
        return str(tmp_path)

    def _query(self, base_dir, **kwargs):
        defaults = {
            "bbox": SI_BBOX,
            "start": datetime(2021, 6, 1, tzinfo=UTC),
            "end": datetime(2021, 6, 30, tzinfo=UTC),
            "cloud_cover_max": 75,
            "base_url": base_dir,
            "today": date(2026, 9, 21),
        }
        defaults.update(kwargs)
        return query_scenes(**defaults)

    def test_filters_and_sorts_by_cloud_cover(self, base_dir):
        items = self._query(base_dir)
        assert [i.id for i in items] == [
            "S2B_33TVM_20210610_0_L2A",  # 0.5% cloud
            "S2A_33TVM_20210605_0_L2A",  # 5.0% cloud
        ]

    def test_excludes_scenes_over_cloud_threshold(self, base_dir):
        ids = {i.id for i in self._query(base_dir)}
        assert "S2A_33TVM_20210615_0_L2A" not in ids

    def test_excludes_scenes_outside_window(self, base_dir):
        ids = {i.id for i in self._query(base_dir)}
        assert "S2A_33TVM_20210705_0_L2A" not in ids

    def test_excludes_scenes_outside_bbox(self, base_dir):
        ids = {i.id for i in self._query(base_dir)}
        assert "S2A_33TWM_20210605_0_L2A" not in ids

    def test_item_carries_datetime_properties_and_assets(self, base_dir):
        from pystac.extensions.eo import EOExtension

        item = self._query(base_dir)[0]
        assert item.datetime == datetime(2021, 6, 10, 10, 0, tzinfo=UTC)
        assert item.properties["eo:cloud_cover"] == 0.5
        # Selection reads cloud cover through EOExtension; the extension must
        # be declared on the reconstructed item or that read raises.
        assert EOExtension.ext(item).cloud_cover == 0.5
        assert item.properties["s2:nodata_pixel_percentage"] == 3.2
        # Asset hrefs are synthesized from the id via the sentinel-cogs bucket
        # layout (reading the mirror's wide assets column is far too slow).
        base = (
            "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs"
            "/33/T/VM/2021/6/S2B_33TVM_20210610_0_L2A"
        )
        assert item.assets["red"].href == f"{base}/B04.tif"
        assert item.assets["scl"].href == f"{base}/SCL.tif"
        assert item.assets["visual"].href == f"{base}/TCI.tif"
        assert item.assets["nir"].href == f"{base}/B08.tif"
        assert item.bbox == [14.5, 45.5, 15.6, 46.5]

    def test_single_digit_zone_asset_paths(self, tmp_path):
        _write_part(
            tmp_path / "year=2021" / "z01-15.parquet",
            [
                {
                    "id": "S2A_8VLM_20210605_0_L2A",
                    "datetime": datetime(2021, 6, 5, 10, 0, tzinfo=UTC),
                    "tile": "8VLM",
                    "bbox": [-135.1, 57.5, -133.9, 58.5],
                }
            ],
        )
        items = query_scenes(
            bbox=(-134.6, 57.9, -134.4, 58.1),
            start=datetime(2021, 6, 1, tzinfo=UTC),
            end=datetime(2021, 6, 30, tzinfo=UTC),
            cloud_cover_max=75,
            base_url=str(tmp_path),
            today=date(2026, 9, 21),
        )
        assert items[0].assets["red"].href == (
            "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs"
            "/8/V/LM/2021/6/S2A_8VLM_20210605_0_L2A/B04.tif"
        )

    def test_item_without_nodata_metadata_omits_property(self, base_dir):
        items = self._query(base_dir)
        by_id = {i.id: i for i in items}
        assert "s2:nodata_pixel_percentage" not in by_id["S2A_33TVM_20210605_0_L2A"].properties

    def test_item_self_href_points_at_earth_search_item(self, base_dir):
        item = self._query(base_dir)[0]
        href = item.get_self_href()
        assert href is not None
        assert href.endswith(f"/collections/sentinel-2-l2a/items/{item.id}")

    def test_duplicate_ids_across_parts_are_deduped(self, tmp_path):
        row = {
            "id": "S2A_33TVM_20260805_0_L2A",
            "datetime": datetime(2026, 8, 5, 10, 0, tzinfo=UTC),
            "cloud": 2.0,
        }
        _write_part(tmp_path / "year=2026" / "z32-35.parquet", [row])
        _write_part(tmp_path / "year=2026" / "live.parquet", [row])
        items = query_scenes(
            bbox=SI_BBOX,
            start=datetime(2026, 8, 1, tzinfo=UTC),
            end=datetime(2026, 8, 31, tzinfo=UTC),
            cloud_cover_max=75,
            base_url=str(tmp_path),
            today=date(2026, 9, 21),
        )
        assert [i.id for i in items] == ["S2A_33TVM_20260805_0_L2A"]

    def test_no_parts_in_range_returns_empty(self, base_dir):
        items = self._query(
            base_dir,
            start=datetime(2015, 6, 1, tzinfo=UTC),
            end=datetime(2015, 6, 30, tzinfo=UTC),
        )
        assert items == []

    def test_window_crossing_year_boundary_uses_month_set(self, tmp_path):
        # December and January scenes in adjacent year parts
        _write_part(
            tmp_path / "year=2024" / "z32-35.parquet",
            [
                {
                    "id": "S2A_33TVM_20241220_0_L2A",
                    "datetime": datetime(2024, 12, 20, 10, 0, tzinfo=UTC),
                }
            ],
        )
        _write_part(
            tmp_path / "year=2025" / "z32-35.parquet",
            [
                {
                    "id": "S2A_33TVM_20250110_0_L2A",
                    "datetime": datetime(2025, 1, 10, 10, 0, tzinfo=UTC),
                }
            ],
        )
        items = query_scenes(
            bbox=SI_BBOX,
            start=datetime(2024, 12, 15, tzinfo=UTC),
            end=datetime(2025, 1, 15, tzinfo=UTC),
            cloud_cover_max=75,
            base_url=str(tmp_path),
            today=date(2026, 9, 21),
        )
        assert {i.id for i in items} == {
            "S2A_33TVM_20241220_0_L2A",
            "S2A_33TVM_20250110_0_L2A",
        }
