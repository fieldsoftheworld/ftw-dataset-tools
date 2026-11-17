"""Tests for bringing a published collection up to date (api/backfill.py)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from ftw_dataset_tools.api import admin_names, backfill, field_embeddings, stac
from tests.test_api.test_stac import _write_field_embeddings, build_catalog

if TYPE_CHECKING:
    from pathlib import Path

GRID_ID = "ftw-33UXP0410"
ITEM_ID = f"{GRID_ID}_2024"


def _chips_with_admin_names(path: Path, admin2: str | None = "Bezirk") -> Path:
    import geopandas as gpd
    from shapely.geometry import box

    gpd.GeoDataFrame(
        {
            "id": [GRID_ID],
            "field_coverage_pct": [50.0],
            "admin0_iso3": ["AUT"],
            "admin0_name": ["Austria"],
            "admin1_name": ["Niederösterreich"],
            "admin2_name": [admin2],
        },
        geometry=[box(0, 0, 1, 1)],
        crs="EPSG:4326",
    ).to_parquet(path)
    return path


def _item(root: Path) -> dict:
    return json.loads((root / "chips" / "33UXP" / ITEM_ID / f"{ITEM_ID}.json").read_text())


def _collection(root: Path) -> dict:
    return json.loads((root / "collection.json").read_text())


def _published(tmp_path: Path) -> tuple[Path, dict]:
    """A collection built before admin names and field embeddings existed."""
    root = tmp_path / "published"
    root.mkdir()
    build_catalog(root)
    return root, _collection(root)


def _links(doc: dict, rel: str) -> list[dict]:
    return [link for link in doc["links"] if link["rel"] == rel]


class TestItemProperties:
    def test_match_what_the_stac_stage_publishes(self, tmp_path: Path) -> None:
        chips = _chips_with_admin_names(tmp_path / "ds_chips.parquet", admin2=None)
        build_catalog(tmp_path, chips_path=chips)

        props = backfill.chip_item_properties(chips, year=2024)

        published = {
            k: v for k, v in _item(tmp_path)["properties"].items() if k.startswith("ftw:admin")
        }
        assert props == {ITEM_ID: published}
        assert "ftw:admin2_name" not in props[ITEM_ID]

    def test_update_replaces_stale_names_and_reports_a_change(self) -> None:
        item = {"properties": {"ftw:split": "train", "ftw:admin2_name": "Old", "a": 1}}

        changed = backfill.update_item(item, {"ftw:admin0_name": "Austria"})

        assert changed
        assert item["properties"] == {"ftw:split": "train", "a": 1, "ftw:admin0_name": "Austria"}
        assert not backfill.update_item(item, {"ftw:admin0_name": "Austria"})

    def test_update_leaves_other_chip_properties(self) -> None:
        item = {"properties": {"ftw:split": "train"}}

        backfill.update_item(item, {"ftw:split": "test", "ftw:admin0_iso3": "AUT"})

        assert item["properties"] == {"ftw:split": "train", "ftw:admin0_iso3": "AUT"}


class TestUpdateCollection:
    def _fresh_build(self, tmp_path: Path, embeddings: Path) -> dict:
        """The same collection built with both steps on, to compare against."""
        root = tmp_path / "fresh"
        root.mkdir()
        chips = _chips_with_admin_names(root / "ds_chips.parquet")
        build_catalog(root, chips_path=chips, field_embeddings_file=embeddings)
        return _collection(root)

    def test_matches_a_fresh_build(self, tmp_path: Path) -> None:
        root, doc = _published(tmp_path)
        chips = _chips_with_admin_names(root / "ds_chips.parquet")
        embeddings = _write_field_embeddings(
            root / "ds_field_embeddings.parquet", ("aef", "olmoearth")
        )

        backfill.update_collection(
            doc, chips_file=chips, admin_names_added=True, field_embeddings_file=embeddings
        )

        fresh = self._fresh_build(tmp_path, embeddings)
        assert doc["assets"]["field_embeddings"] == fresh["assets"]["field_embeddings"]
        assert doc["assets"]["chips"] == fresh["assets"]["chips"]
        assert _links(doc, "license") == _links(fresh, "license")
        gaul = [
            link for link in _links(doc, "derived_from") if link["href"] == admin_names.GAUL_URL
        ]
        assert len(gaul) == 1
        assert [link for link in _links(doc, "derived_from") if link not in gaul] == _links(
            fresh, "derived_from"
        )
        assert set(fresh["stac_extensions"]) <= set(doc["stac_extensions"])

    def test_records_the_backfill_without_touching_the_build_config(self, tmp_path: Path) -> None:
        root, doc = _published(tmp_path)
        doc["ftw:config"] = {"config": {"year": 2024}}
        chips = _chips_with_admin_names(root / "ds_chips.parquet")

        backfill.update_collection(
            doc, chips_file=chips, admin_names_added=True, field_embeddings_file=None
        )

        assert doc["ftw:config"]["config"] == {"year": 2024}
        [run] = doc["ftw:config"][backfill.BACKFILL_KEY]
        assert run["steps"] == ["admin_names"]
        assert run["ftwd_version"]

    def test_running_again_changes_nothing(self, tmp_path: Path) -> None:
        root, doc = _published(tmp_path)
        chips = _chips_with_admin_names(root / "ds_chips.parquet")
        embeddings = _write_field_embeddings(
            root / "ds_field_embeddings.parquet", ("aef", "olmoearth")
        )
        kwargs = {"chips_file": chips, "admin_names_added": True}

        backfill.update_collection(doc, field_embeddings_file=embeddings, **kwargs)
        once = json.loads(json.dumps(doc))
        backfill.update_collection(doc, field_embeddings_file=embeddings, **kwargs)

        for run in (*once["ftw:config"]["backfills"], *doc["ftw:config"]["backfills"]):
            run.pop("generated_at")
        assert doc == once

    def test_the_record_follows_the_latest_run(self, tmp_path: Path) -> None:
        root, doc = _published(tmp_path)
        chips = _chips_with_admin_names(root / "ds_chips.parquet")
        embeddings = _write_field_embeddings(root / "ds_field_embeddings.parquet", ("aef",))
        backfill.update_collection(
            doc, chips_file=chips, admin_names_added=True, field_embeddings_file=embeddings
        )

        backfill.update_collection(
            doc, chips_file=chips, admin_names_added=True, field_embeddings_file=None
        )
        assert [r["steps"] for r in doc["ftw:config"]["backfills"]] == [["admin_names"]]
        backfill.update_collection(
            doc, chips_file=chips, admin_names_added=False, field_embeddings_file=None
        )
        assert "backfills" not in doc["ftw:config"]

    def test_refreshes_the_items_mirror_columns(self, tmp_path: Path) -> None:
        root, doc = _published(tmp_path)
        chips = _chips_with_admin_names(root / "ds_chips.parquet")
        fresh = tmp_path / "fresh"
        fresh.mkdir()
        build_catalog(fresh, chips_path=_chips_with_admin_names(fresh / "ds_chips.parquet"))
        backfill.update_collection(
            doc, chips_file=chips, admin_names_added=True, field_embeddings_file=None
        )

        backfill.refresh_parquet_asset(doc, "items", fresh / "items.parquet")

        names = [c["name"] for c in doc["assets"]["items"]["table:columns"]]
        assert "ftw:admin0_name" in names
        assert doc["assets"]["items"] == _collection(fresh)["assets"]["items"]

    def test_dropping_the_embeddings_removes_their_asset_and_links(self, tmp_path: Path) -> None:
        root, doc = _published(tmp_path)
        chips = _chips_with_admin_names(root / "ds_chips.parquet")
        embeddings = _write_field_embeddings(
            root / "ds_field_embeddings.parquet", ("aef", "olmoearth")
        )
        backfill.update_collection(
            doc, chips_file=chips, admin_names_added=True, field_embeddings_file=embeddings
        )

        backfill.update_collection(
            doc, chips_file=chips, admin_names_added=True, field_embeddings_file=None
        )

        assert stac.FIELD_EMBEDDINGS_ASSET not in doc["assets"]
        assert _links(doc, "license") == []
        product_urls = {p.url for p in field_embeddings.PRODUCTS.values()}
        assert not [link for link in doc["links"] if link["href"] in product_urls]

    def test_new_links_sit_before_the_child_links(self, tmp_path: Path) -> None:
        root, doc = _published(tmp_path)
        chips = _chips_with_admin_names(root / "ds_chips.parquet")

        backfill.update_collection(
            doc, chips_file=chips, admin_names_added=True, field_embeddings_file=None
        )

        rels = [link["rel"] for link in doc["links"]]
        assert rels.index("derived_from") < rels.index("child")

    def test_a_collection_without_chips_is_refused(self, tmp_path: Path) -> None:
        root, doc = _published(tmp_path)
        del doc["assets"]["chips"]

        with pytest.raises(backfill.BackfillError, match="no chips asset"):
            backfill.update_collection(
                doc,
                chips_file=root / "ds_chips.parquet",
                admin_names_added=True,
                field_embeddings_file=None,
            )

    def test_a_missing_embeddings_file_is_refused_before_any_edit(self, tmp_path: Path) -> None:
        root, doc = _published(tmp_path)
        before = json.loads(json.dumps(doc))

        with pytest.raises(backfill.BackfillError, match="no field embeddings file"):
            backfill.update_collection(
                doc,
                chips_file=root / "ds_chips.parquet",
                admin_names_added=True,
                field_embeddings_file=root / "missing.parquet",
            )
        assert doc == before


class TestRewriteDocs:
    def test_documents_the_new_columns_and_asset(self, tmp_path: Path) -> None:
        root, doc = _published(tmp_path)
        chips = _chips_with_admin_names(root / "ds_chips.parquet")
        embeddings = _write_field_embeddings(root / "ds_field_embeddings.parquet", ("aef",))
        backfill.update_collection(
            doc, chips_file=chips, admin_names_added=True, field_embeddings_file=embeddings
        )
        (root / "collection.json").write_text(json.dumps(doc))

        written = backfill.rewrite_docs(root, root / "collection.json")

        assert sorted(p.name for p in written) == ["AGENTS.md", "README.md"]
        readme = (root / "README.md").read_text()
        agents = (root / "AGENTS.md").read_text()
        assert "## Field embeddings" in readme
        assert "`admin1_name`" in agents

    def test_honours_the_build_config_turning_a_document_off(self, tmp_path: Path) -> None:
        root, doc = _published(tmp_path)
        doc["ftw:config"] = {"config": {"stages": {"docs": {"agents": False}}}}
        (root / "collection.json").write_text(json.dumps(doc))

        written = backfill.rewrite_docs(root, root / "collection.json")

        assert [p.name for p in written] == ["README.md"]

    def test_writes_nothing_when_the_build_turned_both_documents_off(self, tmp_path: Path) -> None:
        root, doc = _published(tmp_path)
        doc["ftw:config"] = {"config": {"stages": {"docs": {"readme": False, "agents": False}}}}
        (root / "collection.json").write_text(json.dumps(doc))

        assert backfill.rewrite_docs(root, root / "collection.json") == []
        assert not (root / "README.md").exists()

    def test_lists_exactly_the_styles_written(self, tmp_path: Path) -> None:
        root, doc = _published(tmp_path)
        doc["assets"]["chips_tiles"] = {"href": "./ds_chips.pmtiles", "roles": ["visual"]}
        doc["assets"]["style-gone"] = {"href": "./styles/gone.json", "roles": ["style"]}
        (root / "collection.json").write_text(json.dumps(doc))

        backfill.rewrite_docs(root, root / "collection.json")

        assets = _collection(root)["assets"]
        listed = sorted(key for key in assets if key.startswith("style-"))
        on_disk = sorted(f"style-{path.stem}" for path in (root / "styles").glob("*.json"))
        assert "style-gone" not in listed
        assert listed
        assert set(listed) <= set(on_disk)
        assert assets["chips_tiles"] == doc["assets"]["chips_tiles"]

    def test_unchanged_styles_leave_the_collection_alone(self, tmp_path: Path) -> None:
        root, doc = _published(tmp_path)
        doc["assets"]["chips_tiles"] = {"href": "./ds_chips.pmtiles", "roles": ["visual"]}
        (root / "collection.json").write_text(json.dumps(doc))
        backfill.rewrite_docs(root, root / "collection.json")
        published = _collection(root)
        for key in published["assets"]:
            if key.startswith("style-"):
                published["assets"][key]["file:size"] = 123
        (root / "collection.json").write_text(json.dumps(published))

        backfill.rewrite_docs(root, root / "collection.json")

        assert _collection(root) == published

    def test_without_tiles_no_styles_are_written_or_listed(self, tmp_path: Path) -> None:
        root, doc = _published(tmp_path)
        doc["assets"]["style-gone"] = {"href": "./styles/gone.json", "roles": ["style"]}
        (root / "collection.json").write_text(json.dumps(doc))

        backfill.rewrite_docs(root, root / "collection.json")

        assert not any(key.startswith("style-") for key in _collection(root)["assets"])
        assert not (root / "styles").exists()

    def test_a_collection_without_chips_is_refused(self, tmp_path: Path) -> None:
        root, doc = _published(tmp_path)
        del doc["assets"]["chips"]
        (root / "collection.json").write_text(json.dumps(doc))

        with pytest.raises(backfill.BackfillError, match="no chips asset"):
            backfill.rewrite_docs(root, root / "collection.json")

    def test_a_collection_without_fields_is_refused(self, tmp_path: Path) -> None:
        _, doc = _published(tmp_path)
        del doc["assets"]["fields"]

        with pytest.raises(backfill.BackfillError, match="no fields asset"):
            backfill.field_polygons_file(doc, tmp_path)

    def test_measures_the_filtered_fields_when_the_build_filtered(self, tmp_path: Path) -> None:
        root = tmp_path / "published"
        root.mkdir()
        build_catalog(root, filtered=True)

        path = backfill.field_polygons_file(_collection(root), root)

        assert path == root / "ds_fields_filtered.parquet"
