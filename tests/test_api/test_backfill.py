"""Tests for bringing a published collection up to date (api/backfill.py)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from ftw_dataset_tools.api import admin_names, backfill, field_embeddings, stac
from ftw_dataset_tools.api.assets import CROP_NAME_NOTES
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

    def test_crop_names_refresh_the_field_asset_and_are_recorded(self, tmp_path: Path) -> None:
        import geopandas as gpd
        from shapely.geometry import box

        root, doc = _published(tmp_path)
        chips = _chips_with_admin_names(root / "ds_chips.parquet")
        fields = root / "ds_fields.parquet"
        gpd.GeoDataFrame(
            {"id": ["a"], "crop:code": ["10"], "crop:name": ["Weizen"]},
            geometry=[box(0, 0, 1, 1)],
            crs="EPSG:4326",
        ).to_parquet(fields)

        backfill.update_collection(
            doc,
            chips_file=chips,
            admin_names_added=True,
            field_embeddings_file=None,
            field_files={"fields": fields},
            crop_names={"source": "fr/year=2024/fr-2024.parquet", "by": "code"},
        )

        columns = [c["name"] for c in doc["assets"]["fields"]["table:columns"]]
        assert "crop:name" in columns
        assert doc["assets"]["fields"]["file:size"] == fields.stat().st_size
        [run] = doc["ftw:config"][backfill.BACKFILL_KEY]
        assert run["steps"] == ["admin_names", "crop_names"]
        assert run["crop_names"] == {"source": "fr/year=2024/fr-2024.parquet", "by": "code"}

    def test_field_files_are_refreshed_without_crop_names(self, tmp_path: Path) -> None:
        """A run that joins no names still describes the field files it publishes."""
        import geopandas as gpd
        from shapely.geometry import box

        root, doc = _published(tmp_path)
        chips = _chips_with_admin_names(root / "ds_chips.parquet")
        fields = root / "ds_fields.parquet"
        gpd.GeoDataFrame(
            {"id": ["a"], "crop:code": ["10"]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:4326"
        ).to_parquet(fields)
        doc["assets"]["fields"]["table:columns"] = [{"name": "crop:name", "type": "string"}]
        doc["assets"]["fields"]["file:size"] = 1

        backfill.update_collection(
            doc,
            chips_file=chips,
            admin_names_added=True,
            field_embeddings_file=None,
            field_files={"fields": fields},
        )

        columns = [c["name"] for c in doc["assets"]["fields"]["table:columns"]]
        assert "crop:name" not in columns
        assert doc["assets"]["fields"]["file:size"] == fields.stat().st_size
        [run] = doc["ftw:config"][backfill.BACKFILL_KEY]
        assert run["steps"] == ["admin_names"]
        assert "crop_names" not in run

    def test_crop_names_without_details_are_still_recorded(self, tmp_path: Path) -> None:
        root, doc = _published(tmp_path)
        chips = _chips_with_admin_names(root / "ds_chips.parquet")
        fields = root / "ds_fields.parquet"
        fields.write_bytes((root / "ds_chips.parquet").read_bytes())
        backfill.update_collection(
            doc,
            chips_file=chips,
            admin_names_added=True,
            field_embeddings_file=None,
            field_files={"fields": fields},
            crop_names={},
        )
        [run] = doc["ftw:config"][backfill.BACKFILL_KEY]
        assert run["steps"] == ["admin_names", "crop_names"]
        assert run["crop_names"] == {}

    def test_crop_names_without_field_files_are_refused(self, tmp_path: Path) -> None:
        root, doc = _published(tmp_path)
        chips = _chips_with_admin_names(root / "ds_chips.parquet")
        before = json.loads(json.dumps(doc))
        with pytest.raises(backfill.BackfillError, match="no field file"):
            backfill.update_collection(
                doc,
                chips_file=chips,
                admin_names_added=True,
                field_embeddings_file=None,
                crop_names={"source": "x", "by": "id"},
            )
        assert doc == before

    def test_crop_names_on_both_field_assets(self, tmp_path: Path) -> None:
        import geopandas as gpd
        from shapely.geometry import box

        root, doc = _published(tmp_path)
        chips = _chips_with_admin_names(root / "ds_chips.parquet")
        files = {}
        for key in ("fields", "fields_filtered"):
            files[key] = root / f"ds_{key}.parquet"
            gpd.GeoDataFrame(
                {"id": ["a"], "crop:code": ["10"], "crop:name": ["Weizen"]},
                geometry=[box(0, 0, 1, 1)],
                crs="EPSG:4326",
            ).to_parquet(files[key])
        doc["assets"]["fields_filtered"] = {**doc["assets"]["fields"], "href": "./x.parquet"}

        backfill.update_collection(
            doc,
            chips_file=chips,
            admin_names_added=True,
            field_embeddings_file=None,
            field_files=files,
            crop_names={"source": "s", "by": "id"},
        )
        for key in files:
            assert "crop:name" in [c["name"] for c in doc["assets"][key]["table:columns"]]

        backfill.update_collection(
            doc, chips_file=chips, admin_names_added=True, field_embeddings_file=None
        )
        [run] = doc["ftw:config"][backfill.BACKFILL_KEY]
        assert run["steps"] == ["admin_names"], "a run without crop names records none"

    def test_a_file_that_fails_to_read_leaves_the_collection_alone(self, tmp_path: Path) -> None:
        root, doc = _published(tmp_path)
        chips = _chips_with_admin_names(root / "ds_chips.parquet")
        broken = root / "broken.parquet"
        broken.write_bytes(b"not parquet")
        before = json.loads(json.dumps(doc))
        with pytest.raises(Exception, match=r"(?i)parquet"):
            backfill.update_collection(
                doc,
                chips_file=chips,
                admin_names_added=True,
                field_embeddings_file=None,
                field_files={"fields": broken},
            )
        assert doc == before

    def test_crop_names_for_a_missing_asset_or_file_are_refused(self, tmp_path: Path) -> None:
        root, doc = _published(tmp_path)
        chips = _chips_with_admin_names(root / "ds_chips.parquet")
        before = json.loads(json.dumps(doc))
        for files, match in (
            ({"fields_filtered": root / "ds_fields.parquet"}, "no fields_filtered asset"),
            ({"fields": root / "nope.parquet"}, "no fields file"),
        ):
            with pytest.raises(backfill.BackfillError, match=match):
                backfill.update_collection(
                    doc,
                    chips_file=chips,
                    admin_names_added=True,
                    field_embeddings_file=None,
                    field_files=files,
                )
            assert doc == before

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


class TestCropNameDescriptions:
    """A run that joins crop names describes the name columns of every field asset."""

    def _run(self, tmp_path: Path, crop_names: dict | None) -> dict:
        import geopandas as gpd
        from shapely.geometry import box

        root, doc = _published(tmp_path)
        chips = _chips_with_admin_names(root / "ds_chips.parquet")
        files = {}
        for key in ("fields", "boundary_lines"):
            files[key] = root / f"ds_{key}.parquet"
            gpd.GeoDataFrame(
                {"id": ["a"], "crop:code": ["10"], "crop:name": ["Weizen"]},
                geometry=[box(0, 0, 1, 1)],
                crs="EPSG:4326",
            ).to_parquet(files[key])
        doc["assets"]["boundary_lines"] = {**doc["assets"]["fields"], "href": "./b.parquet"}
        backfill.update_collection(
            doc,
            chips_file=chips,
            admin_names_added=True,
            field_embeddings_file=None,
            field_files=files,
            crop_names=crop_names,
        )
        return doc

    @staticmethod
    def _note(doc: dict, key: str, name: str = "crop:name") -> str | None:
        [column] = [c for c in doc["assets"][key]["table:columns"] if c["name"] == name]
        return column.get("description")

    def test_by_id_names_the_source(self, tmp_path: Path) -> None:
        doc = self._run(tmp_path, {"source": "pt/year=2021/pt-2021.parquet", "by": "id"})
        for key in ("fields", "boundary_lines"):
            note = self._note(doc, key)
            assert "pt/year=2021/pt-2021.parquet" in note
            assert "matched by crop:code" not in note
        assert self._note(doc, "fields", "crop:code") is None

    def test_by_code_says_so(self, tmp_path: Path) -> None:
        doc = self._run(tmp_path, {"source": "fr/year=2024/fr-2024.parquet", "by": "code"})
        assert "matched by crop:code" in self._note(doc, "fields")

    def test_without_crop_names_the_general_description(self, tmp_path: Path) -> None:
        """A refresh that joined no names still says what the column is, not where from."""
        doc = self._run(tmp_path, None)
        assert self._note(doc, "fields") == CROP_NAME_NOTES["crop:name"]

    def test_notes_without_a_source(self) -> None:
        notes = backfill.crop_name_notes({})
        assert notes["crop:name"] == f"{CROP_NAME_NOTES['crop:name']}, from the harmonized edition"
        assert "English" in notes["crop:name_en"]
