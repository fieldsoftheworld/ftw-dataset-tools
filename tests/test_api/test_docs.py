"""Tests for README/AGENTS generation."""

import json
from pathlib import Path


def _collection_dir(tmp_path: Path):
    """A minimal collection on disk: collection.json, chips/fields parquet, items.parquet."""
    import geopandas as gpd
    from shapely.geometry import box

    chips = gpd.GeoDataFrame(
        {
            "id": ["ftw-33UXP0001", "ftw-33UXP0002"],
            "split": ["train", "test"],
            "field_coverage_pct": [20.0, 70.0],
            "hcat_dominant_code": [3301010101, 3302000000],
            "hcat_dominant_name_en": ["Winter wheat", "Pasture"],
        },
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
        crs="EPSG:4326",
    )
    chips.to_parquet(tmp_path / "ds_chips.parquet")
    fields = gpd.GeoDataFrame(
        {
            "id": [1, 2],
            "hcat:code": [3301010101, 3302000000],
            "hcat:name_en": ["Winter wheat", "Pasture"],
        },
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
        crs="EPSG:4326",
    )
    fields.to_parquet(tmp_path / "ds_fields.parquet")
    items = gpd.GeoDataFrame(
        {
            "id": ["ftw-33UXP0001_2024", "ftw-33UXP0002_2024"],
            "collection": ["ds", "ds"],
            "ftw:split": ["train", "test"],
        },
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
        crs="EPSG:4326",
    )
    items.to_parquet(tmp_path / "items.parquet")
    collection = {
        "type": "Collection",
        "id": "ds",
        "stac_version": "1.1.0",
        "description": "Test chips",
        "title": "Test",
        "license": "CC0-1.0",
        "extent": {
            "spatial": {"bbox": [[0, 0, 2, 1]]},
            "temporal": {"interval": [["2024-01-01T00:00:00Z", "2024-12-31T00:00:00Z"]]},
        },
        "providers": [{"name": "Producer X", "roles": ["producer"], "url": "https://x.example"}],
        "links": [],
        "assets": {
            "fields": {
                "href": "./ds_fields.parquet",
                "type": "application/vnd.apache.parquet",
                "roles": ["data"],
            },
            "chips": {
                "href": "./ds_chips.parquet",
                "type": "application/vnd.apache.parquet",
                "roles": ["data"],
            },
            "items": {
                "href": "./items.parquet",
                "type": "application/vnd.apache.parquet",
                "roles": ["collection-mirror"],
            },
        },
        "item_assets": {"semantic_2class_mask": {"type": "image/tiff", "roles": ["labels"]}},
        "ftw:split_type": "block3x3",
    }
    (tmp_path / "collection.json").write_text(json.dumps(collection, indent=2))
    return (
        tmp_path / "collection.json",
        tmp_path / "ds_chips.parquet",
        tmp_path / "ds_fields.parquet",
    )


class TestCollectStats:
    def test_counts_and_crops(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.docs import collect_stats

        _, chips, fields = _collection_dir(tmp_path)
        stats = collect_stats(tmp_path, chips, fields)

        assert stats["chips_total"] == 2
        assert stats["split_counts"] == {"train": 1, "test": 1}
        assert stats["fields_total"] == 2
        assert [c for c, _, _ in stats["top_crops"]] == [3301010101, 3302000000]
        assert stats["imagery"] is None

    def test_split_type_from_collection_or_config(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.docs import collect_stats

        _, chips, fields = _collection_dir(tmp_path)

        # Falls back to the collection's ftw:split_type when no config is given.
        assert collect_stats(tmp_path, chips, fields)["split_type"] == "block3x3"

        # An explicit config_dict split_type takes precedence.
        stats = collect_stats(
            tmp_path,
            chips,
            fields,
            {"stages": {"splits": {"split_type": "random-uniform"}}},
        )
        assert stats["split_type"] == "random-uniform"


class TestWriteDocs:
    def test_readme_and_agents_are_computed_and_linked(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.docs import write_docs

        coll_path, chips, fields = _collection_dir(tmp_path)
        styles = []
        paths = write_docs(
            tmp_path,
            coll_path,
            chips,
            fields,
            styles,
            {"stages": {"splits": {"split_type": "block3x3"}}},
        )

        assert [p.name for p in paths] == ["README.md", "AGENTS.md"]
        readme = (tmp_path / "README.md").read_text()
        assert "# Test" in readme
        assert "| train | 1 |" in readme and "| test | 1 |" in readme
        assert "Winter wheat" in readme
        assert "[Producer X](https://x.example)" in readme
        assert "https://" not in readme.replace("](https://", "")  # every URL is a markdown link

        agents = (tmp_path / "AGENTS.md").read_text()
        assert "## Overview" in agents and "## Example queries" in agents
        assert "read_parquet('items.parquet')" in agents
        assert "-- result:" in agents  # queries were executed and results inlined
        assert "```sql" in agents

    def test_agents_only(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.docs import write_docs

        coll_path, chips, fields = _collection_dir(tmp_path)
        paths = write_docs(tmp_path, coll_path, chips, fields, [], {}, readme=False)

        assert [p.name for p in paths] == ["AGENTS.md"]
        assert not (tmp_path / "README.md").exists()


def _minimal_collection_dir(tmp_path: Path):
    """A collection whose chips carry only id and field_coverage_pct."""
    import geopandas as gpd
    from shapely.geometry import box

    gpd.GeoDataFrame(
        {"id": ["a", "b"], "field_coverage_pct": [10.0, 90.0]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
        crs="EPSG:4326",
    ).to_parquet(tmp_path / "min_chips.parquet")
    gpd.GeoDataFrame({"id": [1]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:4326").to_parquet(
        tmp_path / "min_fields.parquet"
    )
    gpd.GeoDataFrame(
        {"id": ["a", "b"]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
        crs="EPSG:4326",
    ).to_parquet(tmp_path / "items.parquet")
    collection = {
        "type": "Collection",
        "id": "min",
        "description": "Minimal",
        "title": "Minimal",
        "license": "CC-BY-4.0",
        "links": [],
        "assets": {
            "chips": {"href": "./min_chips.parquet"},
            "fields": {"href": "./min_fields.parquet"},
            "items": {"href": "./items.parquet"},
        },
    }
    (tmp_path / "collection.json").write_text(json.dumps(collection))
    return (
        tmp_path / "collection.json",
        tmp_path / "min_chips.parquet",
        tmp_path / "min_fields.parquet",
    )


class TestDegradedCollection:
    def test_absent_columns_degrade_rather_than_placeholder(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.docs import collect_stats, write_docs

        coll_path, chips, fields = _minimal_collection_dir(tmp_path)
        stats = collect_stats(tmp_path, chips, fields)

        assert stats["split_counts"] == {}
        assert stats["top_crops"] == []
        assert stats["mask_types"] == []
        assert stats["imagery"] is None
        assert stats["coverage_quantiles"][50] == 50.0

        paths = write_docs(tmp_path, coll_path, chips, fields, [], {})
        assert [p.name for p in paths] == ["README.md", "AGENTS.md"]

        readme = (tmp_path / "README.md").read_text()
        assert "| split | chips |" not in readme
        assert "## Crops" not in readme
        assert "## Styles" not in readme
        assert "| percentile | field coverage |" in readme

        agents = (tmp_path / "AGENTS.md").read_text()
        # Queries whose columns are absent are dropped, not printed with an error.
        assert "ftw:split" not in agents
        assert "hcat_dominant_code" not in agents
        assert "ORDER BY field_coverage_pct DESC" in agents
        assert "This collection stands alone" in agents


class TestVia:
    def test_via_link_is_a_markdown_link(self) -> None:
        from ftw_dataset_tools.api.docs import render_agents, render_readme

        collection = {
            "title": "Test",
            "description": "d",
            "links": [{"rel": "via", "href": "https://src.example", "title": "Source X"}],
            "providers": [{"name": "No URL Provider", "roles": ["licensor"]}],
        }
        stats = {
            "chips_total": 1,
            "fields_total": 0,
            "split_counts": {},
            "coverage_quantiles": {},
            "top_crops": [],
            "imagery": None,
            "mask_types": [],
            "chip_columns": [],
            "item_properties": [],
        }
        readme = render_readme(collection, stats, [], {})
        assert "[Source X](https://src.example)" in readme
        assert "- No URL Provider: licensor" in readme
        assert "https://" not in readme.replace("](https://", "")

        agents = render_agents(collection, stats, [])
        assert "[Source X](https://src.example)" in agents
        assert "https://" not in agents.replace("](https://", "")


class TestRegisterDocsAssets:
    """Tests for JSON-editing the saved collection with tiles, styles and docs."""

    def _styles(self, tmp_path: Path):
        from ftw_dataset_tools.api.styles import StyleResult

        styles_dir = tmp_path / "styles"
        styles_dir.mkdir(exist_ok=True)
        for style_id in ("split", "outline"):
            (styles_dir / f"{style_id}.json").write_text("{}")
        return [
            StyleResult("split", styles_dir / "split.json", "Chips by split", [], True),
            StyleResult("outline", styles_dir / "outline.json", "Field outlines", [], False),
        ]

    def _tiles(self, tmp_path: Path) -> dict[str, Path]:
        for name in ("chips.pmtiles", "fields.pmtiles"):
            (tmp_path / name).write_bytes(b"PMTiles" * 10)
        return {
            "chips_tiles": tmp_path / "chips.pmtiles",
            "fields_tiles": tmp_path / "fields.pmtiles",
        }

    def test_registers_tiles_styles_and_docs(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.docs import register_docs_assets

        coll_path, _, _ = _collection_dir(tmp_path)
        docs = [tmp_path / "README.md", tmp_path / "AGENTS.md"]
        for doc in docs:
            doc.write_text("# doc")

        register_docs_assets(
            coll_path,
            tiles=self._tiles(tmp_path),
            styles=self._styles(tmp_path),
            docs=docs,
        )

        coll = json.loads(coll_path.read_text())
        chips_tiles = coll["assets"]["chips_tiles"]
        assert chips_tiles["type"] == "application/vnd.pmtiles"
        assert chips_tiles["roles"] == ["visual"]
        assert chips_tiles["title"] == "Chips (PMTiles)"
        assert chips_tiles["href"] == "./chips.pmtiles"
        assert chips_tiles["file:size"] == (tmp_path / "chips.pmtiles").stat().st_size
        assert coll["assets"]["fields_tiles"]["title"] == "Field boundaries (PMTiles)"
        assert coll["assets"]["fields_tiles"]["href"] == "./fields.pmtiles"

        default_style = coll["assets"]["style-split"]
        assert default_style["type"] == "application/vnd.mapbox.style+json"
        assert default_style["roles"] == ["style", "default"]
        assert default_style["title"] == "Chips by split"
        assert default_style["href"] == "./styles/split.json"
        assert coll["assets"]["style-outline"]["roles"] == ["style"]

        links = {(link["rel"], link["href"]): link for link in coll["links"]}
        assert links[("describedby", "./README.md")]["type"] == "text/markdown"
        assert links[("describedby", "./README.md")]["title"] == "Collection README"
        assert links[("agents", "./AGENTS.md")]["title"] == "Collection agent guide"

    def test_preserves_existing_keys_and_appends(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.docs import register_docs_assets

        coll_path, _, _ = _collection_dir(tmp_path)
        before = json.loads(coll_path.read_text())

        register_docs_assets(coll_path, tiles={}, styles=[], docs=[tmp_path / "README.md"])

        coll = json.loads(coll_path.read_text())
        assert list(coll)[: len(before)] == list(before)  # top-level key order preserved
        assert list(coll["assets"])[:3] == list(before["assets"])[:3]
        assert coll["assets"]["fields"] == before["assets"]["fields"]
        assert coll_path.read_text().startswith('{\n  "type"')  # indent 2

    def test_is_idempotent(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.docs import register_docs_assets

        coll_path, _, _ = _collection_dir(tmp_path)
        docs = [tmp_path / "README.md", tmp_path / "AGENTS.md"]
        for doc in docs:
            doc.write_text("# doc")
        kwargs = {"tiles": self._tiles(tmp_path), "styles": self._styles(tmp_path), "docs": docs}

        register_docs_assets(coll_path, **kwargs)
        first = coll_path.read_text()
        register_docs_assets(coll_path, **kwargs)
        second = coll_path.read_text()

        assert first == second
        coll = json.loads(second)
        rels = [link["rel"] for link in coll["links"]]
        assert rels.count("describedby") == 1 and rels.count("agents") == 1
        assert len([k for k in coll["assets"] if k.startswith("style-")]) == 2

    def test_only_written_docs_are_linked(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.docs import register_docs_assets

        coll_path, _, _ = _collection_dir(tmp_path)
        (tmp_path / "AGENTS.md").write_text("# doc")

        register_docs_assets(coll_path, tiles={}, styles=[], docs=[tmp_path / "AGENTS.md"])

        coll = json.loads(coll_path.read_text())
        rels = {link["rel"] for link in coll["links"]}
        assert rels == {"agents"}


def _stats(**overrides) -> dict:
    """A stats dict with every optional measurement empty, for renderer-level tests."""
    stats = {
        "chips_total": 2,
        "fields_total": 0,
        "split_counts": {},
        "coverage_quantiles": {},
        "top_crops": [],
        "imagery": None,
        "mask_types": [],
        "chip_columns": [],
        "item_properties": [],
    }
    stats.update(overrides)
    return stats


class TestConditionalProse:
    """ "Suggested uses" and "Limitations" only claim what the data supports."""

    def test_imagery_limitation_only_when_imagery_present(self) -> None:
        from ftw_dataset_tools.api.docs import render_readme

        collection = {"title": "T", "description": "d", "links": []}
        without = render_readme(collection, _stats(), [], {})
        assert "crop calendar" not in without
        assert "imagery" not in without.lower()

        imagery = {
            "chips_with_imagery": 2,
            "planting": {
                "min": "2024-04-01T10:00:00Z",
                "max": "2024-04-20T10:00:00Z",
                "cloud_cover_avg": 3.5,
                "cloud_cover_max": 7.0,
            },
            "harvest": {"min": "2024-08-01T10:00:00Z", "max": "2024-08-20T10:00:00Z"},
        }
        with_imagery = render_readme(collection, _stats(imagery=imagery), [], {})
        assert "2 chips have Sentinel-2 scenes selected" in with_imagery
        assert "2024-04-01T10:00:00Z and 2024-04-20T10:00:00Z" in with_imagery
        assert "3.5% cloud cover" in with_imagery
        assert "Harvest imagery was acquired between 2024-08-01T10:00:00Z" in with_imagery
        assert "crop calendar" in with_imagery

    def test_crop_suggestion_only_when_crops_measured(self) -> None:
        from ftw_dataset_tools.api.docs import render_readme

        collection = {"title": "T", "description": "d", "links": []}
        assert "HCAT" not in render_readme(collection, _stats(), [], {})
        with_crops = render_readme(
            collection, _stats(top_crops=[(3301010101, "Winter wheat", 1.0)]), [], {}
        )
        assert "harmonized HCAT codes" in with_crops

    def test_grid_suggestion_only_when_chips_carry_an_id(self) -> None:
        from ftw_dataset_tools.api.docs import render_readme

        collection = {"title": "T", "description": "d", "links": []}
        assert "grid" not in render_readme(collection, _stats(), [], {})
        with_id = render_readme(collection, _stats(chip_columns=["id", "geometry"]), [], {})
        assert "grid" in with_id

    def test_split_claim_only_when_splits_measured(self) -> None:
        from ftw_dataset_tools.api.docs import render_readme

        collection = {"title": "T", "description": "d", "links": []}
        assert "split" not in render_readme(collection, _stats(), [], {})
        with_splits = render_readme(collection, _stats(split_counts={"train": 2}), [], {})
        assert "pre-assigned" in with_splits

    def test_minimal_collection_readme_never_mentions_imagery(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.docs import write_docs

        coll_path, chips, fields = _minimal_collection_dir(tmp_path)
        write_docs(tmp_path, coll_path, chips, fields, [], {})

        readme = (tmp_path / "README.md").read_text()
        assert "imagery" not in readme.lower()
        assert "Sentinel" not in readme
        assert "HCAT" not in readme


class TestStylesSection:
    def test_lists_style_titles_and_marks_the_default(self) -> None:
        from ftw_dataset_tools.api.docs import render_readme
        from ftw_dataset_tools.api.styles import StyleResult

        styles = [
            StyleResult("split", Path("styles/split.json"), "Chips by split", [], True),
            StyleResult("outline", Path("styles/outline.json"), "Field outlines", [], False),
        ]
        readme = render_readme({"title": "T", "links": []}, _stats(), styles, {})

        assert "## Styles" in readme
        assert "- **Chips by split** (the default view):" in readme
        assert "- **Field outlines**:" in readme
        assert "train / val / test" in readme
        assert "for reading the boundaries themselves" in readme


class TestImageryStats:
    def test_reads_season_child_items_from_disk(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.docs import imagery_stats

        chip_dir = tmp_path / "chips" / "33UXP" / "ftw-33UXP0001"
        chip_dir.mkdir(parents=True)
        (chip_dir / "ftw-33UXP0001.json").write_text(json.dumps({"id": "ftw-33UXP0001"}))
        for season, when, cloud in (
            ("planting", "2024-04-05T10:00:00Z", 2.0),
            ("harvest", "2024-08-11T10:00:00Z", 8.0),
        ):
            (chip_dir / f"ftw-33UXP0001_{season}_s2.json").write_text(
                json.dumps(
                    {
                        "id": f"ftw-33UXP0001_{season}_s2",
                        "properties": {
                            "ftw:season": season,
                            "datetime": when,
                            "eo:cloud_cover": cloud,
                        },
                    }
                )
            )

        stats = imagery_stats(tmp_path)

        assert stats is not None
        assert stats["chips_with_imagery"] == 1
        assert stats["planting"]["min"] == "2024-04-05T10:00:00Z"
        assert stats["planting"]["cloud_cover_max"] == 2.0
        assert stats["harvest"]["max"] == "2024-08-11T10:00:00Z"

    def test_returns_none_without_child_items(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.docs import imagery_stats

        assert imagery_stats(tmp_path) is None


class TestRegisterDocsAssetsPruning:
    """A rerun that produces less must leave the collection describing only what exists."""

    def _register(self, coll_path: Path, tmp_path: Path, **kwargs):
        from ftw_dataset_tools.api.docs import register_docs_assets

        helper = TestRegisterDocsAssets()
        defaults = {
            "tiles": helper._tiles(tmp_path),
            "styles": helper._styles(tmp_path),
            "docs": [tmp_path / "README.md", tmp_path / "AGENTS.md"],
        }
        for doc in defaults["docs"]:
            doc.write_text("# doc")
        defaults.update(kwargs)
        register_docs_assets(coll_path, **defaults)

    def _with_links(self, tmp_path: Path) -> Path:
        """The shared collection, plus the links the stac stage really writes."""
        coll_path, _, _ = _collection_dir(tmp_path)
        collection = json.loads(coll_path.read_text())
        collection["links"] = [
            {"rel": "root", "href": "./collection.json", "type": "application/json"},
            {"rel": "child", "href": "./chips/33UXP/collection.json"},
            {"rel": "via", "href": "https://x.example/collection.json"},
        ]
        coll_path.write_text(json.dumps(collection, indent=2))
        return coll_path

    def test_dropped_tiles_and_styles_are_removed(self, tmp_path: Path) -> None:
        coll_path = self._with_links(tmp_path)

        self._register(coll_path, tmp_path)
        self._register(coll_path, tmp_path, tiles={}, styles=[])

        coll = json.loads(coll_path.read_text())
        assert "chips_tiles" not in coll["assets"] and "fields_tiles" not in coll["assets"]
        assert [k for k in coll["assets"] if k.startswith("style-")] == []
        # The documents this run still produced stay linked.
        rels = {link["rel"] for link in coll["links"]}
        assert {"describedby", "agents"} <= rels

    def test_dropped_docs_lose_their_links(self, tmp_path: Path) -> None:
        coll_path = self._with_links(tmp_path)

        self._register(coll_path, tmp_path)
        self._register(coll_path, tmp_path, docs=[tmp_path / "AGENTS.md"])

        coll = json.loads(coll_path.read_text())
        rels = [link["rel"] for link in coll["links"]]
        assert "describedby" not in rels
        assert rels.count("agents") == 1
        # Tiles and styles were still produced, so they stay.
        assert "chips_tiles" in coll["assets"]

    def test_unrelated_assets_and_links_survive_pruning(self, tmp_path: Path) -> None:
        coll_path = self._with_links(tmp_path)
        before = json.loads(coll_path.read_text())

        self._register(coll_path, tmp_path)
        self._register(coll_path, tmp_path, tiles={}, styles=[], docs=[])

        coll = json.loads(coll_path.read_text())
        assert list(coll["assets"]) == list(before["assets"])
        for key in ("fields", "chips", "items"):
            assert coll["assets"][key] == before["assets"][key]
        assert coll["links"] == before["links"]

    def test_pruning_leaves_a_steady_state_idempotent(self, tmp_path: Path) -> None:
        coll_path = self._with_links(tmp_path)

        self._register(coll_path, tmp_path)
        self._register(coll_path, tmp_path, tiles={}, styles=[])
        first = coll_path.read_text()
        self._register(coll_path, tmp_path, tiles={}, styles=[])

        assert coll_path.read_text() == first


class TestSplitQualityNote:
    """AGENTS.md's split caveat matches how splits were actually assigned."""

    def test_block_split_says_spatially_blocked(self) -> None:
        from ftw_dataset_tools.api.docs import render_agents

        collection = {"title": "T", "links": []}
        stats = _stats(split_counts={"train": 2}, split_type="block3x3")

        agents = render_agents(collection, stats, [])

        assert "spatially blocked" in agents

    def test_random_uniform_split_warns_about_possible_leakage(self) -> None:
        from ftw_dataset_tools.api.docs import render_agents

        collection = {"title": "T", "links": []}
        stats = _stats(split_counts={"train": 2}, split_type="random-uniform")

        agents = render_agents(collection, stats, [])

        assert "uniformly at random" in agents
        assert "spatially blocked" not in agents

    def test_predefined_split_credits_the_source_dataset(self) -> None:
        from ftw_dataset_tools.api.docs import render_agents

        collection = {"title": "T", "links": []}
        stats = _stats(split_counts={"train": 2}, split_type="predefined")

        agents = render_agents(collection, stats, [])

        assert "source dataset" in agents
        assert "spatially blocked" not in agents
        assert "uniformly at random" not in agents

    def test_unknown_split_type_omits_the_note(self) -> None:
        from ftw_dataset_tools.api.docs import render_agents

        collection = {"title": "T", "links": []}
        stats = _stats(split_counts={"train": 2}, split_type=None)

        agents = render_agents(collection, stats, [])

        assert "spatially blocked" not in agents
        assert "uniformly at random" not in agents
        assert "source dataset" not in agents

    def test_no_split_column_omits_the_note_even_with_a_split_type(self) -> None:
        from ftw_dataset_tools.api.docs import render_agents

        collection = {"title": "T", "links": []}
        stats = _stats(split_type="block3x3")

        agents = render_agents(collection, stats, [])

        assert "spatially blocked" not in agents


class TestSchemaNotes:
    """gzd/mgrs_10km/hcat_top and the ftw:hcat_* properties get real descriptions."""

    def test_grid_columns_have_specific_notes(self) -> None:
        from ftw_dataset_tools.api.docs import render_agents

        collection = {"title": "T", "links": []}
        stats = _stats(chip_columns=["gzd", "mgrs_10km", "hcat_top"])

        agents = render_agents(collection, stats, [])

        assert "carried through from the source dataset" not in agents
        assert "MGRS grid zone designator" in agents
        assert "MGRS 100 km square plus 10 km cell identifier" in agents
        assert "ordered by share (top 5)" in agents

    def test_ftw_hcat_properties_reuse_the_chip_column_notes(self) -> None:
        from ftw_dataset_tools.api.docs import render_agents

        collection = {"title": "T", "links": []}
        stats = _stats(
            item_properties=[
                "ftw:hcat_dominant_code",
                "ftw:hcat_dominant_name_en",
                "ftw:hcat_dominant_pct",
                "ftw:hcat_top",
            ]
        )

        agents = render_agents(collection, stats, [])

        assert "see the item JSON" not in agents
        assert "ordered by share (top 5)" in agents


class TestLicenseLink:
    """README/AGENTS license mentions link out when there is somewhere to link to."""

    def test_spdx_license_links_to_spdx_org(self) -> None:
        from ftw_dataset_tools.api.docs import _readme_provenance

        collection = {"license": "CC-BY-4.0", "links": []}

        text = _readme_provenance(collection, {})

        assert "[CC-BY-4.0](https://spdx.org/licenses/CC-BY-4.0.html)" in text

    def test_other_license_with_titled_rel_link_uses_the_title(self) -> None:
        from ftw_dataset_tools.api.docs import _readme_provenance

        collection = {
            "license": "other",
            "links": [
                {"rel": "license", "href": "https://x.example/terms", "title": "Custom Terms"}
            ],
        }

        text = _readme_provenance(collection, {})

        assert "[Custom Terms](https://x.example/terms)" in text

    def test_other_license_with_untitled_rel_link_uses_a_generic_title(self) -> None:
        from ftw_dataset_tools.api.docs import _readme_provenance

        collection = {
            "license": "proprietary",
            "links": [{"rel": "license", "href": "https://x.example/terms"}],
        }

        text = _readme_provenance(collection, {})

        assert "[License terms](https://x.example/terms)" in text

    def test_other_license_without_a_rel_link_is_bare(self) -> None:
        from ftw_dataset_tools.api.docs import _readme_provenance

        collection = {"license": "other", "links": []}

        text = _readme_provenance(collection, {})

        assert "License: other" in text
        assert "[other]" not in text

    def test_agents_overview_links_the_license_too(self) -> None:
        from ftw_dataset_tools.api.docs import render_agents

        collection = {"title": "T", "links": [], "license": "CC0-1.0"}

        agents = render_agents(collection, _stats(), [])

        assert "[CC0-1.0](https://spdx.org/licenses/CC0-1.0.html)" in agents


class TestGrammar:
    """ "covering the most of" is grammatically broken; the fix drops "the"."""

    def test_docs_notes_avoid_the_double_article(self) -> None:
        from ftw_dataset_tools.api.docs import CHIP_COLUMN_NOTES, STYLE_BLURBS, _readme_crops

        assert "covering the most of" not in CHIP_COLUMN_NOTES["hcat_dominant_code"]
        assert "covering the most of" not in STYLE_BLURBS["dominant-crop"]
        text = _readme_crops(_stats(top_crops=[(3301010101, "Winter wheat", 1.0)]))
        assert "covering the most of" not in text

    def test_style_descriptions_avoid_the_double_article(self) -> None:
        from ftw_dataset_tools.api.styles import dominant_crop_style

        rows = [(3301010101, "Winter wheat", 1.0)]
        _, style, _ = dominant_crop_style(rows, tiles_href="../chips.pmtiles", layer="chips")

        assert "covering the most of" not in style["metadata"]["description"]
