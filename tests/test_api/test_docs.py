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
