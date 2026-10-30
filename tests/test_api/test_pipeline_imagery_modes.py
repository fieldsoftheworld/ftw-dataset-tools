"""Pipeline stages with ``imagery_mode: [scenes, mosaics]``: both modes in one catalog."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from ftw_dataset_tools.api import pipeline
from ftw_dataset_tools.api.config import DatasetConfig
from ftw_dataset_tools.api.imagery import mosaic_selection, selection_workflow
from ftw_dataset_tools.api.imagery.slots import MOSAIC_SLOTS
from ftw_dataset_tools.api.imagery.summary import selection_summary_lines

if TYPE_CHECKING:
    from pathlib import Path


_BOTH = {"imagery_mode": ["scenes", "mosaics"]}


def _ctx(fields: Path, tmp_path: Path, stages: dict) -> pipeline.PipelineContext:
    out = tmp_path / "out"
    config = DatasetConfig.from_dict(
        {"fields_file": str(fields), "output_dir": str(out), "year": 2023, "stages": stages}
    )
    ctx = pipeline.build_context(config)
    out.mkdir(parents=True, exist_ok=True)
    (out / "collection.json").write_text("{}")
    return ctx


def _record_selection(monkeypatch: pytest.MonkeyPatch, *, mosaics_exist: bool) -> list[dict]:
    calls: list[dict] = []

    def fake_select(**kwargs: object) -> selection_workflow.SelectionWorkflowResult:
        calls.append(kwargs)
        return selection_workflow.SelectionWorkflowResult(successful=1)

    monkeypatch.setattr(mosaic_selection, "year_available", lambda _year: mosaics_exist)
    monkeypatch.setattr(selection_workflow, "_select_mode", fake_select)
    return calls


class TestSelectBothModes:
    def test_mosaics_use_the_data_year_like_the_scenes(
        self, sample_geoparquet_4326: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = _record_selection(monkeypatch, mosaics_exist=True)
        ctx = _ctx(sample_geoparquet_4326, tmp_path, {"select_images": _BOTH})
        ctx.config.year = None
        ctx.effective_year = 2021  # as if from determination_datetime
        pipeline.stage_select_images(ctx)
        assert [(c["imagery_mode"], c["year"]) for c in calls] == [
            ("scenes", 2021),
            ("mosaics", 2021),
        ]

    def test_runs_scenes_then_mosaics(
        self, sample_geoparquet_4326: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = _record_selection(monkeypatch, mosaics_exist=True)
        ctx = _ctx(sample_geoparquet_4326, tmp_path, {"select_images": _BOTH})
        pipeline.stage_select_images(ctx)

        assert [(c["imagery_mode"], c["year"]) for c in calls] == [
            ("scenes", 2023),
            ("mosaics", 2023),
        ]
        assert ctx.selection_result.successful == 2
        assert selection_summary_lines(ctx.selection_result) == [
            "  Imagery selection (scenes): 1 ok, 0 skipped, 0 failed",
            "  Imagery selection (mosaics): 1 ok, 0 skipped, 0 failed",
        ]

    def test_year_without_mosaics_passes_validation_alongside_scenes(
        self, sample_geoparquet_4326: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _record_selection(monkeypatch, mosaics_exist=False)
        ctx = _ctx(sample_geoparquet_4326, tmp_path, {"select_images": _BOTH})
        pipeline._validate_stage_selection(ctx, ["select_images"])  # no error

    def test_year_without_mosaics_fails_validation_for_mosaics_alone(
        self, sample_geoparquet_4326: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _record_selection(monkeypatch, mosaics_exist=False)
        ctx = _ctx(sample_geoparquet_4326, tmp_path, {"select_images": {"imagery_mode": "mosaics"}})
        with pytest.raises(mosaic_selection.MosaicYearError, match="2023"):
            pipeline._validate_stage_selection(ctx, ["select_images"])


def _download(
    fields: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, select: dict
) -> tuple[list[dict], list[dict]]:
    downloads: list[dict] = []
    previews: list[dict] = []
    empty = SimpleNamespace(
        successful=0, skipped=0, failed=0, failed_details=[], skipped_details=[]
    )

    def fake_download(**kwargs: object) -> object:
        downloads.append(kwargs)
        return empty

    def fake_preview(**kwargs: object) -> object:
        previews.append(kwargs)
        return empty

    monkeypatch.setattr(pipeline, "download_imagery_for_catalog", fake_download)
    monkeypatch.setattr(pipeline, "preview_imagery_for_catalog", fake_preview)
    stages = {"select_images": select, "download_images": {"mode": "preview"}}
    pipeline.stage_download_images(_ctx(fields, tmp_path, stages))
    return downloads, previews


class TestDownloadBothModes:
    def test_preview_previews_scenes_and_clips_mosaics(
        self, sample_geoparquet_4326: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        downloads, previews = _download(sample_geoparquet_4326, tmp_path, monkeypatch, _BOTH)
        assert len(previews) == 1
        assert [d["slots"] for d in downloads] == [MOSAIC_SLOTS]

    def test_preview_with_scenes_alone_downloads_nothing(
        self, sample_geoparquet_4326: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        downloads, previews = _download(
            sample_geoparquet_4326, tmp_path, monkeypatch, {"imagery_mode": "scenes"}
        )
        assert len(previews) == 1
        assert downloads == []

    def test_clip_downloads_every_slot(
        self, sample_geoparquet_4326: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        downloads: list[dict] = []

        def fake_download(**kwargs: object) -> object:
            downloads.append(kwargs)
            return SimpleNamespace(successful=0, skipped=0, failed=0, failed_details=[])

        monkeypatch.setattr(pipeline, "download_imagery_for_catalog", fake_download)
        ctx = _ctx(sample_geoparquet_4326, tmp_path, {"select_images": _BOTH})
        pipeline.stage_download_images(ctx)
        assert downloads[0]["slots"] is None
