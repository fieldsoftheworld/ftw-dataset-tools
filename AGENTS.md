# Agent guidelines

Follow the [Development](README.md#development) section of the README. In addition:

## Code

- `commands/` holds thin Click wrappers; the logic lives in `api/`. `api/` never imports click
  (a pre-commit hook rejects it): it raises domain exceptions defined where the problem is
  found, and `commands/` turns them into `ClickException` or `BadParameter`. Follow
  `StageInputError` (`api/pipeline.py`) or `MaskReadError` (`api/assets.py`).
- `api/` logs through `get_logger(__name__)` from `api/logging_config.py` (not
  `logging.getLogger`), with no colours or `Warning:` prefixes in messages. On a thread pool,
  wrap each task in `capture_logs(task.logs)` and `replay` them in the caller so each task's
  lines stay together. Only numeric progress callbacks stay callbacks.
- `ftwd -v` is global and commands read it with `is_verbose()`; don't add `-v/--verbose` to new
  commands. The aliases on `select-images`, `convert-previews` and `create-dataset-summary` stay
  for old scripts.
- `ftwd run`, `create-dataset` and the standalone commands share one `api/` function per
  operation (e.g. `download_imagery_for_catalog()`). Change it there and test every caller.
- DuckDB for queries, geoparquet-io (`gpio.read(...).write(...)`) for GeoParquet I/O, rasterio
  for rasters. GeoPandas only for in-memory steps the others can't do, never for I/O.
- Aim for xenon grade A in new functions; the hook enforces `--max-absolute=D --max-modules=C
  --max-average=B`.

## Outputs

- Most `ftwd run` stages rebuild their output, but a rerun keeps earlier work: image selections,
  downloads (`stages.download_images.resume`, on by default), field embeddings for unchanged
  fields, and masks with `stages.masks.skip_existing`. After changing selection or download
  code, use a fresh `output_dir`, `ftwd select-images --force` or `resume: false`.
- Chip ids, mask values, column names and STAC fields are read by
  fieldsoftheworld/benchmark-data-catalog and by users of the published benchmark. Say in the
  PR when a change alters them, and whether published datasets need a rebuild.
- Never commit datasets, downloaded inputs or run outputs.

## Tests and checks

- New code comes with tests for the happy path and the errors; every fix with a test that fails
  without it. Tests mirror the source: `tests/test_api/` for `api/`, `tests/test_cli/`
  (CliRunner) for `commands/`. Mark slow tests `slow` and tests that need the network `network`.
- Run `uv run pre-commit run --all-files` until it passes without modifying files, then
  `uv run pytest` (the hook skips `slow` tests).

## Writing

- Commit messages: one short imperative line ("Fix bbox check for antimeridian chips"), no
  period.
- Keep PR descriptions, review comments and replies short. Answer questions in a sentence or
  two, and don't repeat what the author already did.
- Write `#123` only for issues and PRs of this repository, `owner/repo#123` for others, and no
  `#` in front of other numbers.
- Measure or run the code instead of assuming how it behaves.
