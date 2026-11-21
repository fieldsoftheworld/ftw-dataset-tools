# Agent guidelines

Follow the [Development](README.md#development) section of the README. In addition:

## Code

- `commands/` holds thin Click wrappers; the logic lives in `api/`. `api/` never imports click
  (a pre-commit hook rejects it): it raises domain exceptions defined where the problem is
  found, and `commands/` turns them into `ClickException` or `BadParameter`. Follow
  `StageInputError` (`api/pipeline.py`) or `MaskReadError` (`api/assets.py`).
- `api/` logs through `get_logger(__name__)` from `api/logging_config.py`, with no colours or
  `Warning:` prefixes in messages. Only numeric `(done, total)` progress stays a callback.
- `ftwd -v` is global and commands read it with `is_verbose()`; don't add `-v/--verbose` to new
  commands.
- `ftwd run`, `create-dataset` and the standalone commands share one `api/` function per
  operation (e.g. `download_imagery_for_catalog()`). Change it there and test both paths.
- DuckDB for queries, geoparquet-io (`gpio.read(...).write(...)`) for GeoParquet I/O, rasterio
  for rasters. GeoPandas only for in-memory steps the others can't do, never for I/O.
- Keep new functions at xenon grade A where possible; the hook fails above D.

## Outputs

- `ftwd run` stages reuse what is already on disk, so a rerun after a code change needs a fresh
  output directory or the stage forced.
- Chip ids, mask values, column names and STAC fields are read by
  fieldsoftheworld/benchmark-data-catalog and by users of the published benchmark. Say in the
  PR when a change alters them, and whether published datasets need a rebuild.
- Never commit data or build outputs.

## Tests and checks

- Every fix comes with a test that fails without it. Tests mirror the source: `tests/test_api/`
  for `api/`, `tests/test_cli/` (CliRunner) for `commands/`. Mark tests `slow` or `network`
  where they are.
- Run `pre-commit run --all-files` until it passes without modifying files, then
  `uv run pytest`.

## Writing

- Commit messages: one short imperative line ("Fix bbox check for antimeridian chips"), no
  period.
- Keep PR descriptions, review comments and replies short. Answer questions in a sentence or
  two, and don't repeat what the author already did.
- Write `#123` only for issues and PRs of this repository, `owner/repo#123` for others, and no
  `#` in front of other numbers.
- Measure or run the code instead of assuming how it behaves.
