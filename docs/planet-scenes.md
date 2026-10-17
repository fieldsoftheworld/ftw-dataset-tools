# PlanetScope SR and UDM2 downloads

`ftwd planet-scenes` turns dated AOIs into four-band PlanetScope surface
reflectance crops and eight-band UDM2 companions. Search, activation, extraction,
and quality reporting run independently, locally or as SLURM array jobs. Python
functions live in `ftw_dataset_tools.api.planet`; the CLI wraps those functions.

## Install and authenticate

```bash
uv sync --locked --dev
export PL_API_KEY='your Planet API key'
uv run ftwd planet-scenes --help
```

`PLANET_API_KEY` is also accepted. Supply credentials through your shell or
cluster secret mechanism. No hardcoded account, partition, filesystem path, or
credentials are required. The account needs access to PSScene
`ortho_analytic_4b_sr` and `ortho_udm2`. Activation and access to imagery remain
subject to Planet permissions and quota.

## Input

Use UTF-8 JSONL with one object per AOI/time window:

```json
{"id":"example_a","geometry":{"type":"Polygon","coordinates":[[[-90.01,38.60],[-90.00,38.60],[-90.00,38.61],[-90.01,38.61],[-90.01,38.60]]]},"start":"2024-06-01","end":"2024-07-01","target_date":"2024-06-15"}
```

Geometry is an EPSG:4326 Polygon or MultiPolygon; split antimeridian-crossing
AOIs first. IDs are unique, start with a letter/digit, and use only letters,
digits, `.`, `_`, and `-`. Give each seasonal window its own ID. The interval is
`start <= acquisition < end`; the target must fall inside it. Naive dates mean
UTC; timezone offsets are normalized. Bounds are never expanded automatically.

For existing FTW country indexes:

```bash
uv run ftwd planet-scenes manifest data/ftw aois.jsonl
```

The converter reads `COUNTRY/index.jsonl` containing `id`, `geometry_4326`,
`win_a_date`, `win_b_date`, and `[start, end]` in `win_a_range`/`win_b_range`.
Check the source index's end-date convention against the half-open interval
above. Output IDs include country and window. The converter refuses to
overwrite a manifest. For other catalogs, export those six manifest fields
directly; these commands do not alter STAC items or the Sentinel-2 workflow.

## Local workflow

```bash
uv run ftwd planet-scenes search aois.jsonl --work-dir runs/scenes --workers 4
uv run ftwd planet-scenes activate --work-dir runs/scenes --workers 4
uv run ftwd planet-scenes extract --work-dir runs/scenes --workers 4
uv run ftwd planet-scenes quality --work-dir runs/scenes
```

Search uses standard-quality scenes, a scene cloud ceiling of 0.1 and AOI
coverage of at least 0.99. It follows pagination and ranks by time distance,
cloud fraction, then scene ID. Up to ten candidates are retained. Tune with
`--max-cloud`, `--min-coverage`, and `--max-candidates`.

`activate` is an optional prewarming stage. SR and UDM2 are independent tasks,
deduplicated by scene/asset, with bounded polling (`--activation-timeout`,
default 3600 seconds per activation). `extract` can activate assets itself.
It groups all AOIs sharing a scene, opens each asset once, and reads only the
required windows. URLs are obtained afresh, kept in memory, and refreshed once
if a read fails. Request retries are bounded and cover HTTP 429/5xx and transport
errors; access errors fail immediately.

## Outputs and recovery

```text
runs/scenes/
  inputs.json                   # frozen manifest
  search/config.json            # frozen search settings and shard count
  search/shard-00000.jsonl       # candidates and empty/failed searches
  activate/shard-00000.jsonl     # activation status, no signed URLs
  extract/shard-00000.jsonl      # scene status and pending AOI/assets
  selections.jsonl              # optional alternate ranks
  quality.jsonl                 # append-only current quality observations
  images/ID_sr.tif
  images/ID_udm2.tif
```

Crops preserve each source's native CRS, pixel grid, bands, and data type.
They cover the AOI bounding box; they are not warped to a label grid. The SR
and UDM2 source grids should be checked before pixelwise use. Pixels beyond
the UDM2 scene edge are marked as blackfill. Each output records its source
scene/asset and input signature. Atomic writes preserve old outputs if an
attempt fails; old outputs with a different signature are not current data.

Rerun a failed stage with the same arguments. Completed searches are reused;
failed and empty searches retry. An empty search is a recorded outcome, not a
network error. Finish all search shards before activation/extraction. Repeated
extracts validate files, not just progress records: missing/corrupt SR or UDM2
is repaired independently. Changed input geometry, dates, or search settings
require a new work directory. Worker counts may change. Keep a single writer
per stage/shard and run stages in order. Never run two extracts of the same
shard simultaneously or edit selections while extraction is running.

Read journals with `ftw_dataset_tools.api.planet.state.latest(path)` (last row
per `id` wins), or `collect(directory)` for all shard journals. Failed jobs
produce a nonzero exit code, with credential-safe error class names. Inspect
the job ID and retry stage when diagnosing a failure. A torn final journal
line can be recovered; malformed input and interior corruption fail explicitly.

Quality fractions use AOI pixel centers, including polygon holes and blackfill
in the denominator. Band 8 is decoded as a bit field: bit 0 is blackfill, and
any nonzero value is unusable. Confidence averages only observed pixels.
`usable` requires clear fraction >= `--min-clear` (default 0.95) and unusable
fraction <= its complement. UDM2.1's retained heavy-haze band may be all zero.
A quality failure does not delete data or automatically change a selected scene.

For a cloudy patch, inspect its cached candidate list and choose an alternate:

```bash
uv run ftwd planet-scenes select example_a --rank 1 --work-dir runs/scenes
uv run ftwd planet-scenes extract --work-dir runs/scenes
uv run ftwd planet-scenes quality --work-dir runs/scenes
```

Ranks are zero-based. Changing the selection replaces both companions on the
next successful extract; stale UDM2 is rejected by the quality stage until then.
Rerun quality after each recovery. Retain `inputs.json`, search results,
selections, and quality records alongside the imagery for provenance.

## SLURM

Install the environment once on a filesystem visible to compute nodes. Set
your scheduler account/partition at submission time. From the repository root:

```bash
mkdir -p logs
export MANIFEST="$PWD/aois.jsonl" WORK_DIR="$PWD/runs/scenes" NUM_SHARDS=8 WORKERS=4
search_job=$(sbatch --parsable --array=0-7 examples/planet-scenes/stage.sbatch search)
activate_job=$(sbatch --parsable --array=0-7 --dependency=afterok:$search_job examples/planet-scenes/stage.sbatch activate)
extract_job=$(sbatch --parsable --array=0-7 --dependency=afterok:$activate_job examples/planet-scenes/stage.sbatch extract)
sbatch --dependency=afterok:$extract_job examples/planet-scenes/stage.sbatch quality
```

Search shards by AOI ID; activation/extraction shard by scene ID using SHA-256,
so Python hash randomization cannot duplicate or omit scenes. Use the same
shard count when rerunning a stage. Quality uses one process. The template
uses the already-installed `.venv/bin/ftwd`; adjust `FTWD` if needed. Total API
load is array concurrency times workers: start conservatively, limit concurrent
array tasks with `%N`, and reduce workers if rate limited.

## Validation and references

Offline tests exercise paginated search, deadline/refresh behavior, real raster
crops, companion recovery, scene grouping, changed selections, and UDM2 masks.
They do not prove account access, quota, or cluster throughput. Before a large
run, use a single authorized AOI and inspect SR/UDM2 pixels and quality output.

See Planet's [Data API search](https://docs.planet.com/develop/apis/data/item-search/),
[asset activation](https://docs.planet.com/develop/apis/data/items/), and
[UDM band definitions](https://docs.planet.com/data/imagery/udm/).
