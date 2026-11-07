# PlanetScope downloads

`ftwd planet-scenes` downloads four-band surface reflectance and eight-band
UDM2 crops for polygons from fiboa/other GeoParquet files or JSONL. Each stage
is also a Python function in `ftw_dataset_tools.api.planet.scene_workflow`.

## Run

```bash
uv sync --locked --dev
export PL_API_KEY='your Planet API key'

uv run ftwd planet-scenes search fields.parquet --work-dir downloads/scenes \
    --start 2024-06-01 --end 2024-07-01 --target-date 2024-06-15
uv run ftwd planet-scenes activate --work-dir downloads/scenes
uv run ftwd planet-scenes extract --work-dir downloads/scenes --workers 4
uv run ftwd planet-scenes quality --work-dir downloads/scenes
```

Activation is optional; extraction activates assets when needed. Both
`PL_API_KEY` and `PLANET_API_KEY` are accepted. The account needs download
access to `ortho_analytic_4b_sr` and `ortho_udm2`; Planet permissions and quota
apply. Download URLs stay in memory rather than being saved in journals.

## Polygon and date options

GeoParquet geometry is read as WKB using its declared CRS and transformed to
WGS84 for search. The primary geometry column is detected automatically.

| Argument | Use |
| --- | --- |
| `--id-column` | Polygon identifier; defaults to `id` |
| `--geometry-column` | Override the primary geometry column |
| `--input-crs` | Override the input CRS; required if explicitly unknown |
| `--limit` | Process only the first N polygons for a small trial |
| `--date-column` | Per-polygon target date; defaults to fiboa's `determination_datetime` |
| `--target-date` | Override the target for all polygons |
| `--start`, `--end` | Override the search interval for all polygons |
| `--search-days` | Days on each side of the target when bounds are absent; default 30 |
| `--max-cloud` | Scene cloud fraction ceiling; default 0.1 |
| `--min-coverage` | Minimum fraction of polygon covered by the scene; default 0.99 |
| `--max-candidates` | Number of ranked alternatives to retain; default 10 |

A row's `target_date`, `start`, and `end` columns can also supply dates. CLI
values take precedence. Without a target, an explicit start/end pair uses its
midpoint. Intervals are `[start, end)`; naive dates mean UTC, and explicit
UTC offsets are preserved correctly. No seasonal windows or countries are
hardcoded. JSONL accepts the same attributes plus GeoJSON `geometry`:

```json
{"id":"parcel-1","geometry":{"type":"Polygon","coordinates":[[[-90.01,38.60],[-90.00,38.60],[-90.00,38.61],[-90.01,38.61],[-90.01,38.60]]]},"target_date":"2024-06-15"}
```

IDs must be unique and non-null. IDs containing filename-unsafe characters
receive stable hashed output names; `inputs.json` retains their `source_id`.
Invalid polygons are rejected. Split antimeridian-crossing polygons first.

## Outputs and recovery

Search follows all result pages, ranks candidates by date distance and cloud
fraction, and saves alternatives in `search/shard-*.jsonl`. Extraction groups
polygons by scene and opens each asset once for range reads. Outputs are
`images/ID_sr.tif` and `images/ID_udm2.tif`, cropped to the polygon's bounding
box on each asset's native grid. No label-grid reprojection is applied.

Rerun the same command after interruption. Failed/empty searches retry;
missing or corrupt SR and UDM2 crops are repaired independently. Activation
polling is bounded by `--activation-timeout` (3600 seconds by default). HTTP
429/5xx and transport errors retry with bounded backoff. Failed raster reads
retry once with a refreshed URL. Atomic writes preserve earlier files, and
input signatures prevent those files from being mistaken for current outputs.
Changed polygons, dates, or search settings require a new work directory.

Quality is written to `quality.jsonl`. Fractions use polygon pixel centers,
including blackfill in the denominator; confidence averages observed pixels.
UDM2's bit-encoded unusable band is handled separately from its class bands.
`usable` requires clear fraction >= `--min-clear` (default 0.95) and unusable
fraction <= its complement. UDM2.1's heavy-haze band remains present but is zero.
To replace a cloudy selection with a cached alternative:

```bash
uv run ftwd planet-scenes select parcel-1 --rank 1 --work-dir downloads/scenes
uv run ftwd planet-scenes extract --work-dir downloads/scenes
uv run ftwd planet-scenes quality --work-dir downloads/scenes
```

Ranks are zero-based. Both companions are replaced after selection changes.
Quality refuses to score a stale companion. Journals are append-only; read
current records with `state.latest(path)` or `state.collect(directory)`.
Failures return a nonzero exit code and record error types without signed URLs.
An empty search is recorded separately from a request failure.

## Parallel work

`--workers` controls local threads. Search, activation, and extraction also
accept `--shard-id` and `--num-shards`, for use with any job runner. Search
shards by polygon ID; activation/extraction shard by scene ID using stable
hashes. Finish every shard of a stage before starting the next. Keep one
writer per shard, retain the shard count when retrying, and do not change
selections during extraction. Quality runs in one process. Combined API load
is the number of active jobs times workers.

Offline tests run the CLI from projected GeoParquet through real raster crops
and UDM2 quality, with mocked Planet responses. Live account access, quota,
and throughput require a small authorized download before scaling up.

References: [Data API search](https://docs.planet.com/develop/apis/data/item-search/),
[activation](https://docs.planet.com/develop/apis/data/items/),
[UDM band definitions](https://docs.planet.com/data/imagery/udm/).
