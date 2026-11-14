# Planet Basemaps downloads

`ftwd planet-basemaps` downloads mosaic quads and creates ordered seasonal
stacks for polygons from fiboa/other GeoParquet files or JSONL. The reusable
Python stages are in `ftw_dataset_tools.api.planet.basemap_workflow`.

## Run

```bash
uv sync --locked --dev
export PL_API_KEY='your Planet API key'

uv run ftwd planet-basemaps plan fields.parquet --work-dir downloads/basemaps \
    --mosaic global_quarterly_2024q1_mosaic \
    --mosaic global_quarterly_2024q2_mosaic \
    --mosaic global_quarterly_2024q3_mosaic \
    --mosaic global_quarterly_2024q4_mosaic
uv run ftwd planet-basemaps download --work-dir downloads/basemaps --workers 4
uv run ftwd planet-basemaps assemble --work-dir downloads/basemaps --workers 4
```

`PLANET_API_KEY` is also accepted. Your account must have quad-download
access to the requested mosaics. Catalog visibility does not establish polygon
coverage. Planning checks actual quads for every polygon and requested period.

## Input and options

GeoParquet geometry is read as WKB using its declared CRS and transformed to
WGS84 for planning. The primary geometry column is detected automatically.

| Argument | Use |
| --- | --- |
| `--id-column` | Polygon identifier; defaults to `id` |
| `--geometry-column` | Override the primary geometry column |
| `--input-crs` | Override the input CRS; required if explicitly unknown |
| `--limit` | Process only the first N polygons for a small trial |
| `--mosaic` | Exact mosaic name; repeat in output order |
| `--band` | Source band index at assembly; repeat in output order, default 1, 2, 3 |
| `--workers` | Local concurrent tasks; default 4 |

Mosaic names can instead come from an ordered `mosaics` list per input row,
allowing different years or products for different polygons. CLI mosaic names
override row lists. Names must match exactly; no alternate period or product
is substituted. JSONL accepts GeoJSON `geometry` and the same attributes:

```json
{"id":"parcel-1","geometry":{"type":"Polygon","coordinates":[[[-90.01,38.60],[-90.00,38.60],[-90.00,38.61],[-90.01,38.61],[-90.01,38.60]]]},"mosaics":["global_quarterly_2024q1_mosaic"]}
```

IDs must be unique and non-null. Filename-unsafe IDs receive stable hashed
output names, with original `source_id` values retained in `inputs.json`.
Invalid polygons are rejected; split antimeridian-crossing polygons first.
No country lists, fixed years, or FTW-specific polygon layouts are required.

## Outputs

Plans are saved in `plan/shard-*.jsonl`; unique quads are cached under
`quads/MOSAIC_ID/QUAD_ID.tif`. Assembly writes `images/ID.tif`, using the first
quad's native pixel grid. All requested periods must share CRS, pixel
resolution, alignment, and data type. Incompatible grids fail explicitly.
Web Mercator pixel size is in projected units, not constant ground resolution.

The band order is requested mosaic order followed by requested source band
order. Four quarters with bands 1–3 produce twelve bands. GeoTIFF descriptions
identify each mosaic/band; tags retain mosaic IDs, provenance, and coverage.
Choose source bands appropriate to the product; RGB visual and analytic
products have different band definitions.

The output internal mask marks pixels valid in every requested period and
inside the polygon. Valid zero-valued colors are preserved. Read using
`source.read(masked=True)` or `source.dataset_mask()`. The assembly journal
reports each period's `valid_fractions` and the `joint_valid_fraction`.
Inspect these before use. An empty period or missing quad fails assembly;
there is no unmarked zero-filled seasonal placeholder.

## Resume and parallel work

Rerun a stage with the same arguments after interruption. Failed/empty plans
retry; valid cached quads are reused. Corrupt quads are redownloaded through a
fresh authenticated URL. Downloads and assembly validate temporary rasters
before atomic replacement, preserving existing files on failure. Input
signatures prevent stale stacks from being reused. Changed input polygons or
mosaic order require a new work directory.

Request retries cover HTTP 429/5xx and transport errors with bounded backoff;
401/403/404 fail immediately. Failed jobs return nonzero and record their IDs
and error types. Signed URLs are not written into plans or journals. Journals
are append-only: use `state.latest(path)` or `state.collect(directory)` for
current results. Retain inputs, plans, and assembly records with your outputs.

Every stage accepts `--shard-id` and `--num-shards` for any job runner. Planning
shards by polygon/mosaic, downloading by unique mosaic/quad, and assembly by
polygon. Stable hashes prevent Python hash randomization from changing task
ownership. Finish all shards before the next stage, keep one writer per shard,
and retain shard counts for retries. Total API load is active jobs times
workers; reduce concurrency when rate limited.

Offline tests cover the CLI from projected GeoParquet through adjacent-quad
crops, seasonal ordering, alpha masks, and cache recovery with mocked HTTP.
Smoke-test a small authorized area to establish live access and coverage.

References: [Basemaps overview](https://docs.planet.com/develop/apis/basemaps/),
[API reference](https://docs.planet.com/develop/apis/basemaps/reference/).
