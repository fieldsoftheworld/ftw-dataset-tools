# Planet Basemaps downloads

`ftwd planet-basemaps` plans available mosaic quads for AOIs, downloads each
unique quad into a shared cache, and assembles one ordered seasonal stack per
AOI. Stages can run locally or in SLURM arrays. The reusable Python functions
are in `ftw_dataset_tools.api.planet.basemap_workflow`, with API and raster
operations in separate modules.

## Install and authenticate

```bash
uv sync --locked --dev
export PL_API_KEY='your Planet API key'
uv run ftwd planet-basemaps --help
```

`PLANET_API_KEY` is also accepted. Supply the key through your shell or
cluster secret mechanism. The account must have quad-download access to each
requested mosaic. Product availability and quota depend on that account;
seeing a mosaic in the catalog does not establish coverage of your AOIs.

## Input and mosaic selection

Use JSONL containing unique IDs and WGS84 Polygon/MultiPolygon geometry:

```json
{"id":"example","geometry":{"type":"Polygon","coordinates":[[[-90.01,38.60],[-90.00,38.60],[-90.00,38.61],[-90.01,38.61],[-90.01,38.60]]]}}
```

IDs start with a letter/digit and contain only letters, digits, `.`, `_`, and
`-`. Split antimeridian-crossing AOIs before use. If AOIs need different years
or products, add a `mosaics` list to each row:

```json
{"id":"example","geometry":{"type":"Polygon","coordinates":[[[-90.01,38.60],[-90.00,38.60],[-90.00,38.61],[-90.01,38.61],[-90.01,38.60]]]},"mosaics":["global_quarterly_2024q1_mosaic","global_quarterly_2024q2_mosaic"]}
```

Alternatively, repeat `--mosaic` at plan time to set the ordered list for every
AOI. CLI names override per-row lists. Names must match exactly: no fallback to
another month, quarter, or product. Use the monthly or quarterly names exposed
by your own Planet catalog; these examples do not establish entitlement.

The planner queries actual quad availability for **every AOI and mosaic**,
following pagination. It does not infer availability from a mosaic's broad
bounding box or reuse a reference quarter's quad list for other periods.
Plans retain identifiers and AOI geometry, never signed download links.

## Local workflow

```bash
uv run ftwd planet-basemaps plan aois.jsonl --work-dir runs/basemaps \
    --mosaic global_quarterly_2024q1_mosaic \
    --mosaic global_quarterly_2024q2_mosaic \
    --mosaic global_quarterly_2024q3_mosaic \
    --mosaic global_quarterly_2024q4_mosaic
uv run ftwd planet-basemaps download --work-dir runs/basemaps --workers 4
uv run ftwd planet-basemaps assemble --work-dir runs/basemaps --workers 4
```

Assembly defaults to source bands 1, 2, 3, suitable for RGB visual mosaics.
Choose bands explicitly for other products, e.g. `--band 1 --band 2 --band 3
--band 4`. The source band order is preserved, not interpreted as a universal
RGB/NIR order. Verify the selected product's band definitions.

Four RGB quarters produce twelve bands: Q1 bands 1–3, Q2 bands 1–3, Q3 bands
1–3, Q4 bands 1–3. The order is exactly the requested mosaic order, even if it
is not chronological. Output band descriptions identify mosaic name and source
band. GeoTIFF tags retain mosaic names, IDs, input signature, and each period's
fraction of valid AOI pixels.

## Native grids and missing data

Assembly crops on the first quad's native pixel grid, without reprojection or
resampling. Every quad and period must share CRS, resolution, pixel alignment,
and data type. Incompatible products fail explicitly; assemble them separately.
Web Mercator pixel size is in projected units, not constant ground resolution.
These outputs do not automatically align to labels or change existing STAC
items, and no labels or training/inference artifacts are generated.

Only the required windows are read from cached quads. Alpha/nodata masks
determine coverage, preserving valid zero-valued colors. The output's internal
mask marks pixels valid in **all** requested periods and inside the AOI;
`nodata` is unset because zero may be a real color. Read with
`rasterio.open(path).read(masked=True)` or inspect `dataset_mask()`.

No available quads is recorded as `empty` at planning time. Assembly fails
for an empty period or missing/unreadable quad; it never silently substitutes a
different period or adds an unmarked placeholder. Partial coverage within an
available quad is represented by its mask and reported as `valid_fractions`
and `joint_valid_fraction` in the assembly journal. Inspect those fractions
before using a stack. AOIs/stacks above the memory guard must be split.

## Resume and handoff artifacts

```text
runs/basemaps/
  inputs.json                    # frozen AOIs and ordered mosaic names
  plan/config.json               # frozen planning shard count
  plan/shard-00000.jsonl          # actual mosaic/quad availability per AOI
  download/shard-00000.jsonl      # unique-quad download outcomes
  assemble/shard-00000.jsonl      # output coverage and failures
  quads/MOSAIC_ID/QUAD_ID.tif     # shared validated cache
  images/AOI_ID.tif              # ordered seasonal stack
```

Rerun the same stage after interruption. Successful plans are reused; failed
and empty plans retry. A change to manifest or mosaic order requires a new
work directory. Failed jobs produce a nonzero exit code. Downloads retry
429/5xx and transport errors with bounded backoff; 401/403/404 fail immediately.
Use the journal's job ID and error class to identify a failing task without
logging a signed URL or credentials.

The cache is keyed by `(mosaic_id, quad_id)` across all AOIs. Downloads follow
the authenticated `/full` endpoint to obtain a fresh link, stream into unique
temporary files, read back the raster, and atomically rename it. Corrupt cache
files are redownloaded. Atomic replacement preserves an earlier valid file on
failure. Assembly similarly validates provenance and pixels before skipping a
stack. Retain the input, plan, and assembly records with outputs for colleagues.

One process owns each stage/shard journal. Run stages sequentially and keep the
shard count fixed for retries. Do not launch duplicate copies of a shard.
Worker counts may change. Journals are append-only: use
`ftw_dataset_tools.api.planet.state.latest(path)` or `collect(directory)` to
read the last record per ID, including failures following earlier successes.
Only a torn final journal line is recoverable; corrupt input or interior
records fail explicitly.

## SLURM

Install the environment once on a shared filesystem; pass your account and
partition to `sbatch` as appropriate. Put mosaic lists in the manifest, then
submit from the repository root:

```bash
mkdir -p logs
export MANIFEST="$PWD/aois.jsonl" WORK_DIR="$PWD/runs/basemaps" NUM_SHARDS=8 WORKERS=4
plan_job=$(sbatch --parsable --array=0-7 examples/planet-basemaps/stage.sbatch plan)
download_job=$(sbatch --parsable --array=0-7 --dependency=afterok:$plan_job examples/planet-basemaps/stage.sbatch download)
sbatch --array=0-7 --dependency=afterok:$download_job examples/planet-basemaps/stage.sbatch assemble
```

Planning shards by AOI/mosaic, download by unique mosaic/quad, and assembly by
AOI. All use stable SHA-256 assignments. Array concurrency times workers sets
the account's combined load; start small, optionally use `--array=0-7%2`, and
reduce concurrency when rate limited. `FTWD` overrides the template's default
`$PWD/.venv/bin/ftwd`. No cluster-specific account, paths, or modules are assumed.

## Validation and references

Offline tests exercise the real CLI, paginated metadata, shared downloads,
adjacent-quad crops, seasonal band order, alpha handling, corrupt-cache repair,
missing periods, incompatible grids, and sharding. They do not establish
live account coverage, quota, or cluster performance. Smoke-test one authorized
AOI, inspect the stack and validity mask, then scale up.

See the [Basemaps API overview](https://docs.planet.com/develop/apis/basemaps/)
and [API reference](https://docs.planet.com/develop/apis/basemaps/reference/).
