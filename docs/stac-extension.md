# FTW STAC Extension

This document describes the FTW (Fields of The World) STAC extension properties used by ftw-dataset-tools for satellite imagery metadata.

## Overview

The FTW extension adds properties to STAC items that describe:
- Crop calendar timing (planting/harvest dates)
- Image selection parameters
- Source imagery provenance

## Extension Prefix

All FTW properties use the `ftw:` prefix.

## Properties

### Parent Chip Item Properties

These properties are added to parent chip items after image selection:

| Property | Type | Description |
|----------|------|-------------|
| `ftw:calendar_year` | integer | Calendar year for the crop cycle (e.g., 2024) |
| `ftw:planting_day` | integer | Day of year (1-365) for planting from crop calendar |
| `ftw:harvest_day` | integer | Day of year (1-365) for harvest from crop calendar |
| `ftw:stac_host` | string | Source STAC catalog used (always "earthsearch") |
| `ftw:cloud_cover_chip_threshold` | number | Chip-level cloud cover threshold percentage (0-100). Note: 2 means 2%, not 0.02 |
| `ftw:buffer_days` | integer | Search buffer in days around crop calendar dates |
| `ftw:num_buffer_expansions` | integer | Number of times to expand buffer if no cloud-free scenes found |
| `ftw:buffer_expansion_size` | integer | Days added to buffer on each expansion |
| `ftw:planting_buffer_used` | integer | Buffer in days that actually produced the planting scene |
| `ftw:harvest_buffer_used` | integer | Buffer in days that actually produced the harvest scene |
| `ftw:expansions_performed` | integer | Number of buffer expansions performed before a scene was found |
| `ftw:planting_cloud_cover` | number | Cloud cover percentage of the selected planting scene |
| `ftw:harvest_cloud_cover` | number | Cloud cover percentage of the selected harvest scene |

### Build-time Chip Item Properties

These properties are set when the chip items are generated from the chips GeoParquet,
independently of imagery:

| Property | Type | Description |
|----------|------|-------------|
| `ftw:split` | string | Train/val/test split assignment: "train", "val", or "test" |
| `ftw:field_coverage_pct` | number | Percentage of the chip covered by field polygons |
| `ftw:hcat_dominant_code` | integer | HCAT code with the largest area share in the chip |
| `ftw:hcat_dominant_name_en` | string | English name of the dominant HCAT code |
| `ftw:hcat_dominant_pct` | number | Area share (0-100) of the dominant HCAT code |
| `ftw:hcat_top` | array | Top HCAT codes by area, each `{code, name_en, pct}` |

The `ftw:hcat_*` properties are only present when the field polygons carry the fiboa
HCAT extension (an `hcat:code` column); otherwise crop composition is skipped and these
properties are omitted from the item. The name falls back to the `hcat:name` column when
`hcat:name_en` is absent. The percentages are shares of the chip's total field-covered
area, so they sum below 100 when some of the fields in the chip carry no HCAT code. This
is a different denominator from `ftw:field_coverage_pct`, which is a share of the chip's
own area: a chip that is 20% fields, all of them wheat, has
`ftw:field_coverage_pct: 20` and `ftw:hcat_dominant_pct: 100`.

### Child S2 Item Properties

These properties are added to child Sentinel-2 items (planting and harvest):

| Property | Type | Description |
|----------|------|-------------|
| `ftw:season` | string | Season identifier: "planting" or "harvest" |
| `ftw:source` | string | Image source identifier: "sentinel-2" |
| `ftw:calendar_year` | integer | Calendar year for the crop cycle |
| `ftw:cloud_cover_source` | string | Source of cloud cover value: "scene" or "pixel" |

Standard EO extension property:

| Property | Type | Description |
|----------|------|-------------|
| `eo:cloud_cover` | number | Cloud cover percentage of the source scene, rounded to 2 decimal places |

### Collection Properties

These are added to the chips collection (and, for `ftw:config`, to the root catalog too),
describing how the whole dataset was built:

| Property | Type | Description |
|----------|------|-------------|
| `ftw:split_type` | string | Split strategy used: `random-uniform` or `block3x3` |
| `ftw:split_seed` | integer | Random seed used for split assignment |
| `ftw:split_percents` | integer[3] | Train/val/test split percentages |
| `ftw:mask_types` | string[] | Mask types generated for the dataset |
| `ftw:mask_resolution_m` | number | Mask pixel resolution in meters |
| `ftw:presence_only` | boolean | Whether labels are presence-only (background class value is 3 instead of 0) |
| `ftw:min_coverage_pct` | number | Minimum field-coverage percentage required to keep a grid cell |
| `ftw:cloud_cover_chip_threshold` | number | Chip-level cloud cover threshold percentage (present only when image selection is enabled) |
| `ftw:nodata_max` | number | Maximum allowed nodata percentage (0-100) for a selected scene (present only when image selection is enabled) |
| `ftw:buffer_days` | integer | Search buffer in days around crop calendar dates (present only when image selection is enabled) |
| `ftw:num_buffer_expansions` | integer | Number of times to expand the buffer if no cloud-free scenes are found (present only when image selection is enabled) |
| `ftw:buffer_expansion_size` | integer | Days added to the buffer on each expansion (present only when image selection is enabled) |
| `ftw:config` | object | Resolved config provenance for the run that produced the dataset |

`license`, `providers`, `keywords`, `version` (from the
[version extension](https://github.com/stac-extensions/version)) and `updated` are not
`ftw:` properties; they come from the config's `metadata` block (see
`configs/examples/config.yaml`).

When the config sets `source_via`, the collection also gets a `via` link pointing at the
upstream collection:

```json
{"rel": "via", "href": "https://.../collection.json", "type": "application/json", "title": "Source field boundary collection"}
```

`ftw:config` (the resolved config, embedded verbatim) additionally carries two
provenance keys describing where the input fields file came from:

| Key | Type | Description |
|-----|------|-------------|
| `source` | object | `href` (the source URL, or just the filename for a local input), `via` (the `source_via` value, or null), `sha256` and `size` of the fetched/local file, `fetched_at` (ISO 8601 UTC timestamp, or null if served from cache or a local file) |
| `ftwd_git_commit` | string \| null | Git commit `ftwd` was installed from, or null if it wasn't installed from a git checkout |

## Item Structure

### Parent Chip Item

After image selection, a parent chip item contains:

```json
{
  "id": "ftw-34UFF1628_2024",
  "type": "Feature",
  "properties": {
    "start_datetime": "2024-03-15T10:30:00Z",
    "end_datetime": "2024-09-28T10:28:00Z",
    "ftw:calendar_year": 2024,
    "ftw:planting_day": 75,
    "ftw:harvest_day": 274,
    "ftw:stac_host": "earthsearch",
    "ftw:cloud_cover_chip_threshold": 2.0,
    "ftw:buffer_days": 14,
    "ftw:num_buffer_expansions": 3,
    "ftw:buffer_expansion_size": 14,
    "ftw:planting_cloud_cover": 0.42,
    "ftw:harvest_cloud_cover": 1.13
  },
  "links": [
    {"rel": "ftw:planting", "href": "./ftw-34UFF1628_2024_planting_s2.json"},
    {"rel": "ftw:harvest", "href": "./ftw-34UFF1628_2024_harvest_s2.json"}
  ]
}
```

`start_datetime` / `end_datetime` span the acquisition dates of the two selected
scenes. The `ftw:planting` and `ftw:harvest` links are what mark a chip as having
imagery: `ftwd select-images` skips chips that already have both (use `--force` to
re-select).

### Child S2 Item

Child items reference the source Sentinel-2 scene and contain remote asset links:

```json
{
  "id": "ftw-34UFF1628_2024_planting_s2",
  "type": "Feature",
  "datetime": "2024-03-15T10:30:00Z",
  "properties": {
    "ftw:season": "planting",
    "ftw:source": "sentinel-2",
    "ftw:calendar_year": 2024
  },
  "assets": {
    "red": {"href": "https://earth-search.aws.element84.com/.../B04.tif"},
    "green": {"href": "https://earth-search.aws.element84.com/.../B03.tif"},
    "blue": {"href": "https://earth-search.aws.element84.com/.../B02.tif"},
    "nir": {"href": "https://earth-search.aws.element84.com/.../B08.tif"},
    "clipped": {
      "href": "./ftw-34UFF1628_2024_planting_image_s2.tif",
      "type": "image/tiff; application=geotiff; profile=cloud-optimized",
      "title": "Clipped 4-band image (red,green,blue,nir)",
      "roles": ["data"]
    }
  },
  "links": [
    {"rel": "ftw:parent_chip", "href": "./ftw-34UFF1628_2024.json"},
    {"rel": "via", "href": "https://earth-search.aws.element84.com/.../S2A_....json"}
  ]
}
```

## Asset Metadata

Every asset that points at a file ftwd itself writes (masks, clipped imagery, thumbnails,
parquet outputs) carries `type`, at least one role, and `file:size`
([file extension](https://github.com/stac-extensions/file)). Assets that reference remote
source scenes (the Sentinel-2 band assets on child items) are carried through as provided
by the upstream catalog. `file:checksum` (multihash sha2-256) is added when
`stages.stac.checksums: true`, and only to the assets the `stac` stage writes: the label
masks on chip items and the parquet collection assets (`fields`, `boundary_lines`,
`chips`, `items`). Clipped imagery and thumbnails never carry one, because the
`select_images` and `download_images` stages add those assets after `stac` has run.
Checksums are off by default because they are slow on large datasets.

Raster assets (masks and clipped imagery) carry `raster:bands`
([raster extension](https://github.com/stac-extensions/raster)) with `data_type`,
`nodata` when set, `spatial_resolution`, and `statistics` (minimum, maximum, mean,
stddev, and valid_percent whenever a value was excluded from them). A presence-only
instance mask is the case where those come apart: its background is excluded from the
statistics, so `valid_percent` is present, while the band declares no `nodata` at all.
The same statistics are embedded in the
COG as GDAL `STATISTICS_*` band tags, never in an `.aux.xml` sidecar.

Semantic mask assets add `classification:classes`
([classification extension](https://github.com/stac-extensions/classification)):

| Mask | Classes |
|------|---------|
| `semantic_2class_mask` | 0 background, 1 field (background is 3 when `presence_only` is set) |
| `semantic_3class_mask` | 0 background, 1 field, 2 boundary |
| `instance_mask` | no class list; background (0, or 3 for presence-only) marks non-field pixels and is excluded from the band statistics — declared as the band's `nodata` only when it is 0 — other values are instance ids |
| | Field ids that are float-like (e.g. `'111205887.0'`) or otherwise non-numeric are coerced to integers, or replaced with sequential ids (1..n) for that chip when any id in it can't be coerced. |
| `decode_boundary_mask` | 0 background, 1 boundary |
| `decode_distance_mask` | no class list; float32 normalized distance in [0, 1], with a `decode_distance_max_px` dataset tag |

The `items` asset on the chips collection is the stac-geoparquet mirror of the items,
with media type `application/vnd.apache.parquet` and role `collection-mirror`.

Parquet assets (fields, boundary lines, chips, items) also carry `table:columns` (name,
type) and `table:row_count` from the
[table extension](https://github.com/stac-extensions/table).

The docs stage adds two more asset shapes to the collection when it runs (see
[Output Layout](#output-layout) above):

| Asset key | Media type | Roles | Description |
|-----------|------------|-------|--------------|
| `chips_tiles` | `application/vnd.pmtiles` | `visual` | Vector tiles of the chips (`chips.pmtiles`) |
| `fields_tiles` | `application/vnd.pmtiles` | `visual` | Vector tiles of the fields (`fields.pmtiles`) |
| `style-<id>` | `application/vnd.mapbox.style+json` | `style`, plus `default` on the first style written | A MapLibre GL style JSON under `styles/`, e.g. `style-split` for `styles/split.json` |

`chips_tiles` and `fields_tiles` also carry `file:size`. Re-running the docs stage
replaces these assets and the `describedby`/`agents` links in place (matched by key, or
by `rel` + `href`) rather than duplicating them, and drops any tile, style or document
entry a previous run wrote that this run did not produce.

## Rendering

A chip item's own COGs are label masks and a 16-bit four-band scene stack, so a browser
that picks one to draw unaided shows a near-black square. Three pieces of metadata make a
chip item render meaningfully.

**Visual season assets.** Once imagery has been selected, the parent chip item carries
`planting_visual` and `harvest_visual`: the selected Sentinel-2 scene's true-colour COG
(Earth Search's `visual` asset), by absolute `https` href, with role `visual`, plus
`ftw:scene` (the source scene id) and the scene's `datetime`. They are the natural default
for a client that scores candidate assets by role. The downloaded, chip-clipped
`planting_image` / `harvest_image` assets keep role `data` and are unchanged.

**Colour hints.** Colours for the categorical masks come from
`classification:classes[].color_hint` (6-digit hex, no `#`) — this is the primary
rendering mechanism for those rasters. Field interiors are `009E73` and boundaries
`D55E00` (Okabe-Ito, colour-blind safe); background carries no hint, because it is meant
to be transparent rather than coloured.

**Renders.** Items carry `renders` under `properties` and the collection carries it at the
top level, as the [render extension](https://github.com/stac-extensions/render) v2.0.0
schema requires for each type. The categorical masks get an entry with only `assets`,
`title` and `nodata`, so a viewer that ignores `classification:classes` still hides the
background — deliberately no `colormap`, so the class hints stay the single source of
colour. Only the continuous rasters get a ramp, and a chip that has season imagery also
gets a true-colour render per season:

| Render | Assets | Definition |
|--------|--------|------------|
| `semantic_2class` | `semantic_2class_mask` | `nodata`: the background value |
| `semantic_3class` | `semantic_3class_mask` | `nodata`: the background value |
| `decode_boundary` | `decode_boundary_mask` | `nodata: 0` |
| `decode_distance` | `decode_distance_mask` | `rescale: [[0, 1]]`, `nodata` from the band, else 0 |
| `instance` | `instance_mask` | `rescale: [[band minimum, band maximum]]`, `nodata`: the background value, `colormap_name: viridis` |
| `planting_rgb` | `planting_image`, else `planting_visual` | `bidx` + `rescale` — see below |
| `harvest_rgb` | `harvest_image`, else `harvest_visual` | `bidx` + `rescale` — see below |

`nodata` is the dataset's background pixel value — 0 normally, **3** when
`presence_only` is set — for the three class-valued masks (`semantic_2class`,
`semantic_3class`, `instance`). The DECODE layers always fold their background into 0, so
they always use 0.

Instance ids are global rather than per-chip, so the instance mask's background is excluded
from its embedded statistics: the render then stretches from the chip's smallest field id
to its largest instead of from zero, and a chip whose ids all sit in the hundreds of
thousands still shows its fields apart. Where those statistics are missing or degenerate
the render falls back to `[[0, 1]]`.

Two caveats on that:

- The instance band declares `nodata` **only when the background is 0**. A presence-only
  background of 3 is also a legal instance id (ids are renumbered `1..n` for a chip whose
  raw ids cannot be coerced), so declaring it would make a real field vanish for any
  nodata-respecting reader. Presence-only instance masks therefore exclude 3 from their
  statistics but leave the band's `nodata` unset.
- Masks written before this behaviour existed carry statistics that include the background
  and so still yield a `[[0, maximum]]` stretch. Re-run the `masks` stage (then `stac`) to
  pick up the tighter range.

Item mask renders are keyed by mask kind and only cover the masks that chip actually has;
the collection mirrors the same definitions keyed by asset name, as a default for clients
that read the collection first. The collection's instance render carries no `rescale` at
all — there is no meaningful collection-wide id range — so use the item's. The season
renders below are item-level only: they depend on per-chip imagery and statistics, so the
collection has no equivalent.

### Season imagery renders

`planting_rgb` and `harvest_rgb` are true colour, titled `"<Season> season (true colour)"`.
One is emitted per season the chip has imagery for, drawn from whichever asset that season
has:

- **`<season>_image`** — the chip-clipped GeoTIFF, when it is on disk. `bidx` selects the
  red, green and blue bands **by name**, read from the band descriptions the download
  stage writes into `raster:bands[].description`; a file written without any descriptions
  falls back to its first three bands. A stack whose named bands do not include all of
  red, green and blue (`--bands nir,red,green`, say) is *never* guessed at — the season
  falls back to its `visual` asset instead of publishing false colour under a true-colour
  title. `rescale` is per band, taken from that band's own embedded statistics and clipped
  to `mean ± 2σ` within the band's extremes, so one bright cloud edge does not crush the
  image; a band with no mean/stddev, or whose clipped range collapses, uses its raw
  minimum/maximum. `nodata` is published only when all three colour bands declare the same
  fill (0 for an all-reflectance stack; a stack sharing `scl`/`cloud`/`aot` declares none,
  because 0 is a real measurement there).
- **`<season>_visual`** — the full scene's true-colour COG, used when the chip has a
  selection but no local download. It is 8-bit, so its stretch is fixed:
  `bidx: [1, 2, 3]`, `rescale: [[0, 255], [0, 255], [0, 255]]`, `nodata: 0`. A COG reader
  only fetches the tiles the chip covers.

**`portolan:render_order`.** An item that has season imagery also carries this property: an
array of keys **into that item's own `renders` object, bottom first**, naming the layer
stack a viewer should open the chip on. It is always the season's true-colour imagery with
the field labels drawn over it:

```json
"portolan:render_order": ["planting_rgb", "decode_boundary"]
```

The base layer is `planting_rgb` when the chip has planting imagery, else `harvest_rgb`.
The overlay is `decode_boundary` when the chip has a DECODE boundary mask — an outline, so
the imagery stays visible inside each field — and `semantic_2class` otherwise; a chip with
neither carries the base layer alone. Every key names a render of the same item, and the
overlay's `nodata` makes its background transparent so the imagery shows through.

The property is **absent** on a chip with no season imagery of any kind: a mask-only stack
would show nothing a viewer's single-asset default does not already show.

This is a browser-side Portolan convention, pending standardisation in portolan-spec issue
#41, and it costs nothing to a reader that does not know it: a client that ignores the property keeps
its own default asset choice, a key naming no render is skipped, and a render whose assets
cannot be drawn contributes nothing.

## Output Layout

The output directory is a self-contained STAC collection. `collection.json` sits at the root, and chip items are organized into sub-catalogs by MGRS 100 km square to keep directory sizes manageable:

```
{name}/
├── collection.json                   # STAC collection root
├── {name}_fields.parquet             # Field boundaries in EPSG:4326
├── {name}_fields_filtered.parquet    # Filtered fields (if class filter applied)
├── {name}_chips.parquet              # Chip definitions with field coverage stats
├── {name}_boundary_lines.parquet     # Boundary lines from vector data
├── items.parquet                     # Collection mirror (STAC items as Parquet; only if any chip has masks)
├── README.md                         # What the collection contains (docs stage)
├── AGENTS.md                         # Schema, quality notes, executed example queries (docs stage)
├── chips.pmtiles                     # Chip vector tiles (docs stage; only when tippecanoe ran)
├── fields.pmtiles                    # Field vector tiles (docs stage; only when tippecanoe ran)
├── styles/                           # MapLibre GL styles for the PMTiles (docs stage)
│   └── {style_id}.json               # e.g. split.json, field-coverage.json, dominant-crop.json, crops.json, outline.json
└── chips/
    ├── {mgrs100k}/
    │   ├── catalog.json              # Sub-catalog for MGRS 100 km square
    │   └── {item_id}/                # Item JSON and its assets, flat, side by side
    │       ├── {item_id}.json                   # Chip item
    │       ├── {item_id}_{mask_type}.tif        # Masks (if masks generated)
    │       ├── {item_id}_{season}_s2.json       # Scene items (if imagery selected)
    │       ├── {item_id}_{season}_image_s2.tif  # Clipped imagery (if downloaded)
    │       └── {item_id}_{season}_image_s2.jpg  # Thumbnails (if downloaded)
    └── other/                        # For non-FTW grid ids
        ├── catalog.json
        └── {item_id}/...
```

**Docs stage outputs** (the final `docs` stage, run by both `ftwd run` and
`ftwd create-dataset`): `README.md` and `AGENTS.md`
are generated from the collection's measured contents and linked from `collection.json`
via `describedby` and `agents` links, respectively. `chips.pmtiles` / `fields.pmtiles` and
the styles under `styles/` are only written when the `tippecanoe` binary is available (or
`stages.docs.pmtiles` is not set to `false`); each style is registered as a `style-<id>`
asset on the collection. Every STAC object this tool writes — the collection, every
sub-catalog and every item — declares the
[Portolan schema](https://schemas.portolan-sdi.org/portolan/v0.1.2/schema.json) URI in
`stac_extensions`.

**MGRS square rule:** The sub-catalog id is the MGRS 100 km square extracted from FTW grid ids (e.g., `33UXP` from `ftw-33UXP0410`). Grid ids that don't match the FTW naming convention are placed under `other`.

**Items parquet:** The `items.parquet` asset with role `collection-mirror` exists only if at least one chip item has masks. It is a geoparquet mirror of all chip items in the collection.

**Item assets:** The collection's `item_assets` declares the possible assets on chip items: mask types (`instance_mask`, `semantic_2class_mask`, `semantic_3class_mask`, `decode_boundary_mask`, `decode_distance_mask`), imagery (`planting_image`, `harvest_image`), and `thumbnail`.

**Collection reference:** Every chip item carries a `collection` field and link pointing to the dataset collection (the one holding `collection.json`). The collection's `root` link points to itself (downstream Portolan catalogs rewrite `root` when ingesting the output).

## Link Relations

| Relation | Description |
|----------|-------------|
| `ftw:planting` | Links a parent chip item to its planting-season S2 child item |
| `ftw:harvest` | Links a parent chip item to its harvest-season S2 child item |
| `ftw:parent_chip` | Links a child S2 item back to its parent chip item |
| `via` | Links to the original source STAC item in the remote catalog |

## Asset Roles

| Asset Key | Description |
|-----------|-------------|
| `red` | Red band (B04) |
| `green` | Green band (B03) |
| `blue` | Blue band (B02) |
| `nir` | Near-infrared band (B08) |
| `scl` | Scene Classification Layer |
| `cloud_probability` | Cloud probability mask |
| `clipped` | Local clipped multi-band image (after download) |
| `planting_visual` / `harvest_visual` | The season's source scene as a true-colour COG (role `visual`, remote href) |

## File Naming Convention

All files include the calendar year for consistency. Chip item files and their assets live in `chips/{mgrs100k}/{item_id}/`:

- Parent item: `chips/{mgrs100k}/{item_id}/{item_id}.json`
- Child S2 items: `chips/{mgrs100k}/{item_id}/{item_id}_{season}_s2.json`
- Clipped imagery: `chips/{mgrs100k}/{item_id}/{item_id}_{season}_image_s2.tif`
- Mask files: `chips/{mgrs100k}/{item_id}/{item_id}_{mask_type}.tif`

where `{mgrs100k}` is the MGRS 100 km square (or `other` for custom grids) and `{item_id}` is the chip identifier.

## Future Work

A formal STAC extension schema will be published at:
- Repository: `fieldsoftheworld/ftw-stac-extension`
- Based on: `stac-extensions/template`

The extension will include:
- JSON Schema definitions
- Validation examples
- Best practices documentation
