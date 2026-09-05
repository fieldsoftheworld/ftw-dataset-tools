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
| `ftw:nodata_max` | number | Maximum allowed nodata fraction for a selected scene (present only when image selection is enabled) |
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
| `source` | object | `href` (URL or absolute local path), `via` (the `source_via` value, or null), `sha256` and `size` of the fetched/local file, `fetched_at` (ISO 8601 UTC timestamp, or null if served from cache or a local file) |
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
`stages.stac.checksums: true`; it is off by default because it is slow on large datasets.

Raster assets (masks and clipped imagery) carry `raster:bands`
([raster extension](https://github.com/stac-extensions/raster)) with `data_type`,
`nodata` when set, `spatial_resolution`, and `statistics` (minimum, maximum, mean,
stddev, and valid_percent when nodata is set). The same statistics are embedded in the
COG as GDAL `STATISTICS_*` band tags, never in an `.aux.xml` sidecar.

Semantic mask assets add `classification:classes`
([classification extension](https://github.com/stac-extensions/classification)):

| Mask | Classes |
|------|---------|
| `semantic_2class_mask` | 0 background, 1 field (background is 3 when `presence_only` is set) |
| `semantic_3class_mask` | 0 background, 1 field, 2 boundary |
| `instance_mask` | no class list; background (0, or 3 for presence-only) marks non-field pixels, other values are instance ids |
| `decode_boundary_mask` | 0 background, 1 boundary |
| `decode_distance_mask` | no class list; float32 normalized distance in [0, 1], with a `decode_distance_max_px` dataset tag |

The `items` asset on the chips collection is the stac-geoparquet mirror of the items,
with media type `application/vnd.apache.parquet` and role `collection-mirror`.

Parquet assets (fields, boundary lines, chips, items) also carry `table:columns` (name,
type) and `table:row_count` from the
[table extension](https://github.com/stac-extensions/table).

## Output Layout

The output directory is a self-contained STAC collection. `collection.json` sits at the root, and chip items are organized into sub-catalogs by MGRS 100 km square to keep directory sizes manageable:

```
{name}/
├── collection.json                   # STAC collection root
├── {name}_fields.parquet             # Field boundaries in EPSG:4326
├── {name}_fields_filtered.parquet    # Filtered fields (if class filter applied)
├── {name}_boundary_lines.parquet     # Boundary lines from vector data
├── items.parquet                     # Collection mirror (STAC items as Parquet; only if any chip has masks)
└── chips/
    ├── {mgrs100k}/
    │   ├── catalog.json              # Sub-catalog for MGRS 100 km square
    │   └── {item_id}/
    │       ├── {item_id}.json        # Chip item
    │       ├── masks/                # Mask files (if masks generated)
    │       ├── imagery/              # Clipped imagery (if selected/downloaded)
    │       └── ...
    └── other/                         # For non-FTW grid ids
        ├── catalog.json
        └── {item_id}/...
```

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
