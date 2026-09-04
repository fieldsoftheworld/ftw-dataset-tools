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

Every asset written by ftwd carries `type`, at least one role, and `file:size`
([file extension](https://github.com/stac-extensions/file)). `file:checksum` (multihash
sha2-256) is added when `stages.stac.checksums: true`; it is off by default because it is
slow on large datasets.

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
| `instance_mask` | no class list; 0 is background, other values are per-field instance ids |
| `decode_boundary_mask` | 0 background, 1 boundary |
| `decode_distance_mask` | no class list; float32 normalized distance in [0, 1], with a `decode_distance_max_px` dataset tag |

The `items` asset on the chips collection is the stac-geoparquet mirror of the items,
with media type `application/vnd.apache.parquet` and role `collection-mirror`.

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

All files include the calendar year for consistency:

- Parent item: `{grid_id}_{year}.json`
- Child S2 items: `{grid_id}_{year}_{season}_s2.json`
- Clipped imagery: `{grid_id}_{year}_{season}_image_s2.tif`
- Mask files: `{grid_id}_{year}_{mask_type}.tif`

## Future Work

A formal STAC extension schema will be published at:
- Repository: `fieldsoftheworld/ftw-stac-extension`
- Based on: `stac-extensions/template`

The extension will include:
- JSON Schema definitions
- Validation examples
- Best practices documentation
