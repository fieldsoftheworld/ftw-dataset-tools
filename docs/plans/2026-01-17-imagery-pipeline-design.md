# Imagery Pipeline Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend ftwd to create complete ML training datasets including satellite imagery, not just masks. Two new operations for selecting and downloading Sentinel-2 imagery based on crop calendar dates.

**Tech Stack:** Python, DuckDB, PySTAC, Rasterio, Click, odc-stac, pystac-client

---

## Overview

Two new CLI commands plus integration into existing workflow:

1. **`select-images`** - Query STAC catalogs to find optimal cloud-free Sentinel-2 scenes for each chip based on crop calendar dates, create child STAC items with remote asset links
2. **`download-images`** - Download and clip the selected imagery, add local "clipped" asset to child STAC items
3. **`refine-cloud-cover`** - Optional pixel-level cloud check on already-selected images

---

## File Naming & Directory Structure

All files include year for consistency:

```
{dataset}-chips/
├── collection.json
├── items.parquet
├── ftw-34UFF1628_2024/
│   ├── ftw-34UFF1628_2024.json                      # Parent chip item
│   ├── ftw-34UFF1628_2024_planting_s2.json          # Planting S2 child item
│   ├── ftw-34UFF1628_2024_harvest_s2.json           # Harvest S2 child item
│   ├── ftw-34UFF1628_2024_planting_image_s2.tif     # Planting imagery
│   ├── ftw-34UFF1628_2024_harvest_image_s2.tif      # Harvest imagery
│   ├── ftw-34UFF1628_2024_instance.tif              # Masks (now with year)
│   ├── ftw-34UFF1628_2024_semantic_2_class.tif
│   └── ftw-34UFF1628_2024_semantic_3_class.tif
```

**STAC item IDs match filenames** (without .json extension).

---

## STAC Structure & Workflow

### After `select-images`

Child items created with remote asset links:

```json
// ftw-34UFF1628_2024_planting_s2.json
{
  "id": "ftw-34UFF1628_2024_planting_s2",
  "bbox": [/* from chip */],
  "datetime": "2024-03-15T10:30:00Z",
  "properties": {
    "ftw:season": "planting",
    "ftw:source": "sentinel-2",
    "ftw:calendar_year": 2024
  },
  "assets": {
    "red": {"href": "https://earth-search.aws.../B04.tif"},
    "green": {"href": "https://earth-search.aws.../B03.tif"},
    "blue": {"href": "https://earth-search.aws.../B02.tif"},
    "nir": {"href": "https://earth-search.aws.../B08.tif"},
    "cloud_probability": {"href": "https://earth-search.aws.../..."}
  },
  "links": [
    {"rel": "derived_from", "href": "./ftw-34UFF1628_2024.json"},
    {"rel": "via", "href": "https://earth-search.aws.../S2A_....json"}
  ]
}
```

Note: `eo:cloud_cover` only populated if scene-level is 0.

### After `download-images`

Adds local clipped asset:

```json
{
  "assets": {
    "red": {"href": "https://..."},
    "green": {"href": "https://..."},
    "blue": {"href": "https://..."},
    "nir": {"href": "https://..."},
    "cloud_probability": {"href": "https://..."},
    "clipped": {
      "href": "./ftw-34UFF1628_2024_planting_image_s2.tif",
      "type": "image/tiff; application=geotiff; profile=cloud-optimized",
      "title": "Clipped 4-band image (R,G,B,NIR)",
      "roles": ["data"]
    }
  }
}
```

### Parent Item Properties

After selection, parent chip item updated with:

```json
{
  "properties": {
    "start_datetime": "2024-03-15T...",
    "end_datetime": "2024-09-30T...",
    "ftw:planting_day": 92,
    "ftw:harvest_day": 274,
    "ftw:calendar_year": 2024,
    "ftw:stac_host": "earthsearch",
    "ftw:cloud_cover_scene_threshold": 10,
    "ftw:buffer_days": 14,
    "ftw:pixel_check": false
  }
}
```

---

## CLI Commands

### `ftwd select-images`

```
ftwd select-images <catalog-path> \
  --year 2024 \
  --stac-host earthsearch \
  --cloud-cover-scene 10 \
  --pixel-check \
  --cloud-cover-pixel 0 \
  --buffer-days 14 \
  --include-alternatives \
  --on-missing skip|fail|best-available \
  --resume \
  --output-report skipped.json
```

- Default: scene-level cloud cover only (no pixel check)
- `--pixel-check`: Enable pixel-level cloud filtering
- `--cloud-cover-pixel 0`: Threshold when pixel-check is enabled (default 0%)
- When `--pixel-check` is on: hybrid approach (scene <50% first filter, then pixel check, skip pixel if scene <0.1%)

### `ftwd refine-cloud-cover`

```
ftwd refine-cloud-cover <catalog-path> \
  --cloud-cover-pixel 0 \
  --update-items \
  --output-report cloud-report.json
```

- Reads cloud mask COGs for actual chip area
- Reports actual cloud coverage per chip
- With `--update-items`: populates `eo:cloud_cover` and optionally re-selects if above threshold

### `ftwd download-images`

```
ftwd download-images <catalog-path> \
  --bands red,green,blue,nir \
  --resume \
  --num-workers 4 \
  --output-report download-report.json
```

### `ftwd create-dataset` (updated)

```
ftwd create-dataset <input-file> <output-dir> \
  --name my-dataset \
  --year 2024 \
  # ... existing flags ...
  --select-images \
  --download-images \
  --cloud-cover-scene 10 \
  --include-alternatives
```

When `--select-images` is passed, runs image selection after mask creation.
When `--download-images` is passed (implies `--select-images`), also downloads the imagery.

---

## Cloud Filtering & Crop Calendar Logic

### Scene Selection Flow

1. Look up crop calendar for chip bbox -> get planting day and harvest day
2. Convert to dates for the given year (handling southern hemisphere year wraparound)
3. Query STAC for scenes within +/-buffer_days of each date
4. Filter by scene-level cloud cover (default <10%)
5. If `--pixel-check` enabled:
   - Skip scenes with scene-level >=50%
   - Skip pixel check if scene-level <0.1%
   - Otherwise read cloud mask COG for chip bbox, calculate actual coverage
   - Filter by pixel-level threshold (default 0%)
6. Select scene with lowest cloud cover (or closest to target date as tiebreaker)
7. If no scene found: handle per `--on-missing` setting

### Crop Calendar Files

Downloaded on first use to `~/.cache/ftw-tools/crop_calendar/`:

- Summer crop start: `sc-sos-3x3-v2-cog.tiff`
- Summer crop end: `sc-eos-3x3-v2-cog.tiff`

---

## Error Handling

**`--on-missing` options:**
- `skip` (default): Don't create child items, log warning, continue
- `fail`: Stop and report error
- `best-available`: Relax threshold, use whatever is available

**Summary report generated at end:**

```json
{
  "total_processed": 1500,
  "successful": 1485,
  "skipped": [
    {"chip": "ftw-34UFF1628_2024", "reason": "no cloud-free planting scene", "candidates_checked": 12}
  ],
  "failed": [
    {"chip": "ftw-34UFF9999_2024", "error": "STAC API timeout"}
  ]
}
```

---

## Resume & Progress Tracking

**Progress file** (stored in output directory):

```json
// {dataset}-chips/.ftw-progress.json
{
  "operation": "select-images",
  "started_at": "2024-03-15T10:30:00Z",
  "updated_at": "2024-03-15T10:45:00Z",
  "total_chips": 1500,
  "completed": ["ftw-34UFF1628_2024", "ftw-34UFF1629_2024"],
  "skipped": {
    "ftw-34UFF1630_2024": "no cloud-free planting scene"
  },
  "failed": {
    "ftw-34UFF9999_2024": "STAC API timeout"
  },
  "parameters": {
    "year": 2024,
    "cloud_cover_scene": 10,
    "buffer_days": 14
  }
}
```

**Resume behavior:**
- `--resume`: Read progress file, skip already-completed chips
- Without `--resume`: Start fresh, overwrite progress file
- Also checks if output files already exist

---

## FTW STAC Extension

**Repository:** Clone `stac-extensions/template` -> `fieldsoftheworld/ftw-stac-extension`

**Extension prefix:** `ftw:`

**Flattened properties (easier to query):**

| Property | Type | Description |
|----------|------|-------------|
| `ftw:planting_day` | integer | Day of year (1-365) for planting |
| `ftw:harvest_day` | integer | Day of year (1-365) for harvest |
| `ftw:calendar_year` | integer | Calendar year for crop cycle |
| `ftw:stac_host` | string | Source STAC catalog used |
| `ftw:cloud_cover_scene_threshold` | number | Scene-level threshold used |
| `ftw:cloud_cover_pixel_threshold` | number | Pixel-level threshold (if used) |
| `ftw:buffer_days` | integer | Search buffer in days |
| `ftw:pixel_check` | boolean | Whether pixel-level check was used |
| `ftw:season` | string | "planting" or "harvest" |
| `ftw:source` | string | Image source identifier (e.g., "sentinel-2") |

---

## Code Organization

Structured for future extraction to ftw-common:

```
src/ftw_dataset_tools/
├── api/
│   ├── imagery/                  # New module - future ftw-common candidate
│   │   ├── __init__.py
│   │   ├── crop_calendar.py      # Crop calendar lookup (from ftw-baselines)
│   │   ├── scene_selection.py    # STAC queries, cloud filtering
│   │   ├── image_download.py     # Clipping and downloading COGs
│   │   └── settings.py           # Constants, URLs, band configs
│   ├── stac.py                   # Updated for new structure
│   └── masks.py                  # Updated for year-based naming
├── commands/
│   ├── select_images.py          # New CLI command
│   ├── download_images.py        # New CLI command
│   ├── refine_cloud_cover.py     # New CLI command
│   └── create_dataset.py         # Updated with image flags
```

---

## Implementation Order

### Phase 1: Foundation (update existing code)
1. Update `api/masks.py` - add year to filename helpers
2. Update `api/stac.py` - add year to ChipInfo, directory names, item IDs
3. Update `api/dataset.py` - pass year through, create year-based directories
4. Update tests for new naming convention

### Phase 2: Imagery module (new code)
5. Create `api/imagery/settings.py` - constants, URLs, band configs
6. Create `api/imagery/crop_calendar.py` - copy from ftw-baselines, adapt
7. Create `api/imagery/scene_selection.py` - STAC queries, cloud filtering
8. Create `api/imagery/image_download.py` - COG clipping, writing

### Phase 3: CLI commands
9. Create `commands/select_images.py` - standalone command
10. Create `commands/download_images.py` - standalone command
11. Create `commands/refine_cloud_cover.py` - pixel-level check command
12. Update `commands/create_dataset.py` - add `--select-images`, `--download-images` flags

### Phase 4: STAC extension
13. Clone stac-extensions/template repo
14. Define ftw extension schema and examples
15. Publish to fieldsoftheworld org

---

## Default Values

| Parameter | Default |
|-----------|---------|
| STAC host | EarthSearch (AWS) |
| Scene-level cloud cover | 10% |
| Pixel-level cloud cover | 0% |
| Buffer days | 14 |
| Bands | R, G, B, NIR + cloud probability |
| On missing | skip (with warning) |
