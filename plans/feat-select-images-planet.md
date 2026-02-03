# feat: Add Planet Imagery Selection and Download Commands

## Overview

Add two new CLI commands for PlanetScope satellite imagery:

1. **`ftwd select-images-planet`** - Query Planet STAC API, assess clear coverage, select best scenes, generate thumbnails, create STAC child items
2. **`ftwd download-images-planet`** - Activate assets and download imagery (with `--activate-only` mode for batch workflows)

This follows the established S2 pattern (`select-images` + `download-images`). **No premature abstraction** - we copy the S2 pattern directly and refactor shared utilities only if duplication becomes painful after both implementations exist.

## Problem Statement

The FTW dataset currently only supports Sentinel-2 imagery. Users need PlanetScope imagery for:
- Higher resolution (3-4m vs 10m) for fine-grained field boundaries
- Alternative coverage in regions with persistent S2 cloud cover
- Validation studies comparing S2 vs Planet-derived boundaries

## Proposed Solution

### Two-Command Workflow

```
Step 1: SELECT (fast, ~9s per chip)
┌────────────────────────────────────────────────────────────────┐
│ ftwd select-images-planet ./dataset --year 2024                │
│   - Query Planet STAC API                                      │
│   - Assess clear coverage (estimate mode)                      │
│   - Iterative buffer expansion (14 days default, 3 iterations) │
│   - Generate thumbnails                                        │
│   - Create STAC child items with all metadata for download     │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
Step 2: DOWNLOAD (10-30 min activation wait per asset)
┌────────────────────────────────────────────────────────────────┐
│ ftwd download-images-planet ./dataset --bands red green blue   │
│   - Read scene info from STAC items (no re-query needed)       │
│   - Activate assets (or --activate-only to queue all first)    │
│   - Wait for activation, download, clip to chip bounds         │
│   - Update STAC items with local asset paths                   │
└────────────────────────────────────────────────────────────────┘
```

### Command Interfaces

#### `select-images-planet`

```bash
# Basic usage
ftwd select-images-planet ./my-dataset --year 2024

# Custom iteration parameters
ftwd select-images-planet ./dataset --year 2024 \
  --buffer-days 21 \
  --num-iterations 5

# Verbose output showing iteration details
ftwd select-images-planet ./dataset --year 2024 -v

# Force re-selection
ftwd select-images-planet ./dataset --year 2024 --force
```

#### `download-images-planet`

```bash
# Download with specific bands
ftwd download-images-planet ./dataset --bands red green blue nir

# Activate all assets first (queue activations, exit immediately)
ftwd download-images-planet ./dataset --activate-only

# Then download later (assets already active)
ftwd download-images-planet ./dataset --bands red green blue nir --resume
```

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **No shared refactor** | Copy S2 pattern directly | Avoid premature abstraction; refactor later if needed |
| **Flat file structure** | 3 files in `api/imagery/` | Match S2 structure, simpler imports |
| **SDK vs HTTP** | Planet Python SDK | Built-in auth, rate limiting, activation polling |
| **Coverage mode** | Estimate only (v1) | Fast, synchronous; add UDM2 later if requested |
| **Separate commands** | select + download | Match S2 pattern; activation takes 10-30min |

## Technical Approach

### File Structure (3 new files, flat)

```
src/ftw_dataset_tools/
├── commands/
│   ├── select_images_planet.py     # CLI command
│   └── download_images_planet.py   # CLI command
└── api/
    └── imagery/
        ├── planet_client.py        # Auth, settings, SDK init (~100 lines)
        ├── planet_selection.py     # Queries, coverage, selection, thumbnails (~300 lines)
        └── planet_download.py      # Activation, download, clipping (~200 lines)
```

### Data Structures (Minimal)

```python
# api/imagery/planet_selection.py

@dataclass
class PlanetScene:
    """A selected PlanetScope scene. Wraps pystac.Item like S2's SelectedScene."""
    item: pystac.Item
    season: Literal["planting", "harvest"]
    clear_coverage: float  # 0-100, from coverage API
    datetime: datetime
    stac_url: str


@dataclass
class PlanetSelectionResult:
    """Result of Planet imagery selection for a chip. Mirrors S2's SceneSelectionResult."""
    chip_id: str
    bbox: tuple[float, float, float, float]
    year: int
    crop_calendar: CropCalendarDates
    planting_scene: PlanetScene | None = None
    harvest_scene: PlanetScene | None = None
    skipped_reason: str | None = None
    candidates_checked: int = 0
    iterations_used: int = 0
```

### STAC Child Item Structure (Minimal)

Store only what's needed for download command to work:

```json
{
  "id": "ftw-34UFF1628_2024_planting_planet",
  "properties": {
    "datetime": "2024-06-15T10:30:00Z",
    "ftw:source": "planet",
    "ftw:season": "planting",
    "ftw:year": 2024,
    "ftw:scene_id": "20240615_103000_38_2461",
    "ftw:collection": "PSScene",
    "ftw:clear_coverage": 97.5,
    "ftw:cloud_cover": 2.5
  },
  "assets": {
    "thumbnail": {
      "href": "./ftw-34UFF1628_2024_planting_planet_thumb.png",
      "type": "image/png",
      "roles": ["thumbnail"]
    }
  },
  "links": [
    {
      "rel": "derived_from",
      "href": "https://api.planet.com/x/data/collections/PSScene/items/20240615_103000_38_2461",
      "type": "application/geo+json"
    }
  ]
}
```

### Error Handling Strategy

| Scenario | Handling |
|----------|----------|
| Coverage API timeout | Fall back to `cloud_cover` property from item metadata |
| Thumbnail 404 | Log warning, skip thumbnail, continue selection |
| Activation timeout | Mark in STAC as `ftw:activation_failed`, skip, continue |
| Download failure | Log error, skip item, continue with others |
| Invalid API key | Fail fast with clear error message before processing |
| No Planet coverage for chip | Respect `--on-missing` (skip or fail) |

## Implementation Phases

### Phase 1: Selection Command (End-to-End)

**Goal:** Working `select-images-planet` command that creates complete STAC items with thumbnails.

**Files to create:**
- `api/imagery/planet_client.py`
- `api/imagery/planet_selection.py`
- `commands/select_images_planet.py`
- `tests/test_planet_selection.py`

**Tasks:**

- [ ] Create `planet_client.py`:
  ```python
  # Settings
  PLANET_STAC_URL = "https://api.planet.com/x/data/"
  PLANET_TILES_URL = "https://tiles.planet.com/data/v1/"
  DEFAULT_BUFFER_DAYS = 14
  DEFAULT_NUM_ITERATIONS = 3
  VALID_BANDS = ["blue", "green", "red", "nir", "coastal_blue", "green_i", "yellow", "red_edge"]

  class PlanetClient:
      def __init__(self, api_key: str | None = None):
          """Initialize with API key from param or PL_API_KEY env var."""

      def validate_auth(self) -> bool:
          """Test API key validity with a simple API call."""

      def get_stac_client(self) -> pystac_client.Client:
          """Get authenticated PySTAC client for Planet STAC API."""

      def get_sdk_client(self) -> Planet:
          """Get Planet SDK client for Data API operations."""
  ```

- [ ] Create `planet_selection.py` (copy S2 pattern from `scene_selection.py`):
  ```python
  def select_planet_scenes_for_chip(
      client: PlanetClient,
      chip_id: str,
      chip_bbox: tuple[float, float, float, float],
      chip_geometry: dict,  # For coverage API
      year: int,
      buffer_days: int = 14,
      num_iterations: int = 3,
      cloud_cover_threshold: float = 2.0,
      on_progress: Callable[[str], None] | None = None,
  ) -> PlanetSelectionResult:
      """Select best Planet scenes for planting and harvest seasons.

      Copies the iteration pattern from S2's select_scenes_for_chip():
      - Start with buffer_days around target date
      - Expand buffer if no good scenes found
      - Track checked scene IDs to avoid re-evaluation
      - Return best scene per season based on clear_coverage
      """

  def get_clear_coverage(
      client: PlanetClient,
      item_id: str,
      geometry: dict,
  ) -> float:
      """Get clear coverage percentage using Planet's estimate API.

      Falls back to item's cloud_cover property on API error.
      """

  def generate_thumbnail(
      client: PlanetClient,
      item_id: str,
      output_path: Path,
      width: int = 256,
  ) -> Path | None:
      """Generate thumbnail via Planet tiles endpoint. Returns None on failure."""
  ```

- [ ] Create CLI command:
  ```python
  @click.command("select-images-planet")
  @click.argument("input_path", type=click.Path(exists=True))
  @click.option("--year", type=int, required=True)
  @click.option("--buffer-days", type=int, default=14, show_default=True,
                help="Initial buffer days around target date")
  @click.option("--num-iterations", type=int, default=3, show_default=True,
                help="Number of buffer expansion iterations")
  @click.option("--cloud-cover-chip", type=float, default=2.0, show_default=True,
                help="Maximum cloud cover percentage (maps to min clear coverage)")
  @click.option("--on-missing", type=click.Choice(["skip", "fail"]), default="skip")
  @click.option("--force", is_flag=True, help="Re-select even if Planet imagery exists")
  @click.option("--verbose", "-v", is_flag=True)
  def select_images_planet(...):
  ```

- [ ] Register command in `cli.py`
- [ ] Add `planet>=3.0.0` to `pyproject.toml`
- [ ] Write tests with mocked Planet API responses
- [ ] Add one `@pytest.mark.network` test against real Planet API

**Success Criteria:**
- Command runs on test dataset
- STAC child items created with thumbnails
- Parent items updated with `ftw:planting_planet` links
- Handles missing coverage gracefully

---

### Phase 2: Download Command (End-to-End)

**Goal:** Working `download-images-planet` command with `--activate-only` mode.

**Files to create:**
- `api/imagery/planet_download.py`
- `commands/download_images_planet.py`
- `tests/test_planet_download.py`

**Tasks:**

- [ ] Create `planet_download.py`:
  ```python
  def activate_asset(
      client: PlanetClient,
      item_id: str,
      asset_type: str,
  ) -> dict:
      """Start asset activation (non-blocking). Returns status dict."""

  def wait_for_activation(
      client: PlanetClient,
      item_id: str,
      asset_type: str,
      timeout: int = 3600,
      poll_interval: int = 30,
      on_progress: Callable[[str], None] | None = None,
  ) -> dict:
      """Wait for asset activation with polling. Returns status dict."""

  def download_and_clip_scene(
      client: PlanetClient,
      item_id: str,
      asset_type: str,
      bbox: tuple[float, float, float, float],
      output_path: Path,
      bands: list[str] | None = None,
      on_progress: Callable[[str], None] | None = None,
  ) -> Path:
      """Download asset, clip to bbox, return output path."""
  ```

- [ ] Create CLI command:
  ```python
  @click.command("download-images-planet")
  @click.argument("catalog_path", type=click.Path(exists=True))
  @click.option("--bands", type=click.Choice(VALID_BANDS), multiple=True,
                default=("red", "green", "blue", "nir"), show_default=True)
  @click.option("--asset-type", default="ortho_analytic_4b", show_default=True)
  @click.option("--resolution", type=float, default=3.0, show_default=True)
  @click.option("--activate-only", is_flag=True,
                help="Only activate assets, don't download (for batch workflows)")
  @click.option("--resume", is_flag=True, help="Skip already downloaded items")
  @click.option("--verbose", "-v", is_flag=True)
  def download_images_planet(...):
  ```

- [ ] Read scene info from STAC items (`ftw:scene_id`, `ftw:collection`)
- [ ] Implement activate-only mode (queue all, exit)
- [ ] Implement full download with progress
- [ ] Update STAC items with local asset paths
- [ ] Register command in `cli.py`
- [ ] Write tests

**Success Criteria:**
- `--activate-only` queues activations and exits quickly
- Normal mode activates, waits, downloads, clips
- `--resume` skips already-downloaded items
- STAC items updated with local asset hrefs

---

### Phase 3: Integration & Polish (Optional)

**Goal:** Integration with `create-dataset`, stats, documentation.

**Tasks:**

- [ ] Add `--select-images-planet` flag to `create_dataset.py`
- [ ] Add `--download-images-planet` flag to `create_dataset.py`
- [ ] Implement `--show-stats` for Planet coverage
- [ ] Update README with Planet workflow documentation
- [ ] Add cost/quota documentation
- [ ] Full integration test

**Success Criteria:**
- Full pipeline works via `create-dataset`
- Documentation complete

## Acceptance Criteria

### Functional Requirements

**Authentication:**
- [ ] Commands validate Planet API key before processing
- [ ] API key accepted via `PL_API_KEY` env var
- [ ] Clear error message for missing/invalid API key

**Selection (`select-images-planet`):**
- [ ] Iterates with configurable `--buffer-days` (default 14)
- [ ] Configurable `--num-iterations` (default 3)
- [ ] Generates thumbnails via Planet tiles endpoint
- [ ] Creates STAC child items with metadata needed for download
- [ ] `--force` re-selects even if Planet imagery exists
- [ ] `--on-missing skip` skips chips with no coverage

**Download (`download-images-planet`):**
- [ ] Reads scene info from STAC items (no re-query)
- [ ] `--bands` option for band selection
- [ ] `--activate-only` queues activations without waiting
- [ ] Normal mode activates, waits, downloads
- [ ] `--resume` skips already downloaded items
- [ ] Clips imagery to chip bounds
- [ ] Updates STAC items with local asset paths

### Non-Functional Requirements

- [ ] Respects Planet rate limits (via SDK)
- [ ] Selection processes 100 chips in <15 minutes
- [ ] API key never logged or written to files
- [ ] Graceful cancellation via Ctrl+C

### Quality Gates

- [ ] Unit test coverage ≥80% for new code
- [ ] One e2e test with real Planet API (marked `@pytest.mark.network`)
- [ ] All existing tests pass
- [ ] Ruff/pre-commit checks pass
- [ ] CLI help text complete with `show_default=True`

## Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| `planet` | ≥3.0.0 | Planet Python SDK |
| `pystac-client` | existing | STAC API queries |
| `rasterio` | existing | Raster clipping |

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Long activation times (10-30 min) | `--activate-only` mode for batch workflows |
| Coverage API errors | Fall back to item's `cloud_cover` property |
| Thumbnail failures | Non-blocking; skip and continue |
| Rate limiting | SDK handles automatically |

## Future Considerations (Deferred)

These are explicitly **not** in scope for v1:

- `--coverage-mode accurate` using UDM2 (add if users request)
- Shared `selection_utils.py` refactor (add if third imagery source needed)
- SkySat / Basemaps support
- Checksum validation (trust SDK)
- Spectral harmonization with S2

## References

### Internal (Copy These Patterns)

- S2 selection: `src/ftw_dataset_tools/commands/select_images.py`
- S2 scene selection: `src/ftw_dataset_tools/api/imagery/scene_selection.py`
- S2 download: `src/ftw_dataset_tools/api/imagery/image_download.py`

### External

- [Planet STAC API](https://docs.planet.com/develop/apis/data/stac/)
- [Planet Clear Coverage API](https://docs.planet.com/develop/apis/data/items/)
- [Planet Python SDK](https://planet-sdk-for-python-v2.readthedocs.io/en/latest/)
- [Planet Tiles API](https://developers.planet.com/docs/basemaps/tile-services/)
