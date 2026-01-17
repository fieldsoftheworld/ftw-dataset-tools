# Claude Code Instructions for ftw-dataset-tools

This file contains project-specific instructions for Claude Code when working in this repository.

## Project Overview

ftw-dataset-tools (ftwd) is a Python CLI tool for creating the Fields of the World (FTW) benchmark dataset. It uses Click for CLI, DuckDB for data processing, geoparquet-io for GeoParquet I/O, Rasterio for raster operations, and follows modern Python packaging standards.

**Entry point**: `ftwd` command defined in `src/ftw_dataset_tools/cli.py`

---

## Planning and Research Before Coding

**Always research before implementing.** This is the most important guideline.

### Before Writing Any Code

1. **Understand the request fully** - Ask clarifying questions if scope is ambiguous
2. **Search for existing patterns** - Check if similar functionality exists
3. **Identify affected files** - Map out what needs to change
4. **Check for utilities** - Review `api/` modules for reusable code
5. **Understand test requirements** - Look at existing tests for the area you're modifying

### Research Commands

```bash
# Find files by pattern
ls src/ftw_dataset_tools/api/
ls src/ftw_dataset_tools/commands/

# Search for function usage
grep -r "function_name" src/ftw_dataset_tools/

# Check how similar features are implemented
grep -r "pattern_to_find" --include="*.py"

# Run tests in specific area
pytest tests/test_<area>.py -v
```

### Questions to Answer Before Coding

1. Does this feature already exist partially?
2. What existing utilities can I reuse?
3. How do similar features handle errors?
4. What's the test coverage expectation?
5. Are there edge cases mentioned in similar code?

---

## Codebase Architecture

Understanding the structure prevents architectural mistakes.

### Directory Layout

```
src/ftw_dataset_tools/
├── __init__.py          # Package init with version
├── cli.py               # CLI entry point, command registration
├── commands/
│   ├── __init__.py
│   ├── create_dataset.py    # Main dataset creation pipeline
│   ├── create_chips.py      # Chip definitions with coverage stats
│   ├── create_masks.py      # Raster mask generation
│   ├── create_boundaries.py # Vector to boundary line conversion
│   ├── create_ftw_grid.py   # Hierarchical grid creation
│   └── get_grid.py          # Grid cell retrieval
└── api/
    ├── __init__.py
    ├── dataset.py       # Dataset creation logic
    ├── geo.py           # Geometry utilities
    ├── grid.py          # Grid operations
    ├── boundaries.py    # Boundary processing
    ├── masks.py         # Mask generation
    ├── ftw_grid.py      # FTW grid logic
    ├── field_stats.py   # Field coverage statistics
    └── stac.py          # STAC catalog generation
```

### Key Patterns

**1. Commands/API Separation**
CLI commands are thin wrappers. Business logic lives in `api/`.

```python
# In commands/mycommand.py - CLI wrapper
@click.command()
@click.argument("input_file")
@click.option("--verbose", "-v", is_flag=True)
def mycommand(input_file, verbose):
    """Command docstring."""
    from ftw_dataset_tools.api.mymodule import mycommand_impl
    mycommand_impl(input_file, verbose)

# In api/mymodule.py - Business logic
def mycommand_impl(input_file, verbose):
    # Actual implementation here
```

**2. Error Handling**
Use Click's error handling for user-facing errors:
```python
from click import ClickException, BadParameter

# For general errors
raise ClickException("Human readable error message")

# For parameter validation
raise BadParameter("Invalid value for --option")
```

---

## Key Dependencies and Usage

### DuckDB (SQL engine)
- Primary data processing engine for Parquet files
- Spatial extension for geometry operations
- Reading/writing GeoParquet files

```python
import duckdb

con = duckdb.connect()
con.install_extension("spatial")
con.load_extension("spatial")
con.execute("SELECT * FROM read_parquet('file.parquet')")
```

**IMPORTANT:** Use DuckDB for data processing/queries and geoparquet-io for reading/writing GeoParquet files. Do NOT use GeoPandas or pandas for file I/O - they are only acceptable for specific transformations that geoparquet-io doesn't support.

### geoparquet-io (GeoParquet I/O)
- **Primary library for reading/writing GeoParquet files**
- Use the fluent API for all GeoParquet operations
- Preferred over GeoPandas for vector I/O

```python
import geoparquet_io as gpio

# Read, transform, and write GeoParquet
gpio.read('input.parquet') \
    .add_bbox() \
    .reproject('EPSG:4326') \
    .write('output.parquet')

# Simple read and add bbox
gpio.read('input.parquet').add_bbox().write('output.parquet')
```

**Do NOT use GeoPandas or pandas for GeoParquet I/O.** Always prefer geoparquet-io.

### Rasterio (Raster operations)
- Reading/writing raster files (GeoTIFF)
- Rasterization of vector data

```python
import rasterio
from rasterio.features import rasterize

with rasterio.open(file_path) as src:
    data = src.read()
```

### Click (CLI framework)
- All CLI commands use Click decorators
- Commands registered in `cli.py`

---

## Logging (TODO)

> **Note:** This section describes the target logging infrastructure. It is not yet implemented.
> See GitHub issue for tracking: https://github.com/fieldsoftheworld/ftw-dataset-tools/issues/14

**Target: Never use `click.echo()` in `api/` modules. Always use logging helpers.**

`click.echo()` is allowed in `commands/` for direct CLI output, but `api/` modules should use a logger for testability and library compatibility.

### Future Implementation

```python
from ftw_dataset_tools.api.logging_config import success, warn, error, info, debug, progress

success("Operation completed")  # Green - for completed operations
warn("Something to note")       # Yellow - for warnings
error("Something went wrong")   # Red - for errors
info("Informational message")   # Cyan - for tips/context
debug("Debug details")          # Only shown when verbose=True
progress("Processing...")       # Plain text - for status updates
```

### Why Not click.echo() in api/?

1. **Testability**: Logger output is captured by pytest; click.echo requires special handling
2. **Library usage**: When ftwd is used as a library, users can configure logging handlers
3. **Consistency**: Single source of truth for all output formatting
4. **Verbosity control**: Debug messages are automatically hidden unless `--verbose` is passed

---

## Git Commit Messages

Keep commit messages brief and focused:

- **Maximum 1-2 lines** - Single sentence preferred
- Use imperative mood: "Add feature" not "Added feature"
- Start with a verb: Add, Fix, Update, Remove, Refactor, Improve
- No period at the end, no emoji
- Focus on *what* changed, not *how*

**Good examples:**
```
Add spatial filtering to ftwd create-chips command
Fix bbox validation for antimeridian-crossing geometries
Remove deprecated --format flag from reproject command
```

**Bad examples:**
```
Updated the create_chips.py file to add new functionality for filtering rows and columns with various options including bbox support
```

Do NOT include the standard Claude Code footer. Keep commits minimal.

---

## Code Formatting with Ruff

This project uses Ruff for linting and formatting.

### Before Any Commit
```bash
pre-commit run --all-files
```

### Ruff Configuration (`pyproject.toml`)
- **Line length**: 100 characters
- **Target Python**: 3.11+
- **Enabled rules**: E, W, F, I, B, C4, UP, ARG, SIM, TCH, PTH, RUF
- Use double quotes, f-strings, comprehensions
- Imports: stdlib, third-party, first-party (`ftw_dataset_tools`)

### Auto-fix
```bash
ruff check --fix .
ruff format .
```

---

## Code Complexity (TODO)

> **Note:** Complexity monitoring is not yet enforced via pre-commit.
> See GitHub issue for tracking: https://github.com/fieldsoftheworld/ftw-dataset-tools/issues/15

**Target: Aim for grade 'A' complexity on all new code.**

### Complexity Grades
- **A**: Simple, easy to understand (TARGET)
- **B**: Acceptable, low complexity
- **C-F**: Needs refactoring

### Target Pre-commit Thresholds
```bash
# Strict check - aim for this
xenon --max-absolute=A --max-modules=A --max-average=A src/ftw_dataset_tools/

# Target pre-commit threshold
xenon --max-absolute=D --max-modules=C --max-average=B src/ftw_dataset_tools/
```

### Reducing Complexity

**1. Extract helper functions**
```python
# BAD: Long function with many branches
def process_file(file, options):
    if options.a:
        # 20 lines of code
    elif options.b:
        # 20 lines of code

# GOOD: Extracted helpers
def process_file(file, options):
    if options.a:
        return _process_option_a(file)
    elif options.b:
        return _process_option_b(file)
```

**2. Early returns (guard clauses)**
```python
# BAD: Nested conditions
def validate(data):
    if data:
        if data.valid:
            if data.complete:
                return process(data)
    return None

# GOOD: Early returns
def validate(data):
    if not data:
        return None
    if not data.valid:
        return None
    if not data.complete:
        return None
    return process(data)
```

**3. Use data structures instead of branching**
```python
# BAD: Long if-elif chain
if format == "json":
    return json_handler()
elif format == "csv":
    return csv_handler()

# GOOD: Dictionary dispatch
handlers = {
    "json": json_handler,
    "csv": csv_handler,
}
return handlers[format]()
```

**4. Single responsibility per function**
- Each function does one thing
- If you need "and" to describe it, split it
- Max 30-40 lines per function

---

## Testing Guidelines

### Test Coverage Requirements

**All new code must have tests. This is mandatory, not optional.**

- Every commit with code changes should include corresponding tests
- Test both happy paths and error cases

```bash
# Check coverage for a specific file
pytest --cov=src/ftw_dataset_tools/api/mymodule --cov-report=term-missing tests/test_mymodule.py

# Check overall coverage
pytest --cov=src/ftw_dataset_tools --cov-report=term-missing
```

### Test Structure
```
tests/
├── conftest.py          # Shared fixtures
├── data/                # Test data files
├── test_<module>.py     # Tests mirror source structure
```

### Running Tests
```bash
# All tests
pytest

# Skip slow tests
pytest -m "not slow"

# Skip network tests
pytest -m "not network"

# Specific module
pytest tests/test_create_chips.py -v

# Single test
pytest tests/test_create_chips.py::TestCreateChips::test_valid_input -v
```

### Test Patterns Used

**1. Class-based organization**
```python
class TestCreateChips:
    """Tests for create_chips function."""

    def test_valid_input(self):
        result = create_chips(input_file)
        assert result is not None
```

**2. Fixtures for temp files**
```python
@pytest.fixture
def output_file(self):
    tmp_path = Path(tempfile.gettempdir()) / f"test_{uuid.uuid4()}.parquet"
    yield str(tmp_path)
    if tmp_path.exists():
        tmp_path.unlink()
```

**3. Markers for conditional tests**
```python
@pytest.mark.slow
class TestRemoteFiles:
    """Tests requiring network access."""
```

**4. CLI testing with CliRunner**
```python
from click.testing import CliRunner
from ftw_dataset_tools.commands.create_chips import create_chips

runner = CliRunner()
result = runner.invoke(create_chips, [input_file, output_file])
assert result.exit_code == 0
```

---

## Pre-Commit Workflow Summary

Before every commit:
```bash
# 1. Run all checks
pre-commit run --all-files

# 2. Fix any issues
ruff check --fix .
ruff format .

# 3. Run tests
pytest

# 4. If tests pass, commit
git add .
git commit -m "Brief description of change"
```

---

## Quick Reference

### Adding a New CLI Command

1. Define core logic in `api/newmodule.py`
2. Add CLI wrapper in `commands/newcommand.py`
3. Register command in `cli.py`
4. Add tests in `tests/test_newcommand.py`

### Adding a CLI Option

1. Add option decorator to command in `commands/`
2. Pass to api function
3. Update tests

### Modifying Core Logic

1. Read existing tests first
2. Understand the function's contract
3. Make changes
4. Run tests: `pytest tests/test_<module>.py -v`

### Working with GeoParquet Files

**Always use geoparquet-io** for reading and writing GeoParquet files:

```python
import geoparquet_io as gpio

# Read and write with best practices (bbox, proper metadata)
gpio.read('input.parquet').add_bbox().write('output.parquet')

# Reproject
gpio.read('input.parquet').reproject('EPSG:4326').write('output.parquet')

# Chain multiple operations
gpio.read('input.parquet') \
    .add_bbox() \
    .reproject('EPSG:4326') \
    .write('output.parquet')
```

**Do NOT use:**
- `geopandas.read_parquet()` / `gdf.to_parquet()` - use `gpio.read()` / `.write()` instead
- `pandas.read_parquet()` - use `gpio.read()` instead
- Internal geoparquet-io modules - use the fluent API (`gpio.read()...`)

**Exception:** GeoPandas is acceptable for in-memory geometry transformations that require GeoDataFrame methods not available in geoparquet-io or DuckDB.
