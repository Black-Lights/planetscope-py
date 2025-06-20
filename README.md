# PlanetScope-py

A professional Python library for PlanetScope satellite imagery analysis, providing comprehensive tools for scene discovery, metadata analysis, and spatial-temporal density calculations using Planet's Data API.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Library Status](https://img.shields.io/badge/Library%20Status-Complete-green.svg)](#current-status)
[![Spatial Analysis](https://img.shields.io/badge/Spatial%20Analysis-Complete-green.svg)](#spatial-analysis-engine-complete)
[![Visualization](https://img.shields.io/badge/Visualization-Basic-yellow.svg)](#basic-visualization-current)
[![Test Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen.svg)](#testing)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Status

**Current Status**: Spatial Analysis Engine Complete  
**Test Coverage**: 280/280 tests passing (100%)  
**API Integration**: Fully functional with real Planet API  
**Spatial Analysis**: Multi-algorithm density calculations complete  
**Visualization**: Basic export and plotting (Phase 4 advanced features planned)  
**Python Support**: 3.10+  
**License**: MIT  

## Overview

PlanetScope-py is designed for remote sensing researchers, GIS analysts, and Earth observation professionals who need reliable tools for working with PlanetScope satellite imagery. The library provides a robust foundation for scene inventory management and sophisticated spatial-temporal analysis workflows.

## Features

### Foundation (Complete)
- **Authentication System**: Hierarchical API key detection with secure credential management
- **Configuration Management**: Multi-source configuration with environment variable support
- **Input Validation**: Comprehensive geometry, date, and parameter validation
- **Exception Handling**: Professional error hierarchy with detailed context and troubleshooting guidance
- **Security**: API key masking, secure session management, and credential protection
- **Cross-Platform**: Full compatibility with Windows, macOS, and Linux environments

### Planet API Integration (Complete)
- **Scene Discovery**: Robust search functionality with advanced filtering capabilities
- **Metadata Processing**: Comprehensive scene metadata extraction and analysis
- **Rate Limiting**: Intelligent rate limiting with exponential backoff and retry logic
- **API Response Handling**: Optimized response caching and pagination support
- **Date Formatting**: Planet API compliant date formatting with end-of-day handling
- **Geometry Validation**: Multi-format geometry support (GeoJSON, Shapely, WKT)
- **Batch Operations**: Support for multiple geometry searches with parallel processing
- **Quality Assessment**: Scene filtering based on cloud cover, sun elevation, and quality metrics
- **Preview Support**: Scene preview URL generation for visual inspection
- **Real-World Testing**: Verified with actual Planet API calls and data retrieval

### Spatial Analysis Engine (Complete)
- **Multi-Algorithm Density Calculation**: Three computational methods (rasterization, vector overlay, adaptive grid)
- **High-Resolution Analysis**: Support for 3m to 1000m grid resolutions
- **Performance Optimization**: Automatic method selection based on dataset characteristics
- **Memory Efficient Processing**: Adaptive grid and chunking for large areas
- **Basic Visualization**: GeoTIFF export with QGIS styling and summary plots

### Basic Visualization and Export (Current)
- **GeoTIFF Export**: GIS-compatible files with automatic QGIS styling
- **Summary Plots**: Four-panel density visualization with statistics
- **Basic Formats**: Export to NumPy arrays and CSV files
- **Statistical Analysis**: Core density statistics and quality metrics
- **GIS Compatibility**: Direct compatibility with QGIS and ArcGIS

### Advanced Visualization and Export (Planned)
- **Interactive Mapping**: Web-based interactive maps with zoom and pan
- **Timeline Plotting**: Temporal density analysis with time-series visualization
- **Advanced Export**: Enhanced GeoJSON export with metadata and styling
- **Report Generation**: Automated analysis reports with charts and recommendations
- **Dashboard Integration**: Web dashboard for monitoring and analysis workflows

## Installation

### Standard Installation
```bash
# Development installation (recommended)
git clone https://github.com/Black-Lights/planetscope-py.git
cd planetscope-py
pip install -e .
```

### Development Installation
```bash
# Clone the repository
git clone https://github.com/Black-Lights/planetscope-py.git
cd planetscope-py

# Create and activate a virtual environment (recommended)
python -m venv myvenv
# On Windows:
myvenv\Scripts\activate
# On macOS/Linux:
source myvenv/bin/activate

# Install the package in editable mode and dev requirements
pip install -e .
pip install -r requirements-dev.txt
```

## Authentication

PlanetScope-py supports multiple authentication methods with automatic discovery in order of priority:

### Method 1: Environment Variable (Recommended)
```bash
# Linux/macOS
export PL_API_KEY="your_planet_api_key_here"

# Windows Command Prompt
set PL_API_KEY=your_planet_api_key_here

# Windows PowerShell
$env:PL_API_KEY="your_planet_api_key_here"
```

### Method 2: Configuration File
Create `~/.planet.json` in your home directory:
```json
{
    "api_key": "your_planet_api_key_here"
}
```

### Method 3: Direct Parameter
```python
from planetscope_py import PlanetAuth
auth = PlanetAuth(api_key="your_planet_api_key_here")
```

### Obtaining Your API Key
Get your Planet API key from [Planet Account Settings](https://www.planet.com/account/#/).

## Quick Start

### Basic Scene Search
```python
from planetscope_py import PlanetScopeQuery

# Initialize query system (automatically detects API key)
query = PlanetScopeQuery()

# Define area of interest (example: Milan, Italy)
milan_geometry = {
    "type": "Point",
    "coordinates": [9.1900, 45.4642]  # [longitude, latitude]
}

# Search for scenes
results = query.search_scenes(
    geometry=milan_geometry,
    start_date="2025-01-01",
    end_date="2025-01-31",
    cloud_cover_max=0.2,  # 20% maximum cloud cover
    item_types=["PSScene"]
)

# Check results
print(f"Found {len(results['features'])} scenes")
```

### Spatial Density Analysis
```python
from planetscope_py import SpatialDensityEngine, DensityConfig, DensityMethod
from shapely.geometry import box

# Define region of interest
roi = box(9.04, 45.40, 9.28, 45.52)  # Milan bounding box

# Configure spatial analysis
config = DensityConfig(
    resolution=30.0,  # 30m grid resolution
    method=DensityMethod.AUTO  # Automatic method selection
)

# Initialize spatial analysis engine
engine = SpatialDensityEngine(config)

# Calculate spatial density
density_result = engine.calculate_density(
    scene_footprints=results['features'],
    roi_geometry=roi
)

print(f"Analysis completed using {density_result.method_used.value} method")
print(f"Grid size: {density_result.grid_info['width']}×{density_result.grid_info['height']}")
print(f"Density range: {density_result.stats['min']}-{density_result.stats['max']} scenes per cell")
```

### Visualization and Export
```python
from planetscope_py.visualization import DensityVisualizer
import os

# Create visualizer
visualizer = DensityVisualizer()

# Create output directory
os.makedirs("analysis_output", exist_ok=True)

# Generate basic summary visualization
fig = visualizer.create_summary_plot(
    density_result=density_result,
    save_path="analysis_output/density_summary.png"
)

# Export GeoTIFF with QGIS styling
visualizer.export_density_geotiff_with_style(
    density_result=density_result,
    output_path="analysis_output/density_map.tif"
)

print("Basic visualization files created:")
print("  - analysis_output/density_summary.png")
print("  - analysis_output/density_map.tif") 
print("  - analysis_output/density_map.qml")
```

### Advanced Usage with Metadata Processing
```python
from planetscope_py import PlanetScopeQuery, MetadataProcessor
from planetscope_py.utils import validate_geometry, calculate_area_km2

# Create a 100km x 100km polygon around Milan
milan_polygon = {
    "type": "Polygon",
    "coordinates": [[
        [8.55, 45.91],   # Northwest
        [9.83, 45.91],   # Northeast  
        [9.83, 45.02],   # Southeast
        [8.55, 45.02],   # Southwest
        [8.55, 45.91]    # Close polygon
    ]]
}

# Validate geometry and calculate area
validated_geom = validate_geometry(milan_polygon)
area_km2 = calculate_area_km2(validated_geom)
print(f"Search area: {area_km2:.2f} km²")

# Advanced search with multiple filters
query = PlanetScopeQuery()
results = query.search_scenes(
    geometry=milan_polygon,
    start_date="2024-06-01", 
    end_date="2024-08-31",
    cloud_cover_max=0.20,
    sun_elevation_min=25,
    item_types=["PSScene"]
)

# Process metadata for quality assessment
processor = MetadataProcessor()
assessment = processor.assess_coverage_quality(
    scenes=results["features"],
    target_geometry=milan_polygon
)

print(f"Total scenes: {assessment['total_scenes']}")

# Safe access to assessment results
if assessment['total_scenes'] > 0:
    print(f"Quality analysis: {assessment['quality_analysis']}")
else:
    print("No scenes found for specified criteria")
    
print(f"Recommendations: {assessment['recommendations']}")
```

## Core Components

### Authentication Management
```python
from planetscope_py import PlanetAuth

# Automatic API key discovery
auth = PlanetAuth()

# Check authentication status
if auth.is_authenticated:
    print("Successfully authenticated with Planet API")
    
# Get session for API requests
session = auth.get_session()

# Get authentication tuple for manual requests
username, password = auth.get_auth_tuple()
```

### Configuration System
```python
from planetscope_py import PlanetScopeConfig

# Load default configuration
config = PlanetScopeConfig()

# Access configuration values
base_url = config.base_url
rate_limits = config.rate_limits
timeouts = config.timeouts

# Modify configuration
config.set('max_retries', 5)
config.set('max_roi_area_km2', 15000)

# Export configuration
config_dict = config.to_dict()
```

### Planet API Query System
```python
from planetscope_py import PlanetScopeQuery

query = PlanetScopeQuery()

# Scene search with comprehensive filtering
results = query.search_scenes(
    geometry=geometry,
    start_date="2025-01-01",
    end_date="2025-01-31",
    cloud_cover_max=0.2,
    sun_elevation_min=30,
    item_types=["PSScene"]
)

# Get scene statistics
stats = query.get_scene_stats(
    geometry=geometry,
    start_date="2025-01-01",
    end_date="2025-01-31"
)

# Batch search across multiple geometries
batch_results = query.batch_search(
    geometries=[geom1, geom2, geom3],
    start_date="2025-01-01",
    end_date="2025-01-31"
)

# Filter scenes by quality criteria
quality_scenes = query.filter_scenes_by_quality(
    scenes=results['features'],
    min_quality=0.7,
    max_cloud_cover=0.15,
    exclude_night=True
)

# Get scene preview URLs
previews = query.get_scene_previews(["scene_id_1", "scene_id_2"])
```

### Spatial Density Analysis
```python
from planetscope_py import SpatialDensityEngine, DensityConfig, DensityMethod

# Configure analysis with automatic method selection
config = DensityConfig(
    resolution=100.0,  # 100m grid cells
    method=DensityMethod.AUTO,  # Auto-select optimal method
    max_memory_gb=8.0,
    parallel_workers=4
)

# Initialize engine
engine = SpatialDensityEngine(config)

# Calculate density across three methods:
# - Rasterization: Fastest for most cases
# - Vector Overlay: Highest precision
# - Adaptive Grid: Memory efficient for large areas
result = engine.calculate_density(
    scene_footprints=scenes,
    roi_geometry=roi
)

# Access results
print(f"Method used: {result.method_used.value}")
print(f"Computation time: {result.computation_time:.2f}s")
print(f"Density statistics: {result.stats}")
```

### Metadata Processing
```python
from planetscope_py import MetadataProcessor

processor = MetadataProcessor()

# Extract comprehensive metadata from a scene
metadata = processor.extract_scene_metadata(scene)

# Assess coverage quality for a collection of scenes
assessment = processor.assess_coverage_quality(
    scenes=scenes,
    target_geometry=roi_geometry
)

# Filter scenes based on metadata criteria
filtered_scenes, stats = processor.filter_by_metadata_criteria(
    scenes=scenes,
    criteria={
        "max_cloud_cover": 0.2,
        "min_sun_elevation": 40.0,
        "min_usable_data": 0.85
    }
)
```

### Utility Functions
```python
from planetscope_py.utils import (
    validate_geometry,
    validate_date_range,
    validate_cloud_cover,
    create_point_geometry,
    create_bbox_geometry
)

# Validate GeoJSON geometry
geometry = {
    "type": "Polygon",
    "coordinates": [[[-122.5, 37.7], [-122.3, 37.7], 
                     [-122.3, 37.8], [-122.5, 37.8], [-122.5, 37.7]]]
}
validate_geometry(geometry)

# Validate date range with proper Planet API formatting
start_date, end_date = validate_date_range("2025-01-01", "2025-01-31")

# Validate cloud cover
cloud_cover = validate_cloud_cover(0.15)  # 15%

# Create geometries
point = create_point_geometry(-122.4, 37.75)
bbox = create_bbox_geometry(-122.5, 37.7, -122.3, 37.8)
```

### Exception Handling
```python
from planetscope_py.exceptions import (
    PlanetScopeError,
    AuthenticationError,
    ValidationError,
    ConfigurationError,
    APIError,
    RateLimitError
)

try:
    results = query.search_scenes(geometry, start_date, end_date)
except AuthenticationError as e:
    print(f"Authentication failed: {e.message}")
    print(f"Available methods: {e.details.get('methods', [])}")
except ValidationError as e:
    print(f"Validation error: {e.message}")
    print(f"Details: {e.details}")
except APIError as e:
    print(f"API error: {e.message}")
except RateLimitError as e:
    print(f"Rate limit exceeded: {e.message}")
except ConfigurationError as e:
    print(f"Configuration error: {e.message}")
    print(f"Details: {e.details}")
```

## Testing

### Running Tests
```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage report
python -m pytest tests/ --cov=planetscope_py --cov-report=html

# Run specific test modules
python -m pytest tests/test_auth.py -v
python -m pytest tests/test_config.py -v
python -m pytest tests/test_exceptions.py -v
python -m pytest tests/test_utils.py -v
python -m pytest tests/test_query.py -v
python -m pytest tests/test_metadata.py -v
python -m pytest tests/test_rate_limiter.py -v
python -m pytest tests/test_density_engine.py -v
python -m pytest tests/test_visualization.py -v
```

### Test Coverage
Current test coverage: **280/280 tests passing (100%)**

| Component | Tests | Status |
|-----------|-------|--------|
| Authentication | 24 | All passing |
| Configuration | 21 | All passing |
| Exceptions | 48 | All passing |
| Utilities | 54 | All passing |
| Planet API Query | 45+ | All passing |
| Metadata Processing | 30+ | All passing |
| Rate Limiting | 25+ | All passing |
| Spatial Analysis | 35+ | All passing |
| Basic Visualization | 18+ | All passing |

**Total: 280 tests with 100% coverage**

## Development Roadmap

### Foundation (Complete)
- Robust authentication system with hierarchical API key detection
- Advanced configuration management with environment support
- Comprehensive exception handling with detailed error context
- Complete utility functions with geometry and date validation
- 100% test coverage with professional testing practices
- Security-first design with credential masking and protection

### Planet API Integration (Complete)
- Scene discovery and search capabilities with real API integration
- Metadata extraction and processing with comprehensive analysis
- Rate limiting and request optimization with retry logic
- Response caching and pagination handling
- Advanced filtering and selection tools
- Date formatting with proper Planet API compliance
- Geometry validation for multiple input formats
- Batch operations for multiple geometries
- Quality-based scene filtering
- Preview URL generation
- Comprehensive test suite

### Spatial Analysis Engine (Complete)
- Multi-algorithm spatial density calculations (rasterization, vector overlay, adaptive grid)
- High-resolution analysis support (3m to 1000m grid resolutions)
- Performance optimization with automatic method selection
- Memory efficient processing for large areas
- Statistical analysis and quality metrics

### Basic Visualization (Current)
- GeoTIFF export with QGIS integration
- Summary plots with statistical analysis
- Basic export formats (NumPy, CSV)
- GIS software compatibility

### Advanced Visualization and Export (Planned)
- Interactive web-based mapping and visualization
- Timeline and temporal density plotting
- Enhanced export capabilities (GeoJSON with styling, detailed reports)
- Automated report generation with charts and recommendations
- Dashboard integration for monitoring workflows

## API Reference

### Configuration Defaults
```python
# Planet API Configuration
BASE_URL = "https://api.planet.com/data/v1"
DEFAULT_ITEM_TYPES = ["PSScene"]
DEFAULT_ASSET_TYPES = ["ortho_analytic_4b", "ortho_analytic_4b_xml"]

# Rate Limits (requests per second)
RATE_LIMITS = {
    "search": 10,
    "activate": 5,
    "download": 15,
    "general": 10
}

# Timeout Settings (seconds)
TIMEOUTS = {
    "connect": 10.0,
    "read": 30.0,
    "activation_poll": 300.0,
    "download": 3600.0
}

# Validation Limits
MAX_ROI_AREA_KM2 = 10000
MAX_GEOMETRY_VERTICES = 1000
DEFAULT_CRS = "EPSG:4326"
```

### Exception Hierarchy
```
PlanetScopeError (Base)
├── AuthenticationError     # API key and authentication issues
├── ValidationError        # Input validation failures
├── RateLimitError         # API rate limit exceeded
├── APIError               # Planet API communication errors
├── ConfigurationError     # Configuration file and setup issues
└── AssetError            # Asset activation and download failures
```

## Real-World Examples

### Example 1: Milan Region Analysis
```python
# Search for scenes over Milan region in January 2025
from planetscope_py import PlanetScopeQuery

query = PlanetScopeQuery()

# 100km x 100km area around Milan
milan_geometry = {
    "type": "Polygon", 
    "coordinates": [[
        [8.55, 45.91], [9.83, 45.91], [9.83, 45.02], [8.55, 45.02], [8.55, 45.91]
    ]]
}

results = query.search_scenes(
    geometry=milan_geometry,
    start_date="2025-01-01",
    end_date="2025-01-31", 
    cloud_cover_max=0.2
)

print(f"Found {len(results['features'])} scenes over Milan")
```

### Example 2: High-Resolution Density Analysis
```python
# Perform 3m resolution analysis for detailed coverage mapping
from planetscope_py import SpatialDensityEngine, DensityConfig, DensityMethod
from shapely.geometry import box

roi = box(9.1, 45.4, 9.2, 45.5)  # Small area for high-resolution

config = DensityConfig(
    resolution=3.0,  # 3m grid cells (sub-pixel analysis)
    method=DensityMethod.RASTERIZATION,  # Optimal for high resolution
    max_memory_gb=16.0
)

engine = SpatialDensityEngine(config)
result = engine.calculate_density(scenes, roi)

print(f"High-resolution analysis: {result.grid_info['width']}×{result.grid_info['height']} cells")
print(f"Sub-pixel detail: 3m grid vs ~10m scene resolution")
```

### Example 3: Batch Processing Multiple ROIs
```python
# Process multiple regions simultaneously
regions = [
    {"type": "Point", "coordinates": [9.19, 45.46]},    # Milan
    {"type": "Point", "coordinates": [11.26, 43.77]},   # Florence
    {"type": "Point", "coordinates": [12.49, 41.90]}    # Rome
]

batch_results = query.batch_search(
    geometries=regions,
    start_date="2025-01-01", 
    end_date="2025-01-15"
)

for i, result in enumerate(batch_results):
    if result["success"]:
        scenes = result["result"]["features"]
        print(f"Region {i+1}: {len(scenes)} scenes found")
```

### Example 4: Quality-Based Filtering
```python
# Filter scenes by quality criteria
high_quality_scenes = query.filter_scenes_by_quality(
    scenes=search_results['features'],
    min_quality=0.8,
    max_cloud_cover=0.1,
    exclude_night=True
)

print(f"High quality scenes: {len(high_quality_scenes)}")
```

## Documentation

### Available Resources
- **Complete Wiki**: [GitHub Wiki](https://github.com/Black-Lights/planetscope-py/wiki) with comprehensive guides
- **API Reference**: Complete function and class documentation
- **Getting Started Guide**: Installation and authentication setup
- **Spatial Analysis Guide**: Complete spatial analysis workflows
- **Examples & Tutorials**: Real-world usage examples

### Wiki Documentation Structure
- **Getting Started**: Installation, setup, authentication
- **Core Features**: Scene discovery, metadata analysis, spatial analysis
- **Advanced Features**: Computational methods, performance optimization
- **Examples**: Complete workflow tutorials
- **Reference**: Comprehensive API documentation

## Development

### Code Quality Standards
- **Testing**: Comprehensive test coverage with pytest (280/280 tests passing)
- **Type Hints**: Progressive type annotation implementation with Python 3.10+ support
- **Documentation**: Detailed docstrings and comprehensive README with examples
- **Security**: Credential protection and secure error handling with API key masking
- **Cross-Platform**: Full compatibility with Windows, macOS, and Linux environments
- **Professional Structure**: Modular design with clear separation of concerns

### Contributing
1. Fork the repository on GitHub
2. Create a feature branch: `git checkout -b feature-name`
3. Install development dependencies: `pip install -r requirements-dev.txt`
4. Make changes and add comprehensive tests
5. Ensure all tests pass: `python -m pytest tests/ -v`
6. Run code quality checks: `black . && flake8 . && mypy planetscope_py/`
7. Submit a pull request with detailed description

### Development Setup
```bash
# Clone repository
git clone https://github.com/Black-Lights/planetscope-py.git
cd planetscope-py

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .
pip install -r requirements-dev.txt

# Verify installation
python -m pytest tests/ -v
```

## Requirements

### System Requirements
- Python 3.10 or higher
- Active internet connection for Planet API access
- Valid Planet API key

### Core Dependencies
- requests: HTTP client with session management
- shapely: Geometric operations and validation
- pyproj: Coordinate transformations and CRS handling
- numpy: Numerical computations
- pandas: Data manipulation and analysis
- python-dateutil: Date parsing and operations
- orjson: High-performance JSON processing

### Optional Dependencies (for spatial analysis)
- rasterio: Raster data I/O and processing
- geopandas: Geospatial data analysis
- matplotlib: Plotting and visualization

## Support

- **Issues**: [GitHub Issues](https://github.com/Black-Lights/planetscope-py/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Black-Lights/planetscope-py/discussions)
- **Documentation**: [Project Wiki](https://github.com/Black-Lights/planetscope-py/wiki)

## Citation

If you use this library in your research, please cite:

```bibtex
@software{planetscope_py_2025,
  title = {PlanetScope-py: Professional Python library for PlanetScope satellite imagery analysis},
  author = {Ammar and Umayr},
  year = {2025},
  url = {https://github.com/Black-Lights/planetscope-py},
  note = {Python library for satellite imagery analysis using Planet's Data API}
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Planet Labs PBC** for providing the Planet API and PlanetScope imagery
- **Dr. Daniela Stroppiana** - Project Advisor
- **Prof. Giovanna Venuti** - Project Supervisor
- **Politecnico di Milano** - Geoinformatics Engineering Program

## Authors

**Ammar & Umayr**  
Geoinformatics Engineering Students  
Politecnico di Milano

---

**Note**: This project is independently developed and is not officially affiliated with Planet Labs PBC. It is designed to work with Planet's publicly available APIs following their terms of service.