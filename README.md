# PlanetScope-py

A professional Python library for PlanetScope satellite imagery analysis, providing comprehensive tools for scene discovery, metadata analysis, spatial-temporal density calculations, asset management, and data export using Planet's Data API.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Library Status](https://img.shields.io/badge/Library%20Status-Production-green.svg)](#current-status)
[![Spatial Analysis](https://img.shields.io/badge/Spatial%20Analysis-Complete-green.svg)](#spatial-analysis-engine-complete)
[![Temporal Analysis](https://img.shields.io/badge/Temporal%20Analysis-Complete-green.svg)](#temporal-analysis-complete)
[![Asset Management](https://img.shields.io/badge/Asset%20Management-Complete-green.svg)](#asset-management-complete)
[![Test Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen.svg)](#testing)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Status

**Current Status**: Enhanced Temporal Analysis & Asset Management  
**Version**: 4.0.0a1  
**Test Coverage**: 349 tests passing (100%)  
**API Integration**: Fully functional with real Planet API  
**Spatial Analysis**: Multi-algorithm density calculations complete  
**Temporal Analysis**: 3D data cubes, seasonal patterns, gap analysis  
**Asset Management**: Quota monitoring, downloads, progress tracking  
**GeoPackage Export**: Scene polygons with imagery support  
**Python Support**: 3.10+  
**License**: MIT  

## Overview

PlanetScope-py is designed for remote sensing researchers, GIS analysts, and Earth observation professionals who need reliable tools for working with PlanetScope satellite imagery. The library provides a robust foundation for scene inventory management, sophisticated spatial-temporal analysis workflows, and professional data export capabilities.

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

### Temporal Analysis (Complete)
- **3D Spatiotemporal Data Cubes**: Multi-dimensional analysis with (lat, lon, time) dimensions
- **Seasonal Pattern Detection**: Automated identification of acquisition patterns and seasonal trends
- **Temporal Gap Analysis**: Detection and analysis of coverage gaps with severity assessment
- **Time Series Analytics**: Comprehensive temporal statistics and trend analysis
- **Temporal Resolution Support**: Configurable analysis from daily to annual scales
- **Cross-Platform Grid Compatibility**: Standardized temporal data structures

### Asset Management (Complete)
- **Intelligent Quota Monitoring**: Real-time tracking of Planet subscription usage
- **Asset Activation & Download**: Automated asset processing with progress tracking
- **Download Management**: Parallel downloads with retry logic and error recovery
- **User Confirmation System**: Interactive prompts for download decisions
- **ROI Clipping Support**: Automatic scene clipping to regions of interest
- **Data Usage Warnings**: Proactive alerts about subscription limits

### GeoPackage Export (Complete)
- **Professional Scene Polygons**: Comprehensive GeoPackage export with full metadata
- **Multi-Layer Support**: Vector polygons and raster imagery in single file
- **Comprehensive Attribute Schema**: Rich metadata tables with quality metrics
- **GIS Software Integration**: Direct compatibility with QGIS, ArcGIS, and other tools
- **Cross-Platform Standards**: Standardized schemas for maximum compatibility
- **Imagery Integration**: Optional inclusion of downloaded scene imagery

### Basic Visualization and Export (Current)
- **GeoTIFF Export**: GIS-compatible files with automatic QGIS styling
- **Summary Plots**: Multi-panel density visualization with statistics
- **Data Format Support**: Export to NumPy arrays, CSV, and GeoPackage files
- **Statistical Analysis**: Core density statistics and quality metrics
- **GIS Compatibility**: Direct compatibility with QGIS and ArcGIS

## Installation

### Standard Installation
```bash
# Development installation (recommended)
git clone https://github.com/Black-Lights/planetscope-py.git
cd planetscope-py

# Creating a virtual environment
python -m venv planetscope_env
.\planetscope_env\Scripts\Activate.ps1 # Windows
pip install wheel setuptools
python.exe -m pip install --upgrade pip

# Install enhanced dependencies for full functionality
pip install -r requirements-dev.txt
pip install -e . 
```

### Development Installation
```bash
# Clone the repository
git clone https://github.com/Black-Lights/planetscope-py.git
cd planetscope-py

# Create and activate a virtual environment (recommended)
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

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

### Temporal Analysis
```python
from planetscope_py import TemporalAnalyzer, TemporalConfig, TemporalResolution

# Initialize temporal analyzer
temporal_config = TemporalConfig(
    temporal_resolution=TemporalResolution.WEEKLY,
    spatial_resolution=100.0  # 100m spatial grid
)
temporal_analyzer = TemporalAnalyzer(temporal_config)

# Create 3D spatiotemporal data cube
datacube = temporal_analyzer.create_spatiotemporal_datacube(
    scenes=results['features'],
    roi=roi,
    start_date="2024-01-01",
    end_date="2024-12-31"
)

# Analyze acquisition patterns
patterns = temporal_analyzer.analyze_acquisition_patterns(datacube)
print(f"Found {len(patterns['seasonal_patterns'])} seasonal patterns")

# Detect temporal gaps
gaps = temporal_analyzer.detect_temporal_gaps(datacube)
print(f"Detected {len(gaps)} temporal gaps")
```

### Asset Management and Downloads
```python
from planetscope_py import AssetManager

# Initialize asset manager
asset_manager = AssetManager(query.auth)

# Check current quota usage
quota_info = await asset_manager.get_quota_info()
print(f"Current usage: {quota_info.used_area_km2:.1f} / {quota_info.limit_area_km2} km²")

# Select scenes for download (with user confirmation)
selected_scenes = results['features'][:10]  # First 10 scenes

# Calculate download impact
download_info = asset_manager.calculate_download_impact(selected_scenes, roi)
print(f"Download will use: {download_info['area_km2']:.1f} km²")

# Download assets with progress tracking
if download_info['area_km2'] < quota_info.remaining_area_km2:
    downloads = await asset_manager.activate_and_download_assets(
        scenes=selected_scenes,
        asset_types=["ortho_analytic_4b"],
        clip_to_roi=roi  # Optional ROI clipping
    )
    print(f"Downloaded {len(downloads)} assets")
```

### GeoPackage Export
```python
from planetscope_py import GeoPackageManager, GeoPackageConfig

# Configure GeoPackage export
geopackage_config = GeoPackageConfig(
    include_imagery=True,      # Include downloaded imagery
    clip_to_roi=True,         # Clip images to ROI
    attribute_schema="comprehensive"  # Full metadata attributes
)

# Initialize GeoPackage manager
geopackage_manager = GeoPackageManager(config=geopackage_config)

# Create comprehensive GeoPackage
output_path = "milan_analysis.gpkg"
layer_info = geopackage_manager.create_scene_geopackage(
    scenes=results['features'],
    output_path=output_path,
    roi=roi,
    downloaded_files=downloads if 'downloads' in locals() else None
)

print(f"Created GeoPackage: {output_path}")
print(f"Vector layer: {layer_info.feature_count} scene polygons")
if geopackage_config.include_imagery:
    print(f"Raster layers: Included downloaded imagery")
```

### Complete Analysis Workflow
```python
# Complete analysis workflow with all features
from planetscope_py import (
    PlanetScopeQuery, SpatialDensityEngine, TemporalAnalyzer,
    AssetManager, GeoPackageManager
)

async def complete_analysis_workflow():
    # 1. Scene discovery
    query = PlanetScopeQuery()
    results = query.search_scenes(
        geometry=milan_geometry,
        start_date="2024-01-01",
        end_date="2024-12-31",
        cloud_cover_max=0.3
    )
    
    # 2. Spatial analysis
    spatial_engine = SpatialDensityEngine()
    spatial_result = spatial_engine.calculate_density(results['features'], roi)
    
    # 3. Temporal analysis
    temporal_analyzer = TemporalAnalyzer()
    datacube = temporal_analyzer.create_spatiotemporal_datacube(results['features'], roi)
    temporal_patterns = temporal_analyzer.analyze_acquisition_patterns(datacube)
    
    # 4. Asset management (with user confirmation)
    asset_manager = AssetManager(query.auth)
    quota_info = await asset_manager.get_quota_info()
    
    if quota_info.remaining_area_km2 > 100:  # Check available quota
        downloads = await asset_manager.activate_and_download_assets(
            scenes=results['features'][:20],  # Download subset
            clip_to_roi=roi
        )
    else:
        downloads = None
        print("Insufficient quota for downloads")
    
    # 5. Export to GeoPackage
    geopackage_manager = GeoPackageManager()
    geopackage_manager.create_scene_geopackage(
        scenes=results['features'],
        output_path="complete_analysis.gpkg",
        roi=roi,
        downloaded_files=downloads
    )
    
    return {
        'scenes': len(results['features']),
        'spatial_analysis': spatial_result,
        'temporal_patterns': temporal_patterns,
        'downloads': len(downloads) if downloads else 0
    }

# Run complete workflow
# results = await complete_analysis_workflow()
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
```

### Planet API Query System
```python
from planetscope_py import PlanetScopeQuery

query = PlanetScopeQuery()

# Advanced scene search with comprehensive filtering
results = query.search_scenes(
    geometry=geometry,
    start_date="2025-01-01",
    end_date="2025-01-31",
    cloud_cover_max=0.2,
    sun_elevation_min=30,
    item_types=["PSScene"]
)

# Get scene statistics
stats = query.get_scene_stats(geometry, "2025-01-01", "2025-01-31")

# Batch search across multiple geometries
batch_results = query.batch_search([geom1, geom2, geom3], "2025-01-01", "2025-01-31")
```

### Spatial Analysis Engine
```python
from planetscope_py import SpatialDensityEngine, DensityConfig, DensityMethod

# Configure analysis with automatic method selection
config = DensityConfig(
    resolution=100.0,  # 100m grid cells
    method=DensityMethod.AUTO,  # Auto-select optimal method
    max_memory_gb=8.0,
    parallel_workers=4
)

engine = SpatialDensityEngine(config)
result = engine.calculate_density(scene_footprints=scenes, roi_geometry=roi)
```

### Temporal Analysis Engine
```python
from planetscope_py import TemporalAnalyzer, TemporalConfig, TemporalResolution

# Configure temporal analysis
config = TemporalConfig(
    temporal_resolution=TemporalResolution.MONTHLY,
    spatial_resolution=500.0,
    enable_gap_analysis=True
)

analyzer = TemporalAnalyzer(config)

# Create data cube and analyze patterns
datacube = analyzer.create_spatiotemporal_datacube(scenes, roi)
patterns = analyzer.analyze_acquisition_patterns(datacube)
gaps = analyzer.detect_temporal_gaps(datacube)
recommendations = analyzer.generate_acquisition_recommendations(datacube)
```

## Testing

### Running Tests
```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage report
python -m pytest tests/ --cov=planetscope_py --cov-report=html

# Run specific component tests
python -m pytest tests/test_temporal_analysis.py -v
python -m pytest tests/test_asset_manager.py -v
python -m pytest tests/test_geopackage_manager.py -v
```

### Test Coverage
Current test coverage: **349 tests passing (100%)**

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
| Temporal Analysis | 23 | All passing |
| Asset Management | 23 | All passing |
| GeoPackage Export | 21 | All passing |

**Total: 349 tests with 100% success rate**

## Development Roadmap

### Foundation (Complete)
- Robust authentication system with hierarchical API key detection
- Advanced configuration management with environment support
- Comprehensive exception handling with detailed error context
- Complete utility functions with geometry and date validation
- Security-first design with credential masking and protection

### Planet API Integration (Complete)
- Scene discovery and search capabilities with real API integration
- Metadata extraction and processing with comprehensive analysis
- Rate limiting and request optimization with retry logic
- Response caching and pagination handling
- Advanced filtering and selection tools
- Comprehensive test suite

### Spatial Analysis Engine (Complete)
- Multi-algorithm spatial density calculations (rasterization, vector overlay, adaptive grid)
- High-resolution analysis support (3m to 1000m grid resolutions)
- Performance optimization with automatic method selection
- Memory efficient processing for large areas

### Enhanced Temporal Analysis & Asset Management (Current - Complete)
- Temporal Analysis: 3D spatiotemporal data cubes, seasonal patterns, gap analysis
- Asset Management: Quota monitoring, intelligent downloads, progress tracking
- GeoPackage Export: Professional scene polygons with imagery integration
- Interactive Controls: User confirmations, progress bars, workflow management

### Future Enhancements (Planned)
- **Advanced Visualization**: Interactive web-based mapping and visualization
- **Cloud Integration**: Cloud-based processing for large-scale analysis
- **Machine Learning**: Predictive modeling for acquisition planning
- **API Endpoints**: RESTful API for web service integration

## API Reference

### Configuration Defaults
```python
# Planet API Configuration
BASE_URL = "https://api.planet.com/data/v1"
DEFAULT_ITEM_TYPES = ["PSScene"]
DEFAULT_ASSET_TYPES = ["ortho_analytic_4b", "ortho_analytic_4b_xml"]

# Enhanced timeouts and limits
TIMEOUTS = {
    "connect": 10.0,
    "read": 30.0,
    "activation_poll": 300.0,
    "download": 3600.0
}

# Validation Limits
MAX_ROI_AREA_KM2 = 10000
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

### Enhanced Dependencies
- **Temporal Analysis**: xarray, scipy (for data cubes and analysis)
- **Asset Management**: aiohttp, asyncio (for async downloads)
- **GeoPackage Export**: geopandas, rasterio, fiona (for GIS data export)
- **Optional Interactive**: ipywidgets (for Jupyter notebook integration)

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
  version = {4.0.0},
  url = {https://github.com/Black-Lights/planetscope-py},
  note = {Phase 4: Enhanced temporal analysis, asset management, and data export capabilities}
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
