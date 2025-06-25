# PlanetScope-py

A professional Python library for PlanetScope satellite imagery analysis, providing comprehensive tools for scene discovery, metadata analysis, spatial-temporal density calculations, asset management, and data export using Planet's Data API.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Library Status](https://img.shields.io/badge/Library%20Status-Production-green.svg)](#current-status)
[![Spatial Analysis](https://img.shields.io/badge/Spatial%20Analysis-Complete-green.svg)](#spatial-analysis-complete)
[![Temporal Analysis](https://img.shields.io/badge/Temporal%20Analysis-Complete-green.svg)](#temporal-analysis-complete)
[![Asset Management](https://img.shields.io/badge/Asset%20Management-Complete-green.svg)](#asset-management-complete)
[![Test Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen.svg)](#testing)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Status

**Current Status**: Complete Temporal Analysis & Advanced Data Management  
**Version**: 4.0.1 (Bug Fix Release)  
**Test Coverage**: 349 tests passing (100%)  
**API Integration**: Fully functional with real Planet API  
**Spatial Analysis**: Multi-algorithm density calculations with coordinate system fixes  
**Temporal Analysis**: Grid-based temporal pattern analysis with performance optimization  
**Asset Management**: Quota monitoring, downloads, progress tracking  
**GeoPackage Export**: Scene polygons with imagery support  
**Import Issues**: Fixed in v4.0.1 (workflow and visualization modules now work correctly)  
**Python Support**: 3.10+  
**License**: MIT  

## Key Features (v4.0.0)

### File-Based ROI Support (NEW in v4.0.0)
- **Shapefile Input**: Direct `.shp` file support with automatic CRS reprojection to WGS84
- **GeoJSON Files**: Support for `.geojson` file input with FeatureCollection handling  
- **WKT Support**: WKT string input and `.wkt` file support
- **Multi-Feature Handling**: Automatic union of multiple features in shapefiles and GeoJSON files
- **Universal Compatibility**: All analysis functions support file-based ROI input

### Complete Temporal Analysis Engine
- **Grid-Based Temporal Pattern Analysis**: Complete temporal analysis using coordinate-corrected grid approach
- **Multiple Temporal Metrics**: Coverage days, mean/median intervals, temporal density, and frequency analysis
- **Performance Optimization**: FAST and ACCURATE methods with automatic selection
- **Professional Outputs**: GeoTIFF export with QGIS styling and comprehensive metadata
- **ROI Integration**: Full integration with coordinate system fixes and flexible input formats

### Enhanced Spatial Analysis Engine
- **Multi-Algorithm Density Calculation**: Rasterization, vector overlay, and adaptive grid methods
- **Automatic Method Selection**: Intelligent algorithm selection based on dataset characteristics
- **High-Resolution Support**: 3m to 1000m grid resolutions with sub-pixel accuracy
- **Performance Optimization**: Memory-efficient processing with configurable limits
- **Coordinate System Fixes**: Proper north-to-south orientation with corrected transforms

### Advanced Asset Management
- **Intelligent Quota Monitoring**: Real-time tracking of Planet subscription usage
- **Async Download Management**: Parallel downloads with retry logic and progress tracking
- **ROI Clipping Integration**: Automatic scene clipping during download process
- **User Confirmation System**: Interactive prompts with quota impact calculations
- **Download Verification**: Integrity checking for downloaded assets

### Professional Data Export
- **GeoPackage Creation**: Comprehensive GeoPackage files with metadata integration
- **Multi-Layer Support**: Vector polygons and raster imagery in standardized files
- **GIS Software Integration**: Direct compatibility with QGIS, ArcGIS, and other tools
- **Comprehensive Metadata**: Rich attribute schemas with quality metrics
- **Flexible Schemas**: Support for minimal, standard, and comprehensive attribute schemas

## Overview

PlanetScope-py is designed for remote sensing researchers, GIS analysts, and Earth observation professionals who need reliable tools for working with PlanetScope satellite imagery. The library provides a robust foundation for scene inventory management, sophisticated spatial-temporal analysis workflows, and professional data export capabilities.

## Recent Updates

### v4.0.1 (2025-06-25) - Bug Fix Release
- 🐛 **Fixed critical import issues** - `quick_planet_analysis` and visualization functions now work correctly
- ✅ **Resolved module availability detection** - Fixed `_WORKFLOWS_AVAILABLE` and `_VISUALIZATION_AVAILABLE` flags
- 📋 **Enhanced error messages** - Clear installation instructions when dependencies are missing
- 🔧 **Improved debugging** - Added success confirmations for module loading

### v4.0.0 (2025-06-25) - Major Release
- 🆕 **Complete temporal analysis** with grid-based approach and performance optimization
- 📊 **Enhanced spatial analysis** with coordinate system fixes and multi-algorithm support  
- 🗃️ **Advanced data export** with professional GeoPackage creation
- ⚡ **Asset management** with quota monitoring and async downloads

## Quick Start

### Basic Scene Search

#### Method 1: Direct Geometry Definition
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

#### Method 2: File-Based ROI (NEW in v4.0.0)
```python
from planetscope_py import PlanetScopeQuery

# Initialize query system (automatically detects API key)
query = PlanetScopeQuery()

# Use shapefile directly as ROI
results = query.search_scenes(
    geometry=r'C:\path\to\study_area.shp',  # Direct file path support
    start_date="2025-01-01",
    end_date="2025-01-31",
    cloud_cover_max=0.2,  # 20% maximum cloud cover
    item_types=["PSScene"]
)

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

### Temporal Analysis Engine (NEW in v4.0.0)

#### Method 1: Direct Geometry Definition
```python
from planetscope_py import TemporalAnalyzer, TemporalConfig, TemporalMetric

# Configure temporal analysis
config = TemporalConfig(
    spatial_resolution=100.0,
    metrics=[TemporalMetric.COVERAGE_DAYS, TemporalMetric.MEAN_INTERVAL],
    optimization_method="auto"  # FAST or ACCURATE method selection
)

analyzer = TemporalAnalyzer(config)

# Analyze temporal patterns
temporal_result = analyzer.analyze_temporal_patterns(
    scene_footprints=results['features'],
    roi_geometry=roi,
    start_date="2025-01-01",
    end_date="2025-01-31"
)

print(f"Analysis completed in {temporal_result.computation_time:.1f} seconds")
print(f"Mean coverage days: {temporal_result.temporal_stats['mean_coverage_days']:.1f}")
print(f"Temporal metrics calculated: {len(temporal_result.metric_arrays)}")
```

#### Method 2: File-Based ROI with One-Line Function (NEW in v4.0.0)
```python
from planetscope_py import analyze_roi_temporal_patterns

# Complete temporal analysis with shapefile input
result = analyze_roi_temporal_patterns(
    r'C:\path\to\milan_roi.shp',  # Shapefile input
    "2025-01-01/2025-03-31",
    spatial_resolution=500,
    optimization_level="fast",  # Use FAST vectorized method
    clip_to_roi=True,
    cloud_cover_max=0.3,
    create_visualizations=True  # Creates comprehensive 4-panel summary
)

print(f"Found {result['scenes_found']} scenes")
print(f"Mean coverage days: {result['temporal_result'].temporal_stats['mean_coverage_days']:.1f}")
print(f"Computation time: {result['temporal_result'].computation_time:.1f} seconds")
print(f"Output directory: {result['output_directory']}")
```

### Asset Management
```python
from planetscope_py import AssetManager

# Initialize asset manager
asset_manager = AssetManager()

# Check quota information
quota_info = await asset_manager.get_quota_info()
print(f"Available area: {quota_info.remaining_area_km2:.1f} km²")

# Download assets with ROI clipping
if quota_info.remaining_area_km2 > 100:
    downloads = await asset_manager.activate_and_download_assets(
        scenes=results['features'][:10],
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
    # 1. Scene discovery with multiple ROI input options
    query = PlanetScopeQuery()
    
    # Option A: Use shapefile directly
    roi_shapefile = r'C:\GIS\study_areas\milan_area.shp'
    results = query.search_scenes(
        geometry=roi_shapefile,  # Direct shapefile support
        start_date="2024-01-01",
        end_date="2024-12-31",
        cloud_cover_max=0.3
    )
    
    # Option B: Use traditional geometry (alternative)
    # results = query.search_scenes(
    #     geometry=milan_geometry,
    #     start_date="2024-01-01",
    #     end_date="2024-12-31",
    #     cloud_cover_max=0.3
    # )
    
    # 2. Spatial analysis with file-based ROI
    spatial_engine = SpatialDensityEngine()
    spatial_result = spatial_engine.calculate_density(results['features'], roi_shapefile)
    
    # 3. Temporal analysis with file-based ROI (one-line function)
    temporal_result = analyze_roi_temporal_patterns(
        roi_shapefile,  # Same shapefile for consistency
        "2024-01-01/2024-12-31",
        spatial_resolution=100,
        optimization_level="auto",  # Automatic FAST/ACCURATE selection
        clip_to_roi=True,
        create_visualizations=True,
        export_geotiffs=True
    )
    
    # 4. Asset management (with file-based ROI clipping)
    asset_manager = AssetManager(query.auth)
    quota_info = await asset_manager.get_quota_info()
    
    if quota_info.remaining_area_km2 > 100:  # Check available quota
        downloads = await asset_manager.activate_and_download_assets(
            scenes=results['features'][:20],  # Download subset
            clip_to_roi=roi_shapefile  # ROI clipping with file support
        )
    else:
        downloads = None
        print("Insufficient quota for downloads")
    
    # 5. Export to GeoPackage with file-based ROI
    geopackage_manager = GeoPackageManager()
    geopackage_manager.create_scene_geopackage(
        scenes=results['features'],
        output_path="complete_analysis.gpkg",
        roi=roi_shapefile,  # File-based ROI support
        downloaded_files=downloads
    )
    
    return {
        'scenes': len(results['features']),
        'spatial_analysis': spatial_result,
        'temporal_analysis': temporal_result,
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

# Performance benchmarks (Milan dataset: 43 scenes, 355 km²)
# - Rasterization: 0.03-0.09s for 100m-30m resolutions
# - Vector Overlay: 53-203s with highest precision
# - Adaptive Grid: 9-15s with memory efficiency
```

### Temporal Analysis Engine
```python
from planetscope_py import TemporalAnalyzer, TemporalConfig, TemporalMetric

# Configure temporal analysis
config = TemporalConfig(
    spatial_resolution=100.0,
    metrics=[
        TemporalMetric.COVERAGE_DAYS,
        TemporalMetric.MEAN_INTERVAL,
        TemporalMetric.TEMPORAL_DENSITY
    ],
    optimization_method="auto"
)

analyzer = TemporalAnalyzer(config)
result = analyzer.analyze_temporal_patterns(scenes, roi, start_date, end_date)

# Export temporal results
analyzer.export_temporal_geotiffs(result, "temporal_analysis", roi)
```

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
- **Multi-Algorithm Calculation**: Three computational methods (rasterization, vector overlay, adaptive grid)
- **Automatic Method Selection**: Intelligent algorithm selection based on dataset characteristics
- **High-Resolution Analysis**: Support for 3m to 1000m grid resolutions with sub-pixel accuracy
- **Performance Optimization**: Memory-efficient processing with adaptive chunking
- **Coordinate System Fixes**: Proper CRS handling and transformation accuracy
- **Professional Visualization**: Four-panel summary plots with comprehensive statistics
- **GeoTIFF Export**: GIS-compatible export with automatic QGIS styling
- **Cross-Platform Compatibility**: Standardized grid structures and coordinate handling

### Temporal Analysis (Complete - NEW in v4.0.0)
- **Grid-Based Pattern Analysis**: Temporal analysis using same grid approach as spatial density
- **Multiple Temporal Metrics**: Coverage days, interval statistics, temporal density, frequency
- **Performance Optimization**: FAST (vectorized) and ACCURATE (cell-by-cell) methods
- **Temporal Statistics**: Comprehensive statistical analysis and gap detection
- **Professional Export**: GeoTIFF files with QML styling and metadata
- **ROI Integration**: Full integration with coordinate system fixes
- **Visualization**: Four-panel summary plots and comprehensive reporting

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

### Visualization and Export
- **Professional Visualization**: Multi-panel summary plots with comprehensive statistics
- **GeoTIFF Export**: GIS-compatible export with automatic QGIS styling
- **Statistical Analysis**: Comprehensive statistics for all analysis types
- **Multiple Export Formats**: NumPy arrays, CSV, and GeoPackage formats
- **Cross-Platform Standards**: Standardized file formats and metadata schemas

## Installation

### Standard Installation
```bash
pip install planetscope-py
```

### Enhanced Installation (with all optional features)
```bash
pip install planetscope-py[all]
```

### Bug Fix Update (v4.0.1)
If you're experiencing import issues with `quick_planet_analysis` or visualization functions:
```bash
pip install --upgrade planetscope-py
```

### Development Installation
```bash
# Clone the repository
git clone https://github.com/Black-Lights/planetscope-py.git
cd planetscope-py

# Create virtual environment
python -m venv planetscope_env
source planetscope_env/bin/activate  # Linux/macOS
# or
planetscope_env\Scripts\activate     # Windows

# Install development dependencies
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
- Cross-platform compatibility testing and validation

### Planet API Integration (Complete)
- Full Planet API integration with all major endpoints
- Advanced scene search with sophisticated filtering
- Comprehensive metadata processing and quality assessment
- Intelligent rate limiting and error recovery
- Real-world testing with actual Planet API data

### Spatial Analysis (Complete)
- Multi-algorithm spatial density calculations
- Performance optimization with automatic method selection
- High-resolution analysis capabilities (3m-1000m)
- Professional visualization and export tools
- Memory-efficient processing for large datasets

### Temporal Analysis (Complete - v4.0.0)
- Grid-based temporal pattern analysis
- Multiple temporal metrics and statistics
- Performance optimization with FAST/ACCURATE methods
- Professional visualization and export capabilities
- Integration with existing spatial analysis framework

### Asset Management (Complete)
- Intelligent quota monitoring and usage tracking
- Automated asset activation and download management
- Parallel downloads with progress tracking and retry logic
- User confirmation workflows with impact assessment
- ROI-based clipping and processing capabilities

### Data Export (Complete)
- Professional GeoPackage creation with comprehensive metadata
- Multi-layer support with vector and raster integration
- GIS software compatibility and styling
- Flexible schema support for different use cases
- Cross-platform file format standards

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

### Enhanced Dependencies (v4.0.0)
- **Spatial Analysis**: rasterio, geopandas (for coordinate fixes and export)
- **Temporal Analysis**: xarray, scipy (for data structures and statistical analysis)
- **Asset Management**: aiohttp (for async downloads)
- **GeoPackage Export**: geopandas, rasterio, fiona (for GIS data export)
- **Visualization**: matplotlib (for plotting and visualization)
- **Optional Interactive**: ipywidgets (for Jupyter notebook integration)

## API Reference

### Core Classes
- `PlanetAuth`: Authentication management with multiple methods
- `PlanetScopeQuery`: Scene discovery and metadata processing
- `SpatialDensityEngine`: Multi-algorithm spatial analysis
- `TemporalAnalyzer`: Grid-based temporal pattern analysis
- `AssetManager`: Quota monitoring and download management
- `GeoPackageManager`: Professional data export system

### Configuration Classes
- `DensityConfig`: Spatial analysis configuration
- `TemporalConfig`: Temporal analysis configuration
- `GeoPackageConfig`: Export configuration
- `AssetConfig`: Asset management configuration

### Result Classes
- `DensityResult`: Spatial analysis results with statistics
- `TemporalResult`: Temporal analysis results with metrics
- `AssetStatus`: Asset activation and download status
- `QuotaInfo`: Real-time quota information

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
  note = {Complete temporal analysis and advanced data management capabilities}
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