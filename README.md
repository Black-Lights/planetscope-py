# PlanetScope-py

A professional Python library for PlanetScope satellite imagery analysis, providing comprehensive tools for scene discovery, metadata analysis, spatial-temporal density calculations, asset management, and data export using Planet's Data API.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Library Status](https://img.shields.io/badge/Library%20Status-Production-green.svg)](#current-status)
[![Spatial Analysis](https://img.shields.io/badge/Spatial%20Analysis-Complete-green.svg)](#spatial-analysis-complete)
[![Temporal Analysis](https://img.shields.io/badge/Temporal%20Analysis-Complete-green.svg)](#temporal-analysis-complete)
[![Asset Management](https://img.shields.io/badge/Asset%20Management-Complete-green.svg)](#asset-management-complete)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Status

**Current Status**: Enhanced Metadata Processing & Serialization Fixes  
**Version**: 4.1.0 (Metadata & Serialization Fix Release)  
**Test Coverage**: 349 tests passing                
**API Integration**: Fully functional with real Planet API  
**Spatial Analysis**: Multi-algorithm density calculations with coordinate system fixes  
**Temporal Analysis**: Grid-based temporal pattern analysis with enhanced turbo colormap visualizations  
**Asset Management**: Quota monitoring, downloads, progress tracking  
**GeoPackage Export**: Scene polygons with imagery support  
**Metadata Processing**: Enhanced scene ID extraction from all Planet API endpoints  
**JSON Serialization**: Complete metadata export without truncation  
**Python Support**: 3.10+  
**License**: MIT  

## Key Features (v4.1.0)

### Enhanced Metadata Processing (NEW in v4.1.0)
- **Multi-Source Scene ID Extraction**: Robust scene ID detection from various Planet API response formats
- **Planet API Compatibility**: Works with Search, Stats, Orders, and other API endpoints  
- **Fallback ID Detection**: Comprehensive checking of properties.id, top-level id, item_id, and scene_id fields
- **Complete JSON Export**: Fixed truncated metadata files with proper numpy type serialization
- **Error Recovery**: Graceful handling of missing or malformed scene identifiers

### JSON Serialization Fixes (NEW in v4.1.0)
- **Numpy Type Conversion**: Comprehensive conversion of numpy types to JSON-compatible formats
- **Complete Metadata Export**: No more truncated analysis_metadata.json files
- **Type Safety**: Handles numpy.int64, numpy.float64, numpy.ndarray, and numpy.nan values
- **Nested Object Support**: Recursive conversion of complex data structures
- **Memory Efficiency**: Optimized handling of large arrays during serialization

### Enhanced Temporal Analysis Visualizations (NEW in v4.1.0)
- **Turbo Colormap**: Improved temporal analysis visualizations with better color contrast
- **Consistent Summary Tables**: Temporal summary tables now match spatial density format
- **Professional Presentation**: Enhanced visual consistency across all analysis types
- **Better Data Interpretation**: Improved visibility and data understanding in temporal plots

### File-Based ROI Support
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

### v4.1.0 (2025-06-26) - Metadata & Serialization Fix Release
- **Enhanced scene ID extraction** - Now works with all Planet API response formats (Search, Stats, Orders)
- **Fixed JSON serialization** - Complete metadata export without truncation issues
- **Improved temporal visualizations** - Turbo colormap for better data interpretation
- **Summary table consistency** - Temporal tables now match spatial density format
- **Interactive manager integration** - Enhanced preview and interactive manager configuration

### v4.0.1 (2025-06-25) - Bug Fix Release
- **Fixed critical import issues** - `quick_planet_analysis` and visualization functions now work correctly
- **Resolved module availability detection** - Fixed `_WORKFLOWS_AVAILABLE` and `_VISUALIZATION_AVAILABLE` flags
- **Enhanced error messages** - Clear installation instructions when dependencies are missing
- **Improved debugging** - Added success confirmations for module loading

### v4.0.0 (2025-06-25) - Major Release
- **Complete temporal analysis** with grid-based approach and performance optimization
- **Enhanced spatial analysis** with coordinate system fixes and multi-algorithm support  
- **Advanced data export** with professional GeoPackage creation
- **Asset management** with quota monitoring and async downloads

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

# v4.1.0 Enhancement: Metadata processing now handles all API response formats
for scene in results['features'][:3]:
    metadata = query.metadata_processor.extract_scene_metadata(scene)
    print(f"Scene ID: {metadata['scene_id']}")  # Now works reliably!
```

#### Method 2: File-Based ROI
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

# v4.1.0 Enhancement: Complete metadata export without truncation
print(f"Complete metadata saved to analysis_metadata.json")
```

### Temporal Analysis Engine

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

# v4.1.0 Enhancement: Visualizations now use turbo colormap for better contrast
print("Enhanced temporal visualizations with turbo colormap available")
```

#### Method 2: File-Based ROI with One-Line Function
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
    create_visualizations=True  # Creates comprehensive 4-panel summary with turbo colormap
)

print(f"Found {result['scenes_found']} scenes")
print(f"Mean coverage days: {result['temporal_result'].temporal_stats['mean_coverage_days']:.1f}")
print(f"Computation time: {result['temporal_result'].computation_time:.1f} seconds")
print(f"Output directory: {result['output_directory']}")

# v4.1.0 Enhancement: Complete JSON metadata export
print("Complete temporal analysis metadata exported successfully")
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

# v4.1.0 Enhancement: Enhanced scene metadata in GeoPackage attributes
print("GeoPackage includes enhanced scene metadata from all API sources")
```

### Complete Analysis Workflow
```python
# Complete analysis workflow with all features and v4.1.0 enhancements
from planetscope_py import (
    PlanetScopeQuery, SpatialDensityEngine, TemporalAnalyzer,
    AssetManager, GeoPackageManager
)

async def complete_analysis_workflow():
    # 1. Scene discovery with enhanced metadata processing
    query = PlanetScopeQuery()
    
    # Option A: Use shapefile directly
    roi_shapefile = r'C:\GIS\study_areas\milan_area.shp'
    results = query.search_scenes(
        geometry=roi_shapefile,  # Direct shapefile support
        start_date="2024-01-01",
        end_date="2024-12-31",
        cloud_cover_max=0.3
    )
    
    # v4.1.0: Enhanced metadata processing for all scenes
    print("Processing scene metadata with enhanced ID extraction...")
    valid_scenes = []
    for scene in results['features']:
        metadata = query.metadata_processor.extract_scene_metadata(scene)
        if metadata['scene_id']:  # Now reliably extracts scene IDs
            valid_scenes.append(scene)
    
    print(f"Successfully processed {len(valid_scenes)} scenes with valid metadata")
    
    # 2. Spatial analysis with enhanced JSON export
    spatial_engine = SpatialDensityEngine()
    spatial_result = spatial_engine.calculate_density(valid_scenes, roi_shapefile)
    print("Spatial analysis completed with complete metadata export")
    
    # 3. Temporal analysis with enhanced visualizations
    temporal_result = analyze_roi_temporal_patterns(
        roi_shapefile,  # Same shapefile for consistency
        "2024-01-01/2024-12-31",
        spatial_resolution=100,
        optimization_level="auto",  # Automatic FAST/ACCURATE selection
        clip_to_roi=True,
        create_visualizations=True,  # Enhanced turbo colormap visualizations
        export_geotiffs=True
    )
    print("Temporal analysis completed with enhanced turbo colormap visualizations")
    
    # 4. Asset management (with file-based ROI clipping)
    asset_manager = AssetManager(query.auth)
    quota_info = await asset_manager.get_quota_info()
    
    if quota_info.remaining_area_km2 > 100:  # Check available quota
        downloads = await asset_manager.activate_and_download_assets(
            scenes=valid_scenes[:20],  # Download subset
            clip_to_roi=roi_shapefile  # ROI clipping with file support
        )
    else:
        downloads = None
        print("Insufficient quota for downloads")
    
    # 5. Export to GeoPackage with enhanced metadata
    geopackage_manager = GeoPackageManager()
    geopackage_manager.create_scene_geopackage(
        scenes=valid_scenes,
        output_path="complete_analysis_v4_1_0.gpkg",
        roi=roi_shapefile,  # File-based ROI support
        downloaded_files=downloads
    )
    
    return {
        'scenes': len(valid_scenes),
        'spatial_analysis': spatial_result,
        'temporal_analysis': temporal_result,
        'downloads': len(downloads) if downloads else 0,
        'metadata_enhancements': 'v4.1.0 - Complete scene ID extraction and JSON export'
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

### Planet API Query System (Enhanced in v4.1.0)
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

# v4.1.0: Enhanced metadata extraction works with all API endpoints
for scene in results['features']:
    metadata = query.metadata_processor.extract_scene_metadata(scene)
    print(f"Reliable Scene ID: {metadata['scene_id']}")  # Now always works!

# Get scene statistics
stats = query.get_scene_stats(geometry, "2025-01-01", "2025-01-31")

# Batch search across multiple geometries
batch_results = query.batch_search([geom1, geom2, geom3], "2025-01-01", "2025-01-31")
```

### Spatial Analysis Engine (Enhanced JSON Export in v4.1.0)
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

# v4.1.0: Complete metadata export without truncation
print("Complete analysis metadata exported successfully")

# Performance benchmarks (Milan dataset: 43 scenes, 355 km²)
# - Rasterization: 0.03-0.09s for 100m-30m resolutions
# - Vector Overlay: 53-203s with highest precision
# - Adaptive Grid: 9-15s with memory efficiency
```

### Temporal Analysis Engine (Enhanced Visualizations in v4.1.0)
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

# Export temporal results with enhanced visualizations
analyzer.export_temporal_geotiffs(result, "temporal_analysis", roi)

# v4.1.0: Enhanced turbo colormap visualizations
print("Temporal analysis visualizations now use turbo colormap for better contrast")
```

## Features

### Foundation (Complete)
- **Authentication System**: Hierarchical API key detection with secure credential management
- **Configuration Management**: Multi-source configuration with environment variable support
- **Input Validation**: Comprehensive geometry, date, and parameter validation
- **Exception Handling**: Professional error hierarchy with detailed context and troubleshooting guidance
- **Security**: API key masking, secure session management, and credential protection
- **Cross-Platform**: Full compatibility with Windows, macOS, and Linux environments

### Planet API Integration (Enhanced in v4.1.0)
- **Scene Discovery**: Robust search functionality with advanced filtering capabilities
- **Enhanced Metadata Processing**: Multi-source scene ID extraction from all Planet API endpoints
- **Comprehensive Error Recovery**: Graceful handling of missing or malformed scene identifiers
- **Rate Limiting**: Intelligent rate limiting with exponential backoff and retry logic
- **API Response Handling**: Optimized response caching and pagination support
- **Date Formatting**: Planet API compliant date formatting with end-of-day handling
- **Geometry Validation**: Multi-format geometry support (GeoJSON, Shapely, WKT)
- **Batch Operations**: Support for multiple geometry searches with parallel processing
- **Quality Assessment**: Scene filtering based on cloud cover, sun elevation, and quality metrics
- **Preview Support**: Scene preview URL generation for visual inspection
- **Real-World Testing**: Verified with actual Planet API calls and data retrieval

### Spatial Analysis Engine (Enhanced Export in v4.1.0)
- **Multi-Algorithm Calculation**: Three computational methods (rasterization, vector overlay, adaptive grid)
- **Automatic Method Selection**: Intelligent algorithm selection based on dataset characteristics
- **High-Resolution Analysis**: Support for 3m to 1000m grid resolutions with sub-pixel accuracy
- **Performance Optimization**: Memory-efficient processing with adaptive chunking
- **Coordinate System Fixes**: Proper CRS handling and transformation accuracy
- **Professional Visualization**: Four-panel summary plots with comprehensive statistics
- **Enhanced JSON Export**: Complete metadata files without truncation issues
- **GeoTIFF Export**: GIS-compatible export with automatic QGIS styling
- **Cross-Platform Compatibility**: Standardized grid structures and coordinate handling

### Temporal Analysis (Enhanced Visualizations in v4.1.0)
- **Grid-Based Pattern Analysis**: Temporal analysis using same grid approach as spatial density
- **Multiple Temporal Metrics**: Coverage days, interval statistics, temporal density, frequency
- **Performance Optimization**: FAST (vectorized) and ACCURATE (cell-by-cell) methods
- **Temporal Statistics**: Comprehensive statistical analysis and gap detection
- **Enhanced Visualizations**: Turbo colormap for better data interpretation and contrast
- **Professional Export**: GeoTIFF files with QML styling and complete metadata
- **Consistent Summary Tables**: Temporal summary tables match spatial density format
- **ROI Integration**: Full integration with coordinate system fixes
- **Complete JSON Export**: Fixed serialization issues for comprehensive metadata

### Asset Management (Complete)
- **Intelligent Quota Monitoring**: Real-time tracking of Planet subscription usage
- **Asset Activation & Download**: Automated asset processing with progress tracking
- **Download Management**: Parallel downloads with retry logic and error recovery
- **User Confirmation System**: Interactive prompts for download decisions
- **ROI Clipping Support**: Automatic scene clipping to regions of interest
- **Data Usage Warnings**: Proactive alerts about subscription limits

### GeoPackage Export (Enhanced Metadata in v4.1.0)
- **Professional Scene Polygons**: Comprehensive GeoPackage export with enhanced metadata
- **Multi-Layer Support**: Vector polygons and raster imagery in single file
- **Enhanced Attribute Schema**: Rich metadata tables with reliable scene IDs from all sources
- **GIS Software Integration**: Direct compatibility with QGIS, ArcGIS, and other tools
- **Cross-Platform Standards**: Standardized schemas for maximum compatibility
- **Imagery Integration**: Optional inclusion of downloaded scene imagery

### Visualization and Export (Enhanced in v4.1.0)
- **Enhanced Temporal Visualizations**: Turbo colormap for better data interpretation
- **Professional Visualization**: Multi-panel summary plots with comprehensive statistics
- **Consistent Formatting**: Temporal summary tables match spatial density format
- **Complete JSON Export**: Fixed serialization issues for all metadata files
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

### Update to Latest Version (v4.1.0)
If you're upgrading from previous versions to get the metadata and serialization fixes:
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
| Metadata Processing | 30+ | All passing (Enhanced in v4.1.0) |
| Rate Limiting | 25+ | All passing |
| Spatial Analysis | 35+ | All passing |
| Temporal Analysis | 23 | All passing (Enhanced in v4.1.0) |
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

### Planet API Integration (Enhanced in v4.1.0)
- Full Planet API integration with all major endpoints
- Enhanced scene ID extraction from all Planet API response formats
- Comprehensive metadata processing with error recovery
- Intelligent rate limiting and error recovery
- Real-world testing with actual Planet API data

### Spatial Analysis (Enhanced Export in v4.1.0)
- Multi-algorithm spatial density calculations
- Performance optimization with automatic method selection
- High-resolution analysis capabilities (3m-1000m)
- Professional visualization and export tools
- Complete JSON metadata export without truncation
- Memory-efficient processing for large datasets

### Temporal Analysis (Enhanced Visualizations in v4.1.0)
- Grid-based temporal pattern analysis
- Multiple temporal metrics and statistics
- Performance optimization with FAST/ACCURATE methods
- Enhanced turbo colormap visualizations for better data interpretation
- Consistent summary table formatting with spatial analysis
- Complete JSON metadata export
- Integration with existing spatial analysis framework

### Asset Management (Complete)
- Intelligent quota monitoring and usage tracking
- Automated asset activation and download management
- Parallel downloads with progress tracking and retry logic
- User confirmation workflows with impact assessment
- ROI-based clipping and processing capabilities

### Data Export (Enhanced in v4.1.0)
- Professional GeoPackage creation with enhanced metadata
- Multi-layer support with vector and raster integration
- Enhanced scene ID attributes from all API sources
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

### Enhanced Dependencies (v4.1.0)
- **Spatial Analysis**: rasterio, geopandas (for coordinate fixes and export)
- **Temporal Analysis**: xarray, scipy (for data structures and statistical analysis)
- **Asset Management**: aiohttp (for async downloads)
- **GeoPackage Export**: geopandas, rasterio, fiona (for GIS data export)
- **Visualization**: matplotlib (for plotting and enhanced visualizations)
- **Optional Interactive**: ipywidgets (for Jupyter notebook integration)

## API Reference

### Core Classes
- `PlanetAuth`: Authentication management with multiple methods
- `PlanetScopeQuery`: Scene discovery and enhanced metadata processing
- `SpatialDensityEngine`: Multi-algorithm spatial analysis with complete export
- `TemporalAnalyzer`: Grid-based temporal pattern analysis with enhanced visualizations
- `AssetManager`: Quota monitoring and download management
- `GeoPackageManager`: Professional data export system with enhanced metadata

### Configuration Classes
- `DensityConfig`: Spatial analysis configuration
- `TemporalConfig`: Temporal analysis configuration with visualization options
- `GeoPackageConfig`: Export configuration with enhanced schema support
- `AssetConfig`: Asset management configuration

### Result Classes
- `DensityResult`: Spatial analysis results with complete JSON export
- `TemporalResult`: Temporal analysis results with enhanced metrics and visualizations
- `AssetStatus`: Asset activation and download status
- `QuotaInfo`: Real-time quota information

## Support

- **Issues**: [GitHub Issues](https://github.com/Black-Lights/planetscope-py/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Black-Lights/planetscope-py/discussions)
- **Documentation**: [Project Wiki](https://github.com/Black-Lights/planetscope-py/wiki)

## What's Fixed in v4.1.0

### Critical Metadata Processing Issues
- **Scene ID Extraction**: Now works reliably with all Planet API endpoints (Search, Stats, Orders)
- **Multi-Source ID Detection**: Checks properties.id, top-level id, item_id, and scene_id fields
- **Error Recovery**: Graceful handling of missing or malformed scene identifiers
- **API Compatibility**: Enhanced compatibility across different Planet API response formats

### JSON Serialization Problems
- **Complete Metadata Export**: Fixed truncated analysis_metadata.json files
- **Numpy Type Conversion**: Proper handling of numpy.int64, numpy.float64, numpy.ndarray types
- **Memory Efficiency**: Optimized serialization of large data structures
- **Nested Object Support**: Recursive conversion of complex metadata dictionaries

### Temporal Analysis Enhancements
- **Turbo Colormap**: Improved visualization contrast and data interpretation
- **Summary Table Consistency**: Temporal tables now match spatial density format
- **Professional Presentation**: Enhanced visual consistency across analysis types
- **Better Color Schemes**: Standardized color palettes for all visualizations

### Integration Improvements
- **Interactive Manager**: Enhanced preview and interactive manager configuration
- **Module Loading**: Improved module availability detection and reporting
- **Error Messages**: Clear feedback about component status and missing dependencies

## Citation

If you use this library in your research, please cite:

```bibtex
@software{planetscope_py_2025,
  title = {PlanetScope-py: Professional Python library for PlanetScope satellite imagery analysis},
  author = {Ammar and Umayr},
  year = {2025},
  version = {4.1.0},
  url = {https://github.com/Black-Lights/planetscope-py},
  note = {Enhanced metadata processing and serialization fixes with complete temporal analysis}
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
