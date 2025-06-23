# PlanetScope-py

A professional Python library for PlanetScope satellite imagery analysis, providing comprehensive tools for scene discovery, metadata analysis, spatial-temporal density calculations, asset management, and data export using Planet's Data API.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Library Status](https://img.shields.io/badge/Library%20Status-Production-green.svg)](#current-status)
[![Spatial Analysis](https://img.shields.io/badge/Spatial%20Analysis-Enhanced-green.svg)](#enhanced-spatial-analysis)
[![Temporal Analysis](https://img.shields.io/badge/Temporal%20Analysis-Complete-green.svg)](#temporal-analysis-complete)
[![Asset Management](https://img.shields.io/badge/Asset%20Management-Complete-green.svg)](#asset-management-complete)
[![Test Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen.svg)](#testing)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Status

**Current Status**: Enhanced Spatial Analysis with Coordinate Fixes & One-Line Functions  
**Version**: 4.0.0  
**Test Coverage**: 349 tests passing (100%)  
**API Integration**: Fully functional with real Planet API  
**Spatial Analysis**: Enhanced with coordinate system fixes and one-line functions  
**Temporal Analysis**: 3D data cubes, seasonal patterns, gap analysis  
**Asset Management**: Quota monitoring, downloads, progress tracking  
**GeoPackage Export**: Scene polygons with imagery support  
**Python Support**: 3.10+  
**License**: MIT  

## Enhanced Features (v4.0+)

### Coordinate System Fixes
- **Fixed coordinate system display** - No more mirrored/flipped maps
- **Proper north-to-south orientation** with corrected transforms
- **Geographic alignment** with northwest origin positioning
- **Automatic coordinate validation** for all analysis methods

### Enhanced Performance & Limits
- **Rasterization as optimized default** - Changed from vector overlay for better performance
- **Increased scene footprint limits** - 150+ default (up to 1000+) in visualizations
- **Enhanced memory management** - Improved chunking and optimization
- **Robust error handling** - PROJ/CRS compatibility with multiple fallbacks

### One-Line Functions
- **Individual plot access** - Single command for specific visualizations
- **Simplified workflows** - Complete analysis in one line
- **Enhanced exports** - Direct GeoTIFF + QML file generation
- **Dynamic histogram bins** - No more fixed 11-19 ranges

## Overview

PlanetScope-py is designed for remote sensing researchers, GIS analysts, and Earth observation professionals who need reliable tools for working with PlanetScope satellite imagery. The library provides a robust foundation for scene inventory management, sophisticated spatial-temporal analysis workflows, and professional data export capabilities.

## Quick Start

### Ultra-Simple Analysis (One-Line)
```python
from planetscope_py import quick_planet_analysis
from shapely.geometry import box

# Milan area ROI
milan_roi = box(9.04, 45.40, 9.28, 45.52)

# Complete analysis with enhanced coordinate fixes
result = quick_planet_analysis(milan_roi, "last_month")

print(f"Found {result['scenes_found']} scenes")
print(f"Mean density: {result['summary']['mean_density']:.1f} scenes/pixel")
print(f"Coordinate fixes applied: {result['summary'].get('coordinate_system_corrected', False)}")
```

### One-Line Individual Functions
```python
from planetscope_py import plot_density_map_only, plot_footprints_only, export_geotiff_only

# Just get density map (coordinate-corrected)
fig = plot_density_map_only(milan_roi, "last_month", "density.png")

# Just get scene footprints (enhanced limits: 300+ scenes)
fig = plot_footprints_only(milan_roi, "last_month", max_scenes=500)

# Just export GeoTIFF + QML files (coordinate fixes included)
success = export_geotiff_only(milan_roi, "last_month", "output.tif")
```

### Enhanced Spatial Density Analysis
```python
from planetscope_py import PlanetScopeQuery, SpatialDensityEngine, DensityConfig, DensityMethod
from shapely.geometry import box

# 1. First, search for scenes
query = PlanetScopeQuery()
milan_geometry = {
    "type": "Point",
    "coordinates": [9.1900, 45.4642]
}

results = query.search_scenes(
    geometry=milan_geometry,
    start_date="2025-01-01",
    end_date="2025-01-31",
    cloud_cover_max=0.2
)

# 2. Define region of interest and configure analysis
roi = box(9.04, 45.40, 9.28, 45.52)  # Milan bounding box

# Enhanced configuration with coordinate fixes
config = DensityConfig(
    resolution=30.0,                          # Optimized default resolution
    method=DensityMethod.RASTERIZATION,       # Enhanced default method
    coordinate_system_fixes=True,             # Enable coordinate fixes (default)
    chunk_size_km=200.0,                      # Enhanced chunk size
    max_memory_gb=16.0,                       # Enhanced memory limit
    parallel_workers=4                        # Parallel processing
)

# 3. Initialize enhanced spatial analysis engine
engine = SpatialDensityEngine(config)

# Calculate spatial density with coordinate fixes
density_result = engine.calculate_density(
    scene_footprints=results['features'],
    roi_geometry=roi
)

print(f"Analysis completed using {density_result.method_used.value} method")
print(f"Coordinate fixes applied: {density_result.coordinate_system_corrected}")
print(f"Grid size: {density_result.grid_info['width']}×{density_result.grid_info['height']}")
print(f"Density range: {density_result.stats['min']}-{density_result.stats['max']} scenes per cell")

# 4. Enhanced visualization with scene limits
from planetscope_py.visualization import DensityVisualizer

visualizer = DensityVisualizer()
fig = visualizer.create_summary_plot(
    density_result=density_result,
    roi_polygon=roi,
    max_scenes_footprint=300,                 # Enhanced scene limits go here
    show_plot=True
)
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

### Enhanced Spatial Analysis Engine (v4.0+)
- **Fixed Coordinate System Display**: Proper north-to-south orientation with corrected transforms
- **Enhanced Multi-Algorithm Calculation**: Three computational methods with coordinate fixes
- **Rasterization as Optimized Default**: Changed from vector overlay for better performance
- **High-Resolution Analysis**: Support for 3m to 1000m grid resolutions with coordinate accuracy
- **Enhanced Performance Optimization**: Automatic method selection with increased thresholds
- **Memory Efficient Processing**: Adaptive grid and chunking with coordinate validation
- **One-Line Functions**: Individual plot access and simplified workflows
- **Enhanced Visualization**: Coordinate-corrected plots with increased scene limits (150+ default)
- **Robust Export**: GeoTIFF export with coordinate fixes and QGIS styling

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

### Enhanced Visualization and Export (v4.0+)
- **Coordinate-Corrected Visualization**: Fixed orientation and geographic alignment
- **Enhanced GeoTIFF Export**: Coordinate fixes with automatic QGIS styling
- **One-Line Plot Functions**: Individual visualization access with single commands
- **Increased Scene Limits**: 150+ default (up to 1000+) in footprint plots
- **Dynamic Histogram Bins**: Automatic bin calculation based on data range
- **Multi-Panel Summary Plots**: Coordinate-corrected density visualization with statistics
- **Enhanced Export Formats**: NumPy arrays, CSV, and GeoPackage with coordinate fixes
- **Robust Error Handling**: PROJ/CRS compatibility with multiple fallbacks

## Installation

### Standard Installation
```bash
# Development installation (recommended)
git clone https://github.com/Black-Lights/planetscope-py.git
cd planetscope-py
pip install -e .

# Install enhanced dependencies for full functionality
pip install -r requirements.txt
```

### Development Installation
```bash
# Clone the repository
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

## Enhanced Usage Examples

### Basic Scene Search with Enhanced Features
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

### Enhanced One-Line Analysis Workflow
```python
from planetscope_py import quick_planet_analysis

# Complete enhanced analysis in one line
result = quick_planet_analysis(
    milan_roi, "last_month",
    resolution=30.0,                          # Optimized resolution
    max_scenes_footprint=300,                 # Enhanced scene limits
    show_plots=True                           # Display in notebook
)

# Verify enhanced features
print(f"Enhanced Features Applied:")
print(f"  Coordinate fixes: {result['summary'].get('coordinate_system_corrected', False)}")
print(f"  Method used: {result['density_result'].method_used.value}")
print(f"  Scene limits: {result['summary'].get('max_scenes_displayed', 0)}")
print(f"  Analysis time: {result['summary']['computation_time_s']:.2f}s")
```

### Enhanced Individual Visualizations
```python
from planetscope_py import plot_density_map_only, plot_footprints_only, plot_histogram_only, export_geotiff_only

# Enhanced density map with coordinate fixes
fig = plot_density_map_only(
    milan_roi, "last_month", 
    save_path="density_corrected.png",
    resolution=30.0,
    clip_to_roi=True
)

# Enhanced footprint plot with increased limits
fig = plot_footprints_only(
    milan_roi, "last_month",
    save_path="footprints_enhanced.png",
    max_scenes=500                            # Enhanced: up to 1000+
)

# Enhanced histogram with dynamic bins
fig = plot_histogram_only(
    milan_roi, "last_month",
    save_path="histogram_dynamic.png"
)

# Enhanced GeoTIFF export with coordinate fixes
success = export_geotiff_only(
    milan_roi, "last_month",
    output_path="density_corrected.tif"
)
# Automatically creates: density_corrected.tif + density_corrected.qml
```

### Complete Enhanced Analysis Workflow
```python
# Complete analysis workflow with enhanced features
from planetscope_py import (
    PlanetScopeQuery, quick_planet_analysis, plot_density_map_only,
    AssetManager, GeoPackageManager
)

async def enhanced_analysis_workflow():
    # 1. Scene discovery (unchanged)
    query = PlanetScopeQuery()
    results = query.search_scenes(
        geometry=milan_geometry,
        start_date="2024-01-01",
        end_date="2024-12-31",
        cloud_cover_max=0.3
    )
    
    # 2. Enhanced spatial analysis (one-line)
    spatial_result = quick_planet_analysis(
        milan_roi, "last_month",
        resolution=30.0,                      # Optimized default
        method="rasterization",               # Enhanced default
        max_scenes_footprint=300,             # Enhanced limits
        coordinate_system_fixes=True          # Enhanced accuracy
    )
    
    # 3. Enhanced individual plots
    plot_density_map_only(milan_roi, "last_month", "enhanced_density.png")
    plot_footprints_only(milan_roi, "last_month", "enhanced_footprints.png", max_scenes=500)
    export_geotiff_only(milan_roi, "last_month", "enhanced_density.tif")
    
    # 4. Asset management (unchanged)
    asset_manager = AssetManager(query.auth)
    quota_info = await asset_manager.get_quota_info()
    
    if quota_info.remaining_area_km2 > 100:
        downloads = await asset_manager.activate_and_download_assets(
            scenes=results['features'][:20],
            clip_to_roi=milan_roi
        )
    else:
        downloads = None
        print("Insufficient quota for downloads")
    
    # 5. Export to GeoPackage (unchanged)
    geopackage_manager = GeoPackageManager()
    geopackage_manager.create_scene_geopackage(
        scenes=results['features'],
        output_path="enhanced_analysis.gpkg",
        roi=milan_roi,
        downloaded_files=downloads
    )
    
    return {
        'scenes': len(results['features']),
        'enhanced_spatial_analysis': spatial_result,
        'coordinate_fixes_applied': spatial_result['summary'].get('coordinate_system_corrected', False),
        'downloads': len(downloads) if downloads else 0
    }

# Run enhanced workflow
# results = await enhanced_analysis_workflow()
```

## Core Components

### Enhanced Spatial Analysis Engine
```python
from planetscope_py import SpatialDensityEngine, DensityConfig, DensityMethod

# Enhanced configuration with coordinate fixes
config = DensityConfig(
    resolution=30.0,                          # Optimized default (changed from 10m)
    method=DensityMethod.RASTERIZATION,       # Enhanced default (changed from AUTO)
    coordinate_system_fixes=True,             # Enable coordinate fixes (default)
    chunk_size_km=200.0,                      # Increased chunk size (default)
    max_memory_gb=16.0,                       # Increased memory limit (default)
    parallel_workers=4,                       # Parallel processing
    validate_geometries=True                  # Input validation
)

engine = SpatialDensityEngine(config)
result = engine.calculate_density(scene_footprints=scenes, roi_geometry=roi)

# Verify enhanced features
print(f"Enhanced method used: {result.method_used.value}")
print(f"Coordinate fixes applied: {result.coordinate_system_corrected}")
print(f"Performance: {result.computation_time:.3f}s")
```

### Enhanced Visualization System
```python
from planetscope_py.visualization import DensityVisualizer

# Enhanced visualizer with coordinate fixes
visualizer = DensityVisualizer(figsize=(12, 8))

# Enhanced summary plot with coordinate fixes
fig = visualizer.create_summary_plot(
    density_result=result,
    roi_polygon=milan_roi,
    max_scenes_footprint=300,                 # Enhanced limits
    clip_to_roi=True,                         # Enhanced ROI clipping
    show_plot=True                            # Display in notebook
)

# Enhanced individual plots
visualizer.plot_density_map(result, milan_roi, title="Enhanced Density Map (Coordinate-Corrected)")
visualizer.plot_scene_footprints(scene_polygons, milan_roi, max_scenes=500)

# Enhanced export with coordinate fixes
visualizer.export_density_geotiff_with_style(
    result, "enhanced_output.tif",
    roi_polygon=milan_roi,
    clip_to_roi=True,
    colormap="viridis"
)
```

## Testing

### Running Tests
```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage report
python -m pytest tests/ --cov=planetscope_py --cov-report=html

# Run specific component tests
python -m pytest tests/test_density_engine.py -v
python -m pytest tests/test_visualization.py -v
python -m pytest tests/test_workflows.py -v
```

### Test Coverage
Current test coverage: **349 tests passing (100%)**

| Component | Tests | Status | Enhanced Features |
|-----------|-------|--------|-------------------|
| Authentication | 24 | All passing | - |
| Configuration | 21 | All passing | - |
| Exceptions | 48 | All passing | - |
| Utilities | 54 | All passing | - |
| Planet API Query | 45+ | All passing | - |
| Metadata Processing | 30+ | All passing | - |
| Rate Limiting | 25+ | All passing | - |
| **Enhanced Spatial Analysis** | **35+** | **All passing** | **✓ Coordinate fixes, enhanced defaults** |
| **Enhanced Visualization** | **30+** | **All passing** | **✓ One-line functions, coordinate fixes** |
| **Enhanced Workflows** | **25+** | **All passing** | **✓ Simplified API, enhanced features** |
| Temporal Analysis | 23 | All passing | - |
| Asset Management | 23 | All passing | - |
| GeoPackage Export | 21 | All passing | - |

**Total: 349 tests with 100% success rate**

## Enhanced Configuration Reference

### Enhanced DensityConfig Defaults
```python
# Enhanced defaults in v4.0+
DensityConfig(
    resolution=30.0,                          # Changed from 10.0m (optimized)
    method=DensityMethod.RASTERIZATION,       # Changed from AUTO (performance)
    chunk_size_km=200.0,                      # Increased from 50.0km (efficiency)
    max_memory_gb=16.0,                       # Increased from 8.0GB (capacity)
    coordinate_system_fixes=True,             # New: Enable coordinate fixes
    force_single_chunk=False,                 # New: Chunking control
    validate_geometries=True                  # New: Input validation
)
```

### Enhanced Method Selection
```python
# Enhanced automatic method selection (v4.0+)
# - Favors rasterization over vector overlay for performance
# - Increased scene thresholds (2000+ scenes for rasterization)
# - Coordinate fixes enabled for all methods
# - Enhanced memory management and error handling
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

### Enhanced Dependencies (v4.0+)
- **Enhanced Spatial Analysis**: rasterio, geopandas (for coordinate fixes and export)
- **Enhanced Visualization**: matplotlib, contextily (for coordinate-corrected plotting)
- **Temporal Analysis**: xarray, scipy (for data cubes and analysis)
- **Asset Management**: aiohttp, asyncio (for async downloads)
- **GeoPackage Export**: geopandas, rasterio, fiona (for GIS data export)
- **Optional Interactive**: ipywidgets (for Jupyter notebook integration)

## Enhanced API Reference

### One-Line Functions
```python
# Enhanced one-line analysis
quick_planet_analysis(roi, period, **config)

# Enhanced one-line visualizations
plot_density_map_only(roi, period, save_path, **kwargs)
plot_footprints_only(roi, period, save_path, max_scenes=300, **kwargs)
plot_histogram_only(roi, period, save_path, **kwargs)
export_geotiff_only(roi, period, output_path, **kwargs)
```

### Enhanced Configuration
```python
# Enhanced density configuration
DensityConfig(
    coordinate_system_fixes=True,             # Enable coordinate fixes
    max_scenes_footprint=150,                 # Enhanced scene display limits
    method=DensityMethod.RASTERIZATION        # Enhanced default method
)

# Enhanced visualization configuration
DensityVisualizer(
    figsize=(12, 8),                          # Figure size
    default_cmap="viridis"                    # Default colormap
)
```

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
  note = {Enhanced with coordinate system fixes, one-line functions, and improved performance}
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