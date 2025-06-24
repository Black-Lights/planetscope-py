#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PlanetScope-py: Professional Python library for PlanetScope satellite imagery analysis.

ENHANCED VERSION with one-line functions for individual outputs.

NEW FEATURES:
- Individual plot access functions
- Fixed coordinate system display
- Increased scene footprint limits
- GeoTIFF-only export functions

Author: Ammar & Umayr
Version: 4.0.0 (Enhanced)
"""

import logging
import warnings
from typing import Dict, Any, Optional, Union, List

# Add these imports for type hints
try:
    from shapely.geometry import Polygon
except ImportError:
    # Fallback if shapely not available
    Polygon = None

from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Version information
from ._version import __version__, __version_info__

# Core Infrastructure
try:
    from .auth import PlanetAuth
    from .config import PlanetScopeConfig, default_config
    from .exceptions import (
        PlanetScopeError, AuthenticationError, ValidationError, 
        RateLimitError, APIError, ConfigurationError, AssetError
    )
    from .utils import (
        validate_geometry, calculate_area_km2, transform_geometry,
        create_bounding_box, buffer_geometry
    )
    _CORE_AVAILABLE = True
except ImportError as e:
    _CORE_AVAILABLE = False
    warnings.warn(f"Core infrastructure not available: {e}")

# Planet API Integration
try:
    from .query import PlanetScopeQuery
    from .metadata import MetadataProcessor
    from .rate_limiter import RateLimiter, RetryableSession, CircuitBreaker
    _PLANET_API_AVAILABLE = True
except ImportError as e:
    _PLANET_API_AVAILABLE = False
    warnings.warn(f"Planet API integration not available: {e}")

# Spatial Analysis
_SPATIAL_ANALYSIS_AVAILABLE = False
try:
    from .density_engine import (
        SpatialDensityEngine, DensityConfig, DensityMethod, DensityResult
    )
    _SPATIAL_ANALYSIS_AVAILABLE = True
except ImportError:
    pass

# Enhanced Visualization with Fixes
_VISUALIZATION_AVAILABLE = False
try:
    from .visualization import (
        DensityVisualizer, plot_density_only, plot_footprints_only, 
        plot_histogram_only, export_geotiff_only
    )
    _VISUALIZATION_AVAILABLE = True
except ImportError:
    pass

# Adaptive Grid Engine
_ADAPTIVE_GRID_AVAILABLE = False
try:
    from .adaptive_grid import AdaptiveGridEngine, AdaptiveGridConfig
    _ADAPTIVE_GRID_AVAILABLE = True
except ImportError:
    pass

# Performance Optimizer
_OPTIMIZER_AVAILABLE = False
try:
    from .optimizer import PerformanceOptimizer, DatasetCharacteristics, PerformanceProfile
    _OPTIMIZER_AVAILABLE = True
except ImportError:
    pass

# Temporal Analysis
_TEMPORAL_ANALYSIS_AVAILABLE = False
try:
    from .temporal_analysis import (
        TemporalAnalyzer, TemporalConfig, TemporalResolution,
        SeasonalPeriod, TemporalGap, SeasonalPattern
    )
    _TEMPORAL_ANALYSIS_AVAILABLE = True
except ImportError:
    pass

# Asset Management
_ASSET_MANAGEMENT_AVAILABLE = False
try:
    from .asset_manager import AssetManager, AssetStatus, QuotaInfo, DownloadJob
    _ASSET_MANAGEMENT_AVAILABLE = True
except ImportError:
    pass

# GeoPackage Export
_GEOPACKAGE_AVAILABLE = False
try:
    from .geopackage_manager import (
        GeoPackageManager, GeoPackageConfig, LayerInfo, RasterInfo
    )
    _GEOPACKAGE_AVAILABLE = True
except ImportError:
    pass


# Enhanced GeoPackage One-Liner Functions
_GEOPACKAGE_ONELINERS_AVAILABLE = False
try:
    from .geopackage_oneliners import (
        quick_geopackage_export, create_milan_geopackage, create_clipped_geopackage,
        create_full_grid_geopackage, export_scenes_to_geopackage,
        quick_scene_search_and_export, validate_geopackage_output,
        batch_geopackage_export, get_geopackage_usage_examples
    )
    _GEOPACKAGE_ONELINERS_AVAILABLE = True
except ImportError as e:
    _GEOPACKAGE_ONELINERS_AVAILABLE = False
    import warnings
    warnings.warn(f"GeoPackage one-liners not available: {e}")

# Preview Management
_PREVIEW_MANAGEMENT_AVAILABLE = False
try:
    from .preview_manager import PreviewManager
    _PREVIEW_MANAGEMENT_AVAILABLE = True
except ImportError:
    pass

# Interactive Management
_INTERACTIVE_AVAILABLE = False
try:
    from .interactive_manager import InteractiveManager
    _INTERACTIVE_AVAILABLE = True
except ImportError:
    pass

# Enhanced Workflow API with Fixes
_WORKFLOWS_AVAILABLE = False
try:
    from .workflows import (
        analyze_density, quick_analysis, batch_analysis, temporal_analysis_workflow,
        # NEW: One-line functions for individual outputs
        quick_density_plot, quick_footprints_plot, quick_geotiff_export
    )
    _WORKFLOWS_AVAILABLE = True
except ImportError:
    pass

# Configuration Presets
_CONFIG_PRESETS_AVAILABLE = False
try:
    from .config import PresetConfigs
    _CONFIG_PRESETS_AVAILABLE = True
except ImportError:
    pass


# ENHANCED HIGH-LEVEL API FUNCTIONS

def create_scene_geopackage(
    roi: Union["Polygon", list, dict],  # Use quotes for forward reference
    time_period: str = "last_month",
    output_path: Optional[str] = None,
    clip_to_roi: bool = True,
    **kwargs
) -> str:
    """
    HIGH-LEVEL API: Create GeoPackage with scene footprints.
    
    ENHANCED one-line function for GeoPackage creation with Planet scene footprints.
    
    Args:
        roi: Region of interest as Shapely Polygon, coordinate list, or GeoJSON dict
        time_period: Time period specification:
            - "last_month": Previous 30 days
            - "last_3_months": Previous 90 days
            - "YYYY-MM-DD/YYYY-MM-DD": Custom date range
        output_path: Path for output GeoPackage (auto-generated if None)
        clip_to_roi: Whether to clip scene footprints to ROI shape (default: True)
        **kwargs: Additional parameters:
            - cloud_cover_max (float): Maximum cloud cover threshold (default: 0.3)
            - schema (str): Attribute schema ("minimal", "standard", "comprehensive")
            - sun_elevation_min (float): Minimum sun elevation in degrees
            - ground_control (bool): Require ground control points
            - quality_category (str): Required quality category
            - item_types (list): Planet item types to search
    
    Returns:
        str: Path to created GeoPackage file
    
    Example:
        >>> from planetscope_py import create_scene_geopackage
        >>> from shapely.geometry import Polygon
        >>> 
        >>> milan_roi = Polygon([
        ...     [8.7, 45.1], [9.8, 44.9], [10.3, 45.3], [10.1, 45.9],
        ...     [9.5, 46.2], [8.9, 46.0], [8.5, 45.6], [8.7, 45.1]
        ... ])
        >>> 
        >>> # One-liner to create clipped GeoPackage
        >>> gpkg_path = create_scene_geopackage(milan_roi, "2025-01-01/2025-01-31")
        >>> print(f"Created: {gpkg_path}")
    """
    if not _GEOPACKAGE_ONELINERS_AVAILABLE:
        raise ImportError(
            "GeoPackage one-liner functions not available. "
            "Please create planetscope_py/geopackage_oneliners.py with the one-liner functions. "
            "See the artifact code provided for the complete implementation."
        )
    
    return quick_geopackage_export(
        roi=roi,
        time_period=time_period,
        output_path=output_path,
        clip_to_roi=clip_to_roi,
        **kwargs
    )


def quick_milan_geopackage(
    time_period: str = "2025-01-01/2025-01-31",
    output_path: Optional[str] = None,
    size: str = "large",
    **kwargs
) -> str:
    """
    HIGH-LEVEL API: Create Milan area GeoPackage with predefined polygon.
    
    Args:
        time_period: Time period for scene search (default: Jan 2025)
        output_path: Output path (auto-generated if None)
        size: Milan area size preset:
            - "city_center": ~100 km² (central Milan only)
            - "small": ~500 km² (Milan + close suburbs)  
            - "medium": ~1200 km² (Greater Milan area)
            - "large": ~2000+ km² (Milan metropolitan region)
        **kwargs: Additional parameters (cloud_cover_max, schema, etc.)
    
    Returns:
        str: Path to created GeoPackage file
    """
    if not _GEOPACKAGE_ONELINERS_AVAILABLE:
        raise ImportError(
            "GeoPackage one-liner functions not available. "
            "Please create planetscope_py/geopackage_oneliners.py with the one-liner functions."
        )
    
    return create_milan_geopackage(
        time_period=time_period,
        output_path=output_path,
        size=size,
        **kwargs
    )


def search_and_export_scenes(
    roi: Union["Polygon", list, dict],
    start_date: str,
    end_date: str,
    output_path: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    HIGH-LEVEL API: Search scenes and export to GeoPackage with detailed results.
    
    Args:
        roi: Region of interest (Polygon, coordinate list, or GeoJSON dict)
        start_date: Start date in "YYYY-MM-DD" format
        end_date: End date in "YYYY-MM-DD" format
        output_path: Path for output GeoPackage (auto-generated if None)
        **kwargs: Additional search and export parameters
    
    Returns:
        dict: Comprehensive results including scene count, statistics, file path
    """
    if not _GEOPACKAGE_ONELINERS_AVAILABLE:
        raise ImportError(
            "GeoPackage one-liner functions not available. "
            "Please create planetscope_py/geopackage_oneliners.py with the one-liner functions."
        )
    
    return quick_scene_search_and_export(
        roi=roi,
        start_date=start_date,
        end_date=end_date,
        output_path=output_path,
        **kwargs
    )


def validate_scene_geopackage(geopackage_path: str) -> Dict[str, Any]:
    """
    HIGH-LEVEL API: Validate GeoPackage and get comprehensive statistics.
    
    Args:
        geopackage_path: Path to GeoPackage file to validate
    
    Returns:
        dict: Validation results and statistics
    
    Example:
        >>> from planetscope_py import validate_scene_geopackage
        >>> 
        >>> stats = validate_scene_geopackage("milan_scenes.gpkg")
        >>> if stats['valid']:
        ...     print(f" Valid GeoPackage with {stats['feature_count']} scenes")
    """
    if not _GEOPACKAGE_ONELINERS_AVAILABLE:
        raise ImportError(
            "GeoPackage one-liner functions not available. "
            "Please ensure geopackage_oneliners.py is created and all dependencies are installed."
        )
    
    return validate_geopackage_output(geopackage_path)

def analyze_roi_density(roi_polygon, time_period="2025-01-01/2025-01-31", **kwargs):
    """
    Complete density analysis for a region of interest.
    
    ENHANCED with coordinate system fixes and increased scene footprint limits.
    
    Args:
        roi_polygon: Region of interest as Shapely Polygon or coordinate list
        time_period: Time period as "start_date/end_date" string or tuple
        **kwargs: Optional parameters including:
            resolution (float): Analysis resolution in meters (default: 30.0)
            cloud_cover_max (float): Maximum cloud cover threshold (default: 0.2)
            output_dir (str): Output directory (default: "./planetscope_analysis")
            method (str): Density calculation method (default: "rasterization")
            clip_to_roi (bool): Clip outputs to ROI shape (default: True)
            create_visualizations (bool): Generate plots (default: True)
            export_geotiff (bool): Export GeoTIFF (default: True)
            max_scenes_footprint (int): Max scenes in footprint plot (default: 150)
    
    Returns:
        dict: Analysis results with coordinate-corrected outputs
    
    Example:
        >>> from planetscope_py import analyze_roi_density
        >>> from shapely.geometry import Polygon
        >>> 
        >>> milan_roi = Polygon([
        ...     [8.7, 45.1], [9.8, 44.9], [10.3, 45.3], [10.1, 45.9],
        ...     [9.5, 46.2], [8.9, 46.0], [8.5, 45.6], [8.7, 45.1]
        ... ])
        >>> 
        >>> result = analyze_roi_density(milan_roi, "2025-01-01/2025-01-31")
        >>> print(f"Found {result['scenes_found']} scenes")
        >>> print(f"Mean density: {result['density_result'].stats['mean']:.1f}")
    """
    if not _WORKFLOWS_AVAILABLE:
        raise ImportError(
            "Workflows module not available. Please ensure all dependencies are installed."
        )
    
    return analyze_density(roi_polygon, time_period, **kwargs)


def quick_planet_analysis(roi, period="last_month", output_dir="./output", show_plots=True, **config):
    """
    Simplified analysis function with minimal parameters.
    
    ENHANCED with coordinate fixes and increased scene limits.
    
    Args:
        roi: Region of interest as Shapely Polygon or coordinate list
        period: Time period specification:
            - "last_month": Previous 30 days
            - "last_3_months": Previous 90 days  
            - "YYYY-MM-DD/YYYY-MM-DD": Custom date range
        output_dir: Directory for saving results
        show_plots: Whether to display plots in notebook cells (default: True)
        **config: Configuration overrides:
            - resolution: Analysis resolution in meters (default: 30.0)
            - cloud_cover_max: Maximum cloud cover threshold (default: 0.2)
            - method: Density calculation method (default: "rasterization")
            - max_scenes_footprint: Max scenes in footprint plot (default: 150)
    
    Returns:
        dict: Complete analysis results with fixed coordinate system
    
    Example:
        >>> from planetscope_py import quick_planet_analysis
        >>> 
        >>> # Basic usage
        >>> result = quick_planet_analysis(milan_polygon, "last_month")
        >>> 
        >>> # With custom parameters
        >>> result = quick_planet_analysis(
        ...     milan_polygon, "2025-01-01/2025-01-31", 
        ...     resolution=50, max_scenes_footprint=300
        ... )
    """
    if not _WORKFLOWS_AVAILABLE:
        raise ImportError(
            "Workflows module not available. Please ensure all dependencies are installed."
        )
    
    return quick_analysis(roi, period, output_dir, show_plots=show_plots, **config)


# NEW: ONE-LINE FUNCTIONS FOR INDIVIDUAL OUTPUTS

def plot_density_map_only(roi_polygon, time_period="last_month", save_path=None, **kwargs):
    """
    ONE-LINE function to generate only the density map plot.
    
    FIXED coordinate system display - no more mirrored images!
    
    Args:
        roi_polygon: ROI as Shapely Polygon or coordinate list
        time_period: Time period (default: "last_month")
        save_path: Path to save plot (optional)
        **kwargs: Additional parameters (resolution, cloud_cover_max, etc.)
    
    Returns:
        matplotlib.Figure: Density map plot with corrected orientation
    
    Example:
        >>> from planetscope_py import plot_density_map_only
        >>> 
        >>> # Just get the density plot
        >>> fig = plot_density_map_only(milan_roi, "2025-01-01/2025-01-31", "density.png")
        >>> 
        >>> # With custom resolution
        >>> fig = plot_density_map_only(milan_roi, "last_month", resolution=50)
    """
    if not _WORKFLOWS_AVAILABLE:
        raise ImportError(
            "Workflows module not available. Please ensure all dependencies are installed."
        )
    
    return quick_density_plot(roi_polygon, time_period, save_path, **kwargs)


def plot_footprints_only(roi_polygon, time_period="last_month", save_path=None, max_scenes=300, **kwargs):
    """
    ONE-LINE function to generate only the scene footprints plot.
    
    ENHANCED with increased scene limits (300+ default instead of 50).
    
    Args:
        roi_polygon: ROI as Shapely Polygon or coordinate list
        time_period: Time period (default: "last_month")
        save_path: Path to save plot (optional)
        max_scenes: Maximum scenes to display (default: 300, increased from 50)
        **kwargs: Additional parameters
    
    Returns:
        matplotlib.Figure: Scene footprints plot
    
    Example:
        >>> from planetscope_py import plot_footprints_only
        >>> 
        >>> # Show more scenes (default now 300)
        >>> fig = plot_footprints_only(milan_roi, "2025-01-01/2025-01-31", "footprints.png")
        >>> 
        >>> # Show all scenes if reasonable number
        >>> fig = plot_footprints_only(milan_roi, "last_month", max_scenes=1000)
    """
    if not _WORKFLOWS_AVAILABLE:
        raise ImportError(
            "Workflows module not available. Please ensure all dependencies are installed."
        )
    
    return quick_footprints_plot(roi_polygon, time_period, save_path, max_scenes, **kwargs)


def export_geotiff_only(roi_polygon, time_period="last_month", output_path="density.tif", **kwargs):
    """
    ONE-LINE function to generate only GeoTIFF + QML files.
    
    ENHANCED with coordinate fixes and robust PROJ error handling.
    
    Args:
        roi_polygon: ROI as Shapely Polygon or coordinate list
        time_period: Time period (default: "last_month")
        output_path: Path for GeoTIFF output (default: "density.tif")
        **kwargs: Additional parameters (clip_to_roi, resolution, etc.)
    
    Returns:
        bool: True if export successful, False otherwise
    
    Example:
        >>> from planetscope_py import export_geotiff_only
        >>> 
        >>> # Just get the GeoTIFF files
        >>> success = export_geotiff_only(milan_roi, "2025-01-01/2025-01-31", "milan_density.tif")
        >>> 
        >>> # Will also create milan_density.qml automatically
        >>> # With ROI clipping
        >>> success = export_geotiff_only(milan_roi, "last_month", "output.tif", clip_to_roi=True)
    """
    if not _WORKFLOWS_AVAILABLE:
        raise ImportError(
            "Workflows module not available. Please ensure all dependencies are installed."
        )
    
    return quick_geotiff_export(roi_polygon, time_period, output_path, **kwargs)

# Plot_histogram_only wrapper function
def plot_histogram_only(roi_polygon, time_period="last_month", save_path=None, **kwargs):
    """
    ONE-LINE function to generate only the histogram plot.
    
    ENHANCED with proper dynamic bins (no more fixed 11-19 ranges).
    
    Args:
        roi_polygon: ROI as Shapely Polygon or coordinate list
        time_period: Time period (default: "last_month")
        save_path: Path to save plot (optional)
        **kwargs: Additional parameters (clip_to_roi, bins, etc.)
    
    Returns:
        matplotlib.Figure: Histogram plot with dynamic bins
    
    Example:
        >>> from planetscope_py import plot_histogram_only
        >>> 
        >>> # Just get the histogram with proper bins
        >>> fig = plot_histogram_only(milan_roi, "2025-01-01/2025-01-31", "histogram.png")
        >>> 
        >>> # With custom parameters
        >>> fig = plot_histogram_only(milan_roi, "last_month", clip_to_roi=True)
    """
    if not _WORKFLOWS_AVAILABLE:
        raise ImportError(
            "Workflows module not available. Please ensure all dependencies are installed."
        )
    
    # This would call the workflow function or direct visualization function
    try:
        from .visualization import plot_histogram_only as viz_histogram
        # Analyze first to get density result
        result = quick_analysis(roi_polygon, time_period, create_visualizations=False, 
                              export_geotiff=False, **kwargs)
        if result['density_result'] is not None:
            return viz_histogram(result['density_result'], roi_polygon, save_path, **kwargs)
    except ImportError:
        raise ImportError("Visualization module not available")


# Package Exports - ENHANCED
__all__ = [
    # Version
    "__version__",
    
    # ENHANCED High-Level API
    "analyze_roi_density",
    "quick_planet_analysis",
    
    # NEW: One-Line Functions for Individual Outputs
    "plot_density_map_only",
    "plot_footprints_only", 
    "export_geotiff_only",
    
    # Core Infrastructure
    "PlanetAuth",
    "PlanetScopeConfig", 
    "default_config",
    
    # Exceptions
    "PlanetScopeError",
    "AuthenticationError",
    "ValidationError", 
    "RateLimitError",
    "APIError",
    "ConfigurationError",
    "AssetError",
    
    # Utilities
    "validate_geometry",
    "calculate_area_km2",
    "transform_geometry",
    "create_bounding_box",
    "buffer_geometry",
    
    # Planet API Integration
    "PlanetScopeQuery",
    "MetadataProcessor",
    "RateLimiter",
    "RetryableSession", 
    "CircuitBreaker",
]

# Conditional exports based on module availability
if _SPATIAL_ANALYSIS_AVAILABLE:
    __all__.extend([
        "SpatialDensityEngine",
        "DensityConfig", 
        "DensityMethod",
        "DensityResult",
    ])

if _VISUALIZATION_AVAILABLE:
    __all__.extend([
        "DensityVisualizer",
        "plot_density_only",     # Direct visualization functions
        "plot_footprints_only",  # (different from workflow functions)
        "export_geotiff_only",
        "plot_histogram_only",  # NEW: Histogram plot function
    ])

if _ADAPTIVE_GRID_AVAILABLE:
    __all__.extend([
        "AdaptiveGridEngine",
        "AdaptiveGridConfig",
    ])

if _OPTIMIZER_AVAILABLE:
    __all__.extend([
        "PerformanceOptimizer",
        "DatasetCharacteristics",
        "PerformanceProfile",
    ])

if _TEMPORAL_ANALYSIS_AVAILABLE:
    __all__.extend([
        "TemporalAnalyzer",
        "TemporalConfig",
        "TemporalResolution",
        "SeasonalPeriod", 
        "TemporalGap",
        "SeasonalPattern",
    ])

if _ASSET_MANAGEMENT_AVAILABLE:
    __all__.extend([
        "AssetManager",
        "AssetStatus",
        "QuotaInfo",
        "DownloadJob",
    ])

if _GEOPACKAGE_AVAILABLE:
    __all__.extend([
        "GeoPackageManager",
        "GeoPackageConfig",
        "LayerInfo",
        "RasterInfo",
    ])

# ADD these to your existing __all__ list:
if _GEOPACKAGE_ONELINERS_AVAILABLE:
    __all__.extend([
        # HIGH-LEVEL GeoPackage API
        "create_scene_geopackage",
        "quick_milan_geopackage", 
        "search_and_export_scenes",
        "validate_scene_geopackage",
        
        # ONE-LINE FUNCTIONS (optional - for power users)
        "quick_geopackage_export",
        "create_milan_geopackage",
        "create_clipped_geopackage", 
        "create_full_grid_geopackage",
        "export_scenes_to_geopackage",
        "quick_scene_search_and_export",
        "validate_geopackage_output",
        "batch_geopackage_export",
    ])

if _PREVIEW_MANAGEMENT_AVAILABLE:
    __all__.extend([
        "PreviewManager",
    ])

if _INTERACTIVE_AVAILABLE:
    __all__.extend([
        "InteractiveManager",
    ])

if _WORKFLOWS_AVAILABLE:
    __all__.extend([
        "analyze_density",
        "quick_analysis", 
        "batch_analysis",
        "temporal_analysis_workflow",
        # One-line workflow functions
        "quick_density_plot",
        "quick_footprints_plot", 
        "quick_geotiff_export",
    ])

if _CONFIG_PRESETS_AVAILABLE:
    __all__.extend([
        "PresetConfigs",
    ])

# Package Metadata
__author__ = "Ammar & Umayr"
__email__ = "mohammadammarmughees@gmail.com"
__description__ = (
    "Professional Python library for PlanetScope satellite imagery analysis with "
    "enhanced coordinate system fixes, increased scene footprint limits, and one-line functions"
)
__url__ = "https://github.com/Black-Lights/planetscope-py"
__license__ = "MIT"

# Diagnostic Functions
def get_component_status():
    """Get availability status of all library components."""
    return {
        "core_infrastructure": _CORE_AVAILABLE,
        "planet_api_integration": _PLANET_API_AVAILABLE,
        "spatial_analysis": {
            "density_engine": _SPATIAL_ANALYSIS_AVAILABLE,
            "adaptive_grid": _ADAPTIVE_GRID_AVAILABLE,
            "optimizer": _OPTIMIZER_AVAILABLE,
            "visualization": _VISUALIZATION_AVAILABLE,
        },
        "advanced_features": {
            "temporal_analysis": _TEMPORAL_ANALYSIS_AVAILABLE,
            "asset_management": _ASSET_MANAGEMENT_AVAILABLE,
            "geopackage_export": _GEOPACKAGE_AVAILABLE,
            "geopackage_oneliners": _GEOPACKAGE_ONELINERS_AVAILABLE,  # ADD THIS LINE
            "preview_management": _PREVIEW_MANAGEMENT_AVAILABLE,
            "interactive_features": _INTERACTIVE_AVAILABLE,
        },
        "workflows": {
            "high_level_api": _WORKFLOWS_AVAILABLE,
            "config_presets": _CONFIG_PRESETS_AVAILABLE,
        }
    }


def check_module_status():
    """Display detailed status of all library modules."""
    status = get_component_status()
    
    print("PlanetScope-py Module Status (Enhanced)")
    print("=" * 45)
    
    # Core Components
    print("\nCore Infrastructure:")
    print(f"  Authentication: {'Available' if status['core_infrastructure'] else 'Not Available'}")
    print(f"  Configuration: {'Available' if status['core_infrastructure'] else 'Not Available'}")
    print(f"  Planet API: {'Available' if status['planet_api_integration'] else 'Not Available'}")
    
    # Spatial Analysis
    print("\nSpatial Analysis (Enhanced):")
    spatial = status['spatial_analysis']
    for component, available in spatial.items():
        status_text = "Available" if available else "Not Available"
        print(f"  {component.replace('_', ' ').title()}: {status_text}")
    
    # Advanced Features
    print("\nAdvanced Features:")
    advanced = status['advanced_features']
    for component, available in advanced.items():
        status_text = "Available" if available else "Not Available"
        print(f"  {component.replace('_', ' ').title()}: {status_text}")
    
    # NEW: GeoPackage One-Liners Status
    print("\nGeoPackage Features (NEW):")
    print(f"  GeoPackage Manager: {'Available' if _GEOPACKAGE_AVAILABLE else 'Not Available'}")
    print(f"  GeoPackage One-Liners: {'Available' if _GEOPACKAGE_ONELINERS_AVAILABLE else 'Not Available'}")
    
    # Workflows
    print("\nWorkflow API (Enhanced):")
    workflows = status['workflows']
    for component, available in workflows.items():
        status_text = "Available" if available else "Not Available"
        print(f"  {component.replace('_', ' ').title()}: {status_text}")
    
    # Summary
    total_components = (
        len(spatial) + len(advanced) + len(workflows) + 4  # +4 for core + geopackage components
    )
    available_components = (
        sum(spatial.values()) + sum(advanced.values()) + sum(workflows.values()) + 
        int(status['core_infrastructure']) + int(status['planet_api_integration']) +
        int(_GEOPACKAGE_AVAILABLE) + int(_GEOPACKAGE_ONELINERS_AVAILABLE)
    )
    
    print(f"\nSummary: {available_components}/{total_components} components available")
    
    if available_components < total_components:
        print("\nMissing components may require additional dependencies.")
        print("Refer to documentation for installation instructions.")





def get_usage_examples():
    """Display usage examples for the ENHANCED simplified API."""
    print("PlanetScope-py Usage Examples (Enhanced)")
    print("=" * 42)
    
    print("\n1. Complete Analysis (1-line):")
    print("   from planetscope_py import analyze_roi_density")
    print("   result = analyze_roi_density(milan_roi, '2025-01-01/2025-01-31')")
    
    print("\n2. Ultra-Simple Analysis:")
    print("   from planetscope_py import quick_planet_analysis")
    print("   result = quick_planet_analysis(milan_polygon, 'last_month')")
    
    print("\n3. NEW: Individual Plot Functions (1-line each):")
    print("   from planetscope_py import plot_density_map_only, plot_footprints_only")
    print("   ")
    print("   # Just get density map (FIXED orientation)")
    print("   fig = plot_density_map_only(milan_roi, 'last_month', 'density.png')")
    print("   ")
    print("   # Just get footprints (300+ scenes default)")
    print("   fig = plot_footprints_only(milan_roi, 'last_month', max_scenes=500)")
    
    print("\n4. NEW: GeoTIFF-Only Export (1-line):")
    print("   from planetscope_py import export_geotiff_only")
    print("   ")
    print("   # Just get GeoTIFF + QML files")
    print("   success = export_geotiff_only(milan_roi, 'last_month', 'output.tif')")
    print("   # Just get histogram (proper dynamic bins)")
    print("   fig = plot_histogram_only(milan_roi, 'last_month', 'histogram.png')")
    
    print("\n5. NEW: GeoPackage One-Liners (ENHANCED):")
    if _GEOPACKAGE_ONELINERS_AVAILABLE:
        print("   from planetscope_py import create_scene_geopackage, quick_milan_geopackage")
        print("   ")
        print("   # Create GeoPackage with scene footprints")
        print("   gpkg = create_scene_geopackage(milan_roi, '2025-01-01/2025-01-31')")
        print("   ")
        print("   # Milan preset with different sizes")
        print("   milan_gpkg = quick_milan_geopackage('last_month', size='large')")
        print("   ")
        print("   # Search + export with detailed results")
        print("   result = search_and_export_scenes(roi, '2025-01-01', '2025-01-31')")
        print("   print(f'Found {result[\"scenes_found\"]} scenes')")
        print("   ")
        print("   # Validate GeoPackage")
        print("   stats = validate_scene_geopackage('output.gpkg')")
        print("   print(f'Valid: {stats[\"valid\"]}, Features: {stats[\"feature_count\"]}')")
    else:
        print("   # GeoPackage one-liners not available")
    
    print("\n6. Enhanced Parameters:")
    print("   # Increased scene footprint limits")
    print("   result = quick_planet_analysis(roi, 'last_month', max_scenes_footprint=300)")
    print("   ")
    print("   # Custom resolution with coordinate fixes")
    print("   result = analyze_roi_density(roi, period, resolution=10)")
    print("   ")
    print("   # GeoPackage with clipping")
    if _GEOPACKAGE_ONELINERS_AVAILABLE:
        print("   gpkg = create_scene_geopackage(roi, period, clip_to_roi=True)")
    
    print("\n7. Configuration Presets (if available):")
    if _CONFIG_PRESETS_AVAILABLE:
        print("   from planetscope_py import PresetConfigs")
        print("   config = PresetConfigs.ULTRA_HIGH_DETAIL()")
        print("   result = analyze_roi_density(roi, period, config=config)")
    else:
        print("   # Configuration presets not available")
    
    print("\nENHANCEMENTS in this version:")
    print("✓ Fixed coordinate system display (no more mirroring)")
    print("✓ Increased scene footprint limits (150+ default, up to 1000+)")
    print("✓ One-line functions for individual outputs")
    print("✓ Robust PROJ error handling")
    print("✓ Enhanced ROI polygon clipping")
    if _GEOPACKAGE_ONELINERS_AVAILABLE:
        print("✓ NEW: GeoPackage one-liner functions")
        print("✓ NEW: Milan area presets with different sizes")
        print("✓ NEW: Scene search + export with comprehensive stats")


# Enhanced help function
def help():
    """Display comprehensive help for the enhanced PlanetScope-py library."""
    print("PlanetScope-py Enhanced Help")
    print("=" * 30)
    print()
    print("This library provides professional tools for PlanetScope satellite imagery analysis")
    print("with enhanced coordinate system fixes, simplified one-line functions, and GeoPackage export.")
    print()
    
    check_module_status()
    print()
    get_usage_examples()
    print()
    
    print("For more detailed documentation, visit:")
    print("https://github.com/Black-Lights/planetscope-py")
    print()
    print("Common Issues Fixed:")
    print("• Mirrored/flipped density maps")
    print("• Limited scene footprint display (50 → 150+)")
    print("• Complex multi-step workflows") 
    print("• PROJ database compatibility issues")
    if _GEOPACKAGE_ONELINERS_AVAILABLE:
        print("• Complex GeoPackage creation workflows")
        print("• Manual scene clipping and attribute management")

def demo_milan_comparison():
    """Show comparison between your original code and one-liner equivalent."""
    print(" Milan GeoPackage Demo: Before vs After")
    print("=" * 50)
    
    print("\nYOUR ORIGINAL CODE:")
    print("─" * 20)
    print("""
from datetime import datetime
from shapely.geometry import Polygon
from planetscope_py import PlanetScopeQuery, GeoPackageManager, GeoPackageConfig

# Create large irregular polygon around Milan (~2000+ km²)
milan_polygon = Polygon([
    [8.7, 45.1], [9.8, 44.9], [10.3, 45.3], [10.1, 45.9],
    [9.5, 46.2], [8.9, 46.0], [8.5, 45.6], [8.7, 45.1]
])

# Search scenes
query = PlanetScopeQuery()
results = query.search_scenes(
    geometry=milan_polygon,
    start_date="2025-01-05",
    end_date="2025-01-25", 
    cloud_cover_max=1.0,
    item_types=["PSScene"]
)

scenes = results.get('features', [])

# Create clipped GeoPackage
if scenes:
    config = GeoPackageConfig(
        clip_to_roi=True,
        attribute_schema="standard"
    )
    
    manager = GeoPackageManager(config=config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"milan_large_{timestamp}.gpkg"
    
    manager.create_scene_geopackage(
        scenes=scenes,
        output_path=output_path,
        roi=milan_polygon
    )
    """)
    
    print("\nNEW ONE-LINER EQUIVALENTS:")
    print("─" * 27)
    print("""
# Option 1: Direct one-liner (your exact polygon)
from planetscope_py import create_scene_geopackage
from shapely.geometry import Polygon

milan_polygon = Polygon([
    [8.7, 45.1], [9.8, 44.9], [10.3, 45.3], [10.1, 45.9],
    [9.5, 46.2], [8.9, 46.0], [8.5, 45.6], [8.7, 45.1]
])

gpkg_path = create_scene_geopackage(
    milan_polygon, "2025-01-05/2025-01-25", 
    clip_to_roi=True, cloud_cover_max=1.0
)

# Option 2: Even simpler with Milan preset
from planetscope_py import quick_milan_geopackage

milan_gpkg = quick_milan_geopackage(
    "2025-01-05/2025-01-25", size="large", cloud_cover_max=1.0
)

# Option 3: With detailed results
from planetscope_py import search_and_export_scenes

result = search_and_export_scenes(
    milan_polygon, "2025-01-05", "2025-01-25",
    clip_to_roi=True, cloud_cover_max=1.0
)

print(f" Found {result['scenes_found']} scenes")
print(f" Coverage: {result['coverage_ratio']:.1%}")
print(f" GeoPackage: {result['geopackage_path']}")
    """)
    
    print("\nBENEFITS OF ONE-LINERS:")
    print(" 20+ lines → 3 lines")
    print(" No manual timestamp handling")
    print(" No config object creation")
    print(" Built-in validation and error handling")
    print(" Automatic statistics and reporting")
    print(" Preset polygons for common areas")
    print(" Batch processing capabilities")


if __name__ == "__main__":
    demo_milan_comparison()