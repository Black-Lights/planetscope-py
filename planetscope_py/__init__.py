#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PlanetScope-py: Professional Python library for PlanetScope satellite imagery analysis.

ENHANCED VERSION with clean temporal analysis integration.

NEW FEATURES:
- Clean temporal analysis module (grid-based temporal patterns)
- Individual plot access functions
- Fixed coordinate system display
- Increased scene footprint limits
- GeoTIFF-only export functions
- Complete temporal analysis implementation

Author: Ammar & Umayr
Version: 4.1.0 (Enhanced + Complete Temporal Analysis)
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

# CLEAN Temporal Analysis - COMPLETE IMPLEMENTATION
_TEMPORAL_ANALYSIS_AVAILABLE = False
try:
    from .temporal_analysis import (
        TemporalAnalyzer, TemporalConfig, TemporalMetric, TemporalResolution,
        TemporalResult, analyze_temporal_patterns
    )
    _TEMPORAL_ANALYSIS_AVAILABLE = True
except ImportError as e:
    _TEMPORAL_ANALYSIS_AVAILABLE = False
    warnings.warn(f"Clean temporal analysis not available: {e}")

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


# NEW: TEMPORAL ANALYSIS HIGH-LEVEL FUNCTIONS

def analyze_roi_temporal_patterns(
    roi_polygon: Union["Polygon", list, dict],
    time_period: str = "2025-01-01/2025-03-31",
    spatial_resolution: float = 30.0,
    clip_to_roi: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    HIGH-LEVEL API: Complete temporal pattern analysis for a region of interest.
    
    NEW FUNCTION for grid-based temporal analysis with same coordinate fixes as spatial density.
    
    Args:
        roi_polygon: Region of interest as Shapely Polygon, coordinate list, or GeoJSON dict
        time_period: Analysis time period as "YYYY-MM-DD/YYYY-MM-DD" string
        spatial_resolution: Spatial grid resolution in meters (default: 30m)
        clip_to_roi: If True, clip analysis to ROI shape. If False, analyze full grid
        **kwargs: Additional parameters:
            - cloud_cover_max (float): Maximum cloud cover threshold (default: 0.3)
            - metrics (list): List of TemporalMetric to calculate (default: key metrics)
            - min_scenes_per_cell (int): Minimum scenes required per cell (default: 2)
            - output_dir (str): Output directory (default: "./temporal_analysis")
            - create_visualizations (bool): Generate plots (default: True)
            - export_geotiffs (bool): Export GeoTIFF files (default: True)
            - optimization_level (str): "fast", "accurate", or "auto" (default: "auto")
    
    Returns:
        dict: Complete temporal analysis results including:
            - temporal_result: TemporalResult object with all metrics
            - visualizations: Dictionary of plot file paths
            - exports: Dictionary of exported file paths
            - summary: Analysis summary statistics
    
    Example:
        >>> from planetscope_py import analyze_roi_temporal_patterns
        >>> from shapely.geometry import Polygon
        >>> 
        >>> milan_roi = Polygon([
        ...     [8.7, 45.1], [9.8, 44.9], [10.3, 45.3], [10.1, 45.9],
        ...     [9.5, 46.2], [8.9, 46.0], [8.5, 45.6], [8.7, 45.1]
        ... ])
        >>> 
        >>> # Complete temporal analysis
        >>> result = analyze_roi_temporal_patterns(
        ...     milan_roi, "2025-01-01/2025-03-31",
        ...     spatial_resolution=100, clip_to_roi=True
        ... )
        >>> 
        >>> print(f"Found {result['scenes_found']} scenes")
        >>> print(f"Mean coverage days: {result['temporal_result'].temporal_stats['mean_coverage_days']:.1f}")
        >>> print(f"Output directory: {result['output_directory']}")
    """
    if not _TEMPORAL_ANALYSIS_AVAILABLE:
        raise ImportError(
            "Temporal analysis module not available. "
            "Please ensure planetscope_py/temporal_analysis.py is created and all dependencies are installed."
        )
    
    return analyze_temporal_patterns(
        roi_polygon=roi_polygon,
        time_period=time_period,
        spatial_resolution=spatial_resolution,
        clip_to_roi=clip_to_roi,
        **kwargs
    )


def quick_temporal_analysis(
    roi: Union["Polygon", list, dict],
    period: str = "last_3_months",
    output_dir: str = "./temporal_output",
    spatial_resolution: float = 100.0,
    **kwargs
) -> Dict[str, Any]:
    """
    HIGH-LEVEL API: Simplified temporal analysis with minimal parameters.
    
    Args:
        roi: Region of interest as Shapely Polygon, coordinate list, or GeoJSON dict
        period: Time period specification:
            - "last_month": Previous 30 days
            - "last_3_months": Previous 90 days (default for temporal analysis)
            - "YYYY-MM-DD/YYYY-MM-DD": Custom date range
        output_dir: Directory for saving results
        spatial_resolution: Spatial grid resolution in meters (default: 100m)
        **kwargs: Additional configuration overrides
    
    Returns:
        dict: Complete temporal analysis results
    
    Example:
        >>> from planetscope_py import quick_temporal_analysis
        >>> 
        >>> # Basic temporal analysis
        >>> result = quick_temporal_analysis(milan_polygon, "last_3_months")
        >>> 
        >>> # Custom resolution and period
        >>> result = quick_temporal_analysis(
        ...     milan_polygon, "2025-01-01/2025-06-30",
        ...     spatial_resolution=50
        ... )
    """
    # Parse period shortcuts
    if period == "last_month":
        from datetime import datetime, timedelta
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        time_period = f"{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
    elif period == "last_3_months":
        from datetime import datetime, timedelta
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
        time_period = f"{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
    else:
        time_period = period
    
    return analyze_roi_temporal_patterns(
        roi_polygon=roi,
        time_period=time_period,
        spatial_resolution=spatial_resolution,
        output_dir=output_dir,
        **kwargs
    )


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


# Package Exports - ENHANCED WITH COMPLETE TEMPORAL ANALYSIS
__all__ = [
    # Version
    "__version__",
    
    # ENHANCED High-Level API
    "analyze_roi_density",
    "quick_planet_analysis",
    
    # NEW: Complete Temporal Analysis API
    "analyze_roi_temporal_patterns",
    "quick_temporal_analysis",
    
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

# NEW: Complete Temporal Analysis exports
if _TEMPORAL_ANALYSIS_AVAILABLE:
    __all__.extend([
        "TemporalAnalyzer",
        "TemporalConfig",
        "TemporalMetric",
        "TemporalResolution",
        "TemporalResult",
        "analyze_temporal_patterns",
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
    "enhanced coordinate system fixes, complete temporal analysis, increased scene footprint limits, and one-line functions"
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
        "temporal_analysis": {
            "complete_temporal_analysis": _TEMPORAL_ANALYSIS_AVAILABLE,  # NEW: Complete implementation
        },
        "advanced_features": {
            "asset_management": _ASSET_MANAGEMENT_AVAILABLE,
            "geopackage_export": _GEOPACKAGE_AVAILABLE,
            "geopackage_oneliners": _GEOPACKAGE_ONELINERS_AVAILABLE,
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
    
    print("PlanetScope-py Module Status (Enhanced + Complete Temporal Analysis)")
    print("=" * 70)
    
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
    
    # NEW: Complete Temporal Analysis
    print("\nTemporal Analysis (NEW - Complete Implementation):")
    temporal = status['temporal_analysis']
    for component, available in temporal.items():
        status_text = "Available" if available else "Not Available"
        print(f"  {component.replace('_', ' ').title()}: {status_text}")
    
    # Advanced Features
    print("\nAdvanced Features:")
    advanced = status['advanced_features']
    for component, available in advanced.items():
        status_text = "Available" if available else "Not Available"
        print(f"  {component.replace('_', ' ').title()}: {status_text}")
    
    # Workflows
    print("\nWorkflow API (Enhanced):")
    workflows = status['workflows']
    for component, available in workflows.items():
        status_text = "Available" if available else "Not Available"
        print(f"  {component.replace('_', ' ').title()}: {status_text}")
    
    # Summary
    total_components = (
        len(spatial) + len(temporal) + len(advanced) + len(workflows) + 2  # +2 for core components
    )
    available_components = (
        sum(spatial.values()) + sum(temporal.values()) + sum(advanced.values()) + sum(workflows.values()) + 
        int(status['core_infrastructure']) + int(status['planet_api_integration'])
    )
    
    print(f"\nSummary: {available_components}/{total_components} components available")
    
    if available_components < total_components:
        print("\nMissing components may require additional dependencies.")
        print("Refer to documentation for installation instructions.")


def get_usage_examples():
    """Display usage examples for the ENHANCED + COMPLETE TEMPORAL ANALYSIS simplified API."""
    print("PlanetScope-py Usage Examples (Enhanced + Complete Temporal Analysis)")
    print("=" * 75)
    
    print("\n1. Complete Spatial Analysis (1-line):")
    print("   from planetscope_py import analyze_roi_density")
    print("   result = analyze_roi_density(milan_roi, '2025-01-01/2025-01-31')")
    
    print("\n2. NEW: Complete Temporal Analysis (1-line):")
    print("   from planetscope_py import analyze_roi_temporal_patterns")
    print("   result = analyze_roi_temporal_patterns(milan_roi, '2025-01-01/2025-03-31')")
    print("   print(f'Mean coverage days: {result[\"temporal_result\"].temporal_stats[\"mean_coverage_days\"]:.1f}')")
    
    print("\n3. Ultra-Simple Analysis:")
    print("   from planetscope_py import quick_planet_analysis, quick_temporal_analysis")
    print("   # Spatial analysis")
    print("   spatial_result = quick_planet_analysis(milan_polygon, 'last_month')")
    print("   # Temporal analysis")
    print("   temporal_result = quick_temporal_analysis(milan_polygon, 'last_3_months')")
    
    print("\n4. Individual Plot Functions (1-line each):")
    print("   from planetscope_py import plot_density_map_only, plot_footprints_only")
    print("   ")
    print("   # Just get density map (FIXED orientation)")
    print("   fig = plot_density_map_only(milan_roi, 'last_month', 'density.png')")
    print("   ")
    print("   # Just get footprints (300+ scenes default)")
    print("   fig = plot_footprints_only(milan_roi, 'last_month', max_scenes=500)")
    
    print("\n5. GeoTIFF-Only Export (1-line):")
    print("   from planetscope_py import export_geotiff_only")
    print("   ")
    print("   # Just get GeoTIFF + QML files")
    print("   success = export_geotiff_only(milan_roi, 'last_month', 'output.tif')")
    
    print("\n6. NEW: Complete Temporal Analysis Examples:")
    if _TEMPORAL_ANALYSIS_AVAILABLE:
        print("   from planetscope_py import TemporalAnalyzer, TemporalConfig, TemporalMetric")
        print("   ")
        print("   # Custom temporal analysis")
        print("   config = TemporalConfig(")
        print("       spatial_resolution=100,")
        print("       metrics=[TemporalMetric.COVERAGE_DAYS, TemporalMetric.MEAN_INTERVAL],")
        print("       optimization_method='fast'")
        print("   )")
        print("   analyzer = TemporalAnalyzer(config)")
        print("   result = analyzer.analyze_temporal_patterns(scenes, roi, start, end)")
        print("   ")
        print("   # Export temporal GeoTIFFs")
        print("   files = analyzer.export_temporal_geotiffs(result, './output', clip_to_roi=True)")
        print("   print(f'Exported: {list(files.keys())}')")
    else:
        print("   # Temporal analysis not available")
    
    print("\n7. Performance Optimization:")
    print("   # Fast temporal analysis for large areas")
    print("   result = analyze_roi_temporal_patterns(")
    print("       roi, '2025-01-01/2025-06-30',")
    print("       spatial_resolution=500,  # Larger cells = faster")
    print("       optimization_level='fast'  # Use fast vectorized method")
    print("   )")
    print("   ")
    print("   # Accurate temporal analysis for detailed study")
    print("   result = analyze_roi_temporal_patterns(")
    print("       small_roi, '2025-01-01/2025-01-31',")
    print("       spatial_resolution=30,   # Fine resolution")
    print("       optimization_level='accurate'  # Cell-by-cell processing")
    print("   )")
    
    print("\nNEW ENHANCEMENTS in this version:")
    print("✓ Complete temporal analysis implementation")
    print("✓ Grid-based temporal pattern analysis (same as spatial density)")
    print("✓ Multiple temporal metrics (coverage days, intervals, density)")
    print("✓ FAST and ACCURATE optimization methods")
    print("✓ Professional GeoTIFF export with proper styling")
    print("✓ Integration with visualization module for temporal plots")
    print("✓ High-level one-line functions for temporal analysis")
    print("✓ Comprehensive statistics and metadata export")
    print("✓ Same coordinate system fixes as spatial density")
    print("✓ ROI clipping support for temporal analysis")


def demo_temporal_analysis():
    """Show complete temporal analysis capabilities and usage examples."""
    print("🕒 PlanetScope-py Complete Temporal Analysis Demo")
    print("=" * 55)
    
    print("\nCOMPLETE TEMPORAL ANALYSIS CAPABILITIES:")
    print("─" * 45)
    
    print("\n✅ Grid-Based Temporal Analysis:")
    print("   • Same grid approach as spatial density analysis")
    print("   • Coordinate system fixes applied")
    print("   • ROI clipping support (clip_to_roi parameter)")
    print("   • Daily temporal resolution")
    print("   • FAST and ACCURATE optimization methods")
    
    print("\n✅ Temporal Metrics Calculated:")
    print("   • Coverage Days: Number of days with scene coverage per grid cell")
    print("   • Mean/Median Intervals: Days between consecutive scenes")
    print("   • Temporal Density: Scenes per day over the analysis period")
    print("   • Coverage Frequency: Percentage of days with coverage")
    print("   • Min/Max Intervals: Range of temporal gaps")
    
    print("\n✅ Professional Outputs:")
    print("   • Multiple GeoTIFF files (one per metric)")
    print("   • QML style files for QGIS visualization")
    print("   • JSON metadata with comprehensive statistics")
    print("   • Integration with visualization module")
    
    print("\n✅ Performance Optimization:")
    print("   • FAST method: Vectorized operations (10-50x faster)")
    print("   • ACCURATE method: Cell-by-cell processing (slower but precise)")
    print("   • AUTO selection: Automatically chooses based on grid size")
    
    print("\nUSAGE EXAMPLES:")
    print("─" * 15)
    
    print("\n1. HIGH-LEVEL ONE-LINER:")
    print("   from planetscope_py import analyze_roi_temporal_patterns")
    print("   ")
    print("   result = analyze_roi_temporal_patterns(")
    print("       milan_roi, '2025-01-01/2025-03-31',")
    print("       spatial_resolution=100, clip_to_roi=True,")
    print("       optimization_level='fast'")
    print("   )")
    print("   ")
    print("   print(f'Mean coverage: {result[\"temporal_result\"].temporal_stats[\"mean_coverage_days\"]:.1f} days')")
    print("   print(f'Exported files: {list(result[\"exports\"].keys())}')")
    
    print("\n2. CUSTOM CONFIGURATION:")
    if _TEMPORAL_ANALYSIS_AVAILABLE:
        print("   from planetscope_py import TemporalAnalyzer, TemporalConfig, TemporalMetric")
        print("   ")
        print("   # Configure specific metrics")
        print("   config = TemporalConfig(")
        print("       spatial_resolution=50,")
        print("       metrics=[")
        print("           TemporalMetric.COVERAGE_DAYS,")
        print("           TemporalMetric.MEAN_INTERVAL,")
        print("           TemporalMetric.TEMPORAL_DENSITY")
        print("       ],")
        print("       min_scenes_per_cell=3,")
        print("       optimization_method='fast'")
        print("   )")
        print("   ")
        print("   analyzer = TemporalAnalyzer(config)")
        print("   result = analyzer.analyze_temporal_patterns(scenes, roi, start, end)")
    else:
        print("   # Temporal analysis module not yet available")
        print("   # Will be available once temporal_analysis.py is created")
    
    print("\n3. SIMPLIFIED WORKFLOW:")
    print("   from planetscope_py import quick_temporal_analysis")
    print("   ")
    print("   # Ultra-simple temporal analysis")
    print("   result = quick_temporal_analysis(milan_polygon, 'last_3_months')")
    print("   ")
    print("   # Access results")
    print("   temporal_result = result['temporal_result']")
    print("   for metric, array in temporal_result.metric_arrays.items():")
    print("       print(f'{metric.value}: {array.shape} grid')")
    
    print("\nPERFORMANCE COMPARISON:")
    print("─" * 25)
    print("┌─────────────────────┬──────────────────────┬─────────────────────┐")
    print("│ Grid Size           │ FAST Method          │ ACCURATE Method     │")
    print("├─────────────────────┼──────────────────────┼─────────────────────┤")
    print("│ Small (<100k cells) │ 30 seconds           │ 2-5 minutes         │")
    print("│ Medium (100k-500k)  │ 2-5 minutes          │ 10-30 minutes       │")
    print("│ Large (>500k cells) │ 5-15 minutes         │ 1-3 hours           │")
    print("│ Very Large (>1M)    │ 10-30 minutes        │ 3+ hours            │")
    print("└─────────────────────┴──────────────────────┴─────────────────────┘")
    
    print("\nWHY TEMPORAL ANALYSIS MATTERS:")
    print("• Identify acquisition gaps and irregular coverage")
    print("• Plan optimal data acquisition strategies")
    print("• Understand seasonal and temporal patterns")
    print("• Assess data availability for time series analysis")
    print("• Optimize monitoring and change detection workflows")
    
    print(f"\nTEMPORAL ANALYSIS STATUS: {'✅ AVAILABLE' if _TEMPORAL_ANALYSIS_AVAILABLE else '❌ NOT AVAILABLE'}")


# Enhanced help function
def help():
    """Display comprehensive help for the enhanced PlanetScope-py library."""
    print("PlanetScope-py Enhanced Help (+ Complete Temporal Analysis)")
    print("=" * 65)
    print()
    print("This library provides professional tools for PlanetScope satellite imagery analysis")
    print("with enhanced coordinate system fixes, complete temporal analysis, simplified one-line functions, and GeoPackage export.")
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
    print("• Missing temporal pattern analysis capabilities")
    print("• Performance issues with large grids")
    if _GEOPACKAGE_ONELINERS_AVAILABLE:
        print("• Complex GeoPackage creation workflows")
        print("• Manual scene clipping and attribute management")


if __name__ == "__main__":
    demo_temporal_analysis()