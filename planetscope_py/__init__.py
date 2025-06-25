#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PlanetScope-py: Professional Python library for PlanetScope satellite imagery analysis.

ENHANCED VERSION with clean temporal analysis integration and critical fixes.

NEW FEATURES:
- Clean temporal analysis module (grid-based temporal patterns)
- Individual plot access functions
- Fixed coordinate system display
- Increased scene footprint limits
- GeoTIFF-only export functions
- Complete temporal analysis implementation

CRITICAL FIXES (v4.1.0):
- Enhanced scene ID extraction (handles different API response formats)
- JSON serialization fixes for metadata export
- Temporal analysis visualizations with turbo colormap
- Summary table formatting consistency
- Interactive and preview manager integration

Author: Ammar & Umayr
Version: 4.1.0 (Enhanced + Metadata Fixes + JSON Serialization)
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
    print("✓ Visualization module loaded successfully (v4.1.0 with turbo colormap)")
except ImportError as e:
    _VISUALIZATION_AVAILABLE = False
    warnings.warn(f"Visualization not available: {e}")

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

# Enhanced Preview Management - FIXED IMPORT
_PREVIEW_MANAGEMENT_AVAILABLE = False
try:
    from .preview_manager import PreviewManager
    _PREVIEW_MANAGEMENT_AVAILABLE = True
    print("✓ Preview manager loaded successfully (v4.1.0 with enhanced integration)")
except ImportError as e:
    _PREVIEW_MANAGEMENT_AVAILABLE = False
    warnings.warn(f"Preview manager not available: {e}. Install: pip install folium shapely")

# Enhanced Interactive Management - COMPLETE UPDATE  
_INTERACTIVE_AVAILABLE = False
try:
    from .interactive_manager import (
        InteractiveManager,
        create_roi_selector,
        quick_roi_map,
        jupyter_roi_selector,
        jupyter_quick_analysis,
        # NEW: Shapely integration functions
        jupyter_get_shapely_roi,
        export_shapely_objects,
        create_shapely_polygon_from_coords,
        # FIXED: quick_preview_with_shapely is actually in interactive_manager.py
        quick_preview_with_shapely,
        # NEW: Workflow display function
        display_jupyter_workflow_example,
    )
    _INTERACTIVE_AVAILABLE = True
    print("✓ Interactive manager loaded successfully (v4.1.0 with config fixes)")
except ImportError as e:
    _INTERACTIVE_AVAILABLE = False
    warnings.warn(f"Interactive manager not available: {e}. Install: pip install folium shapely")

# Enhanced Workflow API with Fixes
_WORKFLOWS_AVAILABLE = False
try:
    from .workflows import (
        analyze_density, quick_analysis, batch_analysis, temporal_analysis_workflow,
        # NEW: One-line functions for individual outputs
        quick_density_plot, quick_footprints_plot, quick_geotiff_export
    )
    _WORKFLOWS_AVAILABLE = True
    print("✓ Workflows module loaded successfully (v4.1.0 with JSON serialization fixes)")
except ImportError as e:
    _WORKFLOWS_AVAILABLE = False
    warnings.warn(f"Workflows not available: {e}")

# Configuration Presets - FIXED
_CONFIG_PRESETS_AVAILABLE = False
try:
    from .config import PlanetScopeConfig, default_config
    _CONFIG_PRESETS_AVAILABLE = True
    print("✓ Configuration module loaded successfully (v4.1.0 with enhanced metadata extraction)")
except ImportError as e:
    _CONFIG_PRESETS_AVAILABLE = False
    warnings.warn(f"Configuration not available: {e}")


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


def create_scene_preview_map(
    roi: Union["Polygon", list, dict, str],  # Can now accept file paths
    time_period: str = "last_month",
    max_scenes: int = 10,
    **kwargs
) -> Optional[Any]:
    """
    HIGH-LEVEL API: Create interactive preview map with actual satellite imagery.
    
    Enhanced function that combines ROI loading with preview map creation.
    
    Args:
        roi: Region of interest as:
            - Shapely Polygon object
            - Coordinate list [[lon, lat], ...]
            - GeoJSON dict
            - File path to GeoJSON/Shapefile
        time_period: Time period specification:
            - "last_month": Previous 30 days
            - "last_3_months": Previous 90 days
            - "YYYY-MM-DD/YYYY-MM-DD": Custom date range
        max_scenes: Maximum scenes to display on map
        **kwargs: Additional parameters:
            - cloud_cover_max (float): Maximum cloud cover threshold
            - show_scene_info (bool): Show scene information in popups
    
    Returns:
        Folium map with actual Planet satellite imagery overlays
    
    Example:
        >>> from planetscope_py import create_scene_preview_map
        >>> 
        >>> # Using file path
        >>> preview_map = create_scene_preview_map("roi_selection.geojson", "last_month")
        >>> 
        >>> # Using Shapely polygon
        >>> preview_map = create_scene_preview_map(roi_polygon, "2025-01-01/2025-01-31")
        >>> 
        >>> # Display in Jupyter
        >>> preview_map
    """
    if not _PREVIEW_MANAGEMENT_AVAILABLE or not _PLANET_API_AVAILABLE:
        raise ImportError(
            "Preview and query functionality not available. "
            "Install missing dependencies: pip install folium shapely"
        )
    
    try:
        # Load ROI from various sources
        if isinstance(roi, str):
            # File path
            if not _INTERACTIVE_AVAILABLE:
                raise ImportError("Interactive manager needed for file loading")
            
            from .interactive_manager import InteractiveManager
            manager = InteractiveManager()
            polygons = manager.load_roi_from_file(roi)
            roi_polygon = polygons[0] if polygons else None
            
            if roi_polygon is None:
                raise ValidationError(f"Could not load ROI from {roi}")
                
        elif isinstance(roi, list):
            # Coordinate list
            from shapely.geometry import Polygon
            roi_polygon = Polygon(roi)
            
        elif isinstance(roi, dict):
            # GeoJSON-like dict
            from shapely.geometry import shape
            roi_polygon = shape(roi)
            
        elif hasattr(roi, 'exterior'):
            # Already a Shapely polygon
            roi_polygon = roi
            
        else:
            raise ValidationError(f"Unsupported ROI type: {type(roi)}")
        
        # Parse time period
        if time_period == "last_month":
            from datetime import datetime, timedelta
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            time_period_str = f"{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
        elif time_period == "last_3_months":
            from datetime import datetime, timedelta
            end_date = datetime.now()
            start_date = end_date - timedelta(days=90)
            time_period_str = f"{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
        else:
            time_period_str = time_period
        
        # FIXED: Use the enhanced preview function from interactive_manager
        if _INTERACTIVE_AVAILABLE:
            from .query import PlanetScopeQuery
            query = PlanetScopeQuery()
            
            return quick_preview_with_shapely(
                query_instance=query,
                roi_polygon=roi_polygon,
                time_period=time_period_str,
                max_scenes=max_scenes
            )
        else:
            raise ImportError("Interactive manager not available")
            
    except Exception as e:
        logger.error(f"Scene preview creation failed: {e}")
        raise PlanetScopeError(f"Failed to create scene preview: {e}")


def jupyter_complete_workflow_demo():
    """
    Display complete Jupyter workflow demonstration with both ROI selection and preview.
    
    Shows step-by-step process from ROI selection to imagery preview to analysis.
    """
    try:
        from IPython.display import display, HTML
        
        workflow_html = """
        <div style="border: 3px solid #007acc; border-radius: 12px; padding: 25px; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); font-family: 'Segoe UI', Arial, sans-serif; margin: 20px 0;">
            <h1 style="color: #007acc; margin-top: 0; text-align: center;"> Complete PlanetScope-py Jupyter Workflow</h1>
            
            <div style="background-color: #e8f4fd; padding: 20px; border-radius: 8px; margin: 15px 0; border-left: 5px solid #007acc;">
                <h2 style="color: #005c99; margin-top: 0;"> Step 1: Interactive ROI Selection</h2>
                <pre style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; border: 1px solid #dee2e6; overflow-x: auto;"><code># Create interactive map for ROI selection
from planetscope_py import jupyter_roi_selector

map_obj = jupyter_roi_selector("milan")
map_obj  # Draw your ROI on this map and export as 'roi_selection.geojson'</code></pre>
            </div>
            
            <div style="background-color: #fff3cd; padding: 20px; border-radius: 8px; margin: 15px 0; border-left: 5px solid #ffc107;">
                <h2 style="color: #856404; margin-top: 0;"> Step 2: Preview Actual Satellite Imagery</h2>
                <pre style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; border: 1px solid #dee2e6; overflow-x: auto;"><code># Create preview map with actual Planet satellite imagery
from planetscope_py import create_scene_preview_map

preview_map = create_scene_preview_map(
    "roi_selection.geojson",  # Your exported ROI
    "2025-01-01/2025-01-31",  # Time period
    max_scenes=20             # Show up to 20 scenes
)
preview_map  # Interactive map with real satellite imagery!</code></pre>
            </div>
            
            <div style="background-color: #d4edda; padding: 20px; border-radius: 8px; margin: 15px 0; border-left: 5px solid #28a745;">
                <h2 style="color: #155724; margin-top: 0;"> Step 3: Run Spatial Density Analysis</h2>
                <pre style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; border: 1px solid #dee2e6; overflow-x: auto;"><code># Run spatial analysis on your ROI
from planetscope_py import jupyter_quick_analysis

spatial_result = jupyter_quick_analysis(
    "roi_selection.geojson",
    "2025-01-01/2025-01-31", 
    "spatial"
)

print(f"Found {spatial_result['scenes_found']} scenes")
print(f"Mean density: {spatial_result['density_result'].stats['mean']:.1f}")

# View visualizations
spatial_result['visualizations']</code></pre>
            </div>
            
            <div style="background-color: #f8d7da; padding: 20px; border-radius: 8px; margin: 15px 0; border-left: 5px solid #dc3545;">
                <h2 style="color: #721c24; margin-top: 0;"> Step 4: Run Temporal Pattern Analysis</h2>
                <pre style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; border: 1px solid #dee2e6; overflow-x: auto;"><code># Run temporal analysis to understand acquisition patterns
temporal_result = jupyter_quick_analysis(
    "roi_selection.geojson",
    "2025-01-01/2025-03-31",  # 3-month period for temporal analysis
    "temporal"
)

print(f"Temporal scenes: {temporal_result['scenes_found']}")
print(f"Mean coverage days: {temporal_result['temporal_result'].temporal_stats['mean_coverage_days']:.1f}")

# View temporal visualizations (now with turbo colormap!)
temporal_result['visualizations']</code></pre>
            </div>
            
            <div style="background-color: #e2e3e5; padding: 20px; border-radius: 8px; margin: 15px 0; border-left: 5px solid #6c757d;">
                <h2 style="color: #495057; margin-top: 0;"> Step 5: Work with Shapely Objects</h2>
                <pre style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; border: 1px solid #dee2e6; overflow-x: auto;"><code># Get Shapely polygon object for custom analysis
from planetscope_py import jupyter_get_shapely_roi

roi_polygon = jupyter_get_shapely_roi("roi_selection.geojson")
print(f"ROI area: {roi_polygon.area:.6f} square degrees")
print(f"ROI bounds: {roi_polygon.bounds}")

# Use directly with any analysis function
from planetscope_py import analyze_roi_density
custom_result = analyze_roi_density(roi_polygon, "2025-01-01/2025-01-31", resolution=50)</code></pre>
            </div>
            
            <div style="background-color: #d1ecf1; padding: 20px; border-radius: 8px; margin: 15px 0; border-left: 5px solid #0c5460;">
                <h2 style="color: #0c5460; margin-top: 0;"> Step 6: Export and Save Results</h2>
                <pre style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; border: 1px solid #dee2e6; overflow-x: auto;"><code># All outputs are automatically saved with proper metadata
print("Spatial analysis outputs:")
print(f"  Directory: {spatial_result['output_directory']}")
print(f"  Files: {list(spatial_result['exports'].keys())}")

print("\\nTemporal analysis outputs:")
print(f"  Directory: {temporal_result['output_directory']}")
print(f"  Files: {list(temporal_result['exports'].keys())}")

# Export ROI in multiple formats
from planetscope_py import export_shapely_objects
roi_exports = export_shapely_objects([roi_polygon])
print(f"\\nROI exported as: {list(roi_exports.keys())}")</code></pre>
            </div>
            
            <div style="background-color: #fff; padding: 20px; border-radius: 8px; margin: 15px 0; border: 2px solid #007acc;">
                <h2 style="color: #007acc; margin-top: 0;"> Key Benefits of This Workflow (v4.1.0)</h2>
                <ul style="line-height: 1.8; font-size: 16px;">
                    <li><strong>Visual ROI Selection:</strong> No need to manually code coordinates</li>
                    <li><strong>Imagery Preview:</strong> See actual satellite data before analysis</li>
                    <li><strong>Comprehensive Analysis:</strong> Both spatial density and temporal patterns</li>
                    <li><strong>Professional Outputs:</strong> GeoTIFF files, visualizations, and complete metadata</li>
                    <li><strong>Enhanced Scene ID Extraction:</strong> Works with all Planet API endpoints</li>
                    <li><strong>Fixed JSON Serialization:</strong> No more truncated metadata files</li>
                    <li><strong>Improved Visualizations:</strong> Turbo colormap for better temporal analysis</li>
                    <li><strong>Shapely Integration:</strong> Work directly with geometry objects</li>
                    <li><strong>Jupyter Optimized:</strong> All visualizations display in notebooks</li>
                </ul>
            </div>
        </div>
        """
        
        display(HTML(workflow_html))
        
    except ImportError:
        # Fallback for non-Jupyter environments
        print("🛰️ Complete PlanetScope-py Jupyter Workflow")
        print("=" * 50)
        print()
        print("STEP 1: Interactive ROI Selection")
        print("from planetscope_py import jupyter_roi_selector")
        print("map_obj = jupyter_roi_selector('milan')")
        print()
        print("STEP 2: Preview Satellite Imagery")
        print("from planetscope_py import create_scene_preview_map")
        print("preview_map = create_scene_preview_map('roi_selection.geojson', 'last_month')")
        print()
        print("STEP 3: Run Analysis")
        print("from planetscope_py import jupyter_quick_analysis")
        print("result = jupyter_quick_analysis('roi_selection.geojson', 'last_month', 'both')")
        print()
        print("For full workflow guide, run this in a Jupyter notebook!")


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

# FIXED: Enhanced exports section
if _PREVIEW_MANAGEMENT_AVAILABLE:
    __all__.extend([
        "PreviewManager",
    ])

if _INTERACTIVE_AVAILABLE:
    __all__.extend([
        "InteractiveManager",
        "create_roi_selector",
        "quick_roi_map",
        "jupyter_roi_selector", 
        "jupyter_quick_analysis",
        # FIXED: Shapely integration functions (including quick_preview_with_shapely)
        "jupyter_get_shapely_roi",
        "export_shapely_objects",
        "create_shapely_polygon_from_coords",
        "quick_preview_with_shapely",  # FIXED: Now properly exported from interactive_manager
        # NEW: Workflow helper
        "display_jupyter_workflow_example",
        "jupyter_complete_workflow_demo",
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
    "enhanced coordinate system fixes, complete temporal analysis, metadata extraction improvements, "
    "JSON serialization fixes, and enhanced visualization capabilities"
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
    
    print("PlanetScope-py Module Status (v4.1.0 - Enhanced + Metadata Fixes)")
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
    print("\nTemporal Analysis (Complete Implementation with Turbo Colormap):")
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
    print("\nWorkflow API (Enhanced with JSON Fixes):")
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
    print("PlanetScope-py Usage Examples (v4.1.0 - Enhanced + Metadata Fixes)")
    print("=" * 75)
    
    print("\n1. Complete Spatial Analysis (1-line):")
    print("   from planetscope_py import analyze_roi_density")
    print("   result = analyze_roi_density(milan_roi, '2025-01-01/2025-01-31')")
    
    print("\n2. NEW: Complete Temporal Analysis (1-line with turbo colormap):")
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
    
    print("\n6. FIXED: Preview with Shapely (now works!):")
    print("   from planetscope_py import quick_preview_with_shapely, PlanetScopeQuery")
    print("   ")
    print("   query = PlanetScopeQuery()")
    print("   preview_map = quick_preview_with_shapely(")
    print("       query, milan_roi, '2025-01-01/2025-01-31', max_scenes=20")
    print("   )")
    print("   preview_map  # Display in Jupyter")
    
    print("\n7. Performance Optimization:")
    print("   # Fast temporal analysis for large areas")
    print("   result = analyze_roi_temporal_patterns(")
    print("       roi, '2025-01-01/2025-06-30',")
    print("       spatial_resolution=500,  # Larger cells = faster")
    print("       optimization_level='fast'  # Use fast vectorized method")
    print("   )")
    
    print("\nFIXED ISSUES IN v4.1.0:")
    print("✓ Enhanced scene ID extraction from all Planet API response formats")
    print("✓ JSON serialization fixes for complete metadata export")
    print("✓ Temporal analysis visualizations with turbo colormap")
    print("✓ Summary table formatting matches spatial density analysis")
    print("✓ Interactive and preview manager integration")
    print("✓ No more truncated metadata JSON files")


def demo_temporal_analysis():
    """Show complete temporal analysis capabilities and usage examples."""
    print(" PlanetScope-py Complete Temporal Analysis Demo (v4.1.0)")
    print("=" * 60)
    
    print("\nCOMPLETE TEMPORAL ANALYSIS CAPABILITIES:")
    print("─" * 45)
    
    print("\n✓ Grid-Based Temporal Analysis:")
    print("   • Same grid approach as spatial density analysis")
    print("   • Coordinate system fixes applied")
    print("   • ROI clipping support (clip_to_roi parameter)")
    print("   • Daily temporal resolution")
    print("   • FAST and ACCURATE optimization methods")
    
    print("\n✓ Temporal Metrics Calculated:")
    print("   • Coverage Days: Number of days with scene coverage per grid cell")
    print("   • Mean/Median Intervals: Days between consecutive scenes")
    print("   • Temporal Density: Scenes per day over the analysis period")
    print("   • Coverage Frequency: Percentage of days with coverage")
    print("   • Min/Max Intervals: Range of temporal gaps")
    
    print("\n✓ Professional Outputs (v4.1.0 Enhanced):")
    print("   • Multiple GeoTIFF files (one per metric)")
    print("   • QML style files for QGIS visualization with turbo colormap")
    print("   • Complete JSON metadata with proper serialization")
    print("   • Enhanced summary tables matching spatial density format")
    print("   • Integration with visualization module")
    
    print("\n✓ Performance Optimization:")
    print("   • FAST method: Vectorized operations (10-50x faster)")
    print("   • ACCURATE method: Cell-by-cell processing (slower but precise)")
    print("   • AUTO selection: Automatically chooses based on grid size")
    
    print(f"\nTEMPORAL ANALYSIS STATUS: {'✓ AVAILABLE' if _TEMPORAL_ANALYSIS_AVAILABLE else ' NOT AVAILABLE'}")


# Enhanced help function
def help():
    """Display comprehensive help for the enhanced PlanetScope-py library."""
    print("PlanetScope-py Enhanced Help (v4.1.0 - Metadata & JSON Fixes)")
    print("=" * 70)
    print()
    print("This library provides professional tools for PlanetScope satellite imagery analysis")
    print("with enhanced coordinate system fixes, complete temporal analysis, improved metadata")
    print("extraction, JSON serialization fixes, and enhanced visualization capabilities.")
    print()
    
    check_module_status()
    print()
    get_usage_examples()
    print()
    
    print("For more detailed documentation, visit:")
    print("https://github.com/Black-Lights/planetscope-py")
    print()
    print("Common Issues Fixed in v4.1.0:")
    print("• Enhanced scene ID extraction from different Planet API endpoints")
    print("• JSON serialization errors in metadata export (numpy type conversion)")
    print("• Truncated metadata JSON files")
    print("• Temporal analysis visualizations now use turbo colormap")
    print("• Summary table formatting consistency between spatial and temporal")
    print("• Interactive and preview manager configuration issues")
    print("• Mirrored/flipped density maps")
    print("• Limited scene footprint display (50 → 150+)")
    print("• Complex multi-step workflows") 
    print("• PROJ database compatibility issues")
    print("• Missing temporal pattern analysis capabilities")
    print("• Performance issues with large grids")
    print("• ImportError for quick_preview_with_shapely function")
    if _GEOPACKAGE_ONELINERS_AVAILABLE:
        print("• Complex GeoPackage creation workflows")
        print("• Manual scene clipping and attribute management")


# No main execution block in __init__.py - demos should be in separate files