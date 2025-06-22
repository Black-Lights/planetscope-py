#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PlanetScope-py: Professional Python library for PlanetScope satellite imagery analysis.

A comprehensive tool for scene discovery, metadata analysis, spatial-temporal
density calculations, asset management, and data export using Planet's APIs.
"""

from ._version import __version__

# ============================================
# CORE INFRASTRUCTURE (Always Available)
# ============================================

# Core Infrastructure
from .auth import PlanetAuth
from .config import PlanetScopeConfig, default_config
from .exceptions import (
    PlanetScopeError,
    AuthenticationError,
    ValidationError,
    RateLimitError,
    APIError,
    ConfigurationError,
    AssetError,
)
from .utils import (
    validate_geometry,
    calculate_area_km2,
    transform_geometry,
    create_bounding_box,
    buffer_geometry,
)

# Planet API Integration
from .query import PlanetScopeQuery
from .metadata import MetadataProcessor
from .rate_limiter import RateLimiter, RetryableSession, CircuitBreaker

# ============================================
# SPATIAL ANALYSIS (Phase 3)
# ============================================

# Spatial Analysis Engine
_SPATIAL_ANALYSIS_AVAILABLE = False
try:
    from .density_engine import (
        SpatialDensityEngine,
        DensityConfig,
        DensityMethod,
        DensityResult,
    )

    _SPATIAL_ANALYSIS_AVAILABLE = True
except ImportError:
    pass

# Advanced Spatial Components
_ADAPTIVE_GRID_AVAILABLE = False
try:
    from .adaptive_grid import AdaptiveGridEngine, AdaptiveGridConfig

    _ADAPTIVE_GRID_AVAILABLE = True
except ImportError:
    pass

_OPTIMIZER_AVAILABLE = False
try:
    from .optimizer import (
        PerformanceOptimizer,
        DatasetCharacteristics,
        PerformanceProfile,
    )

    _OPTIMIZER_AVAILABLE = True
except ImportError:
    pass

_VISUALIZATION_AVAILABLE = False
try:
    from .visualization import DensityVisualizer

    _VISUALIZATION_AVAILABLE = True
except ImportError:
    pass

# ============================================
# PHASE 4: Enhanced Features
# ============================================

# Temporal Analysis
_TEMPORAL_ANALYSIS_AVAILABLE = False
try:
    from .temporal_analysis import (
        TemporalAnalyzer,
        TemporalConfig,
        TemporalResolution,
        SeasonalPeriod,
        TemporalGap,
        SeasonalPattern,
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
        GeoPackageManager,
        GeoPackageConfig,
        LayerInfo,
        RasterInfo,
    )

    _GEOPACKAGE_AVAILABLE = True
except ImportError:
    pass

# Preview Management
_PREVIEW_MANAGEMENT_AVAILABLE = False
try:
    from .preview_manager import PreviewManager

    _PREVIEW_MANAGEMENT_AVAILABLE = True
except ImportError:
    pass

# Interactive Management (optional)
_INTERACTIVE_AVAILABLE = False
try:
    from .interactive_manager import InteractiveManager

    _INTERACTIVE_AVAILABLE = True
except ImportError:
    pass

# ============================================
# EXPORTS (__all__)
# ============================================

__all__ = [
    # Version
    "__version__",
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

# Add spatial analysis exports if available
if _SPATIAL_ANALYSIS_AVAILABLE:
    __all__.extend(
        ["SpatialDensityEngine", "DensityConfig", "DensityMethod", "DensityResult"]
    )

if _ADAPTIVE_GRID_AVAILABLE:
    __all__.extend(["AdaptiveGridEngine", "AdaptiveGridConfig"])

if _OPTIMIZER_AVAILABLE:
    __all__.extend(
        ["PerformanceOptimizer", "DatasetCharacteristics", "PerformanceProfile"]
    )

if _VISUALIZATION_AVAILABLE:
    __all__.extend(["DensityVisualizer"])

# Add Phase 4 exports if available
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

if _PREVIEW_MANAGEMENT_AVAILABLE:
    __all__.extend(["PreviewManager"])

if _INTERACTIVE_AVAILABLE:
    __all__.extend(["InteractiveManager"])

# ============================================
# PACKAGE METADATA
# ============================================

__author__ = "Ammar & Umayr"
__email__ = "mohammadammarmughees@gmail.com"
__description__ = (
    "Professional Python library for PlanetScope satellite imagery analysis with "
    "enhanced temporal analysis, asset management, and data export capabilities"
)
__url__ = "https://github.com/Black-Lights/planetscope-py"
__license__ = "MIT"

# ============================================
# DIAGNOSTIC FUNCTIONS
# ============================================


def get_component_status():
    """Get the current availability status of all components."""
    return {
        "core_infrastructure": True,
        "planet_api_integration": True,
        "spatial_analysis": {
            "density_engine": _SPATIAL_ANALYSIS_AVAILABLE,
            "adaptive_grid": _ADAPTIVE_GRID_AVAILABLE,
            "optimizer": _OPTIMIZER_AVAILABLE,
            "visualization": _VISUALIZATION_AVAILABLE,
        },
        "phase4_features": {
            "temporal_analysis": _TEMPORAL_ANALYSIS_AVAILABLE,
            "asset_management": _ASSET_MANAGEMENT_AVAILABLE,
            "geopackage_export": _GEOPACKAGE_AVAILABLE,
            "preview_management": _PREVIEW_MANAGEMENT_AVAILABLE,
            "interactive_features": _INTERACTIVE_AVAILABLE,
        },
    }


def check_phase4_status():
    """Check Phase 4 component availability."""
    phase4_components = {
        "temporal_analysis": _TEMPORAL_ANALYSIS_AVAILABLE,
        "asset_management": _ASSET_MANAGEMENT_AVAILABLE,
        "geopackage_export": _GEOPACKAGE_AVAILABLE,
        "preview_management": _PREVIEW_MANAGEMENT_AVAILABLE,
        "interactive_features": _INTERACTIVE_AVAILABLE,
    }
    
    available_count = sum(phase4_components.values())
    total_count = len(phase4_components)
    
    print(f"Phase 4 Status: {available_count}/{total_count} components available")
    
    for component, available in phase4_components.items():
        status = "Available" if available else "Not Available"
        print(f"  {component}: {status}")
    
    return phase4_components


def check_preview_capabilities():
    """Check preview-related capabilities."""
    capabilities = {
        "scene_preview_urls": True,  # Always available (core query functionality)
        "tile_service_api": True,    # Always available (core query functionality)
        "interactive_maps": _PREVIEW_MANAGEMENT_AVAILABLE,
        "folium_integration": False,
        "static_tile_urls": True,    # Always available (core query functionality)
    }
    
    # Check for optional dependencies
    try:
        import folium
        capabilities["folium_integration"] = True
    except ImportError:
        pass
    
    try:
        import shapely
        capabilities["shapely_support"] = True
    except ImportError:
        capabilities["shapely_support"] = False
    
    print("Preview Capabilities:")
    for capability, available in capabilities.items():
        status = "Available" if available else "Not Available"
        print(f"  {capability}: {status}")
    
    if not capabilities["folium_integration"]:
        print("\nTo enable interactive maps: pip install folium")
    
    if not capabilities["shapely_support"]:
        print("To enable full geometry support: pip install shapely")
    
    return capabilities


# Convenience function for backward compatibility
def get_scene_previews_help():
    """Display help information about scene preview functionality."""
    print("PlanetScope Scene Preview Help")
    print("=" * 35)
    print()
    print("The library now uses Planet's official Tile Service API for previews.")
    print()
    print("Basic Usage:")
    print("  query = PlanetScopeQuery()")
    print("  results = query.search_scenes(geometry, start_date, end_date)")
    print("  scene_ids = [scene['id'] for scene in results['features'][:5]]")
    print("  previews = query.get_scene_previews(scene_ids)")
    print()
    print("Returns tile URL templates for interactive maps (Folium, Leaflet, etc.)")
    print()
    print("For static URLs with actual coordinates:")
    print("  tile_info = query.get_scene_tile_urls(scene_ids, zoom_level=12)")
    print()
    print("For interactive maps:")
    if _PREVIEW_MANAGEMENT_AVAILABLE:
        print("  preview_manager = PreviewManager(query)")
        print("  map_obj = preview_manager.create_interactive_map(results, roi)")
        print("  preview_manager.save_interactive_map(map_obj, 'scenes.html')")
    else:
        print("  Install preview_manager module for enhanced capabilities")
    print()
    print("Check capabilities with: check_preview_capabilities()")