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
# PHASE 4: FORCE IMPORTS (No silent failures)
# ============================================

# Temporal Analysis - Force import since we know it works
print("Loading temporal analysis...")
from .temporal_analysis import (
    TemporalAnalyzer,
    TemporalConfig,
    TemporalResolution,
    SeasonalPeriod,
    TemporalGap,
    SeasonalPattern,
)

_TEMPORAL_ANALYSIS_AVAILABLE = True
print("✅ Temporal analysis loaded successfully")

# Asset Management - Force import since debug showed it works
print("Loading asset management...")
from .asset_manager import AssetManager, AssetStatus, QuotaInfo, DownloadJob

_ASSET_MANAGEMENT_AVAILABLE = True
print("✅ Asset management loaded successfully")

# GeoPackage Export - Force import since debug showed it works
print("Loading GeoPackage management...")
from .geopackage_manager import (
    GeoPackageManager,
    GeoPackageConfig,
    LayerInfo,
    RasterInfo,
)

_GEOPACKAGE_AVAILABLE = True
print("✅ GeoPackage management loaded successfully")

# Interactive Management (optional)
_INTERACTIVE_AVAILABLE = False
try:
    from .interactive_manager import InteractiveManager

    _INTERACTIVE_AVAILABLE = True
    print("✅ Interactive management loaded successfully")
except ImportError:
    print("ℹ️ Interactive management not available (optional)")

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

# Add Phase 4 exports (these should always be available now)
__all__.extend(
    [
        # Temporal Analysis
        "TemporalAnalyzer",
        "TemporalConfig",
        "TemporalResolution",
        "SeasonalPeriod",
        "TemporalGap",
        "SeasonalPattern",
        # Asset Management
        "AssetManager",
        "AssetStatus",
        "QuotaInfo",
        "DownloadJob",
        # GeoPackage Export
        "GeoPackageManager",
        "GeoPackageConfig",
        "LayerInfo",
        "RasterInfo",
    ]
)

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
            "interactive_features": _INTERACTIVE_AVAILABLE,
        },
    }


def check_phase4_status():
    """Check Phase 4 component availability."""
    print("=== PlanetScope-py Phase 4 Status ===")
    print(f"Temporal Analysis: {'✅' if _TEMPORAL_ANALYSIS_AVAILABLE else '❌'}")
    print(f"Asset Management: {'✅' if _ASSET_MANAGEMENT_AVAILABLE else '❌'}")
    print(f"GeoPackage Export: {'✅' if _GEOPACKAGE_AVAILABLE else '❌'}")
    print(f"Interactive Features: {'✅' if _INTERACTIVE_AVAILABLE else '❌'}")

    return {
        "temporal_analysis": _TEMPORAL_ANALYSIS_AVAILABLE,
        "asset_management": _ASSET_MANAGEMENT_AVAILABLE,
        "geopackage_export": _GEOPACKAGE_AVAILABLE,
        "interactive_features": _INTERACTIVE_AVAILABLE,
    }


def has_temporal_analysis():
    """Check if temporal analysis components are available."""
    return _TEMPORAL_ANALYSIS_AVAILABLE


def has_asset_management():
    """Check if asset management components are available."""
    return _ASSET_MANAGEMENT_AVAILABLE


def has_geopackage_export():
    """Check if GeoPackage export components are available."""
    return _GEOPACKAGE_AVAILABLE


def has_phase4_complete():
    """Check if all Phase 4 features are available."""
    return all(
        [
            _TEMPORAL_ANALYSIS_AVAILABLE,
            _ASSET_MANAGEMENT_AVAILABLE,
            _GEOPACKAGE_AVAILABLE,
        ]
    )


def show_library_status():
    """Display current library component availability."""
    print("PlanetScope-py Library Status")
    print("=" * 40)

    status = get_component_status()
    for component, component_status in status.items():
        if isinstance(component_status, dict):
            print(f"{component.replace('_', ' ').title()}:")
            for subcomponent, sub_status in component_status.items():
                status_icon = "✅" if sub_status else "❌"
                print(f"   {subcomponent.replace('_', ' ').title()}: {status_icon}")
        else:
            status_icon = "✅" if component_status else "❌"
            print(f"{component.replace('_', ' ').title()}: {status_icon}")

    print(f"\nPackage Version: {__version__}")
    print(f"Total exports in __all__: {len(__all__)}")


# Print successful load message
print(f"🚀 PlanetScope-py v{__version__} - Phase 4 Enhanced")
print(f"✅ All Phase 4 features loaded successfully!")
print(f"📦 {len(__all__)} components available for import")
