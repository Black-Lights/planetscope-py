#!/usr/bin/env python3
"""
Version information for planetscope-py.

This file contains the version number and related metadata for the
planetscope-py library, following semantic versioning principles.
"""

# Version components
__version_info__ = (4, 0, 0, "alpha", 1)

# Main version string
__version__ = "4.0.0a1"

# Version status
__version_status__ = "alpha"

# Build information
__build_date__ = "2025-06-20"
__build_number__ = "001"

# Phase information
__phase__ = "Phase 4: Enhanced Temporal Analysis & Asset Management"
__phase_number__ = 4

# Feature set information
__features__ = {
    "core_infrastructure": True,
    "planet_api_integration": True,
    "spatial_analysis": True,
    "adaptive_grid": True,
    "performance_optimization": True,
    "basic_visualization": True,
    "temporal_analysis": True,  # NEW in Phase 4
    "asset_management": True,  # NEW in Phase 4
    "geopackage_export": True,  # NEW in Phase 4
    "interactive_controls": True,  # NEW in Phase 4
}

# Compatibility information
__python_requires__ = ">=3.10"
__supported_platforms__ = ["Windows", "macOS", "Linux"]

# API version for backward compatibility
__api_version__ = "2.1"  # Enhanced API with Phase 4 features

# Development status for PyPI classifiers
__development_status__ = (
    "4 - Beta"  # Will be updated to "5 - Production/Stable" after testing
)

# Package metadata
__package_name__ = "planetscope-py"
__package_description__ = (
    "Professional Python library for PlanetScope satellite imagery analysis "
    "with enhanced temporal analysis, asset management, and data export capabilities"
)

# Version history
__version_history__ = {
    "1.0.0": "Foundation and Core Infrastructure",
    "2.0.0": "Planet API Integration Complete",
    "3.0.0": "Spatial Analysis Engine Complete",
    "4.0.0": "Enhanced Temporal Analysis & Asset Management",
}

# Release notes for current version
__release_notes__ = """
PlanetScope-py v4.0.0 Alpha 1 - Enhanced Temporal Analysis & Asset Management

NEW FEATURES:
- Advanced temporal analysis with 3D spatiotemporal data cubes
- Intelligent asset management with quota monitoring
- Professional GeoPackage export with imagery support
- Interactive user controls and progress tracking

ENHANCED FEATURES:
- Multi-algorithm spatial density calculations (3 methods)
- Comprehensive metadata processing and quality assessment
- Cross-platform grid compatibility
- Professional visualization and export capabilities

IMPROVEMENTS:
- Async asset download management with error recovery
- Real-time quota checking across multiple Planet APIs
- ROI clipping support for downloads and exports
- Comprehensive test coverage (300+ tests)

TECHNICAL DETAILS:
- Python 3.10+ required
- Support for xarray-based data cubes
- GeoPackage with raster and vector layers
- Async/await pattern for downloads
- Professional error handling and logging

DEPENDENCIES:
- Core: requests, shapely, pandas, numpy
- Spatial: geopandas, rasterio, matplotlib
- Temporal: xarray, scipy
- Asset Management: aiohttp, asyncio
- Export: fiona, sqlite3
- Interactive: ipywidgets (optional)
"""

# Deprecation warnings for future versions
__deprecation_warnings__ = []

# Feature flags for development
__feature_flags__ = {
    "enable_caching": True,
    "enable_async_downloads": True,
    "enable_progress_tracking": True,
    "enable_quota_monitoring": True,
    "enable_roi_clipping": True,
    "enable_interactive_widgets": True,
}


def get_version():
    """Get the current version string."""
    return __version__


def get_version_info():
    """Get detailed version information."""
    return {
        "version": __version__,
        "version_info": __version_info__,
        "status": __version_status__,
        "phase": __phase__,
        "phase_number": __phase_number__,
        "build_date": __build_date__,
        "build_number": __build_number__,
        "api_version": __api_version__,
        "python_requires": __python_requires__,
        "supported_platforms": __supported_platforms__,
        "features": __features__,
    }


def show_version_info():
    """Display comprehensive version information."""
    print(f"PlanetScope-py {__version__}")
    print(f"Phase: {__phase__}")
    print(f"Build: {__build_date__} #{__build_number__}")
    print(f"Python: {__python_requires__}")
    print(f"Status: {__development_status__}")
    print()

    print("Available Features:")
    for feature, available in __features__.items():
        status = "+" if available else "-"
        feature_name = feature.replace("_", " ").title()
        print(f"  {status} {feature_name}")

    print()
    print("Supported Platforms:")
    for platform in __supported_platforms__:
        print(f"  - {platform}")


def check_version_compatibility(required_version: str) -> bool:
    """
    Check if current version meets requirement.

    Args:
        required_version: Minimum required version (e.g., "3.0.0")

    Returns:
        True if current version meets requirement
    """
    try:
        from packaging import version

        return version.parse(__version__) >= version.parse(required_version)
    except ImportError:
        # Fallback comparison if packaging not available
        current = tuple(map(int, __version__.split(".")[:3]))
        required = tuple(map(int, required_version.split(".")[:3]))
        return current >= required


def get_feature_availability():
    """Get current feature availability status."""
    try:
        # Check actual imports to verify availability
        import planetscope_py

        actual_features = {}

        # Check core features
        try:
            from planetscope_py import PlanetScopeQuery

            actual_features["planet_api_integration"] = True
        except ImportError:
            actual_features["planet_api_integration"] = False

        # Check spatial analysis
        try:
            from planetscope_py import SpatialDensityEngine

            actual_features["spatial_analysis"] = True
        except ImportError:
            actual_features["spatial_analysis"] = False

        # Check Phase 4 features
        try:
            from planetscope_py import TemporalAnalyzer

            actual_features["temporal_analysis"] = True
        except ImportError:
            actual_features["temporal_analysis"] = False

        try:
            from planetscope_py import AssetManager

            actual_features["asset_management"] = True
        except ImportError:
            actual_features["asset_management"] = False

        try:
            from planetscope_py import GeoPackageManager

            actual_features["geopackage_export"] = True
        except ImportError:
            actual_features["geopackage_export"] = False

        try:
            from planetscope_py import InteractiveManager

            actual_features["interactive_controls"] = True
        except ImportError:
            actual_features["interactive_controls"] = False

        return actual_features

    except ImportError:
        return {}


# Version validation
def validate_version_format():
    """Validate that version follows semantic versioning."""
    import re

    # Semantic versioning pattern for alpha/beta/rc versions
    semver_pattern = r"^(\d+)\.(\d+)\.(\d+)(a|b|rc)(\d+)$"

    if not re.match(semver_pattern, __version__):
        raise ValueError(f"Version {__version__} does not follow semantic versioning")

    return True


# Automatic validation on import
try:
    validate_version_format()
except ValueError as e:
    import warnings

    warnings.warn(f"Version format warning: {e}", UserWarning)

# Export public interface
__all__ = [
    "__version__",
    "__version_info__",
    "__version_status__",
    "__phase__",
    "__phase_number__",
    "__features__",
    "__api_version__",
    "__python_requires__",
    "__release_notes__",
    "get_version",
    "get_version_info",
    "show_version_info",
    "check_version_compatibility",
    "get_feature_availability",
]
