#!/usr/bin/env python3
"""
Version information for planetscope-py.

This file contains the version number and related metadata for the
planetscope-py library, following semantic versioning principles.

Author: Ammar & Umayr
Version: 4.1.0 (Enhanced + Metadata Fixes + JSON Serialization)
"""

# Version components
__version_info__ = (4, 0, 1)

# Main version string
__version__ = "4.0.1"

# Version status
__version_status__ = "stable"

# Build information
__build_date__ = "2025-06-25"
__build_number__ = "002"

# Phase information
__phase__ = "Phase 4: Complete Temporal Analysis & Advanced Data Management"
__phase_number__ = 4

# Feature set information
__features__ = {
    "core_infrastructure": True,
    "planet_api_integration": True,
    "spatial_analysis": True,
    "temporal_analysis": True,
    "asset_management": True,
    "geopackage_export": True,
    "adaptive_grid": True,
    "performance_optimization": True,
    "visualization": True,
    "async_operations": True,
    "import_fixes": True,  # NEW in v4.0.1
}

# Compatibility information
__python_requires__ = ">=3.10"
__supported_platforms__ = ["Windows", "macOS", "Linux"]

# API version for backward compatibility
__api_version__ = "2.1"

# Development status for PyPI classifiers
__development_status__ = "5 - Production/Stable"

# Package metadata
__package_name__ = "planetscope-py"
__package_description__ = (
    "Professional Python library for PlanetScope satellite imagery analysis "
    "with complete temporal analysis, spatial density analysis, and advanced data export capabilities"
)

# Version history
__version_history__ = {
    "1.0.0": "Foundation and Core Infrastructure",
    "2.0.0": "Planet API Integration Complete",
    "3.0.0": "Spatial Analysis Engine Complete",
    "4.0.0": "Complete Temporal Analysis & Advanced Data Management",
    "4.0.1": "Bug Fix Release - Fixed Module Availability Issues",
}

# Release notes for current version
__release_notes__ = """
PlanetScope-py v4.0.1 - Bug Fix Release

CRITICAL FIXES:
- Fixed workflow module availability detection
- Fixed silent import failures in __init__.py  
- Fixed quick_planet_analysis function not working
- Fixed visualization module import issues
- Added proper error messages for missing dependencies

IMPROVEMENTS:
- Enhanced import debugging with success confirmations
- Better dependency installation instructions in error messages
- Improved module status reporting accuracy
- More robust error handling throughout the library

TECHNICAL DETAILS:
- Replaced silent 'except ImportError: pass' with proper warnings
- Added debug print statements for successful module loads
- Enhanced _WORKFLOWS_AVAILABLE and _VISUALIZATION_AVAILABLE flag setting
- Improved check_module_status() function accuracy

USER IMPACT:
This release ensures that quick_planet_analysis and all workflow functions 
work correctly. Users should no longer encounter ImportError issues when 
all dependencies are properly installed.

INSTALLATION:
pip install --upgrade planetscope-py

VERIFICATION:
After upgrading, users can verify the fix with:
import planetscope_py
planetscope_py.check_module_status()
from planetscope_py import quick_planet_analysis  # Should work now
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
    "enable_grid_optimization": True,
    "enable_coordinate_fixes": True,
    "enable_import_debugging": True,  # NEW in v4.0.1
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
        status = "✓" if available else "✗"
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

        # Check temporal analysis
        try:
            from planetscope_py import TemporalAnalyzer

            actual_features["temporal_analysis"] = True
        except ImportError:
            actual_features["temporal_analysis"] = False

        # Check asset management
        try:
            from planetscope_py import AssetManager

            actual_features["asset_management"] = True
        except ImportError:
            actual_features["asset_management"] = False

        # Check GeoPackage export
        try:
            from planetscope_py import GeoPackageManager

            actual_features["geopackage_export"] = True
        except ImportError:
            actual_features["geopackage_export"] = False

        # Check workflow functions (v4.0.1 fix verification)
        try:
            from planetscope_py import quick_planet_analysis

            actual_features["workflow_functions"] = True
        except ImportError:
            actual_features["workflow_functions"] = False

        # Check visualization (v4.0.1 fix verification)
        try:
            from planetscope_py import plot_density_map_only

            actual_features["visualization_functions"] = True
        except ImportError:
            actual_features["visualization_functions"] = False

        return actual_features

    except ImportError:
        return {}


# Version validation
def validate_version_format():
    """Validate that version follows semantic versioning."""
    import re

    # Semantic versioning pattern for stable versions
    semver_pattern = r"^(\d+)\.(\d+)\.(\d+)$"

    if not re.match(semver_pattern, __version__):
        raise ValueError(f"Version {__version__} does not follow semantic versioning")

    return True


# Test import fixes (v4.0.1 specific)
def test_import_fixes():
    """Test that v4.0.1 import fixes are working."""
    try:
        import planetscope_py
        
        # Test workflow availability
        workflow_available = planetscope_py._WORKFLOWS_AVAILABLE
        
        # Test visualization availability  
        viz_available = planetscope_py._VISUALIZATION_AVAILABLE
        
        # Test actual function imports
        try:
            from planetscope_py import quick_planet_analysis
            workflow_import = True
        except ImportError:
            workflow_import = False
            
        try:
            from planetscope_py import plot_density_map_only
            viz_import = True
        except ImportError:
            viz_import = False
        
        return {
            "workflow_flag": workflow_available,
            "visualization_flag": viz_available,
            "workflow_import": workflow_import,
            "visualization_import": viz_import,
            "fix_successful": workflow_available and viz_available and workflow_import and viz_import
        }
        
    except Exception as e:
        return {"error": str(e), "fix_successful": False}


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
    "test_import_fixes",  # NEW in v4.0.1
]