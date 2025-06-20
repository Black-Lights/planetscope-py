#!/usr/bin/env python3
"""
PlanetScope-py: Professional Python library for PlanetScope satellite imagery analysis.

A comprehensive tool for scene discovery, metadata analysis, and spatial-temporal 
density calculations using Planet's Data API.

Example usage:
    from planetscope_py import PlanetScopeQuery, MetadataProcessor, SpatialDensityEngine
    
    # Initialize query system
    query = PlanetScopeQuery()
    
    # Search for scenes
    results = query.search_scenes(
        geometry=my_roi,
        start_date="2024-01-01", 
        end_date="2024-03-31",
        cloud_cover_max=0.2
    )
    
    # Process metadata
    processor = MetadataProcessor()
    assessment = processor.assess_coverage_quality(
        scenes=results["features"],
        target_geometry=my_roi
    )
    
    # Calculate spatial density
    from planetscope_py.density_engine import DensityConfig, DensityMethod
    
    config = DensityConfig(
        resolution=30.0,  # 30m resolution
        method=DensityMethod.AUTO  # Auto-select best method
    )
    
    density_engine = SpatialDensityEngine(config)
    density_result = density_engine.calculate_density(
        scene_footprints=results["features"],
        roi_geometry=my_roi
    )
"""

from ._version import __version__

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
    AssetError
)
from .utils import (
    validate_geometry,
    calculate_area_km2,
    transform_geometry,
    create_bounding_box,
    buffer_geometry
)

# Planet API Integration
from .query import PlanetScopeQuery
from .metadata import MetadataProcessor
from .rate_limiter import RateLimiter, RetryableSession, CircuitBreaker

# Spatial Analysis Engine
try:
    from .density_engine import (
        SpatialDensityEngine,
        DensityConfig, 
        DensityMethod,
        DensityResult
    )
    _SPATIAL_ANALYSIS_AVAILABLE = True
except ImportError:
    _SPATIAL_ANALYSIS_AVAILABLE = False

# Advanced Spatial Components
try:
    from .adaptive_grid import AdaptiveGridEngine, AdaptiveGridConfig
    _ADAPTIVE_GRID_AVAILABLE = True
except ImportError:
    _ADAPTIVE_GRID_AVAILABLE = False

try:
    from .optimizer import PerformanceOptimizer, DatasetCharacteristics, PerformanceProfile
    _OPTIMIZER_AVAILABLE = True
except ImportError:
    _OPTIMIZER_AVAILABLE = False

try:
    from .visualization import DensityVisualizer
    _VISUALIZATION_AVAILABLE = True
except ImportError:
    _VISUALIZATION_AVAILABLE = False

# Main exports for easy access
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
    __all__.extend([
        "SpatialDensityEngine",
        "DensityConfig",
        "DensityMethod", 
        "DensityResult"
    ])

if _ADAPTIVE_GRID_AVAILABLE:
    __all__.extend([
        "AdaptiveGridEngine",
        "AdaptiveGridConfig"
    ])

if _OPTIMIZER_AVAILABLE:
    __all__.extend([
        "PerformanceOptimizer",
        "DatasetCharacteristics",
        "PerformanceProfile"
    ])

if _VISUALIZATION_AVAILABLE:
    __all__.extend([
        "DensityVisualizer"
    ])

# Package metadata
__author__ = "Ammar & Umayr"
__email__ = "mohammadammarmughees@gmail.com"
__description__ = (
    "Professional Python library for PlanetScope satellite imagery analysis"
)
__url__ = "https://github.com/Black-Lights/planetscope-py"
__license__ = "MIT"

# Component availability status
def get_component_status():
    """Get the current availability status of all components."""
    return {
        "Core Infrastructure": "Available",
        "Planet API Integration": "Available", 
        "Spatial Analysis Engine": "Available" if _SPATIAL_ANALYSIS_AVAILABLE else "Not Available",
        "Advanced Components": {
            "Adaptive Grid": "Available" if _ADAPTIVE_GRID_AVAILABLE else "Not Available",
            "Performance Optimizer": "Available" if _OPTIMIZER_AVAILABLE else "Not Available",
            "Visualization": "Available" if _VISUALIZATION_AVAILABLE else "Not Available"
        }
    }

# Auto-integration of advanced methods
def _integrate_advanced_methods():
    """Automatically integrate advanced methods if available."""
    if _SPATIAL_ANALYSIS_AVAILABLE and _ADAPTIVE_GRID_AVAILABLE:
        try:
            from .adaptive_grid import integrate_adaptive_grid
            
            # Get the SpatialDensityEngine class
            density_engine_class = globals().get('SpatialDensityEngine')
            if density_engine_class:
                # Apply integration to the class itself
                integrate_adaptive_grid(density_engine_class)
                
                import logging
                logger = logging.getLogger(__name__)
                logger.info("Adaptive grid integration applied automatically")
                
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to auto-integrate adaptive grid: {e}", ImportWarning)

# Dependency checking
def check_dependencies():
    """Check if all required dependencies are available."""
    missing_deps = []
    
    # Core dependencies
    try:
        import requests
    except ImportError:
        missing_deps.append("requests")
        
    try:
        import shapely
    except ImportError:
        missing_deps.append("shapely")
        
    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy")
        
    try:
        import pandas  
    except ImportError:
        missing_deps.append("pandas")
    
    # Spatial analysis dependencies
    if _SPATIAL_ANALYSIS_AVAILABLE:
        try:
            import rasterio
        except ImportError:
            missing_deps.append("rasterio (required for spatial analysis)")
            
        try:
            import geopandas
        except ImportError:
            missing_deps.append("geopandas (required for spatial analysis)")
    
    if _VISUALIZATION_AVAILABLE:
        try:
            import matplotlib
        except ImportError:
            missing_deps.append("matplotlib (required for visualization)")
    
    if missing_deps:
        raise ImportError(
            f"Missing required dependencies: {', '.join(missing_deps)}. "
            "Please install them using: pip install planetscope-py[all]"
        )

def check_spatial_dependencies():
    """Check spatial analysis specific dependencies."""
    spatial_deps = []
    
    try:
        import rasterio
    except ImportError:
        spatial_deps.append("rasterio")
        
    try:
        import geopandas
    except ImportError:
        spatial_deps.append("geopandas")
        
    try:
        import matplotlib
    except ImportError:
        spatial_deps.append("matplotlib")
    
    return spatial_deps

# Perform dependency check on import
try:
    check_dependencies()
except ImportError as e:
    import warnings
    warnings.warn(
        f"Some dependencies are missing: {e}. "
        "Some functionality may not be available.",
        ImportWarning
    )

# Auto-integrate advanced methods
try:
    _integrate_advanced_methods()
except Exception:
    pass  # Fail silently if integration is not possible

# User information functions
def show_library_status():
    """Display current library component availability."""
    print("PlanetScope-py Library Status")
    print("=" * 40)
    
    status = get_component_status()
    for component, status_text in status.items():
        if isinstance(status_text, dict):
            print(f"{component}:")
            for subcomponent, comp_status in status_text.items():
                print(f"   {subcomponent}: {comp_status}")
        else:
            print(f"{component}: {status_text}")
    
    print(f"\nPackage Version: {__version__}")
    
    # Check dependencies
    missing_deps = check_spatial_dependencies()
    if missing_deps:
        print(f"\nOptional dependencies missing for full spatial functionality:")
        for dep in missing_deps:
            print(f"   - {dep}")
        print(f"\nInstall with: pip install {' '.join(missing_deps)}")
    else:
        print(f"\nAll dependencies satisfied")
    
    print(f"\nQuick Start:")
    print(f"   from planetscope_py import PlanetScopeQuery, SpatialDensityEngine")
    print(f"   query = PlanetScopeQuery()")
    if _SPATIAL_ANALYSIS_AVAILABLE:
        print(f"   engine = SpatialDensityEngine()")
        print(f"   # Ready for spatial analysis")

# Main library interface
class PlanetScopeLibrary:
    """Main library interface with configuration options."""
    
    def __init__(self):
        self.core_available = True
        self.api_integration_available = True  
        self.spatial_analysis_available = _SPATIAL_ANALYSIS_AVAILABLE
        self.adaptive_grid_available = _ADAPTIVE_GRID_AVAILABLE
        self.optimizer_available = _OPTIMIZER_AVAILABLE
        self.visualization_available = _VISUALIZATION_AVAILABLE
    
    def __repr__(self):
        spatial_status = "Available" if self.spatial_analysis_available else "Not Available"
        return f"PlanetScopeLibrary v{__version__} (Spatial Analysis: {spatial_status})"

# Create default instance
library = PlanetScopeLibrary()

# Utility functions
def get_version():
    """Get library version."""
    return __version__

def get_build_info():
    """Get build information."""
    return {
        "version": __version__,
        "author": __author__,
        "component_status": get_component_status(),
        "adaptive_grid_integrated": _ADAPTIVE_GRID_AVAILABLE and _SPATIAL_ANALYSIS_AVAILABLE
    }

def has_spatial_analysis():
    """Check if spatial analysis components are available."""
    return _SPATIAL_ANALYSIS_AVAILABLE

def has_adaptive_grid():
    """Check if adaptive grid is available."""
    return _ADAPTIVE_GRID_AVAILABLE

def has_visualization():
    """Check if visualization components are available."""
    return _VISUALIZATION_AVAILABLE