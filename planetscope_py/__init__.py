#!/usr/bin/env python3
"""
PlanetScope-py: Professional Python library for PlanetScope satellite imagery analysis.

A comprehensive tool for scene discovery, metadata analysis, and spatial-temporal
density calculations using Planet's Data API.

Example usage:
    from planetscope_py import PlanetScopeQuery, MetadataProcessor
    
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
"""

from ._version import __version__

# Phase 1 - Core Infrastructure
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

# Phase 2 - Planet API Integration
from .query import PlanetScopeQuery
from .metadata import MetadataProcessor
from .rate_limiter import RateLimiter, RetryableSession, CircuitBreaker

# Main exports for easy access
__all__ = [
    # Version
    "__version__",
    
    # Core Infrastructure (Phase 1)
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
    
    # Planet API Integration (Phase 2)
    "PlanetScopeQuery",
    "MetadataProcessor",
    "RateLimiter",
    "RetryableSession",
    "CircuitBreaker",
]

# Package metadata
__author__ = "Ammar & Umayr"
__email__ = "mohammadammarmughees@gmail.com"
__description__ = (
    "Professional Python library for PlanetScope satellite imagery analysis"
)
__url__ = "https://github.com/Black-Lights/planetscope-py"
__license__ = "MIT"

# Compatibility check
def check_dependencies():
    """Check if all required dependencies are available."""
    missing_deps = []
    
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
    
    if missing_deps:
        raise ImportError(
            f"Missing required dependencies: {', '.join(missing_deps)}. "
            "Please install them using: pip install planetscope-py[all]"
        )

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