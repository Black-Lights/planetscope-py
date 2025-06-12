"""
PlanetScope-py: Professional Python library for PlanetScope satellite imagery analysis.

A comprehensive tool for scene discovery, metadata analysis, and spatial-temporal 
density calculations using Planet's Data API.

Example usage:
    from planetscope_py import PlanetScopeClient
    
    # Initialize client (auto-detects API key)
    client = PlanetScopeClient()
    
    # Search for scenes
    scenes = client.search_scenes(
        geometry=my_roi,
        date_range=("2024-01-01", "2024-12-31"),
        cloud_cover_max=0.2
    )
    
    # Analyze density
    density = client.calculate_density(scenes, resolution=30)
"""

from ._version import __version__
from .auth import PlanetAuth
from .config import PlanetScopeConfig
from .exceptions import (
    PlanetScopeError,
    AuthenticationError, 
    ValidationError,
    RateLimitError,
    APIError
)

# Core classes that will be implemented in later phases
__all__ = [
    '__version__',
    'PlanetAuth',
    'PlanetScopeConfig', 
    'PlanetScopeError',
    'AuthenticationError',
    'ValidationError', 
    'RateLimitError',
    'APIError'
]

# Package metadata
__author__ = "Ammar & Umayr"
__email__ = "mohammadammarmughees@gmail.com"
__description__ = "Professional Python library for PlanetScope satellite imagery analysis"
__url__ = "https://github.com/Black-Lights/planetscope-py"