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
        date_range=("2025-02-01", "2025-01-30"),
        cloud_cover_max=0.2
    )

    # Analyze density
    density = client.calculate_density(scenes, resolution=30)
"""


from ._version import __version__
from .auth import PlanetAuth
from .config import PlanetScopeConfig
from .exceptions import (APIError, AuthenticationError, PlanetScopeError,
                         RateLimitError, ValidationError)

__all__ = [
    "__version__",
    "PlanetAuth",
    "PlanetScopeConfig",
    "PlanetScopeError",
    "AuthenticationError",
    "ValidationError",
    "RateLimitError",
    "APIError",
]

__author__ = "Ammar & Umayr"
__email__ = "mohammadammarmughees@gmail.com"
__description__ = (
    "Professional Python library for PlanetScope satellite imagery analysis"
)
__url__ = "https://github.com/Black-Lights/planetscope-py"
