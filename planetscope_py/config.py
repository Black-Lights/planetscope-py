"""Configuration management for planetscope-py.

This module handles all configuration settings, default values, and environment setup
following Planet API conventions and best practices.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

from .exceptions import ConfigurationError


class PlanetScopeConfig:
    """Configuration manager for planetscope-py.

    Handles default settings, user configuration, and environment variables
    following Planet's established patterns.

    Attributes:
        base_url: Planet Data API base URL
        tile_url: Planet Tile Service API base URL
        item_types: Default item types for searches
        asset_types: Default asset types for downloads
        rate_limits: API rate limits per endpoint type
        timeouts: Request timeout settings
        max_retries: Maximum retry attempts for failed requests
    """

    # Planet API Configuration
    BASE_URL = "https://api.planet.com/data/v1"
    TILE_URL = "https://tiles.planet.com/data/v1"

    # Default search parameters
    DEFAULT_ITEM_TYPES = ["PSScene"]
    DEFAULT_ASSET_TYPES = ["ortho_analytic_4b", "ortho_analytic_4b_xml"]

    # Rate limits (requests per second) based on Planet API documentation
    RATE_LIMITS = {"search": 10, "activate": 5, "download": 15, "general": 10}

    # Request timeout settings (seconds)
    TIMEOUTS = {
        "connect": 10.0,
        "read": 30.0,
        "activation_poll": 300.0,  # 5 minutes for asset activation
        "download": 3600.0,  # 1 hour for large downloads
    }

    # Retry configuration
    MAX_RETRIES = 3
    BACKOFF_FACTOR = 1.0  # Exponential backoff multiplier

    # Geometry validation limits
    MAX_ROI_AREA_KM2 = 10000  # Maximum ROI area in square kilometers
    MAX_GEOMETRY_VERTICES = 1000  # Maximum vertices in polygon

    # Default output settings
    DEFAULT_OUTPUT_FORMAT = "GeoTIFF"
    DEFAULT_CRS = "EPSG:4326"

    def __init__(self, config_file: Optional[Union[str, Path]] = None):
        """Initialize configuration.

        Args:
            config_file: Path to custom configuration file (optional)
        """
        self._config_data = {}
        self._setup_logging()

        # Load configuration from multiple sources in order of priority:
        # 1. Default values (already set as class attributes)
        # 2. System config file (~/.planet.json)
        # 3. Custom config file (if provided)
        # 4. Environment variables

        self._load_system_config()
        if config_file:
            self._load_custom_config(config_file)
        self._load_env_config()

    def _setup_logging(self) -> None:
        """Setup default logging configuration."""
        log_level = os.environ.get("PLANETSCOPE_LOG_LEVEL", "INFO").upper()

        logging.basicConfig(
            level=getattr(logging, log_level, logging.INFO),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Suppress noisy third-party loggers
        logging.getLogger("requests").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)

    def _load_system_config(self) -> None:
        """Load configuration from ~/.planet.json if it exists."""
        try:
            config_path = Path.home() / ".planet.json"
        except (RuntimeError, OSError):
            # Fallback for cases where home directory detection fails
            home_dir = os.environ.get(
                "USERPROFILE", os.environ.get("HOME", os.getcwd())
            )
            config_path = Path(home_dir) / ".planet.json"

        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    system_config = json.load(f)
                    self._config_data.update(system_config)
            except (json.JSONDecodeError, IOError) as e:
                raise ConfigurationError(
                    f"Failed to load system config file: {config_path}",
                    {"error": str(e), "file": str(config_path)},
                )

    def _load_custom_config(self, config_file: Union[str, Path]) -> None:
        """Load configuration from custom file.

        Args:
            config_file: Path to configuration file
        """
        config_path = Path(config_file)
        if not config_path.exists():
            raise ConfigurationError(
                f"Configuration file not found: {config_path}",
                {"file": str(config_path)},
            )

        try:
            with open(config_path, "r") as f:
                custom_config = json.load(f)
                self._config_data.update(custom_config)
        except (json.JSONDecodeError, IOError) as e:
            raise ConfigurationError(
                f"Failed to load config file: {config_path}",
                {"error": str(e), "file": str(config_path)},
            )

    def _load_env_config(self) -> None:
        """Load configuration from environment variables."""
        env_mapping = {
            "PLANETSCOPE_BASE_URL": "base_url",
            "PLANETSCOPE_TILE_URL": "tile_url",
            "PLANETSCOPE_MAX_RETRIES": "max_retries",
            "PLANETSCOPE_MAX_ROI_AREA": "max_roi_area_km2",
            "PLANETSCOPE_DEFAULT_CRS": "default_crs",
        }

        for env_var, config_key in env_mapping.items():
            if env_var in os.environ:
                value = os.environ[env_var]
                # Convert to appropriate type
                if config_key in ["max_retries", "max_roi_area_km2"]:
                    try:
                        value = int(value)
                    except ValueError:
                        continue
                self._config_data[config_key] = value

    def get(self, key: str, default=None):
        """Get configuration value.

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        # Check custom config first, then class attributes
        if key in self._config_data:
            return self._config_data[key]

        # Convert key to class attribute format
        attr_name = key.upper().replace("-", "_")
        return getattr(self, attr_name, default)

    def set(self, key: str, value) -> None:
        """Set configuration value.

        Args:
            key: Configuration key
            value: Configuration value
        """
        self._config_data[key] = value

    def to_dict(self) -> Dict:
        """Export configuration as dictionary.

        Returns:
            Dictionary containing all configuration values
        """
        config = {
            "base_url": self.get("base_url", self.BASE_URL),
            "tile_url": self.get("tile_url", self.TILE_URL),
            "item_types": self.get("item_types", self.DEFAULT_ITEM_TYPES),
            "asset_types": self.get("asset_types", self.DEFAULT_ASSET_TYPES),
            "rate_limits": self.get("rate_limits", self.RATE_LIMITS),
            "timeouts": self.get("timeouts", self.TIMEOUTS),
            "max_retries": self.get("max_retries", self.MAX_RETRIES),
            "max_roi_area_km2": self.get("max_roi_area_km2", self.MAX_ROI_AREA_KM2),
            "default_crs": self.get("default_crs", self.DEFAULT_CRS),
        }
        config.update(self._config_data)
        return config

    @property
    def base_url(self) -> str:
        """Get Planet Data API base URL."""
        return self.get("base_url", self.BASE_URL)

    @property
    def tile_url(self) -> str:
        """Get Planet Tile Service API base URL."""
        return self.get("tile_url", self.TILE_URL)

    @property
    def item_types(self) -> List[str]:
        """Get default item types."""
        return self.get("item_types", self.DEFAULT_ITEM_TYPES)

    @property
    def asset_types(self) -> List[str]:
        """Get default asset types."""
        return self.get("asset_types", self.DEFAULT_ASSET_TYPES)

    @property
    def rate_limits(self) -> Dict[str, int]:
        """Get API rate limits."""
        return self.get("rate_limits", self.RATE_LIMITS)

    @property
    def timeouts(self) -> Dict[str, float]:
        """Get request timeouts."""
        return self.get("timeouts", self.TIMEOUTS)


# Global configuration instance
default_config = PlanetScopeConfig()