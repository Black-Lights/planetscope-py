"""Configuration management for planetscope-py.

This module handles all configuration settings, default values, and environment setup
following Planet API conventions and best practices.

ENHANCED VERSION: Includes configuration for all modules:
- Spatial density analysis
- Temporal analysis  
- GeoPackage export
- Visualization
- Performance optimization

Author: Ammar & Umayr
Version: 4.1.0 (Enhanced + Metadata Fixes + JSON Serialization)
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

    ENHANCED: Now includes settings for all modern modules including
    spatial density, temporal analysis, GeoPackage export, and visualization.
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

    # ====================================================================
    # NEW: SPATIAL DENSITY ANALYSIS CONFIGURATION
    # ====================================================================
    
    # Spatial density default settings
    SPATIAL_DENSITY_DEFAULTS = {
        "spatial_resolution": 30.0,  # meters
        "chunk_size_km": 100.0,  # km
        "max_memory_gb": 8.0,  # GB
        "parallel_workers": 4,
        "no_data_value": -9999.0,
        "coordinate_system_fixes": True,
        "optimization_method": "auto",  # "fast", "accurate", "auto"
        "force_single_chunk": False,
        "validate_geometries": True,
        "max_scenes_footprint": 150,  # For visualization
    }

    # ====================================================================
    # NEW: TEMPORAL ANALYSIS CONFIGURATION  
    # ====================================================================
    
    # Temporal analysis default settings
    TEMPORAL_ANALYSIS_DEFAULTS = {
        "spatial_resolution": 30.0,  # meters (same grid as spatial density)
        "temporal_resolution": "daily",  # "daily", "weekly", "monthly"
        "chunk_size_km": 200.0,  # km (larger for temporal)
        "max_memory_gb": 16.0,  # GB (more for temporal)
        "parallel_workers": 4,
        "no_data_value": -9999.0,
        "coordinate_system_fixes": True,
        "force_single_chunk": False,
        "validate_geometries": True,
        "min_scenes_per_cell": 2,  # Minimum scenes for temporal analysis
        "optimization_method": "auto",  # "fast", "accurate", "auto"
        "default_metrics": ["coverage_days", "mean_interval", "temporal_density"],
    }

    # ====================================================================
    # NEW: GEOPACKAGE EXPORT CONFIGURATION
    # ====================================================================
    
    # GeoPackage export settings
    GEOPACKAGE_DEFAULTS = {
        "attribute_schema": "standard",  # "minimal", "standard", "comprehensive"
        "include_previews": False,
        "include_styling": True,
        "compression": "lzw",
        "coordinate_precision": 6,  # decimal places
        "max_features_per_layer": 50000,
        "enable_spatial_index": True,
        "metadata_level": "standard",  # "minimal", "standard", "comprehensive"
        "export_format": "GPKG",  # "GPKG", "SHP", "GEOJSON"
    }

    # ====================================================================
    # NEW: VISUALIZATION CONFIGURATION
    # ====================================================================
    
    # Visualization settings
    VISUALIZATION_DEFAULTS = {
        "default_colormap": "turbo",
        "figure_size": (12, 8),
        "dpi": 300,
        "max_scenes_display": 300,  # Increased limit
        "coordinate_system_fixes": True,
        "enable_roi_clipping": True,
        "histogram_bins": "auto",  # or integer
        "save_format": "png",
        "show_statistics": True,
        "enable_interactive": False,
    }

    # ====================================================================
    # NEW: PERFORMANCE OPTIMIZATION CONFIGURATION
    # ====================================================================
    
    # Performance optimization settings
    PERFORMANCE_DEFAULTS = {
        "enable_caching": True,
        "cache_size_mb": 512,
        "parallel_processing": True,
        "memory_monitoring": True,
        "progress_reporting": True,
        "optimization_level": "balanced",  # "fast", "balanced", "accurate"
        "enable_gpu_acceleration": False,  # Future feature
        "batch_size": 1000,  # For large collections
    }

    # ====================================================================
    # NEW: QUALITY CONTROL CONFIGURATION
    # ====================================================================
    
    # Quality control thresholds
    QUALITY_DEFAULTS = {
        "max_cloud_cover": 0.3,  # 30%
        "min_sun_elevation": 15.0,  # degrees
        "min_usable_data": 0.8,  # 80%
        "enable_quality_filtering": True,
        "quality_categories": ["excellent", "good", "fair", "poor"],
        "auto_reject_poor": False,
    }

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
            # Original settings
            "PLANETSCOPE_BASE_URL": "base_url",
            "PLANETSCOPE_TILE_URL": "tile_url",
            "PLANETSCOPE_MAX_RETRIES": "max_retries",
            "PLANETSCOPE_MAX_ROI_AREA": "max_roi_area_km2",
            "PLANETSCOPE_DEFAULT_CRS": "default_crs",
            
            # NEW: Spatial density settings
            "PLANETSCOPE_SPATIAL_RESOLUTION": "spatial_resolution",
            "PLANETSCOPE_OPTIMIZATION_METHOD": "optimization_method",
            "PLANETSCOPE_MAX_MEMORY_GB": "max_memory_gb",
            "PLANETSCOPE_PARALLEL_WORKERS": "parallel_workers",
            
            # NEW: Temporal analysis settings
            "PLANETSCOPE_TEMPORAL_RESOLUTION": "temporal_resolution",
            "PLANETSCOPE_MIN_SCENES_PER_CELL": "min_scenes_per_cell",
            
            # NEW: Quality settings
            "PLANETSCOPE_MAX_CLOUD_COVER": "max_cloud_cover",
            "PLANETSCOPE_MIN_SUN_ELEVATION": "min_sun_elevation",
        }

        for env_var, config_key in env_mapping.items():
            if env_var in os.environ:
                value = os.environ[env_var]
                # Convert to appropriate type
                if config_key in ["max_retries", "max_roi_area_km2", "parallel_workers", "min_scenes_per_cell"]:
                    try:
                        value = int(value)
                    except ValueError:
                        continue
                elif config_key in ["spatial_resolution", "max_memory_gb", "max_cloud_cover", "min_sun_elevation"]:
                    try:
                        value = float(value)
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

        # Check in default dictionaries
        for defaults_dict in [
            self.SPATIAL_DENSITY_DEFAULTS,
            self.TEMPORAL_ANALYSIS_DEFAULTS,
            self.GEOPACKAGE_DEFAULTS,
            self.VISUALIZATION_DEFAULTS,
            self.PERFORMANCE_DEFAULTS,
            self.QUALITY_DEFAULTS,
        ]:
            if key in defaults_dict:
                return defaults_dict[key]

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
            # Original settings
            "base_url": self.get("base_url", self.BASE_URL),
            "tile_url": self.get("tile_url", self.TILE_URL),
            "item_types": self.get("item_types", self.DEFAULT_ITEM_TYPES),
            "asset_types": self.get("asset_types", self.DEFAULT_ASSET_TYPES),
            "rate_limits": self.get("rate_limits", self.RATE_LIMITS),
            "timeouts": self.get("timeouts", self.TIMEOUTS),
            "max_retries": self.get("max_retries", self.MAX_RETRIES),
            "max_roi_area_km2": self.get("max_roi_area_km2", self.MAX_ROI_AREA_KM2),
            "default_crs": self.get("default_crs", self.DEFAULT_CRS),
            
            # NEW: Module-specific settings
            "spatial_density": self.SPATIAL_DENSITY_DEFAULTS.copy(),
            "temporal_analysis": self.TEMPORAL_ANALYSIS_DEFAULTS.copy(),
            "geopackage": self.GEOPACKAGE_DEFAULTS.copy(),
            "visualization": self.VISUALIZATION_DEFAULTS.copy(),
            "performance": self.PERFORMANCE_DEFAULTS.copy(),
            "quality": self.QUALITY_DEFAULTS.copy(),
        }
        
        # Override with custom settings
        config.update(self._config_data)
        return config

    # ====================================================================
    # ENHANCED PROPERTY ACCESSORS
    # ====================================================================

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

    # NEW: Enhanced property accessors for modern modules
    
    @property
    def spatial_density_config(self) -> Dict:
        """Get spatial density analysis configuration."""
        return self.get("spatial_density", self.SPATIAL_DENSITY_DEFAULTS)

    @property
    def temporal_analysis_config(self) -> Dict:
        """Get temporal analysis configuration."""
        return self.get("temporal_analysis", self.TEMPORAL_ANALYSIS_DEFAULTS)

    @property
    def geopackage_config(self) -> Dict:
        """Get GeoPackage export configuration."""
        return self.get("geopackage", self.GEOPACKAGE_DEFAULTS)

    @property
    def visualization_config(self) -> Dict:
        """Get visualization configuration."""
        return self.get("visualization", self.VISUALIZATION_DEFAULTS)

    @property
    def performance_config(self) -> Dict:
        """Get performance optimization configuration."""
        return self.get("performance", self.PERFORMANCE_DEFAULTS)

    @property
    def quality_config(self) -> Dict:
        """Get quality control configuration."""
        return self.get("quality", self.QUALITY_DEFAULTS)


class PresetConfigs:
    """Preset configurations for common use cases."""
    
    @staticmethod
    def get_high_resolution():
        """Get configuration for high-resolution analysis."""
        config = PlanetScopeConfig()
        config.set("spatial_resolution", 10.0)
        config.set("optimization_method", "accurate")
        config.set("max_scenes_display", 500)
        return config
    
    @staticmethod
    def get_fast_analysis():
        """Get configuration for fast analysis."""
        config = PlanetScopeConfig()
        config.set("spatial_resolution", 100.0)
        config.set("optimization_method", "fast")
        config.set("max_memory_gb", 4.0)
        config.set("parallel_workers", 2)
        return config
    
    @staticmethod
    def get_temporal_analysis():
        """Get configuration optimized for temporal analysis."""
        config = PlanetScopeConfig()
        config.set("spatial_resolution", 50.0)
        config.set("temporal_resolution", "daily")
        config.set("min_scenes_per_cell", 3)
        config.set("optimization_method", "fast")
        config.set("max_memory_gb", 16.0)
        return config
    
    @staticmethod
    def get_production_quality():
        """Get configuration for production-quality analysis."""
        config = PlanetScopeConfig()
        config.set("spatial_resolution", 30.0)
        config.set("optimization_method", "accurate")
        config.set("coordinate_system_fixes", True)
        config.set("validate_geometries", True)
        config.set("enable_quality_filtering", True)
        config.set("max_cloud_cover", 0.2)
        return config


# Global configuration instance
default_config = PlanetScopeConfig()