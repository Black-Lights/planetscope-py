"""Tests for planetscope_py.config module."""

import json
import os
from pathlib import Path
from unittest.mock import Mock, mock_open, patch

import pytest

from planetscope_py.config import PlanetScopeConfig
from planetscope_py.exceptions import ConfigurationError


class TestPlanetScopeConfig:
    """Test cases for PlanetScopeConfig class."""

    def test_init_with_defaults(self):
        """Test initialization with default values."""
        config = PlanetScopeConfig()

        assert config.base_url == "https://api.planet.com/data/v1"
        assert config.item_types == ["PSScene"]
        assert config.asset_types == ["ortho_analytic_4b", "ortho_analytic_4b_xml"]
        assert config.rate_limits["search"] == 10
        assert config.timeouts["connect"] == 10.0
        assert config.MAX_RETRIES == 3

    def test_init_with_custom_config_file(self, tmp_path):
        """Test initialization with custom configuration file."""
        # Create temporary config file
        config_file = tmp_path / "custom_config.json"
        config_data = {
            "max_retries": 5,
            "base_url": "https://custom.api.com/v1",
            "item_types": ["REOrthoTile"],
        }

        with open(config_file, "w") as f:
            json.dump(config_data, f)

        config = PlanetScopeConfig(config_file=config_file)

        assert config.get("max_retries") == 5
        assert config.get("base_url") == "https://custom.api.com/v1"
        assert config.get("item_types") == ["REOrthoTile"]

    def test_init_with_nonexistent_config_file(self):
        """Test initialization with non-existent config file raises error."""
        with pytest.raises(ConfigurationError) as exc_info:
            PlanetScopeConfig(config_file="/nonexistent/config.json")

        assert "Configuration file not found" in str(exc_info.value)

    def test_init_with_invalid_json_config(self, tmp_path):
        """Test initialization with invalid JSON config file."""
        config_file = tmp_path / "invalid_config.json"
        with open(config_file, "w") as f:
            f.write("{ invalid json }")

        with pytest.raises(ConfigurationError) as exc_info:
            PlanetScopeConfig(config_file=config_file)

        assert "Failed to load config file" in str(exc_info.value)

    def test_system_config_loading(self, monkeypatch):
        """Test loading configuration from ~/.planet.json."""
        system_config = {"api_key": "system_api_key", "max_retries": 7}

        mock_file_content = json.dumps(system_config)

        # Mock the system config file
        mock_path = Mock()
        mock_path.exists.return_value = True

        monkeypatch.setattr(
            "pathlib.Path.home", lambda: Mock(__truediv__=lambda self, other: mock_path)
        )

        with patch("builtins.open", mock_open(read_data=mock_file_content)):
            config = PlanetScopeConfig()

            assert config.get("api_key") == "system_api_key"
            assert config.get("max_retries") == 7

    def test_system_config_invalid_json(self, monkeypatch):
        """Test handling of invalid JSON in system config file."""
        mock_path = Mock()
        mock_path.exists.return_value = True

        monkeypatch.setattr(
            "pathlib.Path.home", lambda: Mock(__truediv__=lambda self, other: mock_path)
        )

        with patch("builtins.open", mock_open(read_data="{ invalid json }")):
            with pytest.raises(ConfigurationError) as exc_info:
                PlanetScopeConfig()

            assert "Failed to load system config file" in str(exc_info.value)

    def test_system_config_missing_file(self, monkeypatch):
        """Test behavior when system config file doesn't exist."""
        mock_path = Mock()
        mock_path.exists.return_value = False

        monkeypatch.setattr(
            "pathlib.Path.home", lambda: Mock(__truediv__=lambda self, other: mock_path)
        )

        # Should not raise exception, just use defaults
        config = PlanetScopeConfig()
        assert config.base_url == "https://api.planet.com/data/v1"

    def test_environment_variable_loading(self, monkeypatch):
        """Test loading configuration from environment variables."""
        env_vars = {
            "PLANETSCOPE_BASE_URL": "https://env.api.com/v1",
            "PLANETSCOPE_MAX_RETRIES": "8",
            "PLANETSCOPE_MAX_ROI_AREA": "5000",
            "PLANETSCOPE_DEFAULT_CRS": "EPSG:3857",
        }

        for key, value in env_vars.items():
            monkeypatch.setenv(key, value)

        config = PlanetScopeConfig()

        assert config.get("base_url") == "https://env.api.com/v1"
        assert config.get("max_retries") == 8
        assert config.get("max_roi_area_km2") == 5000
        assert config.get("default_crs") == "EPSG:3857"

    def test_environment_variable_invalid_types(self, monkeypatch):
        """Test that invalid environment variable types are ignored."""
        monkeypatch.setenv("PLANETSCOPE_MAX_RETRIES", "not_a_number")

        config = PlanetScopeConfig()

        # Should use default value since conversion failed
        assert config.get("max_retries") == config.MAX_RETRIES

    def test_get_method(self):
        """Test the get method for retrieving configuration values."""
        config = PlanetScopeConfig()

        # Test getting existing value
        assert config.get("base_url") == "https://api.planet.com/data/v1"

        # Test getting non-existent value with default
        assert config.get("nonexistent_key", "default_value") == "default_value"

        # Test getting non-existent value without default
        assert config.get("nonexistent_key") is None

    def test_set_method(self):
        """Test the set method for updating configuration values."""
        config = PlanetScopeConfig()

        # Set new value
        config.set("custom_setting", "custom_value")
        assert config.get("custom_setting") == "custom_value"

        # Override existing value
        config.set("max_retries", 10)
        assert config.get("max_retries") == 10

    def test_to_dict_method(self):
        """Test the to_dict method for exporting configuration."""
        config = PlanetScopeConfig()
        config.set("custom_key", "custom_value")

        config_dict = config.to_dict()

        # Check that all expected keys are present
        expected_keys = [
            "base_url",
            "item_types",
            "asset_types",
            "rate_limits",
            "timeouts",
            "max_retries",
            "max_roi_area_km2",
            "default_crs",
        ]

        for key in expected_keys:
            assert key in config_dict

        # Check custom key is included
        assert config_dict["custom_key"] == "custom_value"

        # Check specific values
        assert config_dict["base_url"] == "https://api.planet.com/data/v1"
        assert config_dict["item_types"] == ["PSScene"]

    def test_property_accessors(self):
        """Test property accessors for common configuration values."""
        config = PlanetScopeConfig()

        assert config.base_url == "https://api.planet.com/data/v1"
        assert config.item_types == ["PSScene"]
        assert config.asset_types == ["ortho_analytic_4b", "ortho_analytic_4b_xml"]
        assert isinstance(config.rate_limits, dict)
        assert isinstance(config.timeouts, dict)

        # Test that rate_limits has expected keys
        assert "search" in config.rate_limits
        assert "activate" in config.rate_limits
        assert "download" in config.rate_limits

        # Test that timeouts has expected keys
        assert "connect" in config.timeouts
        assert "read" in config.timeouts

    def test_configuration_precedence(self, tmp_path, monkeypatch):
        """Test that configuration sources have correct precedence."""
        # Environment variable (highest precedence)
        monkeypatch.setenv("PLANETSCOPE_MAX_RETRIES", "15")

        # Custom config file (medium precedence)
        config_file = tmp_path / "test_config.json"
        config_data = {"max_retries": 12}
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        # System config (lower precedence)
        system_config = {"max_retries": 8}
        mock_file_content = json.dumps(system_config)

        mock_path = Mock()
        mock_path.exists.return_value = True
        monkeypatch.setattr(
            "pathlib.Path.home", lambda: Mock(__truediv__=lambda self, other: mock_path)
        )

        with patch("builtins.open", mock_open(read_data=mock_file_content)):
            config = PlanetScopeConfig(config_file=config_file)

            # Environment variable should take precedence
            assert config.get("max_retries") == 15

    def test_default_constants(self):
        """Test that default constants are set correctly."""
        assert PlanetScopeConfig.BASE_URL == "https://api.planet.com/data/v1"
        assert PlanetScopeConfig.TILE_URL == "https://tiles.planet.com/data/v1"
        assert PlanetScopeConfig.DEFAULT_ITEM_TYPES == ["PSScene"]
        assert "ortho_analytic_4b" in PlanetScopeConfig.DEFAULT_ASSET_TYPES

        # Test rate limits
        assert PlanetScopeConfig.RATE_LIMITS["search"] == 10
        assert PlanetScopeConfig.RATE_LIMITS["activate"] == 5
        assert PlanetScopeConfig.RATE_LIMITS["download"] == 15

        # Test timeouts
        assert PlanetScopeConfig.TIMEOUTS["connect"] == 10.0
        assert PlanetScopeConfig.TIMEOUTS["read"] == 30.0

        # Test other constants
        assert PlanetScopeConfig.MAX_RETRIES == 3
        assert PlanetScopeConfig.MAX_ROI_AREA_KM2 == 10000
        assert PlanetScopeConfig.DEFAULT_CRS == "EPSG:4326"

    @pytest.mark.parametrize(
        "env_var,config_key,test_value",
        [
            ("PLANETSCOPE_BASE_URL", "base_url", "https://test.api.com"),
            ("PLANETSCOPE_MAX_RETRIES", "max_retries", "5"),
            ("PLANETSCOPE_MAX_ROI_AREA", "max_roi_area_km2", "8000"),
            ("PLANETSCOPE_DEFAULT_CRS", "default_crs", "EPSG:4326"),
        ],
    )
    def test_environment_variable_mapping(
        self, env_var, config_key, test_value, monkeypatch
    ):
        """Test that environment variables map correctly to config keys."""
        monkeypatch.setenv(env_var, test_value)

        config = PlanetScopeConfig()

        expected_value = test_value
        if config_key in ["max_retries", "max_roi_area_km2"]:
            expected_value = int(test_value)

        assert config.get(config_key) == expected_value


class TestDefaultConfig:
    """Test cases for the default global config instance."""

    def test_default_config_exists(self):
        """Test that default_config instance exists and is properly configured."""
        from planetscope_py.config import default_config

        assert isinstance(default_config, PlanetScopeConfig)
        assert default_config.base_url == "https://api.planet.com/data/v1"

    def test_default_config_is_singleton_like(self):
        """Test that default_config behaves consistently."""
        from planetscope_py.config import default_config

        # Modify the default config
        original_retries = default_config.get("max_retries", 3)
        default_config.set("max_retries", 99)

        # Import again and check persistence
        from planetscope_py.config import default_config as config2

        assert config2.get("max_retries") == 99

        # Restore original value
        default_config.set("max_retries", original_retries)
