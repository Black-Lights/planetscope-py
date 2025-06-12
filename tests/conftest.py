"""Pytest configuration and shared fixtures for planetscope-py tests."""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from planetscope_py.config import PlanetScopeConfig


@pytest.fixture
def sample_api_key():
    """Provide a sample API key for testing."""
    return "pl_test_key_12345_abcdef"


@pytest.fixture
def sample_point_geometry():
    """Provide a sample Point geometry."""
    return {"type": "Point", "coordinates": [-122.4194, 37.7749]}  # San Francisco


@pytest.fixture
def sample_polygon_geometry():
    """Provide a sample Polygon geometry."""
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [-122.5, 37.7],
                [-122.3, 37.7],
                [-122.3, 37.8],
                [-122.5, 37.8],
                [-122.5, 37.7],
            ]
        ],
    }


@pytest.fixture
def sample_large_polygon():
    """Provide a large polygon that exceeds size limits."""
    # Create a large polygon covering roughly 100x100 degree area
    return {
        "type": "Polygon",
        "coordinates": [[[-50, -50], [50, -50], [50, 50], [-50, 50], [-50, -50]]],
    }


@pytest.fixture
def sample_invalid_geometry():
    """Provide an invalid geometry for testing."""
    return {
        "type": "Polygon",
        "coordinates": [
            [[0, 0], [1, 0], [1, 1]]  # Not closed - missing final coordinate
        ],
    }


@pytest.fixture
def sample_config_file_content():
    """Provide sample config file content."""
    return json.dumps(
        {
            "api_key": "config_file_api_key",
            "base_url": "https://api.planet.com/data/v1",
            "max_retries": 5,
        }
    )


@pytest.fixture
def mock_config():
    """Provide a mock configuration for testing."""
    config = PlanetScopeConfig()
    # Override some values for testing
    config.set("max_retries", 2)
    config.set("max_roi_area_km2", 1000)
    return config


@pytest.fixture
def mock_successful_response():
    """Provide a mock successful HTTP response."""
    response = Mock()
    response.status_code = 200
    response.json.return_value = {"status": "success"}
    response.text = '{"status": "success"}'
    response.headers = {"Content-Type": "application/json"}
    return response


@pytest.fixture
def mock_unauthorized_response():
    """Provide a mock unauthorized HTTP response."""
    response = Mock()
    response.status_code = 401
    response.json.return_value = {"error": "Unauthorized"}
    response.text = '{"error": "Unauthorized"}'
    response.headers = {"Content-Type": "application/json"}
    return response


@pytest.fixture
def mock_rate_limit_response():
    """Provide a mock rate limit HTTP response."""
    response = Mock()
    response.status_code = 429
    response.json.return_value = {"error": "Rate limit exceeded"}
    response.text = '{"error": "Rate limit exceeded"}'
    response.headers = {"Content-Type": "application/json", "Retry-After": "60"}
    return response


@pytest.fixture
def sample_search_results():
    """Provide sample Planet API search results."""
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "id": "20250101_123456_78_9abc",
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [-122.5, 37.7],
                            [-122.4, 37.7],
                            [-122.4, 37.8],
                            [-122.5, 37.8],
                            [-122.5, 37.7],
                        ]
                    ],
                },
                "properties": {
                    "item_type": "PSScene",
                    "acquired": "2025-01-01T12:34:56.000Z",
                    "cloud_cover": 0.1,
                    "pixel_resolution": 3.0,
                    "strip_id": "123456",
                },
                "_permissions": ["assets.basic_analytic_4b:download"],
                "_links": {
                    "_self": "https://api.planet.com/data/v1/item-types/PSScene/items/20250101_123456_78_9abc",
                    "assets": "https://api.planet.com/data/v1/item-types/PSScene/items/20250101_123456_78_9abc/assets",
                },
            },
            {
                "id": "20250102_654321_87_cba9",
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [-122.6, 37.6],
                            [-122.5, 37.6],
                            [-122.5, 37.7],
                            [-122.6, 37.7],
                            [-122.6, 37.6],
                        ]
                    ],
                },
                "properties": {
                    "item_type": "PSScene",
                    "acquired": "2025-01-02T09:15:30.000Z",
                    "cloud_cover": 0.05,
                    "pixel_resolution": 3.0,
                    "strip_id": "654321",
                },
                "_permissions": ["assets.basic_analytic_4b:download"],
                "_links": {
                    "_self": "https://api.planet.com/data/v1/item-types/PSScene/items/20250102_654321_87_cba9",
                    "assets": "https://api.planet.com/data/v1/item-types/PSScene/items/20250102_654321_87_cba9/assets",
                },
            },
        ],
    }


@pytest.fixture
def sample_asset_info():
    """Provide sample Planet API asset information."""
    return {
        "ortho_analytic_4b": {
            "status": "active",
            "type": "image/tiff; application=geotiff; profile=cloud-optimized",
            "_links": {
                "_self": "https://api.planet.com/data/v1/item-types/PSScene/items/test/assets/ortho_analytic_4b",
                "activate": "https://api.planet.com/data/v1/item-types/PSScene/items/test/assets/ortho_analytic_4b/activate",
                "download": "https://api.planet.com/data/v1/item-types/PSScene/items/test/assets/ortho_analytic_4b/download",
            },
            "location": "https://storage.googleapis.com/download-url/ortho_analytic_4b.tif",
        },
        "ortho_analytic_4b_xml": {
            "status": "active",
            "type": "text/xml",
            "_links": {
                "_self": "https://api.planet.com/data/v1/item-types/PSScene/items/test/assets/ortho_analytic_4b_xml",
                "activate": "https://api.planet.com/data/v1/item-types/PSScene/items/test/assets/ortho_analytic_4b_xml/activate",
                "download": "https://api.planet.com/data/v1/item-types/PSScene/items/test/assets/ortho_analytic_4b_xml/download",
            },
            "location": "https://storage.googleapis.com/download-url/ortho_analytic_4b_metadata.xml",
        },
    }


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    """Clean environment variables before each test."""
    # Remove Planet-related environment variables
    env_vars_to_remove = [
        "PL_API_KEY",
        "PLANETSCOPE_BASE_URL",
        "PLANETSCOPE_MAX_RETRIES",
        "PLANETSCOPE_LOG_LEVEL",
    ]

    for var in env_vars_to_remove:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def temp_config_file(tmp_path, sample_config_file_content):
    """Create a temporary config file."""
    config_file = tmp_path / "planet_config.json"
    config_file.write_text(sample_config_file_content)
    return config_file


@pytest.fixture
def mock_home_config(monkeypatch, sample_config_file_content):
    """Mock ~/.planet.json config file."""
    mock_path = Mock()
    mock_path.exists.return_value = True

    # Mock pathlib.Path.home() to return our mock path
    monkeypatch.setattr(
        "pathlib.Path.home", lambda: Mock(__truediv__=lambda self, other: mock_path)
    )

    # Mock open to return our config content
    mock_open_func = Mock()
    mock_open_func.return_value.__enter__ = Mock(
        return_value=Mock(read=Mock(return_value=sample_config_file_content))
    )
    mock_open_func.return_value.__exit__ = Mock(return_value=None)

    monkeypatch.setattr("builtins.open", mock_open_func)

    return mock_path


# Custom pytest markers for different test categories
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests for individual functions")
    config.addinivalue_line(
        "markers", "integration: Integration tests for multiple components"
    )
    config.addinivalue_line("markers", "auth: Authentication-related tests")
    config.addinivalue_line("markers", "validation: Input validation tests")
    config.addinivalue_line("markers", "slow: Slow-running tests")


# Test utilities
def assert_valid_geometry(geometry):
    """Helper to assert geometry is valid."""
    from planetscope_py.utils import validate_geometry

    # Should not raise exception
    validated = validate_geometry(geometry)
    assert validated == geometry


def assert_raises_validation_error(func, *args, **kwargs):
    """Helper to assert function raises ValidationError."""
    from planetscope_py.exceptions import ValidationError

    with pytest.raises(ValidationError):
        func(*args, **kwargs)
