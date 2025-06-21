"""
Pytest configuration and shared fixtures for planetscope-py tests.

PERMANENT WINDOWS FIX: This completely overrides pytest's temp directory
behavior to avoid Windows username whitespace permission issues.
"""

import json
import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Generator, Dict, Any
from unittest.mock import Mock, patch, MagicMock
import pytest

# Add the parent directory to the Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from planetscope_py.config import PlanetScopeConfig


# =====================================================
# PERMANENT WINDOWS FIX - Override pytest's tmp_path
# =====================================================


def pytest_configure(config):
    """Configure pytest with Windows compatibility fixes."""
    # Force project-local temp directory
    project_root = Path(__file__).parent.parent
    temp_base = project_root / ".pytest_temp"

    # Ensure it exists
    temp_base.mkdir(exist_ok=True)

    # Override all temp-related environment variables
    os.environ["TMPDIR"] = str(temp_base)
    os.environ["TMP"] = str(temp_base)
    os.environ["TEMP"] = str(temp_base)

    # Configure markers
    config.addinivalue_line("markers", "unit: Unit tests for individual functions")
    config.addinivalue_line(
        "markers", "integration: Integration tests for multiple components"
    )
    config.addinivalue_line("markers", "auth: Authentication-related tests")
    config.addinivalue_line("markers", "validation: Input validation tests")
    config.addinivalue_line("markers", "slow: Slow-running tests")
    config.addinivalue_line("markers", "network: Tests that require network access")
    config.addinivalue_line("markers", "config: Configuration tests")
    config.addinivalue_line("markers", "utils: Utility tests")
    config.addinivalue_line("markers", "exceptions: Exception tests")
    # Phase 2 additions (minimal)
    config.addinivalue_line("markers", "query: Planet API query system tests")
    config.addinivalue_line("markers", "metadata: Metadata processing tests")
    config.addinivalue_line(
        "markers", "rate_limit: Rate limiting and retry logic tests"
    )
    config.addinivalue_line("markers", "mock_api: Tests using mock Planet API")
    config.addinivalue_line("markers", "api: Tests that require real Planet API key")


@pytest.fixture
def tmp_path(request):
    """
    COMPLETE OVERRIDE of pytest's tmp_path fixture.

    This completely bypasses pytest's temp directory logic that causes
    Windows permission issues with usernames containing spaces.
    """
    import uuid

    # Use project-local temp directory
    project_root = Path(__file__).parent.parent
    base_temp = project_root / ".pytest_temp"
    base_temp.mkdir(exist_ok=True)

    # Create unique directory for this test
    test_name = request.node.name
    safe_test_name = "".join(c for c in test_name if c.isalnum() or c in "-_")
    unique_id = uuid.uuid4().hex[:8]

    temp_dir = base_temp / f"{safe_test_name}_{unique_id}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    yield temp_dir

    # Cleanup
    try:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
    except (OSError, PermissionError):
        # If cleanup fails, leave it - it's in .gitignore
        pass


@pytest.fixture
def tmp_path_factory(request):
    """
    COMPLETE OVERRIDE of pytest's tmp_path_factory fixture.

    This ensures all temporary directory creation goes through our
    project-local directory system.
    """

    class SafeTmpPathFactory:
        def __init__(self):
            self.project_root = Path(__file__).parent.parent
            self.base_temp = self.project_root / ".pytest_temp"
            self.base_temp.mkdir(exist_ok=True)
            self._counter = 0

        def mktemp(self, basename="tmp", numbered=True):
            """Create a temporary directory."""
            import uuid

            if numbered:
                self._counter += 1
                dir_name = f"{basename}_{self._counter}_{uuid.uuid4().hex[:6]}"
            else:
                dir_name = f"{basename}_{uuid.uuid4().hex[:8]}"

            temp_dir = self.base_temp / dir_name
            temp_dir.mkdir(parents=True, exist_ok=True)
            return temp_dir

        def getbasetemp(self):
            """Get the base temp directory."""
            return self.base_temp

    return SafeTmpPathFactory()


# =====================================================
# AUTHENTICATION AND CONFIGURATION FIXTURES
# =====================================================


@pytest.fixture
def sample_api_key():
    """Provide a sample API key for testing."""
    return "pl_test_key_12345_abcdef"


@pytest.fixture
def mock_config():
    """Provide a mock configuration for testing."""
    config = PlanetScopeConfig()
    config.set("max_retries", 2)
    config.set("max_roi_area_km2", 1000)
    return config


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


# =====================================================
# GEOMETRY FIXTURES
# =====================================================


@pytest.fixture
def sample_point_geometry():
    """Provide a sample Point geometry."""
    return {"type": "Point", "coordinates": [-122.4194, 37.7749]}


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
    return {
        "type": "Polygon",
        "coordinates": [[[-50, -50], [50, -50], [50, 50], [-50, 50], [-50, -50]]],
    }


@pytest.fixture
def sample_invalid_geometry():
    """Provide an invalid geometry for testing."""
    return {
        "type": "Polygon",
        "coordinates": [[[0, 0], [1, 0], [1, 1]]],  # Not closed
    }


# =====================================================
# SAFE FILE FIXTURES
# =====================================================


@pytest.fixture
def temp_config_file(tmp_path, sample_config_file_content):
    """Create a temporary config file using our safe tmp_path."""
    config_file = tmp_path / "planet_config.json"
    config_file.write_text(sample_config_file_content)
    return config_file


@pytest.fixture
def mock_home_config(monkeypatch, sample_config_file_content):
    """Mock ~/.planet.json config file."""
    mock_path = Mock()
    mock_path.exists.return_value = True

    monkeypatch.setattr(
        "pathlib.Path.home", lambda: Mock(__truediv__=lambda self, other: mock_path)
    )

    mock_open_func = Mock()
    mock_open_func.return_value.__enter__ = Mock(
        return_value=Mock(read=Mock(return_value=sample_config_file_content))
    )
    mock_open_func.return_value.__exit__ = Mock(return_value=None)

    monkeypatch.setattr("builtins.open", mock_open_func)
    return mock_path


# =====================================================
# ENVIRONMENT CLEANUP
# =====================================================


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    """Clean environment variables before each test."""
    env_vars_to_remove = [
        "PL_API_KEY",
        "PLANET_API_KEY",
        "PLANETSCOPE_BASE_URL",
        "PLANETSCOPE_MAX_RETRIES",
        "PLANETSCOPE_LOG_LEVEL",
    ]

    for var in env_vars_to_remove:
        monkeypatch.delenv(var, raising=False)


# =====================================================
# HTTP RESPONSE MOCKS
# =====================================================


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


# =====================================================
# PLANET API DATA FIXTURES
# =====================================================


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
            }
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
        }
    }


# =====================================================
# PHASE 2 ADDITIONS - MINIMAL NEW FIXTURES
# =====================================================


@pytest.fixture
def mock_planet_api():
    """Provide mock Planet API for Phase 2 tests."""
    try:
        from tests.mock_planet_api import MockPlanetAPI

        return MockPlanetAPI()
    except ImportError:
        # If mock_planet_api.py doesn't exist yet, return basic mock
        return Mock()


@pytest.fixture
def sample_scene_metadata():
    """Provide sample scene metadata with realistic Planet API dates."""
    return {
        "scene_id": "test_scene_001",
        "item_type": "PSScene",
        "satellite_id": "test_satellite",
        "acquired": "2024-01-15T14:30:00.12345Z",  # Realistic format
        "cloud_cover": 0.15,
        "sun_elevation": 45.2,
        "usable_data": 0.92,
        "overall_quality": 0.85,
        "suitability": "good",
    }


@pytest.fixture
def sample_scenes_collection():
    """Provide sample scenes collection for Phase 2 tests."""
    return [
        {
            "type": "Feature",
            "id": "scene_1",
            "properties": {
                "id": "scene_1",
                "item_type": "PSScene",
                "acquired": "2024-01-15T14:30:00.000Z",
                "cloud_cover": 0.05,
                "sun_elevation": 50.0,
                "usable_data": 0.95,
            },
        },
        {
            "type": "Feature",
            "id": "scene_2",
            "properties": {
                "id": "scene_2",
                "item_type": "PSScene",
                "acquired": "2024-02-10T15:45:00.000Z",
                "cloud_cover": 0.25,
                "sun_elevation": 35.0,
                "usable_data": 0.80,
            },
        },
    ]


@pytest.fixture
def mock_rate_limiter():
    """Provide mock rate limiter for Phase 2 tests."""
    limiter = Mock()
    limiter.make_request.return_value = Mock(
        status_code=200, json=lambda: {"status": "ok"}
    )
    limiter.get_current_rate_status.return_value = {
        "search": {"limit": 10, "current_rate": 2}
    }
    return limiter


# =====================================================
# UTILITY FUNCTIONS
# =====================================================


def assert_valid_geometry(geometry):
    """Helper to assert geometry is valid."""
    from planetscope_py.utils import validate_geometry

    validated = validate_geometry(geometry)
    assert validated == geometry


def assert_raises_validation_error(func, *args, **kwargs):
    """Helper to assert function raises ValidationError."""
    from planetscope_py.exceptions import ValidationError

    with pytest.raises(ValidationError):
        func(*args, **kwargs)


# Test constants
TEST_API_KEY = "pl_test_key_12345_abcdef"
TEST_BASE_URL = "https://api.planet.com/data/v1"
