"""Tests for planetscope_py.exceptions module."""

import pytest

from planetscope_py.exceptions import (
    APIError,
    AssetError,
    AuthenticationError,
    ConfigurationError,
    PlanetScopeError,
    RateLimitError,
    ValidationError,
)


class TestPlanetScopeError:
    """Test cases for base PlanetScopeError class."""

    def test_basic_exception(self):
        """Test basic exception creation and properties."""
        message = "Test error message"
        error = PlanetScopeError(message)

        assert str(error) == message
        assert error.message == message
        assert error.details == {}

    def test_exception_with_details(self):
        """Test exception with additional details."""
        message = "Test error"
        details = {"code": 123, "context": "test_function"}
        error = PlanetScopeError(message, details)

        assert error.message == message
        assert error.details == details
        assert "Details: " in str(error)
        assert "123" in str(error)

    def test_exception_inheritance(self):
        """Test that PlanetScopeError inherits from Exception."""
        error = PlanetScopeError("test")
        assert isinstance(error, Exception)

    def test_exception_str_representation(self):
        """Test string representation of exception."""
        # Without details
        error1 = PlanetScopeError("Simple error")
        assert str(error1) == "Simple error"

        # With details
        error2 = PlanetScopeError("Complex error", {"key": "value"})
        result = str(error2)
        assert "Complex error" in result
        assert "Details:" in result
        assert "key" in result

    def test_exception_with_none_details(self):
        """Test exception with None details parameter."""
        error = PlanetScopeError("Test message", None)
        assert error.details == {}

    def test_exception_with_empty_details(self):
        """Test exception with empty details dictionary."""
        error = PlanetScopeError("Test message", {})
        assert error.details == {}
        assert str(error) == "Test message"


class TestAuthenticationError:
    """Test cases for AuthenticationError class."""

    def test_authentication_error_inheritance(self):
        """Test that AuthenticationError inherits from PlanetScopeError."""
        error = AuthenticationError("Auth failed")
        assert isinstance(error, PlanetScopeError)
        assert isinstance(error, Exception)

    def test_authentication_error_with_status_code(self):
        """Test authentication error with HTTP status code."""
        message = "Invalid API key"
        details = {
            "status_code": 401,
            "hint": "Check your API key at planet.com/account",
        }
        error = AuthenticationError(message, details)

        assert error.message == message
        assert error.details["status_code"] == 401
        assert "hint" in error.details
        assert "Invalid API key" in str(error)

    def test_authentication_error_context(self):
        """Test authentication error with contextual information."""
        details = {
            "methods": [
                "Set PL_API_KEY environment variable",
                "Create ~/.planet.json config file",
            ],
            "help_url": "https://www.planet.com/account/#/",
        }
        error = AuthenticationError("No API key found", details)

        assert "methods" in error.details
        assert len(error.details["methods"]) == 2
        assert "help_url" in error.details
        assert error.details["help_url"] == "https://www.planet.com/account/#/"

    def test_authentication_error_api_key_masking(self):
        """Test authentication error with masked API key."""
        details = {"api_key_prefix": "pl_12...89ab", "status_code": 401}
        error = AuthenticationError("Invalid API key", details)

        assert "api_key_prefix" in error.details
        assert "..." in error.details["api_key_prefix"]


class TestValidationError:
    """Test cases for ValidationError class."""

    def test_validation_error_inheritance(self):
        """Test that ValidationError inherits from PlanetScopeError."""
        error = ValidationError("Validation failed")
        assert isinstance(error, PlanetScopeError)
        assert isinstance(error, Exception)

    def test_geometry_validation_error(self):
        """Test validation error for geometry issues."""
        geometry = {"type": "Point", "coordinates": [200, 0]}  # Invalid longitude
        details = {
            "geometry": geometry,
            "error": "Longitude must be between -180 and 180",
            "bounds": (200, 0, 200, 0),
        }
        error = ValidationError("Invalid geometry", details)

        assert "Invalid geometry" in str(error)
        assert error.details["geometry"] == geometry
        assert "bounds" in error.details

    def test_date_validation_error(self):
        """Test validation error for date issues."""
        details = {
            "start_date": "2025-12-31",
            "end_date": "2025-01-01",
            "error": "Start date must be before end date",
        }
        error = ValidationError("Invalid date range", details)

        assert error.details["start_date"] == "2025-12-31"
        assert error.details["end_date"] == "2025-01-01"
        assert "Invalid date range" in str(error)

    def test_parameter_validation_error(self):
        """Test validation error for parameter issues."""
        details = {
            "parameter": "cloud_cover",
            "value": -0.5,
            "valid_range": "0.0 to 1.0",
        }
        error = ValidationError("Invalid parameter value", details)

        assert error.details["parameter"] == "cloud_cover"
        assert error.details["value"] == -0.5
        assert error.details["valid_range"] == "0.0 to 1.0"

    def test_polygon_validation_error(self):
        """Test validation error for polygon-specific issues."""
        geometry = {
            "type": "Polygon",
            "coordinates": [[[0, 0], [1, 0], [1, 1]]],  # Not closed
        }
        details = {
            "geometry": geometry,
            "error": "Polygon must be closed",
            "first": [0, 0],
            "last": [1, 1],
        }
        error = ValidationError("Invalid polygon geometry", details)

        assert error.details["first"] == [0, 0]
        assert error.details["last"] == [1, 1]

    def test_item_type_validation_error(self):
        """Test validation error for invalid item types."""
        details = {
            "item_type": "InvalidType",
            "valid_types": ["PSScene", "REOrthoTile", "REScene"],
        }
        error = ValidationError("Invalid item type: InvalidType", details)

        assert error.details["item_type"] == "InvalidType"
        assert "PSScene" in error.details["valid_types"]


class TestRateLimitError:
    """Test cases for RateLimitError class."""

    def test_rate_limit_error_inheritance(self):
        """Test that RateLimitError inherits from PlanetScopeError."""
        error = RateLimitError("Rate limit exceeded")
        assert isinstance(error, PlanetScopeError)
        assert isinstance(error, Exception)

    def test_rate_limit_with_retry_info(self):
        """Test rate limit error with retry information."""
        details = {
            "retry_after": 60,
            "endpoint": "search",
            "limit": "10 requests per second",
        }
        error = RateLimitError("Rate limit exceeded", details)

        assert error.details["retry_after"] == 60
        assert error.details["endpoint"] == "search"
        assert "Rate limit exceeded" in str(error)

    def test_rate_limit_different_endpoints(self):
        """Test rate limit errors for different API endpoints."""
        endpoints = ["search", "activate", "download"]
        limits = [10, 5, 15]

        for endpoint, limit in zip(endpoints, limits):
            details = {"endpoint": endpoint, "limit": limit}
            error = RateLimitError(f"Rate limit for {endpoint}", details)

            assert error.details["endpoint"] == endpoint
            assert error.details["limit"] == limit
            assert endpoint in str(error)

    def test_rate_limit_with_headers(self):
        """Test rate limit error with HTTP headers."""
        details = {
            "retry_after": 120,
            "remaining": 0,
            "reset_time": "2025-06-12T14:30:00Z",
            "headers": {
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": "1718197800",
            },
        }
        error = RateLimitError("API rate limit exceeded", details)

        assert error.details["remaining"] == 0
        assert "headers" in error.details
        assert error.details["headers"]["X-RateLimit-Remaining"] == "0"


class TestAPIError:
    """Test cases for APIError class."""

    def test_api_error_inheritance(self):
        """Test that APIError inherits from PlanetScopeError."""
        error = APIError("API error")
        assert isinstance(error, PlanetScopeError)
        assert isinstance(error, Exception)

    def test_api_error_with_http_status(self):
        """Test API error with HTTP status information."""
        details = {
            "status_code": 500,
            "response": "Internal Server Error",
            "url": "https://api.planet.com/data/v1/quick-search",
        }
        error = APIError("Server error", details)

        assert error.details["status_code"] == 500
        assert "Server error" in str(error)
        assert error.details["url"].startswith("https://api.planet.com")

    def test_api_error_with_planet_response(self):
        """Test API error with Planet API response details."""
        details = {
            "status_code": 400,
            "response": {
                "error": "Invalid request",
                "details": "Missing required parameter: geometry",
            },
            "request_id": "abc123",
        }
        error = APIError("Bad request", details)

        assert error.details["status_code"] == 400
        assert "request_id" in error.details
        assert error.details["request_id"] == "abc123"
        assert isinstance(error.details["response"], dict)

    def test_api_error_network_issues(self):
        """Test API error for network-related issues."""
        details = {
            "error_type": "connection_timeout",
            "timeout": 30.0,
            "url": "https://api.planet.com/data/v1/",
            "attempt": 3,
        }
        error = APIError("Connection timeout", details)

        assert error.details["error_type"] == "connection_timeout"
        assert error.details["timeout"] == 30.0
        assert error.details["attempt"] == 3

    def test_api_error_json_parse_error(self):
        """Test API error for JSON parsing issues."""
        details = {
            "status_code": 200,
            "response_text": "<!DOCTYPE html><html>...",
            "content_type": "text/html",
            "expected": "application/json",
        }
        error = APIError("Invalid JSON response", details)

        assert error.details["content_type"] == "text/html"
        assert error.details["expected"] == "application/json"


class TestConfigurationError:
    """Test cases for ConfigurationError class."""

    def test_configuration_error_inheritance(self):
        """Test that ConfigurationError inherits from PlanetScopeError."""
        error = ConfigurationError("Config error")
        assert isinstance(error, PlanetScopeError)
        assert isinstance(error, Exception)

    def test_config_file_error(self):
        """Test configuration error for file issues."""
        details = {"file": "~/.planet.json", "error": "File not found"}
        error = ConfigurationError("Invalid config file", details)

        assert error.details["file"] == "~/.planet.json"
        assert error.details["error"] == "File not found"
        assert "Invalid config file" in str(error)

    def test_config_parsing_error(self):
        """Test configuration error for parsing issues."""
        details = {
            "file": "/path/to/config.json",
            "error": "Invalid JSON syntax",
            "line": 5,
            "column": 12,
        }
        error = ConfigurationError("Failed to parse config", details)

        assert error.details["line"] == 5
        assert error.details["column"] == 12
        assert "Failed to parse config" in str(error)

    def test_config_env_variable_error(self):
        """Test configuration error for environment variable issues."""
        details = {
            "env_var": "PLANETSCOPE_MAX_RETRIES",
            "value": "not_a_number",
            "expected_type": "integer",
        }
        error = ConfigurationError("Invalid environment variable", details)

        assert error.details["env_var"] == "PLANETSCOPE_MAX_RETRIES"
        assert error.details["expected_type"] == "integer"

    def test_config_permission_error(self):
        """Test configuration error for permission issues."""
        details = {
            "file": "/etc/planet/config.json",
            "error": "Permission denied",
            "permissions": "644",
        }
        error = ConfigurationError("Cannot read config file", details)

        assert error.details["permissions"] == "644"
        assert "Permission denied" in error.details["error"]


class TestAssetError:
    """Test cases for AssetError class."""

    def test_asset_error_inheritance(self):
        """Test that AssetError inherits from PlanetScopeError."""
        error = AssetError("Asset error")
        assert isinstance(error, PlanetScopeError)
        assert isinstance(error, Exception)

    def test_asset_activation_error(self):
        """Test asset error for activation issues."""
        details = {
            "asset_id": "ortho_analytic_4b",
            "item_id": "20250101_123456_78_9abc",
            "status": "failed",
            "reason": "Insufficient permissions",
        }
        error = AssetError("Asset activation failed", details)

        assert error.details["asset_id"] == "ortho_analytic_4b"
        assert error.details["status"] == "failed"
        assert "Asset activation failed" in str(error)

    def test_asset_download_error(self):
        """Test asset error for download issues."""
        details = {
            "asset_url": "https://storage.googleapis.com/...",
            "error": "Connection timeout",
            "retry_count": 3,
            "file_size": 1024000,
        }
        error = AssetError("Download failed", details)

        assert "asset_url" in error.details
        assert error.details["retry_count"] == 3
        assert error.details["file_size"] == 1024000

    def test_asset_not_available_error(self):
        """Test asset error when asset is not available."""
        details = {
            "asset_type": "ortho_visual",
            "item_id": "20250101_123456_78_9abc",
            "available_assets": ["ortho_analytic_4b", "ortho_analytic_4b_xml"],
        }
        error = AssetError("Asset type not available", details)

        assert error.details["asset_type"] == "ortho_visual"
        assert len(error.details["available_assets"]) == 2

    def test_asset_activation_timeout(self):
        """Test asset error for activation timeout."""
        details = {
            "asset_id": "ortho_analytic_4b",
            "timeout": 300,
            "elapsed": 310,
            "status": "activating",
        }
        error = AssetError("Asset activation timeout", details)

        assert error.details["timeout"] == 300
        assert error.details["elapsed"] == 310
        assert error.details["status"] == "activating"


class TestExceptionChaining:
    """Test cases for exception chaining and context."""

    def test_exception_with_cause(self):
        """Test that exceptions can be properly chained."""
        try:
            # Simulate a nested exception scenario
            try:
                raise ValueError("Original error")
            except ValueError as e:
                raise ValidationError(
                    "Validation failed due to underlying error",
                    {"original_error": str(e)},
                ) from e
        except ValidationError as validation_error:
            assert "Validation failed" in str(validation_error)
            assert "original_error" in validation_error.details
            assert validation_error.__cause__ is not None
            assert isinstance(validation_error.__cause__, ValueError)

    def test_exception_context_preservation(self):
        """Test that exception context is preserved."""
        details = {
            "function": "validate_geometry",
            "input": {"type": "Invalid"},
            "stack_info": "test_context",
        }
        error = ValidationError("Context test", details)

        assert error.details["function"] == "validate_geometry"
        assert error.details["stack_info"] == "test_context"
        assert error.details["input"]["type"] == "Invalid"

    def test_nested_exception_handling(self):
        """Test handling of nested exceptions."""
        try:
            try:
                try:
                    raise ConnectionError("Network failure")
                except ConnectionError as e:
                    raise APIError(
                        "API connection failed", {"network_error": str(e)}
                    ) from e
            except APIError as e:
                raise AuthenticationError(
                    "Auth failed due to network", {"api_error": str(e)}
                ) from e
        except AuthenticationError as final_error:
            assert "Auth failed due to network" in str(final_error)
            assert "api_error" in final_error.details
            assert final_error.__cause__ is not None


class TestExceptionIntegration:
    """Integration tests for exception usage patterns."""

    def test_typical_authentication_flow(self):
        """Test typical authentication error flow."""
        # Simulate authentication failure
        try:
            api_key = None
            if not api_key:
                raise AuthenticationError(
                    "No Planet API key found",
                    {
                        "methods": [
                            "Set PL_API_KEY environment variable",
                            "Create ~/.planet.json config file",
                        ],
                        "help_url": "https://www.planet.com/account/#/",
                    },
                )
        except AuthenticationError as e:
            assert "No Planet API key found" in str(e)
            assert len(e.details["methods"]) == 2
            assert e.details["help_url"].startswith("https://")

    def test_typical_validation_flow(self):
        """Test typical validation error flow."""
        # Simulate geometry validation failure
        invalid_geometry = {"type": "Point", "coordinates": [200, 0]}

        try:
            if invalid_geometry["coordinates"][0] > 180:
                raise ValidationError(
                    "Invalid longitude coordinate",
                    {
                        "geometry": invalid_geometry,
                        "coordinate": invalid_geometry["coordinates"][0],
                        "valid_range": "[-180, 180]",
                    },
                )
        except ValidationError as e:
            assert "Invalid longitude" in str(e)
            assert e.details["coordinate"] == 200
            assert e.details["valid_range"] == "[-180, 180]"

    def test_typical_api_error_flow(self):
        """Test typical API error handling flow."""
        # Simulate API error response
        try:
            status_code = 500
            if status_code >= 500:
                raise APIError(
                    "Planet API server error",
                    {
                        "status_code": status_code,
                        "retry_after": 60,
                        "url": "https://api.planet.com/data/v1/quick-search",
                    },
                )
        except APIError as e:
            assert "server error" in str(e)
            assert e.details["status_code"] == 500
            assert e.details["retry_after"] == 60

    @pytest.mark.parametrize(
        "exception_class,message,details",
        [
            (AuthenticationError, "Auth failed", {"status": 401}),
            (ValidationError, "Invalid input", {"field": "geometry"}),
            (RateLimitError, "Too many requests", {"retry_after": 30}),
            (APIError, "Server error", {"status_code": 500}),
            (ConfigurationError, "Bad config", {"file": "config.json"}),
            (AssetError, "Asset failed", {"asset_id": "test_asset"}),
        ],
    )
    def test_all_exception_types(self, exception_class, message, details):
        """Test that all exception types work correctly."""
        error = exception_class(message, details)

        assert isinstance(error, PlanetScopeError)
        assert error.message == message
        assert error.details == details
        assert message in str(error)

    def test_exception_details_immutability(self):
        """Test that exception details cannot be accidentally modified."""
        original_details = {"key": "value", "nested": {"inner": "data"}}
        error = PlanetScopeError("Test", original_details.copy())

        # Modifying the original dict shouldn't affect the exception
        original_details["key"] = "modified"
        assert error.details["key"] == "value"

    def test_exception_with_complex_details(self):
        """Test exception with complex nested details."""
        details = {
            "request": {
                "url": "https://api.planet.com/data/v1/quick-search",
                "method": "POST",
                "headers": {"Content-Type": "application/json"},
                "body": {"geometry": {"type": "Point", "coordinates": [0, 0]}},
            },
            "response": {
                "status": 400,
                "headers": {"Content-Type": "application/json"},
                "body": {"error": "Invalid geometry"},
            },
            "metadata": {
                "timestamp": "2025-06-12T14:30:00Z",
                "request_id": "req_123456",
                "user_agent": "planetscope-py/0.1.0",
            },
        }
        error = APIError("Complex API error", details)

        assert error.details["request"]["method"] == "POST"
        assert error.details["response"]["status"] == 400
        assert error.details["metadata"]["request_id"] == "req_123456"
        assert "Complex API error" in str(error)


class TestExceptionDocstrings:
    """Test that exceptions have proper documentation."""

    def test_base_exception_docstring(self):
        """Test that base exception has docstring."""
        assert PlanetScopeError.__doc__ is not None
        assert "Base exception class" in PlanetScopeError.__doc__

    def test_all_exceptions_have_docstrings(self):
        """Test that all exception classes have docstrings."""
        exception_classes = [
            PlanetScopeError,
            AuthenticationError,
            ValidationError,
            RateLimitError,
            APIError,
            ConfigurationError,
            AssetError,
        ]

        for exc_class in exception_classes:
            assert exc_class.__doc__ is not None
            assert len(exc_class.__doc__.strip()) > 0

    def test_exception_docstring_content(self):
        """Test that exception docstrings contain useful information."""
        # Check specific exceptions for key phrases
        assert "authentication" in AuthenticationError.__doc__.lower()
        assert "validation" in ValidationError.__doc__.lower()
        assert "rate limit" in RateLimitError.__doc__.lower()
        assert "api" in APIError.__doc__.lower()
        assert "configuration" in ConfigurationError.__doc__.lower()
        assert "asset" in AssetError.__doc__.lower()
