"""Custom exception classes for planetscope-py.

This module defines all custom exceptions used throughout the library,
following Planet API error patterns and providing clear, actionable error messages.
"""

from typing import Any, Dict, Optional


class PlanetScopeError(Exception):
    """Base exception class for all planetscope-py errors.

    All custom exceptions in this library inherit from this base class.

    Args:
        message: Human-readable error description
        details: Optional dictionary with additional error context
    """

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message}. Details: {self.details}"
        return self.message


class AuthenticationError(PlanetScopeError):
    """Raised when API authentication fails.

    This occurs when:
    - No API key is found (env var, config file, or parameter)
    - API key is invalid or expired
    - API key lacks required permissions

    Example:
        raise AuthenticationError(
            "Invalid API key",
            {"status_code": 401, "hint": "Check your API key at planet.com/account"}
        )
    """


class ValidationError(PlanetScopeError):
    """Raised when input validation fails.

    This occurs when:
    - Invalid GeoJSON geometry provided
    - Date ranges are malformed or invalid
    - Parameters are outside acceptable ranges
    - ROI size exceeds limits

    Example:
        raise ValidationError(
            "Invalid geometry",
            {"geometry": geom, "error": "Polygon must be closed"}
        )
    """


class RateLimitError(PlanetScopeError):
    """Raised when API rate limits are exceeded.

    Planet API rate limits (per API key):
    - General endpoints: 10 requests/second
    - Activation endpoints: 5 requests/second
    - Download endpoints: 15 requests/second

    Example:
        raise RateLimitError(
            "Rate limit exceeded",
            {"retry_after": 60, "endpoint": "search"}
        )
    """


class APIError(PlanetScopeError):
    """Raised when Planet API returns an error.

    This covers HTTP errors, server errors, and API-specific errors
    that aren't authentication or rate limiting issues.

    Example:
        raise APIError(
            "Server error",
            {"status_code": 500, "response": response_text}
        )
    """


class ConfigurationError(PlanetScopeError):
    """Raised when configuration is invalid or missing.

    This occurs when:
    - Configuration files are malformed
    - Required settings are missing
    - Invalid configuration values provided

    Example:
        raise ConfigurationError(
            "Invalid config file",
            {"file": "~/.planet.json", "error": "Not valid JSON"}
        )
    """


class AssetError(PlanetScopeError):
    """Raised when asset operations fail.

    This occurs when:
    - Asset activation fails or times out
    - Requested asset type not available
    - Download permissions insufficient

    Example:
        raise AssetError(
            "Asset activation failed",
            {"asset_id": "ortho_analytic_4b", "status": "failed"}
        )
    """
