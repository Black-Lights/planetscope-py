"""Tests for planetscope_py.auth module."""

import json
import os
from pathlib import Path
from unittest.mock import Mock, mock_open, patch

import pytest
import requests

from planetscope_py.auth import PlanetAuth
from planetscope_py.exceptions import AuthenticationError, ConfigurationError


class TestPlanetAuth:
    """Test cases for PlanetAuth class."""

    def test_init_with_explicit_key(self):
        """Test initialization with explicit API key."""
        api_key = "test_key_12345"

        with patch.object(PlanetAuth, "_validate_api_key"):
            auth = PlanetAuth(api_key=api_key)
            assert auth._api_key == api_key
            assert auth.is_authenticated

    def test_init_with_env_var(self):
        """Test initialization with environment variable."""
        api_key = "env_key_67890"

        with patch.dict(os.environ, {"PL_API_KEY": api_key}):
            with patch.object(PlanetAuth, "_validate_api_key"):
                auth = PlanetAuth()
                assert auth._api_key == api_key

    def test_init_with_config_file(self):
        """Test initialization with config file."""
        api_key = "config_key_11111"
        config_data = {"api_key": api_key}

        mock_file_content = json.dumps(config_data)

        with patch.dict(os.environ, {}, clear=True):  # Clear env vars
            with patch("pathlib.Path.exists", return_value=True):
                with patch("builtins.open", mock_open(read_data=mock_file_content)):
                    with patch.object(PlanetAuth, "_validate_api_key"):
                        auth = PlanetAuth()
                        assert auth._api_key == api_key

    def test_init_no_key_found(self):
        """Test initialization when no API key is found."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("pathlib.Path.exists", return_value=False):
                with pytest.raises(AuthenticationError) as exc_info:
                    PlanetAuth()

                assert "No Planet API key found" in str(exc_info.value)
                assert "methods" in exc_info.value.details

    def test_api_key_priority_explicit_over_env(self):
        """Test that explicit key takes priority over environment variable."""
        explicit_key = "explicit_key"
        env_key = "env_key"

        with patch.dict(os.environ, {"PL_API_KEY": env_key}):
            with patch.object(PlanetAuth, "_validate_api_key"):
                auth = PlanetAuth(api_key=explicit_key)
                assert auth._api_key == explicit_key

    def test_api_key_priority_env_over_config(self):
        """Test that env var takes priority over config file."""
        env_key = "env_key"
        config_key = "config_key"
        config_data = {"api_key": config_key}

        mock_file_content = json.dumps(config_data)

        with patch.dict(os.environ, {"PL_API_KEY": env_key}):
            with patch("pathlib.Path.exists", return_value=True):
                with patch("builtins.open", mock_open(read_data=mock_file_content)):
                    with patch.object(PlanetAuth, "_validate_api_key"):
                        auth = PlanetAuth()
                        assert auth._api_key == env_key

    def test_ignore_paste_placeholder(self):
        """Test that PASTE placeholder values are ignored."""
        with patch.dict(os.environ, {"PL_API_KEY": "PASTE_YOUR_API_KEY_HERE"}):
            with patch("pathlib.Path.exists", return_value=False):
                with pytest.raises(AuthenticationError):
                    PlanetAuth()

    def test_validate_api_key_success(self):
        """Test successful API key validation."""
        api_key = "valid_key"

        mock_response = Mock()
        mock_response.status_code = 200

        with patch("requests.get", return_value=mock_response):
            # Should not raise exception
            auth = PlanetAuth(api_key=api_key)
            assert auth._api_key == api_key

    def test_validate_api_key_unauthorized(self):
        """Test API key validation with 401 response."""
        api_key = "invalid_key"

        mock_response = Mock()
        mock_response.status_code = 401

        with patch("requests.get", return_value=mock_response):
            with pytest.raises(AuthenticationError) as exc_info:
                PlanetAuth(api_key=api_key)

            assert "Invalid Planet API key" in str(exc_info.value)
            assert exc_info.value.details["status_code"] == 401

    def test_validate_api_key_server_error(self):
        """Test API key validation with server error."""
        api_key = "test_key"

        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        with patch("requests.get", return_value=mock_response):
            with pytest.raises(AuthenticationError) as exc_info:
                PlanetAuth(api_key=api_key)

            assert "status 500" in str(exc_info.value)

    def test_validate_api_key_network_error(self):
        """Test API key validation with network error."""
        api_key = "test_key"

        with patch("requests.get", side_effect=requests.exceptions.ConnectionError()):
            with pytest.raises(AuthenticationError) as exc_info:
                PlanetAuth(api_key=api_key)

            assert "network error" in str(exc_info.value)

    def test_config_file_invalid_json(self):
        """Test handling of invalid JSON in config file."""
        invalid_json = "{ invalid json }"

        with patch.dict(os.environ, {}, clear=True):
            with patch("pathlib.Path.exists", return_value=True):
                with patch("builtins.open", mock_open(read_data=invalid_json)):
                    with pytest.raises(ConfigurationError) as exc_info:
                        PlanetAuth()

                    assert "Failed to load system config file" in str(exc_info.value)

    def test_get_session(self):
        """Test getting authenticated session."""
        api_key = "test_key"

        with patch.object(PlanetAuth, "_validate_api_key"):
            auth = PlanetAuth(api_key=api_key)
            session = auth.get_session()

            assert isinstance(session, requests.Session)
            assert session.auth.username == api_key
            assert session.auth.password == ""

    def test_get_session_caching(self):
        """Test that session is cached and reused."""
        api_key = "test_key"

        with patch.object(PlanetAuth, "_validate_api_key"):
            auth = PlanetAuth(api_key=api_key)
            session1 = auth.get_session()
            session2 = auth.get_session()

            assert session1 is session2

    def test_refresh_session(self):
        """Test session refresh creates new session."""
        api_key = "test_key"

        with patch.object(PlanetAuth, "_validate_api_key"):
            auth = PlanetAuth(api_key=api_key)
            session1 = auth.get_session()
            session2 = auth.refresh_session()

            assert session1 is not session2
            assert isinstance(session2, requests.Session)

    def test_get_auth_tuple(self):
        """Test getting authentication tuple."""
        api_key = "test_key"

        with patch.object(PlanetAuth, "_validate_api_key"):
            auth = PlanetAuth(api_key=api_key)
            auth_tuple = auth.get_auth_tuple()

            assert auth_tuple == (api_key, "")

    def test_api_key_property_masking(self):
        """Test that API key property returns masked value."""
        api_key = "very_long_test_key_12345"

        with patch.object(PlanetAuth, "_validate_api_key"):
            auth = PlanetAuth(api_key=api_key)
            masked = auth.api_key

            assert "very" in masked
            assert "2345" in masked
            assert "..." in masked
            assert len(masked) < len(api_key)

    def test_api_key_property_short_key(self):
        """Test API key masking with short key."""
        api_key = "short"

        with patch.object(PlanetAuth, "_validate_api_key"):
            auth = PlanetAuth(api_key=api_key)
            masked = auth.api_key

            assert masked == "***"

    def test_session_headers(self):
        """Test that session has correct headers."""
        api_key = "test_key"

        with patch.object(PlanetAuth, "_validate_api_key"):
            auth = PlanetAuth(api_key=api_key)
            session = auth.get_session()

            assert "User-Agent" in session.headers
            assert "planetscope-py" in session.headers["User-Agent"]
            assert session.headers["Content-Type"] == "application/json"

    @pytest.mark.parametrize(
        "env_key,expected",
        [
            ("", None),
            ("   ", None),
            ("PASTE_YOUR_KEY", None),
            ("valid_key", "valid_key"),
            ("  valid_key  ", "valid_key"),  # Should be stripped
        ],
    )
    def test_env_key_handling(self, env_key, expected):
        """Test various environment variable values."""
        with patch.dict(os.environ, {"PL_API_KEY": env_key}):
            with patch("pathlib.Path.exists", return_value=False):
                if expected:
                    with patch.object(PlanetAuth, "_validate_api_key"):
                        auth = PlanetAuth()
                        assert auth._api_key == expected
                else:
                    with pytest.raises(AuthenticationError):
                        PlanetAuth()
