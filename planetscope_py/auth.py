"""Authentication handling for planetscope-py.

This module implements Planet API authentication following the patterns
established in Planet's official Python client and examples.
"""

import json
import os
import requests
from pathlib import Path
from typing import Optional, Tuple
from requests.auth import HTTPBasicAuth
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import PlanetScopeConfig
from .exceptions import AuthenticationError, ConfigurationError


class PlanetAuth:
    """Authentication manager for Planet Data API.
    
    Handles API key discovery, validation, and session management following
    Planet's established authentication patterns.
    
    API Key Priority Order:
    1. Direct parameter to constructor
    2. PL_API_KEY environment variable  
    3. ~/.planet.json config file
    4. Raise AuthenticationError if none found
    
    Example:
        # Auto-detect API key
        auth = PlanetAuth()
        
        # Explicit API key
        auth = PlanetAuth(api_key="your_key_here")
        
        # Get authenticated session
        session = auth.get_session()
        response = session.get("https://api.planet.com/data/v1/")
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[PlanetScopeConfig] = None
    ):
        """Initialize authentication.
        
        Args:
            api_key: Direct API key (highest priority)
            config: Configuration instance (optional)
            
        Raises:
            AuthenticationError: If no valid API key found
        """
        self.config = config or PlanetScopeConfig()
        self._api_key = None
        self._session = None
        
        # Discover API key following priority order
        self._api_key = self._discover_api_key(api_key)
        
        if not self._api_key:
            raise AuthenticationError(
                "No Planet API key found. Please set PL_API_KEY environment variable, "
                "create ~/.planet.json config file, or pass api_key parameter.",
                {
                    "methods": [
                        "Pass api_key parameter to PlanetAuth()",
                        "Set PL_API_KEY environment variable", 
                        "Create ~/.planet.json with 'api_key' field",
                    ],
                    "help_url": "https://www.planet.com/account/#/"
                }
            )
        
        # Validate API key on initialization
        self._validate_api_key()
    
    def _discover_api_key(self, explicit_key: Optional[str]) -> Optional[str]:
        """Discover API key from multiple sources.
        
        Args:
            explicit_key: API key passed directly to constructor
            
        Returns:
            API key string or None if not found
        """
        # 1. Explicit parameter (highest priority)
        if explicit_key:
            return explicit_key.strip()
        
        # 2. Environment variable
        env_key = os.environ.get('PL_API_KEY', '').strip()
        if env_key and not env_key.startswith('PASTE'):
            return env_key
        
        # 3. Configuration file ~/.planet.json
        config_key = self._load_api_key_from_config()
        if config_key:
            return config_key
        
        return None
    
    def _load_api_key_from_config(self) -> Optional[str]:
        """Load API key from ~/.planet.json config file.
        
        Returns:
            API key from config file or None if not found
        """
        try:
            config_path = Path.home() / ".planet.json"
        except (RuntimeError, OSError):
            # Fallback for cases where home directory detection fails
            import os
            home_dir = os.environ.get('USERPROFILE', os.environ.get('HOME', os.getcwd()))
            config_path = Path(home_dir) / ".planet.json"
        
        if not config_path.exists():
            return None
        
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
                api_key = config_data.get('api_key', '').strip()
                if api_key and not api_key.startswith('PASTE'):
                    return api_key
        except (json.JSONDecodeError, IOError, KeyError) as e:
            raise ConfigurationError(
                f"Failed to read API key from {config_path}",
                {"error": str(e), "file": str(config_path)}
            )
        
        return None
    
    def _validate_api_key(self) -> None:
        """Validate API key by making a test request.
        
        Raises:
            AuthenticationError: If API key is invalid
        """
        try:
            # Make a simple request to validate the key
            test_url = f"{self.config.base_url}/"
            response = requests.get(
                test_url,
                auth=HTTPBasicAuth(self._api_key, ''),
                timeout=self.config.timeouts['connect']
            )
            
            if response.status_code == 401:
                raise AuthenticationError(
                    "Invalid Planet API key",
                    {
                        "status_code": 401,
                        "hint": "Check your API key at https://www.planet.com/account/#/",
                        "api_key_prefix": f"{self._api_key[:8]}..." if len(self._api_key) > 8 else "***"
                    }
                )
            elif response.status_code != 200:
                raise AuthenticationError(
                    f"API key validation failed with status {response.status_code}",
                    {
                        "status_code": response.status_code,
                        "response": response.text[:200]
                    }
                )
                
        except requests.exceptions.RequestException as e:
            raise AuthenticationError(
                "Failed to validate API key due to network error",
                {"error": str(e)}
            )
    
    def get_session(self) -> requests.Session:
        """Get authenticated requests session with retry logic.
        
        Returns:
            Configured requests.Session with authentication and retry handling
        """
        if self._session is None:
            self._session = self._create_session()
        return self._session
    
    def _create_session(self) -> requests.Session:
        """Create configured requests session.
        
        Returns:
            Configured requests.Session instance
        """
        session = requests.Session()
        
        # Set authentication
        session.auth = HTTPBasicAuth(self._api_key, '')
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=self.config.MAX_RETRIES,
            backoff_factor=self.config.BACKOFF_FACTOR,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST", "PUT", "DELETE"]
        )
        
        # Mount adapter with retry strategy
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Set default headers
        session.headers.update({
            'User-Agent': f'planetscope-py/{self._get_version()}',
            'Content-Type': 'application/json'
        })
        
        return session
    
    def _get_version(self) -> str:
        """Get package version for User-Agent header."""
        try:
            from ._version import __version__
            return __version__
        except ImportError:
            return "unknown"
    
    def get_auth_tuple(self) -> Tuple[str, str]:
        """Get authentication tuple for requests.
        
        Returns:
            Tuple of (api_key, empty_password) for HTTPBasicAuth
        """
        return (self._api_key, '')
    
    @property
    def api_key(self) -> str:
        """Get the API key (masked for security).
        
        Returns:
            Masked API key string for logging/debugging
        """
        if len(self._api_key) > 8:
            return f"{self._api_key[:4]}...{self._api_key[-4:]}"
        return "***"
    
    @property
    def is_authenticated(self) -> bool:
        """Check if authentication is valid.
        
        Returns:
            True if API key is available and validated
        """
        return bool(self._api_key)
    
    def refresh_session(self) -> requests.Session:
        """Refresh the session (create new one).
        
        Useful if session state becomes invalid or needs reset.
        
        Returns:
            New authenticated session
        """
        self._session = None
        return self.get_session()