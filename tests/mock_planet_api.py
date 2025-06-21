#!/usr/bin/env python3
"""Mock Planet API for testing.

Provides realistic mock responses for Planet API endpoints
to enable comprehensive testing without actual API calls.
"""

import json
import time
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from urllib.parse import urlparse, parse_qs

import requests


class MockPlanetAPI:
    """Mock Planet API for testing purposes.

    Simulates Planet API responses for various endpoints including
    search, stats, assets, and download operations.

    Attributes:
        base_url: Mock API base URL
        rate_limits: Simulated rate limits per endpoint
        response_delays: Simulated response delays
        failure_scenarios: Configuration for simulating failures
    """

    def __init__(self):
        """Initialize mock Planet API."""
        self.base_url = "https://api.planet.com/data/v1"

        # Rate limiting simulation
        self.rate_limits = {"search": 10, "stats": 10, "assets": 20, "download": 5}
        self.request_counts = {endpoint: 0 for endpoint in self.rate_limits}
        self.last_reset = {endpoint: time.time() for endpoint in self.rate_limits}

        # Response configuration
        self.response_delays = {
            "search": (0.1, 0.5),  # Min, max delay in seconds
            "stats": (0.2, 0.8),
            "assets": (0.05, 0.2),
            "download": (1.0, 3.0),
        }

        # Failure simulation
        self.failure_scenarios = {
            "rate_limit_probability": 0.0,  # Probability of 429 response
            "server_error_probability": 0.0,  # Probability of 5xx response
            "network_error_probability": 0.0,  # Probability of network error
        }

        # Sample data
        self._initialize_sample_data()

    def _initialize_sample_data(self):
        """Initialize sample data for mock responses."""
        self.sample_scenes = self._generate_sample_scenes(100)
        self.sample_assets = self._generate_sample_assets()

    def _generate_sample_scenes(self, count: int) -> List[Dict]:
        """Generate sample scene data."""
        scenes = []
        base_date = datetime(2024, 1, 1)

        satellites = ["Planet_1", "Planet_2", "Planet_3", "Planet_4"]

        for i in range(count):
            # Generate acquisition date
            days_offset = random.randint(0, 365)
            acquired = base_date + timedelta(days=days_offset)

            # Generate geometry (small random polygons)
            base_lon = -122.4 + random.uniform(-0.1, 0.1)
            base_lat = 37.8 + random.uniform(-0.1, 0.1)
            size = random.uniform(0.01, 0.05)

            geometry = {
                "type": "Polygon",
                "coordinates": [
                    [
                        [base_lon, base_lat],
                        [base_lon + size, base_lat],
                        [base_lon + size, base_lat + size],
                        [base_lon, base_lat + size],
                        [base_lon, base_lat],
                    ]
                ],
            }

            # Generate properties
            scene = {
                "type": "Feature",
                "id": f"mock_scene_{i:04d}",
                "geometry": geometry,
                "properties": {
                    "id": f"mock_scene_{i:04d}",
                    "item_type": "PSScene",
                    "satellite_id": random.choice(satellites),
                    "provider": "planet",
                    "acquired": acquired.isoformat() + "Z",
                    "published": (acquired + timedelta(hours=2)).isoformat() + "Z",
                    "updated": (acquired + timedelta(hours=2)).isoformat() + "Z",
                    "cloud_cover": random.uniform(0.0, 0.8),
                    "sun_azimuth": random.uniform(0, 360),
                    "sun_elevation": random.uniform(10, 70),
                    "usable_data": random.uniform(0.6, 1.0),
                    "quality_category": random.choice(["standard", "test"]),
                    "pixel_resolution": 3.0,
                    "ground_control": random.choice([True, False]),
                },
            }
            scenes.append(scene)

        return scenes

    def _generate_sample_assets(self) -> Dict[str, Dict]:
        """Generate sample asset definitions."""
        return {
            "ortho_analytic_4b": {
                "type": "ortho_analytic_4b",
                "status": "inactive",
                "_links": {
                    "activate": "https://api.planet.com/data/v1/assets/activate",
                    "_self": "https://api.planet.com/data/v1/assets/ortho_analytic_4b",
                },
            },
            "ortho_analytic_4b_xml": {
                "type": "ortho_analytic_4b_xml",
                "status": "inactive",
                "_links": {
                    "activate": "https://api.planet.com/data/v1/assets/activate",
                    "_self": "https://api.planet.com/data/v1/assets/ortho_analytic_4b_xml",
                },
            },
            "ortho_visual": {
                "type": "ortho_visual",
                "status": "active",
                "_links": {
                    "download": "https://api.planet.com/data/v1/download/mock_visual.tif",
                    "thumbnail": "https://api.planet.com/data/v1/thumb/mock_thumb.jpg",
                    "_self": "https://api.planet.com/data/v1/assets/ortho_visual",
                },
            },
        }

    def mock_request(self, method: str, url: str, **kwargs) -> requests.Response:
        """Mock HTTP request to Planet API.

        Args:
            method: HTTP method
            url: Request URL
            **kwargs: Request parameters

        Returns:
            Mock response object
        """
        # Parse URL to determine endpoint
        parsed_url = urlparse(url)
        endpoint_type = self._classify_endpoint(url)

        # Simulate rate limiting
        if self._should_rate_limit(endpoint_type):
            return self._create_rate_limit_response()

        # Simulate random failures
        failure_response = self._check_failure_scenarios()
        if failure_response:
            return failure_response

        # Simulate response delay
        self._simulate_delay(endpoint_type)

        # Route to appropriate handler
        if "/quick-search" in parsed_url.path:
            return self._handle_search_request(method, url, **kwargs)
        elif "/stats" in parsed_url.path:
            return self._handle_stats_request(method, url, **kwargs)
        elif "/assets" in parsed_url.path:
            return self._handle_assets_request(method, url, **kwargs)
        elif "/item-types" in parsed_url.path and parsed_url.path.endswith(
            "/item-types"
        ):
            return self._handle_item_types_request(method, url, **kwargs)
        elif "/activate" in parsed_url.path:
            return self._handle_activate_request(method, url, **kwargs)
        elif "/download" in parsed_url.path:
            return self._handle_download_request(method, url, **kwargs)
        else:
            return self._create_not_found_response()

    def _classify_endpoint(self, url: str) -> str:
        """Classify endpoint type for rate limiting."""
        if any(path in url for path in ["/quick-search", "/searches", "/stats"]):
            return "search"
        elif "/assets" in url:
            return "assets"
        elif "/download" in url:
            return "download"
        else:
            return "general"

    def _should_rate_limit(self, endpoint_type: str) -> bool:
        """Check if request should be rate limited."""
        current_time = time.time()

        # Reset counters every minute
        if current_time - self.last_reset[endpoint_type] > 60:
            self.request_counts[endpoint_type] = 0
            self.last_reset[endpoint_type] = current_time

        # Check if over limit
        self.request_counts[endpoint_type] += 1
        return self.request_counts[endpoint_type] > self.rate_limits[endpoint_type]

    def _check_failure_scenarios(self) -> Optional[requests.Response]:
        """Check if request should fail based on configured scenarios."""
        # Rate limit failure
        if random.random() < self.failure_scenarios["rate_limit_probability"]:
            return self._create_rate_limit_response()

        # Server error failure
        if random.random() < self.failure_scenarios["server_error_probability"]:
            return self._create_server_error_response()

        # Network error (raises exception)
        if random.random() < self.failure_scenarios["network_error_probability"]:
            raise requests.exceptions.ConnectionError("Mock network error")

        return None

    def _simulate_delay(self, endpoint_type: str):
        """Simulate response delay."""
        if endpoint_type in self.response_delays:
            min_delay, max_delay = self.response_delays[endpoint_type]
            delay = random.uniform(min_delay, max_delay)
            time.sleep(delay)

    def _handle_search_request(
        self, method: str, url: str, **kwargs
    ) -> requests.Response:
        """Handle search request."""
        if method != "POST":
            return self._create_method_not_allowed_response()

        # Parse search request
        search_data = kwargs.get("json", {})
        search_filter = search_data.get("filter", {})
        item_types = search_data.get("item_types", ["PSScene"])

        # Filter scenes based on search criteria
        filtered_scenes = self._filter_scenes(search_filter)

        # Create response
        response_data = {"type": "FeatureCollection", "features": filtered_scenes}

        return self._create_json_response(200, response_data)

    def _handle_stats_request(
        self, method: str, url: str, **kwargs
    ) -> requests.Response:
        """Handle stats request."""
        if method != "POST":
            return self._create_method_not_allowed_response()

        # Generate mock stats based on search criteria
        stats_data = kwargs.get("json", {})
        interval = stats_data.get("interval", "month")

        # Generate monthly buckets
        buckets = []
        base_date = datetime(2024, 1, 1)

        for i in range(12):  # 12 months
            month_start = base_date.replace(month=i + 1)
            bucket = {
                "start_time": month_start.isoformat() + "Z",
                "count": random.randint(5, 25),
            }
            buckets.append(bucket)

        response_data = {"buckets": buckets, "interval": interval}

        return self._create_json_response(200, response_data)

    def _handle_assets_request(
        self, method: str, url: str, **kwargs
    ) -> requests.Response:
        """Handle assets request."""
        if method != "GET":
            return self._create_method_not_allowed_response()

        return self._create_json_response(200, self.sample_assets)

    def _handle_item_types_request(
        self, method: str, url: str, **kwargs
    ) -> requests.Response:
        """Handle item types request."""
        if method != "GET":
            return self._create_method_not_allowed_response()

        item_types = {
            "item_types": [
                {
                    "id": "PSScene",
                    "display_name": "PlanetScope Scene",
                    "display_description": "PlanetScope basic scene product",
                },
                {
                    "id": "REOrthoTile",
                    "display_name": "RapidEye Ortho Tile",
                    "display_description": "RapidEye orthorectified image",
                },
            ]
        }

        return self
