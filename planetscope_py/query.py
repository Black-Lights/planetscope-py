#!/usr/bin/env python3
"""Planet API query and scene discovery system.

This module implements comprehensive Planet Data API interaction capabilities
including scene search, filtering, preview handling, and batch operations.
Based on Planet's Data API v1 and following RASD specifications.
"""

import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple, Any
from urllib.parse import urlencode
from collections import defaultdict
import statistics
from datetime import datetime, timedelta, timezone
import requests
from shapely.geometry import shape, Point, Polygon
from shapely.validation import make_valid

from .auth import PlanetAuth
from .config import default_config
from .exceptions import (
    PlanetScopeError,
    APIError,
    ValidationError,
    RateLimitError,
    AssetError,
)
from .rate_limiter import RateLimiter
from .utils import (
    validate_geometry,
    calculate_area_km2,
    validate_date_range,  # <-- This was missing!
)

logger = logging.getLogger(__name__)


class PlanetScopeQuery:
    """Planet API query system for scene discovery and filtering.

    Implements comprehensive search capabilities with intelligent filtering,
    batch operations, and preview handling following Planet API patterns.

    Attributes:
        auth: PlanetAuth instance for API authentication
        rate_limiter: RateLimiter for API request management
        config: Configuration settings
        session: HTTP session for API requests
    """

    def __init__(self, api_key: Optional[str] = None, config: Optional[Dict] = None):
        """Initialize Planet query system.

        Args:
            api_key: Planet API key (optional, uses auth hierarchy)
            config: Custom configuration settings (optional)
        """
        self.auth = PlanetAuth(api_key)
        self.config = default_config
        if config:
            for key, value in config.items():
                self.config.set(key, value)

        self.session = self.auth.get_session()
        self.rate_limiter = RateLimiter(
            rates=self.config.rate_limits, session=self.session
        )

        # Search state management
        self._last_search_results = None
        self._last_search_stats = None

        logger.info("PlanetScopeQuery initialized successfully")

    def search_scenes(
        self,
        geometry: Union[Dict, Polygon, str],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        item_types: Optional[List[str]] = None,
        cloud_cover_max: float = 0.2,
        limit: Optional[int] = None,
        **kwargs,
    ) -> Dict:
        """Search for Planet scenes based on spatiotemporal criteria with proper API call handling and pagination.

        Executes a search request to Planet's Data API with comprehensive error handling,
        result processing, and automatic pagination to retrieve ALL matching scenes.
        Supports both dictionary and Shapely geometry inputs.

        Args:
            geometry (Union[Dict, Polygon, str]): Search area as GeoJSON dict, Shapely Polygon,
                                                or geometry string. Supports coordinate systems
                                                in WGS84 (EPSG:4326).
            start_date (Union[str, datetime]): Start date for temporal filtering.
                                            Accepts ISO format strings or datetime objects.
            end_date (Union[str, datetime]): End date for temporal filtering.
                                            Accepts ISO format strings or datetime objects.
            item_types (Optional[List[str]]): Planet item types to search. Defaults to ["PSScene"].
            cloud_cover_max (float): Maximum cloud cover threshold (0.0-1.0). Default: 0.2.
            limit (Optional[int]): Maximum number of scenes to return. If None, returns all available.
            **kwargs: Additional search parameters:
                    - sun_elevation_min (float): Minimum sun elevation in degrees
                    - ground_control (bool): Require ground control points
                    - quality_category (str): Required quality category

        Returns:
            Dict: Search results containing:
                - 'features': List of matching Planet scene features (ALL scenes, not limited to 250)
                - 'stats': Search statistics (total_scenes, cloud_cover_stats, etc.)
                - 'pagination': Pagination information (pages_fetched, total_scenes, etc.)
                - 'search_params': Original search parameters for reference

        Raises:
            ValidationError: Invalid geometry or search parameters
            APIError: Planet API communication errors or invalid responses
            RateLimitError: API rate limits exceeded

        Example:
            >>> results = query.search_scenes(
            ...     geometry=my_polygon,
            ...     start_date="2024-01-01",
            ...     end_date="2024-01-31",
            ...     cloud_cover_max=0.1
            ... )
            >>> print(f"Found {len(results['features'])} scenes")  # Can be > 250!
        """
        try:
            logger.info(
                f"Executing search for {len(item_types or ['PSScene'])} item types"
            )

            # Build search filter with proper geometry handling
            search_filter = self._build_search_filter(
                geometry=geometry,
                start_date=start_date,
                end_date=end_date,
                cloud_cover_max=cloud_cover_max,
                **kwargs,
            )

            # Use default item types if none provided
            if item_types is None:
                item_types = ["PSScene"]

            # Prepare search request
            search_request = {"item_types": item_types, "filter": search_filter}

            # Make API request with proper error handling
            base_url = getattr(
                self.config, "base_url", "https://api.planet.com/data/v1"
            )
            url = f"{base_url}/quick-search"

            # === PAGINATION IMPLEMENTATION ===
            all_features = []
            page_count = 0
            next_url = None
            total_scenes_found = 0

            logger.info("Starting paginated search...")

            while True:
                page_count += 1

                if next_url:
                    # Use next page URL from previous response
                    response = self.rate_limiter.make_request("GET", next_url)
                    logger.debug(f"Fetching page {page_count} from pagination link")
                else:
                    # First page - use POST request with search criteria
                    response = self.rate_limiter.make_request(
                        "POST", url, json=search_request
                    )
                    logger.debug(f"Executing initial search (page {page_count})")

                # Handle API response
                if response.status_code != 200:
                    raise APIError(
                        f"Search failed with status {response.status_code}: {response.text}"
                    )

                # Parse response
                page_results = response.json()
                page_features = page_results.get("features", [])

                logger.info(f"Page {page_count}: {len(page_features)} scenes")

                # Add features from this page
                all_features.extend(page_features)
                total_scenes_found += len(page_features)

                # Check for next page in Planet API response
                links = page_results.get("_links", {})
                next_url = links.get("_next")

                # Stop conditions
                if not next_url:
                    logger.info(f"No more pages available")
                    break

                if limit and len(all_features) >= limit:
                    logger.info(f"Reached user-specified limit of {limit} scenes")
                    all_features = all_features[:limit]  # Trim to exact limit
                    break

                if len(page_features) == 0:
                    logger.info(f"Empty page received, stopping pagination")
                    break

                # Safety: Prevent infinite loops (Planet API shouldn't do this, but just in case)
                if page_count >= 100:  # Max 100 pages = 25,000 scenes
                    logger.warning(
                        f"Reached maximum page limit (100 pages), stopping pagination"
                    )
                    break

            logger.info(
                f"Pagination complete: {len(all_features)} total scenes from {page_count} pages"
            )

            # Create combined results with all features
            search_results = {
                "features": all_features,
                "_pagination": {
                    "total_pages": page_count,
                    "total_features": len(all_features),
                    "pagination_used": page_count > 1,
                    "last_page_url": next_url,
                },
            }

            # Calculate search statistics on ALL results
            search_stats = self._calculate_search_stats(search_results)

            # Store results for later access
            self._last_search_results = search_results
            self._last_search_stats = search_stats

            logger.info(
                f"Search completed: {search_stats.get('total_scenes', 0)} scenes found across {page_count} pages"
            )

            # Return formatted results with pagination info
            return {
                "features": all_features,
                "stats": search_stats,
                "pagination": {
                    "pages_fetched": page_count,
                    "total_scenes": len(all_features),
                    "used_pagination": page_count > 1,
                    "hit_limit": bool(limit and len(all_features) >= limit),
                    "max_possible_scenes": "unlimited" if not limit else limit,
                },
                "search_params": {
                    "item_types": item_types,
                    "geometry": geometry,
                    "start_date": start_date,
                    "end_date": end_date,
                    "cloud_cover_max": cloud_cover_max,
                    "limit": limit,
                    **kwargs,
                },
            }

        except requests.exceptions.ConnectionError as e:
            raise APIError(f"Network error during search: {e}")
        except Exception as e:
            if isinstance(e, (ValidationError, APIError, RateLimitError)):
                raise
            raise APIError(f"Unexpected error during search: {e}")

    # Supporting helper function for calculating stats (if not already present)
    def _calculate_search_stats(self, search_results: Dict) -> Dict:
        """Calculate comprehensive statistics for search results."""
        features = search_results.get("features", [])

        if not features:
            return {
                "total_scenes": 0,
                "date_range": None,
                "cloud_cover_stats": None,
                "item_type_distribution": {},
                "coverage_area_km2": 0,
            }

        # Basic stats
        total_scenes = len(features)

        # Date range analysis
        acquisition_dates = []
        cloud_covers = []
        item_types = []

        for feature in features:
            props = feature.get("properties", {})

            # Collect acquisition dates
            acquired = props.get("acquired")
            if acquired:
                acquisition_dates.append(acquired)

            # Collect cloud cover values
            cloud_cover = props.get("cloud_cover")
            if cloud_cover is not None:
                cloud_covers.append(cloud_cover)

            # Collect item types
            item_type = props.get("item_type")
            if item_type:
                item_types.append(item_type)

        # Calculate date range
        date_range = None
        if acquisition_dates:
            date_range = {
                "start": min(acquisition_dates),
                "end": max(acquisition_dates),
                "span_days": (
                    pd.to_datetime(max(acquisition_dates))
                    - pd.to_datetime(min(acquisition_dates))
                ).days,
            }

        # Calculate cloud cover statistics
        cloud_cover_stats = None
        if cloud_covers:
            cloud_cover_stats = {
                "mean": float(np.mean(cloud_covers)),
                "min": float(np.min(cloud_covers)),
                "max": float(np.max(cloud_covers)),
                "std": float(np.std(cloud_covers)),
                "median": float(np.median(cloud_covers)),
            }

        # Item type distribution
        from collections import Counter

        item_type_distribution = dict(Counter(item_types))

        return {
            "total_scenes": total_scenes,
            "date_range": date_range,
            "cloud_cover_stats": cloud_cover_stats,
            "item_type_distribution": item_type_distribution,
            "acquisition_dates_available": len(acquisition_dates),
            "cloud_cover_values_available": len(cloud_covers),
        }

    def get_scene_stats(
        self,
        geometry: Union[Dict, Polygon, str],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        item_types: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict:
        """Get statistics for scenes matching search criteria.

        Args:
            geometry: Search area geometry
            start_date: Start date for temporal filter
            end_date: End date for temporal filter
            item_types: Planet item types to analyze
            **kwargs: Additional search filters

        Returns:
            Dictionary containing detailed scene statistics
        """
        try:
            # Build search filter
            search_filter = self._build_search_filter(
                geometry=geometry, start_date=start_date, end_date=end_date, **kwargs
            )

            if item_types is None:
                item_types = self.config.item_types

            # Prepare stats request
            stats_request = {
                "item_types": item_types,
                "interval": "month",  # Monthly aggregation
                "filter": search_filter,
            }

            # Execute stats request
            endpoint = f"{self.config.base_url}/stats"

            response = self.rate_limiter.make_request(
                method="POST",
                url=endpoint,
                json=stats_request,
                timeout=self.config.timeouts["read"],
            )

            if response.status_code != 200:
                raise APIError(
                    f"Stats request failed with status {response.status_code}",
                    details={"response": response.text},
                )

            stats_data = response.json()

            # Process and enhance stats
            processed_stats = self._process_stats_response(stats_data)

            logger.info("Scene statistics calculated successfully")

            return processed_stats

        except Exception as e:
            if isinstance(e, PlanetScopeError):
                raise
            raise APIError(f"Error calculating scene stats: {str(e)}")

    def filter_scenes_by_quality(
        self,
        scenes: List[Dict],
        min_quality: float = 0.7,
        max_cloud_cover: float = 0.2,
        exclude_night: bool = True,
    ) -> List[Dict]:
        """Filter scenes based on quality criteria.

        Args:
            scenes: List of scene features from search results
            min_quality: Minimum quality score (0.0-1.0)
            max_cloud_cover: Maximum cloud cover (0.0-1.0)
            exclude_night: Exclude nighttime acquisitions

        Returns:
            Filtered list of scene features
        """
        filtered_scenes = []

        for scene in scenes:
            properties = scene.get("properties", {})

            # Check cloud cover
            cloud_cover = properties.get("cloud_cover", 1.0)
            if cloud_cover > max_cloud_cover:
                continue

            # Check quality score if available
            quality_category = properties.get("quality_category", "standard")
            if quality_category == "test":  # Exclude test data
                continue

            # Check for nighttime imagery
            if exclude_night:
                sun_elevation = properties.get("sun_elevation")
                if sun_elevation is not None and sun_elevation < 10:
                    continue

            # Check usable data percentage
            usable_data = properties.get("usable_data", 0.0)
            if usable_data < min_quality:
                continue

            filtered_scenes.append(scene)

        logger.info(
            f"Quality filtering: {len(filtered_scenes)}/{len(scenes)} scenes passed"
        )

        return filtered_scenes

    def get_scene_previews(self, scene_ids: List[str]) -> Dict[str, str]:
        """Get preview URLs for specified scenes.

        Args:
            scene_ids: List of Planet scene IDs

        Returns:
            Dictionary mapping scene IDs to preview URLs
        """
        preview_urls = {}

        for scene_id in scene_ids:
            try:
                # Get assets for scene
                assets_url = (
                    f"{self.config.base_url}/item-types/PSScene/items/{scene_id}/assets"
                )

                response = self.rate_limiter.make_request(
                    method="GET", url=assets_url, timeout=self.config.timeouts["read"]
                )

                if response.status_code == 200:
                    assets = response.json()

                    # Look for visual asset with preview
                    for asset_type, asset_info in assets.items():
                        if "visual" in asset_type.lower():
                            preview_link = asset_info.get("_links", {}).get("thumbnail")
                            if preview_link:
                                preview_urls[scene_id] = preview_link
                                break

            except Exception as e:
                logger.warning(f"Failed to get preview for scene {scene_id}: {e}")
                continue

        return preview_urls

    def batch_search(
        self,
        geometries: List[Union[Dict, Polygon]],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        **kwargs,
    ) -> List[Dict]:
        """Execute batch search across multiple geometries.

        Args:
            geometries: List of search areas
            start_date: Start date for temporal filter
            end_date: End date for temporal filter
            **kwargs: Additional search parameters

        Returns:
            List of search results for each geometry
        """
        batch_results = []

        for i, geometry in enumerate(geometries):
            try:
                logger.info(f"Processing batch search {i+1}/{len(geometries)}")

                result = self.search_scenes(
                    geometry=geometry,
                    start_date=start_date,
                    end_date=end_date,
                    **kwargs,
                )

                batch_results.append(
                    {"geometry_index": i, "result": result, "success": True}
                )

                # Add small delay between requests
                time.sleep(0.1)

            except Exception as e:
                logger.error(f"Batch search failed for geometry {i}: {e}")
                batch_results.append(
                    {"geometry_index": i, "error": str(e), "success": False}
                )

        return batch_results

    def _build_search_filter(
        self, geometry, start_date, end_date, cloud_cover_max=0.2, **kwargs
    ):
        """Build Planet API search filter with proper Planet API date formatting.

        Constructs a Planet API compatible filter object ensuring all dates are formatted
        exactly as Planet API expects: YYYY-MM-DDTHH:MM:SS.ffffffZ (6-digit microseconds + Z).

        Args:
            geometry (Union[Dict, Polygon]): Search area geometry
            start_date (Union[str, datetime]): Start date for temporal filter
            end_date (Union[str, datetime]): End date for temporal filter
            cloud_cover_max (float): Maximum cloud cover threshold (0.0-1.0)
            **kwargs: Additional filter parameters

        Returns:
            Dict: Planet API filter object with properly formatted dates

        Raises:
            ValidationError: Invalid geometry, dates, or area constraints

        Example:
            >>> # All these date inputs will work:
            >>> filter_obj = query._build_search_filter(
            ...     geometry=polygon,
            ...     start_date="2024-01-01",                    # Simple date string
            ...     end_date=datetime(2024, 1, 31),             # Datetime object
            ... )
            >>> # Dates become: "2024-01-01T00:00:00.000000Z" and "2024-01-31T23:59:59.999999Z"
        """

        # Handle Shapely geometry objects FIRST
        if hasattr(geometry, "__geo_interface__"):
            geom_dict = geometry.__geo_interface__
        elif isinstance(geometry, dict):
            geom_dict = geometry
        else:
            raise ValidationError(
                f"Geometry must be a dictionary or Shapely object. "
                f"Details: {{'geometry': {geometry}, 'type': '{type(geometry).__name__}'}}"
            )

        # Validate the geometry dict using utils function
        geom_dict = validate_geometry(geom_dict)

        # Validate area constraints
        area_km2 = calculate_area_km2(geom_dict)
        max_area = getattr(self.config, "max_roi_area_km2", 10000)
        if area_km2 > max_area:
            raise ValidationError(
                f"Search area ({area_km2:.1f} km²) exceeds maximum allowed ({max_area} km²)"
            )

        # **CRITICAL FIX**: Use validate_date_range from utils.py
        # This ensures consistent date formatting throughout the library
        try:
            start_date_formatted, end_date_formatted = validate_date_range(
                start_date, end_date
            )
            logger.debug(
                f"Formatted dates - Start: {start_date_formatted}, End: {end_date_formatted}"
            )
        except Exception as e:
            raise ValidationError(f"Date formatting failed: {str(e)}")

        # Build filter components list
        filter_components = []

        # 1. Geometry filter (required)
        filter_components.append(
            {"type": "GeometryFilter", "field_name": "geometry", "config": geom_dict}
        )

        # 2. Date range filter (required) - Using properly formatted dates from utils.py
        filter_components.append(
            {
                "type": "DateRangeFilter",
                "field_name": "acquired",
                "config": {
                    "gte": start_date_formatted,  # Now properly formatted with end-of-day logic
                    "lte": end_date_formatted,  # Now properly formatted with end-of-day logic
                },
            }
        )

        # 3. Cloud cover filter (optional)
        if cloud_cover_max is not None:
            filter_components.append(
                {
                    "type": "RangeFilter",
                    "field_name": "cloud_cover",
                    "config": {"lte": cloud_cover_max},
                }
            )

        # 4. Additional filters from kwargs
        for field_name, value in kwargs.items():
            if field_name.endswith("_min"):
                # Minimum value filters (e.g., sun_elevation_min)
                actual_field = field_name.replace("_min", "")
                filter_components.append(
                    {
                        "type": "RangeFilter",
                        "field_name": actual_field,
                        "config": {"gte": value},
                    }
                )
            elif field_name.endswith("_max"):
                # Maximum value filters (e.g., view_angle_max)
                actual_field = field_name.replace("_max", "")
                filter_components.append(
                    {
                        "type": "RangeFilter",
                        "field_name": actual_field,
                        "config": {"lte": value},
                    }
                )
            elif isinstance(value, bool):
                # Boolean filters (e.g., ground_control)
                filter_components.append(
                    {
                        "type": "StringInFilter",
                        "field_name": field_name,
                        "config": [value],
                    }
                )
            elif isinstance(value, (list, tuple)):
                # List-based filters
                filter_components.append(
                    {
                        "type": "StringInFilter",
                        "field_name": field_name,
                        "config": list(value),
                    }
                )

        # Combine all filters with AndFilter
        search_filter = {"type": "AndFilter", "config": filter_components}

        logger.debug(f"Built search filter with {len(filter_components)} components")

        return search_filter

    def _calculate_search_stats(self, search_response):
        """Calculate comprehensive statistics from Planet API search response.

        Processes search results to extract statistical information about found scenes
        including cloud cover distribution, temporal patterns, and satellite coverage.

        Args:
            search_response (Dict): Raw search response from Planet API containing:
                                - 'features': List of scene feature objects
                                - Each feature has 'properties' with metadata

        Returns:
            Dict: Comprehensive statistics containing:
                - 'total_scenes': Total number of scenes found
                - 'cloud_cover_stats': Cloud cover distribution (min, max, mean, count)
                - 'acquisition_dates': Sorted list of acquisition dates
                - 'item_types': Count of each item type found
                - 'satellites': Count of scenes per satellite

        Example:
            >>> response = {"features": [{"properties": {"cloud_cover": 0.1, ...}}, ...]}
            >>> stats = query._calculate_search_stats(response)
            >>> print(f"Found {stats['total_scenes']} scenes")
            >>> print(f"Cloud cover range: {stats['cloud_cover_stats']['min']}-{stats['cloud_cover_stats']['max']}")

        Note:
            - Handles missing or None values gracefully
            - Returns empty structures for empty search results
            - Calculates statistics only from valid non-None values
            - Dates are sorted chronologically for temporal analysis
        """
        features = search_response.get("features", [])

        if not features:
            return {
                "total_scenes": 0,
                "cloud_cover_stats": {},
                "acquisition_dates": [],
                "item_types": {},
                "satellites": {},
            }

        # Extract statistics
        cloud_covers = []
        acquisition_dates = []
        item_types = defaultdict(int)
        satellites = defaultdict(int)

        for feature in features:
            props = feature.get("properties", {})

            # Cloud cover
            cc = props.get("cloud_cover")
            if cc is not None:
                cloud_covers.append(cc)

            # Acquisition date
            acquired = props.get("acquired")
            if acquired:
                acquisition_dates.append(acquired)

            # Item type
            item_type = props.get("item_type")
            if item_type:
                item_types[item_type] += 1

            # Satellite
            satellite = props.get("satellite_id")
            if satellite:
                satellites[satellite] += 1

        # Calculate cloud cover statistics
        cloud_cover_stats = {}
        if cloud_covers:
            cloud_cover_stats = {
                "min": min(cloud_covers),
                "max": max(cloud_covers),
                "mean": sum(cloud_covers) / len(cloud_covers),
                "count": len(cloud_covers),
            }

        return {
            "total_scenes": len(features),
            "cloud_cover_stats": cloud_cover_stats,
            "acquisition_dates": sorted(acquisition_dates),
            "item_types": dict(item_types),
            "satellites": dict(satellites),
        }

    def _process_stats_response(self, stats_data: Dict) -> Dict:
        """Process and enhance Planet API stats response.

        Args:
            stats_data: Raw stats response from Planet API

        Returns:
            Processed statistics dictionary
        """
        processed = {
            "buckets": stats_data.get("buckets", []),
            "interval": stats_data.get("interval", "month"),
            "total_scenes": 0,
            "temporal_distribution": {},
        }

        # Calculate totals and temporal distribution
        for bucket in processed["buckets"]:
            count = bucket.get("count", 0)
            processed["total_scenes"] += count

            start_time = bucket.get("start_time")
            if start_time:
                processed["temporal_distribution"][start_time] = count

        return processed
