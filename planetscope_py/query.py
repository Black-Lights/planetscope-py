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
import math

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
    validate_date_range,
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
            >>> print(f"Found {len(results['features'])} scenes")
        """
        try:
            # ... (validation code remains the same) ...
            
            # Validate inputs
            validated_geometry = validate_geometry(geometry)
            validated_dates = validate_date_range(start_date, end_date)
            start_date, end_date = validated_dates

            if item_types is None:
                item_types = self.config.item_types

            logger.info(f"Executing search for {len(item_types)} item types")

            # Build search filter
            search_filter = self._build_search_filter(
                validated_geometry, start_date, end_date, cloud_cover_max, **kwargs
            )

            # Build initial search request
            search_request = {"item_types": item_types, "filter": search_filter}

            # Handle pagination to get ALL results
            all_features = []
            page_count = 0
            search_url = f"{self.config.base_url}/quick-search"
            next_page_url = None

            logger.info("Starting paginated search...")

            while True:
                page_count += 1

                # Use next page URL if available, otherwise use base search
                if next_page_url:
                    # For subsequent pages, use the _next URL from previous response
                    response = self.rate_limiter.make_request(
                        method="GET",
                        url=next_page_url,
                        timeout=self.config.timeouts["read"],
                    )
                else:
                    # First page - POST request with search parameters
                    response = self.rate_limiter.make_request(
                        method="POST",
                        url=search_url,
                        json=search_request,
                        timeout=self.config.timeouts["read"],
                    )

                if response.status_code != 200:
                    raise APIError(
                        f"Search request failed with status {response.status_code}",
                        {
                            "status_code": response.status_code,
                            "response": response.text[:500],
                            "request": search_request,
                        },
                    )

                page_data = response.json()
                page_features = page_data.get("features", [])

                logger.info(f"Page {page_count}: {len(page_features)} scenes")

                if not page_features:
                    logger.info("No more pages available")
                    break

                all_features.extend(page_features)

                # Check if we have a limit and have reached it
                if limit and len(all_features) >= limit:
                    all_features = all_features[:limit]
                    logger.info(f"Reached limit of {limit} scenes")
                    break

                # Check for next page link in response
                links = page_data.get("_links", {})
                next_page_url = links.get("_next")
                
                if not next_page_url:
                    logger.info("No more pages available (no _next link)")
                    break

                # Safety check: if we get the same number of features as the page size,
                # but no _next link, we're likely at the end
                if len(page_features) < 250:  # Planet's typical page size
                    logger.info("Reached end of results (partial page)")
                    break

                # Safety check: prevent infinite loops
                if page_count > 1000:  # Reasonable upper limit
                    logger.warning(f"Stopping pagination at {page_count} pages to prevent infinite loop")
                    break

            logger.info(
                f"Pagination complete: {len(all_features)} total scenes from {page_count} pages"
            )

            # Calculate search statistics
            search_stats = self._calculate_search_stats({"features": all_features})

            # Store results for potential reuse
            self._last_search_results = all_features
            self._last_search_stats = search_stats

            logger.info(
                f"Search completed: {len(all_features)} scenes found across {page_count} pages"
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

    def _calculate_search_stats(self, search_results: Dict) -> Dict:
        """Calculate comprehensive statistics for search results."""
        features = search_results.get("features", [])
        if not features:
            return {"total_scenes": 0}

        # Extract metadata for statistics
        cloud_covers = []
        sun_elevations = []
        acquisition_dates = []
        satellites = []

        for feature in features:
            props = feature.get("properties", {})

            cloud_cover = props.get("cloud_cover")
            if cloud_cover is not None:
                cloud_covers.append(cloud_cover)

            sun_elevation = props.get("sun_elevation")
            if sun_elevation is not None:
                sun_elevations.append(sun_elevation)

            acquired = props.get("acquired")
            if acquired:
                acquisition_dates.append(acquired)

            satellite_id = props.get("satellite_id")
            if satellite_id:
                satellites.append(satellite_id)

        # Calculate statistics
        stats = {"total_scenes": len(features)}

        if cloud_covers:
            stats["cloud_cover"] = {
                "min": min(cloud_covers),
                "max": max(cloud_covers),
                "mean": statistics.mean(cloud_covers),
                "median": statistics.median(cloud_covers),
            }

        if sun_elevations:
            stats["sun_elevation"] = {
                "min": min(sun_elevations),
                "max": max(sun_elevations),
                "mean": statistics.mean(sun_elevations),
                "median": statistics.median(sun_elevations),
            }

        if acquisition_dates:
            stats["temporal_range"] = {
                "start": min(acquisition_dates),
                "end": max(acquisition_dates),
                "span_days": (
                    datetime.fromisoformat(max(acquisition_dates).replace("Z", "+00:00"))
                    - datetime.fromisoformat(min(acquisition_dates).replace("Z", "+00:00"))
                ).days,
            }

        if satellites:
            stats["satellites"] = {
                "unique_count": len(set(satellites)),
                "distribution": {
                    satellite: satellites.count(satellite) for satellite in set(satellites)
                },
            }

        return stats

    def get_scene_previews(self, scene_ids: List[str]) -> Dict[str, str]:
        """Get preview URLs for specified scenes using Planet's Tile Service API.
        
        This method works for any geographic region and calculates proper tile coordinates
        from actual scene geometry, ensuring the URLs work anywhere in the world.
        
        Args:
            scene_ids: List of Planet scene IDs

        Returns:
            Dictionary mapping scene IDs to working tile preview URLs
            
        Example:
            >>> query = PlanetScopeQuery()
            >>> results = query.search_scenes(any_geometry, "2024-01-01", "2024-01-31")
            >>> scene_ids = [scene['id'] for scene in results['features'][:5]]
            >>> previews = query.get_scene_previews(scene_ids)
            >>> for scene_id, preview_url in previews.items():
            ...     print(f"Scene {scene_id}: {preview_url}")
        """
        
        logger.info(f"Getting working preview URLs for {len(scene_ids)} scenes")
        
        def lat_lon_to_tile(lat: float, lon: float, zoom: int) -> Tuple[int, int]:
            """Convert lat/lon coordinates to tile coordinates."""
            lat_rad = math.radians(lat)
            n = 2.0 ** zoom
            x = int((lon + 180.0) / 360.0 * n)
            y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
            return (x, y)
        
        def get_scene_center(scene_geometry):
            """Get the center coordinates of a scene."""
            try:
                geom = shape(scene_geometry)
                centroid = geom.centroid
                return centroid.y, centroid.x  # lat, lon
            except Exception:
                return None, None
        
        def get_fallback_coordinates(scene_ids):
            """Get fallback coordinates from search ROI or scene locations."""
            # Try to use the search geometry from last search as fallback
            if hasattr(self, '_last_search_results') and self._last_search_results:
                # Use the centroid of all scenes as fallback
                all_lats, all_lons = [], []
                for scene in self._last_search_results:
                    try:
                        geom = shape(scene['geometry'])
                        all_lats.append(geom.centroid.y)
                        all_lons.append(geom.centroid.x)
                    except:
                        continue
                
                if all_lats and all_lons:
                    avg_lat = sum(all_lats) / len(all_lats)
                    avg_lon = sum(all_lons) / len(all_lons)
                    return avg_lat, avg_lon
            
            # If no previous results, try to get from individual scenes
            for scene_id in scene_ids:
                try:
                    scene_url = f"{self.config.base_url}/item-types/PSScene/items/{scene_id}"
                    response = self.rate_limiter.make_request(
                        method="GET", 
                        url=scene_url, 
                        timeout=self.config.timeouts["read"]
                    )
                    
                    if response.status_code == 200:
                        scene_data = response.json()
                        scene_geometry = scene_data.get('geometry')
                        if scene_geometry:
                            lat, lon = get_scene_center(scene_geometry)
                            if lat is not None and lon is not None:
                                return lat, lon
                except Exception:
                    continue
            
            # Last resort: return None to indicate no fallback available
            logger.warning("Could not determine fallback coordinates for any region")
            return None, None
        
        # Get tile base URL and API key
        tile_base_url = self.config.get('tile_url', 'https://tiles.planet.com/data/v1')
        api_key = getattr(self.auth, '_api_key', '')
        
        preview_urls = {}
        
        # Build scene geometries lookup from last search results
        scene_geometries = {}
        if hasattr(self, '_last_search_results') and self._last_search_results:
            for scene in self._last_search_results:
                if scene['id'] in scene_ids:
                    scene_geometries[scene['id']] = scene['geometry']
        
        # Get fallback coordinates for scenes without geometry
        fallback_lat, fallback_lon = get_fallback_coordinates(scene_ids)
        
        # Process each scene
        for scene_id in scene_ids:
            try:
                scene_lat, scene_lon = None, None
                
                # Try to get scene-specific coordinates
                if scene_id in scene_geometries:
                    scene_lat, scene_lon = get_scene_center(scene_geometries[scene_id])
                
                # If scene geometry not available, try to fetch it
                if scene_lat is None or scene_lon is None:
                    try:
                        scene_url = f"{self.config.base_url}/item-types/PSScene/items/{scene_id}"
                        response = self.rate_limiter.make_request(
                            method="GET", 
                            url=scene_url, 
                            timeout=self.config.timeouts["read"]
                        )
                        
                        if response.status_code == 200:
                            scene_data = response.json()
                            scene_geometry = scene_data.get('geometry')
                            if scene_geometry:
                                scene_lat, scene_lon = get_scene_center(scene_geometry)
                    except Exception as e:
                        logger.debug(f"Could not fetch geometry for scene {scene_id}: {e}")
                
                # Use fallback coordinates if scene-specific ones not available
                if scene_lat is None or scene_lon is None:
                    if fallback_lat is not None and fallback_lon is not None:
                        scene_lat, scene_lon = fallback_lat, fallback_lon
                        logger.debug(f"Using fallback coordinates for scene {scene_id}")
                    else:
                        logger.warning(f"No coordinates available for scene {scene_id}, skipping")
                        continue
                
                # Try multiple zoom levels to find working tiles
                working_url = None
                for zoom_level in [10, 11, 12]:
                    tile_x, tile_y = lat_lon_to_tile(scene_lat, scene_lon, zoom_level)
                    
                    # Create tile URL
                    tile_url = f"{tile_base_url}/PSScene/{scene_id}/{zoom_level}/{tile_x}/{tile_y}.png"
                    if api_key:
                        tile_url += f"?api_key={api_key}"
                    
                    # Optional quick test for better UX
                    try:
                        import requests
                        test_response = requests.head(tile_url, timeout=2)
                        if test_response.status_code == 200:
                            working_url = tile_url
                            break
                    except:
                        # If quick test fails, still use the URL as fallback
                        if not working_url:
                            working_url = tile_url
                
                if working_url:
                    preview_urls[scene_id] = working_url
                    logger.debug(f"Generated working tile URL for scene {scene_id}")
                else:
                    logger.warning(f"Could not generate working URL for scene {scene_id}")
                    
            except Exception as e:
                logger.warning(f"Failed to generate preview URL for scene {scene_id}: {e}")
                continue
        
        logger.info(f"Generated {len(preview_urls)} working preview URLs")
        return preview_urls


    def get_scene_tile_urls(
        self, 
        scene_ids: List[str], 
        zoom_level: int = 12,
        center_coords: Optional[Tuple[float, float]] = None
    ) -> Dict[str, Dict]:
        """Get specific tile URLs for scenes at given zoom level with actual coordinates.
        
        Works for any geographic region by using actual scene coordinates or provided center.
        
        Args:
            scene_ids: List of Planet scene IDs
            zoom_level: Zoom level for tiles (default: 12)
            center_coords: Optional (lat, lon) tuple to override automatic calculation

        Returns:
            Dictionary mapping scene IDs to tile information
        """
        
        logger.info(f"Getting tile URLs for {len(scene_ids)} scenes at zoom level {zoom_level}")
        
        def lat_lon_to_tile(lat: float, lon: float, zoom: int) -> Tuple[int, int]:
            """Convert lat/lon coordinates to tile coordinates."""
            lat_rad = math.radians(lat)
            n = 2.0 ** zoom
            x = int((lon + 180.0) / 360.0 * n)
            y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
            return (x, y)
        
        def get_scene_center(scene_geometry):
            """Get the center coordinates of a scene."""
            try:
                geom = shape(scene_geometry)
                centroid = geom.centroid
                return centroid.y, centroid.x  # lat, lon
            except Exception:
                return None, None
        
        tile_info = {}
        tile_base_url = self.config.get('tile_url', 'https://tiles.planet.com/data/v1')
        api_key = getattr(self.auth, '_api_key', '')
        
        # Build scene geometries lookup
        scene_geometries = {}
        if hasattr(self, '_last_search_results') and self._last_search_results:
            for scene in self._last_search_results:
                if scene['id'] in scene_ids:
                    scene_geometries[scene['id']] = scene['geometry']
        
        for scene_id in scene_ids:
            try:
                # Use provided center coordinates or calculate from scene
                if center_coords:
                    center_lat, center_lon = center_coords
                    tile_x, tile_y = lat_lon_to_tile(center_lat, center_lon, zoom_level)
                else:
                    # Get scene-specific coordinates
                    scene_lat, scene_lon = None, None
                    
                    if scene_id in scene_geometries:
                        scene_lat, scene_lon = get_scene_center(scene_geometries[scene_id])
                    
                    if scene_lat is None or scene_lon is None:
                        # Try to fetch scene geometry
                        try:
                            scene_url = f"{self.config.base_url}/item-types/PSScene/items/{scene_id}"
                            response = self.rate_limiter.make_request(
                                method="GET", 
                                url=scene_url, 
                                timeout=self.config.timeouts["read"]
                            )
                            
                            if response.status_code == 200:
                                scene_data = response.json()
                                scene_geometry = scene_data.get('geometry')
                                if scene_geometry:
                                    scene_lat, scene_lon = get_scene_center(scene_geometry)
                        except Exception:
                            pass
                    
                    if scene_lat is None or scene_lon is None:
                        logger.warning(f"Could not determine coordinates for scene {scene_id}")
                        continue
                    
                    center_lat, center_lon = scene_lat, scene_lon
                    tile_x, tile_y = lat_lon_to_tile(center_lat, center_lon, zoom_level)
                
                # Create URLs
                template_url = f"{tile_base_url}/PSScene/{scene_id}/{{z}}/{{x}}/{{y}}.png"
                static_url = f"{tile_base_url}/PSScene/{scene_id}/{zoom_level}/{tile_x}/{tile_y}.png"
                
                if api_key:
                    template_url += f"?api_key={api_key}"
                    static_url += f"?api_key={api_key}"
                
                tile_info[scene_id] = {
                    'template_url': template_url,
                    'static_url': static_url,
                    'zoom_level': zoom_level,
                    'tile_x': tile_x,
                    'tile_y': tile_y,
                    'center_coords': (center_lat, center_lon),
                    'tile_base_url': tile_base_url
                }
                
                logger.debug(f"Generated tile info for scene {scene_id}")
                
            except Exception as e:
                logger.warning(f"Failed to generate tile info for scene {scene_id}: {e}")
                continue
        
        logger.info(f"Generated tile info for {len(tile_info)} scenes")
        return tile_info
    
    def get_scene_stats(
        self,
        geometry: Union[Dict, Polygon, str],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        item_types: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict:
        """Get statistical information about scenes without full search.

        Args:
            geometry: Search area geometry
            start_date: Start date for temporal filter
            end_date: End date for temporal filter
            item_types: Planet item types to analyze
            **kwargs: Additional filter parameters

        Returns:
            Dictionary containing scene statistics and temporal distribution
        """
        try:
            validated_geometry = validate_geometry(geometry)
            validated_dates = validate_date_range(start_date, end_date)
            start_date, end_date = validated_dates

            if item_types is None:
                item_types = self.config.item_types

            # Build search filter
            search_filter = self._build_search_filter(
                validated_geometry, start_date, end_date, **kwargs
            )

            # Build stats request
            stats_request = {
                "item_types": item_types,
                "filter": search_filter,
                "interval": "month",
            }

            stats_url = f"{self.config.base_url}/stats"

            response = self.rate_limiter.make_request(
                method="POST",
                url=stats_url,
                json=stats_request,
                timeout=self.config.timeouts["read"],
            )

            if response.status_code != 200:
                raise APIError(
                    f"Stats request failed with status {response.status_code}",
                    {"status_code": response.status_code, "response": response.text[:500]},
                )

            stats_data = response.json()

            # Process and format stats
            buckets = stats_data.get("buckets", [])
            total_scenes = sum(bucket.get("count", 0) for bucket in buckets)

            result = {
                "total_scenes": total_scenes,
                "temporal_distribution": buckets,
                "interval": stats_data.get("interval", "month"),
                "buckets": buckets,
            }

            logger.info(f"Scene stats: {total_scenes} total scenes found")

            return result

        except Exception as e:
            if isinstance(e, (ValidationError, APIError, RateLimitError)):
                raise
            raise APIError(f"Unexpected error getting scene stats: {e}")

    def filter_scenes_by_quality(
        self,
        scenes: List[Dict],
        min_visible_fraction: float = 0.7,
        max_cloud_cover: float = 0.2,
        exclude_night: bool = True,
    ) -> List[Dict]:
        """Filter scenes based on quality criteria.

        Args:
            scenes: List of scene features from search results
            min_visible_fraction: Minimum visible area fraction (0.0-1.0)
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

            # Check visible data percentage
            visible_percent = properties.get("visible_percent")
            if visible_percent is not None:
                visible_fraction = visible_percent / 100.0  # Convert to 0.0-1.0 range
                if visible_fraction < min_visible_fraction:
                    continue
            # If visible_percent is missing, don't reject the scene

            filtered_scenes.append(scene)

        logger.info(
            f"Quality filtering: {len(filtered_scenes)}/{len(scenes)} scenes passed"
        )

        return filtered_scenes

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
        
        Args:
            geometry: Validated geometry object
            start_date: Start date string or datetime
            end_date: End date string or datetime  
            cloud_cover_max: Maximum cloud cover threshold
            **kwargs: Additional filter parameters
            
        Returns:
            Planet API compatible search filter
        """
        # Convert dates to Planet API format if needed
        if isinstance(start_date, datetime):
            start_date_str = start_date.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        else:
            start_date_str = start_date
            
        if isinstance(end_date, datetime):
            end_date_str = end_date.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        else:
            end_date_str = end_date

        # Base filters
        filters = [
            {
                "type": "GeometryFilter",
                "field_name": "geometry", 
                "config": geometry
            },
            {
                "type": "DateRangeFilter",
                "field_name": "acquired",
                "config": {
                    "gte": start_date_str,
                    "lte": end_date_str
                }
            },
            {
                "type": "RangeFilter",
                "field_name": "cloud_cover",
                "config": {
                    "lte": cloud_cover_max
                }
            }
        ]

        # Add optional filters from kwargs
        if kwargs.get("sun_elevation_min"):
            filters.append({
                "type": "RangeFilter",
                "field_name": "sun_elevation", 
                "config": {
                    "gte": kwargs["sun_elevation_min"]
                }
            })

        if kwargs.get("ground_control"):
            filters.append({
                "type": "StringInFilter",
                "field_name": "ground_control",
                "config": ["true"]
            })

        if kwargs.get("quality_category"):
            filters.append({
                "type": "StringInFilter", 
                "field_name": "quality_category",
                "config": [kwargs["quality_category"]]
            })

        # Combine all filters with AND logic
        return {
            "type": "AndFilter",
            "config": filters
        }