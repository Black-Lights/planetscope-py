#!/usr/bin/env python3
"""Preview management for PlanetScope scenes.

This module provides advanced preview capabilities using Planet's Tile Service API,
including interactive map generation and static preview creation.

Author: Ammar & Umayr
Version: 4.1.0 (Enhanced + Metadata Fixes + JSON Serialization)
"""

import logging
import math
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any

# Add Shapely imports
try:
    from shapely.geometry import Polygon, shape
    _SHAPELY_AVAILABLE = True
except ImportError:
    _SHAPELY_AVAILABLE = False
    # Create dummy classes to avoid NameError
    class Polygon:
        pass

# Add other optional imports
try:
    import geopandas as gpd
    _GEOPANDAS_AVAILABLE = True
except ImportError:
    _GEOPANDAS_AVAILABLE = False

logger = logging.getLogger(__name__)


class PreviewManager:
    """Manages scene previews using Planet's Tile Service API.
    
    Provides methods for generating tile URLs, creating interactive maps,
    and managing preview visualizations following Planet's official approach.
    """
    
    def __init__(self, query_instance):
        """Initialize preview manager.
        
        Args:
            query_instance: PlanetScopeQuery instance for API access
        """
        self.query = query_instance
        self.config = query_instance.config
        self.auth = query_instance.auth
        
        # Planet tile service configuration
        self.tile_base_url = self.config.get('tile_url', 'https://tiles.planet.com/data/v1')
        
        logger.info("PreviewManager initialized")
    
    def generate_tile_urls(self, scene_ids: List[str]) -> Dict[str, str]:
        """Generate tile URLs for scenes using Planet's Tile Service API.
        
        Args:
            scene_ids: List of Planet scene IDs
            
        Returns:
            Dictionary mapping scene IDs to tile template URLs
        """
        tile_urls = {}
        api_key = getattr(self.auth, '_api_key', '')
        
        for scene_id in scene_ids:
            template_url = f"{self.tile_base_url}/PSScene/{scene_id}/{{z}}/{{x}}/{{y}}.png"
            if api_key:
                template_url += f"?api_key={api_key}"
            
            tile_urls[scene_id] = template_url
        
        logger.info(f"Generated {len(tile_urls)} tile URLs")
        return tile_urls
    
    def create_interactive_map(self, 
                            search_results: Dict,
                            roi_geometry: Optional[Any] = None,
                            max_scenes: int = 10) -> Optional[Any]:
        """Create interactive Folium map with Planet tile layers and improved centering."""
        try:
            import folium
            from shapely.geometry import shape
        except ImportError:
            logger.error("Folium and/or Shapely not available. Install with: pip install folium shapely")
            return None
        
        # Calculate map center with improved logic
        map_center = self._calculate_map_center(search_results, roi_geometry)
        
        # Calculate appropriate zoom level based on ROI size
        zoom_level = self._calculate_zoom_level(roi_geometry, search_results)
        
        # Create base map with calculated center and zoom
        folium_map = folium.Map(
            location=map_center, 
            zoom_start=zoom_level,
            tiles='OpenStreetMap'
        )
        
        # Add ROI if provided (this should be added first to ensure it's visible)
        if roi_geometry:
            self._add_roi_to_map(folium_map, roi_geometry)
        
        # Add scene tile layers
        scenes_added = self._add_scene_tiles_to_map(
            folium_map, 
            search_results['features'][:max_scenes]
        )
        
        # Add layer control
        folium.LayerControl().add_to(folium_map)
        
        logger.info(f"Created interactive map centered at {map_center} with zoom {zoom_level}")
        logger.info(f"Added {scenes_added} scene layers")
        return folium_map
    
    def save_interactive_map(self, 
                           folium_map: Any, 
                           filename: str = "planet_preview_map.html") -> str:
        """Save interactive map to HTML file.
        
        Args:
            folium_map: Folium map object
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        try:
            folium_map.save(filename)
            logger.info(f"Interactive map saved to {filename}")
            
            # Security warning
            logger.warning(f"Saved HTML file contains API key in tile URLs. Share carefully.")
            
            return filename
        except Exception as e:
            logger.error(f"Failed to save map: {e}")
            raise
    
    def get_static_tile_urls(self, 
                           scene_ids: List[str], 
                           zoom_level: int = 12,
                           center_coords: Optional[Tuple[float, float]] = None) -> Dict[str, Dict]:
        """Get static tile URLs for specific coordinates and zoom level.
        
        Args:
            scene_ids: List of Planet scene IDs
            zoom_level: Tile zoom level
            center_coords: (lat, lon) coordinates for tile center (optional)
            
        Returns:
            Dictionary with static tile information for each scene
        """
        static_tiles = {}
        api_key = getattr(self.auth, '_api_key', '')
        
        for scene_id in scene_ids:
            # Calculate tile coordinates
            if center_coords:
                tile_x, tile_y = self._lat_lon_to_tile(center_coords[0], center_coords[1], zoom_level)
            else:
                # Use center tile as default
                tile_x = tile_y = 2 ** zoom_level // 2
            
            # Create static tile URL
            static_url = f"{self.tile_base_url}/PSScene/{scene_id}/{zoom_level}/{tile_x}/{tile_y}.png"
            if api_key:
                static_url += f"?api_key={api_key}"
            
            static_tiles[scene_id] = {
                'url': static_url,
                'zoom_level': zoom_level,
                'tile_x': tile_x,
                'tile_y': tile_y,
                'center_coords': center_coords
            }
        
        logger.info(f"Generated static tile URLs for {len(static_tiles)} scenes")
        return static_tiles
    
    def _calculate_map_center(self, search_results: Dict, roi_geometry: Any = None) -> List[float]:
        """Calculate center point for map with improved ROI handling."""
        try:
            # Priority 1: Use ROI geometry if provided
            if roi_geometry is not None:
                try:
                    # Handle different ROI geometry formats
                    if hasattr(roi_geometry, 'centroid'):
                        # Shapely Polygon object
                        centroid = roi_geometry.centroid
                        center_coords = [centroid.y, centroid.x]
                        logger.info(f"Using ROI centroid: {center_coords}")
                        return center_coords
                    
                    elif isinstance(roi_geometry, dict):
                        # GeoJSON-like dictionary
                        from shapely.geometry import shape
                        polygon = shape(roi_geometry)
                        centroid = polygon.centroid
                        center_coords = [centroid.y, centroid.x]
                        logger.info(f"Using ROI centroid from GeoJSON: {center_coords}")
                        return center_coords
                    
                    elif hasattr(roi_geometry, '__geo_interface__'):
                        # Object with __geo_interface__ (like Shapely objects)
                        from shapely.geometry import shape
                        polygon = shape(roi_geometry.__geo_interface__)
                        centroid = polygon.centroid
                        center_coords = [centroid.y, centroid.x]
                        logger.info(f"Using ROI centroid from geo_interface: {center_coords}")
                        return center_coords
                    
                    else:
                        logger.warning(f"Unknown ROI geometry type: {type(roi_geometry)}")
                        
                except Exception as e:
                    logger.warning(f"Failed to calculate ROI centroid: {e}")
            
            # Priority 2: Use bounds of all scenes to calculate center
            if search_results and 'features' in search_results and search_results['features']:
                try:
                    from shapely.geometry import shape
                    
                    # Collect all scene geometries
                    scene_geometries = []
                    for scene in search_results['features']:
                        try:
                            geom = shape(scene['geometry'])
                            scene_geometries.append(geom)
                        except Exception as e:
                            logger.warning(f"Failed to process scene geometry: {e}")
                            continue
                    
                    if scene_geometries:
                        # Calculate bounds of all scenes
                        all_bounds = []
                        for geom in scene_geometries:
                            bounds = geom.bounds  # (minx, miny, maxx, maxy)
                            all_bounds.append(bounds)
                        
                        if all_bounds:
                            # Calculate overall bounds
                            min_x = min(bound[0] for bound in all_bounds)
                            min_y = min(bound[1] for bound in all_bounds)
                            max_x = max(bound[2] for bound in all_bounds)
                            max_y = max(bound[3] for bound in all_bounds)
                            
                            # Calculate center from bounds
                            center_lon = (min_x + max_x) / 2
                            center_lat = (min_y + max_y) / 2
                            center_coords = [center_lat, center_lon]
                            
                            logger.info(f"Using scenes bounds center: {center_coords}")
                            return center_coords
                    
                except Exception as e:
                    logger.warning(f"Failed to calculate scenes bounds center: {e}")
            
            # Priority 3: Use first scene centroid as fallback
            if search_results and 'features' in search_results and search_results['features']:
                try:
                    from shapely.geometry import shape
                    first_scene = search_results['features'][0]
                    geom = shape(first_scene['geometry'])
                    centroid = geom.centroid
                    center_coords = [centroid.y, centroid.x]
                    logger.info(f"Using first scene centroid: {center_coords}")
                    return center_coords
                except Exception as e:
                    logger.warning(f"Failed to use first scene centroid: {e}")
            
            # Priority 4: Default fallback (should rarely happen)
            logger.warning("Using default fallback coordinates (0, 0)")
            return [0.0, 0.0]
            
        except Exception as e:
            logger.error(f"Map center calculation failed completely: {e}")
            return [0.0, 0.0]
    
    def _add_scene_tiles_to_map(self, folium_map: Any, scenes: List[Dict]) -> int:
        """Add scene tile layers to map."""
        try:
            import folium
            
            api_key = getattr(self.auth, '_api_key', '')
            scenes_added = 0
            
            for scene in scenes:
                item_id = scene['id']
                props = scene['properties']
                acquired = props.get('acquired', 'Unknown')
                cloud_cover = props.get('cloud_cover', 0) * 100
                
                # Create tile URL
                tile_url = f"{self.tile_base_url}/PSScene/{item_id}/{{z}}/{{x}}/{{y}}.png"
                if api_key:
                    tile_url += f"?api_key={api_key}"
                
                # Create layer name
                layer_name = f"Scene {item_id[:12]}... ({acquired[:10]}, {cloud_cover:.1f}% cloud)"
                
                # Add tile layer
                folium.TileLayer(
                    tiles=tile_url,
                    attr='Planet Labs PBC',
                    name=layer_name,
                    overlay=True,
                    control=True
                ).add_to(folium_map)
                
                scenes_added += 1
            
            return scenes_added
            
        except Exception as e:
            logger.error(f"Failed to add scene tiles: {e}")
            return 0
    
    def _lat_lon_to_tile(self, lat: float, lon: float, zoom: int) -> Tuple[int, int]:
        """Convert lat/lon to tile coordinates."""
        lat_rad = math.radians(lat)
        n = 2.0 ** zoom
        x = int((lon + 180.0) / 360.0 * n)
        y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        return (x, y)
    
    # Also add this improved _calculate_zoom_level method if it's missing
    def _calculate_zoom_level(self, roi_geometry: Any = None, search_results: Dict = None) -> int:
        """Calculate appropriate zoom level based on ROI or scene coverage."""
        try:
            # Default zoom
            default_zoom = 12
            
            if roi_geometry is not None and not isinstance(roi_geometry, str):
                try:
                    # Handle different geometry types
                    if hasattr(roi_geometry, 'bounds'):
                        bounds = roi_geometry.bounds
                    elif isinstance(roi_geometry, dict):
                        from shapely.geometry import shape
                        polygon = shape(roi_geometry)
                        bounds = polygon.bounds
                    else:
                        return default_zoom
                    
                    # Calculate approximate area in degrees
                    width = bounds[2] - bounds[0]  # max_x - min_x
                    height = bounds[3] - bounds[1]  # max_y - min_y
                    max_dimension = max(width, height)
                    
                    # Zoom level based on dimension size
                    if max_dimension > 10:  # Very large area
                        return 6
                    elif max_dimension > 5:  # Large area
                        return 8
                    elif max_dimension > 1:  # Medium area
                        return 10
                    elif max_dimension > 0.1:  # Small area
                        return 12
                    elif max_dimension > 0.01:  # Very small area
                        return 14
                    else:  # Tiny area
                        return 16
                        
                except Exception as e:
                    logger.warning(f"Failed to calculate zoom from ROI: {e}")
            
            return default_zoom
            
        except Exception as e:
            logger.warning(f"Zoom calculation failed: {e}")
            return 12
        
    def load_roi_from_various_sources(self, roi_source: Union[str, "Polygon", Dict, List]) -> Optional["Polygon"]:
        """
        Load ROI from various source types for use with Preview Manager.
        
        Args:
            roi_source: Can be:
                - str: Path to GeoJSON/Shapefile
                - Polygon: Shapely polygon object
                - Dict: GeoJSON-like dictionary
                - List: List of coordinates
                
        Returns:
            Polygon: Loaded Shapely polygon or None if Shapely not available
        """
        if not _SHAPELY_AVAILABLE:
            logger.error("Shapely not available. Install with: pip install shapely")
            return None
            
        try:
            if isinstance(roi_source, str):
                # File path - load using InteractiveManager
                try:
                    from .interactive_manager import InteractiveManager
                    manager = InteractiveManager()
                    polygons = manager.load_roi_from_file(roi_source)
                    return polygons[0] if polygons else None
                except ImportError:
                    logger.error("InteractiveManager not available for file loading")
                    return None
                    
            elif hasattr(roi_source, 'exterior'):  # Check if it's a Polygon-like object
                # Already a Shapely polygon
                return roi_source
                
            elif isinstance(roi_source, dict):
                # GeoJSON-like dictionary
                if 'type' in roi_source and 'coordinates' in roi_source:
                    # Direct geometry
                    return shape(roi_source)
                elif 'features' in roi_source:
                    # FeatureCollection
                    first_feature = roi_source['features'][0]
                    return shape(first_feature['geometry'])
                elif 'geometry' in roi_source:
                    # Feature
                    return shape(roi_source['geometry'])
                else:
                    return shape(roi_source)
                    
            elif isinstance(roi_source, list):
                # List of coordinates
                return Polygon(roi_source)
                
            else:
                logger.error(f"Unsupported ROI source type: {type(roi_source)}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to load ROI from source: {e}")
            return None


    def create_interactive_map_from_file(self, 
                                    search_results: Dict,
                                    roi_file_path: str,
                                    max_scenes: int = 10) -> Optional[Any]:
        """
        Convenience method to create map directly from ROI file.
        
        Args:
            search_results: Planet API search results
            roi_file_path: Path to ROI file (GeoJSON, Shapefile, etc.)
            max_scenes: Maximum scenes to display
            
        Returns:
            Folium map object
        """
        # Load ROI from file
        roi_polygon = self.load_roi_from_various_sources(roi_file_path)
        
        if roi_polygon is None:
            logger.error(f"Could not load ROI from {roi_file_path}")
            return self.create_interactive_map(search_results, None, max_scenes)
        
        # Create map with loaded ROI
        return self.create_interactive_map(search_results, roi_polygon, max_scenes)

    def _add_roi_to_map(self, folium_map: Any, roi_geometry: Any) -> None:
        """Add ROI polygon to map with improved geometry handling."""
        try:
            import folium
            
            # Handle different ROI geometry formats
            roi_geojson = None
            
            if hasattr(roi_geometry, '__geo_interface__'):
                # Shapely geometry object
                roi_geojson = roi_geometry.__geo_interface__
            elif isinstance(roi_geometry, dict):
                # Already a GeoJSON-like dictionary
                if 'type' in roi_geometry and 'coordinates' in roi_geometry:
                    roi_geojson = roi_geometry
                elif 'geometry' in roi_geometry:
                    roi_geojson = roi_geometry['geometry']
                else:
                    roi_geojson = roi_geometry
            elif isinstance(roi_geometry, str):
                # String - could be a filename, warn user
                logger.warning(f"ROI geometry is a string: {roi_geometry}. Expected geometry object, not filename.")
                return
            else:
                logger.warning(f"Unsupported ROI geometry type: {type(roi_geometry)}")
                return
            
            if roi_geojson:
                # Style function for ROI
                style_function = lambda feature: {
                    'fillOpacity': 0.1,
                    'fillColor': 'red',
                    'color': 'red',
                    'weight': 3,
                    'dashArray': '5, 5'
                }
                
                # Add ROI to map
                folium.GeoJson(
                    roi_geojson,
                    style_function=style_function,
                    name="Search Area (ROI)",
                    tooltip="Region of Interest"
                ).add_to(folium_map)
                
                logger.info("ROI polygon added to map")
            else:
                logger.warning("Could not extract valid GeoJSON from ROI geometry")
                
        except Exception as e:
            logger.warning(f"Could not add ROI to map: {e}")
