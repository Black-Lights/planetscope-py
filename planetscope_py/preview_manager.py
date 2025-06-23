#!/usr/bin/env python3
"""Preview management for PlanetScope scenes.

This module provides advanced preview capabilities using Planet's Tile Service API,
including interactive map generation and static preview creation.
"""

import logging
import math
from typing import Dict, List, Optional, Union, Tuple, Any

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
        """Create interactive Folium map with Planet tile layers.
        
        Args:
            search_results: Planet API search results
            roi_geometry: Region of interest geometry (optional)
            max_scenes: Maximum number of scenes to add to map
            
        Returns:
            Folium map object or None if folium not available
        """
        try:
            import folium
            from shapely.geometry import shape
        except ImportError:
            logger.error("Folium and/or Shapely not available. Install with: pip install folium shapely")
            return None
        
        # Calculate map center
        map_center = self._calculate_map_center(search_results, roi_geometry)
        
        # Create base map
        folium_map = folium.Map(location=map_center, zoom_start=12)
        
        # Add ROI if provided
        if roi_geometry:
            self._add_roi_to_map(folium_map, roi_geometry)
        
        # Add scene tile layers
        scenes_added = self._add_scene_tiles_to_map(
            folium_map, 
            search_results['features'][:max_scenes]
        )
        
        # Add layer control
        folium.LayerControl().add_to(folium_map)
        
        logger.info(f"Created interactive map with {scenes_added} scene layers")
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
        """Calculate center point for map."""
        try:
            from shapely.geometry import shape
            
            if roi_geometry:
                if hasattr(roi_geometry, 'centroid'):
                    centroid = roi_geometry.centroid
                else:
                    polygon = shape(roi_geometry)
                    centroid = polygon.centroid
                return [centroid.y, centroid.x]
            else:
                # Use first scene center
                first_scene = search_results['features'][0]
                geom = shape(first_scene['geometry'])
                centroid = geom.centroid
                return [centroid.y, centroid.x]
        except Exception:
            # Fallback to default location
            return [0.0, 0.0]
    
    def _add_roi_to_map(self, folium_map: Any, roi_geometry: Any) -> None:
        """Add ROI polygon to map."""
        try:
            import folium
            
            style_function = lambda feature: {
                'fillOpacity': 0,
                'color': 'red',
                'weight': 3,
                'dashArray': '5, 5'
            }
            
            if hasattr(roi_geometry, '__geo_interface__'):
                roi_geojson = roi_geometry.__geo_interface__
            else:
                roi_geojson = roi_geometry
            
            folium.GeoJson(
                roi_geojson,
                style_function=style_function,
                name="Search Area"
            ).add_to(folium_map)
            
        except Exception as e:
            logger.warning(f"Could not add ROI to map: {e}")
    
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
